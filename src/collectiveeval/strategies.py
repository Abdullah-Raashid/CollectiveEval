"""Inference strategies implemented over one task/provider interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Protocol

from collectiveeval.aggregation import majority_vote, summarize_peer
from collectiveeval.budget import BudgetExceeded
from collectiveeval.context import StrategyContext
from collectiveeval.core import (
    BenchmarkExample,
    Candidate,
    ModelSpec,
    StrategyResult,
    finish_result,
)
from collectiveeval.router import HeuristicRouter, UncertaintySignals, extract_uncertainty_signals
from collectiveeval.validation import validate_output_schema


class RouterPolicy(Protocol):
    def route(self, signals: UncertaintySignals) -> str: ...

    def uncertainty_score(self, signals: UncertaintySignals) -> float: ...


class AgentStrategy(ABC):
    """Base interface for every strategy."""

    name: str

    @abstractmethod
    async def run(self, example: BenchmarkExample, context: StrategyContext) -> StrategyResult:
        """Run the strategy for one example."""

    def _finish(
        self,
        example: BenchmarkExample,
        selected: Candidate,
        candidates: list[Candidate],
        context: StrategyContext,
        metadata: dict[str, object] | None = None,
    ) -> StrategyResult:
        return finish_result(
            example_id=example.id,
            strategy=self.name,
            output=selected.output,
            confidence=selected.confidence,
            candidates=candidates,
            context_usage=context.ledger.usage_dict(),
            metadata={**(metadata or {}), **context.ledger.metadata_dict()},
        )


class SingleAgent(AgentStrategy):
    """One model call, one structured answer."""

    name = "single_agent"

    def __init__(self, model: ModelSpec | None = None) -> None:
        self.model = model or ModelSpec()

    async def run(self, example: BenchmarkExample, context: StrategyContext) -> StrategyResult:
        response = await context.call_model(
            example=example, model=self.model, strategy=self.name, role="generator"
        )
        candidate = Candidate(
            output=response.content,
            confidence=response.confidence,
            model_id=self.model.model_id,
            role="generator",
        )
        return self._finish(example, candidate, [candidate], context)


class SelfConsistency(AgentStrategy):
    """Independent generations plus deterministic majority aggregation."""

    name = "self_consistency"

    def __init__(self, k: int = 4, model: ModelSpec | None = None) -> None:
        if k not in {2, 4}:
            raise ValueError("SelfConsistency supports k=2 or k=4 in Phase 1")
        self.k = k
        self.model = model or ModelSpec(temperature=0.7)

    async def run(self, example: BenchmarkExample, context: StrategyContext) -> StrategyResult:
        async def sample(sample_index: int) -> tuple[int, Candidate]:
            sample_model = (
                self.model.model_copy(update={"seed": self.model.seed + sample_index})
                if self.model.seed is not None
                else self.model
            )
            response = await context.call_model(
                example=example,
                model=sample_model,
                strategy=self.name,
                role="sample",
                round_index=sample_index,
            )
            return (
                sample_index,
                Candidate(
                    output=response.content,
                    confidence=response.confidence,
                    model_id=self.model.model_id,
                    role=f"sample_{sample_index}",
                ),
            )

        results = await asyncio.gather(
            *(sample(sample_index) for sample_index in range(self.k)),
            return_exceptions=True,
        )
        candidates = []
        for result in results:
            if isinstance(result, BudgetExceeded):
                continue
            if isinstance(result, BaseException):
                raise result
            _, candidate = result
            candidates.append(candidate)
        if not candidates:
            raise BudgetExceeded("SelfConsistency could not make any model calls")
        selected = majority_vote(candidates)
        return self._finish(example, selected, candidates, context, {"k": self.k})


class CriticReviser(AgentStrategy):
    """Generator -> specialist critic -> revision, with one repair retry."""

    name = "critic_reviser"

    def __init__(
        self,
        generator: ModelSpec | None = None,
        critic: ModelSpec | None = None,
        reviser: ModelSpec | None = None,
        critic_enabled: bool = True,
        revision_enabled: bool = True,
    ) -> None:
        self.generator = generator or ModelSpec(role="generator")
        self.critic = critic or self.generator.model_copy(update={"role": "critic"})
        self.reviser = reviser or self.generator.model_copy(update={"role": "reviser"})
        self.critic_enabled = critic_enabled
        self.revision_enabled = revision_enabled

    async def run(self, example: BenchmarkExample, context: StrategyContext) -> StrategyResult:
        generator_response = await context.call_model(
            example=example, model=self.generator, strategy=self.name, role="generator"
        )
        generated = Candidate(
            output=generator_response.content,
            confidence=generator_response.confidence,
            model_id=self.generator.model_id,
            role="generator",
        )
        candidates = [generated]

        if not self.critic_enabled:
            return self._finish(
                example,
                generated,
                candidates,
                context,
                {"critic_enabled": False, "revised": False},
            )
        if not context.has_remaining_call():
            return self._finish(example, generated, candidates, context, {"stopped": "budget"})

        critic_response = await context.call_model(
            example=example,
            model=self.critic,
            strategy=self.name,
            role="critic",
            candidate=generated.output,
        )
        critic_says_revision = bool(critic_response.content.get("needs_revision", False))
        schema_valid, _ = validate_output_schema(example, generated.output)
        if (not critic_says_revision and schema_valid) or not self.revision_enabled:
            return self._finish(
                example,
                generated,
                candidates,
                context,
                {
                    "critic": critic_response.content,
                    "revised": False,
                    "revision_enabled": self.revision_enabled,
                },
            )

        if not context.has_remaining_call():
            return self._finish(example, generated, candidates, context, {"stopped": "budget"})

        revision_response = await context.call_model(
            example=example,
            model=self.reviser,
            strategy=self.name,
            role="reviser",
            candidate=generated.output,
            extra={"critic": critic_response.content},
        )
        revised = Candidate(
            output=revision_response.content,
            confidence=revision_response.confidence,
            model_id=self.reviser.model_id,
            role="reviser",
        )
        candidates.append(revised)

        schema_valid, _ = validate_output_schema(example, revised.output)
        if schema_valid:
            return self._finish(
                example,
                revised,
                candidates,
                context,
                {"critic": critic_response.content, "revised": True},
            )

        if context.has_remaining_call():
            repair_response = await context.call_model(
                example=example,
                model=self.reviser,
                strategy=self.name,
                role="repair",
                candidate=revised.output,
                extra={"critic": critic_response.content},
            )
            repaired = Candidate(
                output=repair_response.content,
                confidence=repair_response.confidence,
                model_id=self.reviser.model_id,
                role="repair",
            )
            candidates.append(repaired)
            return self._finish(
                example,
                repaired,
                candidates,
                context,
                {"critic": critic_response.content, "revised": True, "repair_retry": True},
            )

        return self._finish(
            example,
            revised,
            candidates,
            context,
            {"critic": critic_response.content, "revised": True, "schema_valid": False},
        )


class MultiAgentDebate(AgentStrategy):
    """N agents answer, view concise peer summaries, then revise for rounds."""

    name = "multi_agent_debate"

    def __init__(
        self,
        agents: int = 3,
        rounds: int = 2,
        model: ModelSpec | None = None,
        peer_evidence: bool = True,
        specialist_roles: bool = True,
        homogeneous_agents: bool = False,
    ) -> None:
        if agents not in {2, 3, 4}:
            raise ValueError("MultiAgentDebate supports 2, 3, or 4 agents")
        if rounds not in {1, 2, 3}:
            raise ValueError("MultiAgentDebate supports 1, 2, or 3 total rounds")
        self.agents = agents
        self.rounds = rounds
        self.model = model or ModelSpec(temperature=0.4)
        self.peer_evidence = peer_evidence
        self.specialist_roles = specialist_roles
        self.homogeneous_agents = homogeneous_agents

    async def run(self, example: BenchmarkExample, context: StrategyContext) -> StrategyResult:
        all_candidates: list[Candidate] = []
        current_round: list[Candidate] = []
        peer_summaries: list[dict[str, object]] = []

        for round_index in range(self.rounds):

            async def agent_call(
                agent_index: int,
                round_number: int,
                summaries: list[dict[str, object]],
            ) -> tuple[int, Candidate]:
                role = f"agent_{agent_index}" if self.specialist_roles else "generalist"
                model = (
                    self.model
                    if self.homogeneous_agents
                    else self.model.model_copy(update={"role": role})
                )
                response = await context.call_model(
                    example=example,
                    model=model,
                    strategy=self.name,
                    role=role,
                    round_index=round_number,
                    peer_summaries=summaries if self.peer_evidence else [],
                )
                return (
                    agent_index,
                    Candidate(
                        output=response.content,
                        confidence=response.confidence,
                        model_id=model.model_id,
                        role=role,
                        metadata={"round": round_number},
                    ),
                )

            results = await asyncio.gather(
                *(
                    agent_call(agent_index, round_index, list(peer_summaries))
                    for agent_index in range(self.agents)
                ),
                return_exceptions=True,
            )
            next_round = []
            for result in results:
                if isinstance(result, BudgetExceeded):
                    continue
                if isinstance(result, BaseException):
                    raise result
                _, candidate = result
                next_round.append(candidate)
                all_candidates.append(candidate)
            if next_round:
                current_round = next_round
                peer_summaries = (
                    [summarize_peer(candidate) for candidate in current_round]
                    if self.peer_evidence
                    else []
                )
            if not context.has_remaining_call():
                break

        if not current_round:
            raise BudgetExceeded("MultiAgentDebate could not make any model calls")
        selected = majority_vote(current_round)
        return self._finish(
            example,
            selected,
            all_candidates,
            context,
            {
                "agents": self.agents,
                "rounds": self.rounds,
                "peer_evidence": self.peer_evidence,
                "specialist_roles": self.specialist_roles,
                "homogeneous_agents": self.homogeneous_agents,
            },
        )


class HeterogeneousPanel(AgentStrategy):
    """Different provider/model/role candidates plus deterministic selection."""

    name = "heterogeneous_panel"

    def __init__(self, models: list[ModelSpec] | None = None, homogeneous: bool = False) -> None:
        self.models = models or [
            ModelSpec(model="mock-accurate", role="qa_specialist"),
            ModelSpec(model="mock-accurate-alt", role="business_specialist"),
        ]
        if homogeneous and self.models:
            first = self.models[0]
            self.models = [
                first.model_copy(update={"role": f"panelist_{i}"}) for i in range(len(self.models))
            ]
        self.homogeneous = homogeneous
        if len(self.models) < 2:
            raise ValueError("HeterogeneousPanel needs at least two model specs")

    async def run(self, example: BenchmarkExample, context: StrategyContext) -> StrategyResult:
        async def panel_call(index: int, model: ModelSpec) -> tuple[int, Candidate]:
            role = model.role or f"panelist_{index}"
            response = await context.call_model(
                example=example,
                model=model,
                strategy=self.name,
                role=role,
                round_index=index,
            )
            return (
                index,
                Candidate(
                    output=response.content,
                    confidence=response.confidence,
                    model_id=model.model_id,
                    role=role,
                ),
            )

        results = await asyncio.gather(
            *(panel_call(index, model) for index, model in enumerate(self.models)),
            return_exceptions=True,
        )
        candidates = []
        for result in results:
            if isinstance(result, BudgetExceeded):
                continue
            if isinstance(result, BaseException):
                raise result
            _, candidate = result
            candidates.append(candidate)
        if not candidates:
            raise BudgetExceeded("HeterogeneousPanel could not make any model calls")
        selected = majority_vote(candidates)
        return self._finish(
            example,
            selected,
            candidates,
            context,
            {"models": len(self.models), "homogeneous": self.homogeneous},
        )


class AdaptiveRouterStrategy(AgentStrategy):
    """Cheap first pass with heuristic escalation to critic or debate."""

    name = "adaptive_router"

    def __init__(
        self,
        cheap_model: ModelSpec | None = None,
        router: RouterPolicy | None = None,
        critic_strategy: CriticReviser | None = None,
        debate_strategy: MultiAgentDebate | None = None,
        uncertainty_signals_enabled: bool = True,
    ) -> None:
        self.cheap_model = cheap_model or ModelSpec(model="mock-accurate-cheap")
        self.router = router or HeuristicRouter()

        if critic_strategy is not None:
            self.critic_strategy = critic_strategy
        elif self.cheap_model.provider == "mock":
            # Preserve deterministic legacy mock fixtures.
            self.critic_strategy = CriticReviser(generator=ModelSpec())
        else:
            # Real Phase 8 providers must propagate into every nested role.
            self.critic_strategy = CriticReviser(
                generator=self.cheap_model.model_copy(update={"role": "generator"}),
                critic=self.cheap_model.model_copy(update={"role": "critic"}),
                reviser=self.cheap_model.model_copy(update={"role": "reviser"}),
            )

        if debate_strategy is not None:
            self.debate_strategy = debate_strategy
        elif self.cheap_model.provider == "mock":
            # Preserve deterministic legacy mock fixtures.
            self.debate_strategy = MultiAgentDebate(agents=2, rounds=2)
        else:
            self.debate_strategy = MultiAgentDebate(
                agents=2,
                rounds=2,
                model=self.cheap_model.model_copy(update={"temperature": 0.5}),
            )
        self.uncertainty_signals_enabled = uncertainty_signals_enabled

    async def run(self, example: BenchmarkExample, context: StrategyContext) -> StrategyResult:
        first_response = await context.call_model(
            example=example,
            model=self.cheap_model,
            strategy=self.name,
            role="cheap_single_agent",
        )
        initial = Candidate(
            output=first_response.content,
            confidence=first_response.confidence,
            model_id=self.cheap_model.model_id,
            role="cheap_single_agent",
        )
        signals = extract_uncertainty_signals(
            example,
            initial.output,
            confidence=initial.confidence,
            model_id=initial.model_id,
        )
        action = self.router.route(signals) if self.uncertainty_signals_enabled else "accept"
        if action == "accept" or not context.has_remaining_call():
            return self._finish(
                example,
                initial,
                [initial],
                context,
                {
                    "route": action,
                    "escalation_incomplete": action != "accept",
                    "budget_exhausted": action != "accept",
                    "budget_exhaustion_reason": "model call budget exhausted"
                    if action != "accept"
                    else None,
                    "uncertainty_score": self.router.uncertainty_score(signals),
                },
            )

        try:
            if action == "critic":
                escalated = await self.critic_strategy.run(example, context)
            else:
                escalated = await self.debate_strategy.run(example, context)
        except BudgetExceeded as exc:
            return self._finish(
                example,
                initial,
                [initial],
                context,
                {
                    "route": action,
                    "budget_exhausted": True,
                    "budget_exhaustion_reason": str(exc),
                    "escalation_incomplete": True,
                    "uncertainty_score": self.router.uncertainty_score(signals),
                },
            )

        candidates = [initial, *escalated.candidates]
        selected = Candidate(
            output=escalated.output,
            confidence=escalated.confidence,
            model_id="escalated",
            role=action,
        )
        return self._finish(
            example,
            selected,
            candidates,
            context,
            {"route": action, "uncertainty_score": self.router.uncertainty_score(signals)},
        )
