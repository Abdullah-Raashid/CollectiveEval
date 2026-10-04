"""Build strategy instances from experiment configuration."""

from __future__ import annotations

from typing import Any

from collectiveeval.budget import InferenceBudget
from collectiveeval.budget_policy import resolve_budget_policy
from collectiveeval.core import ModelSpec
from collectiveeval.router import HeuristicRouter, LearnedRouter
from collectiveeval.strategies import (
    AdaptiveRouterStrategy,
    AgentStrategy,
    CriticReviser,
    HeterogeneousPanel,
    MultiAgentDebate,
    SelfConsistency,
    SingleAgent,
)


def model_from_config(config: dict[str, Any] | None) -> ModelSpec:
    """Create a `ModelSpec` from common config aliases."""

    config = config or {}
    return ModelSpec(
        provider=str(config.get("provider", "mock")),
        model=str(config.get("model", config.get("name", "mock-accurate"))),
        role=config.get("role"),
        temperature=float(config.get("temperature", 0.0)),
        top_p=config.get("top_p"),
        max_tokens=config.get("max_tokens"),
        seed=config.get("seed"),
        timeout_s=float(config.get("timeout_s", 30.0)),
        base_url=config.get("base_url"),
        api_key_env=config.get("api_key_env"),
        mock_mode=config.get("mock_mode"),
        provider_options=dict(config.get("provider_options", {})),
        input_cost_per_1k=float(config.get("input_cost_per_1k", 0.0)),
        output_cost_per_1k=float(config.get("output_cost_per_1k", 0.0)),
    )


def budget_from_config(config: dict[str, Any] | None) -> InferenceBudget:
    """Create an inference budget from config using strict known fields."""

    return resolve_budget_policy({"budget": config or {}}).scientific_budget.to_inference_budget()


def strategy_from_config(config: dict[str, Any]) -> AgentStrategy:
    """Build one of the six Phase 1 strategies from YAML config."""

    matrix_metadata = config.get("matrix")
    if isinstance(matrix_metadata, dict) and matrix_metadata.get("execution_ready") is False:
        entry_name = matrix_metadata.get("entry_name", "<unknown>")
        raise ValueError(f"matrix entry is not execution-ready: {entry_name}")

    strategy_config = dict(config.get("strategy", {}))
    validate_strategy_config(strategy_config)
    strategy_type = str(strategy_config.get("type", "single_agent")).lower()
    model = model_from_config(config.get("model"))

    if strategy_type in {"single", "single_agent"}:
        return SingleAgent(model=model)
    if strategy_type in {"self_consistency", "self-consistency"}:
        return SelfConsistency(k=int(strategy_config.get("k", 4)), model=model)
    if strategy_type in {"debate", "multi_agent_debate", "multi-agent-debate"}:
        return MultiAgentDebate(
            agents=int(strategy_config.get("agents", 3)),
            rounds=int(strategy_config.get("rounds", 2)),
            model=model,
            peer_evidence=bool(strategy_config.get("peer_evidence", True)),
            specialist_roles=bool(strategy_config.get("specialist_roles", True)),
            homogeneous_agents=bool(strategy_config.get("homogeneous_agents", False)),
        )
    if strategy_type in {"critic", "critic_reviser", "critic-reviser"}:
        return CriticReviser(
            generator=model,
            critic_enabled=bool(strategy_config.get("critic_enabled", True)),
            revision_enabled=bool(strategy_config.get("revision_enabled", True)),
        )
    if strategy_type in {"panel", "heterogeneous_panel", "heterogeneous-panel"}:
        model_configs = strategy_config.get("models")
        models = None
        if isinstance(model_configs, list):
            models = [model_from_config(item) for item in model_configs]
        return HeterogeneousPanel(
            models=models,
            homogeneous=bool(strategy_config.get("homogeneous", False)),
        )
    if strategy_type in {"adaptive", "adaptive_router", "adaptive-router"}:
        cheap_model_config = strategy_config.get("cheap_model")
        cheap_model = model_from_config(cheap_model_config) if cheap_model_config else model
        router_type = str(strategy_config.get("router", "heuristic")).lower()
        router: HeuristicRouter | LearnedRouter
        if router_type == "heuristic":
            router = HeuristicRouter()
        elif router_type == "learned":
            artifact = strategy_config.get("router_artifact")
            if not artifact:
                raise ValueError("strategy.router=learned requires strategy.router_artifact")
            router = LearnedRouter.load(str(artifact))
        else:
            raise ValueError(f"unknown adaptive router type: {router_type}")
        uncertainty_signals_enabled = bool(
            strategy_config.get(
                "router_uncertainty_signals",
                strategy_config.get("uncertainty_signals", True),
            )
        )
        return AdaptiveRouterStrategy(
            cheap_model=cheap_model,
            router=router,
            uncertainty_signals_enabled=uncertainty_signals_enabled,
        )

    raise ValueError(f"unknown strategy type: {strategy_type}")


def validate_strategy_config(strategy_config: dict[str, Any]) -> None:
    """Reject known experimental variables that are not implemented."""

    strategy_type = str(strategy_config.get("type", "single_agent")).lower()
    unsupported_by_type: dict[str, set[str]] = {}
    ignored = sorted(unsupported_by_type.get(strategy_type, set()) & set(strategy_config))
    if ignored:
        raise ValueError(
            f"unsupported experimental parameter(s) for {strategy_type}: {', '.join(ignored)}"
        )
