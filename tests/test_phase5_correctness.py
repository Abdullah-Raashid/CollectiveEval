import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from collectiveeval.budget import BudgetLedger, InferenceBudget
from collectiveeval.context import StrategyContext
from collectiveeval.core import (
    BenchmarkExample,
    ModelOutput,
    ModelSpec,
    ProviderErrorType,
    ProviderRequest,
    StrategyResult,
    TaskType,
    TokenUsage,
    UsageSource,
)
from collectiveeval.parsing import OutputParseError, parse_provider_output
from collectiveeval.providers import (
    MockProvider,
    ModelProvider,
    OpenAICompatibleProvider,
    ProviderCapabilities,
    ProviderError,
    VLLMCompatibleProvider,
    build_provider_registry,
    normalize_openai_response,
)
from collectiveeval.reporting import compare_runs_report
from collectiveeval.router import (
    HeuristicRouter,
    LearnedRouter,
    RouterTrainingExample,
    extract_uncertainty_signals,
)
from collectiveeval.storage import SQLiteStore
from collectiveeval.strategies import AdaptiveRouterStrategy, CriticReviser, MultiAgentDebate
from collectiveeval.strategy_factory import strategy_from_config


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def qa_example() -> BenchmarkExample:
    return BenchmarkExample(
        id="qa-phase5",
        task_type=TaskType.GROUNDED_QA,
        input={
            "source_text": "第3条: 解約通知は30日前まで。",
            "question": "何日前?",
            "evidence_units": [{"id": "doc-3", "text": "第3条: 解約通知は30日前まで。"}],
        },
        gold={"answerable": True, "answer": "30日前", "evidence": ["doc-3"]},
        metadata={
            "source": "synthetic_unit_test",
            "difficulty": "hard",
            "tags": ["qa", "phase5"],
            "split": "dev",
        },
    )


def qa_output(answer: str = "30日前", *, confidence: float = 0.9) -> dict[str, Any]:
    return {
        "answer": answer,
        "citations": ["doc-3"] if answer == "30日前" else [],
        "abstain": False,
        "confidence": confidence,
    }


class StaticProvider(ModelProvider):
    provider_id = "static"
    capabilities = ProviderCapabilities(supports_usage=True, supports_json_mode=True)

    def __init__(
        self,
        *,
        output: dict[str, Any] | None = None,
        raw_output: str | None = None,
        usage_source: UsageSource | None = None,
        usage: TokenUsage | None = None,
        critic_output: dict[str, Any] | None = None,
    ) -> None:
        self.output = output or qa_output()
        self.raw_output = raw_output
        self.critic_output = critic_output
        self.usage_source = usage_source
        self.usage = usage or TokenUsage()
        self.requests: list[ProviderRequest] = []

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        self.requests.append(request)
        selected_output = (
            self.critic_output
            if request.role == "critic" and self.critic_output is not None
            else self.output
        )
        return ModelOutput(
            content=dict(selected_output) if self.raw_output is None else {},
            raw_output=self.raw_output,
            confidence=0.8,
            usage=self.usage,
            usage_source=self.usage_source,
            provider=request.model.provider,
            model=request.model.model,
        )


class FlakyProvider(ModelProvider):
    provider_id = "flaky"
    capabilities = ProviderCapabilities()

    def __init__(self, *, failures: int, retryable: bool) -> None:
        self.failures = failures
        self.retryable = retryable
        self.calls = 0

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError(
                ProviderErrorType.TIMEOUT,
                "synthetic timeout",
                retryable=self.retryable,
            )
        return ModelOutput(content=qa_output(), confidence=0.8)


class Phase5CorrectnessTests(unittest.TestCase):
    def test_adaptive_router_config_selects_heuristic_and_learned_strictly(self) -> None:
        heuristic = strategy_from_config({"strategy": {"type": "adaptive", "router": "heuristic"}})
        self.assertIsInstance(heuristic, AdaptiveRouterStrategy)
        self.assertIsInstance(heuristic.router, HeuristicRouter)

        with tempfile.TemporaryDirectory() as raw_tmpdir:
            artifact = Path(raw_tmpdir) / "router.pkl"
            good = extract_uncertainty_signals(
                qa_example(),
                qa_output(confidence=0.95),
                confidence=0.95,
                model_id="mock/strong",
            )
            bad = extract_uncertainty_signals(
                qa_example(),
                qa_output("不明", confidence=0.1),
                confidence=0.1,
                model_id="mock/cheap-weak",
            )
            learned_router = LearnedRouter()
            learned_router.train(
                [
                    RouterTrainingExample(good, escalation_helped=False, split="dev"),
                    RouterTrainingExample(bad, escalation_helped=True, split="dev"),
                ]
            )
            learned_router.save(artifact)

            learned = strategy_from_config(
                {
                    "strategy": {
                        "type": "adaptive",
                        "router": "learned",
                        "router_artifact": str(artifact),
                    }
                }
            )
            self.assertIsInstance(learned.router, LearnedRouter)

        with self.assertRaisesRegex(ValueError, "router_artifact"):
            strategy_from_config({"strategy": {"type": "adaptive", "router": "learned"}})
        with self.assertRaisesRegex(ValueError, "unknown adaptive router"):
            strategy_from_config({"strategy": {"type": "adaptive", "router": "silent_fallback"}})

    def test_ablation_parameters_are_executable_strategy_controls(self) -> None:
        debate = strategy_from_config(
            {
                "strategy": {
                    "type": "multi_agent_debate",
                    "peer_evidence": False,
                    "specialist_roles": False,
                    "homogeneous_agents": True,
                }
            }
        )
        self.assertIsInstance(debate, MultiAgentDebate)
        self.assertFalse(debate.peer_evidence)
        self.assertFalse(debate.specialist_roles)
        self.assertTrue(debate.homogeneous_agents)

        critic = strategy_from_config(
            {"strategy": {"type": "critic_reviser", "critic_enabled": False}}
        )
        self.assertIsInstance(critic, CriticReviser)
        self.assertFalse(critic.critic_enabled)

        router = strategy_from_config(
            {"strategy": {"type": "adaptive_router", "router_uncertainty_signals": False}}
        )
        self.assertIsInstance(router, AdaptiveRouterStrategy)
        self.assertFalse(router.uncertainty_signals_enabled)

    def test_provider_registry_is_config_driven_and_rejects_unknown_providers(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown provider"):
            build_provider_registry({"model": {"provider": "bogus", "name": "x"}})

        registry = build_provider_registry(
            {
                "model": {"provider": "vllm", "name": "served-model"},
                "providers": {"vllm": {"base_url": "http://localhost:8000/v1"}},
            }
        )
        self.assertIsInstance(registry["vllm"], VLLMCompatibleProvider)
        self.assertEqual(MockProvider(mock_mode="mock-noisy").mock_mode, "noisy")

    def test_openai_response_normalization_preserves_provider_usage(self) -> None:
        request = ProviderRequest(
            example=qa_example(),
            model=ModelSpec(
                provider="openai",
                model="gpt-test",
                input_cost_per_1k=0.001,
                output_cost_per_1k=0.002,
            ),
            strategy="single_agent",
            role="generator",
            prompt="Return JSON",
            prompt_version="test.v1",
        )
        response = normalize_openai_response(
            {
                "choices": [
                    {
                        "message": {"content": json.dumps(qa_output(), ensure_ascii=False)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
            request=request,
            latency_ms=12.0,
            provider_id=OpenAICompatibleProvider.provider_id,
        )

        self.assertEqual(response.raw_output, json.dumps(qa_output(), ensure_ascii=False))
        self.assertEqual(response.usage.input_tokens, 11)
        self.assertEqual(response.usage.output_tokens, 7)
        self.assertEqual(response.usage_source, UsageSource.PROVIDER_REPORTED)

    def test_context_uses_provider_tokens_or_estimates_when_absent(self) -> None:
        provider_reported = StaticProvider(
            usage_source=UsageSource.PROVIDER_REPORTED,
            usage=TokenUsage(input_tokens=11, output_tokens=7),
        )
        context = StrategyContext(
            {"static": provider_reported},
            BudgetLedger(InferenceBudget(max_calls=1), run_id="usage-run"),
        )
        response = run(
            context.call_model(
                example=qa_example(),
                model=ModelSpec(provider="static", model="reported"),
                strategy="single_agent",
                role="generator",
            )
        )
        self.assertEqual(response.usage_source, UsageSource.PROVIDER_REPORTED)
        self.assertEqual(context.ledger.records[0].input_tokens, 11)
        self.assertEqual(context.ledger.records[0].output_tokens, 7)

        estimated = StaticProvider()
        context = StrategyContext(
            {"static": estimated},
            BudgetLedger(InferenceBudget(max_calls=1), run_id="usage-run"),
        )
        response = run(
            context.call_model(
                example=qa_example(),
                model=ModelSpec(provider="static", model="estimated"),
                strategy="single_agent",
                role="generator",
            )
        )
        self.assertEqual(response.usage_source, UsageSource.ESTIMATED)
        self.assertGreater(context.ledger.records[0].input_tokens, 0)
        self.assertGreater(context.ledger.records[0].output_tokens, 0)

    def test_retryable_provider_errors_are_retried_and_nonretryable_errors_raise(self) -> None:
        provider = FlakyProvider(failures=1, retryable=True)
        context = StrategyContext(
            {"flaky": provider},
            BudgetLedger(InferenceBudget(max_calls=2), run_id="retry-run"),
            max_retries=1,
        )
        run(
            context.call_model(
                example=qa_example(),
                model=ModelSpec(provider="flaky", model="eventual"),
                strategy="single_agent",
                role="generator",
            )
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(context.ledger.records), 2)
        self.assertEqual(context.ledger.records[0].outcome, "TIMEOUT")
        self.assertEqual(context.ledger.records[1].attempt_index, 1)
        self.assertEqual(context.ledger.logical_model_calls, 1)

        provider = FlakyProvider(failures=1, retryable=False)
        context = StrategyContext(
            {"flaky": provider},
            BudgetLedger(InferenceBudget(max_calls=1), run_id="retry-run"),
            max_retries=1,
        )
        with self.assertRaises(ProviderError):
            run(
                context.call_model(
                    example=qa_example(),
                    model=ModelSpec(provider="flaky", model="fatal"),
                    strategy="single_agent",
                    role="generator",
                )
            )
        self.assertEqual(context.ledger.records[0].normalized_error, str(ProviderErrorType.TIMEOUT))

    def test_prompt_version_and_parse_metadata_are_persisted_in_call_records(self) -> None:
        raw = f"prefix {json.dumps(qa_output(), ensure_ascii=False)} suffix"
        provider = StaticProvider(raw_output=raw)
        context = StrategyContext(
            {"static": provider},
            BudgetLedger(InferenceBudget(max_calls=1), run_id="parse-run"),
        )
        response = run(
            context.call_model(
                example=qa_example(),
                model=ModelSpec(provider="static", model="raw-json"),
                strategy="single_agent",
                role="generator",
            )
        )

        record = context.ledger.records[0]
        self.assertTrue(record.prompt_version.startswith("task-contracts.v1."))
        self.assertEqual(record.metadata["parse_status"], "REPAIRED")
        self.assertEqual(response.content["answer"], "30日前")

    def test_parser_strict_json_repair_and_parse_error(self) -> None:
        repaired = parse_provider_output(
            qa_example(),
            f"Here is JSON: {json.dumps(qa_output(), ensure_ascii=False)}",
        )
        self.assertEqual(repaired.parse_status, "REPAIRED")

        with self.assertRaises(OutputParseError):
            parse_provider_output(qa_example(), "not json and no object")

    def test_critic_prompt_contract_does_not_include_gold_answer(self) -> None:
        provider = StaticProvider(
            output=qa_output("不明", confidence=0.2),
            critic_output={"needs_revision": False, "issues": []},
        )
        context = StrategyContext(
            {"static": provider},
            BudgetLedger(InferenceBudget(max_calls=3), run_id="critic-run"),
        )
        strategy = CriticReviser(
            generator=ModelSpec(provider="static", model="generator"),
            critic=ModelSpec(provider="static", model="critic"),
            reviser=ModelSpec(provider="static", model="reviser"),
        )
        run(strategy.run(qa_example(), context))

        critic_request = provider.requests[1]
        self.assertEqual(critic_request.role, "critic")
        self.assertIsNotNone(critic_request.candidate)
        self.assertNotIn(str(qa_example().gold), critic_request.prompt)
        self.assertEqual(
            critic_request.expected_schema.get("required"),
            ["needs_revision", "issues"],
        )
        self.assertIn("Expected critic response schema", critic_request.prompt)
        self.assertIn("needs_revision", critic_request.prompt)

    def test_matched_budget_comparison_rejects_incompatible_budget(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            store = SQLiteStore(Path(raw_tmpdir) / "runs.sqlite3")
            example = qa_example()
            for index, calls in enumerate((1, 2), start=1):
                experiment_id = f"exp-{index}"
                run_id = f"run-{index}"
                store.insert_experiment(
                    experiment_id=experiment_id,
                    name=experiment_id,
                    config_hash=f"config-{index}",
                    dataset_hash="same-dataset",
                    git_commit="unknown",
                    python_version="test",
                    created_at=f"2026-01-0{index}T00:00:00Z",
                    config={
                        "budget": {"max_calls": calls, "max_total_tokens": 1000},
                        "budget_policy": {"scientific_budget_scope": "per_example"},
                        "model": {"provider": "mock", "name": "mock-accurate"},
                    },
                )
                store.insert_run(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    strategy="single_agent",
                    status="COMPLETED",
                    start_ts=f"2026-01-0{index}T00:00:00Z",
                )
                store.upsert_example(example)
                store.insert_prediction(
                    StrategyResult(
                        example_id=example.id,
                        strategy="single_agent",
                        output=qa_output(),
                        confidence=0.9,
                        model_calls=calls,
                        metadata={"run_id": run_id},
                    )
                )
                store.insert_metric(
                    run_id=run_id,
                    example_id=example.id,
                    metric_name="task_score",
                    metric_value=1.0,
                )

            report = compare_runs_report(store, ["run-1", "run-2"])
            self.assertFalse(report["matched_budget"]["ok"])
            self.assertIn(
                "runs use different matched call budgets",
                report["matched_budget"]["reasons"],
            )
            self.assertIn("CALL_BUDGET_MISMATCH", report["matched_budget"]["reason_codes"])
            self.assertEqual(report["comparisons"], [])


if __name__ == "__main__":
    unittest.main()
