"""Offline budget boundary regressions; no HTTP or Ollama calls."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from collectiveeval import phase8_corrective
from collectiveeval.budget import BudgetExceeded, BudgetLedger, InferenceBudget
from collectiveeval.context import StrategyContext
from collectiveeval.core import (
    BenchmarkExample,
    ModelOutput,
    ModelSpec,
    ProviderErrorType,
    ProviderRequest,
    TaskType,
    TokenUsage,
    UsageSource,
)
from collectiveeval.datasets import write_jsonl
from collectiveeval.providers import ModelProvider, ProviderCapabilities, ProviderError
from collectiveeval.runner import run_experiment
from collectiveeval.storage import SQLiteStore
from collectiveeval.strategies import CriticReviser, SelfConsistency, SingleAgent


def example() -> BenchmarkExample:
    return BenchmarkExample(
        id="dev-budget-fixture",
        task_type=TaskType.GROUNDED_QA,
        input={
            "question": "deadline?",
            "source_text": "30 days",
            "evidence_units": [{"id": "doc", "text": "30 days"}],
        },
        gold={"answer": "30 days", "answerable": True, "evidence": ["doc"]},
        metadata={"source": "unit-fixture", "difficulty": "hard", "split": "dev", "tags": ["qa"]},
    )


def output(inputs: int, outputs: int, *, malformed: bool = False) -> ModelOutput:
    return ModelOutput(
        content={}
        if malformed
        else {"answer": "30 days", "citations": ["doc"], "confidence": 0.8, "abstain": False},
        raw_output="not JSON" if malformed else None,
        confidence=0.8,
        usage=TokenUsage(input_tokens=inputs, output_tokens=outputs),
        usage_source=UsageSource.PROVIDER_REPORTED,
    )


class RecordingProvider(ModelProvider):
    provider_id = "offline-stand-in"
    capabilities = ProviderCapabilities(supports_usage=True, supports_seed=True)

    def __init__(self, responses: list[ModelOutput]) -> None:
        self.responses = responses
        self.requests: list[ProviderRequest] = []

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        self.requests.append(request)
        return self.responses[len(self.requests) - 1]


MODEL = ModelSpec(provider="ollama", model="gemma3:4b", max_tokens=700, seed=100)


def setup_context(
    responses: list[ModelOutput], ceiling: int = 252
) -> tuple[StrategyContext, RecordingProvider, BudgetLedger]:
    provider = RecordingProvider(responses)
    ledger = BudgetLedger(InferenceBudget(max_total_tokens=ceiling), run_id="offline")
    return StrategyContext({"ollama": provider}, ledger), provider, ledger


async def call(context: StrategyContext, task: BenchmarkExample | None = None) -> ModelOutput:
    return await context.call_model(
        example=task or example(), model=MODEL, strategy="single_agent", role="generator"
    )


def test_tiny_budget_clips_cap_sent_to_provider_and_keeps_configured_cap() -> None:
    context, provider, ledger = setup_context([output(100, 20)])
    with patch("collectiveeval.context.estimate_tokens", return_value=180):
        asyncio.run(call(context))
    assert provider.requests[0].model.max_tokens == 72
    assert MODEL.max_tokens == 700
    metadata = ledger.records[0].metadata
    assert metadata["configured_max_tokens"] == 700
    assert metadata["effective_max_tokens"] == 72
    assert metadata["admission_estimated_input_tokens"] == 180


@pytest.mark.parametrize("estimated_input", [252, 253])
def test_impossible_input_rejected_without_fake_usage(estimated_input: int) -> None:
    context, provider, ledger = setup_context([])
    with (
        patch("collectiveeval.context.estimate_tokens", return_value=estimated_input),
        pytest.raises(BudgetExceeded),
    ):
        asyncio.run(call(context))
    assert not provider.requests and ledger.model_calls == ledger.total_tokens == 0
    assert ledger.budget_events[0]["event_type"] == "PRE_CALL_BUDGET_REJECTION"
    assert ledger.reserved_calls == 0


def test_completed_253_tokens_preserved_under_252_ceiling_and_output_retained() -> None:
    context, provider, ledger = setup_context([output(251, 2)])

    async def exercise() -> None:
        with patch("collectiveeval.context.estimate_tokens", return_value=251):
            result = await SingleAgent(MODEL).run(example(), context)
            assert result.output["answer"] == "30 days"
            assert result.metadata["budget_overrun_tokens"] == 1
            with pytest.raises(BudgetExceeded):
                await call(context)

    asyncio.run(exercise())
    assert len(provider.requests) == ledger.model_calls == 1
    assert provider.requests[0].model.max_tokens == 1
    assert ledger.input_tokens == 251 and ledger.output_tokens == 2 and ledger.total_tokens == 253
    assert ledger.records[0].metadata["post_call_budget_overrun"]
    assert not context.has_remaining_call()


def test_allowance_not_charged_and_reported_usage_is_authoritative() -> None:
    context, _, ledger = setup_context([output(120, 11)])
    with patch("collectiveeval.context.estimate_tokens", return_value=180):
        asyncio.run(call(context))
    assert ledger.total_tokens == 131
    assert ledger.remaining_dict()["remaining_total_tokens"] == 121
    assert not ledger.records[0].metadata["post_call_budget_overrun"]
    assert ledger.records[0].usage_source == "PROVIDER_REPORTED"


def test_gold_and_hidden_metadata_do_not_change_estimate_or_cap() -> None:
    values = []
    for answer, evidence, metadata in (
        ("30 days", ["doc"], "small"),
        ("secret" * 1000, ["gold" * 1000], "hidden" * 1000),
    ):
        context, provider, ledger = setup_context([output(20, 10)], ceiling=4000)
        task = example().model_copy(
            update={
                "gold": {**example().gold, "answer": answer, "evidence": evidence},
                "metadata": {**example().metadata, "private": metadata},
            }
        )
        asyncio.run(call(context, task))
        values.append(
            (
                provider.requests[0].prompt,
                ledger.records[0].metadata["admission_estimated_input_tokens"],
                provider.requests[0].model.max_tokens,
            )
        )
    assert values[0] == values[1]


def test_prompt_length_changes_estimated_input_and_output_allowance() -> None:
    estimates = []
    for text in ("short", "long input " * 100):
        context, provider, ledger = setup_context([output(20, 10)], ceiling=700)
        task = example().model_copy(update={"input": {**example().input, "source_text": text}})
        asyncio.run(call(context, task))
        estimates.append(
            (
                ledger.records[0].metadata["admission_estimated_input_tokens"],
                provider.requests[0].model.max_tokens,
            )
        )
    assert estimates[0][0] < estimates[1][0]
    assert estimates[0][1] > estimates[1][1]


def test_parse_failure_keeps_completed_provider_usage() -> None:
    context, provider, ledger = setup_context([output(30, 17, malformed=True)])
    with (
        patch("collectiveeval.context.estimate_tokens", return_value=180),
        pytest.raises(ProviderError) as caught,
    ):
        asyncio.run(call(context))
    assert caught.value.error_type == ProviderErrorType.PARSE_ERROR
    assert len(provider.requests) == ledger.model_calls == 1
    assert ledger.input_tokens == 30 and ledger.output_tokens == 17
    assert ledger.records[0].normalized_error == "PARSE_ERROR"
    assert ledger.records[0].usage_source == "PROVIDER_REPORTED"


def test_sequential_calls_use_actual_remaining_and_stop_at_boundary() -> None:
    context, provider, ledger = setup_context([output(40, 60), output(100, 52)])

    async def exercise() -> None:
        with patch("collectiveeval.context.estimate_tokens", return_value=100):
            await call(context)
            await call(context)
            with pytest.raises(BudgetExceeded):
                await call(context)

    asyncio.run(exercise())
    assert [r.model.max_tokens for r in provider.requests] == [152, 52]
    assert ledger.model_calls == 2 and ledger.total_tokens == 252
    assert not ledger.overrun_details


def test_concurrent_reservations_cannot_double_claim_tokens() -> None:
    async def exercise() -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        provider = RecordingProvider([output(10, 10)])
        original = provider.complete

        async def waiting(request: ProviderRequest) -> ModelOutput:
            entered.set()
            await release.wait()
            return await original(request)

        provider.complete = waiting
        ledger = BudgetLedger(InferenceBudget(max_total_tokens=252), run_id="offline")
        context = StrategyContext({"ollama": provider}, ledger, concurrency_limit=2)
        with patch("collectiveeval.context.estimate_tokens", return_value=180):
            first = asyncio.create_task(call(context))
            await entered.wait()
            with pytest.raises(BudgetExceeded):
                await call(context)
            assert ledger.total_tokens == 0 and ledger.reserved_calls == 1
            release.set()
            await first
        assert ledger.model_calls == 1 and ledger.total_tokens == 20
        assert ledger.reserved_calls == 0

    asyncio.run(exercise())


def test_failed_run_still_persists_completed_parse_error_call(tmp_path: Path) -> None:
    dataset = tmp_path / "dev.jsonl"
    write_jsonl(dataset, [example()])
    config = {
        "dataset": {"path": str(dataset)},
        "experiment": {"name": "parse-failure"},
        "strategy": {"type": "single_agent"},
        "model": MODEL.model_dump(),
        "budget": {"max_total_tokens": 4000},
        "concurrency": {"limit": 1},
    }
    provider = RecordingProvider([output(30, 17, malformed=True)])
    with (
        patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}),
        pytest.raises(ProviderError),
    ):
        asyncio.run(
            run_experiment(config, db_path=tmp_path / "calls.sqlite3", output_dir=tmp_path / "runs")
        )
    store = SQLiteStore(tmp_path / "calls.sqlite3")
    rows = store._fetch_all("SELECT * FROM model_calls")
    assert len(rows) == 1 and rows[0]["input_tokens"] == 30 and rows[0]["output_tokens"] == 17
    assert rows[0]["normalized_error"] == "PARSE_ERROR"
    assert store._fetch_all("SELECT status FROM runs")[0]["status"] == "FAILED"


def test_seed_pairs_are_distinct_and_repeated_execution_reproduces_pairs() -> None:
    pairs = []
    for _ in range(2):
        context, _, ledger = setup_context([output(20, 10), output(20, 10)], ceiling=4000)
        asyncio.run(SelfConsistency(2, MODEL).run(example(), context))
        pairs.append([r.requested_seed for r in ledger.records])
    assert pairs == [[100, 101], [100, 101]]


def test_auth_failure_has_no_invented_inference_usage() -> None:
    provider = RecordingProvider([])

    async def fail(request: ProviderRequest) -> ModelOutput:
        raise ProviderError(ProviderErrorType.AUTH_ERROR, "fixture auth failure")

    provider.complete = fail
    ledger = BudgetLedger(InferenceBudget(max_total_tokens=4000), run_id="offline")
    context = StrategyContext({"ollama": provider}, ledger)
    with pytest.raises(ProviderError):
        asyncio.run(call(context))
    assert ledger.model_calls == 1 and ledger.total_tokens == 0
    assert ledger.records[0].metadata["usage_unavailable"]
    assert ledger.reserved_calls == 0


def test_input_underestimate_overrun_keeps_reported_usage_within_output_cap() -> None:
    context, provider, ledger = setup_context([output(252, 1)])
    with patch("collectiveeval.context.estimate_tokens", return_value=251):
        asyncio.run(call(context))
    assert provider.requests[0].model.max_tokens == 1
    assert ledger.total_tokens == 253 and ledger.records[0].metadata["budget_overrun_tokens"] == 1


def test_output_and_input_subceilings_are_enforced_at_admission() -> None:
    ledger = BudgetLedger(
        InferenceBudget(max_total_tokens=252, max_output_tokens=10, max_input_tokens=180)
    )
    reservation = ledger.reserve_call_start(180, 700)
    assert reservation.effective_max_tokens == 10 and ledger.total_tokens == 0
    ledger.release_reserved_call(reservation)
    with pytest.raises(BudgetExceeded):
        ledger.reserve_call_start(181, 700)


def test_unknown_usage_fallback_remains_explicitly_estimated() -> None:
    response = output(0, 0).model_copy(update={"usage": TokenUsage(), "usage_source": None})
    context, _, ledger = setup_context([response], ceiling=4000)
    asyncio.run(call(context))
    assert ledger.records[0].usage_source == "ESTIMATED"
    assert ledger.input_tokens > 0 and ledger.output_tokens > 0


def test_overrun_output_usage_and_candidate_are_persisted(tmp_path: Path) -> None:
    dataset = tmp_path / "dev.jsonl"
    write_jsonl(dataset, [example()])
    config = {
        "dataset": {"path": str(dataset)},
        "experiment": {"name": "overrun"},
        "strategy": {"type": "single_agent"},
        "model": MODEL.model_dump(),
        "budget": {"max_total_tokens": 252},
        "concurrency": {"limit": 1},
    }
    provider = RecordingProvider([output(252, 1)])
    with (
        patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}),
        patch("collectiveeval.context.estimate_tokens", return_value=251),
    ):
        result = asyncio.run(
            run_experiment(config, db_path=tmp_path / "db.sqlite3", output_dir=tmp_path / "runs")
        )
    store = SQLiteStore(tmp_path / "db.sqlite3")
    row = store.get_run_predictions(result.run_id)[0]
    metadata = json.loads(row["metadata_json"])
    assert metadata["budget_overrun_tokens"] == 1 and metadata["candidates"]
    assert json.loads(row["usage_json"])["total_tokens"] == 253
    assert json.loads(row["output_json"])["answer"] == "30 days"
    assert len(store.get_run_model_calls(result.run_id)) == 1


def test_preparation_is_offline_and_preserves_scientific_settings(tmp_path: Path) -> None:
    if not (phase8_corrective.ORIGINAL / "phase8.sqlite3").exists():
        pytest.skip("Historical pilot unavailable")
    for name in ("rerun_scope.json", "preflight.json"):
        (tmp_path / name).write_bytes((phase8_corrective.CORRECTIVE / name).read_bytes())
    with (
        patch.object(phase8_corrective, "CORRECTIVE", tmp_path),
        patch.object(
            phase8_corrective,
            "discover_local_model_identity",
            side_effect=AssertionError("No Ollama access during preparation"),
        ),
    ):
        preflight = phase8_corrective.prepare()
    assert len(preflight["prepared_reruns"]) == 5
    assert preflight["new_real_model_calls"] == 0
    original_store = phase8_corrective.ReadOnlyPilotStore(
        phase8_corrective.ORIGINAL / "phase8.sqlite3"
    )
    for entry in preflight["prepared_reruns"]:
        config = json.loads(Path(entry["config_path"]).read_text())
        experiment = original_store.get_run_experiment(entry["historical_run_id"])
        assert experiment is not None
        original = json.loads(experiment["config_json"])
        assert config["model"] == original["model"]
        assert config["budget"] == original["budget"]
        assert config["strategy"] == original["strategy"]
        assert phase8_corrective.recursive_model_audit(config)


def test_execution_refuses_unpassed_static_gates_before_provider_access() -> None:
    with (
        patch.object(
            phase8_corrective,
            "discover_local_model_identity",
            side_effect=AssertionError("Must not contact Ollama"),
        ),
        pytest.raises(ValueError, match="Static gates"),
    ):
        phase8_corrective.execute({"status": "REPAIRED_STATIC_GATES_PENDING"})


def test_critic_contract_fenced_json_and_feedback_delivery_still_work() -> None:
    critique = ModelOutput(
        raw_output='prefix ```json\n{"needs_revision":true,"issues":["check deadline"]}\n```',
        usage=TokenUsage(input_tokens=30, output_tokens=20),
        usage_source=UsageSource.PROVIDER_REPORTED,
    )
    context, provider, ledger = setup_context(
        [output(20, 10), critique, output(20, 10)], ceiling=4000
    )
    task = example().model_copy(update={"gold": {**example().gold, "answer": "private-gold-value"}})
    result = asyncio.run(CriticReviser(generator=MODEL).run(task, context))
    assert result.metadata["revised"] and ledger.model_calls == 3
    assert provider.requests[1].expected_schema["required"] == ["needs_revision", "issues"]
    assert "critic-contract.v1" in provider.requests[1].prompt_version
    assert "private-gold-value" not in provider.requests[1].prompt
    assert "check deadline" in provider.requests[2].prompt
    assert {r.model.provider for r in provider.requests} == {"ollama"}


def test_cancellation_releases_only_its_own_reservation() -> None:
    async def exercise() -> None:
        started = asyncio.Event()
        provider = RecordingProvider([])

        async def wait(request: ProviderRequest) -> ModelOutput:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        provider.complete = wait
        ledger = BudgetLedger(InferenceBudget(max_total_tokens=252), run_id="offline")
        context = StrategyContext({"ollama": provider}, ledger)
        with patch("collectiveeval.context.estimate_tokens", return_value=180):
            pending = asyncio.create_task(call(context))
            await started.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert ledger.reserved_calls == 0 and ledger.total_tokens == 0

    asyncio.run(exercise())
