"""Attempt accounting and durable resume regressions, entirely offline."""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from collectiveeval.budget import BudgetExceeded, BudgetLedger, InferenceBudget, UnknownUsageStop
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
from collectiveeval.datasets import file_sha256, write_jsonl
from collectiveeval.providers import ModelProvider, ProviderCapabilities, ProviderError
from collectiveeval.reporting import (
    attempt_accounting_report,
    validate_matched_budget_compatibility,
)
from collectiveeval.runner import run_experiment
from collectiveeval.storage import SQLiteStore


def example(name: str = "dev-attempt-fixture") -> BenchmarkExample:
    return BenchmarkExample(
        id=name,
        task_type=TaskType.GROUNDED_QA,
        input={
            "question": "deadline?",
            "source_text": "30 days",
            "evidence_units": [{"id": "doc", "text": "30 days"}],
        },
        gold={"answer": "30 days", "answerable": True, "evidence": ["doc"]},
        metadata={"source": "unit-fixture", "difficulty": "hard", "split": "dev", "tags": ["qa"]},
    )


def response(malformed: bool = False) -> ModelOutput:
    return ModelOutput(
        content={}
        if malformed
        else {"answer": "30 days", "citations": ["doc"], "confidence": 0.8, "abstain": False},
        raw_output="broken JSON" if malformed else None,
        usage=TokenUsage(input_tokens=30, output_tokens=17),
        usage_source=UsageSource.PROVIDER_REPORTED,
    )


class OfflineProvider(ModelProvider):
    provider_id = "offline-attempt-fixture"
    capabilities = ProviderCapabilities(supports_usage=True)

    def __init__(self, failures: int = 0, malformed: bool = False) -> None:
        self.requests: list[ProviderRequest] = []
        self.failures = failures
        self.malformed = malformed

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        self.requests.append(request)
        await asyncio.sleep(0.01)
        if len(self.requests) <= self.failures:
            raise ProviderError(ProviderErrorType.TIMEOUT, "offline timeout", retryable=True)
        return response(self.malformed)


MODEL = ModelSpec(provider="ollama", model="gemma3:4b", max_tokens=700, seed=100)


async def call(context: StrategyContext) -> ModelOutput:
    return await context.call_model(
        example=example(), model=MODEL, strategy="single_agent", role="generator"
    )


@pytest.mark.parametrize("call_ceiling", [1, 5])
def test_scientific_unknown_usage_stops_without_retry(call_ceiling: int) -> None:
    provider = OfflineProvider(failures=1)
    ledger = BudgetLedger(InferenceBudget(max_calls=call_ceiling, max_total_tokens=4000))
    context = StrategyContext({"ollama": provider}, ledger, max_retries=10, scientific_mode=True)
    with pytest.raises(UnknownUsageStop, match="UNKNOWN_USAGE_STOP"):
        asyncio.run(call(context))
    assert len(provider.requests) == ledger.provider_attempts == ledger.logical_model_calls == 1
    row = ledger.records[0]
    assert row.outcome == "TIMEOUT" and row.latency_ms >= 5
    assert row.input_tokens is row.output_tokens is row.usage_source is None
    assert row.usage_status == "UNKNOWN_NOT_RETURNED"
    assert ledger.metadata_dict()["total_provider_tokens"] is None
    with pytest.raises(UnknownUsageStop):
        asyncio.run(call(context))
    assert len(provider.requests) == 1


def test_general_retry_has_two_attempts_one_logical_call_and_full_latency() -> None:
    provider = OfflineProvider(failures=1)
    ledger = BudgetLedger(InferenceBudget(max_calls=2, max_total_tokens=4000))
    context = StrategyContext({"ollama": provider}, ledger, max_retries=1)
    asyncio.run(call(context))
    first, second = ledger.records
    assert first.attempt_id != second.attempt_id
    assert first.logical_call_id == second.logical_call_id
    assert [first.attempt_index, second.attempt_index] == [0, 1]
    assert ledger.provider_attempts == 2 and ledger.logical_model_calls == 1
    assert first.usage_status == "UNKNOWN_NOT_RETURNED"
    assert second.input_tokens == 30 and second.output_tokens == 17
    assert ledger.latency_ms >= 10
    assert ledger.usage_dict()["logical_call_latency_ms"] >= ledger.latency_ms
    assert not ledger.metadata_dict()["total_provider_tokens_exact"]


def test_general_retry_cannot_bypass_attempt_ceiling() -> None:
    provider = OfflineProvider(failures=1)
    context = StrategyContext(
        {"ollama": provider}, BudgetLedger(InferenceBudget(max_calls=1)), max_retries=1
    )
    with pytest.raises(BudgetExceeded):
        asyncio.run(call(context))
    assert len(provider.requests) == context.ledger.provider_attempts == 1


def test_success_preserves_authoritative_usage_and_separate_counts() -> None:
    context = StrategyContext(
        {"ollama": OfflineProvider()}, BudgetLedger(InferenceBudget(max_calls=1))
    )
    asyncio.run(call(context))
    assert context.ledger.provider_attempts == context.ledger.logical_model_calls == 1
    assert context.ledger.records[0].outcome == "SUCCESS"
    assert context.ledger.total_tokens == 47
    assert context.ledger.metadata_dict()["total_provider_tokens"] == 47


def test_parse_error_is_completed_inference_not_unknown_consumption() -> None:
    context = StrategyContext({"ollama": OfflineProvider(malformed=True)}, scientific_mode=True)
    with pytest.raises(ProviderError):
        asyncio.run(call(context))
    row = context.ledger.records[0]
    assert row.outcome == "PARSE_ERROR_AFTER_RESPONSE"
    assert row.usage_status == "PROVIDER_REPORTED" and row.normalized_error == "PARSE_ERROR"
    assert context.ledger.total_tokens == 47 and not context.ledger.unknown_usage_stop


def test_pre_call_rejection_has_no_attempt_or_provider_request() -> None:
    provider = OfflineProvider()
    context = StrategyContext(
        {"ollama": provider}, BudgetLedger(InferenceBudget(max_total_tokens=1))
    )
    with pytest.raises(BudgetExceeded):
        asyncio.run(call(context))
    assert not provider.requests and not context.ledger.records
    assert context.ledger.provider_attempts == context.ledger.logical_model_calls == 0


@pytest.mark.parametrize(
    "budget", [InferenceBudget(max_calls=1), InferenceBudget(max_total_tokens=700)]
)
def test_concurrent_calls_cannot_bypass_attempt_or_token_reservations(
    budget: InferenceBudget,
) -> None:
    async def execute() -> None:
        provider = OfflineProvider()
        entered, release = asyncio.Event(), asyncio.Event()
        original = provider.complete

        async def waiting(request: ProviderRequest) -> ModelOutput:
            entered.set()
            await release.wait()
            return await original(request)

        provider.complete = waiting
        context = StrategyContext({"ollama": provider}, BudgetLedger(budget), concurrency_limit=2)
        first = asyncio.create_task(call(context))
        await entered.wait()
        with pytest.raises(BudgetExceeded):
            await call(context)
        release.set()
        await first
        assert len(provider.requests) == context.ledger.provider_attempts == 1
        assert context.ledger.reserved_calls == 0

    asyncio.run(execute())


def config(dataset: Path) -> dict:
    return {
        "dataset": {"path": str(dataset)},
        "experiment": {"name": "offline-attempts"},
        "strategy": {"type": "single_agent"},
        "model": MODEL.model_dump(),
        "budget": {"max_calls": 5, "max_total_tokens": 4000},
        "checkpoint": {"resume_completed_examples": True},
        "provider_retry": {"max_retries": 5, "unknown_usage_policy": "UNKNOWN_USAGE_STOP.v1"},
    }


def test_unknown_attempt_is_durable_and_never_automatically_resumed(tmp_path: Path) -> None:
    dataset, db = tmp_path / "dev.jsonl", tmp_path / "attempts.sqlite3"
    write_jsonl(dataset, [example(), example("dev-second")])
    provider = OfflineProvider(failures=1)
    with patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}):
        with pytest.raises(UnknownUsageStop):
            asyncio.run(run_experiment(config(dataset), db_path=db, output_dir=tmp_path / "runs"))
        with pytest.raises(UnknownUsageStop, match="automatic resume/relaunch is forbidden"):
            asyncio.run(run_experiment(config(dataset), db_path=db, output_dir=tmp_path / "runs"))
    assert len(provider.requests) == 1
    store = SQLiteStore(db)
    run = store._fetch_all("SELECT * FROM runs")[0]
    assert run["status"] == "STOPPED_UNKNOWN_USAGE"
    rows = store.get_run_model_calls(run["id"])
    assert len(rows) == 1 and rows[0]["input_tokens"] is rows[0]["output_tokens"] is None
    assert len(store.get_run_predictions(run["id"])) == 1
    accounting = attempt_accounting_report(store, run["id"])
    assert accounting["logical_model_calls"] == accounting["provider_attempts"] == 1
    assert accounting["successful_attempts"] == 0 and accounting["failed_attempts"] == 1
    assert accounting["unknown_usage_attempts"] == 1
    assert (
        accounting["total_provider_tokens"] is None and accounting["known_token_lower_bound"] == 0
    )
    assert accounting["total_attempt_latency_ms"] >= 5
    assert accounting["strategy_wall_clock_latency_ms"] >= 5
    compatibility = validate_matched_budget_compatibility(store, [run["id"], run["id"]])
    assert not compatibility["ok"]
    assert "UNKNOWN_USAGE_ACCOUNTING" in compatibility["reason_codes"]


def test_dispatch_is_persisted_before_provider_and_completion_is_idempotent(tmp_path: Path) -> None:
    dataset, db = tmp_path / "dev.jsonl", tmp_path / "attempts.sqlite3"
    write_jsonl(dataset, [example()])
    provider = OfflineProvider()
    original = provider.complete

    async def inspecting(request: ProviderRequest) -> ModelOutput:
        with sqlite3.connect(db) as connection:
            row = connection.execute(
                "SELECT outcome,input_tokens FROM provider_attempts"
            ).fetchone()
        assert row == ("STARTED", None)
        return await original(request)

    provider.complete = inspecting
    with patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}):
        result = asyncio.run(
            run_experiment(config(dataset), db_path=db, output_dir=tmp_path / "runs")
        )
    store = SQLiteStore(db)
    assert (
        len(store.get_run_model_calls(result.run_id))
        == len(store.get_run_logical_calls(result.run_id))
        == 1
    )
    accounting = attempt_accounting_report(store, result.run_id)
    assert accounting["total_provider_tokens"] == accounting["known_token_lower_bound"] == 47
    assert accounting["successful_attempts"] == 1 and accounting["failed_attempts"] == 0


def test_completed_examples_resume_without_duplicate_attempts(tmp_path: Path) -> None:
    dataset, db = tmp_path / "dev.jsonl", tmp_path / "attempts.sqlite3"
    write_jsonl(dataset, [example(), example("dev-second")])
    provider = OfflineProvider()
    original = SQLiteStore.checkpoint_example
    checkpoints = 0

    def interrupted(
        store: SQLiteStore,
        result: StrategyResult,
        scores: dict[str, float],
        failures: list[dict[str, Any]],
    ) -> None:
        nonlocal checkpoints
        original(store, result, scores, failures)
        checkpoints += 1
        if checkpoints == 1:
            raise RuntimeError("offline interruption after atomic checkpoint")

    with patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}):
        with (
            patch.object(SQLiteStore, "checkpoint_example", interrupted),
            pytest.raises(RuntimeError),
        ):
            asyncio.run(run_experiment(config(dataset), db_path=db, output_dir=tmp_path / "runs"))
        result = asyncio.run(
            run_experiment(config(dataset), db_path=db, output_dir=tmp_path / "runs")
        )
    assert len(provider.requests) == 2
    store = SQLiteStore(db)
    assert (
        len(store.get_run_model_calls(result.run_id))
        == len(store.get_run_predictions(result.run_id))
        == 2
    )
    assert len({r["attempt_id"] for r in store.get_run_model_calls(result.run_id)}) == 2


def test_legacy_schema_is_not_migrated_or_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE model_calls(id INTEGER)")
    before = file_sha256(path)
    with pytest.raises(ValueError, match="Legacy scientific DB is immutable"):
        SQLiteStore(path)
    assert file_sha256(path) == before


def test_cancellation_keeps_sent_attempt_and_prevents_resume(tmp_path: Path) -> None:
    async def execute() -> None:
        dataset, db = tmp_path / "dev.jsonl", tmp_path / "attempts.sqlite3"
        write_jsonl(dataset, [example()])
        provider = OfflineProvider()
        entered = asyncio.Event()

        async def wait(request: ProviderRequest) -> ModelOutput:
            provider.requests.append(request)
            entered.set()
            await asyncio.sleep(10)
            return response()

        provider.complete = wait
        with patch(
            "collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}
        ):
            task = asyncio.create_task(
                run_experiment(config(dataset), db_path=db, output_dir=tmp_path / "runs")
            )
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(UnknownUsageStop):
                await run_experiment(config(dataset), db_path=db, output_dir=tmp_path / "runs")
        store = SQLiteStore(db)
        row = store._fetch_all("SELECT * FROM provider_attempts")[0]
        assert row["outcome"] == "INTERRUPTED" and row["end_timestamp"]
        assert row["usage_status"] == "UNKNOWN_NOT_RETURNED" and row["input_tokens"] is None
        assert len(provider.requests) == 1

    asyncio.run(execute())
