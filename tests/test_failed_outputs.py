from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from collectiveeval.budget import BudgetLedger, InferenceBudget, UnknownUsageStop
from collectiveeval.context import StrategyContext
from collectiveeval.core import (
    BenchmarkExample,
    ModelOutput,
    ModelSpec,
    ProviderRequest,
    TokenUsage,
)
from collectiveeval.datasets import load_jsonl, write_jsonl
from collectiveeval.failed_outputs import (
    FailedOutput,
    FailedPrediction,
    decode_failure,
    evaluate_output,
    failed_prediction,
)
from collectiveeval.metrics import score_prediction
from collectiveeval.providers import ModelProvider, ProviderCapabilities, ProviderError
from collectiveeval.reporting import component_coverage
from collectiveeval.runner import run_experiment
from collectiveeval.storage import SQLiteStore

ROOT = Path(__file__).resolve().parents[1]


def fixtures() -> list[BenchmarkExample]:
    examples = load_jsonl(ROOT / "data/splits/dev.mock.jsonl")
    qa = next(e for e in examples if str(e.task_type) == "grounded_qa")
    examples.append(
        BenchmarkExample.model_validate(
            {
                **qa.model_dump(mode="json"),
                "id": "synthetic-robustness",
                "task_type": "robustness",
            }
        )
    )
    examples.append(
        BenchmarkExample(
            id="synthetic-summary",
            task_type="business_summarization",
            input={"text": "The team approved a launch."},
            gold={
                "summary": "Launch approved.",
                "decisions": ["Launch approved"],
                "action_items": [],
                "risks": [],
                "supported_facts": ["Launch approved"],
            },
            metadata={
                "source": "synthetic-unit-test",
                "difficulty": "easy",
                "tags": ["summary"],
                "split": "dev",
            },
        )
    )
    return examples


def extraction() -> BenchmarkExample:
    return BenchmarkExample(
        id="synthetic-wrong-boolean",
        task_type="structured_extraction",
        input={"text": "Synthetic record."},
        gold={
            "expected": {"auto_renewal": False},
            "json_schema": {
                "type": "object",
                "required": ["auto_renewal"],
                "properties": {"auto_renewal": {"type": ["boolean", "null"]}},
            },
        },
        metadata={
            "source": "synthetic-unit-test",
            "difficulty": "hard",
            "tags": ["schema"],
            "split": "dev",
        },
    )


class ReturnedFailureProvider(ModelProvider):
    provider_id = "ollama"
    capabilities = ProviderCapabilities(supports_usage=True)

    def __init__(self, raw: str, unknown: bool = False) -> None:
        self.raw = raw
        self.requests: list[ProviderRequest] = []
        self.unknown = unknown

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        self.requests.append(request)
        return ModelOutput(
            raw_output=self.raw,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            usage_source="ESTIMATED" if self.unknown else "PROVIDER_REPORTED",
        )


def failed(raw: str, example: BenchmarkExample | None = None) -> FailedPrediction:
    example = example or extraction()
    provider = ReturnedFailureProvider(raw)
    ledger = BudgetLedger(InferenceBudget(max_calls=6, max_total_tokens=12000))
    context = StrategyContext({"ollama": provider}, ledger, scientific_mode=True)
    with pytest.raises(ProviderError):
        asyncio.run(
            context.call_model(
                example=example,
                model=ModelSpec(provider="ollama", model="gemma3:4b", max_tokens=700),
                strategy="single_agent",
                role="generator",
            )
        )
    result = failed_prediction(ledger, example.id, "single_agent")
    assert len(provider.requests) == 1
    return result


def test_wrong_boolean_preserves_original_dictionary_and_existing_safe_metrics() -> None:
    example = extraction()
    original = {"auto_renewal": "wrong-type"}
    result = failed(json.dumps(original), example)
    assert result.output.parsed_json == original
    scores = evaluate_output(example, result.output)
    existing = score_prediction(example, original)
    assert all(scores[key] == value for key, value in existing.items())
    assert scores["schema_validity"] == 0
    assert scores["json_parse_success"] == 1


@pytest.mark.parametrize(
    "raw,syntax",
    [
        ("broken JSON", False),
        ("[1,2]", True),
        ("42", True),
        ('"a string"', True),
        ("null", True),
    ],
)
def test_malformed_and_non_object_are_not_fabricated_dictionaries(raw: str, syntax: bool) -> None:
    result = failed(raw)
    assert result.output.raw_content == raw
    assert result.output.json_syntax_parsed is syntax
    assert result.output.json_object is False
    assert not isinstance(result.output, dict)
    scores = evaluate_output(extraction(), result.output)
    assert scores["task_score"] == 0
    assert scores["json_parse_success"] == float(syntax)
    assert scores["schema_validity"] == 0
    assert scores["field_exact_match_mean"] is None


@pytest.mark.parametrize("task", ["grounded_qa", "robustness"])
def test_unsafe_qa_object_is_preserved_and_components_are_unavailable(task: str) -> None:
    example = next(e for e in fixtures() if str(e.task_type) == task)
    original = {"answer": "test", "citations": None, "abstain": False, "confidence": 0.5}
    result = failed(json.dumps(original), example)
    scores = evaluate_output(example, result.output)
    assert scores["task_score"] == 0
    assert scores["json_parse_success"] == 1
    assert scores["schema_compliance"] == 0
    assert scores["citation_precision"] is None
    assert result.output.parsed_json == original


def test_missing_required_fields_preserve_existing_safe_semantics() -> None:
    result = failed("{}")
    assert result.output.raw_content == "{}"
    assert result.output.parsed_json == {}
    old = score_prediction(extraction(), {})
    assert all(evaluate_output(extraction(), result.output)[k] == v for k, v in old.items())


def test_nested_invalid_fields_are_not_coerced() -> None:
    example = next(e for e in fixtures() if str(e.task_type) == "business_summarization")
    original = {
        "summary": "Synthetic",
        "decisions": [],
        "risks": [],
        "action_items": [{"owner": "Team", "action": "Ship", "deadline": 42}],
    }
    result = failed(json.dumps(original), example)
    assert result.output.parsed_json == original
    assert evaluate_output(example, result.output)["schema_compliance"] == 0


@pytest.mark.parametrize(
    "task",
    [
        "grounded_qa",
        "structured_extraction",
        "business_summarization",
        "robustness",
    ],
)
def test_valid_output_metric_parity_for_every_task(task: str) -> None:
    example = next(e for e in fixtures() if str(e.task_type) == task)
    if task in {"grounded_qa", "robustness"}:
        output = {
            "answer": example.gold["answer"],
            "citations": example.gold["evidence"],
            "confidence": 0.8,
            "abstain": False,
        }
    elif task == "structured_extraction":
        output = example.gold["expected"]
    else:
        output = {
            key: example.gold[key] for key in ["summary", "decisions", "action_items", "risks"]
        }
    assert evaluate_output(example, output) == score_prediction(example, output)


def config(path: Path, strategy: str, ceiling: int) -> dict[str, Any]:
    return {
        "dataset": {"path": str(path)},
        "experiment": {"name": "v3-synthetic"},
        "strategy": {"type": strategy, "k": 2, "agents": 2, "rounds": 2},
        "model": {"provider": "ollama", "model": "gemma3:4b", "max_tokens": 700, "seed": 5},
        "budget": {"max_calls": 6, "max_total_tokens": ceiling},
        "provider_retry": {"max_retries": 0, "unknown_usage_policy": "UNKNOWN_USAGE_STOP.v1"},
        "concurrency": {"limit": 1},
        "checkpoint": {"resume_completed_examples": True},
    }


@pytest.mark.parametrize(
    "strategy",
    [
        "single_agent",
        "self_consistency",
        "critic_reviser",
        "multi_agent_debate",
        "adaptive_router",
    ],
)
@pytest.mark.parametrize("ceiling", [4000, 12000])
def test_uniform_policy_checkpoints_then_continues_without_retry(
    tmp_path: Path,
    strategy: str,
    ceiling: int,
) -> None:
    path = tmp_path / "dev.jsonl"
    examples = [extraction().model_copy(update={"id": f"synthetic-{i}"}) for i in range(2)]
    write_jsonl(path, examples)
    provider = ReturnedFailureProvider("broken JSON")
    db = tmp_path / "new.sqlite3"
    with patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}):
        run = asyncio.run(
            run_experiment(
                config(path, strategy, ceiling),
                db_path=db,
                output_dir=tmp_path / "runs",
                failed_output_policy=True,
            )
        )
    store = SQLiteStore(db)
    assert run.status == "COMPLETED"
    assert len(store.get_run_predictions(run.run_id)) == 2
    assert run.aggregate_metrics["mean_task_score"] == 0
    assert all(
        row["failure_type"] == "SCHEMA_FAILURE" for row in store.get_run_failures(run.run_id)
    )
    calls = store.get_run_model_calls(run.run_id)
    assert all(row["attempt_index"] == 0 and row["retry_count"] == 0 for row in calls)
    assert len(provider.requests) == (
        4 if strategy in {"self_consistency", "multi_agent_debate"} else 2
    )
    coverage = component_coverage(store, run.run_id)
    assert coverage["n_total"] == coverage["failed_output_count"] == 2
    assert coverage["components"]["task_score"]["n_scored"] == 2
    assert coverage["components"]["field_exact_match_mean"]["n_scored"] == 0
    with store.connect() as connection:
        unavailable = connection.execute(
            "SELECT metric_status FROM metrics WHERE metric_value IS NULL"
        ).fetchall()
    assert unavailable and all(row[0] == "UNSCORABLE_FAILED_OUTPUT" for row in unavailable)


def test_non_authoritative_failure_still_stops(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    write_jsonl(path, [extraction()])
    provider = ReturnedFailureProvider("broken JSON", unknown=True)
    with (
        patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}),
        pytest.raises(UnknownUsageStop),
    ):
        asyncio.run(
            run_experiment(
                config(path, "single_agent", 12000),
                db_path=tmp_path / "x.db",
                output_dir=tmp_path / "runs",
                failed_output_policy=True,
            )
        )


def test_failed_result_round_trip_is_typed() -> None:
    result = failed('"not an object"')
    assert isinstance(
        FailedPrediction.model_validate_json(result.model_dump_json()).output, FailedOutput
    )


def test_material_evaluator_defects_still_propagate() -> None:
    result = failed('{"auto_renewal":"wrong"}')
    with (
        patch(
            "collectiveeval.metrics.score_prediction", side_effect=RuntimeError("material defect")
        ),
        pytest.raises(RuntimeError, match="material defect"),
    ):
        evaluate_output(extraction(), result.output)


def test_v3_decoder_does_not_repair_malformed_raw_json() -> None:
    raw = 'not JSON {"auto_renewal":"wrong"} trailing text'
    parsed, syntax, source = decode_failure(raw)
    assert parsed is None and syntax is False
    assert source == "UNPARSEABLE_RAW_RETAINED"


def test_previously_parsed_wrapped_object_is_retained_without_value_changes() -> None:
    original = {"auto_renewal": "wrong"}
    raw = "```json\n" + json.dumps(original) + "\n```"
    result = failed(raw)
    assert result.output.raw_content == raw
    assert result.output.parsed_json == original
    assert result.output.decode_source == "PRESERVED_PARSED_OBJECT"
    assert result.output.raw_json_syntax_parsed is False
    assert result.output.json_syntax_parsed is True


def test_unsafe_qa_aggregation_does_not_fabricate_abstention_components(tmp_path: Path) -> None:
    qa = next(e for e in fixtures() if str(e.task_type) == "grounded_qa")
    path = tmp_path / "dev.jsonl"
    write_jsonl(path, [qa])
    provider = ReturnedFailureProvider(
        json.dumps({"answer": "test", "citations": None, "abstain": False, "confidence": 0.5})
    )
    db = tmp_path / "x.db"
    with patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}):
        run = asyncio.run(
            run_experiment(
                config(path, "single_agent", 12000),
                db_path=db,
                output_dir=tmp_path / "runs",
                failed_output_policy=True,
            )
        )
    assert run.aggregate_metrics["mean_task_score"] == 0
    assert "abstention_f1" not in run.aggregate_metrics
    assert "over_abstention_rate" not in run.aggregate_metrics
    coverage = component_coverage(SQLiteStore(db), run.run_id)
    assert coverage["components"]["predicted_abstain"]["n_scored"] == 0


def test_component_coverage_distinguishes_unavailable_and_non_applicable(tmp_path: Path) -> None:
    qa = next(e for e in fixtures() if str(e.task_type) == "grounded_qa")
    path = tmp_path / "dev.jsonl"
    write_jsonl(path, [extraction(), qa])
    provider = ReturnedFailureProvider("broken JSON")
    db = tmp_path / "x.db"
    with patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}):
        run = asyncio.run(
            run_experiment(
                config(path, "single_agent", 12000),
                db_path=db,
                output_dir=tmp_path / "runs",
                failed_output_policy=True,
            )
        )
    coverage = component_coverage(SQLiteStore(db), run.run_id)
    assert coverage["n_total"] == coverage["failed_output_count"] == 2
    assert coverage["components"]["citation_precision"] == {
        "n_total": 2,
        "n_scored": 0,
        "n_applicable": 1,
        "failed_output_count": 1,
        "n_unavailable": 1,
        "n_not_applicable": 1,
    }
    assert coverage["components"]["task_score"]["n_scored"] == 2


class EscalationFailureProvider(ReturnedFailureProvider):
    def __init__(self, failed_role: str) -> None:
        super().__init__("")
        self.failed_role = failed_role

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        self.requests.append(request)
        output: dict[str, Any] = {"auto_renewal": False}
        if request.role == "critic":
            output = {"needs_revision": True, "issues": ["Synthetic issue"]}
        if request.role == self.failed_role:
            output = (
                {"needs_revision": "wrong-type", "issues": []}
                if request.role == "critic"
                else {"auto_renewal": "wrong-type"}
            )
        return ModelOutput(
            raw_output=json.dumps(output),
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            usage_source="PROVIDER_REPORTED",
        )


@pytest.mark.parametrize("ceiling", [4000, 12000])
@pytest.mark.parametrize(
    "route,failed_role,calls_per_example",
    [
        ("critic", "generator", 2),
        ("critic", "critic", 3),
        ("critic", "reviser", 4),
        ("debate", "agent_1", 3),
    ],
)
def test_router_nested_failures_retain_call_identity_and_frozen_settings(
    tmp_path: Path, ceiling: int, route: str, failed_role: str, calls_per_example: int
) -> None:
    from collectiveeval.heldout_v3 import accounting
    from collectiveeval.pilot_analysis import ReadOnlyPilotStore

    path, db = tmp_path / "dev.jsonl", tmp_path / "x.db"
    examples = [extraction().model_copy(update={"id": f"synthetic-{i}"}) for i in range(2)]
    write_jsonl(path, examples)
    recipe = config(path, "adaptive_router", ceiling)
    recipe["model"].update(
        temperature=0.2, top_p=1.0, provider_options={"model_digest": "synthetic"}
    )
    provider = EscalationFailureProvider(failed_role)
    with (
        patch("collectiveeval.router.HeuristicRouter.route", return_value=route),
        patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}),
    ):
        run = asyncio.run(
            run_experiment(
                recipe,
                db_path=db,
                output_dir=tmp_path / "runs",
                failed_output_policy=True,
            )
        )
    assert run.status == "COMPLETED"
    assert len(provider.requests) == 2 * calls_per_example
    store = ReadOnlyPilotStore(db)
    calls = accounting(store, run.run_id, recipe["model"])
    assert calls["parse_failures"] == 2
    assert calls["attempts"] == 2 * calls_per_example
    for row in store.get_run_predictions(run.run_id):
        output = json.loads(row["output_json"])
        assert row["strategy"] == "adaptive_router"
        assert output["strategy"] == (
            "critic_reviser" if route == "critic" else "multi_agent_debate"
        )
        assert output["role"] == failed_role
    assert all(
        row["retry_count"] == row["attempt_index"] == 0
        for row in store.get_run_model_calls(run.run_id)
    )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE provider_attempts SET temperature=0.9 WHERE role=?", (failed_role,))
    with pytest.raises(ValueError, match="generation settings"):
        accounting(store, run.run_id, recipe["model"])
