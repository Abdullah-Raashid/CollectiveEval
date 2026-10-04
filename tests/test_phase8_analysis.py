import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collectiveeval.budget import BudgetExceeded, BudgetLedger, InferenceBudget
from collectiveeval.context import StrategyContext
from collectiveeval.core import (
    BenchmarkExample,
    ModelOutput,
    ModelSpec,
    TaskType,
    TokenUsage,
    UsageSource,
)
from collectiveeval.pilot import provider_accounting, strategy_result_summary
from collectiveeval.pilot_analysis import ReadOnlyPilotStore, analyze, reconstruct_route
from collectiveeval.providers import MockProvider
from collectiveeval.reporting import validate_matched_budget_compatibility
from collectiveeval.strategies import AdaptiveRouterStrategy, SelfConsistency


def fake_store() -> MagicMock:
    store = MagicMock()
    store.get_run.side_effect = lambda r: {"strategy": "single_agent", "id": r}
    store.get_run_experiment.side_effect = lambda r: {
        "dataset_hash": "same",
        "config_json": json.dumps(
            {
                "phase8": {"condition": r},
                "budget": {"max_total_tokens": 4000},
                "budget_policy": {"scientific_budget": {"scope": "PER_EXAMPLE"}},
            }
        ),
    }
    store.get_run_metrics.return_value = []
    store.get_run_model_calls.side_effect = lambda r: [
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "estimated_cost_usd": 0,
            "retry_count": 0,
            "normalized_error": None,
            "usage_source": "PROVIDER_REPORTED",
            "role": "generator",
            "prompt_version": "task-contracts.v1.grounded_qa",
        }
    ]
    store.run_task_set.return_value = {
        "example_ids": ["dev-a"],
        "splits": ["dev"],
        "task_types": ["grounded_qa"],
    }
    return store


def test_condition_separation_and_duplicate_rejection() -> None:
    store = fake_store()
    result = strategy_result_summary(store, ["natural", "matched_tokens"])
    assert result["natural"]["single_agent"]["run_id"] == "natural"
    assert result["matched_tokens"]["single_agent"]["run_id"] == "matched_tokens"
    with pytest.raises(ValueError, match="Duplicate"):
        strategy_result_summary(store, ["natural", "natural"])


def test_combined_accounting_uses_both_condition_estimates(tmp_path: Path) -> None:
    estimate = {"estimated_model_calls": 2, "estimated_total_tokens": 40}
    (tmp_path / "workload_estimate.json").write_text(
        json.dumps({"natural_condition": estimate, "matched_tokens_condition": estimate})
    )
    with patch("collectiveeval.pilot.PILOT_DIR", tmp_path):
        result = provider_accounting(fake_store(), ["natural", "matched_tokens"], estimate)
    assert result["combined"]["actual"]["model_calls"] == 2
    assert result["combined"]["pre_run_estimate"]["estimated_model_calls"] == 4
    assert result["combined"]["pre_run_estimate"]["estimated_total_tokens"] == 80


def test_role_aware_prompt_check_retains_generator_invariant() -> None:
    store = fake_store()
    generator = {
        "role": "generator",
        "prompt_version": "task-contracts.v1.grounded_qa",
        "input_tokens": 10,
        "output_tokens": 5,
        "retry_count": 0,
    }
    critic = {
        "role": "critic",
        "prompt_version": "task-contracts.v1.grounded_qa.critic-contract.v1",
        "input_tokens": 10,
        "output_tokens": 5,
        "retry_count": 0,
    }
    store.get_run_model_calls.side_effect = lambda r: (
        [generator, critic] if r == "b" else [generator]
    )
    assert validate_matched_budget_compatibility(store, ["a", "b"])["ok"]
    store.get_run_model_calls.side_effect = lambda r: (
        [{**generator, "prompt_version": "changed-contract"}] if r == "b" else [generator]
    )
    result = validate_matched_budget_compatibility(store, ["a", "b"])
    assert "PROMPT_VERSION_MISMATCH" in result["reason_codes"]


def test_router_reconstruction_does_not_guess_acceptance_or_mutate_metadata() -> None:
    metadata = {"budget_exhausted": True}
    result = reconstruct_route(metadata, [{"role": "cheap_single_agent"}, {"role": "generator"}])
    assert result["route"] == "critic"
    assert result["route_source"] == "derived_from_call_trajectory"
    assert metadata == {"budget_exhausted": True}
    assert reconstruct_route({}, [{"role": "cheap_single_agent"}])["route"] == "unknown"
    assert reconstruct_route({}, [{"role": "agent_0"}])["route"] == "debate"


def test_read_only_store_cannot_write(tmp_path: Path) -> None:
    db = tmp_path / "raw.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE evidence (value INTEGER)")
    before = db.read_bytes()
    store = ReadOnlyPilotStore(db)
    with store.connect() as connection, pytest.raises(sqlite3.OperationalError):
        connection.execute("INSERT INTO evidence VALUES (1)")
    assert db.read_bytes() == before


def test_future_router_preserves_route_and_initial_answer_on_budget_failure() -> None:
    example = BenchmarkExample(
        id="dev-fixture",
        task_type=TaskType.GROUNDED_QA,
        input={
            "question": "deadline?",
            "source_text": "30 days",
            "evidence_units": [{"id": "doc", "text": "30 days"}],
        },
        gold={"answer": "30 days", "answerable": True, "evidence": ["doc"]},
        metadata={"source": "unit-test", "difficulty": "hard", "split": "dev", "tags": ["qa"]},
    )
    context = StrategyContext(
        {"mock": MockProvider(mock_mode="gold_fixture")},
        BudgetLedger(InferenceBudget(max_calls=6), run_id="test"),
    )
    strategy = AdaptiveRouterStrategy(cheap_model=ModelSpec())
    strategy.router = MagicMock()
    strategy.router.route.return_value = "critic"
    strategy.router.uncertainty_score.return_value = 0.8
    strategy.critic_strategy.run = AsyncMock(side_effect=BudgetExceeded("token budget"))
    result = asyncio.run(strategy.run(example, context))
    assert result.metadata["route"] == "critic"
    assert result.metadata["budget_exhausted"]
    assert result.output["answer"] == "30 days"


def test_completed_pilot_offline_report_and_immutability(tmp_path: Path) -> None:
    source = Path("reports/pilot_v1").resolve()
    if not (source / "phase8.sqlite3").exists():
        pytest.skip("Completed pilot evidence unavailable")
    for name in ("phase8.sqlite3", "runs"):
        (tmp_path / name).symlink_to(source / name)
    for name in ("phase8_status.json", "workload_estimate.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    with patch(
        "collectiveeval.providers.OpenAICompatibleProvider.complete",
        side_effect=AssertionError("No real inference allowed"),
    ):
        result = analyze(tmp_path)
    assert set(result["strategy_results"]) == {"natural", "matched_tokens"}
    assert all(len(v) == 5 for v in result["strategy_results"].values())
    assert result["provider_accounting"]["combined"]["actual"]["model_calls"] == 706
    assert result["router_summary"]["matched_tokens"]["route_counts"] == {
        "accept": 20,
        "critic": 11,
        "debate": 1,
    }
    assert '"raw_evidence_unchanged": true' in (tmp_path / "phase8_integrity_report.md").read_text()


def test_real_prompt_budget_estimate_does_not_depend_on_gold_or_model_metadata() -> None:
    example = BenchmarkExample(
        id="dev-fixture",
        task_type=TaskType.GROUNDED_QA,
        input={
            "question": "deadline?",
            "source_text": "30 days",
            "evidence_units": [{"id": "doc", "text": "30 days"}],
        },
        gold={"answer": "30 days", "answerable": True, "evidence": ["doc"]},
        metadata={"source": "unit-test", "difficulty": "hard", "split": "dev", "tags": ["qa"]},
    )
    estimates = []
    for gold_answer, identity in (("30 days", "short"), ("x" * 10000, "x" * 10000)):
        ledger = BudgetLedger(InferenceBudget(max_total_tokens=4000), run_id="unit-test")
        provider = MagicMock()
        provider.complete = AsyncMock(
            return_value=ModelOutput(
                content={
                    "answer": "30 days",
                    "citations": ["doc"],
                    "confidence": 0.8,
                    "abstain": False,
                },
                confidence=0.8,
                usage=TokenUsage(input_tokens=20, output_tokens=10),
                usage_source=UsageSource.PROVIDER_REPORTED,
            )
        )
        context = StrategyContext({"ollama": provider}, ledger)
        with patch.object(ledger, "reserve_call_start", wraps=ledger.reserve_call_start) as reserve:
            asyncio.run(
                context.call_model(
                    example=example.model_copy(
                        update={"gold": {**example.gold, "answer": gold_answer}}
                    ),
                    model=ModelSpec(
                        provider="ollama",
                        model="gemma3:4b",
                        provider_options={"model_identity": identity},
                    ),
                    strategy="single_agent",
                    role="generator",
                )
            )
            estimates.append(reserve.call_args.args[0])
    assert estimates[0] == estimates[1]


def test_self_consistency_samples_use_distinct_requested_seeds() -> None:
    example = BenchmarkExample(
        id="dev-fixture",
        task_type=TaskType.GROUNDED_QA,
        input={
            "question": "deadline?",
            "source_text": "30 days",
            "evidence_units": [{"id": "doc", "text": "30 days"}],
        },
        gold={"answer": "30 days", "answerable": True, "evidence": ["doc"]},
        metadata={"source": "unit-test", "difficulty": "hard", "split": "dev", "tags": ["qa"]},
    )
    context = StrategyContext(
        {"mock": MockProvider(mock_mode="gold_fixture")},
        BudgetLedger(InferenceBudget(max_calls=4), run_id="unit-test"),
    )
    asyncio.run(SelfConsistency(k=2, model=ModelSpec(seed=100)).run(example, context))
    assert [r.requested_seed for r in context.ledger.records] == [100, 101]
