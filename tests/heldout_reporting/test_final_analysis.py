from __future__ import annotations

import copy
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from analysis.heldout_final import (
    ROOT,
    STRATEGIES,
    finite_equal,
    frozen_pair,
    frozen_summary,
    mechanism,
    pareto_dominance,
    validate_cohort,
)
from analysis.heldout_render import (
    CONDITIONS,
    application_report,
    json_text,
    public_view,
    safe_prediction,
    select_representatives,
)
from collectiveeval.pilot_analysis import ReadOnlyPilotStore
from collectiveeval.statistics import paired_bootstrap_comparison, summary_stats

SETTINGS = {"bootstrap_replicates": 1000, "confidence": 0.95, "seed": 20261003}


@pytest.mark.parametrize("actual", [["a"], ["a", "a"], ["a", "c"], ["a", "b", "c"]])
def test_cohort_rejects_missing_duplicate_and_extra_examples(actual: list[str]) -> None:
    with pytest.raises(ValueError, match="cohort"):
        validate_cohort(actual, ["a", "b"], "synthetic")


def test_cohort_allows_order_changes_without_exclusions() -> None:
    validate_cohort(["b", "a"], ["a", "b"], "synthetic")


def test_pairing_refuses_silent_intersection() -> None:
    with pytest.raises(ValueError, match="exclude"):
        frozen_pair({"a": 1, "b": 0}, {"a": 1}, "task_score", SETTINGS)
    with pytest.raises(ValueError, match="exclude"):
        frozen_pair({}, {}, "task_score", SETTINGS)


def test_statistics_delegate_to_unchanged_frozen_bootstrap() -> None:
    values = [0.0, 0.25, 0.5, 1.0]
    assert (
        frozen_summary(values, SETTINGS)
        == summary_stats(values, n_bootstrap=1000, confidence=0.95, seed=20261003).to_dict()
    )
    baseline, contender = {"a": 0.25, "b": 1.0}, {"a": 0.5, "b": 0.0}
    assert (
        frozen_pair(baseline, contender, "task_score", SETTINGS)
        == paired_bootstrap_comparison(
            baseline=baseline,
            contender=contender,
            metric="task_score",
            n_bootstrap=1000,
            confidence=0.95,
            seed=20261003,
        ).to_dict()
    )


@pytest.mark.parametrize("actual", [math.nan, math.inf, -math.inf, 1.1])
def test_nonfinite_or_inconsistent_accounting_halts(actual: float) -> None:
    with pytest.raises(ValueError, match="mismatch"):
        finite_equal(actual, 1.0, "synthetic")


def test_budget_stop_uses_persisted_admission_events() -> None:
    rows = [
        {
            "example_id": "synthetic",
            "task": "grounded_qa",
            "task_score": 0.5,
            "metadata": {"budget_events": [{"event_type": "PRE_CALL_BUDGET_REJECTION"}]},
        }
    ]
    result = mechanism(rows, "single_agent")
    assert result["budget_stopped_examples"] == 1
    assert result["failed_outputs"] == 0


def test_pareto_requires_one_strict_improvement_and_stays_condition_local() -> None:
    def point(q: float, tokens: int) -> dict[str, Any]:
        return {"metrics": {"task_score": {"mean": q}, "total_tokens": {"mean": tokens}}}

    result = pareto_dominance(
        {"a": point(0.6, 100), "tie": point(0.6, 100), "b": point(0.5, 100), "c": point(0.7, 200)},
        "total_tokens",
    )
    assert result["a"]["nondominated"] and result["tie"]["nondominated"]
    assert result["c"]["nondominated"]
    assert result["b"]["dominated_by"] == ["a", "tie"]


def test_private_prediction_rendering_drops_raw_and_hidden_reasoning() -> None:
    assert safe_prediction(
        {
            "answer": "ok",
            "raw_content": "secret",
            "analysis": "secret",
            "rationale": "secret",
            "fields": [{"value": 1, "chain_of_thought": "secret"}],
        }
    ) == {"answer": "ok", "fields": [{"value": 1}]}


def test_case_selection_is_stable_and_not_maximum_gain() -> None:
    rows: dict[str, Any] = {c: {s: [] for s in STRATEGIES} for c in CONDITIONS}
    results: dict[str, Any] = {
        c: {s: {"mechanism": {"details": []}} for s in STRATEGIES} for c in CONDITIONS
    }
    for condition in CONDITIONS:
        for eid, gain in (("z-synthetic", 0.5), ("a-synthetic", 0.1)):
            base = {"example_id": eid, "task_score": 0.0, "failure_types": []}
            row = {**base, "task_score": gain}
            rows[condition]["single_agent"].append(base)
            rows[condition]["critic_reviser"].append(row)
            results[condition]["single_agent"]["mechanism"]["details"].append(
                {"example_id": eid, "failed_output": False}
            )
            results[condition]["critic_reviser"]["mechanism"]["details"].append(
                {"example_id": eid, "failed_output": False, "change_from_initial": gain}
            )
    data = {"strategy_results": results}
    selected = select_representatives(data, rows)
    first = selected[0]
    assert first["eligible_count"] == 4
    assert first["selection"]["condition"] == "natural"
    assert first["selection"]["row"]["example_id"] == "a-synthetic"
    for strategies in rows.values():
        for cases in strategies.values():
            cases.reverse()
    assert select_representatives(data, rows) == selected
    assert selected[-1]["selection"] is None


def test_application_lengths_and_resume_scope() -> None:
    report = application_report()
    portfolio = report.split("## Portfolio Description (100 Words)\n\n")[1].split("\n\n##")[0]
    technical = report.split("## Technical Description (200 Words)\n\n")[1].split("\n\n##")[0]
    assert len(portfolio.split()) == 100
    assert len(technical.split()) == 200
    bullets = report.split("## Resume Bullets\n\n")[1].split("\n\n##")[0]
    assert len(bullets.splitlines()) == 4
    assert "synthetic" in report and "interval spans zero" in report


def test_store_is_read_only_at_sqlite_boundary(tmp_path: Path) -> None:
    db = tmp_path / "evidence.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE evidence(value TEXT)")
        conn.execute("INSERT INTO evidence VALUES ('retained')")
    before = db.read_bytes()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        ReadOnlyPilotStore(db)._fetch_all("DELETE FROM evidence")
    assert db.read_bytes() == before


@pytest.fixture
def aggregate_export() -> dict[str, Any]:
    path = ROOT / "docs/heldout_results.json"
    if not path.exists():
        pytest.skip("aggregate export not present")
    return json.loads(path.read_text())


def test_public_export_is_idempotent_and_whitelisted(aggregate_export: dict[str, Any]) -> None:
    source = copy.deepcopy(aggregate_export)
    source["private_rows"] = {"gold": "secret", "raw": "secret"}
    for condition in CONDITIONS:
        for strategy in STRATEGIES:
            row = source["strategy_results"][condition][strategy]
            row["private_outputs"] = "secret"
            row["mechanism"]["details"] = [{"example_id": "private-test", "gold": "secret"}]
            row["mechanism"]["future_private_field"] = "secret"
    exported = public_view(source)
    assert "secret" not in json_text(exported)
    assert exported == aggregate_export


def test_public_results_have_full_failure_denominators(aggregate_export: dict[str, Any]) -> None:
    strategies = aggregate_export["strategy_results"]
    assert (
        sum(s["metrics"]["task_score"]["n"] for d in strategies.values() for s in d.values())
        == 2000
    )
    matched_router = strategies["matched_tokens"]["adaptive_router"]
    coverage = matched_router["component_coverage"]
    assert coverage["n_total"] == 200 and coverage["failed_output_count"] == 6
    assert coverage["components"]["task_score"]["n_scored"] == 200
    assert coverage["components"]["action_item_correctness"]["n_unavailable"] == 6
    assert aggregate_export["accounting_totals"]["failed_attempts"] == 0
    assert aggregate_export["accounting_totals"]["parse_failures"] == 8
    assert matched_router["returned_output_failures"]["length_terminated"] == 6
    change = aggregate_export["condition_comparison"]["adaptive_router"][
        "quality_change_decomposition"
    ]
    assert change["failed_matched_output"]["changed_scores"] == 6
    assert change["valid_matched_output"]["changed_scores"] == 1
    assert sum(g["mean_delta_contribution_over_full_cohort"] for g in change.values()) == (
        pytest.approx(-0.008)
    )


def test_public_material_contains_no_private_example_ids_or_machine_paths(
    aggregate_export: dict[str, Any],
) -> None:
    text = json_text(aggregate_export)
    assert not re.search(r"v3_1-(?:qa|ext|sum|rob)-\d", text)
    assert "/Users/" not in text
    assert "raw_content" not in text and '"gold"' not in text
    assert all(
        "details" not in s["mechanism"]
        for condition in aggregate_export["strategy_results"].values()
        for s in condition.values()
    )
    public_md = (ROOT / "docs/heldout_results.md").read_text()
    assert "/Users/" not in public_md and "```json" not in public_md
    assert "TINY" in public_md and "hard caps" in public_md


def test_two_conditions_are_not_pooled_and_sources_are_bound(
    aggregate_export: dict[str, Any],
) -> None:
    assert set(aggregate_export["strategy_results"]) == set(CONDITIONS)
    natural = aggregate_export["strategy_results"]["natural"]
    matched = aggregate_export["strategy_results"]["matched_tokens"]
    assert (
        natural["adaptive_router"]["metrics"]["task_score"]["mean"]
        > matched["adaptive_router"]["metrics"]["task_score"]["mean"]
    )
    assert len(aggregate_export["analysis_source_hashes"]) == 3
    assert aggregate_export["no_model_calls"] is True
    assert (
        "paired_deltas" in aggregate_export and "raw_evidence_inventory_sha256" in aggregate_export
    )
