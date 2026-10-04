"""Offline corrected-view boundaries; never creates a real provider or opens TEST."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from collectiveeval.phase8_corrective_analysis import (
    CorrectedPilotStore,
    compare_correction,
    completed_scope,
    diversity,
    trajectory_summary,
)


def observation(example_id: str = "dev-fixture") -> dict:
    return {
        "example_id": example_id,
        "task_score": 0.5,
        "output": {"answer": "30 days"},
        "metadata": {"route": "critic", "budget_events": []},
        "candidates": [
            {"output": {"answer": "30 days", "confidence": 0.7}},
            {"output": {"answer": "30 days", "confidence": 0.8}},
        ],
        "calls": [
            {
                "role": "sample",
                "requested_seed": seed,
                "metadata_json": json.dumps({"post_call_budget_overrun": False}),
            }
            for seed in (100, 101)
        ],
        "model_calls": 2,
        "total_tokens": 120,
    }


def test_corrected_store_dispatches_without_merging_evidence() -> None:
    old, new = MagicMock(), MagicMock()
    old.get_run.return_value = {"id": "old"}
    new.get_run.return_value = {"id": "new"}
    store = CorrectedPilotStore({"old": old, "new": new})
    assert store.get_run("old") == {"id": "old"}
    assert store.get_run("new") == {"id": "new"}
    with pytest.raises(KeyError):
        store.get_run("diagnostic")
    old.insert_prediction.assert_not_called()
    new.insert_prediction.assert_not_called()


def test_corrected_store_rejects_non_example_queries() -> None:
    store = CorrectedPilotStore({"old": MagicMock()})
    with pytest.raises(ValueError, match="Only DEV example reads"):
        store._fetch_one("DELETE FROM predictions")


def test_candidate_answer_agreement_is_not_full_identity() -> None:
    result = diversity([observation()])
    assert result["agreement_rate"] == 1
    assert result["disagreement_rate"] == 0
    assert result["exact_identity_rate"] == 0
    assert result["examples"][0]["requested_seeds"] == [100, 101]


def test_diversity_does_not_count_one_candidate_as_agreement() -> None:
    row = observation()
    row["candidates"] = row["candidates"][:1]
    result = diversity([row])
    assert result["paired_candidates"] == 0
    assert result["agreement_rate"] is None


def test_correction_comparison_requires_exact_pairing() -> None:
    with pytest.raises(ValueError, match="exactly paired"):
        compare_correction([observation("dev-a")], [observation("dev-b")])
    new = observation()
    new["task_score"] = 0.7
    new["output"] = {"answer": "31 days"}
    result = compare_correction([observation()], [new])
    assert result["changed_final_predictions"] == 1
    assert result["paired_score_delta"]["mean_difference"] == pytest.approx(0.2)


def test_trajectory_counts_pre_rejection_separately_from_post_overrun() -> None:
    row = observation()
    row["metadata"]["budget_events"] = [{"event_type": "PRE_CALL_BUDGET_REJECTION"}]
    row["calls"][0]["metadata_json"] = json.dumps({"post_call_budget_overrun": True})
    result = trajectory_summary([row])
    assert result["pre_call_budget_rejections"] == 1
    assert result["post_call_overruns"] == 1
    assert result["route_counts"] == {"critic": 1}


def test_corrected_analysis_refuses_incomplete_scope_before_opening_db(tmp_path: Path) -> None:
    (tmp_path / "rerun_scope.json").write_text(json.dumps({"entries": []}))
    (tmp_path / "phase8_2_status.json").write_text(json.dumps({"run_ids": []}))
    with pytest.raises(ValueError, match="All five approved"):
        completed_scope(tmp_path, tmp_path)


def test_corrected_analysis_refuses_unresolved_methodology_stop(tmp_path: Path) -> None:
    (tmp_path / "methodology_stop.json").write_text(json.dumps({"unresolved": True}))
    with pytest.raises(ValueError, match="Unresolved live methodology blocker"):
        completed_scope(tmp_path, tmp_path)
