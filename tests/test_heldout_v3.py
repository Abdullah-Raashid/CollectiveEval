from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from test_failed_outputs import ReturnedFailureProvider, config, extraction

from collectiveeval.core import ModelOutput, ProviderRequest, TokenUsage
from collectiveeval.datasets import file_sha256, write_jsonl
from collectiveeval.failed_outputs import PENDING
from collectiveeval.heldout_v3 import (
    execute_v3,
    prepare_seed,
    verify_carried_rows,
    verify_resume,
)
from collectiveeval.pilot_analysis import ReadOnlyPilotStore
from collectiveeval.providers import ProviderError
from collectiveeval.runner import run_experiment
from collectiveeval.storage import SQLiteStore


class PartialProvider(ReturnedFailureProvider):
    async def complete(self, request: ProviderRequest) -> ModelOutput:
        self.requests.append(request)
        output = {"auto_renewal": False}
        if request.example.id == "failed-synthetic" and request.role == "agent_1":
            output = {"auto_renewal": "wrong-type"}
        return ModelOutput(
            raw_output=json.dumps(output),
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            usage_source="PROVIDER_REPORTED",
        )


def partial_parent(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], ReturnedFailureProvider]:
    parent = tmp_path / "parent"
    (parent / "execution").mkdir(parents=True)
    dataset = tmp_path / "dev.jsonl"
    write_jsonl(
        dataset,
        [
            extraction().model_copy(update={"id": "completed-synthetic"}),
            extraction().model_copy(update={"id": "failed-synthetic"}),
        ],
    )
    recipe = config(dataset, "multi_agent_debate", 12000)
    recipe["model"].update(
        temperature=0.5, top_p=1.0, provider_options={"model_digest": "synthetic"}
    )
    provider = PartialProvider("")
    db = parent / "execution/heldout.sqlite3"
    with (
        patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}),
        pytest.raises(ProviderError),
    ):
        asyncio.run(run_experiment(recipe, db_path=db, output_dir=tmp_path / "old-runs"))
    store = ReadOnlyPilotStore(db)
    run = store._fetch_one("SELECT id FROM runs")
    assert run is not None
    certificate = {"partial_run_id": run["id"], "failed_example_id": "failed-synthetic"}
    # Emulate the original immutable v2 metric schema for a real copy-only migration test.
    with sqlite3.connect(db) as conn:
        conn.executescript("""ALTER TABLE metrics RENAME TO metrics_new;
            DROP INDEX IF EXISTS example_metric_checkpoint;
            CREATE TABLE metrics(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,
            example_id TEXT,metric_name TEXT NOT NULL,metric_value REAL NOT NULL);
            INSERT INTO metrics
                SELECT id,run_id,example_id,metric_name,metric_value FROM metrics_new;
            DROP TABLE metrics_new;""")
    return parent, certificate, recipe, provider


def test_seed_failure_is_checkpointed_without_gold_or_new_request(tmp_path: Path) -> None:
    parent, certificate, _, provider = partial_parent(tmp_path)
    before = file_sha256(parent / "execution/heldout.sqlite3")
    requests = len(provider.requests)
    seed = tmp_path / "seed.sqlite3"
    with patch(
        "collectiveeval.failed_outputs.evaluate_output", side_effect=AssertionError("no scoring")
    ):
        prepare_seed(parent, seed, certificate)
    assert len(provider.requests) == requests
    assert file_sha256(parent / "execution/heldout.sqlite3") == before
    store = ReadOnlyPilotStore(seed)
    row = store._fetch_one("""SELECT output_json,metadata_json FROM predictions
        WHERE example_id='failed-synthetic' """)
    assert row is not None
    output = json.loads(row["output_json"])
    assert output["parsed_json"] == {"auto_renewal": "wrong-type"}
    assert output["raw_content"] == '{"auto_renewal": "wrong-type"}'
    assert json.loads(row["metadata_json"])["evaluation_status"] == PENDING
    assert (
        store._fetch_one("SELECT COUNT(*) AS n FROM metrics WHERE example_id='failed-synthetic'")[
            "n"
        ]
        == 0
    )
    assert store._fetch_one("SELECT COUNT(*) AS n FROM predictions")["n"] == 2
    assert (
        store.get_run_failures(certificate["partial_run_id"])[0]["failure_type"] == "SCHEMA_FAILURE"
    )
    with pytest.raises(ValueError, match="immutable"):
        SQLiteStore(parent / "execution/heldout.sqlite3")


def test_resume_evaluates_preserved_failed_checkpoint_without_regeneration(tmp_path: Path) -> None:
    parent, certificate, recipe, _ = partial_parent(tmp_path)
    seed = tmp_path / "seed.sqlite3"
    prepare_seed(parent, seed, certificate)
    carried = tmp_path / "carried.sqlite3"
    shutil.copyfile(seed, carried)
    provider = ReturnedFailureProvider("MUST NEVER BE SENT")
    before = file_sha256(parent / "execution/heldout.sqlite3")
    with patch("collectiveeval.runner.build_provider_registry", return_value={"ollama": provider}):
        result = asyncio.run(
            run_experiment(
                recipe, db_path=seed, output_dir=tmp_path / "new-runs", failed_output_policy=True
            )
        )
    assert provider.requests == []
    assert result.run_id == certificate["partial_run_id"] and result.examples == 2
    assert file_sha256(parent / "execution/heldout.sqlite3") == before
    store = ReadOnlyPilotStore(seed)
    assert store.get_run(result.run_id)["status"] == "COMPLETED"
    assert len(store.get_run_model_calls(result.run_id)) == 6
    values = store.example_metric_values(result.run_id, "task_score")
    assert set(values) == {"completed-synthetic", "failed-synthetic"}
    assert result.aggregate_metrics["mean_task_score"] == sum(values.values()) / 2
    verify_carried_rows(carried, seed)


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE provider_attempts SET usage_status='UNKNOWN_NOT_RETURNED' WHERE id=1",
        "UPDATE provider_attempts SET outcome='STARTED' WHERE id=1",
        "UPDATE logical_model_calls SET outcome='STARTED' "
        "WHERE logical_call_id=(SELECT logical_call_id FROM provider_attempts WHERE id=1)",
        "DELETE FROM predictions WHERE example_id='failed-synthetic'",
        "UPDATE experiments SET config_hash='unfrozen'",
        "UPDATE experiments SET config_json='{}'",
        "UPDATE predictions SET usage_json=json_set(usage_json,'$.total_tokens',999)",
        "UPDATE predictions SET output_json=json_set(output_json,'$.raw_content','altered') "
        "WHERE example_id='failed-synthetic'",
        "UPDATE predictions SET example_id='wrong-cohort' WHERE example_id='completed-synthetic'",
        "UPDATE runs SET status='COMPLETED'",
    ],
)
def test_resume_does_not_weaken_unknown_inflight_or_identity_guards(
    tmp_path: Path, mutation: str
) -> None:
    parent, certificate, recipe, _ = partial_parent(tmp_path)
    seed = tmp_path / "seed.sqlite3"
    prepare_seed(parent, seed, certificate)
    directory = tmp_path / "protocol"
    (directory / "configs").mkdir(parents=True)
    path = directory / "configs/debate.json"
    path.write_text(json.dumps(recipe))
    store = ReadOnlyPilotStore(seed)
    experiment = store.get_run_experiment(certificate["partial_run_id"])
    protocol = {
        "matrix": [
            {},
            {
                "config_path": "configs/debate.json",
                "resolved_config_sha256": experiment["config_hash"],
            },
        ],
        "benchmark": {
            "test_sha256": experiment["dataset_hash"],
            "test_path": recipe["dataset"]["path"],
        },
    }
    verify_resume(seed, protocol, directory)
    if mutation == "UPDATE runs SET status='COMPLETED'":
        # A partial completed run must halt even when its other accounting is intact.
        with sqlite3.connect(seed) as conn:
            conn.execute("DELETE FROM predictions WHERE example_id='failed-synthetic'")
    with sqlite3.connect(seed) as conn:
        conn.execute(mutation)
    with pytest.raises(ValueError):
        verify_resume(seed, protocol, directory)


def test_failed_v3_preflight_never_dispatches(tmp_path: Path) -> None:
    with (
        patch("collectiveeval.heldout_v3.verify_v3", side_effect=ValueError("bad hash")),
        patch("collectiveeval.heldout_v3.run_experiment_from_path") as run,
        pytest.raises(ValueError, match="hash"),
    ):
        execute_v3(tmp_path, "bad")
    run.assert_not_called()


@pytest.mark.parametrize(
    "table", ["provider_attempts", "logical_model_calls", "predictions", "metrics"]
)
def test_carried_rows_are_opaque_and_cannot_be_modified(tmp_path: Path, table: str) -> None:
    parent, certificate, _, _ = partial_parent(tmp_path)
    seed, working = tmp_path / "seed.sqlite3", tmp_path / "working.sqlite3"
    prepare_seed(parent, seed, certificate)
    shutil.copyfile(seed, working)
    verify_carried_rows(seed, working)
    mutations = {
        "provider_attempts": "UPDATE provider_attempts SET output_tokens=999 WHERE id=1",
        "logical_model_calls": "UPDATE logical_model_calls SET role='altered'",
        "predictions": "UPDATE predictions SET output_json='{}' WHERE id=1",
        "metrics": "UPDATE metrics SET metric_value=999 WHERE id=1",
    }
    with sqlite3.connect(working) as conn:
        conn.execute(mutations[table])
    with pytest.raises(ValueError, match="Inherited"):
        verify_carried_rows(seed, working)
