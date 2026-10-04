"""Validate consistency between SQLite run records and file artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from collectiveeval.storage import SQLiteStore


def verify_run_artifacts(store: SQLiteStore, run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    artifact_dir = Path(str(run.get("artifact_dir") or ""))
    discrepancies: list[str] = []
    if not artifact_dir.exists():
        return {"ok": False, "run_id": run_id, "discrepancies": ["artifact_dir_missing"]}

    manifest = _read_json(artifact_dir / "manifest.json")
    config = yaml.safe_load((artifact_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    predictions = _read_jsonl(artifact_dir / "predictions.jsonl")
    metrics = _read_json(artifact_dir / "metrics.json")
    failures = _read_json(artifact_dir / "failures.json")

    if manifest.get("run_id") != run_id:
        discrepancies.append("manifest_run_id_mismatch")
    if manifest.get("experiment_id") != run.get("experiment_id"):
        discrepancies.append("manifest_experiment_id_mismatch")
    db_predictions = store.get_run_predictions(run_id)
    if len(predictions) != len(db_predictions):
        discrepancies.append("prediction_count_mismatch")
    db_example_ids = sorted(row["example_id"] for row in db_predictions)
    artifact_example_ids = sorted(str(row.get("example_id")) for row in predictions)
    if db_example_ids != artifact_example_ids:
        discrepancies.append("prediction_example_ids_mismatch")
    db_metrics = store.get_run_metrics(run_id)
    aggregate_metrics = metrics.get("aggregate", {}) if isinstance(metrics, dict) else {}
    db_aggregate_names = {row["metric_name"] for row in db_metrics if row.get("example_id") is None}
    if set(aggregate_metrics) - db_aggregate_names:
        discrepancies.append("aggregate_metric_names_mismatch")
    db_failures = store.get_run_failures(run_id)
    if isinstance(failures, list) and len(failures) != len(db_failures):
        discrepancies.append("failure_count_mismatch")
    if manifest.get("config_hash") and not config:
        discrepancies.append("config_artifact_empty")
    return {"ok": not discrepancies, "run_id": run_id, "discrepancies": discrepancies}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
