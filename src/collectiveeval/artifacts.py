"""Experiment artifact writing."""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from collectiveeval.config import redact_secrets
from collectiveeval.failed_outputs import PredictionResult


def write_run_artifacts(
    *,
    artifact_dir: Path,
    config: dict[str, Any],
    predictions: list[PredictionResult],
    metric_rows: list[dict[str, Any]],
    aggregate_metrics: dict[str, float],
    failure_rows: list[dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    log_lines: list[str] | None = None,
) -> dict[str, Path]:
    """Write the required Phase 2 run artifacts and return their paths."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "config": artifact_dir / "config.yaml",
        "predictions": artifact_dir / "predictions.jsonl",
        "metrics": artifact_dir / "metrics.json",
        "summary": artifact_dir / "summary.csv",
        "failures": artifact_dir / "failures.json",
        "manifest": artifact_dir / "manifest.json",
        "environment": artifact_dir / "environment.json",
        "log": artifact_dir / "run.log",
    }

    paths["config"].write_text(
        yaml.safe_dump(redact_secrets(config), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    with paths["predictions"].open("w", encoding="utf-8") as handle:
        for result in predictions:
            handle.write(result.model_dump_json() + "\n")

    metrics_payload = {"aggregate": aggregate_metrics, "examples": metric_rows}
    paths["metrics"].write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["failures"].write_text(
        json.dumps(failure_rows or [], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(manifest or {}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fieldnames = sorted({key for row in metric_rows for key in row})
    with paths["summary"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    environment_payload = {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "created_at": datetime.now(UTC).isoformat(),
        **(environment or {}),
    }
    paths["environment"].write_text(
        json.dumps(environment_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["log"].write_text("\n".join(log_lines or []) + "\n", encoding="utf-8")
    return paths
