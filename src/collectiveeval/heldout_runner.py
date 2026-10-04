"""Explicitly gated held-out execution; default preflight never opens TEST."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from collectiveeval.artifact_verify import verify_run_artifacts
from collectiveeval.budget import UnknownUsageStop
from collectiveeval.datasets import file_sha256, load_jsonl
from collectiveeval.heldout_protocol import (
    PROTOCOL_DIR,
    freeze_protocol,
    read_json,
    verify_protocol,
)
from collectiveeval.phase8_corrective_analysis import validate_attempt_records
from collectiveeval.pilot import Phase8Settings, discover_local_model_identity
from collectiveeval.pilot_analysis import ReadOnlyPilotStore
from collectiveeval.reporting import attempt_accounting_report, compare_runs_report
from collectiveeval.runner import run_experiment_from_path


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_live_model(model: dict[str, Any], identity: dict[str, Any] | None = None) -> None:
    if identity is None:
        identity = discover_local_model_identity(
            Phase8Settings(
                provider=model["provider"],
                model=model["model"],
                api_key_env="LOCAL_NON_SECRET_PLACEHOLDER",
                base_url=model["base_url"],
                input_cost_per_1k=0,
                output_cost_per_1k=0,
                pilot_cap_usd=0,
                smoke_max_cost_usd=0,
                timeout_s=model["timeout_s"],
            )
        )
    expected = model["provider_options"]["model_identity"]
    if (
        identity.get("digest") != model["provider_options"]["model_digest"]
        or identity.get("name") != model["model"]
        or identity.get("details", {}).get("quantization_level")
        != expected["details"]["quantization_level"]
    ):
        raise ValueError("Live local model identity/digest mismatch; no TEST inference authorized")


def verify_test_hash(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError("TEST hash mismatch; no inference authorized")


def verify_resume_evidence(
    db: Path,
    allowed: dict[str, str],
    *,
    dataset_sha256: str | None = None,
    model: dict[str, Any] | None = None,
) -> None:
    if not db.exists():
        return
    store = ReadOnlyPilotStore(db)
    seen: set[str] = set()
    for run in store._fetch_all("SELECT * FROM runs"):
        experiment = store.get_run_experiment(run["id"])
        assert experiment is not None
        if experiment["config_hash"] not in allowed.values():
            raise ValueError("Existing execution contains an unfrozen config")
        if experiment["config_hash"] in seen:
            raise ValueError("Duplicate frozen condition/strategy run; no selection among repeats")
        seen.add(experiment["config_hash"])
        if dataset_sha256 and experiment["dataset_hash"] != dataset_sha256:
            raise ValueError("Existing execution dataset hash differs from frozen TEST")
        attempts = store.get_run_model_calls(run["id"])
        predictions = {r["example_id"] for r in store.get_run_predictions(run["id"])}
        if any(c["usage_status"] == "UNKNOWN_NOT_RETURNED" for c in attempts):
            raise UnknownUsageStop("Unknown sent usage: human review required; no automatic replay")
        if any(
            c["outcome"] != "SUCCESS"
            or c["usage_status"] != "PROVIDER_REPORTED"
            or c["example_id"] not in predictions
            for c in attempts
        ):
            raise ValueError(
                "Failed/estimated/uncheckpointed sent work requires review; no automatic replay"
            )
        if validate_attempt_records(attempts, store.get_run_logical_calls(run["id"])):
            raise ValueError("Incomplete logical/attempt evidence cannot be automatically resumed")
        if model and any(
            c["provider"] != model["provider"]
            or c["model"] != model["model"]
            or json.loads(c["metadata_json"]).get("model_digest")
            != model["provider_options"]["model_digest"]
            for c in attempts
        ):
            raise ValueError("Existing attempt provider/model/digest differs from frozen recipe")
        if run["status"] not in {"RUNNING", "COMPLETED"}:
            raise ValueError("Failed scientific run requires review; no silent retry")


@contextmanager
def execution_lock(directory: Path) -> Iterator[None]:
    path = directory / "execution.lock"
    try:
        handle = path.open("x")
    except FileExistsError as exc:
        raise ValueError(
            "Execution lock present; inspect the previous process/server before resuming"
        ) from exc
    with handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def bind_execution_start(directory: Path, digest: str) -> None:
    marker = directory / "execution_started.json"
    if marker.exists():
        if read_json(marker)["protocol_sha256"] != digest:
            raise ValueError("Protocol cannot change once TEST execution starts")
        return
    with marker.open("x") as handle:
        json.dump({"protocol_sha256": digest, "started_at": datetime.now(UTC).isoformat()}, handle)


def execute_heldout(directory: Path, expected_sha256: str) -> dict[str, Any]:
    checked = verify_protocol(directory, expected_sha256=expected_sha256)
    protocol = checked["protocol"]
    model = protocol["primary_model"]
    execution = directory / "execution"
    execution.mkdir(exist_ok=True)
    db = execution / "heldout.sqlite3"
    if db.exists() and not (execution / "execution_started.json").exists():
        raise ValueError("Unregistered execution DB: frozen start marker is required")
    if (execution / "halted.json").exists():
        raise ValueError(
            "Frozen execution halted; review documented failure before any continuation"
        )
    allowed = {e["name"]: e["resolved_config_sha256"] for e in protocol["matrix"]}
    verify_resume_evidence(
        db, allowed, dataset_sha256=protocol["benchmark"]["test_sha256"], model=model
    )
    # Metadata-only identity requests happen before TEST access and before any generation.
    verify_live_model(model)
    run_ids = []
    with execution_lock(execution):
        bind_execution_start(execution, expected_sha256)
        try:
            benchmark = protocol["benchmark"]
            test_path = Path(benchmark["test_path"])
            verify_test_hash(test_path, benchmark["test_sha256"])
            examples = load_jsonl(test_path)
            expected_ids = [e.id for e in examples]
            if len(examples) != benchmark["test_examples"] or len(set(expected_ids)) != len(
                examples
            ):
                raise ValueError("Frozen TEST count/uniqueness mismatch")
            if any(e.metadata["split"] != "test" for e in examples):
                raise ValueError("Frozen matrix requires the TEST split only")
            for entry in protocol["matrix"]:
                # Recheck frozen identity before each run; no environment overrides.
                verify_protocol(directory, expected_sha256=expected_sha256)
                verify_live_model(model)
                verify_test_hash(test_path, benchmark["test_sha256"])
                verify_resume_evidence(
                    db, allowed, dataset_sha256=benchmark["test_sha256"], model=model
                )
                existing = (
                    ReadOnlyPilotStore(db).find_completed_run_by_config_hash(
                        entry["resolved_config_sha256"]
                    )
                    if db.exists()
                    else None
                )
                if existing is not None:
                    run_id = str(existing["id"])
                else:
                    print(
                        json.dumps(
                            {
                                "stage": "FROZEN_TEST",
                                "entry": entry["name"],
                                "examples": len(examples),
                            }
                        ),
                        flush=True,
                    )
                    run = run_experiment_from_path(
                        directory / entry["config_path"], db_path=db, output_dir=execution / "runs"
                    )
                    if isinstance(run, dict):
                        raise ValueError("Unexpected dry run")
                    run_id = run.run_id
                store = ReadOnlyPilotStore(db)
                if [r["example_id"] for r in store.get_run_predictions(run_id)] != expected_ids:
                    raise ValueError("Run did not checkpoint the exact frozen TEST cohort/order")
                audit = validate_attempt_records(
                    store.get_run_model_calls(run_id), store.get_run_logical_calls(run_id)
                )
                if audit or not verify_run_artifacts(store, run_id)["ok"]:
                    raise ValueError(
                        "Completed-run accounting/artifact integrity failed: " + "; ".join(audit)
                    )
                run_ids.append(run_id)
                write_progress(
                    execution / "progress.json",
                    {
                        "status": "RUNNING",
                        "protocol_sha256": expected_sha256,
                        "run_ids": run_ids,
                        "accounting": {r: attempt_accounting_report(store, r) for r in run_ids},
                    },
                )
            stats = protocol["statistics"]
            reports = {}
            for condition in protocol["conditions"]:
                ids = [
                    r
                    for e, r in zip(protocol["matrix"], run_ids, strict=True)
                    if checked["configs"][e["name"]]["phase9"]["condition"] == condition
                ]
                reports[condition] = {
                    metric: compare_runs_report(
                        store,
                        ids,
                        metric=metric,
                        n_bootstrap=stats["bootstrap_replicates"],
                        seed=stats["seed"],
                    )
                    for metric in ("task_score", "total_tokens", "provider_attempts", "latency_ms")
                }
                if any(
                    not report["matched_budget"]["ok"] for report in reports[condition].values()
                ):
                    raise ValueError("Frozen comparison failed budget compatibility")
            write_progress(execution / "paired_reports.json", reports)
            result = {
                "status": "COMPLETED",
                "protocol_sha256": expected_sha256,
                "run_ids": run_ids,
                "test_examples": len(examples),
                "test_model_evaluation_occurred": True,
            }
            write_progress(execution / "progress.json", result)
            return result
        except BaseException as exc:
            write_progress(
                execution / "halted.json",
                {
                    "status": "HALTED",
                    "protocol_sha256": expected_sha256,
                    "completed_run_ids": run_ids,
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "rule": "Preserve DB/attempts. No silent retry, exclusion, "
                    "prompt change or overwrite.",
                },
            )
            raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze/preflight by default; TEST requires --execute and exact hash"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--protocol-dir", type=Path, default=PROTOCOL_DIR)
    args = parser.parse_args()
    if args.execute and not args.protocol_sha256:
        parser.error("TEST execution requires --protocol-sha256 copied from the frozen manifest")
    if args.freeze:
        result: dict[str, Any] = {
            "status": "FROZEN",
            "protocol_sha256": freeze_protocol(args.protocol_dir),
            "test_file_opened": False,
            "test_execution_started": False,
        }
    elif args.execute:
        result = execute_heldout(args.protocol_dir, args.protocol_sha256)
    else:
        checked = verify_protocol(args.protocol_dir, expected_sha256=args.protocol_sha256)
        result = {
            key: checked[key]
            for key in ("status", "protocol_sha256", "test_file_opened", "test_execution_started")
        }
    print(json.dumps(result, indent=2))
    return 0
