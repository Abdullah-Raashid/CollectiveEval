"""Versioned correction launcher; never writes the original or blocked pilot namespace."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from collectiveeval import phase8_corrective as launch
from collectiveeval.budget import BUDGET_SEMANTICS_VERSION, SCIENTIFIC_ATTEMPT_POLICY
from collectiveeval.datasets import file_sha256
from collectiveeval.pilot_analysis import evidence_hashes

BLOCKED = launch.ORIGINAL / "phase8_2"
REPAIRED = launch.ORIGINAL / "phase8_2_attempts_v2"


def blocked_evidence_hashes() -> dict[str, str]:
    return {
        str(p.relative_to(BLOCKED)): file_sha256(p)
        for p in sorted(BLOCKED.rglob("*"))
        if p.is_file()
    }


def initialize() -> None:
    REPAIRED.mkdir(parents=True, exist_ok=True)
    manifest_path = REPAIRED / "preservation_manifest.json"
    if not manifest_path.exists():
        launch.write_json(
            manifest_path,
            {
                "original_raw_hashes": evidence_hashes(launch.ORIGINAL),
                "blocked_namespace_hashes": blocked_evidence_hashes(),
                "blocked_run_id": "659b9547-e512-4fd5-8305-207114f1a3ed",
                "blocked_accounting_status": "ACCOUNTING_INCOMPLETE_RETRY_ATTEMPTS",
                "blocked_descriptive_observations": {
                    "examples": 32,
                    "seeds": [20261003, 20261004],
                    "disagreement_examples": 9,
                    "identical_examples": 23,
                    "changed_selected_predictions": 0,
                    "mean_task_score": 0.5876176576012103,
                    "known_attempts": 66,
                    "successful_records": 64,
                    "unknown_failed_attempts": 2,
                },
                "blocked_run_reused_in_corrected_view": False,
            },
        )
    for name in ("preflight.json", "rerun_scope.json"):
        path = REPAIRED / name
        if not path.exists():
            launch.write_json(path, json.loads((BLOCKED / name).read_text()))
    verify_preservation()


def verify_preservation() -> dict[str, Any]:
    manifest = json.loads((REPAIRED / "preservation_manifest.json").read_text())
    original_ok = manifest["original_raw_hashes"] == evidence_hashes(launch.ORIGINAL)
    blocked_ok = manifest["blocked_namespace_hashes"] == blocked_evidence_hashes()
    if not original_ok or not blocked_ok:
        raise ValueError("Original/blocked raw evidence changed; stop before inference")
    return {"original_unchanged": original_ok, "blocked_namespace_unchanged": blocked_ok}


@contextmanager
def repaired_namespace() -> Iterator[None]:
    previous = launch.CORRECTIVE
    launch.CORRECTIVE = REPAIRED
    try:
        yield
    finally:
        launch.CORRECTIVE = previous


def prepare_repaired() -> dict[str, Any]:
    initialize()
    with repaired_namespace():
        preflight = launch.prepare()
    preflight.update(
        {
            "execution_version": "provider-attempts.v2",
            "preservation": verify_preservation(),
            "attempt_policy": SCIENTIFIC_ATTEMPT_POLICY,
            "budget_semantics_version": BUDGET_SEMANTICS_VERSION,
            "blocked_run_efficiency_excluded": True,
            "unknown_usage_rule": "Include the stopped example's checkpoint and known-token lower "
            "bound; mark total usage unknown. Halt the run and matrix; no automatic resume/retry. "
            "Do not report a complete matched comparison or freeze Phase 9 from incomplete runs.",
            "interruption_rule": "Completed examples skip safely; admitted incomplete known-usage "
            "trajectories retain their last valid answer without replay. Unknown/in-flight "
            "consumption forbids automatic relaunch under the same config.",
        }
    )
    launch.write_json(REPAIRED / "preflight.json", preflight)
    return preflight


def analyze_repaired() -> dict[str, Any]:
    from collectiveeval.phase8_corrective_analysis import analyze_corrective

    initialize()
    result: dict[str, Any] = analyze_corrective(corrective=REPAIRED)["integrity"]
    preservation = verify_preservation()
    launch.write_json(
        REPAIRED / "phase8_2_status.json",
        {
            "status": "COMPLETED",
            "integrity_status": result["status"],
            "run_ids": result["scientific_run_ids"],
            "diagnostic_run_ids": [d["run_id"] for d in result["diagnostics"]["diagnostics"]],
            "scientific_actuals": result["scientific_actuals"],
            "preservation": preservation,
            "test_evaluated": False,
            "phase9_started": False,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8.2 attempt-accounting correction")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    initialize()
    if args.verify:
        with repaired_namespace():
            launch.verify_static()
    preflight = prepare_repaired()
    if args.execute:
        try:
            with repaired_namespace():
                result = launch.execute(preflight)
        except Exception as exc:
            status_path = REPAIRED / "phase8_2_status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            launch.write_json(
                status_path,
                {
                    **status,
                    "status": "BLOCKED_EXECUTION",
                    "error": str(exc),
                    "phase9_started": False,
                },
            )
            verify_preservation()
            raise
    elif args.analyze:
        result = analyze_repaired()
    else:
        result = {
            "status": preflight["status"],
            "prepared_reruns": 5,
            "real_model_calls": 0,
            "preservation": preflight["preservation"],
        }
    verify_preservation()
    print(json.dumps(result, indent=2))
    return 0
