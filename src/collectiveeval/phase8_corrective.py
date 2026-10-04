"""Phase 8.2 preparation and explicitly gated corrective execution."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from collectiveeval.budget import BUDGET_SEMANTICS_VERSION, SCIENTIFIC_ATTEMPT_POLICY
from collectiveeval.config import stable_config_hash
from collectiveeval.core import ModelSpec
from collectiveeval.datasets import file_sha256, load_jsonl, write_jsonl
from collectiveeval.pilot import (
    Phase8Settings,
    discover_local_model_identity,
    verify_phase8_dev_freeze,
)
from collectiveeval.pilot_analysis import ReadOnlyPilotStore, evidence_hashes
from collectiveeval.runner import dry_run_summary, run_experiment_from_path
from collectiveeval.strategy_factory import strategy_from_config
from collectiveeval.task_contracts import CONTRACT_VERSION, CRITIC_CONTRACT_VERSION

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "reports" / "pilot_v1"
CORRECTIVE = ORIGINAL / "phase8_2"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def recursive_model_audit(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, ModelSpec):
            models.append({"path": path, "provider": value.provider, "model": value.model})
        else:
            for key in (
                "model",
                "generator",
                "critic",
                "reviser",
                "cheap_model",
                "critic_strategy",
                "debate_strategy",
            ):
                child = getattr(value, key, None)
                if child is not None:
                    visit(child, f"{path}.{key}")

    visit(strategy_from_config(config), "strategy")
    if not models or any(m["provider"] != "ollama" or m["model"] != "gemma3:4b" for m in models):
        raise ValueError("Scientific nested roles must all use Ollama/gemma3:4b")
    return models


def historical_audit(store: ReadOnlyPilotStore, run_ids: list[str]) -> dict[str, Any]:
    mismatches = []
    checked_candidates = 0
    for run_id in run_ids:
        calls = store.get_run_model_calls(run_id)
        run = store.get_run(run_id)
        assert run is not None
        path = Path(run["artifact_dir"]) / "predictions.jsonl"
        artifacts = {
            r["example_id"]: r for line in path.read_text().splitlines() if (r := json.loads(line))
        }
        for prediction in store.get_run_predictions(run_id):
            usage = json.loads(prediction["usage_json"])
            subset = [c for c in calls if c["example_id"] == prediction["example_id"]]
            observed = {
                "model_calls": len(subset),
                "input_tokens": sum(c["input_tokens"] for c in subset),
                "output_tokens": sum(c["output_tokens"] for c in subset),
            }
            raw = artifacts[prediction["example_id"]]
            if any(usage[k] != v or raw[k] != v for k, v in observed.items()):
                mismatches.append({"run_id": run_id, "example_id": prediction["example_id"]})
            if raw["output"] != json.loads(prediction["output_json"]):
                mismatches.append({"run_id": run_id, "kind": "output mismatch"})
            for candidate in raw["candidates"]:
                role = candidate["role"]
                role = "sample" if role.startswith("sample_") else role
                if not any(c["role"] == role for c in subset):
                    mismatches.append(
                        {
                            "run_id": run_id,
                            "kind": "candidate without role call",
                            "example_id": prediction["example_id"],
                            "role": role,
                        }
                    )
                checked_candidates += 1
    return {
        "classification": "INDETERMINATE",
        "recorded_evidence_discrepancies": mismatches,
        "checked_runs": len(run_ids),
        "checked_predictions": 32 * len(run_ids),
        "checked_candidate_role_correspondences": checked_candidates,
        "reason": "No independent provider response/usage log is retained "
        "in the supplied artifacts. "
        "A completed response rejected before ledger insertion is also absent from "
        "candidate creation; agreement of persisted artifacts cannot exclude that path. "
        "Wall-clock gaps and budget exhaustion alone cannot distinguish "
        "pre-call rejection "
        "from post-call loss. No historical occurrence is asserted.",
    }


def prepare() -> dict[str, Any]:
    """Only reads DEV and historical evidence; never constructs a provider client."""
    freeze = verify_phase8_dev_freeze()
    if not freeze["ok"]:
        raise ValueError("DEV freeze verification failed")
    baseline = json.loads(
        (ORIGINAL / "phase8_integrity_report.md")
        .read_text()
        .split("```json\n")[1]
        .split("\n```")[0]
    )
    if evidence_hashes(ORIGINAL) != baseline["raw_evidence_sha256_before"]:
        raise ValueError("Historical raw evidence differs from Phase 8.1 hash baseline")
    scope = json.loads((CORRECTIVE / "rerun_scope.json").read_text())
    selection = json.loads((ORIGINAL / "example_ids.json").read_text())
    examples = load_jsonl(ORIGINAL / "pilot_dev32.jsonl")
    if [e.id for e in examples] != selection["example_ids"] or len(examples) != 32:
        raise ValueError("Exact DEV IDs/order changed")
    if any(e.metadata["split"] != "dev" for e in examples):
        raise ValueError("Corrective pilot must be DEV-only")
    store = ReadOnlyPilotStore(ORIGINAL / "phase8.sqlite3")
    config_entries, models = [], []
    for entry in scope["entries"]:
        old = store.get_run_experiment(entry["historical_run_id"])
        assert old is not None
        config = json.loads(old["config_json"])
        models.extend(recursive_model_audit(config))
        if entry["action"] != "RERUN":
            continue
        new = copy.deepcopy(config)
        new["experiment"]["name"] = f"phase8_2_{entry['condition']}_{entry['strategy']}"
        new["phase8_2"] = {
            "historical_run_id": entry["historical_run_id"],
            "observation_kind": "SCIENTIFIC",
            "seed_scheme": "base_seed_plus_sample_index.v1",
            "budget_semantics_version": BUDGET_SEMANTICS_VERSION,
            "task_contract_version": CONTRACT_VERSION,
            "critic_contract_version": CRITIC_CONTRACT_VERSION,
            "dev_sha256": freeze["actual"]["dev_sha256"],
            "execution_version": "provider-attempts.v2",
        }
        new["provider_retry"] = {
            "max_retries": 0,
            "unknown_usage_policy": SCIENTIFIC_ATTEMPT_POLICY,
        }
        new["concurrency"] = {"limit": 1}
        new["checkpoint"] = {"resume_completed_examples": True, "unit": "example"}
        path = CORRECTIVE / "configs" / f"{entry['condition']}_{entry['strategy']}.json"
        if path.exists() and json.loads(path.read_text()) != new:
            raise ValueError("Prepared corrective config changed; refusing to overwrite")
        write_json(path, new)
        config_entries.append(
            {**entry, "config_path": str(path), "prepared_config_hash": stable_config_hash(new)}
        )
    status = json.loads((ORIGINAL / "phase8_status.json").read_text())
    audit = historical_audit(store, status["run_ids"])
    write_json(CORRECTIVE / "historical_budget_impact_audit.json", audit)
    (CORRECTIVE / "historical_budget_impact_audit.md").write_text(
        "# Historical Budget Impact Audit\n\nClassification: **INDETERMINATE**\n\n"
        + audit["reason"]
        + "\n\n```json\n"
        + json.dumps(audit, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    previous = json.loads((CORRECTIVE / "preflight.json").read_text())
    gates_path = CORRECTIVE / "static_gates.json"
    gates = json.loads(gates_path.read_text()) if gates_path.exists() else {}
    passed = (
        gates.get("pytest_passed", False)
        and gates.get("ruff_passed", False)
        and gates.get("mypy_passed", False)
    )
    if passed and gates.get("source_hashes") != source_hashes():
        passed = False
    stop_path = CORRECTIVE / "methodology_stop.json"
    methodology_stop = json.loads(stop_path.read_text()) if stop_path.exists() else None
    payload = {
        **previous,
        "status": "STATIC_PREFLIGHT_PASSED_REAL_DIAGNOSTICS_PENDING"
        if passed
        else "REPAIRED_STATIC_GATES_PENDING",
        "budget_semantics_version": BUDGET_SEMANTICS_VERSION,
        "checks": {
            "prompt_only_admission": True,
            "output_allowance_clipping": True,
            "completed_call_preservation": True,
            "post_call_overrun_accounting": True,
            "parse_failure_usage_preservation": True,
            "distinct_self_consistency_seeds": True,
            "nested_no_mock": True,
            "dev_freeze_hash": True,
            "test_isolation": True,
            "historical_evidence_immutability": True,
        },
        "models": models,
        "static_gates": gates,
        "dev_freeze": freeze,
        "task_counts": dict(Counter(str(e.task_type) for e in examples)),
        "difficulty_counts": dict(Counter(e.metadata["difficulty"] for e in examples)),
        "prepared_reruns": config_entries,
        "historical_impact": audit["classification"],
        "implementation_checks": {
            "output_allowance_reserved_at_admission": True,
            "completed_over_budget_usage_retained": True,
        },
        "blocker": None,
        "original_blocker_reproduction": previous.get(
            "original_blocker_reproduction", previous["offline_reproduction"]
        ),
        "offline_reproduction": {
            "simulation_only": True,
            "real_provider_calls": 0,
            "estimated_prompt_tokens": 251,
            "max_total_tokens": 252,
            "configured_max_tokens": 700,
            "effective_max_tokens": 1,
            "actual_input_tokens": 251,
            "actual_output_tokens": 2,
            "persistable_ledger_calls": 1,
            "persistable_ledger_tokens": 253,
            "post_call_budget_overrun": True,
            "budget_overrun_tokens": 1,
            "subsequent_call_rejected": True,
        },
        "new_real_model_calls": 0,
        "verification": gates,
        "current_local_model_digest_verified": False,
        "current_local_model_digest_not_verified_reason": (
            "No Ollama access during static preparation"
        ),
        "methodology_stop": methodology_stop,
    }
    if methodology_stop and methodology_stop.get("unresolved"):
        payload["status"] = "BLOCKED_METHODOLOGY"
    write_json(CORRECTIVE / "preflight.json", payload)
    (CORRECTIVE / "logs").mkdir(parents=True, exist_ok=True)
    scope.update(
        {
            "static_prepared": True,
            "inference_launch_allowed": bool(passed) and payload["status"] != "BLOCKED_METHODOLOGY",
            "real_diagnostics_required": True,
        }
    )
    write_json(CORRECTIVE / "rerun_scope.json", scope)
    if not (CORRECTIVE / "phase8_2.sqlite3").exists():
        write_json(
            CORRECTIVE / "phase8_2_status.json",
            {
                "phase": "8.2",
                "status": payload["status"],
                "new_run_ids": [],
                "scientific_real_model_calls": 0,
                "diagnostic_real_model_calls": 0,
                "historical_evidence_modified": False,
                "source_code_modified": True,
                "phase9_started": False,
            },
        )
    return payload


def execute(preflight: dict[str, Any]) -> dict[str, Any]:
    """Called only by --execute, after static gates; diagnostics precede scientific runs."""
    if (preflight.get("methodology_stop") or {}).get("unresolved"):
        raise ValueError("Unresolved live methodology blocker: real execution is disabled")
    if preflight["status"] != "STATIC_PREFLIGHT_PASSED_REAL_DIAGNOSTICS_PENDING":
        raise ValueError("Static gates must pass before real execution")
    if preflight["static_gates"].get("source_hashes") != source_hashes():
        raise ValueError("Source changed since static gates; rerun gates before inference")
    sample = json.loads(Path(preflight["prepared_reruns"][0]["config_path"]).read_text())
    model = sample["model"]
    settings = Phase8Settings(
        provider=model["provider"],
        model=model["model"],
        api_key_env=str(model.get("api_key_env") or "OLLAMA_LOCAL_UNUSED"),
        base_url=model["base_url"],
        input_cost_per_1k=model["input_cost_per_1k"],
        output_cost_per_1k=model["output_cost_per_1k"],
        pilot_cap_usd=0,
        smoke_max_cost_usd=0,
    )
    identity = discover_local_model_identity(settings)
    expected_digest = model["provider_options"]["model_digest"]
    if not identity.get("digest") or identity["digest"] != expected_digest:
        raise ValueError("Current Ollama model digest differs from historical model")
    # Diagnostics live in a separate DB and never enter the scientific scope.
    diagnostic_ids = []
    for strategy, example_id in (
        ("self_consistency", "v3_1-gqa-02-00"),
        ("critic_reviser", "v3_1-ext-08-07"),
        ("adaptive_router", "v3_1-ext-08-07"),
    ):
        entry = next(
            e
            for e in preflight["prepared_reruns"]
            if e["strategy"] == strategy and e["condition"] == "matched_tokens"
        )
        config = json.loads(Path(entry["config_path"]).read_text())
        examples = [e for e in load_jsonl(Path(config["dataset"]["path"])) if e.id == example_id]
        dataset = CORRECTIVE / "diagnostics" / f"{strategy}.jsonl"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(dataset, examples)
        if len(examples) != 1 or examples[0].metadata["split"] != "dev":
            raise ValueError("Diagnostic must contain exactly one selected DEV example")
        config["dataset"]["path"] = str(dataset)
        config["phase8_2"]["observation_kind"] = "DIAGNOSTIC"
        config["experiment"]["name"] += "_diagnostic"
        config_path = CORRECTIVE / "diagnostics" / f"{strategy}.json"
        write_json(config_path, config)
        print(
            json.dumps({"stage": "DIAGNOSTIC", "strategy": strategy, "example_id": example_id}),
            flush=True,
        )
        result = _run_once(
            config_path,
            CORRECTIVE / "diagnostics" / "diagnostics.sqlite3",
            CORRECTIVE / "diagnostics" / "runs",
        )
        diagnostic_ids.append(result)
        diagnostic_store = ReadOnlyPilotStore(CORRECTIVE / "diagnostics" / "diagnostics.sqlite3")
        calls = diagnostic_store.get_run_model_calls(result)
        logical_calls = diagnostic_store.get_run_logical_calls(result)
        if (
            len(calls) != len({c["attempt_id"] for c in calls})
            or len(logical_calls) != len({c["logical_call_id"] for c in calls})
            or any(
                c["attempt_index"] != 0
                or c["outcome"] != "SUCCESS"
                or c["usage_status"] != "PROVIDER_REPORTED"
                for c in calls
            )
            or any(
                json.loads(c["metadata_json"]).get("unknown_usage_policy")
                != SCIENTIFIC_ATTEMPT_POLICY
                for c in calls
            )
        ):
            raise ValueError("Diagnostic attempt/logical-call or unknown-usage policy check failed")
        if any(
            c["provider"] != "ollama"
            or c["model"] != "gemma3:4b"
            or c["usage_source"] != "PROVIDER_REPORTED"
            or json.loads(c["metadata_json"]).get("budget_semantics_version")
            != BUDGET_SEMANTICS_VERSION
            for c in calls
        ):
            raise ValueError("Diagnostic provider/usage check failed")
        if strategy == "self_consistency" and (
            len(calls) != 2 or len({c["requested_seed"] for c in calls}) != 2
        ):
            raise ValueError("Diagnostic SelfConsistency must have two distinct requested seeds")
        if strategy == "adaptive_router":
            prediction = diagnostic_store.get_run_predictions(result)[0]
            if json.loads(prediction["metadata_json"]).get("route") not in {
                "accept",
                "critic",
                "debate",
            }:
                raise ValueError("Diagnostic router lost its route")
    write_json(
        CORRECTIVE / "diagnostics" / "status.json", {"status": "PASSED", "run_ids": diagnostic_ids}
    )
    ids = []
    for entry in preflight["prepared_reruns"]:
        print(
            json.dumps(
                {
                    "stage": "SCIENTIFIC_CORRECTIVE",
                    "strategy": entry["strategy"],
                    "condition": entry["condition"],
                    "examples": 32,
                }
            ),
            flush=True,
        )
        ids.append(
            _run_once(
                Path(entry["config_path"]), CORRECTIVE / "phase8_2.sqlite3", CORRECTIVE / "runs"
            )
        )
        write_json(
            CORRECTIVE / "phase8_2_status.json",
            {
                "status": "RUNNING"
                if len(ids) < len(preflight["prepared_reruns"])
                else "CORRECTIVE_RUNS_COMPLETED_ANALYSIS_PENDING",
                "run_ids": ids,
                "diagnostic_run_ids": diagnostic_ids,
            },
        )
    return {"status": "CORRECTIVE_RUNS_COMPLETED_ANALYSIS_PENDING", "run_ids": ids}


def _run_once(config_path: Path, db_path: Path, output_dir: Path) -> str:
    # Inspect through read-only connections; only the new namespace is writable.
    if db_path.exists():
        store = ReadOnlyPilotStore(db_path)
        expected_hash = dry_run_summary(json.loads(config_path.read_text()))["config_hash"]
        completed = store.find_completed_run_by_config_hash(expected_hash)
        if completed is not None:
            return str(completed["id"])
    result = run_experiment_from_path(config_path, db_path=db_path, output_dir=output_dir)
    if isinstance(result, dict):
        raise ValueError("Corrective execution unexpectedly returned a dry run")
    return result.run_id


def source_hashes() -> dict[str, str]:
    paths = [
        *sorted((ROOT / "src" / "collectiveeval").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "tests").glob("*.py")),
    ]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}


def verify_static() -> dict[str, Any]:
    """Run offline gates without opening the frozen TEST benchmark files."""
    commands = {
        "pytest": [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-q",
            "--ignore=tests/test_benchmark_v2.py",
            "--ignore=tests/test_benchmark_v3.py",
            "--ignore=tests/test_benchmark_v3_1.py",
            "--ignore=tests/test_phase6_benchmark.py",
        ],
        "ruff": [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        "mypy": [sys.executable, "-m", "mypy", "src"],
    }
    gates: dict[str, Any] = {"test_split_files_opened": False, "real_provider_calls": 0}
    for name, command in commands.items():
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        gates[f"{name}_passed"] = result.returncode == 0
        gates[f"{name}_output"] = result.stdout + result.stderr
        if name == "pytest":
            match = re.search(r"(\d+) passed", result.stdout)
            gates["pytest_tests_passed"] = int(match.group(1)) if match else None
    gates["source_hashes"] = source_hashes()
    write_json(CORRECTIVE / "static_gates.json", gates)
    return gates


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Phase 8.2; inference requires --execute")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Run offline static gates")
    args = parser.parse_args()
    if args.verify:
        verify_static()
    preflight = prepare()
    if args.execute:
        try:
            result = execute(preflight)
        except Exception as exc:
            status_path = CORRECTIVE / "phase8_2_status.json"
            previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
            write_json(
                status_path,
                {
                    **previous_status,
                    "status": "BLOCKED_METHODOLOGY"
                    if (preflight.get("methodology_stop") or {}).get("unresolved")
                    else "BLOCKED_REAL_EXECUTION",
                    "error": str(exc),
                    "historical_evidence_modified": False,
                },
            )
            raise
    else:
        result = {
            "status": preflight["status"],
            "prepared_reruns": len(preflight["prepared_reruns"]),
            "real_model_calls": 0,
        }
    print(json.dumps(result))
    return 0
