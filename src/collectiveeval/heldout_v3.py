"""Explicit returned-failure amendment; preserve v1/v2, seed v3, never replay."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from collectiveeval.artifact_verify import verify_run_artifacts
from collectiveeval.budget import BudgetLedger, InferenceBudget, LogicalCallRecord, ModelCallRecord
from collectiveeval.config import load_config, stable_config_hash
from collectiveeval.datasets import file_sha256
from collectiveeval.failed_outputs import CONTRACT, PENDING, failed_prediction, failure_annotations
from collectiveeval.heldout_continuation import (
    AUDIT as V1_AUDIT,
)
from collectiveeval.heldout_continuation import (
    CarriedBaselineStore,
    completed_interval,
    test_ids_only,
    verify_v1_preservation,
)
from collectiveeval.heldout_protocol import (
    PROTOCOL_DIR,
    code_identity,
    environment_identity,
    read_json,
    resolved_hash,
    validate_scientific_config,
)
from collectiveeval.heldout_runner import (
    bind_execution_start,
    execution_lock,
    verify_live_model,
    write_progress,
)
from collectiveeval.phase8_corrective import ROOT
from collectiveeval.pilot import verify_phase8_dev_freeze
from collectiveeval.pilot_analysis import ReadOnlyPilotStore
from collectiveeval.reporting import compare_runs_report
from collectiveeval.runner import run_experiment_from_path
from collectiveeval.storage import SCHEMA, SQLiteStore
from collectiveeval.strategies import AdaptiveRouterStrategy
from collectiveeval.strategy_factory import model_from_config

VERSION = "experimental_protocol_v3"
DIRECTORY = ROOT / "reports" / VERSION
PARENT = ROOT / "reports/experimental_protocol_v2"
AUDIT = ROOT / "reports/protocol_v3_failure_audit"
PARENT_SHA = "c91962c4f931d6ae805264fb194560834cf22f57d7938dfe7ee7e8d4cc6e8e99"
CHANGED = {
    "src/collectiveeval/artifacts.py",
    "src/collectiveeval/storage.py",
    "src/collectiveeval/runner.py",
    "src/collectiveeval/reporting.py",
    "src/collectiveeval/context.py",
    "src/collectiveeval/parsing.py",
}
ADDED = {
    "src/collectiveeval/failed_outputs.py",
    "src/collectiveeval/heldout_v3.py",
    "scripts/run_heldout_v3.py",
    "tests/test_failed_outputs.py",
    "tests/test_heldout_v3.py",
}
FAILED_ID = "v3_1-ext-09-01"
RETURNED_RULE = (
    "Known provider usage: retain raw response and attempt, checkpoint typed failed output, "
    "apply failed-output.v3 and continue. No retry, regeneration, repair or exclusion."
)
POLICY = {
    "version": CONTRACT,
    "case_a": "Preserve original schema-invalid object and existing safe task metrics; "
    "JSON parse=1, schema validity=0, SCHEMA_FAILURE. No field repair.",
    "case_b": "Retain malformed/non-object raw content; syntax parse=0/1 respectively; "
    "object/schema validity=0, primary task_score=0; no fabricated dictionary.",
    "case_c": "Retain unsafe schema-invalid object; parse=1, schema validity=0, "
    "primary task_score=0; unavailable components are NULL/UNSCORABLE_FAILED_OUTPUT.",
    "aggregation": "Primary mean includes all 200 examples. Components expose n_total, "
    "n_scored, failed_output_count and n_unavailable.",
    "stop_rules": "Unknown/estimated consumption, timeout with unknown consumption, "
    "in-flight/ambiguous work, corruption and identity mismatch still halt.",
    "uniformity": "All five strategies, both conditions, all four task families.",
    "valid_metrics": "metrics.py and validation.py are unchanged from v2.",
    "historical_checkpoint": "Preserved returned failure is checkpointed without inference; "
    "gold-dependent evaluation is pending explicit execution.",
}


def verify_preserved(parent: Path, preservation: dict[str, Any]) -> None:
    actual = {
        str(p.relative_to(parent)): file_sha256(p) for p in sorted(parent.rglob("*")) if p.is_file()
    }
    if actual != preservation["files"] or str(parent.resolve()) != preservation["v2_root"]:
        raise ValueError("Preserved v2 evidence changed")
    if stable_config_hash(read_json(parent / "protocol.json")) != PARENT_SHA:
        raise ValueError("Wrong v2 protocol")
    verify_v1_preservation(PROTOCOL_DIR, read_json(V1_AUDIT / "v1_preservation_manifest.json"))


def verify_repair_scope(preservation: dict[str, Any]) -> None:
    old, now = preservation["source_hashes_before_repair"], code_identity()
    changed = {name for name in old if now.get(name) != old[name]}
    if set(old) - set(now) or changed - CHANGED or set(now) - set(old) - ADDED:
        raise ValueError("Changes exceed the approved failed-output amendment")


def accounting(store: ReadOnlyPilotStore, run_id: str, model: dict[str, Any]) -> dict[str, Any]:
    run = store._fetch_one("SELECT strategy FROM runs WHERE id=?", (run_id,))
    if run is None:
        raise ValueError("Missing accounting run")
    root_strategy = run["strategy"]
    allowed_strategies = {root_strategy}
    router = None
    if root_strategy == "adaptive_router":
        allowed_strategies.update({"critic_reviser", "multi_agent_debate"})
        router = AdaptiveRouterStrategy(cheap_model=model_from_config(model))
    rows = store._fetch_all(
        """SELECT attempt_id,logical_call_id,attempt_index,outcome,usage_status,usage_source,
        example_id,strategy,provider,model,role,requested_seed,temperature,max_tokens,
        input_tokens,output_tokens,retry_count,start_timestamp,end_timestamp,normalized_error,
        json_extract(metadata_json,'$.model_digest') AS digest,
        json_extract(metadata_json,'$.generation_settings') AS settings,
        json_extract(metadata_json,'$.round_index') AS round_index,
        json_extract(metadata_json,'$.unknown_usage_policy') AS policy,
        json_extract(metadata_json,'$.post_call_budget_overrun') AS overrun
        FROM provider_attempts WHERE run_id=? ORDER BY id""",
        (run_id,),
    )
    logical = store.get_run_logical_calls(run_id)
    by_logical = {r["logical_call_id"]: r for r in logical}
    if len(by_logical) != len(logical) or len({r["attempt_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicated accounting identity")
    if len(rows) != len(logical) or set(by_logical) != {r["logical_call_id"] for r in rows}:
        raise ValueError("Logical/attempt accounting mismatch")
    for row in rows:
        completed_interval(row)
        link = by_logical[row["logical_call_id"]]
        completed_interval(link)
        if (
            row["outcome"] not in {"SUCCESS", "PARSE_ERROR_AFTER_RESPONSE"}
            or link["outcome"] != row["outcome"]
            or link["example_id"] != row["example_id"]
            or link["role"] != row["role"]
            or link["strategy"] != row["strategy"]
            or row["normalized_error"]
            != ("PARSE_ERROR" if row["outcome"] == "PARSE_ERROR_AFTER_RESPONSE" else None)
            or row["usage_status"] != "PROVIDER_REPORTED"
            or row["usage_source"] != "PROVIDER_REPORTED"
            or row["input_tokens"] is None
            or row["output_tokens"] is None
            or row["input_tokens"] < 0
            or row["output_tokens"] < 0
            or row["attempt_index"] != 0
            or row["retry_count"] != 0
            or row["provider"] != model["provider"]
            or row["strategy"] not in allowed_strategies
            or row["model"] != model["model"]
            or row["digest"] != model["provider_options"]["model_digest"]
            or row["policy"] != "UNKNOWN_USAGE_STOP.v1"
        ):
            raise ValueError("Unknown/incomplete/corrupted or unfrozen attempt accounting")
        settings = json.loads(row["settings"])
        expected_temperature = (
            router.debate_strategy.model.temperature
            if router is not None and row["strategy"] == "multi_agent_debate"
            else model["temperature"]
        )
        expected_seed = model["seed"] + (
            int(row["round_index"]) if row["strategy"] == "self_consistency" else 0
        )
        if (
            row["requested_seed"] != expected_seed
            or row["temperature"] != expected_temperature
            or not 0 < row["max_tokens"] <= model["max_tokens"]
            or settings
            != {
                "temperature": expected_temperature,
                "top_p": model["top_p"],
                "max_tokens": row["max_tokens"],
                "seed": expected_seed,
            }
        ):
            raise ValueError("Persisted generation settings differ from frozen recipe")
    return {
        "attempts": len(rows),
        "logical_calls": len(logical),
        "parse_failures": sum(r["outcome"] == "PARSE_ERROR_AFTER_RESPONSE" for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }


def audit_parent(parent: Path = PARENT, audit: Path = AUDIT) -> dict[str, Any]:
    preservation = read_json(audit / "v2_preservation_manifest.json")
    verify_preserved(parent, preservation)
    verify_repair_scope(preservation)
    p = read_json(parent / "protocol.json")
    if p["source_hashes"] != preservation["source_hashes_before_repair"]:
        raise ValueError("Preservation did not capture the frozen v2 implementation")
    ids = test_ids_only(Path(p["benchmark"]["test_path"]), p["benchmark"]["test_sha256"])
    store = ReadOnlyPilotStore(parent / "execution/heldout.sqlite3")
    with closing(store.connect()) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Parent DB corrupted")
    runs = store._fetch_all("SELECT id,status,strategy FROM runs ORDER BY start_ts")
    if len(runs) != 3 or [r["status"] for r in runs] != ["COMPLETED", "COMPLETED", "FAILED"]:
        raise ValueError("Unexpected parent execution scope")
    rows = []
    for run, entry in zip(runs, p["matrix"][1:4], strict=True):
        experiment = store.get_run_experiment(run["id"])
        config = load_config(parent / entry["config_path"])
        if experiment is None or (
            experiment["config_hash"] != entry["resolved_config_sha256"]
            or experiment["dataset_hash"] != p["benchmark"]["test_sha256"]
            or stable_config_hash(json.loads(experiment["config_json"])) != resolved_hash(config)
        ):
            raise ValueError("Parent scientific recipe mismatch")
        checkpoint_ids = [
            r["example_id"]
            for r in store._fetch_all(
                "SELECT example_id FROM predictions WHERE run_id=? ORDER BY id",
                (run["id"],),
            )
        ]
        expected = ids if run["status"] == "COMPLETED" else ids[:31]
        if checkpoint_ids != expected:
            raise ValueError("Missing/duplicated/reordered parent checkpoints")
        counts = accounting(store, run["id"], config["model"])
        if counts["parse_failures"] != (1 if run["status"] == "FAILED" else 0):
            raise ValueError("Unexpected returned failure scope")
        rows.append(
            {
                **run,
                "name": entry["name"],
                "completed_examples": len(checkpoint_ids),
                "checkpoint_ids_sha256": stable_config_hash({"ids": checkpoint_ids}),
                **counts,
            }
        )
    pending = store._fetch_all(
        """SELECT a.example_id,a.attempt_id,a.role,a.outcome,a.input_tokens,a.output_tokens
        FROM provider_attempts a LEFT JOIN predictions p
        ON p.run_id=a.run_id AND p.example_id=a.example_id
        WHERE p.id IS NULL ORDER BY a.id""",
    )
    if (
        len(pending) != 2
        or ids[31] != FAILED_ID
        or any(r["example_id"] != FAILED_ID for r in pending)
        or [r["outcome"] for r in pending] != ["SUCCESS", "PARSE_ERROR_AFTER_RESPONSE"]
    ):
        raise ValueError("Ambiguous uncheckpointed work; no replay authorized")
    baseline = read_json(parent / "baseline_reuse_certificate.json")
    baseline_store = ReadOnlyPilotStore(PROTOCOL_DIR / "execution/heldout.sqlite3")
    baseline_model = load_config(PROTOCOL_DIR / p["matrix"][0]["config_path"])["model"]
    if accounting(baseline_store, baseline["run_id"], baseline_model)["parse_failures"]:
        raise ValueError("Unexpected comparable failure in v1 baseline")
    return {
        "classification": "CONTINUATION_PROVEN_WITH_KNOWN_FAILURE",
        "runs": rows,
        "v1_baseline_run_id": baseline["run_id"],
        "partial_run_id": runs[2]["id"],
        "failed_example_id": FAILED_ID,
        "failed_role": pending[1]["role"],
        "failed_attempt_id": pending[1]["attempt_id"],
        "failed_attempt_input_tokens": pending[1]["input_tokens"],
        "failed_attempt_output_tokens": pending[1]["output_tokens"],
        "completed_run_examples": 600,
        "completed_checkpoints": 631,
        "checkpointed_after_amendment": 632,
        "remaining_new_inference": 1368,
        "completed_runs_comparable_failures": 0,
        "test_quality_inspected": False,
        "test_gold_accessed": False,
    }


def prepare_seed(parent: Path, target: Path, certificate: dict[str, Any]) -> None:
    """Copy bytes first; migrate only the separate copy, never the parent DB."""
    shutil.copyfile(parent / "execution/heldout.sqlite3", target)
    with sqlite3.connect(target) as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(metrics)")}
        if "metric_status" not in columns:
            conn.executescript("""
                ALTER TABLE metrics RENAME TO metrics_v2;
                DROP INDEX IF EXISTS example_metric_checkpoint;
                CREATE TABLE metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,example_id TEXT,
                    metric_name TEXT NOT NULL,metric_value REAL,
                    metric_status TEXT NOT NULL DEFAULT 'SCORED',
                    FOREIGN KEY(run_id) REFERENCES runs(id));
                INSERT INTO metrics(id,run_id,example_id,metric_name,metric_value)
                    SELECT id,run_id,example_id,metric_name,metric_value FROM metrics_v2;
                DROP TABLE metrics_v2;
            """)
        conn.executescript(SCHEMA)
    store = SQLiteStore(target)
    run_id, eid = certificate["partial_run_id"], certificate["failed_example_id"]
    experiment = store.get_run_experiment(run_id)
    run = store.get_run(run_id)
    if experiment is None or run is None:
        raise ValueError("Missing preserved partial run")
    config = json.loads(experiment["config_json"])
    ledger = BudgetLedger(InferenceBudget.model_validate(config["budget"]), run_id=run_id)
    for row in store._fetch_all(
        "SELECT * FROM logical_model_calls WHERE run_id=? AND example_id=?",
        (run_id, eid),
    ):
        logical = LogicalCallRecord.model_validate(row)
        ledger.logical_calls[logical.logical_call_id] = logical
    for row in store._fetch_all(
        "SELECT * FROM provider_attempts WHERE run_id=? AND example_id=? ORDER BY id",
        (run_id, eid),
    ):
        ledger.record_call(
            ModelCallRecord.model_validate({**row, "metadata": json.loads(row["metadata_json"])})
        )
    result = failed_prediction(ledger, eid, "multi_agent_debate", historical_projection=True)
    result.metadata.update(
        run_id=run_id,
        experiment_id=run["experiment_id"],
        historical_failure_checkpoint=True,
        carried_from="experimental_protocol_v2",
        wall_clock_strategy_latency_ms=sum(
            r.lifecycle_latency_ms for r in ledger.logical_calls.values()
        ),
    )
    store.checkpoint_example(result, {}, failure_annotations(result))
    with store.connect() as conn:
        conn.execute("UPDATE runs SET status='RUNNING' WHERE id=?", (run_id,))
    # No evaluator or benchmark loader is called here. The prediction is durable;
    # gold-dependent metrics are evaluated only by the explicit execution command.


def runtime_remaining(parent: dict[str, Any]) -> dict[str, Any]:
    rows = parent["runtime_estimate"]["matrix"][3:]
    slowest: dict[str, float] = {}
    for r in parent["runtime_estimate"]["matrix"]:
        slowest[r["strategy"]] = max(
            slowest.get(r["strategy"], 0), r["observed_extrapolated_seconds"]
        )
    weights = [168 / 200] + [1.0] * (len(rows) - 1)
    return {
        "unfinished_recipes": 7,
        "remaining_new_inference": 1368,
        "expected_provider_attempts": sum(
            r["expected_provider_attempts"] * w for r, w in zip(rows, weights, strict=True)
        ),
        "worst_case_provider_attempts": sum(
            r["worst_case_provider_attempts"] * w for r, w in zip(rows, weights, strict=True)
        ),
        "expected_tokens": sum(
            r["expected_tokens"] * w for r, w in zip(rows, weights, strict=True)
        ),
        "expected_wall_clock_seconds": sum(
            r["observed_extrapolated_seconds"] * w for r, w in zip(rows, weights, strict=True)
        )
        * 1.2,
        "conservative_wall_clock_seconds": sum(
            slowest[r["strategy"]] * w for r, w in zip(rows, weights, strict=True)
        )
        * 1.3,
        "basis": "Frozen DEV extrapolation excluding completed work and the returned failure. "
        "No TEST quality used.",
    }


def verify_seed(seed: Path, certificate: dict[str, Any], protocol: dict[str, Any]) -> None:
    store = ReadOnlyPilotStore(seed)
    run_id, eid = certificate["partial_run_id"], certificate["failed_example_id"]
    row = store._fetch_one(
        """SELECT json_extract(metadata_json,'$.evaluation_status') AS status,
        json_extract(output_json,'$.attempt_id') AS attempt_id,
        json_extract(output_json,'$.input_tokens') AS input_tokens,
        json_extract(output_json,'$.output_tokens') AS output_tokens,
        json_extract(output_json,'$.kind') AS kind
        FROM predictions WHERE run_id=? AND example_id=?""",
        (run_id, eid),
    )
    if row is None or row != {
        "status": PENDING,
        "attempt_id": certificate["failed_attempt_id"],
        "input_tokens": certificate["failed_attempt_input_tokens"],
        "output_tokens": certificate["failed_attempt_output_tokens"],
        "kind": "failed_output",
    }:
        raise ValueError("Preserved failure is not faithfully checkpointed in the seed")
    count = store._fetch_one(
        "SELECT COUNT(*) AS n FROM metrics WHERE run_id=? AND example_id=?", (run_id, eid)
    )
    if count is None or count["n"] != 0:
        raise ValueError("TEST-dependent failure evaluation occurred before authorization")
    ids = test_ids_only(
        Path(protocol["benchmark"]["test_path"]), protocol["benchmark"]["test_sha256"]
    )
    actual = [
        r["example_id"]
        for r in store._fetch_all(
            "SELECT example_id FROM predictions WHERE run_id=? ORDER BY id",
            (run_id,),
        )
    ]
    if actual != ids[:32]:
        raise ValueError("Seed checkpoint count/cohort differs from preserved work")


def freeze_v3(directory: Path = DIRECTORY) -> str:
    if directory.exists():
        raise ValueError("V3 version exists; never overwrite")
    gates = read_json(AUDIT / "v3_static_gates.json")
    if gates["source_hashes"] != code_identity() or not all(
        gates.get(name)
        for name in ("pytest_passed", "ruff_passed", "mypy_passed", "diff_check_passed")
    ):
        raise ValueError("Fresh static gates required before v3 freeze")
    certificate = audit_parent()
    parent = read_json(PARENT / "protocol.json")
    directory.mkdir(parents=True)
    seed = directory / "continuation_seed.sqlite3"
    prepare_seed(PARENT, seed, certificate)
    certificate["seed_sha256"] = file_sha256(seed)
    certificate["failed_checkpoint_evaluation_status"] = PENDING
    protocol = copy.deepcopy(parent)
    protocol.update(
        version=VERSION, source_hashes=code_identity(), reproducibility=environment_identity()
    )
    protocol["reproducibility"]["git_limit"] = (
        "Frozen source hashes bind this uncommitted v3 amendment."
    )
    protocol["failure_rules"]["returned_parse_failure"] = RETURNED_RULE
    protocol["failed_output_contract"] = POLICY
    protocol["continuation"] = {
        "parent_directory": str(PARENT.resolve()),
        "parent_protocol_sha256": PARENT_SHA,
        "certificate_sha256": stable_config_hash(certificate),
        "v2_preservation_manifest_sha256": stable_config_hash(
            read_json(AUDIT / "v2_preservation_manifest.json")
        ),
        "remaining_names": [e["name"] for e in parent["matrix"][3:]],
        "remaining_runtime": runtime_remaining(parent),
        "no_test_quality_before_amendment": True,
        "disclosure": "One v1 and two completed v2 runs retained; 31 v2 Debate checkpoints "
        "plus one known returned failure retained without replay; v3 continues the "
        "remaining 1368 examples under the disclosed failed-output contract.",
    }
    verify_seed(seed, certificate, protocol)
    digest = stable_config_hash(protocol)
    for name, payload in {
        "protocol.json": protocol,
        "continuation_certificate.json": certificate,
        "v2_preservation_manifest.json": read_json(AUDIT / "v2_preservation_manifest.json"),
        "static_gates.json": gates,
        "runtime_estimate.json": runtime_remaining(parent),
    }.items():
        write_progress(directory / name, payload)
    for entry in parent["matrix"]:
        target = directory / entry["config_path"]
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(PARENT / entry["config_path"], target)
    (directory / "protocol.md").write_text(
        f"# Protocol V3\n\nSHA-256: `{digest}`\n\n"
        "V1 began TEST and completed SingleAgent; a JSON/YAML metadata typing defect "
        "led to representation-only v2. V2 completed SelfConsistency/CriticReviser and "
        "31 Debate examples, then halted on a schema-invalid agent_1 response.\n\n"
        "V3 introduces typed failed predictions and the explicitly authorized CASE A/B/C "
        "evaluation contract, not a strategy/model/budget/prompt/schema/sampling redesign. "
        "Valid metrics and safely scoreable invalid-object component semantics are unchanged. "
        "Unknown/ambiguous work and identity/accounting corruption still halt.\n\n"
        "Both prior executions and halt records remain byte-identical. The separate seed "
        "contains completed v2 rows and the failed example checkpoint with raw content and "
        "fully reported usage. It is not regenerated. Its gold-dependent evaluation is pending "
        "explicit execution; no TEST gold, scores or comparative quality were inspected "
        "during amendment. All 200 IDs remain in every recipe's primary denominator.\n\n"
        "See protocol.json for inherited methodology and the exact failure contract, "
        "continuation_certificate.json for carry-forward evidence, and static_gates.json.\n"
    )
    write_progress(
        directory / "protocol_manifest.json",
        {
            "protocol_sha256": digest,
            "version": VERSION,
            "artifact_hashes": {
                str(p.relative_to(directory)): file_sha256(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            },
            "new_inference": False,
            "test_quality_inspected": False,
        },
    )
    verify_v3(directory, digest)
    return digest


def verify_v3(directory: Path = DIRECTORY, digest: str | None = None) -> dict[str, Any]:
    protocol = read_json(directory / "protocol.json")
    manifest = read_json(directory / "protocol_manifest.json")
    actual = stable_config_hash(protocol)
    if (
        protocol["version"] != VERSION
        or actual != manifest["protocol_sha256"]
        or (digest is not None and digest != actual)
    ):
        raise ValueError("V3 protocol hash mismatch")
    if any(
        file_sha256(directory / name) != sha for name, sha in manifest["artifact_hashes"].items()
    ):
        raise ValueError("Frozen v3 artifact changed")
    if protocol["source_hashes"] != code_identity():
        raise ValueError("Frozen v3 source changed")
    env = environment_identity()
    if any(
        protocol["reproducibility"][key] != env[key] for key in ("python_version", "dependencies")
    ):
        raise ValueError("Frozen v3 environment changed")
    certificate = read_json(directory / "continuation_certificate.json")
    current = audit_parent()
    if {
        k: v
        for k, v in certificate.items()
        if k not in {"seed_sha256", "failed_checkpoint_evaluation_status"}
    } != current:
        raise ValueError("Carry-forward certificate no longer matches preserved evidence")
    if stable_config_hash(certificate) != protocol["continuation"]["certificate_sha256"] or (
        file_sha256(directory / "continuation_seed.sqlite3") != certificate["seed_sha256"]
    ):
        raise ValueError("Seed/certificate changed")
    verify_seed(directory / "continuation_seed.sqlite3", certificate, protocol)
    parent = read_json(PARENT / "protocol.json")
    continuation = protocol["continuation"]
    if (
        continuation["parent_directory"] != str(PARENT.resolve())
        or continuation["parent_protocol_sha256"] != PARENT_SHA
        or continuation["remaining_names"] != [e["name"] for e in parent["matrix"][3:]]
        or continuation["remaining_runtime"] != runtime_remaining(parent)
        or continuation["v2_preservation_manifest_sha256"]
        != stable_config_hash(read_json(directory / "v2_preservation_manifest.json"))
    ):
        raise ValueError("Unapproved continuation control change")
    keys = {
        "version",
        "source_hashes",
        "reproducibility",
        "continuation",
        "failure_rules",
        "failed_output_contract",
    }
    if {k: v for k, v in protocol.items() if k not in keys} != {
        k: v for k, v in parent.items() if k not in keys
    }:
        raise ValueError("Scientific methodology changed")
    if (
        protocol["failure_rules"]["returned_parse_failure"] != RETURNED_RULE
        or protocol["failed_output_contract"] != POLICY
        or any(
            protocol["failure_rules"][k] != v
            for k, v in parent["failure_rules"].items()
            if k != "returned_parse_failure"
        )
    ):
        raise ValueError("Unapproved failure/stop policy")
    for entry in protocol["matrix"]:
        path = directory / entry["config_path"]
        config = load_config(path)
        if (
            file_sha256(path) != file_sha256(PARENT / entry["config_path"])
            or resolved_hash(config) != entry["resolved_config_sha256"]
        ):
            raise ValueError("Frozen recipe changed")
        validate_scientific_config(config, protocol["primary_model"])
    if not verify_phase8_dev_freeze()["ok"]:
        raise ValueError("Benchmark DEV/manifest changed")
    return {
        "status": "PASSED",
        "protocol_sha256": actual,
        "protocol": protocol,
        "certificate": certificate,
        "new_inference": False,
        "test_quality_inspected": False,
    }


def verify_resume(db: Path, protocol: dict[str, Any], directory: Path = DIRECTORY) -> None:
    store = ReadOnlyPilotStore(db)
    with closing(store.connect()) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Continuation DB corrupted")
    allowed = {e["resolved_config_sha256"]: e for e in protocol["matrix"][1:]}
    ids = test_ids_only(
        Path(protocol["benchmark"]["test_path"]), protocol["benchmark"]["test_sha256"]
    )
    seen = set()
    for run in store._fetch_all("SELECT id,status FROM runs"):
        experiment = store.get_run_experiment(run["id"])
        if (
            experiment is None
            or experiment["config_hash"] not in allowed
            or experiment["config_hash"] in seen
        ):
            raise ValueError("Unfrozen/duplicated continuation run")
        seen.add(experiment["config_hash"])
        if (
            run["status"] not in {"RUNNING", "COMPLETED"}
            or experiment["dataset_hash"] != protocol["benchmark"]["test_sha256"]
        ):
            raise ValueError("Failed or unfrozen continuation requires review")
        entry = allowed[experiment["config_hash"]]
        config = load_config(directory / entry["config_path"])
        if stable_config_hash(json.loads(experiment["config_json"])) != resolved_hash(config):
            raise ValueError("Persisted continuation recipe changed")
        accounting(store, run["id"], config["model"])
        pending = store._fetch_all(
            """SELECT a.attempt_id FROM provider_attempts a
            LEFT JOIN predictions p ON p.run_id=a.run_id AND p.example_id=a.example_id
            WHERE a.run_id=? AND p.id IS NULL""",
            (run["id"],),
        )
        if pending:
            raise ValueError("Uncheckpointed sent work; no automatic replay")
        checkpoints = store._fetch_all(
            """SELECT p.example_id,p.usage_json,
            json_extract(p.metadata_json,'$.result_kind') AS result_kind,
            json_extract(p.output_json,'$.kind') AS output_kind,
            (SELECT COUNT(*) FROM provider_attempts a WHERE a.run_id=p.run_id
            AND a.example_id=p.example_id) AS attempts,
            (SELECT SUM(input_tokens) FROM provider_attempts a WHERE a.run_id=p.run_id
            AND a.example_id=p.example_id) AS input_tokens,
            (SELECT SUM(output_tokens) FROM provider_attempts a WHERE a.run_id=p.run_id
            AND a.example_id=p.example_id) AS output_tokens,
            (SELECT COUNT(*) FROM provider_attempts a WHERE a.run_id=p.run_id
            AND a.example_id=p.example_id AND outcome='PARSE_ERROR_AFTER_RESPONSE') AS failures
            FROM predictions p WHERE p.run_id=? ORDER BY p.id""",
            (run["id"],),
        )
        recorded = [row["example_id"] for row in checkpoints]
        if recorded != ids[: len(recorded)] or (run["status"] == "COMPLETED" and recorded != ids):
            raise ValueError("Corrupted checkpoint cohort/order")
        for row in checkpoints:
            usage = json.loads(row["usage_json"])
            if any(
                usage.get(key) != (row[key] or 0) for key in ("input_tokens", "output_tokens")
            ) or (
                usage.get("provider_attempts") != row["attempts"]
                or usage.get("total_tokens")
                != (row["input_tokens"] or 0) + (row["output_tokens"] or 0)
                or row["failures"]
                and (
                    row["result_kind"] != "failed_prediction"
                    or row["output_kind"] != "failed_output"
                )
            ):
                raise ValueError("Corrupted failed-output/checkpoint accounting")
            if row["failures"]:
                verify_failed_checkpoint(store, run["id"], row["example_id"], config)


def verify_failed_checkpoint(
    store: ReadOnlyPilotStore, run_id: str, eid: str, config: dict[str, Any]
) -> None:
    row = store._fetch_one(
        "SELECT strategy,output_json,metadata_json FROM predictions "
        "WHERE run_id=? AND example_id=?",
        (run_id, eid),
    )
    if row is None:
        raise ValueError("Missing failed checkpoint")
    metadata = json.loads(row["metadata_json"])
    if metadata.get("evaluation_status") not in {PENDING, "EVALUATED"}:
        raise ValueError("Corrupted failed checkpoint evaluation status")
    ledger = BudgetLedger(InferenceBudget.model_validate(config["budget"]), run_id=run_id)
    for logical in store._fetch_all(
        "SELECT * FROM logical_model_calls WHERE run_id=? AND example_id=?", (run_id, eid)
    ):
        record = LogicalCallRecord.model_validate(logical)
        ledger.logical_calls[record.logical_call_id] = record
    for attempt in store._fetch_all(
        "SELECT * FROM provider_attempts WHERE run_id=? AND example_id=? ORDER BY id",
        (run_id, eid),
    ):
        ledger.record_call(
            ModelCallRecord.model_validate(
                {**attempt, "metadata": json.loads(attempt["metadata_json"])}
            )
        )
    expected = failed_prediction(
        ledger,
        eid,
        row["strategy"],
        historical_projection=bool(metadata.get("historical_failure_checkpoint")),
    )
    if json.loads(row["output_json"]) != expected.output.model_dump(mode="json") or (
        metadata.get("failed_outputs")
        != [output.model_dump(mode="json") for output in expected.failed_outputs]
    ):
        raise ValueError("Failed checkpoint differs from original returned evidence")
    if metadata["evaluation_status"] == PENDING:
        count = store._fetch_one(
            "SELECT COUNT(*) AS n FROM metrics WHERE run_id=? AND example_id=?", (run_id, eid)
        )
        if not metadata.get("historical_failure_checkpoint") or count is None or count["n"]:
            raise ValueError("Ambiguous pending failed checkpoint")


def verify_carried_rows(seed: Path, db: Path) -> None:
    """Compare opaque inherited values for integrity, never evaluate/report quality."""
    original, current = ReadOnlyPilotStore(seed), ReadOnlyPilotStore(db)
    selections = {
        "provider_attempts": ("attempt_id", "*"),
        "logical_model_calls": ("logical_call_id", "*"),
        "predictions": ("id", "id,run_id,example_id,strategy,output_json,confidence,usage_json"),
        "metrics": ("id", "id,run_id,example_id,metric_name,metric_value"),
    }
    with closing(original.connect()) as old, closing(current.connect()) as new:
        for table, (key, columns) in selections.items():
            for row in old.execute(f"SELECT {columns} FROM {table}"):
                actual = new.execute(
                    f"SELECT {columns} FROM {table} WHERE {key}=?", (row[key],)
                ).fetchone()
                if actual is None or dict(actual) != dict(row):
                    raise ValueError(f"Inherited {table} evidence changed in continuation")


def execute_v3(directory: Path, digest: str) -> dict[str, Any]:
    checked = verify_v3(directory, digest)
    protocol = checked["protocol"]
    execution = directory / "execution"
    db = execution / "heldout.sqlite3"
    if (execution / "halted.json").exists():
        raise ValueError("V3 halted; preserve evidence and obtain review")
    if db.exists() and not (execution / "execution_started.json").exists():
        raise ValueError("Unregistered continuation DB")
    verify_live_model(protocol["primary_model"])
    execution.mkdir(exist_ok=True)
    run_ids = [checked["certificate"]["v1_baseline_run_id"]]
    with execution_lock(execution):
        bind_execution_start(execution, digest)
        if not db.exists():
            shutil.copyfile(directory / "continuation_seed.sqlite3", db)
        try:
            ids = test_ids_only(
                Path(protocol["benchmark"]["test_path"]), protocol["benchmark"]["test_sha256"]
            )
            for entry in protocol["matrix"][1:]:
                verify_v3(directory, digest)
                verify_live_model(protocol["primary_model"])
                verify_resume(db, protocol, directory)
                verify_carried_rows(directory / "continuation_seed.sqlite3", db)
                store = ReadOnlyPilotStore(db)
                completed = store.find_completed_run_by_config_hash(entry["resolved_config_sha256"])
                if completed is None:
                    print(
                        json.dumps(
                            {
                                "stage": "FROZEN_TEST_V3",
                                "entry": entry["name"],
                                "new_examples": 168
                                if entry["name"] == "natural_multi_agent_debate"
                                else 200,
                            }
                        ),
                        flush=True,
                    )
                    run = run_experiment_from_path(
                        directory / entry["config_path"],
                        db_path=db,
                        output_dir=execution / "runs",
                        failed_output_policy=True,
                    )
                    if isinstance(run, dict):
                        raise ValueError("Unexpected dry run")
                    run_id = run.run_id
                else:
                    run_id = completed["id"]
                store = ReadOnlyPilotStore(db)
                verify_resume(db, protocol, directory)
                verify_carried_rows(directory / "continuation_seed.sqlite3", db)
                recorded = [
                    r["example_id"]
                    for r in store._fetch_all(
                        "SELECT example_id FROM predictions WHERE run_id=? ORDER BY id", (run_id,)
                    )
                ]
                if recorded != ids or not verify_run_artifacts(store, run_id)["ok"]:
                    raise ValueError("Cohort/checkpoint/artifact corruption")
                run_ids.append(run_id)
                write_progress(
                    execution / "progress.json",
                    {"status": "RUNNING", "run_ids": run_ids, "protocol_sha256": digest},
                )
            combined = CarriedBaselineStore(
                db, PROTOCOL_DIR / "execution/heldout.sqlite3", run_ids[0]
            )
            reports = {}
            for condition in protocol["conditions"]:
                subset = [
                    r
                    for entry, r in zip(protocol["matrix"], run_ids, strict=True)
                    if load_config(directory / entry["config_path"])["phase9"]["condition"]
                    == condition
                ]
                if any(
                    set(combined.example_metric_values(r, "task_score")) != set(ids) for r in subset
                ):
                    raise ValueError("Primary metric denominator is incomplete")
                reports[condition] = {
                    metric: compare_runs_report(
                        combined,
                        subset,
                        metric=metric,
                        n_bootstrap=protocol["statistics"]["bootstrap_replicates"],
                        seed=protocol["statistics"]["seed"],
                    )
                    for metric in ("task_score", "total_tokens", "provider_attempts", "latency_ms")
                }
                if any(not r["matched_budget"]["ok"] for r in reports[condition].values()):
                    raise ValueError("Comparison compatibility failed")
            write_progress(execution / "paired_reports.json", reports)
            write_progress(
                execution / "report_provenance.json",
                {
                    "disclosure": protocol["continuation"]["disclosure"],
                    "contract": POLICY,
                    "protocol_sha256": digest,
                    "run_ids": run_ids,
                    "failed_example_replayed": False,
                },
            )
            result = {"status": "COMPLETED", "protocol_sha256": digest, "run_ids": run_ids}
            write_progress(execution / "progress.json", result)
            return result
        except BaseException as exc:
            write_progress(
                execution / "halted.json",
                {
                    "status": "HALTED",
                    "protocol_sha256": digest,
                    "reason": str(exc),
                    "run_ids": run_ids,
                    "error_type": type(exc).__name__,
                },
            )
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 freeze/preflight; explicit execution only")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--protocol-dir", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    if args.execute and not args.protocol_sha256:
        parser.error("Exact frozen v3 hash required")
    if args.freeze:
        result = {
            "status": "FROZEN",
            "protocol_sha256": freeze_v3(args.protocol_dir),
            "new_inference": False,
        }
    elif args.execute:
        result = execute_v3(args.protocol_dir, args.protocol_sha256)
    else:
        checked = verify_v3(args.protocol_dir, args.protocol_sha256)
        result = {
            k: checked[k]
            for k in ("status", "protocol_sha256", "new_inference", "test_quality_inspected")
        }
    print(json.dumps(result, indent=2))
    return 0
