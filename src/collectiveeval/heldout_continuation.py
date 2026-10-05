"""Representation-only v2 amendment with immutable, certified v1 baseline reuse."""

from __future__ import annotations

import argparse
import copy
import json
import platform
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from collectiveeval.artifact_verify import verify_run_artifacts
from collectiveeval.budget_policy import resolve_budget_policy
from collectiveeval.config import load_config, resolve_env_vars, stable_config_hash
from collectiveeval.core import ModelSpec, ProviderRequest
from collectiveeval.datasets import file_sha256, load_jsonl
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
    verify_resume_evidence,
    verify_test_hash,
    write_progress,
)
from collectiveeval.phase8_corrective import ROOT
from collectiveeval.phase8_corrective_analysis import validate_attempt_records
from collectiveeval.pilot import verify_phase8_dev_freeze
from collectiveeval.pilot_analysis import ReadOnlyPilotStore
from collectiveeval.providers import OpenAICompatibleProvider, _openai_payload
from collectiveeval.reporting import compare_runs_report
from collectiveeval.runner import _config_with_resolved_budget, run_experiment_from_path
from collectiveeval.task_contracts import CONTRACT_VERSION

VERSION = "experimental_protocol_v2"
DIRECTORY = ROOT / "reports" / VERSION
AUDIT = ROOT / "reports" / "protocol_v2_repair_audit"
V1_SHA256 = "e8fc45a7e772d24e6b8e422a1b51e09af07b74b58b950c06c08fb16e61571bce"
METADATA_PATH = (
    "model.provider_options.model_identity.show.model_info."
    "gemma3.vision.attention.layer_norm_epsilon"
)
ALLOWED_CHANGED_SOURCE = {
    "src/collectiveeval/config.py",
    "src/collectiveeval/heldout_protocol.py",
}
ALLOWED_NEW_SOURCE = {
    "src/collectiveeval/heldout_continuation.py",
    "scripts/run_heldout_continuation.py",
    "tests/test_heldout_continuation.py",
}


def representation_diff(a: Any, b: Any, path: str = "") -> list[dict[str, Any]]:
    if type(a) is not type(b):
        return [
            {
                "path": path,
                "frozen_type": type(a).__name__,
                "execution_type": type(b).__name__,
                "frozen_value": a,
                "execution_value": b,
            }
        ]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [{"path": path, "error": "mapping keys differ"}]
        return [d for k in a for d in representation_diff(a[k], b[k], f"{path}.{k}" if path else k)]
    if isinstance(a, list):
        if len(a) != len(b):
            return [{"path": path, "error": "sequence length differs"}]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in representation_diff(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [{"path": path, "frozen_value": a, "execution_value": b}]


def prove_metadata_only(canonical: dict[str, Any], legacy: dict[str, Any]) -> list[dict[str, Any]]:
    differences = representation_diff(canonical, legacy)
    if len(differences) != 1 or differences[0] != {
        "path": METADATA_PATH,
        "frozen_type": "float",
        "execution_type": "str",
        "frozen_value": 1e-6,
        "execution_value": "1e-06",
    }:
        raise ValueError("REUSE_NOT_PROVEN: unexpected semantic or representation difference")
    return [
        {
            **differences[0],
            "classification": "NON_GENERATION_METADATA",
            "change_kind": "REPRESENTATION_ONLY",
        }
    ]


def legacy_yaml(path: Path) -> dict[str, Any]:
    value = resolve_env_vars(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError("Expected legacy config mapping")
    return value


def verify_v1_preservation(v1: Path, preservation: dict[str, Any]) -> None:
    actual = {str(p.relative_to(v1)): file_sha256(p) for p in sorted(v1.rglob("*")) if p.is_file()}
    if actual != preservation["files"]:
        raise ValueError("Preserved v1 protocol/execution evidence changed")
    if stable_config_hash(read_json(v1 / "protocol.json")) != V1_SHA256:
        raise ValueError("Wrong parent protocol identity")
    if (
        preservation["source_hashes_before_repair"]
        != read_json(v1 / "protocol.json")["source_hashes"]
    ):
        raise ValueError("Pre-repair implementation was not the frozen v1 source")


def verify_allowed_source_changes(preservation: dict[str, Any]) -> None:
    before = preservation["source_hashes_before_repair"]
    now = code_identity()
    if set(before) - set(now) or set(now) - set(before) - ALLOWED_NEW_SOURCE:
        raise ValueError("Unapproved source addition/deletion")
    changed = {name for name in before if before[name] != now[name]}
    if changed - ALLOWED_CHANGED_SOURCE:
        raise ValueError("Scientific implementation changed beyond the config-loader repair")


def audit_configs(v1: Path) -> dict[str, Any]:
    protocol = read_json(v1 / "protocol.json")
    rows = []
    example = load_jsonl(ROOT / "data" / "splits" / "dev.mock.jsonl")[0]
    for entry in protocol["matrix"]:
        path = v1 / entry["config_path"]
        canonical = read_json(path)
        legacy = legacy_yaml(path)
        differences = prove_metadata_only(canonical, legacy)
        if resolved_hash(canonical) != entry["resolved_config_sha256"]:
            raise ValueError("Canonical frozen hash does not reproduce")
        if stable_config_hash(load_config(path)) != entry["config_sha256"]:
            raise ValueError("Execution and freeze loaders differ")
        validate_scientific_config(canonical, protocol["primary_model"])
        models = [ModelSpec.model_validate(c["model"]) for c in (canonical, legacy)]
        requests = [
            ProviderRequest(
                example=example,
                model=m,
                strategy=canonical["strategy"]["type"],
                role="generator",
                prompt="offline-equivalence-probe",
            )
            for m in models
        ]
        provider = OpenAICompatibleProvider(provider_id="ollama", base_url=models[0].base_url)
        if (
            _openai_payload(requests[0]) != _openai_payload(requests[1])
            or provider._headers(models[0]) != provider._headers(models[1])
            or models[0].timeout_s != models[1].timeout_s
            or models[0].base_url != models[1].base_url
        ):
            raise ValueError("Provider-facing request semantics changed")
        rows.append(
            {
                "name": entry["name"],
                "differences": differences,
                "canonical_resolved_hash": resolved_hash(canonical),
                "legacy_resolved_hash": resolved_hash(legacy),
                "request_body_headers_endpoint_timeout_equivalent": True,
            }
        )
    return {"configs": rows, "generation_semantic_differences": 0, "test_quality_inspected": False}


def test_ids_only(path: Path, expected_hash: str) -> list[str]:
    verify_test_hash(path, expected_hash)
    # Project only IDs; do not evaluate, report or retain gold/input/answer fields.
    with path.open(encoding="utf-8") as handle:
        return [str(json.loads(line)["id"]) for line in handle if line.strip()]


def completed_interval(record: dict[str, Any]) -> tuple[datetime, datetime]:
    try:
        start = datetime.fromisoformat(record["start_timestamp"])
        end = datetime.fromisoformat(record["end_timestamp"])
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("Invalid completed interval")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("REUSE_NOT_PROVEN: incomplete/invalid call timestamps") from exc
    return start, end


def audit_baseline(v1: Path, preservation: dict[str, Any]) -> dict[str, Any]:
    verify_v1_preservation(v1, preservation)
    verify_allowed_source_changes(preservation)
    protocol = read_json(v1 / "protocol.json")
    halt = read_json(v1 / "execution" / "halted.json")
    if (
        halt["reason"] != "Existing execution contains an unfrozen config"
        or len(halt["completed_run_ids"]) != 1
    ):
        raise ValueError("REUSE_NOT_PROVEN: unexpected halted execution")
    run_id = halt["completed_run_ids"][0]
    entry = protocol["matrix"][0]
    if entry["name"] != "natural_single_agent":
        raise ValueError("REUSE_NOT_PROVEN: unexpected baseline recipe")
    canonical = read_json(v1 / entry["config_path"])
    legacy = legacy_yaml(v1 / entry["config_path"])
    expected_legacy = _config_with_resolved_budget(legacy, resolve_budget_policy(legacy))
    ids = test_ids_only(
        Path(protocol["benchmark"]["test_path"]), protocol["benchmark"]["test_sha256"]
    )
    if len(ids) != protocol["benchmark"]["test_examples"] or len(ids) != len(set(ids)):
        raise ValueError("REUSE_NOT_PROVEN: TEST cohort identity/count mismatch")
    store = ReadOnlyPilotStore(v1 / "execution" / "heldout.sqlite3")
    with closing(store.connect()) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("REUSE_NOT_PROVEN: SQLite integrity failed")
        runs = connection.execute("SELECT id,status,experiment_id FROM runs").fetchall()
        if len(runs) != 1 or runs[0]["id"] != run_id or runs[0]["status"] != "COMPLETED":
            raise ValueError("REUSE_NOT_PROVEN: baseline incomplete or extra runs exist")
        experiment = connection.execute(
            "SELECT config_hash,config_json,dataset_hash FROM experiments WHERE id=?",
            (runs[0]["experiment_id"],),
        ).fetchone()
        persisted = json.loads(experiment["config_json"])
        if stable_config_hash(persisted) != resolved_hash(legacy) or (
            experiment["config_hash"] != resolved_hash(legacy)
            or stable_config_hash(persisted) != stable_config_hash(expected_legacy)
            or experiment["dataset_hash"] != protocol["benchmark"]["test_sha256"]
        ):
            raise ValueError("REUSE_NOT_PROVEN: persisted scientific recipe mismatch")
        prediction_ids = [
            r[0]
            for r in connection.execute(
                "SELECT example_id FROM predictions WHERE run_id=? ORDER BY id", (run_id,)
            )
        ]
        if prediction_ids != ids:
            raise ValueError("REUSE_NOT_PROVEN: missing/duplicate/reordered checkpoints")
        tasks = dict(connection.execute("SELECT example_id,task_type FROM examples"))
        attempts = [
            dict(r)
            for r in connection.execute(
                """SELECT attempt_id,logical_call_id,attempt_index,example_id,
            strategy,provider,model,
            role,prompt_version,requested_seed,temperature,max_tokens,outcome,usage_status,
            usage_source,input_tokens,output_tokens,start_timestamp,end_timestamp,latency_ms,
            retry_count,normalized_error,
            json_extract(metadata_json,'$.generation_settings') AS settings,
            json_extract(metadata_json,'$.model_digest') AS digest,
            json_extract(metadata_json,'$.unknown_usage_policy') AS policy,
            json_extract(metadata_json,'$.parse_status') AS parse_status,
            json_extract(metadata_json,'$.effective_max_tokens') AS effective_max_tokens,
            json_extract(metadata_json,'$.post_call_budget_overrun') AS overrun
            FROM provider_attempts WHERE run_id=? ORDER BY id""",
                (run_id,),
            )
        ]
        logical = [
            dict(r)
            for r in connection.execute(
                "SELECT * FROM logical_model_calls WHERE run_id=?", (run_id,)
            )
        ]
    if len(attempts) != len(ids) or len(logical) != len(ids):
        raise ValueError("REUSE_NOT_PROVEN: logical/attempt count mismatch")
    if (
        len({r["attempt_id"] for r in attempts}) != len(ids)
        or len({r["logical_call_id"] for r in attempts}) != len(ids)
        or [r["example_id"] for r in attempts] != ids
    ):
        raise ValueError("REUSE_NOT_PROVEN: duplicate or missing physical attempts")
    by_logical = {r["logical_call_id"]: r for r in logical}
    if len(by_logical) != len(ids):
        raise ValueError("REUSE_NOT_PROVEN: duplicate logical calls")
    model = canonical["model"]
    settings = {k: model[k] for k in ("temperature", "top_p", "max_tokens", "seed")}
    previous_end = None
    for row in attempts:
        task = tasks[row["example_id"]]
        contract_task = "grounded_qa" if task == "robustness" else task
        matched_logical = by_logical.get(row["logical_call_id"])
        if (
            row["provider"] != model["provider"]
            or row["model"] != model["model"]
            or row["digest"] != model["provider_options"]["model_digest"]
            or row["strategy"] != "single_agent"
            or row["role"] != "generator"
            or row["prompt_version"] != f"{CONTRACT_VERSION}.{contract_task}"
            or row["requested_seed"] != model["seed"]
            or row["temperature"] != model["temperature"]
            or row["max_tokens"] != model["max_tokens"]
            or row["effective_max_tokens"] != model["max_tokens"]
            or json.loads(row["settings"]) != settings
            or row["policy"] != canonical["provider_retry"]["unknown_usage_policy"]
            or row["outcome"] != "SUCCESS"
            or row["usage_status"] != "PROVIDER_REPORTED"
            or row["usage_source"] != "PROVIDER_REPORTED"
            or row["attempt_index"] != 0
            or row["retry_count"] != 0
            or row["normalized_error"]
            or row["parse_status"] not in {"PARSED", "REPAIRED"}
            or row["input_tokens"] is None
            or row["output_tokens"] is None
            or row["input_tokens"] < 0
            or row["output_tokens"] < 0
            or row["input_tokens"] + row["output_tokens"] > canonical["budget"]["max_total_tokens"]
            or not row["start_timestamp"]
            or not row["end_timestamp"]
            or row["latency_ms"] < 0
            or row["overrun"]
            or matched_logical is None
            or matched_logical["outcome"] != "SUCCESS"
            or matched_logical["example_id"] != row["example_id"]
            or matched_logical["strategy"] != row["strategy"]
            or matched_logical["role"] != row["role"]
            or not matched_logical["end_timestamp"]
        ):
            raise ValueError("REUSE_NOT_PROVEN: attempt settings/accounting/contract mismatch")
        start, end = completed_interval(row)
        logical_start, logical_end = completed_interval(matched_logical)
        if (
            start < logical_start
            or end > logical_end
            or previous_end is not None
            and start < previous_end
        ):
            raise ValueError("REUSE_NOT_PROVEN: non-sequential/inconsistent call timestamps")
        previous_end = end
    return {
        "classification": "REUSE_PROVEN_EQUIVALENT",
        "run_id": run_id,
        "parent_protocol_sha256": V1_SHA256,
        "baseline_name": entry["name"],
        "examples": len(ids),
        "example_ids_sha256": stable_config_hash({"ids": ids}),
        "logical_model_calls": len(logical),
        "provider_attempts": len(attempts),
        "input_tokens": sum(r["input_tokens"] for r in attempts),
        "output_tokens": sum(r["output_tokens"] for r in attempts),
        "unknown_usage_attempts": 0,
        "retries": 0,
        "test_sha256": protocol["benchmark"]["test_sha256"],
        "canonical_config_sha256": entry["resolved_config_sha256"],
        "persisted_legacy_config_sha256": resolved_hash(legacy),
        "provider_model_digest": model["provider_options"]["model_digest"],
        "generation_settings": settings,
        "timeout_s": model["timeout_s"],
        "concurrency": canonical["concurrency"],
        "budget": canonical["budget"],
        "retry_policy": canonical["provider_retry"],
        "contracts": protocol["contracts"],
        "v1_test_started_at": read_json(v1 / "execution" / "execution_started.json")["started_at"],
        "v1_evidence_manifest_sha256": stable_config_hash(preservation),
        "test_quality_inspected": False,
        "request_equivalence_proof": "Identical body/header/endpoint/timeout projections; "
        "metadata scalar is never sent. Provider/context/prompt/strategy/schema/budget sources "
        "are unchanged and dataset bytes match. Prompt versions and actual generation settings "
        "match all persisted attempts. This is a code-bound proof, not a provider wire capture.",
    }


def remaining_runtime(protocol: dict[str, Any]) -> dict[str, Any]:
    original = protocol["runtime_estimate"]
    rows = original["matrix"][1:]
    slowest: dict[str, float] = {}
    for row in original["matrix"]:
        slowest[row["strategy"]] = max(
            slowest.get(row["strategy"], 0), row["observed_extrapolated_seconds"]
        )
    return {
        "remaining_runs": len(rows),
        "scientific_predictions": protocol["benchmark"]["test_examples"] * len(rows),
        "expected_provider_attempts": sum(r["expected_provider_attempts"] for r in rows),
        "worst_case_provider_attempts": sum(r["worst_case_provider_attempts"] for r in rows),
        "expected_tokens": sum(r["expected_tokens"] for r in rows),
        "expected_wall_clock_seconds": sum(r["observed_extrapolated_seconds"] for r in rows) * 1.2,
        "conservative_wall_clock_seconds": sum(slowest[r["strategy"]] for r in rows) * 1.3,
        "marginal_api_cost_usd": 0,
        "basis": "Unchanged frozen DEV extrapolation, excluding the carried baseline. "
        "No TEST quality used. Thermal/cold-start/task-mix variation remains possible.",
    }


def freeze_continuation(
    directory: Path = DIRECTORY, v1: Path = PROTOCOL_DIR, audit: Path = AUDIT
) -> str:
    if directory.exists():
        raise ValueError("Continuation version exists; never overwrite a frozen protocol")
    preservation = read_json(audit / "v1_preservation_manifest.json")
    configs_audit = audit_configs(v1)
    certificate = audit_baseline(v1, preservation)
    parent = read_json(v1 / "protocol.json")
    protocol = copy.deepcopy(parent)
    protocol["version"] = VERSION
    protocol["source_hashes"] = code_identity()
    protocol["reproducibility"] = environment_identity()
    protocol["reproducibility"]["git_limit"] = (
        "The published commit plus frozen source hashes identify this loader repair; "
        "the commit alone does not include the uncommitted continuation implementation."
    )
    protocol["continuation"] = {
        "parent_directory": str(v1.resolve()),
        "parent_protocol_sha256": V1_SHA256,
        "change": "JSON files use the central JSON parser, not YAML. No scientific method change.",
        "baseline_status": "COMPLETED_CARRIED_FORWARD_FROM_V1",
        "baseline_run_id": certificate["run_id"],
        "baseline_name": "natural_single_agent",
        "baseline_reuse_certificate_sha256": stable_config_hash(certificate),
        "v1_preservation_manifest_sha256": stable_config_hash(preservation),
        "config_diff_sha256": stable_config_hash(configs_audit),
        "remaining_names": [e["name"] for e in parent["matrix"][1:]],
        "remaining_runtime_estimate": remaining_runtime(parent),
        "v1_test_started_at": certificate["v1_test_started_at"],
        "test_quality_inspected_before_amendment": False,
        "disclosure": "One certified equivalent run executed under v1; nine under v2 "
        "continuation. Methodology unchanged. Preserve all v1 evidence and halted.json.",
    }
    digest = stable_config_hash(protocol)
    directory.mkdir(parents=True)
    artifacts = {
        "protocol.json": protocol,
        "baseline_reuse_certificate.json": certificate,
        "v1_preservation_manifest.json": preservation,
        "config_diff.json": configs_audit,
        "runtime_estimate.json": protocol["continuation"]["remaining_runtime_estimate"],
        "test_matrix.json": {
            "protocol_sha256": digest,
            "entries": [
                {**e, "status": "COMPLETED_CARRIED_FORWARD_FROM_V1" if i == 0 else "PENDING"}
                for i, e in enumerate(protocol["matrix"])
            ],
        },
    }
    for name, payload in artifacts.items():
        write_progress(directory / name, payload)
    for entry in parent["matrix"]:
        path = directory / entry["config_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((v1 / entry["config_path"]).read_bytes())
    note = (
        f"# {VERSION}\n\nProtocol SHA-256: `{digest}`\n\n"
        "Representation-only continuation of v1 after TEST execution began. The loader "
        "now dispatches .json through JSON, retaining YAML for YAML files. All ten configs "
        "are byte-identical to v1; model metadata 1e-06 is no longer mistyped as a string.\n\n"
        f"v1 TEST execution began: {certificate['v1_test_started_at']}. "
        f"Completed baseline: {certificate['run_id']}, 200 examples, 200 logical calls, "
        "200 attempts, 112720 provider-reported tokens, no unknown usage. "
        "Reuse classification: REUSE_PROVEN_EQUIVALENT. No TEST quality results were "
        "inspected before amendment; only cohort IDs, byte hashes and operational settings "
        "were audited. Prediction files were hashed, not interpreted.\n\n"
        "Research question, hypotheses, model/digest, five recipes, budgets, sampling, "
        "contracts, retry policy, metrics, bootstrap, router and exclusions remain unchanged. "
        "There are no ablations or learned router. The only scientific-path change is "
        "JSON loading. New continuation control code certifies read-only baseline reuse.\n\n"
        "The full inherited protocol is in protocol.json. v1 preservation, exact config "
        "diffs and reuse proof are separate hashed artifacts. v1 DB/artifacts/halted.json "
        "are never rewritten. The new DB holds only the remaining nine recipes, in original "
        "order. The final report must disclose one v1 run and nine v2 runs.\n"
    )
    (directory / "protocol.md").write_text(note)
    (directory / "test_matrix.md").write_text(
        "# Continuation Matrix\n\n| Recipe | Status |\n|---|---|\n"
        + "".join(
            f"| {e['name']} | {'COMPLETED_CARRIED_FORWARD_FROM_V1' if i == 0 else 'PENDING'} |\n"
            for i, e in enumerate(parent["matrix"])
        )
    )
    (directory / "integrity_preflight.md").write_text(
        "# Continuation Integrity\n\nNo additional inference during amendment. "
        "Every launch rechecks the certificate, v1 byte hashes, original methodology, "
        "v2 source/dependencies/configs, exact TEST hash/cohort and live model identity. "
        "Unknown/in-flight work blocks automatic replay. v1 halted.json remains intact.\n"
    )
    files = {
        str(p.relative_to(directory)): file_sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    write_progress(
        directory / "protocol_manifest.json",
        {
            "version": VERSION,
            "protocol_sha256": digest,
            "parent_protocol_sha256": V1_SHA256,
            "artifact_hashes": files,
            "additional_inference_during_amendment": False,
            "test_quality_inspected": False,
        },
    )
    verify_continuation(directory, expected_sha256=digest)
    return digest


def verify_continuation(
    directory: Path = DIRECTORY, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    protocol = read_json(directory / "protocol.json")
    manifest = read_json(directory / "protocol_manifest.json")
    digest = stable_config_hash(protocol)
    if (
        protocol["version"] != VERSION
        or digest != manifest["protocol_sha256"]
        or expected_sha256 is not None
        and digest != expected_sha256
    ):
        raise ValueError("Continuation protocol identity mismatch")
    for name, expected in manifest["artifact_hashes"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"Continuation artifact changed: {name}")
    if protocol["source_hashes"] != code_identity():
        raise ValueError("Frozen v2 source changed")
    environment = environment_identity()
    if (
        protocol["reproducibility"]["dependencies"] != environment["dependencies"]
        or protocol["reproducibility"]["python_version"] != platform.python_version()
    ):
        raise ValueError("Frozen v2 Python/dependencies changed")
    preservation = read_json(directory / "v1_preservation_manifest.json")
    continuation = protocol["continuation"]
    v1 = Path(continuation["parent_directory"])
    if str(v1.resolve()) != preservation["v1_root"]:
        raise ValueError("Parent evidence directory changed")
    verify_v1_preservation(v1, preservation)
    verify_allowed_source_changes(preservation)
    parent = read_json(v1 / "protocol.json")
    changed_control = {"version", "source_hashes", "reproducibility", "continuation"}
    if {k: v for k, v in protocol.items() if k not in changed_control} != {
        k: v for k, v in parent.items() if k not in changed_control
    }:
        raise ValueError("Scientific methodology differs from v1")
    if (
        continuation["parent_protocol_sha256"] != V1_SHA256
        or continuation["remaining_names"] != [e["name"] for e in parent["matrix"][1:]]
        or continuation["baseline_status"] != "COMPLETED_CARRIED_FORWARD_FROM_V1"
        or continuation["remaining_runtime_estimate"] != remaining_runtime(parent)
    ):
        raise ValueError("Unapproved continuation matrix/control changes")
    certificate = read_json(directory / "baseline_reuse_certificate.json")
    config_diff = read_json(directory / "config_diff.json")
    if (
        stable_config_hash(preservation) != continuation["v1_preservation_manifest_sha256"]
        or stable_config_hash(certificate) != continuation["baseline_reuse_certificate_sha256"]
        or stable_config_hash(config_diff) != continuation["config_diff_sha256"]
        or certificate != audit_baseline(v1, preservation)
        or config_diff != audit_configs(v1)
        or continuation["baseline_run_id"] != certificate["run_id"]
    ):
        raise ValueError("Baseline reuse certificate/equivalence proof failed")
    configs = {}
    for entry in protocol["matrix"]:
        path = directory / entry["config_path"]
        config = load_config(path)
        if (
            file_sha256(path) != file_sha256(v1 / entry["config_path"])
            or stable_config_hash(config) != entry["config_sha256"]
            or resolved_hash(config) != entry["resolved_config_sha256"]
        ):
            raise ValueError("Continuation config differs from v1")
        validate_scientific_config(config, protocol["primary_model"])
        configs[entry["name"]] = config
    if not verify_phase8_dev_freeze()["ok"]:
        raise ValueError("Benchmark DEV/manifest changed")
    marker = directory / "execution" / "execution_started.json"
    if marker.exists() and read_json(marker)["protocol_sha256"] != digest:
        raise ValueError("Continuation cannot change after execution start")
    return {
        "status": "PASSED",
        "protocol_sha256": digest,
        "protocol": protocol,
        "configs": configs,
        "baseline": certificate,
        "additional_inference": False,
        "test_quality_inspected": False,
        "execution_started": marker.exists(),
    }


class CarriedBaselineStore(ReadOnlyPilotStore):
    """Route per-run report queries without copying or rewriting original rows."""

    def __init__(self, path: Path, baseline: Path, baseline_id: str) -> None:
        super().__init__(path)
        self.baseline = ReadOnlyPilotStore(baseline)
        self.baseline_id = baseline_id

    def _fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        if params and params[0] == self.baseline_id:
            return self.baseline._fetch_one(query, params)
        return super()._fetch_one(query, params)

    def _fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        if params and params[0] == self.baseline_id:
            return self.baseline._fetch_all(query, params)
        return super()._fetch_all(query, params)


def execute_continuation(directory: Path, digest: str) -> dict[str, Any]:
    checked = verify_continuation(directory, expected_sha256=digest)
    protocol = checked["protocol"]
    model = protocol["primary_model"]
    execution = directory / "execution"
    db = execution / "heldout.sqlite3"
    if (execution / "halted.json").exists():
        raise ValueError("v2 halted: review required; no automatic replay")
    if db.exists() and not (execution / "execution_started.json").exists():
        raise ValueError("Unregistered continuation DB")
    allowed = {e["name"]: e["resolved_config_sha256"] for e in protocol["matrix"][1:]}
    verify_resume_evidence(
        db, allowed, dataset_sha256=protocol["benchmark"]["test_sha256"], model=model
    )
    verify_live_model(model)
    execution.mkdir(exist_ok=True)
    run_ids = [checked["baseline"]["run_id"]]
    with execution_lock(execution):
        bind_execution_start(execution, digest)
        try:
            ids = test_ids_only(
                Path(protocol["benchmark"]["test_path"]), protocol["benchmark"]["test_sha256"]
            )
            for entry in protocol["matrix"][1:]:
                verify_continuation(directory, expected_sha256=digest)
                verify_live_model(model)
                verify_resume_evidence(
                    db, allowed, dataset_sha256=protocol["benchmark"]["test_sha256"], model=model
                )
                existing = (
                    ReadOnlyPilotStore(db).find_completed_run_by_config_hash(
                        entry["resolved_config_sha256"]
                    )
                    if db.exists()
                    else None
                )
                if existing is None:
                    print(
                        json.dumps(
                            {
                                "stage": "FROZEN_TEST_V2_CONTINUATION",
                                "entry": entry["name"],
                                "examples": len(ids),
                                "carried_baseline": run_ids[0],
                            }
                        ),
                        flush=True,
                    )
                    result = run_experiment_from_path(
                        directory / entry["config_path"], db_path=db, output_dir=execution / "runs"
                    )
                    if isinstance(result, dict):
                        raise ValueError("Unexpected dry run")
                    run_id = result.run_id
                else:
                    run_id = str(existing["id"])
                store = ReadOnlyPilotStore(db)
                recorded_ids = [
                    r["example_id"]
                    for r in store._fetch_all(
                        "SELECT example_id FROM predictions WHERE run_id=? ORDER BY id", (run_id,)
                    )
                ]
                if recorded_ids != ids:
                    raise ValueError("Continuation cohort/checkpoint mismatch")
                issues = validate_attempt_records(
                    store.get_run_model_calls(run_id), store.get_run_logical_calls(run_id)
                )
                if issues or not verify_run_artifacts(store, run_id)["ok"]:
                    raise ValueError("Continuation accounting/artifact integrity failed")
                run_ids.append(run_id)
                write_progress(
                    execution / "progress.json",
                    {
                        "status": "RUNNING",
                        "protocol_sha256": digest,
                        "run_ids": run_ids,
                        "carried_baseline_run_id": run_ids[0],
                    },
                )
            # Quality analysis occurs only after all nine new scientific runs complete.
            baseline_path = (
                Path(protocol["continuation"]["parent_directory"]) / "execution/heldout.sqlite3"
            )
            combined = CarriedBaselineStore(db, baseline_path, run_ids[0])
            statistics = protocol["statistics"]
            reports = {}
            for condition in protocol["conditions"]:
                condition_ids = [
                    r
                    for e, r in zip(protocol["matrix"], run_ids, strict=True)
                    if checked["configs"][e["name"]]["phase9"]["condition"] == condition
                ]
                reports[condition] = {
                    metric: compare_runs_report(
                        combined,
                        condition_ids,
                        metric=metric,
                        n_bootstrap=statistics["bootstrap_replicates"],
                        seed=statistics["seed"],
                    )
                    for metric in ("task_score", "total_tokens", "provider_attempts", "latency_ms")
                }
                if any(not r["matched_budget"]["ok"] for r in reports[condition].values()):
                    raise ValueError("Continuation comparison failed budget compatibility")
            write_progress(execution / "paired_reports.json", reports)
            write_progress(
                execution / "report_provenance.json",
                {
                    "parent_protocol_sha256": V1_SHA256,
                    "protocol_sha256": digest,
                    "v1_run_ids": run_ids[:1],
                    "v2_run_ids": run_ids[1:],
                    "baseline_reuse": "REUSE_PROVEN_EQUIVALENT",
                    "scientific_methodology_changed": False,
                    "disclosure": protocol["continuation"]["disclosure"],
                },
            )
            result_payload = {
                "status": "COMPLETED",
                "protocol_sha256": digest,
                "run_ids": run_ids,
                "carried_baseline_run_id": run_ids[0],
                "new_runs": 9,
                "test_model_evaluation_occurred": True,
            }
            write_progress(execution / "progress.json", result_payload)
            return result_payload
        except BaseException as exc:
            write_progress(
                execution / "halted.json",
                {
                    "status": "HALTED",
                    "protocol_sha256": digest,
                    "run_ids": run_ids,
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "rule": "Preserve v1 and v2 evidence; no silent replay or overwrite.",
                },
            )
            raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze/preflight v2; inference requires explicit execution"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--protocol-dir", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    if args.execute and not args.protocol_sha256:
        parser.error("Continuation requires the exact frozen v2 --protocol-sha256")
    if args.freeze:
        result = {
            "status": "FROZEN",
            "protocol_sha256": freeze_continuation(args.protocol_dir),
            "additional_inference": False,
        }
    elif args.execute:
        result = execute_continuation(args.protocol_dir, args.protocol_sha256)
    else:
        checked = verify_continuation(args.protocol_dir, expected_sha256=args.protocol_sha256)
        result = {
            k: checked[k]
            for k in (
                "status",
                "protocol_sha256",
                "additional_inference",
                "test_quality_inspected",
                "execution_started",
            )
        }
        result["baseline_reuse"] = checked["baseline"]["classification"]
        result["remaining_runs"] = 9
    print(json.dumps(result, indent=2))
    return 0
