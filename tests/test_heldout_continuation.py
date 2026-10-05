from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from collectiveeval.budget import LogicalCallRecord, ModelCallRecord
from collectiveeval.budget_policy import resolve_budget_policy
from collectiveeval.config import load_config, load_yaml_config, stable_config_hash
from collectiveeval.datasets import file_sha256
from collectiveeval.heldout_continuation import (
    CarriedBaselineStore,
    audit_baseline,
    audit_configs,
    execute_continuation,
    freeze_continuation,
    legacy_yaml,
    prove_metadata_only,
    remaining_runtime,
    representation_diff,
    verify_continuation,
    verify_v1_preservation,
)
from collectiveeval.heldout_protocol import (
    PROTOCOL_DIR,
    STRATEGIES,
    TEMPERATURES,
    code_identity,
    make_test_configs,
    read_json,
    resolved_hash,
)
from collectiveeval.heldout_runner import verify_resume_evidence
from collectiveeval.runner import _config_with_resolved_budget
from collectiveeval.storage import SQLiteStore


def test_json_scientific_notation_is_float_and_yaml_keeps_its_own_semantics(tmp_path: Path) -> None:
    for suffix in ("json", "yaml", "yml"):
        path = tmp_path / f"config.{suffix}"
        path.write_text('{"epsilon": 1e-06}')
        value = load_config(path)["epsilon"]
        assert type(value) is (float if suffix == "json" else str)
    assert read_json(tmp_path / "config.json") == load_yaml_config(tmp_path / "config.json")


def test_json_does_not_accept_yaml_syntax(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("epsilon: 1e-06\n")
    with pytest.raises(json.JSONDecodeError):
        load_config(path)


def test_json_env_resolution_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXAMPLE_MODEL", "local")
    path = tmp_path / "config.json"
    path.write_text('{"model":"${EXAMPLE_MODEL}"}')
    assert load_config(path)["model"] == "local"
    assert read_json(path)["model"] == "${EXAMPLE_MODEL}"


def synthetic_v1(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    v1 = tmp_path / "v1"
    (v1 / "configs").mkdir(parents=True)
    (v1 / "execution").mkdir()
    dataset = tmp_path / "test-fixture.jsonl"
    dataset.write_text(
        "".join(
            json.dumps({"id": f"synthetic-{i}", "gold": {"never_inspect": True}}) + "\n"
            for i in range(2)
        )
    )
    model = {
        "provider": "ollama",
        "model": "gemma3:4b",
        "base_url": "http://localhost:11434/v1",
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 700,
        "seed": 20261003,
        "timeout_s": 60.0,
        "input_cost_per_1k": 0,
        "output_cost_per_1k": 0,
        "provider_options": {
            "model_digest": "synthetic-digest",
            "model_identity": {
                "details": {"quantization_level": "Q4_K_M"},
                "show": {"model_info": {"gemma3.vision.attention.layer_norm_epsilon": 1e-6}},
            },
        },
    }
    templates = {
        s: {
            "model": {**copy.deepcopy(model), "temperature": TEMPERATURES[s]},
            "strategy": {"type": s},
        }
        for s in STRATEGIES
    }
    templates["self_consistency"]["strategy"]["k"] = 2
    templates["multi_agent_debate"]["strategy"].update(agents=2, rounds=2, revision_rounds=1)
    configs = make_test_configs(templates)
    matrix = []
    for name, config in configs.items():
        config["dataset"]["path"] = str(dataset)
        path = f"configs/{name}.json"
        (v1 / path).write_text(json.dumps(config))
        matrix.append(
            {
                "name": name,
                "config_path": path,
                "config_sha256": stable_config_hash(config),
                "resolved_config_sha256": resolved_hash(config),
            }
        )
    runtime_rows = [
        {
            "strategy": configs[e["name"]]["strategy"]["type"],
            "expected_provider_attempts": 10,
            "worst_case_provider_attempts": 20,
            "expected_tokens": 100,
            "observed_extrapolated_seconds": 30,
        }
        for e in matrix
    ]
    protocol = {
        "version": "experimental_protocol_v1",
        "matrix": matrix,
        "primary_model": model,
        "source_hashes": code_identity(),
        "benchmark": {
            "test_path": str(dataset),
            "test_sha256": file_sha256(dataset),
            "test_examples": 2,
        },
        "contracts": {"task": "task-contracts.v1", "critic": "critic-contract.v1"},
        "conditions": {"natural": {}, "matched_tokens": {}},
        "statistics": {"bootstrap_replicates": 1000, "seed": 20261003},
        "runtime_estimate": {"matrix": runtime_rows},
        "metrics": {"primary": "unchanged"},
    }
    (v1 / "protocol.json").write_text(json.dumps(protocol))
    (v1 / "execution/halted.json").write_text(
        json.dumps(
            {
                "reason": "Existing execution contains an unfrozen config",
                "completed_run_ids": ["baseline"],
            }
        )
    )
    (v1 / "execution/execution_started.json").write_text(
        json.dumps({"started_at": "synthetic-time"})
    )
    db = v1 / "execution/heldout.sqlite3"
    store = SQLiteStore(db)
    legacy = legacy_yaml(v1 / matrix[0]["config_path"])
    normalized = _config_with_resolved_budget(legacy, resolve_budget_policy(legacy))
    store.insert_experiment(
        experiment_id="experiment",
        name="baseline",
        config_hash=resolved_hash(legacy),
        dataset_hash=file_sha256(dataset),
        git_commit="synthetic",
        python_version="3",
        created_at="synthetic",
        config=normalized,
    )
    store.insert_run(
        run_id="baseline",
        experiment_id="experiment",
        strategy="single_agent",
        status="COMPLETED",
        start_ts="synthetic",
    )
    with store.connect() as connection:
        for i in range(2):
            eid = f"synthetic-{i}"
            connection.execute(
                "INSERT INTO examples VALUES (?,?,?,?,?)",
                (eid, "grounded_qa", "ja", "test", "DO_NOT_READ"),
            )
            connection.execute(
                "INSERT INTO predictions "
                "(run_id,example_id,strategy,output_json,confidence,usage_json,metadata_json) "
                "VALUES (?,?,?,?,?,?,?)",
                ("baseline", eid, "single_agent", "DO_NOT_READ", 0, "DO_NOT_READ", "DO_NOT_READ"),
            )
    for i in range(2):
        logical = LogicalCallRecord(
            logical_call_id=f"call-{i}",
            run_id="baseline",
            example_id=f"synthetic-{i}",
            strategy="single_agent",
            role="generator",
            start_timestamp=f"2026-01-01T00:00:0{2 * i}+00:00",
            end_timestamp=f"2026-01-01T00:00:0{2 * i + 1}+00:00",
            outcome="SUCCESS",
        )
        store.insert_logical_call(logical)
        store.insert_model_call(
            ModelCallRecord(
                attempt_id=f"attempt-{i}",
                logical_call_id=logical.logical_call_id,
                run_id="baseline",
                example_id=logical.example_id,
                strategy="single_agent",
                provider="ollama",
                model="gemma3:4b",
                role="generator",
                prompt_version="task-contracts.v1.grounded_qa",
                requested_seed=20261003,
                temperature=0.2,
                max_tokens=700,
                input_tokens=10,
                output_tokens=5,
                usage_source="PROVIDER_REPORTED",
                usage_status="PROVIDER_REPORTED",
                outcome="SUCCESS",
                start_timestamp=logical.start_timestamp,
                end_timestamp=logical.end_timestamp,
                latency_ms=1000,
                estimated_cost_usd=0,
                finish_reason="stop",
                metadata={
                    "generation_settings": {
                        "temperature": 0.2,
                        "top_p": 1.0,
                        "max_tokens": 700,
                        "seed": 20261003,
                    },
                    "model_digest": "synthetic-digest",
                    "unknown_usage_policy": "UNKNOWN_USAGE_STOP.v1",
                    "parse_status": "REPAIRED",
                    "effective_max_tokens": 700,
                    "post_call_budget_overrun": False,
                },
            )
        )
    preservation = {
        "v1_root": str(v1.resolve()),
        "source_hashes_before_repair": code_identity(),
        "files": {str(p.relative_to(v1)): file_sha256(p) for p in v1.rglob("*") if p.is_file()},
    }
    audit = tmp_path / "audit"
    audit.mkdir()
    (audit / "v1_preservation_manifest.json").write_text(json.dumps(preservation))
    return v1, audit, protocol


def test_all_ten_synthetic_configs_share_execution_and_freeze_hashes(tmp_path: Path) -> None:
    v1, _, _ = synthetic_v1(tmp_path)
    result = audit_configs(v1)
    assert len(result["configs"]) == 10
    assert result["generation_semantic_differences"] == 0
    for entry in read_json(v1 / "protocol.json")["matrix"]:
        config = load_yaml_config(v1 / entry["config_path"])
        assert resolved_hash(config) == entry["resolved_config_sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.9),
        ("seed", 7),
        ("model", "other"),
        ("timeout_s", 120),
        ("max_tokens", 701),
    ],
)
def test_true_model_semantic_changes_are_not_normalized_away(
    tmp_path: Path, field: str, value: Any
) -> None:
    v1, _, protocol = synthetic_v1(tmp_path)
    path = v1 / protocol["matrix"][0]["config_path"]
    canonical, legacy = read_json(path), legacy_yaml(path)
    legacy["model"][field] = value
    with pytest.raises(ValueError, match="semantic"):
        prove_metadata_only(canonical, legacy)


def test_strategy_budget_retry_and_schema_changes_rejected(tmp_path: Path) -> None:
    v1, _, protocol = synthetic_v1(tmp_path)
    path = v1 / protocol["matrix"][0]["config_path"]
    canonical = read_json(path)
    for key, value in (
        ("strategy", {"type": "adaptive_router"}),
        ("budget", {"max_calls": 7}),
        ("provider_retry", {"max_retries": 1}),
        ("extra_schema", {"type": "object"}),
    ):
        legacy = legacy_yaml(path)
        legacy[key] = value
        with pytest.raises(ValueError, match="semantic"):
            prove_metadata_only(canonical, legacy)


def test_synthetic_baseline_certificate_never_reads_outputs_or_scores(tmp_path: Path) -> None:
    v1, audit, protocol = synthetic_v1(tmp_path)
    with patch("collectiveeval.heldout_continuation.V1_SHA256", stable_config_hash(protocol)):
        certificate = audit_baseline(v1, read_json(audit / "v1_preservation_manifest.json"))
    assert certificate["classification"] == "REUSE_PROVEN_EQUIVALENT"
    assert certificate["provider_attempts"] == certificate["logical_model_calls"] == 2
    assert certificate["test_quality_inspected"] is False


@pytest.mark.parametrize(
    "change",
    ["unknown", "inflight", "seed", "digest", "duplicate", "missing", "timestamps", "overlap"],
)
def test_baseline_reuse_refused_for_incomplete_or_changed_evidence(
    tmp_path: Path, change: str
) -> None:
    v1, audit, protocol = synthetic_v1(tmp_path)
    with sqlite3.connect(v1 / "execution/heldout.sqlite3") as connection:
        mutations = {
            "unknown": (
                "UPDATE provider_attempts SET usage_status='UNKNOWN_NOT_RETURNED' WHERE id=1"
            ),
            "inflight": (
                "UPDATE logical_model_calls SET outcome='STARTED' WHERE logical_call_id='call-0'"
            ),
            "seed": "UPDATE provider_attempts SET requested_seed=7 WHERE id=1",
            "digest": (
                "UPDATE provider_attempts SET "
                "metadata_json=json_set(metadata_json,'$.model_digest','other') WHERE id=1"
            ),
            "duplicate": "UPDATE provider_attempts SET example_id='synthetic-0' WHERE id=2",
            "missing": "DELETE FROM predictions WHERE example_id='synthetic-1'",
            "timestamps": (
                "UPDATE provider_attempts SET end_timestamp='2025-01-01T00:00:00+00:00' WHERE id=1"
            ),
            "overlap": (
                "UPDATE provider_attempts SET start_timestamp='2026-01-01T00:00:00+00:00' "
                "WHERE id=2"
            ),
        }
        connection.execute(mutations[change])
    # Even if an audit snapshot captured already-invalid evidence, semantic checks refuse it.
    preservation = read_json(audit / "v1_preservation_manifest.json")
    preservation["files"]["execution/heldout.sqlite3"] = file_sha256(
        v1 / "execution/heldout.sqlite3"
    )
    with (
        patch("collectiveeval.heldout_continuation.V1_SHA256", stable_config_hash(protocol)),
        pytest.raises(ValueError, match="REUSE_NOT_PROVEN"),
    ):
        audit_baseline(v1, preservation)


def test_v1_mutation_cannot_be_hidden_by_continuation(tmp_path: Path) -> None:
    v1, audit, protocol = synthetic_v1(tmp_path)
    preservation = read_json(audit / "v1_preservation_manifest.json")
    (v1 / "execution/halted.json").write_text("changed")
    with (
        patch("collectiveeval.heldout_continuation.V1_SHA256", stable_config_hash(protocol)),
        pytest.raises(ValueError, match="evidence changed"),
    ):
        verify_v1_preservation(v1, preservation)


def test_v2_freezes_nine_remaining_recipes_and_refuses_tampering(tmp_path: Path) -> None:
    v1, audit, protocol = synthetic_v1(tmp_path)
    v2 = tmp_path / "v2"
    with patch("collectiveeval.heldout_continuation.V1_SHA256", stable_config_hash(protocol)):
        digest = freeze_continuation(v2, v1, audit)
        result = verify_continuation(v2, expected_sha256=digest)
        assert result["baseline"]["classification"] == "REUSE_PROVEN_EQUIVALENT"
        assert len(result["protocol"]["continuation"]["remaining_names"]) == 9
        assert result["protocol"]["matrix"] == protocol["matrix"]
        assert remaining_runtime(protocol)["expected_provider_attempts"] == 90
        with pytest.raises(ValueError, match="exists"):
            freeze_continuation(v2, v1, audit)
        (v2 / "baseline_reuse_certificate.json").write_text("{}")
        with pytest.raises(ValueError, match="artifact changed"):
            verify_continuation(v2, expected_sha256=digest)


def test_bad_preflight_never_reaches_inference(tmp_path: Path) -> None:
    with (
        patch(
            "collectiveeval.heldout_continuation.verify_continuation",
            side_effect=ValueError("bad certificate"),
        ),
        patch("collectiveeval.heldout_continuation.run_experiment_from_path") as run,
        pytest.raises(ValueError, match="certificate"),
    ):
        execute_continuation(tmp_path, "bad-hash")
    run.assert_not_called()


def test_carried_store_routes_to_read_only_baseline(tmp_path: Path) -> None:
    v1, _, _ = synthetic_v1(tmp_path)
    db = tmp_path / "new.sqlite3"
    SQLiteStore(db)
    before = file_sha256(v1 / "execution/heldout.sqlite3")
    store = CarriedBaselineStore(db, v1 / "execution/heldout.sqlite3", "baseline")
    assert store.get_run("baseline")["status"] == "COMPLETED"
    assert file_sha256(v1 / "execution/heldout.sqlite3") == before


def test_resume_rejects_unknown_sent_work(tmp_path: Path) -> None:
    v1, _, _ = synthetic_v1(tmp_path)
    db = v1 / "execution/heldout.sqlite3"
    with sqlite3.connect(db) as connection:
        config_hash = connection.execute("SELECT config_hash FROM experiments").fetchone()[0]
        connection.execute(
            "UPDATE provider_attempts SET usage_status='UNKNOWN_NOT_RETURNED' WHERE id=1"
        )
    with pytest.raises(Exception, match="Unknown sent usage"):
        verify_resume_evidence(db, {"baseline": config_hash})


def test_real_ten_frozen_config_hashes_without_test_outputs() -> None:
    if not (PROTOCOL_DIR / "protocol.json").exists():
        pytest.skip("Private frozen protocol bundle unavailable")
    for entry in read_json(PROTOCOL_DIR / "protocol.json")["matrix"]:
        config = load_yaml_config(PROTOCOL_DIR / entry["config_path"])
        assert stable_config_hash(config) == entry["config_sha256"]
        assert resolved_hash(config) == entry["resolved_config_sha256"]


def test_diff_distinguishes_equal_value_different_types() -> None:
    assert representation_diff({"value": 1}, {"value": 1.0})[0]["path"] == "value"


def test_nine_run_continuation_and_completed_resume_never_replay_baseline(tmp_path: Path) -> None:
    v1, audit, protocol = synthetic_v1(tmp_path)
    v2 = tmp_path / "v2"
    dispatched: list[str] = []

    def fake_run(path: Path, *, db_path: Path, output_dir: Path) -> SimpleNamespace:
        del output_dir
        config = load_config(path)
        run_id = path.stem
        dispatched.append(run_id)
        store = SQLiteStore(db_path)
        normalized = _config_with_resolved_budget(config, resolve_budget_policy(config))
        store.insert_experiment(
            experiment_id=run_id,
            name=run_id,
            config_hash=stable_config_hash(normalized),
            dataset_hash=protocol["benchmark"]["test_sha256"],
            git_commit="synthetic",
            python_version="3",
            created_at="synthetic",
            config=normalized,
        )
        store.insert_run(
            run_id=run_id,
            experiment_id=run_id,
            strategy=config["strategy"]["type"],
            status="COMPLETED",
            start_ts="synthetic",
        )
        with store.connect() as connection:
            for i in range(2):
                eid = f"synthetic-{i}"
                connection.execute(
                    "INSERT OR IGNORE INTO examples VALUES (?,?,?,?,?)",
                    (eid, "grounded_qa", "ja", "test", "DO_NOT_READ"),
                )
                connection.execute(
                    "INSERT INTO predictions "
                    "(run_id,example_id,strategy,output_json,confidence,usage_json,metadata_json) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (run_id, eid, config["strategy"]["type"], "{}", 0, "{}", "{}"),
                )
        for i in range(2):
            logical = LogicalCallRecord(
                logical_call_id=f"{run_id}-{i}",
                run_id=run_id,
                example_id=f"synthetic-{i}",
                strategy=config["strategy"]["type"],
                role="generator",
                start_timestamp="2026-01-01T00:00:00+00:00",
                end_timestamp="2026-01-01T00:00:01+00:00",
                outcome="SUCCESS",
            )
            store.insert_logical_call(logical)
            model = config["model"]
            store.insert_model_call(
                ModelCallRecord(
                    attempt_id=f"{run_id}-attempt-{i}",
                    logical_call_id=logical.logical_call_id,
                    run_id=run_id,
                    example_id=logical.example_id,
                    strategy=logical.strategy,
                    provider="ollama",
                    model="gemma3:4b",
                    role="generator",
                    prompt_version="task-contracts.v1.grounded_qa",
                    requested_seed=model["seed"],
                    temperature=model["temperature"],
                    max_tokens=700,
                    input_tokens=10,
                    output_tokens=5,
                    usage_source="PROVIDER_REPORTED",
                    usage_status="PROVIDER_REPORTED",
                    outcome="SUCCESS",
                    start_timestamp=logical.start_timestamp,
                    end_timestamp=logical.end_timestamp,
                    latency_ms=1000,
                    estimated_cost_usd=0,
                    finish_reason="stop",
                    metadata={
                        "model_digest": "synthetic-digest",
                        "unknown_usage_policy": "UNKNOWN_USAGE_STOP.v1",
                    },
                )
            )
        return SimpleNamespace(run_id=run_id)

    with (
        patch("collectiveeval.heldout_continuation.V1_SHA256", stable_config_hash(protocol)),
        patch("collectiveeval.heldout_continuation.verify_live_model"),
        patch("collectiveeval.heldout_continuation.run_experiment_from_path", side_effect=fake_run),
        patch(
            "collectiveeval.heldout_continuation.verify_run_artifacts", return_value={"ok": True}
        ),
        patch(
            "collectiveeval.heldout_continuation.compare_runs_report",
            return_value={"matched_budget": {"ok": True}},
        ),
    ):
        digest = freeze_continuation(v2, v1, audit)
        result = execute_continuation(v2, digest)
        assert dispatched == [e["name"] for e in protocol["matrix"][1:]]
        assert result["run_ids"][0] == "baseline" and result["new_runs"] == 9
        dispatched.clear()
        resumed = execute_continuation(v2, digest)
        assert dispatched == [] and resumed["run_ids"] == result["run_ids"]
        verify_v1_preservation(v1, read_json(audit / "v1_preservation_manifest.json"))


@pytest.mark.parametrize("guard", ["halt", "unregistered_db", "model_mismatch"])
def test_continuation_launch_guards_never_dispatch(tmp_path: Path, guard: str) -> None:
    v1, audit, protocol = synthetic_v1(tmp_path)
    v2 = tmp_path / "v2"
    with patch("collectiveeval.heldout_continuation.V1_SHA256", stable_config_hash(protocol)):
        digest = freeze_continuation(v2, v1, audit)
        execution = v2 / "execution"
        execution.mkdir()
        if guard == "halt":
            (execution / "halted.json").write_text("{}")
        if guard == "unregistered_db":
            SQLiteStore(execution / "heldout.sqlite3")
        with (
            patch("collectiveeval.heldout_continuation.run_experiment_from_path") as run,
            patch(
                "collectiveeval.heldout_continuation.verify_live_model",
                side_effect=ValueError("model mismatch") if guard == "model_mismatch" else None,
            ),
            pytest.raises(ValueError),
        ):
            execute_continuation(v2, digest)
        run.assert_not_called()


def test_new_hash_cannot_authorize_changed_scientific_methodology(tmp_path: Path) -> None:
    v1, audit, protocol = synthetic_v1(tmp_path)
    v2 = tmp_path / "v2"
    with patch("collectiveeval.heldout_continuation.V1_SHA256", stable_config_hash(protocol)):
        freeze_continuation(v2, v1, audit)
        changed = read_json(v2 / "protocol.json")
        changed["metrics"]["primary"] = "post-hoc-different"
        (v2 / "protocol.json").write_text(json.dumps(changed))
        manifest = read_json(v2 / "protocol_manifest.json")
        manifest["protocol_sha256"] = stable_config_hash(changed)
        manifest["artifact_hashes"]["protocol.json"] = file_sha256(v2 / "protocol.json")
        (v2 / "protocol_manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="methodology differs"):
            verify_continuation(v2, expected_sha256=manifest["protocol_sha256"])
