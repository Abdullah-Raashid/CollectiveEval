"""Synthetic protocol fixtures only: never read the actual held-out benchmark or call Ollama."""

import copy
import json
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from collectiveeval.budget import SCIENTIFIC_ATTEMPT_POLICY, UnknownUsageStop
from collectiveeval.config import stable_config_hash
from collectiveeval.datasets import file_sha256
from collectiveeval.heldout_protocol import (
    BUDGET_SEMANTICS_VERSION,
    EXPECTED_FREEZE,
    PROTOCOL_VERSION,
    TEST_PATH,
    code_identity,
    environment_identity,
    freeze_protocol,
    make_test_configs,
    resolved_hash,
    validate_scientific_config,
    verify_protocol,
)
from collectiveeval.heldout_runner import (
    bind_execution_start,
    execute_heldout,
    execution_lock,
    main,
    verify_live_model,
    verify_resume_evidence,
    verify_test_hash,
)
from collectiveeval.pilot import Phase8Settings, strategy_configs


def configs() -> dict:
    settings = Phase8Settings(
        provider="ollama",
        model="gemma3:4b",
        api_key_env="LOCAL_PLACEHOLDER",
        base_url="http://localhost:11434/v1",
        input_cost_per_1k=0,
        output_cost_per_1k=0,
        pilot_cap_usd=0,
        smoke_max_cost_usd=0,
        model_identity={
            "digest": "fixture-digest",
            "name": "gemma3:4b",
            "details": {"quantization_level": "Q4_K_M"},
        },
    )
    templates = strategy_configs(
        settings, dataset_path=Path("dev-fixture.jsonl"), condition="natural", sample_size=32
    )
    return make_test_configs({c["strategy"]["type"]: c for c in templates.values()})


def test_frozen_matrix_is_ten_explicit_sequential_recipes_without_test_reads() -> None:
    with patch(
        "collectiveeval.heldout_runner.load_jsonl", side_effect=AssertionError("TEST opened")
    ):
        matrix = configs()
        assert len(matrix) == 10
        for config in matrix.values():
            validate_scientific_config(config, matrix["natural_single_agent"]["model"])
            assert config["concurrency"] == {"limit": 1}
            assert config["dataset"]["path"] == str(TEST_PATH)
            assert config["provider_retry"]["max_retries"] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "mock",
        "nested_mock",
        "learned",
        "digest",
        "budget",
        "semantics",
        "retry",
        "concurrency",
        "seed",
        "debate",
    ],
)
def test_scientific_recipe_rejects_substitution_or_drift(defect: str) -> None:
    matrix = configs()
    name = "natural_multi_agent_debate" if defect == "debate" else "natural_adaptive_router"
    candidate = copy.deepcopy(matrix[name])
    if defect == "mock":
        candidate["model"]["provider"] = "mock"
    elif defect == "nested_mock":
        candidate["strategy"]["cheap_model"] = {"provider": "mock", "model": "mock-cheap"}
    elif defect == "learned":
        candidate["strategy"].update(
            router="learned", router_artifact="must-not-read-test-or-dev-artifact"
        )
    elif defect == "digest":
        candidate["model"]["provider_options"]["model_digest"] = "different-digest"
    elif defect == "budget":
        candidate["budget"]["max_total_tokens"] = 1234
    elif defect == "semantics":
        candidate["phase9"]["budget_semantics_version"] = "legacy"
    elif defect == "retry":
        candidate["provider_retry"]["max_retries"] = 1
    elif defect == "concurrency":
        candidate["concurrency"]["limit"] = 2
    elif defect == "seed":
        candidate["model"]["seed"] += 1
    else:
        candidate["strategy"]["rounds"] = 3
    with pytest.raises(ValueError):
        validate_scientific_config(candidate, matrix["natural_single_agent"]["model"])


def test_live_identity_requires_matching_digest_and_quantization_without_generation() -> None:
    model = configs()["natural_single_agent"]["model"]
    identity = model["provider_options"]["model_identity"]
    verify_live_model(model, identity)
    with pytest.raises(ValueError, match="digest"):
        verify_live_model(model, identity | {"digest": "wrong"})
    with pytest.raises(ValueError, match="digest"):
        verify_live_model(model, identity | {"details": {"quantization_level": "wrong"}})


def test_start_marker_is_immutable_and_repeat_binding_is_idempotent(tmp_path: Path) -> None:
    bind_execution_start(tmp_path, "protocol-a")
    before = file_sha256(tmp_path / "execution_started.json")
    bind_execution_start(tmp_path, "protocol-a")
    assert file_sha256(tmp_path / "execution_started.json") == before
    with pytest.raises(ValueError, match="cannot change"):
        bind_execution_start(tmp_path, "protocol-b")
    assert file_sha256(tmp_path / "execution_started.json") == before


def test_existing_protocol_cannot_be_rewritten_even_before_execution(tmp_path: Path) -> None:
    with (
        patch("collectiveeval.heldout_protocol.build_protocol") as build,
        pytest.raises(ValueError, match="immutable"),
    ):
        freeze_protocol(tmp_path)
    build.assert_not_called()


def test_execution_lease_blocks_simultaneous_launches(tmp_path: Path) -> None:
    with (
        execution_lock(tmp_path),
        pytest.raises(ValueError, match="lock"),
        execution_lock(tmp_path),
    ):
        pytest.fail("simultaneous execution admitted")
    assert not (tmp_path / "execution.lock").exists()


def test_hash_check_rejects_mutated_synthetic_test_fixture(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-test-fixture.txt"
    path.write_text("unit fixture only")
    verify_test_hash(path, file_sha256(path))
    with pytest.raises(ValueError, match="TEST hash mismatch"):
        verify_test_hash(path, "bad-hash")


def test_execute_requires_explicit_protocol_hash_before_any_access() -> None:
    with (
        patch("sys.argv", ["run_heldout", "--execute"]),
        patch("collectiveeval.heldout_runner.execute_heldout") as run,
        pytest.raises(SystemExit),
    ):
        main()
    run.assert_not_called()


def test_default_preflight_cannot_open_test_or_call_provider() -> None:
    result = {
        "status": "PASSED",
        "protocol_sha256": "fixture",
        "test_file_opened": False,
        "test_execution_started": False,
    }
    with (
        patch("sys.argv", ["run_heldout"]),
        patch("collectiveeval.heldout_runner.verify_protocol", return_value=result),
        patch("collectiveeval.heldout_runner.load_jsonl") as read,
        patch("collectiveeval.heldout_runner.verify_live_model") as live,
    ):
        assert main() == 0
    read.assert_not_called()
    live.assert_not_called()


@pytest.mark.parametrize("state", ["checkpointed", "uncheckpointed", "unknown"])
def test_resume_never_repeats_sent_uncheckpointed_or_unknown_work(
    tmp_path: Path, state: str
) -> None:
    db = tmp_path / "fixture.sqlite3"
    db.touch()
    store = MagicMock()
    store._fetch_all.return_value = [{"id": "run", "status": "RUNNING"}]
    store.get_run_experiment.return_value = {"config_hash": "frozen"}
    store.get_run_predictions.return_value = (
        [{"example_id": "synthetic-fixture"}] if state == "checkpointed" else []
    )
    store.get_run_model_calls.return_value = [
        {
            "attempt_id": "attempt",
            "logical_call_id": "logical",
            "attempt_index": 0,
            "example_id": "synthetic-fixture",
            "outcome": "SUCCESS",
            "usage_status": "UNKNOWN_NOT_RETURNED" if state == "unknown" else "PROVIDER_REPORTED",
            "input_tokens": None if state == "unknown" else 30,
            "output_tokens": None if state == "unknown" else 17,
            "start_timestamp": "2026-10-04T00:00:00+00:00",
            "end_timestamp": "2026-10-04T00:00:01+00:00",
            "metadata_json": json.dumps({"unknown_usage_policy": SCIENTIFIC_ATTEMPT_POLICY}),
        }
    ]
    store.get_run_logical_calls.return_value = [
        {
            "logical_call_id": "logical",
            "outcome": "SUCCESS",
            "end_timestamp": "2026-10-04T00:00:01+00:00",
        }
    ]
    with patch("collectiveeval.heldout_runner.ReadOnlyPilotStore", return_value=store):
        if state == "checkpointed":
            verify_resume_evidence(db, {"entry": "frozen"})
        else:
            with pytest.raises(UnknownUsageStop if state == "unknown" else ValueError):
                verify_resume_evidence(db, {"entry": "frozen"})


def fixture_bundle(directory: Path) -> str:
    matrix = configs()
    protocol = {
        "version": PROTOCOL_VERSION,
        "budget_semantics_version": BUDGET_SEMANTICS_VERSION,
        "source_hashes": code_identity(),
        "reproducibility": {
            "dependencies": environment_identity()["dependencies"],
            "python_version": platform.python_version(),
        },
        "primary_model": matrix["natural_single_agent"]["model"],
        "benchmark": {"test_path": str(TEST_PATH), "test_sha256": EXPECTED_FREEZE["test_sha256"]},
        "matrix": [
            {
                "name": name,
                "config_path": f"configs/{name}.json",
                "config_sha256": stable_config_hash(c),
                "resolved_config_sha256": resolved_hash(c),
            }
            for name, c in matrix.items()
        ],
    }
    (directory / "configs").mkdir()
    paths = ["protocol.json"]
    (directory / "protocol.json").write_text(json.dumps(protocol))
    for name, config in matrix.items():
        path = f"configs/{name}.json"
        paths.append(path)
        (directory / path).write_text(json.dumps(config))
    digest = stable_config_hash(protocol)
    (directory / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": digest,
                "artifact_hashes": {p: file_sha256(directory / p) for p in paths},
            }
        )
    )
    return digest


def test_frozen_bundle_verification_is_test_blind_and_detects_tampering(tmp_path: Path) -> None:
    digest = fixture_bundle(tmp_path)
    original_hash = file_sha256

    def never_hash_actual_test(path: Path) -> str:
        assert Path(path).resolve() != TEST_PATH.resolve(), "actual TEST opened during preflight"
        return original_hash(path)

    with (
        patch(
            "collectiveeval.heldout_protocol.verify_phase8_dev_freeze",
            return_value={
                "ok": True,
                "test_file_opened": False,
                "actual": {"test_sha256_from_manifest": EXPECTED_FREEZE["test_sha256"]},
            },
        ),
        patch("collectiveeval.heldout_protocol.verify_preservation"),
        patch("collectiveeval.heldout_protocol.file_sha256", side_effect=never_hash_actual_test),
    ):
        result = verify_protocol(tmp_path, expected_sha256=digest)
        assert result["test_file_opened"] is False and len(result["configs"]) == 10
        with pytest.raises(ValueError, match="hash mismatch"):
            verify_protocol(tmp_path, expected_sha256="bad-hash")
        with (
            patch(
                "collectiveeval.heldout_protocol.code_identity",
                return_value={"changed.py": "changed"},
            ),
            pytest.raises(ValueError, match="source changed"),
        ):
            verify_protocol(tmp_path)
        path = tmp_path / "configs/natural_single_agent.json"
        config = json.loads(path.read_text())
        config["budget"]["max_calls"] = 7
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="artifact changed"):
            verify_protocol(tmp_path)


def test_preflight_uses_real_dev_only_adapter_response_contract(tmp_path: Path) -> None:
    digest = fixture_bundle(tmp_path)
    with patch("collectiveeval.heldout_protocol.verify_preservation"):
        result = verify_protocol(tmp_path, expected_sha256=digest)
    assert result["test_file_opened"] is False


def test_export_cannot_claim_frozen_before_self_preflight_passes(tmp_path: Path) -> None:
    candidate = {
        "primary_model": configs()["natural_single_agent"]["model"],
        "benchmark": {"synthetic_fixture": True},
        "contracts": {},
        "source_hashes": {},
        "matrix": [],
        "runtime_estimate": {},
    }
    with (
        patch("collectiveeval.heldout_protocol.build_protocol", return_value=(candidate, {})),
        patch("collectiveeval.heldout_protocol.protocol_markdown", return_value="unit fixture"),
        patch(
            "collectiveeval.heldout_protocol.verify_protocol",
            side_effect=ValueError("preflight failed"),
        ) as verify,
        pytest.raises(ValueError, match="preflight failed"),
    ):
        freeze_protocol(tmp_path / "candidate")
    verify.assert_called_once()
    assert not (tmp_path / "candidate/execution/execution_started.json").exists()


def test_unregistered_scientific_db_cannot_be_adopted_before_frozen_start(tmp_path: Path) -> None:
    (tmp_path / "execution").mkdir()
    (tmp_path / "execution/heldout.sqlite3").touch()
    with (
        patch(
            "collectiveeval.heldout_runner.verify_protocol",
            return_value={
                "protocol": {"primary_model": configs()["natural_single_agent"]["model"]},
            },
        ),
        patch("collectiveeval.heldout_runner.verify_live_model") as live,
        pytest.raises(ValueError, match="Unregistered execution DB"),
    ):
        execute_heldout(tmp_path, "unit-fixture-hash")
    live.assert_not_called()
