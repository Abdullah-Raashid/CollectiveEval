"""Freeze a DEV-derived held-out protocol without opening the TEST benchmark."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from collectiveeval.budget import BUDGET_SEMANTICS_VERSION, SCIENTIFIC_ATTEMPT_POLICY
from collectiveeval.budget_policy import resolve_budget_policy
from collectiveeval.config import load_config, stable_config_hash
from collectiveeval.datasets import file_sha256
from collectiveeval.failures import FailureType
from collectiveeval.phase8_attempt_repair import REPAIRED, verify_preservation
from collectiveeval.phase8_corrective import ORIGINAL, ROOT, recursive_model_audit, source_hashes
from collectiveeval.pilot import EXPECTED_FREEZE, MANIFEST_PATH, TEST_PATH, verify_phase8_dev_freeze
from collectiveeval.pilot_analysis import ReadOnlyPilotStore
from collectiveeval.runner import _config_with_resolved_budget
from collectiveeval.task_contracts import CONTRACT_VERSION, CRITIC_CONTRACT_VERSION

PROTOCOL_VERSION = "experimental_protocol_v1"
PROTOCOL_DIR = ROOT / "reports" / PROTOCOL_VERSION
STRATEGIES = (
    "single_agent",
    "self_consistency",
    "critic_reviser",
    "multi_agent_debate",
    "adaptive_router",
)
CONDITIONS = {"natural": 12000, "matched_tokens": 4000}
WORST_CALLS = dict(zip(STRATEGIES, (1, 2, 4, 4, 5), strict=True))
TEMPERATURES = dict(zip(STRATEGIES, (0.2, 0.6, 0.2, 0.5, 0.2), strict=True))


def read_json(path: Path) -> dict[str, Any]:
    return load_config(path, resolve_env=False)


def code_identity() -> dict[str, str]:
    return source_hashes() | {"pyproject.toml": file_sha256(ROOT / "pyproject.toml")}


def resolved_hash(config: dict[str, Any]) -> str:
    # Use the runner's pure normalization; dry_run_summary would open TEST.
    return stable_config_hash(
        _config_with_resolved_budget(config, resolve_budget_policy(config), max_examples=None)
    )


def validate_scientific_config(config: dict[str, Any], model: dict[str, Any]) -> None:
    strategy = config["strategy"]
    name = strategy["type"]
    if name not in STRATEGIES:
        raise ValueError("Unfrozen strategy")
    if strategy.get("router", "heuristic") != "heuristic" or strategy.get("router_artifact"):
        raise ValueError("Learned/dev router substitution is forbidden")
    recursive_model_audit(config)
    if name == "self_consistency" and strategy.get("k") != 2:
        raise ValueError("SelfConsistency must be K=2 with frozen distinct requested seeds")
    if name == "multi_agent_debate" and (
        strategy.get("agents") != 2 or strategy.get("rounds") != 2
    ):
        raise ValueError("Debate must be two initial agents and one revision round")
    phase = config["phase9"]
    if phase["version"] != PROTOCOL_VERSION or phase["condition"] not in CONDITIONS:
        raise ValueError("Unfrozen condition/version")
    if phase["budget_semantics_version"] != BUDGET_SEMANTICS_VERSION:
        raise ValueError("Budget semantics drift")
    if config["provider_retry"] != {
        "max_retries": 0,
        "unknown_usage_policy": SCIENTIFIC_ATTEMPT_POLICY,
    }:
        raise ValueError("Scientific attempt policy drift")
    if config["concurrency"] != {"limit": 1}:
        raise ValueError("Sequential local resource policy drift")
    if (
        config["budget"]["max_calls"] != 6
        or config["budget"]["max_total_tokens"] != CONDITIONS[phase["condition"]]
    ):
        raise ValueError("Scientific budget drift")
    actual_model = config["model"]
    for key in ("provider", "model", "base_url", "seed", "top_p", "max_tokens", "timeout_s"):
        if actual_model[key] != model[key]:
            raise ValueError(f"Model/generation setting drift: {key}")
    if actual_model["temperature"] != TEMPERATURES[name]:
        raise ValueError("Strategy recipe temperature drift")
    if (
        actual_model["provider_options"]["model_digest"]
        != model["provider_options"]["model_digest"]
    ):
        raise ValueError("Model digest drift")
    if actual_model["input_cost_per_1k"] != 0 or actual_model["output_cost_per_1k"] != 0:
        raise ValueError("Local marginal API price drift")


def scientific_templates() -> dict[str, dict[str, Any]]:
    originals = ReadOnlyPilotStore(ORIGINAL / "phase8.sqlite3")
    experiment = originals.get_run_experiment("80f05b5e-8c7c-41b5-9963-8ccf21cb282b")
    if experiment is None:
        raise ValueError("Missing approved historical SingleAgent baseline")
    result = {"single_agent": json.loads(experiment["config_json"])}
    for name in STRATEGIES[1:]:
        result[name] = read_json(REPAIRED / "configs" / f"matched_tokens_{name}.json")
    return result


def make_test_configs(templates: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    configs = {}
    for condition, ceiling in CONDITIONS.items():
        for name in STRATEGIES:
            config = copy.deepcopy(templates[name])
            config.pop("phase8", None)
            config.pop("phase8_2", None)
            config["dataset"] = {"path": str(TEST_PATH)}
            config["experiment"] = {"name": f"{PROTOCOL_VERSION}_{condition}_{name}"}
            config["budget"] = {"max_calls": 6, "max_total_tokens": ceiling}
            config["budget_policy"] = {"mode": "MATCHED_TOKENS"}
            config["provider_retry"] = {
                "max_retries": 0,
                "unknown_usage_policy": SCIENTIFIC_ATTEMPT_POLICY,
            }
            config["concurrency"] = {"limit": 1}
            config["checkpoint"] = {"resume_completed_examples": True, "unit": "example"}
            config["phase9"] = {
                "version": PROTOCOL_VERSION,
                "condition": condition,
                "budget_semantics_version": BUDGET_SEMANTICS_VERSION,
                "seed_policy": "base_seed_plus_sample_index.v1",
            }
            configs[f"{condition}_{name}"] = config
    return configs


def estimate_test_runtime(analysis: dict[str, Any], count: int) -> dict[str, Any]:
    entries = []
    for condition in CONDITIONS:
        for name in STRATEGIES:
            metrics = analysis["strategy_results"][condition][name]["metrics"]
            entries.append(
                {
                    "condition": condition,
                    "strategy": name,
                    "examples": count,
                    "expected_provider_attempts": count * metrics["model_calls"]["mean"],
                    "worst_case_provider_attempts": count * WORST_CALLS[name],
                    "expected_tokens": count * metrics["total_tokens"]["mean"],
                    "observed_extrapolated_seconds": count
                    * metrics["wall_clock_strategy_latency_ms"]["mean"]
                    / 1000,
                }
            )
    observed = sum(e["observed_extrapolated_seconds"] for e in entries)
    conservative = (
        2
        * count
        * sum(
            max(
                analysis["strategy_results"][c][s]["metrics"]["wall_clock_strategy_latency_ms"][
                    "mean"
                ]
                for c in CONDITIONS
            )
            / 1000
            for s in STRATEGIES
        )
    )
    return {
        "test_examples": count,
        "scientific_predictions": count * len(STRATEGIES) * len(CONDITIONS),
        "matrix": entries,
        "expected_provider_attempts": sum(e["expected_provider_attempts"] for e in entries),
        "worst_case_provider_attempts": sum(e["worst_case_provider_attempts"] for e in entries),
        "expected_tokens": sum(e["expected_tokens"] for e in entries),
        "observed_extrapolated_seconds": observed,
        "expected_wall_clock_seconds": observed * 1.2,
        "conservative_wall_clock_seconds": conservative * 1.3,
        "marginal_api_cost_usd": 0,
        "assumptions": [
            "200/32 linear extrapolation of the corrected stratified DEV pilot; not a guarantee.",
            "Expected time includes 20% overhead; conservative time uses the slower observed "
            "condition for each strategy plus 30% overhead.",
            "Thermal throttling, context defaults, cold starts and task mix can change time.",
            "Worst calls include one critic schema-repair generation, not a provider retry.",
            "Local compute/electricity are not universally free; quality/$ is undefined here.",
        ],
    }


def environment_identity() -> dict[str, Any]:
    packages = {}
    for name in (
        "pydantic",
        "PyYAML",
        "numpy",
        "scikit-learn",
        "fastapi",
        "httpx",
        "pytest",
        "ruff",
        "mypy",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "."],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "git_commit": commit.stdout.strip(),
        "git_dirty": bool(dirty.stdout.strip()),
        "git_status": dirty.stdout.strip(),
        "git_limit": "This project is untracked under the parent checkout; source hashes, "
        "not the parent commit alone, identify the implementation.",
        "python": sys.version,
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": packages,
    }


def build_protocol() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    integrity = read_json(REPAIRED / "integrity.json")
    gates = read_json(REPAIRED / "static_gates.json")
    if integrity["status"] != "PASSED" or len(integrity["scientific_run_ids"]) != 5:
        raise ValueError("Phase 8.2 must be completed and integrity-clean before Phase 9")
    if not all(gates.get(f"{gate}_passed") for gate in ("pytest", "ruff", "mypy")):
        raise ValueError("Static gates must pass before freeze")
    if gates["source_hashes"] != source_hashes():
        raise ValueError("Run DEV-safe static gates again after code additions")
    verify_preservation()
    freeze = verify_phase8_dev_freeze()
    if not freeze["ok"] or freeze["test_file_opened"]:
        raise ValueError("DEV-only benchmark freeze failed")
    manifest = read_json(MANIFEST_PATH)
    count = manifest["files"]["test.jsonl"]["examples"]
    dev_counts = freeze["dev_validation"]["counts"]
    test_tasks = {
        key: value - dev_counts["task_type"][key]
        for key, value in manifest["counts"]["task_type"].items()
    }
    test_difficulty = {
        key: value - dev_counts["difficulty"][key]
        for key, value in manifest["counts"]["difficulty"].items()
    }
    if sum(test_tasks.values()) != count or sum(test_difficulty.values()) != count:
        raise ValueError("Manifest-derived TEST composition mismatch")
    configs = make_test_configs(scientific_templates())
    model = configs["natural_single_agent"]["model"]
    if model["provider_options"]["model_digest"] != integrity["model_digest"]:
        raise ValueError("DEV model identity differs from proposed primary model")
    for config in configs.values():
        validate_scientific_config(config, model)
    analysis = read_json(REPAIRED / "phase8_analysis.json")
    protocol = {
        "version": PROTOCOL_VERSION,
        "research_question": "Under a fixed inference budget, when do multi-agent strategies "
        "outperform single-agent and self-consistency baselines on Japanese enterprise AI tasks, "
        "and when do they fail?",
        "development": {
            "dev_used_for_protocol_development": True,
            "test_accessed": False,
            "corrective_run_ids": integrity["scientific_run_ids"],
            "corrected_dev_sha256": file_sha256(REPAIRED / "corrected_phase8_view.json"),
            "integrity_sha256": file_sha256(REPAIRED / "integrity.json"),
            "preservation": verify_preservation(),
            "interpretation": "CriticReviser/Router matched-token regressions disappeared after "
            "admission correction; Debate remained worse. These are DEV point estimates, "
            "not guaranteed TEST effects or automatic significance claims.",
        },
        "hypotheses": {
            "H1": "Additional orchestration does not uniformly improve quality over SingleAgent.",
            "H2": "CriticReviser can improve some task families under both conditions; "
            "a shared token ceiling need not eliminate gains when it is nonbinding.",
            "H3": "Heuristic routing uses fewer tokens/calls than always-on critique or debate, "
            "with useful but not uniformly better quality tradeoffs.",
            "H4": "The frozen Debate recipe may have unfavorable quality/compute tradeoffs.",
            "H5": "Strategy effects vary by task family and witnessed reasoning requirement.",
        },
        "benchmark": {
            **EXPECTED_FREEZE,
            "test_path": str(TEST_PATH),
            "test_examples": count,
            "test_task_counts": test_tasks,
            "test_difficulty_counts": test_difficulty,
            "composition_source": "Manifest totals minus DEV counts; no TEST file opened.",
            "selection": "All TEST examples in frozen file order; no resampling or exclusions.",
            "provenance": "Synthetic benchmark; labels do not imply real-enterprise provenance.",
        },
        "primary_model": model,
        "generation": {
            "temperatures": TEMPERATURES,
            "router_nested_debate_temperature": 0.5,
            "requested_seed_policy": "base_seed_plus_sample_index.v1",
            "self_consistency_seeds": [model["seed"], model["seed"] + 1],
            "concurrency": 1,
            "retry_max": 0,
            "unknown_usage_policy": SCIENTIFIC_ATTEMPT_POLICY,
            "context": "Model declares 131072 context capacity; effective server context is "
            "not independently measured or overridden. Keep server configuration unchanged.",
            "determinism_limit": "Requested seeds and deterministic aggregation do not guarantee "
            "deterministic real API replay, different outputs or independent samples.",
        },
        "strategies": {
            name: {
                "config": configs[f"natural_{name}"]["strategy"],
                "worst_case_calls_per_example": WORST_CALLS[name],
            }
            for name in STRATEGIES
        },
        "contracts": {
            "task": CONTRACT_VERSION,
            "critic": CRITIC_CONTRACT_VERSION,
            "source_sha256": file_sha256(ROOT / "src/collectiveeval/task_contracts.py"),
            "aggregation": "Canonical majority; K=2 ties use mean confidence then earliest "
            "sample, returning the earliest candidate in the winning group.",
        },
        "conditions": {
            condition: {
                "scope": "PER_EXAMPLE",
                "max_total_tokens": ceiling,
                "max_provider_attempts": 6,
                "purpose": "Operational high ceiling"
                if condition == "natural"
                else "Shared "
                "token allowance; compare actual consumption and flag nonbinding ceilings.",
            }
            for condition, ceiling in CONDITIONS.items()
        },
        "budget_semantics_version": BUDGET_SEMANTICS_VERSION,
        "budget_rules": [
            "Prompt-only estimation includes public output schema, not answer/evaluator metadata.",
            "Central output clipping; actual usage authoritative; unused allowance not charged.",
            "Retain completed output/usage after overrun; block later calls and flag overrun.",
            "max_calls counts provider attempts, not logical generations; count every attempt.",
            "Shared allowance is not equal realized spending. Report actual usage and overruns, "
            "not a claim of exact consumed-token parity.",
        ],
        "metrics": {
            "primary": "Mean deterministic task_score across all 200 examples (50 per family).",
            "task_score_definitions": {
                "grounded_qa_and_robustness": "Answerable: mean(exact_match, token_f1, "
                "citation_f1, 1-unsupported_answer_claims_heuristic); abstaining gives zero. "
                "Unanswerable: one only for abstention with empty answer, otherwise zero.",
                "structured_extraction": "Mean(JSON Schema validity, field exact-match mean, "
                "numeric-field accuracy, date-field accuracy).",
                "business_summarization": "Mean(schema compliance, decision correctness, "
                "action-item correctness, risk correctness, supported-fact coverage).",
            },
            "task_specific": [
                "exact_match",
                "token_f1",
                "citation_precision",
                "citation_recall",
                "citation_f1",
                "abstention_precision",
                "abstention_recall",
                "abstention_f1",
                "over_abstention_rate",
                "under_abstention_rate",
                "schema_compliance",
                "json_schema_validity",
                "field_exact_match_mean",
                "numeric_field_accuracy",
                "date_field_accuracy",
                "decision_extraction_correctness",
                "action_item_correctness",
                "risk_extraction_correctness",
                "supported_fact_coverage",
                "unsupported_claim_rate_heuristic",
            ],
            "efficiency": [
                "logical_model_calls",
                "provider_attempts",
                "failed_attempts",
                "unknown_usage_attempts",
                "provider-reported input/output/total tokens",
                "known token lower bound",
                "attempt latency",
                "logical lifecycle latency",
                "strategy wall-clock latency",
                "quality per 1k tokens",
                "within-condition quality/resource Pareto dominance",
            ],
            "marginal_api_cost_usd": 0,
            "quality_per_dollar": None,
            "failure_taxonomy": [str(value) for value in FailureType],
            "operational_events": [
                "PRE_CALL_BUDGET_REJECTION",
                "POST_CALL_BUDGET_OVERRUN",
                "PARSE_ERROR",
                "TIMEOUT",
                "UNKNOWN_USAGE_STOP",
                "INTERRUPTED_TRAJECTORY_STOP",
            ],
            "limits": "Groundedness and causal-sounding failure annotations are deterministic "
            "heuristics, not human-validated truth. Groundedness is not 1-citation_precision.",
        },
        "statistics": {
            "confidence": 0.95,
            "bootstrap_replicates": 1000,
            "seed": 20261003,
            "procedure": "Existing percentile bootstrap of example means; paired bootstrap of "
            "contender minus SingleAgent within condition over exact common IDs.",
            "breakdowns": ["task_type", "difficulty", "reasoning_family"],
            "tiny_cell_threshold": 5,
            "automatic_significance_claims": False,
            "multiple_comparisons": "Descriptive intervals; no adjusted discovery claims.",
        },
        "router": {
            "primary": "Frozen heuristic only; thresholds critic=0.45, debate=0.75.",
            "learned_router": "Excluded from primary TEST; future DEV-only work. "
            "No mock-trained artifact.",
            "gold_policy": "Gold answers/scores only in retrospective evaluation/training targets, "
            "never inference features. Public task schema is stored in gold.json_schema but is "
            "an exposed task specification, not a reference answer.",
            "difficulty_limit": "Heuristic uses benchmark-provided difficulty metadata; "
            "deployment without supplied difficulty labels is not validated.",
        },
        "ablations": [],
        "failure_rules": {
            "unknown_sent_usage": "Persist NULL tokens/UNKNOWN_NOT_RETURNED and elapsed latency; "
            "stop example/run/matrix; no automatic retry or same-config relaunch.",
            "timeout": "Client timeout does not prove server cancellation. "
            "Never kill/restart Ollama.",
            "returned_parse_failure": "Persist all returned usage and parse error; halt matrix; "
            "no silent replay/exclusion or complete-comparison claim.",
            "estimated_usage": "Persist ESTIMATED. Exact provider tokens become unknown; "
            "halt after affected run, never present it as clean primary efficiency evidence.",
            "budget_stop": "Retain last valid output/fallback and score it in the full cohort; "
            "retain actual usage and pre/post distinction, without performance-based exclusions.",
            "interruption": "Resume fully checkpointed examples only. Any sent, uncheckpointed "
            "trajectory requires review, not replay. Unknown/in-flight attempt blocks relaunch.",
            "corruption": "Stop on config/source/model/hash/schema/persistence mismatch; "
            "no artifact overwrite.",
            "schema_invalid_output": "Zero schema credit; parse/contract failure follows "
            "returned_parse_failure rule. No gold-dependent replacement or selective rerun.",
        },
        "test_access_policy": [
            "Before freeze, no TEST tuning, training, selection, optimization or manual review.",
            "After freeze, only gated execution of all ten frozen runs; no cherry-picking.",
            "Material defect after TEST starts: stop, document and freeze v2; preserve v1.",
        ],
        "limitations": [
            "DEV pilot n=32; one quantized local model; TEST outcomes are not known.",
            "Temperatures differ across strategy recipes. This is a recipe comparison on the same "
            "model, not a temperature-controlled causal isolation of orchestration alone.",
            "Legacy reused DEV efficiency records have no independent provider-side audit.",
            "No ablations, learned router, second model or matched-call condition in the matrix.",
        ],
        "reproducibility": environment_identity(),
        "source_hashes": code_identity(),
        "matrix": [
            {
                "name": name,
                "config_path": f"configs/{name}.json",
                "config_sha256": stable_config_hash(config),
                "resolved_config_sha256": resolved_hash(config),
            }
            for name, config in configs.items()
        ],
        "runtime_estimate": estimate_test_runtime(analysis, count),
    }
    return protocol, configs


def protocol_markdown(protocol: dict[str, Any], digest: str) -> str:
    lines = [
        f"# {PROTOCOL_VERSION}",
        "",
        f"Protocol SHA-256: `{digest}`",
        "",
        "Frozen before TEST access. Phase 9 performs no inference or TEST reads.",
        "",
    ]
    for key, value in protocol.items():
        if key in {"source_hashes", "primary_model"}:
            continue
        lines.extend(
            [
                f"## {key.replace('_', ' ').title()}",
                "",
                "```json",
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    model = protocol["primary_model"]
    lines.extend(
        [
            "## Primary Model",
            "",
            f"Ollama `{model['model']}`, Q4_K_M, local endpoint.",
            f"Digest: `{model['provider_options']['model_digest']}`.",
            "",
            "Full model identity, generation settings, source/config hashes and dependency "
            "versions are frozen in protocol.json.",
            "",
        ]
    )
    return "\n".join(lines)


def freeze_protocol(directory: Path = PROTOCOL_DIR) -> str:
    if directory.exists():
        raise ValueError("Protocol version already exists and is immutable; do not overwrite it")
    protocol, configs = build_protocol()
    digest = stable_config_hash(protocol)
    directory.mkdir(parents=True)
    artifacts: dict[str, str] = {
        "protocol.json": json.dumps(protocol, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        "protocol.md": protocol_markdown(protocol, digest),
        "test_matrix.json": json.dumps(
            {"protocol_sha256": digest, "entries": protocol["matrix"]}, indent=2
        )
        + "\n",
        "test_matrix.md": "# Frozen TEST Matrix\n\n200 examples/run; same provider/digest. "
        "Natural: 12000 tokens/example. Matched: 4000. Six provider attempts/example, "
        "concurrency 1, no provider retries.\n\n| Run | Raw config SHA-256 |\n|---|---|\n"
        + "".join(f"| {e['name']} | `{e['config_sha256']}` |\n" for e in protocol["matrix"]),
        "runtime_estimate.json": json.dumps(protocol["runtime_estimate"], indent=2) + "\n",
        "integrity_preflight.md": "# Frozen Integrity Preflight\n\nPhase 8.2 integrity PASSED; "
        "five clean reruns, original/blocked hashes unchanged. Benchmark DEV/manifest checked; "
        "TEST hash/composition read from manifest only. No TEST file opened.\n\nBefore future "
        "TEST access the launcher checks the requested protocol hash, all artifact/config/source "
        "hashes, budget policy, nested no-mock roles and live model digest. Execution binds an "
        "immutable start marker; changes require a new protocol version.\n",
    }
    artifacts.update(
        {
            f"configs/{name}.json": json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n"
            for name, config in configs.items()
        }
    )
    for name, content in artifacts.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    manifest = {
        "version": PROTOCOL_VERSION,
        "protocol_sha256": digest,
        "benchmark_identity_sha256": stable_config_hash(protocol["benchmark"]),
        "model_identity_sha256": stable_config_hash(protocol["primary_model"]),
        "contracts_sha256": stable_config_hash(protocol["contracts"]),
        "source_identity_sha256": stable_config_hash(protocol["source_hashes"]),
        "config_hashes": {e["name"]: e["config_sha256"] for e in protocol["matrix"]},
        "artifact_hashes": {name: file_sha256(directory / name) for name in artifacts},
        "test_file_opened": False,
        "test_execution_started": False,
    }
    (directory / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    verify_protocol(directory, expected_sha256=digest)
    return digest


def verify_protocol(
    directory: Path = PROTOCOL_DIR, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    protocol = read_json(directory / "protocol.json")
    manifest = read_json(directory / "protocol_manifest.json")
    digest = stable_config_hash(protocol)
    if digest != manifest["protocol_sha256"] or (expected_sha256 and digest != expected_sha256):
        raise ValueError("Protocol payload/hash mismatch")
    if (
        protocol["version"] != PROTOCOL_VERSION
        or protocol["budget_semantics_version"] != BUDGET_SEMANTICS_VERSION
    ):
        raise ValueError("Protocol version/budget semantics mismatch")
    for name, expected in manifest["artifact_hashes"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    if protocol["source_hashes"] != code_identity():
        raise ValueError("Frozen source changed; create a new protocol version")
    if protocol["reproducibility"]["dependencies"] != environment_identity()["dependencies"]:
        raise ValueError("Frozen dependency versions changed")
    if protocol["reproducibility"]["python_version"] != platform.python_version():
        raise ValueError("Frozen Python version changed")
    configs = {}
    if len(protocol["matrix"]) != 10 or len({e["name"] for e in protocol["matrix"]}) != 10:
        raise ValueError("Frozen matrix must contain ten unique runs")
    for entry in protocol["matrix"]:
        config = read_json(directory / entry["config_path"])
        if (
            stable_config_hash(config) != entry["config_sha256"]
            or resolved_hash(config) != entry["resolved_config_sha256"]
        ):
            raise ValueError("Frozen config changed")
        validate_scientific_config(config, protocol["primary_model"])
        if config["dataset"]["path"] != protocol["benchmark"]["test_path"]:
            raise ValueError("Frozen TEST dataset path changed")
        configs[entry["name"]] = config
    marker = directory / "execution" / "execution_started.json"
    if marker.exists() and read_json(marker)["protocol_sha256"] != digest:
        raise ValueError("Protocol immutable after TEST start")
    freeze = verify_phase8_dev_freeze()
    if (
        not freeze["ok"]
        or freeze["test_file_opened"]
        or protocol["benchmark"]["test_sha256"] != freeze["actual"]["test_sha256_from_manifest"]
    ):
        raise ValueError("Benchmark DEV/manifest identity changed")
    verify_preservation()
    return {
        "status": "PASSED",
        "protocol_sha256": digest,
        "test_file_opened": False,
        "test_execution_started": marker.exists(),
        "configs": configs,
        "protocol": protocol,
    }
