"""Read-only final held-out analysis using the unchanged frozen evaluator/statistics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

from collectiveeval.aggregation import canonical_answer, majority_vote
from collectiveeval.artifact_verify import verify_run_artifacts
from collectiveeval.config import load_config, stable_config_hash
from collectiveeval.core import BenchmarkExample, Candidate
from collectiveeval.datasets import file_sha256, load_jsonl
from collectiveeval.failures import FailureType
from collectiveeval.heldout_continuation import CarriedBaselineStore
from collectiveeval.heldout_protocol import PROTOCOL_DIR, read_json, resolved_hash
from collectiveeval.heldout_v3 import (
    DIRECTORY,
    PARENT,
    accounting,
    verify_carried_rows,
    verify_resume,
    verify_v3,
)
from collectiveeval.metrics import score_prediction
from collectiveeval.phase8_corrective_analysis import completed_scope
from collectiveeval.pilot_analysis import ReadOnlyPilotStore, reconstruct_route
from collectiveeval.reporting import component_coverage, validate_matched_budget_compatibility
from collectiveeval.statistics import paired_bootstrap_comparison, summary_stats

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports/heldout_final"
SHA = "0df6313de34514d713f6bc91f372ab12f4072849299b23ae76c5bdd372babf1a"
RUN_IDS = [
    "5936fb81-ef1f-47ce-834f-7dab65771efc",
    "a3f7ad78-86be-4845-bc54-916bc93a0924",
    "7f4de625-d847-4ec8-ba2a-471af9e11728",
    "774ba57e-0e09-403a-add9-59dd81d9c21c",
    "a15f0ac1-74e8-4e06-bf94-318b5b843dcb",
    "1d48943f-cff4-402b-aea5-e8446becca10",
    "494eacbd-6212-4907-9df5-1e20496c267c",
    "67d46d5e-dda5-4295-9fad-3a042eac625c",
    "44031ea9-0182-4e29-9e15-f58765206fa0",
    "f8545d1e-e7a3-4483-9315-5ee765352c20",
]
STRATEGIES = [
    "single_agent",
    "self_consistency",
    "critic_reviser",
    "multi_agent_debate",
    "adaptive_router",
]
LABELS = {
    "single_agent": "SingleAgent",
    "self_consistency": "SelfConsistency K=2",
    "critic_reviser": "CriticReviser",
    "multi_agent_debate": "Debate (2 agents, 1 revision)",
    "adaptive_router": "HeuristicAdaptiveRouter",
}
METRICS = [
    "task_score",
    "logical_model_calls",
    "provider_attempts",
    "total_tokens",
    "latency_ms",
    "logical_call_latency_ms",
    "wall_clock_strategy_latency_ms",
]
LIMITATIONS = [
    "One quantized local Gemma 3 4B model on a frozen synthetic Japanese enterprise benchmark; "
    "not evidence of performance on private enterprise data or other models.",
    "Recipe temperatures differ, and routing also uses its existing nested debate settings; "
    "orchestration is not causally isolated from sampling.",
    "Shared 12,000/4,000-token allowances are not identical realized spending or guaranteed "
    "hard token caps: provider-reported post-call overruns are retained and disclosed.",
    "Bootstrap intervals are descriptive, unadjusted for multiple subgroup comparisons; "
    "cells below five are too small for reliable interpretation.",
    "Groundedness and failure taxonomy are deterministic reference heuristics, not human "
    "entailment judgments. Convergence labels do not prove causal peer-error propagation.",
    "Router under/over-escalation labels are retrospective heuristics, not counterfactual "
    "proof of the optimal route. Accepted-versus-escalated groups are selected, not randomized.",
    "Protocol v3 was an operational/evaluation amendment after a returned schema failure. "
    "Valid metrics and scientific recipes stayed unchanged; all failures remain in the cohort.",
    "The historical failed trajectory's strategy wall-time is reconstructed from recorded "
    "logical lifecycles; attempt latency is measured. No end-to-end remeasurement occurred.",
    "Conditions ran sequentially on one laptop. Thermal state, model residency, and background "
    "load were not randomized; cross-condition latency differences are not causal budget effects.",
    "Client attempt completeness and pinned identities are audited, not independent server-side "
    "wire completeness. Seeds request reproducibility but do not guarantee independent or "
    "deterministic real-API samples.",
    "Local marginal API pricing is zero; compute, energy, and elapsed time are not universally "
    "free. Quality per dollar is not a meaningful discriminator here.",
    "HeterogeneousPanel, learned routing, other models, and ablations are outside this frozen "
    "held-out experiment. TEST was not used for routing features, training, or post-TEST tuning.",
    "Raw databases, private TEST examples, and local protocol bundles remain withheld. "
    "The public aggregate export is not a full raw-bundle reproduction release.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def evidence_snapshot() -> dict[str, str]:
    roots = [
        PROTOCOL_DIR,
        PARENT,
        DIRECTORY,
        ROOT / "data/benchmark_v3_1",
        ROOT / "reports/pilot_v1",
    ]
    return {
        str(path.relative_to(ROOT)): file_sha256(path)
        for directory in roots
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def validate_cohort(actual: list[str], expected: list[str], identity: str) -> None:
    require(
        len(actual) == len(expected)
        and len(set(actual)) == len(actual)
        and set(actual) == set(expected),
        f"Missing/duplicate/extra cohort: {identity}",
    )


def finite_equal(actual: float, expected: float, identity: str) -> None:
    require(
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
        f"Accounting/artifact mismatch: {identity}",
    )


def artifact_rows(store: ReadOnlyPilotStore, run_id: str) -> list[dict[str, Any]]:
    run = store.get_run(run_id)
    require(run is not None, "Missing canonical run")
    assert run is not None
    path = Path(run["artifact_dir"])
    require(verify_run_artifacts(store, run_id)["ok"], f"Artifact verification failed: {run_id}")
    raw = [
        json.loads(line)
        for line in (path / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    db_rows = store.get_run_predictions(run_id)
    validate_cohort([r["example_id"] for r in raw], [r["example_id"] for r in db_rows], run_id)
    by_id = {r["example_id"]: r for r in raw}
    for row in db_rows:
        item = by_id[row["example_id"]]
        require(item["output"] == json.loads(row["output_json"]), "Raw output/DB mismatch")
        metadata = json.loads(row["metadata_json"])
        require(
            {k: v for k, v in metadata.items() if k != "candidates"} == item["metadata"],
            "Prediction metadata/DB mismatch",
        )
        if "candidates" in metadata:
            require(item["candidates"] == metadata["candidates"], "Candidate artifact mismatch")
        usage = json.loads(row["usage_json"])
        for name, value in usage.items():
            source = item[name] if name in item else item["metadata"][name]
            finite_equal(float(source), float(value), f"{run_id}/{name}")
    metric_artifact = json.loads((path / "metrics.json").read_text())
    metric_rows = metric_artifact["examples"]
    validate_cohort(
        [r["example_id"] for r in metric_rows],
        [r["example_id"] for r in db_rows],
        "metric artifact",
    )
    artifact_metrics = {
        r["example_id"]: {k: v for k, v in r.items() if k not in {"example_id", "strategy"}}
        for r in metric_rows
    }
    stored: dict[str, dict[str, Any]] = defaultdict(dict)
    aggregate: dict[str, float] = {}
    for row in store.get_run_metrics(run_id):
        if row["example_id"] is None:
            aggregate[row["metric_name"]] = row["metric_value"]
        else:
            require(row["metric_name"] not in stored[row["example_id"]], "Duplicate metric")
            stored[row["example_id"]][row["metric_name"]] = row["metric_value"]
    require(dict(stored) == artifact_metrics, "Per-example metric artifacts changed")
    require(aggregate == metric_artifact["aggregate"], "Aggregate metric artifacts changed")
    return raw


def run_account(
    store: ReadOnlyPilotStore, run_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    calls, logical = store.get_run_model_calls(run_id), store.get_run_logical_calls(run_id)
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    logical_by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        by_example[call["example_id"]].append(call)
        metadata = json.loads(call["metadata_json"])
        require(
            call["input_tokens"] == metadata["actual_input_tokens"]
            and call["output_tokens"] == metadata["actual_output_tokens"],
            "Returned usage/attempt mismatch",
        )
        require(metadata["base_url_class"] == "local", "Unexpected provider endpoint class")
    for item in logical:
        logical_by_example[item["example_id"]].append(item)
    require(set(by_example) <= {r["example_id"] for r in rows}, "Uncheckpointed attempt")
    for row in rows:
        current = by_example[row["example_id"]]
        require(
            row["provider_attempts"] == len(current)
            and row["logical_model_calls"] == len(logical_by_example[row["example_id"]]),
            "Prediction call counts mismatch",
        )
        for key in ("input_tokens", "output_tokens", "latency_ms", "estimated_cost_usd"):
            finite_equal(float(row[key]), sum(float(c[key]) for c in current), key)
        require(
            row["total_tokens"] == row["input_tokens"] + row["output_tokens"],
            "Prediction token total mismatch",
        )
    events = [e for r in rows for e in r["metadata"].get("budget_events", [])]
    return {
        "predictions": len(rows),
        "logical_model_calls": len(logical),
        "provider_attempts": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "total_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in calls),
        "failed_attempts": sum(
            c["outcome"] not in {"SUCCESS", "PARSE_ERROR_AFTER_RESPONSE", "STARTED"} for c in calls
        ),
        "output_failure_attempts": sum(c["outcome"] == "PARSE_ERROR_AFTER_RESPONSE" for c in calls),
        "retries": sum(c["attempt_index"] > 0 for c in calls),
        "unknown_usage_attempts": sum(c["usage_status"] == "UNKNOWN_NOT_RETURNED" for c in calls),
        "parse_failures": sum(c["outcome"] == "PARSE_ERROR_AFTER_RESPONSE" for c in calls),
        "pre_call_budget_rejections": sum(
            e["event_type"] == "PRE_CALL_BUDGET_REJECTION" for e in events
        ),
        "post_call_overruns": sum(
            bool(json.loads(c["metadata_json"]).get("post_call_budget_overrun")) for c in calls
        ),
        "attempt_latency_ms": sum(c["latency_ms"] for c in calls),
        "logical_lifecycle_latency_ms": sum(c["lifecycle_latency_ms"] for c in logical),
        "strategy_wall_clock_latency_ms": sum(
            r["metadata"]["wall_clock_strategy_latency_ms"] for r in rows
        ),
        "strategy_wall_clock_reconstructed_examples": sum(
            bool(r["metadata"].get("historical_failure_checkpoint")) for r in rows
        ),
        "marginal_api_cost_usd": sum(c["estimated_cost_usd"] for c in calls),
    }


def canonical_integrity(
    snapshot: dict[str, str],
) -> tuple[dict[str, Any], CarriedBaselineStore, dict[str, Any], list[BenchmarkExample]]:
    checked = verify_v3(DIRECTORY, SHA)
    protocol = checked["protocol"]
    progress = read_json(DIRECTORY / "execution/progress.json")
    require(
        progress["status"] == "COMPLETED"
        and progress["protocol_sha256"] == SHA
        and progress["run_ids"] == RUN_IDS,
        "Launcher canonical scope mismatch",
    )
    require(
        not (DIRECTORY / "execution/halted.json").exists()
        and not (DIRECTORY / "execution/execution.lock").exists(),
        "Halted/in-flight execution",
    )
    db = DIRECTORY / "execution/heldout.sqlite3"
    verify_resume(db, protocol, DIRECTORY)
    verify_carried_rows(DIRECTORY / "continuation_seed.sqlite3", db)
    store = CarriedBaselineStore(db, PROTOCOL_DIR / "execution/heldout.sqlite3", RUN_IDS[0])
    for version, expected_runs in (
        (PROTOCOL_DIR, RUN_IDS[:1]),
        (PARENT, RUN_IDS[1:4]),
        (DIRECTORY, RUN_IDS[1:]),
    ):
        raw_store = ReadOnlyPilotStore(version / "execution/heldout.sqlite3")
        require(
            {r["id"] for r in raw_store._fetch_all("SELECT id FROM runs")} == set(expected_runs),
            "Diagnostic/unapproved run in canonical execution DB",
        )
        with closing(raw_store.connect()) as connection:
            require(
                connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok",
                "Corrupt evidence DB",
            )
            require(
                connection.execute(
                    "SELECT COUNT(*) FROM provider_attempts a LEFT JOIN runs r ON a.run_id=r.id "
                    "WHERE r.id IS NULL"
                ).fetchone()[0]
                == 0,
                "Orphan attempt",
            )
    examples = load_jsonl(protocol["benchmark"]["test_path"])
    ids = [e.id for e in examples]
    require(
        len(ids) == len(set(ids)) == 200 and all(e.metadata["split"] == "test" for e in examples),
        "Invalid frozen TEST cohort",
    )
    require(
        file_sha256(protocol["benchmark"]["test_path"]) == protocol["benchmark"]["test_sha256"],
        "TEST hash mismatch",
    )
    baseline_cert = read_json(PARENT / "baseline_reuse_certificate.json")
    reports: list[dict[str, Any]] = []
    attempt_ids: list[str] = []
    for index, (entry, run_id) in enumerate(zip(protocol["matrix"], RUN_IDS, strict=True)):
        config = load_config(DIRECTORY / entry["config_path"])
        run, experiment = store.get_run(run_id), store.get_run_experiment(run_id)
        require(run is not None and experiment is not None, "Missing canonical identity")
        assert run is not None and experiment is not None
        require(
            run["status"] == "COMPLETED" and run["strategy"] == config["strategy"]["type"],
            "Canonical strategy/status mismatch",
        )
        require(
            experiment["dataset_hash"] == protocol["benchmark"]["test_sha256"],
            "Canonical dataset mismatch",
        )
        if index == 0:
            require(
                experiment["config_hash"] == baseline_cert["persisted_legacy_config_sha256"]
                and baseline_cert["classification"] == "REUSE_PROVEN_EQUIVALENT"
                and baseline_cert["canonical_config_sha256"] == entry["resolved_config_sha256"]
                and stable_config_hash(json.loads(experiment["config_json"]))
                == experiment["config_hash"],
                "Baseline reuse identity failed",
            )
        else:
            require(
                experiment["config_hash"] == entry["resolved_config_sha256"]
                and stable_config_hash(json.loads(experiment["config_json"]))
                == resolved_hash(config),
                "Canonical config identity failed",
            )
        predictions = artifact_rows(store, run_id)
        validate_cohort([r["example_id"] for r in predictions], ids, run_id)
        require(
            all(r["strategy"] == run["strategy"] for r in predictions),
            "Prediction strategy mismatch",
        )
        validate_cohort(
            list(store.example_metric_values(run_id, "task_score")), ids, "Primary denominator"
        )
        require(
            all(
                math.isfinite(v) for v in store.example_metric_values(run_id, "task_score").values()
            ),
            "Nonfinite primary score",
        )
        counts = accounting(store, run_id, config["model"])
        account = run_account(store, run_id, predictions)
        require(
            counts["attempts"] == account["provider_attempts"]
            and account["unknown_usage_attempts"] == account["retries"] == 0,
            "Frozen consumption policy violated",
        )
        attempt_ids.extend(c["attempt_id"] for c in store.get_run_model_calls(run_id))
        source = (
            "v1_completed_reused"
            if index == 0
            else "v2_completed_carried"
            if index in {1, 2}
            else "v2_31_complete_plus_1_failure_and_v3_168_new"
            if index == 3
            else "v3_new"
        )
        reports.append(
            {
                "run_id": run_id,
                "strategy": run["strategy"],
                "condition": config["phase9"]["condition"],
                "name": entry["name"],
                "provenance": source,
                "frozen_config_sha256": entry["resolved_config_sha256"],
                "exact_test_cohort": True,
                "accounting": account,
            }
        )
    require(
        len(set(attempt_ids)) == len(attempt_ids), "Duplicate provider request in canonical cohort"
    )
    compatibility = {}
    for condition in protocol["conditions"]:
        selected = [r["run_id"] for r in reports if r["condition"] == condition]
        compatibility[condition] = validate_matched_budget_compatibility(store, selected)
        require(compatibility[condition]["ok"], "Condition comparison incompatibility")
    require(snapshot == evidence_snapshot(), "Scientific evidence modified during integrity audit")
    report = {
        "status": "PASSED",
        "completed_runs": 10,
        "completed_predictions": 2000,
        "protocol_sha256": SHA,
        "model": {k: protocol["primary_model"][k] for k in ("provider", "model")},
        "model_digest": protocol["primary_model"]["provider_options"]["model_digest"],
        "benchmark": protocol["benchmark"],
        "runs": reports,
        "compatibility": compatibility,
        "totals": {
            name: sum(r["accounting"][name] for r in reports) for name in reports[0]["accounting"]
        },
        "continuation_certificate": checked["certificate"],
        "baseline_reuse_certificate_sha256": file_sha256(
            PARENT / "baseline_reuse_certificate.json"
        ),
        "raw_evidence_hashes": snapshot,
        "raw_evidence_unchanged": True,
        "scope": "Read-only client evidence audit; no independent server wire capture.",
        "no_model_calls": True,
    }
    return report, store, protocol, examples


def integrity_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final Held-Out Integrity",
        "",
        "Status: **PASSED**. No new model inference.",
        "",
        f"Canonical runs: {report['completed_runs']}/10; "
        f"predictions: {report['completed_predictions']}/2,000.",
        "Every recipe has exactly 200 unique IDs from the unchanged frozen TEST cohort.",
        "",
        "| Condition | Strategy | n | Logical calls | Attempts | Tokens | Provenance |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for run in report["runs"]:
        a = run["accounting"]
        lines.append(
            f"| {run['condition']} | {LABELS[run['strategy']]} | {a['predictions']} | "
            f"{a['logical_model_calls']} | {a['provider_attempts']} | {a['total_tokens']} | "
            f"{run['provenance']} |"
        )
    lines += [
        "",
        "## Totals",
        "",
        *[f"- {k}: {v}" for k, v in report["totals"].items()],
        "",
        "## Identity And Preservation",
        "",
        f"Protocol: `{SHA}`.",
        f"Model: ollama / gemma3:4b; digest `{report['model_digest']}`.",
        f"TEST SHA-256: `{report['benchmark']['test_sha256']}`.",
        "v1/v2 files and certificates match their frozen preservation manifests. V3 source, "
        "dependency, configuration, artifact and seed checks pass. Carried predictions, metrics "
        "and attempt rows match the immutable seed without replay.",
        "Natural Debate carries 31 completed v2 checkpoints and one preserved returned failure, "
        "then adds 168 new v3 examples: exactly 200 total.",
        "No mock/diagnostic rows, duplicate requests, missing cohorts, retries, "
        "unknown consumption, "
        "unaccounted attempts, or uncheckpointed sent work enter the canonical view.",
        "Eight returned schema failures remain in their denominators under the frozen v3 contract. "
        "The historical failure's strategy wall-time is reconstructed, not remeasured; other "
        "latency totals retain recorded durations.",
        "Frozen failed_attempts counts transport failures, not returned parse failures. The latter "
        "are separate output_failure_attempts/parse_failures; zero transport failures does not "
        "imply zero invalid outputs.",
        "This is a client-side audit, not independent provider wire capture. Full raw file hashes "
        "are in integrity_report.json; databases are opened read-only.",
        "",
    ]
    return "\n".join(lines)


def observations(
    store: ReadOnlyPilotStore, run_id: str, examples: dict[str, BenchmarkExample]
) -> list[dict[str, Any]]:
    calls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in store.get_run_model_calls(run_id):
        calls[call["example_id"]].append({**call, "metadata": json.loads(call["metadata_json"])})
    metric_maps = {name: store.example_metric_values(run_id, name) for name in METRICS}
    failure_maps: dict[str, list[str]] = defaultdict(list)
    for failure in store.get_run_failures(run_id):
        failure_maps[failure["example_id"]].append(failure["failure_type"])
    artifacts = artifact_rows(store, run_id)
    rows = []
    for raw in sorted(artifacts, key=lambda row: row["example_id"]):
        eid = raw["example_id"]
        example = examples[eid]
        rows.append(
            {
                **raw,
                **{name: values[eid] for name, values in metric_maps.items()},
                "calls": calls[eid],
                "example": example,
                "task": str(example.task_type),
                "difficulty": example.metadata["difficulty"],
                "reasoning_family": example.metadata.get("reasoning_family", "unrecorded"),
                "failure_types": failure_maps[eid],
            }
        )
    return rows


def frozen_summary(values: list[float], settings: dict[str, Any]) -> dict[str, Any]:
    return summary_stats(
        values,
        n_bootstrap=settings["bootstrap_replicates"],
        confidence=settings["confidence"],
        seed=settings["seed"],
    ).to_dict()


def frozen_pair(
    baseline: dict[str, float], contender: dict[str, float], metric: str, settings: dict[str, Any]
) -> dict[str, Any]:
    require(
        set(baseline) == set(contender) and bool(baseline),
        "Unpaired comparison would exclude examples",
    )
    return paired_bootstrap_comparison(
        baseline=baseline,
        contender=contender,
        metric=metric,
        n_bootstrap=settings["bootstrap_replicates"],
        confidence=settings["confidence"],
        seed=settings["seed"],
    ).to_dict()


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        **{
            name: sum(float(r[name]) for r in rows) / len(rows) if rows else None
            for name in (
                "task_score",
                "logical_model_calls",
                "provider_attempts",
                "total_tokens",
                "latency_ms",
            )
        },
    }


def successful_outputs(row: dict[str, Any], roles: set[str] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "output": call["metadata"]["structured_output"],
            "role": call["role"],
            "round": call["metadata"]["round_index"],
            "call": call,
        }
        for call in row["calls"]
        if call["outcome"] == "SUCCESS"
        and (roles is None or call["role"] in roles)
        and isinstance(call["metadata"].get("structured_output"), dict)
    ]


def initial_score(row: dict[str, Any], role: str) -> float | None:
    values = successful_outputs(row, {role})
    return score_prediction(row["example"], values[0]["output"])["task_score"] if values else None


def mechanism(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "example_id": row["example_id"],
            "task": row["task"],
            "final_score": row["task_score"],
            "failed_output": row.get("kind") == "failed_prediction",
        }
        if strategy == "critic_reviser":
            initial = initial_score(row, "generator")
            revised = bool(successful_outputs(row, {"reviser", "repair"}))
            item.update(
                initial_score=initial,
                revised=revised,
                explicit_no_revision=row["metadata"].get("revised") is False,
                change_from_initial=None if initial is None else row["task_score"] - initial,
            )
        elif strategy == "self_consistency":
            candidates = row["candidates"]
            outputs = [c["output"] for c in candidates]
            counts = Counter(canonical_answer(output) for output in outputs)
            first = score_prediction(row["example"], outputs[0])["task_score"] if outputs else None
            item.update(
                candidate_count=len(outputs),
                full_output_diversity=len({json.dumps(o, sort_keys=True) for o in outputs}) > 1,
                answer_diversity=len(counts) > 1,
                tie=bool(counts) and max(counts.values()) <= len(outputs) / 2,
                initial_score=first,
                change_from_initial=None if first is None else row["task_score"] - first,
            )
            if candidates:
                selected = majority_vote([Candidate.model_validate(c) for c in candidates])
                require(
                    selected.output == row["output"],
                    "Stored SelfConsistency vote does not match frozen aggregation",
                )
                item["selected_sample_index"] = next(
                    i for i, c in enumerate(candidates) if c["output"] == selected.output
                )
        elif strategy == "multi_agent_debate":
            outputs = successful_outputs(row, {"agent_0", "agent_1"})
            by_round = [[o for o in outputs if o["round"] == index] for index in (0, 1)]
            candidates = [c for c in row["candidates"] if c.get("metadata", {}).get("round") == 0]
            first_output = (
                majority_vote([Candidate.model_validate(c) for c in candidates]).output
                if len(candidates) == 2
                else None
            )
            first_score = (
                score_prediction(row["example"], first_output)["task_score"]
                if first_output is not None
                else None
            )
            item.update(
                completed_rounds=sum(len(group) == 2 for group in by_round),
                attempted_rounds=len({c["metadata"]["round_index"] for c in row["calls"]}),
                candidate_count=len(outputs),
                initial_score=first_score,
                initial_disagreement=len({canonical_answer(o["output"]) for o in by_round[0]}) > 1,
                revision_disagreement=len({canonical_answer(o["output"]) for o in by_round[1]}) > 1,
                selected_prediction_changed=None
                if first_output is None
                else first_output != row["output"],
                change_from_initial=None
                if first_score is None
                else row["task_score"] - first_score,
            )
        elif strategy == "adaptive_router":
            route = reconstruct_route(row["metadata"], row["calls"])
            first = initial_score(row, "cheap_single_agent")
            # If the initial call itself failed, no routing decision was made.
            if first is None:
                route.update(route="not_reached", route_source="initial_output_failure")
            item.update(
                **route,
                initial_score=first,
                change_from_initial=None if first is None else row["task_score"] - first,
            )
        details.append(item)
    by_id = {r["example_id"]: r for r in rows}
    summary: dict[str, Any] = {
        "n": len(rows),
        "failed_outputs": sum(d["failed_output"] for d in details),
        "budget_stopped_examples": sum(
            bool(
                r["metadata"].get("budget_exhaustion_reason")
                or r["metadata"].get("stopped") == "budget"
                or any(
                    e["event_type"] == "PRE_CALL_BUDGET_REJECTION"
                    for e in r["metadata"].get("budget_events", [])
                )
            )
            for r in rows
        ),
        "details": details,
    }
    changes = [
        d["change_from_initial"] for d in details if d.get("change_from_initial") is not None
    ]
    summary.update(
        initial_comparison_coverage=len(changes),
        improved=sum(x > 1e-12 for x in changes),
        regressed=sum(x < -1e-12 for x in changes),
        unchanged=sum(abs(x) <= 1e-12 for x in changes),
    )
    if strategy == "critic_reviser":
        summary.update(
            revision_count=sum(d["revised"] for d in details),
            revision_rate=sum(d["revised"] for d in details) / len(rows),
            explicit_no_revision_count=sum(d["explicit_no_revision"] for d in details),
            no_revision_rate=sum(d["explicit_no_revision"] for d in details) / len(rows),
            score_revised=group_summary([by_id[d["example_id"]] for d in details if d["revised"]]),
            score_unrevised=group_summary(
                [by_id[d["example_id"]] for d in details if not d["revised"]]
            ),
        )
    if strategy == "self_consistency":
        summary.update(
            candidate_count_distribution=dict(Counter(d["candidate_count"] for d in details)),
            full_output_diverse=sum(d["full_output_diversity"] for d in details),
            answer_diverse=sum(d["answer_diversity"] for d in details),
            diversity_rate=sum(d["answer_diversity"] for d in details) / len(rows),
            ties=sum(d["tie"] for d in details),
            selected_sample_distribution=dict(
                Counter(d.get("selected_sample_index", "unavailable") for d in details)
            ),
            interpretation="Canonical answer diversity differs from full-object diversity. K=2 "
            "disagreements are ties, resolved by frozen confidence then earliest order; requested "
            "seeds do not guarantee independent draws.",
        )
    if strategy == "multi_agent_debate":
        summary.update(
            completed_rounds_distribution=dict(Counter(d["completed_rounds"] for d in details)),
            full_two_round_trajectories=sum(d["completed_rounds"] == 2 for d in details),
            initial_disagreements=sum(d["initial_disagreement"] for d in details),
            revision_disagreements=sum(d["revision_disagreement"] for d in details),
            changed_selected_prediction=sum(
                d["selected_prediction_changed"] is True for d in details
            ),
            round_comparison_coverage=sum(
                d["selected_prediction_changed"] is not None for d in details
            ),
            interpretation="Completed round means two successful agent responses. Initial "
            "selection requires stored round-zero candidates; unavailable candidate histories "
            "are not fabricated. Convergence is not causal peer propagation.",
        )
    if strategy == "adaptive_router":
        routes = Counter(d["route"] for d in details)
        summary.update(
            route_counts=dict(routes),
            accepted=routes["accept"],
            escalated=routes["critic"] + routes["debate"],
            accepted_vs_escalated={
                name: group_summary(
                    [by_id[d["example_id"]] for d in details if d["route"] in route_set]
                )
                for name, route_set in (
                    ("accepted", {"accept"}),
                    ("escalated", {"critic", "debate"}),
                )
            },
            retrospective_nonimproving_escalations=sum(
                d["route"] in {"critic", "debate"}
                and d["change_from_initial"] is not None
                and d["change_from_initial"] <= 1e-12
                for d in details
            ),
            accepted_imperfect=sum(
                d["route"] == "accept" and d["final_score"] < 1 for d in details
            ),
            route_reconstructed=sum(
                d["route_source"] == "derived_from_call_trajectory" for d in details
            ),
            actual_calls_distribution=dict(Counter(r["provider_attempts"] for r in rows)),
            interpretation="Retrospective diagnostics use frozen gold-based scores offline only. "
            "They are not inference features or optimal-route counterfactuals; route subgroups "
            "are not randomized.",
        )
    return summary


def pareto_dominance(strategies: dict[str, Any], resource: str) -> dict[str, Any]:
    result = {}
    for name, own in strategies.items():
        q, cost = own["metrics"]["task_score"]["mean"], own["metrics"][resource]["mean"]
        dominated = [
            other
            for other, data in strategies.items()
            if other != name
            and data["metrics"]["task_score"]["mean"] >= q
            and data["metrics"][resource]["mean"] <= cost
            and (
                data["metrics"]["task_score"]["mean"] > q
                or data["metrics"][resource]["mean"] < cost
            )
        ]
        result[name] = {"dominated_by": dominated, "nondominated": not dominated}
    return result


def dev_comparison(result: dict[str, Any]) -> dict[str, Any]:
    export = read_json(ROOT / "docs/phase8_dev_results.json")
    _, _, entries = completed_scope(corrective=ROOT / "reports/pilot_v1/phase8_2_attempts_v2")
    raw = read_json(ROOT / "reports/pilot_v1/phase8_2_attempts_v2/phase8_analysis.json")
    require(
        export["source_analysis_sha256"]
        == file_sha256(ROOT / "reports/pilot_v1/phase8_2_attempts_v2/phase8_analysis.json"),
        "DEV export provenance changed",
    )
    require(len(entries) == 10 and export["pilot_examples"] == 32, "Corrected DEV scope mismatch")
    comparisons: dict[str, Any] = {}
    for condition in result["strategy_results"]:
        comparisons[condition] = {}
        for strategy in STRATEGIES:
            dev = export["paired_deltas"][condition][strategy]["task_score"]
            test = result["paired_deltas"][condition][strategy]["task_score"]
            require(
                dev == raw["paired_deltas"][condition][strategy]["task_score"],
                "DEV public/ raw paired evidence mismatch",
            )
            dv, tv = dev["mean_difference"], test["mean_difference"]
            direction = (
                "reference"
                if strategy == "single_agent"
                else "reversed"
                if dv * tv < 0
                else "weakened"
                if abs(tv) < abs(dv)
                else "same_direction"
            )
            family = {}
            for task, data in result["strategy_results"][condition][strategy]["breakdown"][
                "task"
            ].items():
                dev_s = export["strategy_results"][condition][strategy]["breakdown"]["task"][task]
                dev_b = export["strategy_results"][condition]["single_agent"]["breakdown"]["task"][
                    task
                ]
                family[task] = {
                    "dev_n": dev_s["n"],
                    "test_n": data["n"],
                    "dev_delta": dev_s["mean"] - dev_b["mean"],
                    "test_delta": data["paired_vs_single_agent"]["mean_difference"],
                }
            comparisons[condition][strategy] = {
                "dev": dev,
                "test": test,
                "direction": direction,
                "dev_interval_includes_zero": dev["ci_lower"] <= 0 <= dev["ci_upper"],
                "task_family_deltas": family,
            }
    return {
        "dev_n": 32,
        "test_n": 200,
        "comparisons": comparisons,
        "dev_export_sha256": file_sha256(ROOT / "docs/phase8_dev_results.json"),
        "interpretation": "DEV and TEST are disjoint cohorts, not paired to each other. "
        "Directional labels describe point estimates, not significance, and unchanged "
        "directions can still be uncertain.",
    }


def build_analysis(
    store: CarriedBaselineStore,
    protocol: dict[str, Any],
    examples: list[BenchmarkExample],
    integrity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    by_example = {e.id: e for e in examples}
    settings = protocol["statistics"]
    result: dict[str, Any] = {
        "artifact_kind": "HELDOUT_FINAL_READ_ONLY_ANALYSIS",
        "protocol_sha256": SHA,
        "statistics": settings,
        "benchmark_version": protocol["benchmark"]["version"],
        "test_sha256": protocol["benchmark"]["test_sha256"],
        "dev_sha256": protocol["benchmark"]["dev_sha256"],
        "test_n": len(examples),
        "task_counts": dict(Counter(str(e.task_type) for e in examples)),
        "difficulty_counts": dict(Counter(e.metadata["difficulty"] for e in examples)),
        "strategy_results": {},
        "paired_deltas": {},
        "condition_comparison": {},
        "pareto": {},
        "model": {
            "provider": "ollama",
            "model": "gemma3:4b",
            "digest": integrity["model_digest"],
            "quantization": "Q4_K_M",
            "base_url_class": "local",
        },
        "conditions": protocol["conditions"],
        "accounting_totals": integrity["totals"],
        "provenance": [
            {k: v for k, v in r.items() if k != "accounting"} for r in integrity["runs"]
        ],
        "raw_evidence_inventory_sha256": stable_config_hash(integrity["raw_evidence_hashes"]),
        "analysis_source_hashes": {
            str(p.relative_to(ROOT)): file_sha256(p)
            for p in sorted((ROOT / "analysis").glob("*.py"))
        },
        "failure_contract": protocol["failed_output_contract"],
        "limitations": LIMITATIONS,
        "no_model_calls": True,
    }
    all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for condition in protocol["conditions"]:
        (
            result["strategy_results"][condition],
            result["paired_deltas"][condition],
            all_rows[condition],
        ) = {}, {}, {}
        identities = [r for r in integrity["runs"] if r["condition"] == condition]
        for identity in identities:
            all_rows[condition][identity["strategy"]] = observations(
                store, identity["run_id"], by_example
            )
        baseline = all_rows[condition]["single_agent"]
        for strategy in STRATEGIES:
            rows = all_rows[condition][strategy]
            run_id = next(r["run_id"] for r in identities if r["strategy"] == strategy)
            metrics = {key: frozen_summary([r[key] for r in rows], settings) for key in METRICS}
            breakdown: dict[str, Any] = {}
            for dimension in ("task", "difficulty", "reasoning_family"):
                breakdown[dimension] = {}
                for group in sorted({str(r[dimension]) for r in rows}):
                    chosen = [r for r in rows if str(r[dimension]) == group]
                    ids = {r["example_id"] for r in chosen}
                    paired = frozen_pair(
                        {
                            r["example_id"]: r["task_score"]
                            for r in baseline
                            if r["example_id"] in ids
                        },
                        {r["example_id"]: r["task_score"] for r in chosen},
                        "task_score",
                        settings,
                    )
                    breakdown[dimension][group] = {
                        **frozen_summary([r["task_score"] for r in chosen], settings),
                        "tiny_cell": len(chosen) < settings["tiny_cell_threshold"],
                        "paired_vs_single_agent": paired,
                    }
            failures = Counter(label for r in rows for label in r["failure_types"])
            call_rows = [c for r in rows for c in r["calls"]]
            ceilings = protocol["conditions"][condition]["max_total_tokens"]
            max_overrun = max(0, max(r["total_tokens"] - ceilings for r in rows))
            result["strategy_results"][condition][strategy] = {
                "run_id": run_id,
                "metrics": metrics,
                "breakdown": breakdown,
                "quality_per_1k_tokens": 1000
                * metrics["task_score"]["mean"]
                / metrics["total_tokens"]["mean"],
                "quality_per_1k_definition": "Ratio of mean frozen score to mean realized tokens, "
                "not the mean of per-example ratios.",
                "mean_example_quality_per_1k_tokens": sum(
                    store.example_metric_values(run_id, "quality_per_1k_tokens").values()
                )
                / len(rows),
                "marginal_api_cost_usd": 0,
                "quality_per_dollar": None,
                "failure_counts": {str(label): failures[str(label)] for label in FailureType},
                "failure_by_task": {
                    task: dict(
                        Counter(
                            label for r in rows if r["task"] == task for label in r["failure_types"]
                        )
                    )
                    for task in result["task_counts"]
                },
                "component_coverage": component_coverage(store, run_id),
                "accounting": next(
                    r["accounting"] for r in identities if r["strategy"] == strategy
                ),
                "token_budget": {
                    "ceiling": ceilings,
                    "examples_over_ceiling": sum(r["total_tokens"] > ceilings for r in rows),
                    "max_actual_tokens": max(r["total_tokens"] for r in rows),
                    "max_overrun_tokens": max_overrun,
                    "output_caps_below_700": sum(c["max_tokens"] < 700 for c in call_rows),
                },
                "returned_output_failures": {
                    "by_role": dict(
                        Counter(
                            c["role"]
                            for c in call_rows
                            if c["outcome"] == "PARSE_ERROR_AFTER_RESPONSE"
                        )
                    ),
                    "by_task": dict(
                        Counter(r["task"] for r in rows if r.get("kind") == "failed_prediction")
                    ),
                    "length_terminated": sum(
                        c["outcome"] == "PARSE_ERROR_AFTER_RESPONSE"
                        and c["finish_reason"] == "length"
                        for c in call_rows
                    ),
                    "failed_output_caps": sorted(
                        c["max_tokens"]
                        for c in call_rows
                        if c["outcome"] == "PARSE_ERROR_AFTER_RESPONSE"
                    ),
                },
                "mechanism": mechanism(rows, strategy),
            }
            result["paired_deltas"][condition][strategy] = {
                metric: frozen_pair(
                    {r["example_id"]: r[metric] for r in baseline},
                    {r["example_id"]: r[metric] for r in rows},
                    metric,
                    settings,
                )
                for metric in METRICS
            }
        result["pareto"][condition] = {
            resource: pareto_dominance(result["strategy_results"][condition], resource)
            for resource in (
                "total_tokens",
                "provider_attempts",
                "latency_ms",
                "wall_clock_strategy_latency_ms",
            )
        }
    for strategy in STRATEGIES:
        natural, matched = all_rows["natural"][strategy], all_rows["matched_tokens"][strategy]
        natural_by_id = {r["example_id"]: r for r in natural}
        decomposition = {}
        for label, failed in (("failed_matched_output", True), ("valid_matched_output", False)):
            deltas = [
                r["task_score"] - natural_by_id[r["example_id"]]["task_score"]
                for r in matched
                if (r.get("kind") == "failed_prediction") == failed
            ]
            decomposition[label] = {
                "n": len(deltas),
                "changed_scores": sum(abs(v) > 1e-12 for v in deltas),
                "mean_delta_contribution_over_full_cohort": sum(deltas) / len(matched),
            }
        result["condition_comparison"][strategy] = {
            "direction": "matched_tokens_minus_natural",
            "quality_change_decomposition": decomposition,
            "paired": {
                metric: frozen_pair(
                    {r["example_id"]: r[metric] for r in natural},
                    {r["example_id"]: r[metric] for r in matched},
                    metric,
                    settings,
                )
                for metric in METRICS
            },
            "budget_rejection_delta": result["strategy_results"]["matched_tokens"][strategy][
                "accounting"
            ]["pre_call_budget_rejections"]
            - result["strategy_results"]["natural"][strategy]["accounting"][
                "pre_call_budget_rejections"
            ],
            "post_call_overrun_delta": result["strategy_results"]["matched_tokens"][strategy][
                "accounting"
            ]["post_call_overruns"]
            - result["strategy_results"]["natural"][strategy]["accounting"]["post_call_overruns"],
        }
    frozen_reports = read_json(DIRECTORY / "execution/paired_reports.json")
    for condition in result["strategy_results"]:
        for report in frozen_reports[condition]["task_score"]["runs"]:
            actual = result["strategy_results"][condition][report["run"]["strategy"]]["metrics"][
                "task_score"
            ]
            require(
                actual == report["summary"], "Primary summary differs from frozen launcher report"
            )
        for report in frozen_reports[condition]["task_score"]["comparisons"]:
            strategy = next(
                s
                for s, v in result["strategy_results"][condition].items()
                if v["run_id"] == report["contender_run_id"]
            )
            actual = result["paired_deltas"][condition][strategy]["task_score"]
            require(
                actual
                == {
                    k: v
                    for k, v in report.items()
                    if k not in {"baseline_run_id", "contender_run_id"}
                },
                "Paired interval differs from frozen report",
            )
    result["dev_vs_test"] = dev_comparison(result)
    return result, all_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline final held-out analysis; no provider calls"
    )
    parser.add_argument("--integrity-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="Reproduce and compare; do not write")
    args = parser.parse_args()
    before = evidence_snapshot()
    report, store, protocol, examples = canonical_integrity(before)
    outputs = {
        FINAL / "integrity_report.json": json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        FINAL / "integrity_report.md": integrity_markdown(report),
    }
    if not args.integrity_only:
        from analysis.heldout_render import render_outputs

        result, rows = build_analysis(store, protocol, examples, report)
        outputs.update(render_outputs(result, rows))
    require(evidence_snapshot() == before, "Evidence changed during read-only analysis")
    for path, content in outputs.items():
        if args.check:
            require(
                path.exists() and path.read_text(encoding="utf-8") == content,
                f"Analysis reproducibility mismatch: {path.relative_to(ROOT)}",
            )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "integrity": report["status"],
                "totals": report["totals"],
                "no_model_calls": True,
                "analysis_files": len(outputs),
                "reproducibility_checked": args.check,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
