"""Offline Phase 8 analysis. Never constructs providers or writes scientific rows."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from collectiveeval.aggregation import majority_vote
from collectiveeval.core import BenchmarkExample, Candidate
from collectiveeval.metrics import score_prediction
from collectiveeval.reporting import validate_matched_budget_compatibility
from collectiveeval.statistics import paired_bootstrap_comparison, summary_stats
from collectiveeval.storage import SQLiteStore


class ReadOnlyPilotStore(SQLiteStore):
    """Reuse query methods without running SQLiteStore's schema initialization."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection


def condition_for_run(store: SQLiteStore, run_id: str) -> str:
    experiment = store.get_run_experiment(run_id)
    if experiment is None:
        raise ValueError(f"Missing experiment: {run_id}")
    return str(json.loads(experiment["config_json"])["phase8"]["condition"])


def reconstruct_route(metadata: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    route = metadata.get("route")
    roles = [str(call["role"]) for call in calls]
    source = "explicit_prediction_metadata"
    if route not in {"accept", "critic", "debate"}:
        source = "derived_from_call_trajectory"
        if any(role.startswith("agent_") for role in roles):
            route = "debate"
        elif any(role in {"generator", "critic", "reviser", "repair"} for role in roles):
            route = "critic"
        else:
            route, source = "unknown", "unresolved"
    return {
        "route": route,
        "route_source": source,
        "roles": roles,
        "original_route": metadata.get("route", "unknown"),
        "accepted": route == "accept",
        "escalated": route in {"critic", "debate"},
        "nested_strategy": {"critic": "critic_reviser", "debate": "multi_agent_debate"}.get(
            str(route)
        ),
        "budget_exhausted": bool(metadata.get("budget_exhausted")),
        "budget_exhaustion_reason": metadata.get("budget_exhaustion_reason"),
    }


def accounting(calls: list[dict[str, Any]], estimate: dict[str, Any]) -> dict[str, Any]:
    actual: dict[str, Any] = {
        "model_calls": len(calls),
        "input_tokens": sum(int(c["input_tokens"]) for c in calls),
        "output_tokens": sum(int(c["output_tokens"]) for c in calls),
        "failures": sum(bool(c["normalized_error"]) for c in calls),
        "retries": sum(int(c["retry_count"]) for c in calls),
        "api_cost_usd": sum(float(c["estimated_cost_usd"]) for c in calls),
        "usage_sources": dict(Counter(c["usage_source"] for c in calls)),
    }
    actual["total_tokens"] = actual["input_tokens"] + actual["output_tokens"]
    return {
        "actual": actual,
        "pre_run_estimate": estimate,
        "call_estimate_type": "worst_case_escalation_not_expected_usage",
        "token_estimate_ratio": actual["total_tokens"] / estimate["estimated_total_tokens"],
    }


def evidence_hashes(directory: Path) -> dict[str, str]:
    paths = [directory / "phase8.sqlite3", *sorted((directory / "runs").rglob("*"))]
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file()
    }


def write_report(path: Path, title: str, payload: Any) -> None:
    path.write_text(
        f"# {title}\n\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )


def mechanism_analysis(rows: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    details = []
    for row in rows:
        candidates = row["candidates"]
        outputs = [c["output"] for c in candidates]
        candidate_scores = [
            score_prediction(row["example"], output)["task_score"] for output in outputs
        ]
        item = {
            "example_id": row["example_id"],
            "condition": row["condition"],
            "strategy": strategy,
            "run_id": row["run_id"],
            "candidate_scores": candidate_scores,
            "candidate_schema_compliance": [
                score_prediction(row["example"], output)["schema_compliance"] for output in outputs
            ],
            "disagreement": len({json.dumps(o, sort_keys=True) for o in outputs}) > 1,
            "final_score": row["task_score"],
        }
        if strategy == "critic_reviser" and candidate_scores:
            item["final_minus_generator"] = row["task_score"] - candidate_scores[0]
        if strategy == "multi_agent_debate":
            initial = [c for c in candidates if c["metadata"].get("round") == 0]
            final = [c for c in candidates if c["metadata"].get("round") == 1]
            if initial and final:
                first = majority_vote([Candidate.model_validate(c) for c in initial]).output
                item["revision_changed_selected_answer"] = first != row["output"]
                item["final_minus_initial_round"] = (
                    row["task_score"] - score_prediction(row["example"], first)["task_score"]
                )
        if strategy == "self_consistency" and candidate_scores:
            item["selected_minus_first_sample"] = row["task_score"] - candidate_scores[0]
        details.append(item)
    return {
        "examples": details,
        "candidate_trajectories_available": sum(bool(r["candidates"]) for r in rows),
        "disagreement_examples": [r["example_id"] for r in details if r["disagreement"]],
        "revised_score": summary_stats(
            [r["task_score"] for r in rows if r["metadata"].get("revised") is True]
        ).to_dict(),
        "not_revised_score": summary_stats(
            [r["task_score"] for r in rows if r["metadata"].get("revised") is False]
        ).to_dict(),
        "unsupported_causal_labels": "No majority-wrong/peer-propagation labels inferred "
        "without existing annotations.",
    }


def readable_analysis(result: dict[str, Any]) -> str:
    lines = [
        "# Phase 8.1 DEV Pilot Analysis",
        "",
        "Offline post-hoc analysis of 32 DEV examples on Ollama Gemma 3 4B (Q4_K_M). "
        "No new inference or learned-router training. Bootstrap seed 20261003; 1,000 resamples. "
        "Intervals are descriptive, with no automatic significance claims.",
        "",
    ]
    for condition, strategies in result["strategy_results"].items():
        lines += [
            f"## {condition}",
            "",
            "| Strategy | Score (95% CI) | Calls mean/median | Tokens mean/median | "
            "Summed call latency/example (s) | Strategy wall latency/example (s) | "
            "Quality/1k tokens |",
            "|---|---|---|---|---|---|---|",
        ]
        for strategy, summary in strategies.items():
            m = summary["metrics"]
            q = m["task_score"]
            lines.append(
                f"| {strategy} | {q['mean']:.9f} "
                f"[{q['ci_lower']:.4f}, {q['ci_upper']:.4f}] | "
                f"{m['model_calls']['mean']:.5f}/{m['model_calls']['median']:g} | "
                f"{m['total_tokens']['mean']:.5f}/{m['total_tokens']['median']:g} | "
                f"{m['latency_ms']['mean'] / 1000:.3f} | "
                f"{m['wall_clock_strategy_latency_ms']['mean'] / 1000:.3f} | "
                f"{summary['quality_per_1k_tokens']:.4f} |"
            )
        lines += [
            "",
            "### Paired Deltas Against SingleAgent",
            "",
            "| Strategy | Score delta (95% CI) | Token delta | Call delta | Latency delta (s) |",
            "|---|---|---|---|---|",
        ]
        for strategy, delta in result["paired_deltas"][condition].items():
            if strategy == "single_agent":
                continue
            q = delta["task_score"]
            lines.append(
                f"| {strategy} | {q['mean_difference']:+.9f} "
                f"[{q['ci_lower']:+.4f}, {q['ci_upper']:+.4f}] | "
                f"{delta['total_tokens']['mean_difference']:+.5f} | "
                f"{delta['model_calls']['mean_difference']:+.5f} | "
                f"{delta['latency_ms']['mean_difference'] / 1000:+.3f} |"
            )
        for dimension in ("task", "difficulty", "reasoning_family"):
            lines += [
                "",
                f"### {dimension}",
                "",
                "| Strategy | Group | n | Mean score | Tiny cell (<5) |",
                "|---|---|---|---|---|",
            ]
            for strategy, summary in strategies.items():
                for group, data in summary["breakdown"][dimension].items():
                    lines.append(
                        f"| {strategy} | {group} | {data['n']} | "
                        f"{data['mean']:.6f} | {data['tiny_cell']} |"
                    )
        lines += [
            "",
            "### Efficiency And Mechanisms",
            "",
            "Pareto dominance is descriptive within this DEV condition; no global winner.",
            "",
        ]
        for strategy in strategies:
            lines.append(
                f"- {strategy}: dominated-by per resource: {result['pareto'][condition][strategy]}"
            )
        lines += [
            "",
            "```json",
            json.dumps(
                {s: d["mechanism"] for s, d in strategies.items()}, ensure_ascii=False, indent=2
            ),
            "```",
        ]
    lines += ["", "## Limitations", "", *[f"- {s}" for s in result["limitations"]], ""]
    return "\n".join(lines)


def analyze(
    directory: Path,
    *,
    store: SQLiteStore | None = None,
    run_ids: list[str] | None = None,
    evidence_directory: Path | None = None,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Analyze a read-only view, keeping derived reports separate from raw evidence."""
    evidence_directory = evidence_directory or directory
    before = evidence_hashes(evidence_directory)
    store = store or ReadOnlyPilotStore(evidence_directory / "phase8.sqlite3")
    if run_ids is None:
        status = json.loads((evidence_directory / "phase8_status.json").read_text())
        run_ids = status["run_ids"]
    workload = json.loads((evidence_directory / "workload_estimate.json").read_text())
    groups: dict[str, list[str]] = defaultdict(list)
    summaries: dict[str, dict[str, Any]] = defaultdict(dict)
    observations: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    routes, candidates, all_calls = [], [], []
    review = [
        "# Phase 8.1 Manual Review",
        "",
        "Stored structured outputs only. DEV retrospective analysis.",
    ]
    for run_id in run_ids:
        run = store.get_run(run_id)
        assert run is not None and run["status"] == "COMPLETED"
        artifact_path = Path(run["artifact_dir"]) / "predictions.jsonl"
        if not artifact_path.is_file():
            matches = list((directory / "runs").glob(f"*/{run_id}/predictions.jsonl"))
            if len(matches) != 1:
                raise ValueError(f"Missing/ambiguous prediction artifact for {run_id}")
            artifact_path = matches[0]
        raw_predictions = {
            r["example_id"]: r
            for line in artifact_path.read_text().splitlines()
            if (r := json.loads(line))
        }
        condition = condition_for_run(store, run_id)
        strategy = str(run["strategy"])
        groups[condition].append(run_id)
        calls = store.get_run_model_calls(run_id)
        all_calls.extend(calls)
        call_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for call in calls:
            call_groups[call["example_id"]].append(call)
        scores = store.example_metric_values(run_id, "task_score")
        rows = []
        for prediction in store.get_run_predictions(run_id):
            example_id = prediction["example_id"]
            example_row = store._fetch_one(
                "SELECT payload_json, split FROM examples WHERE example_id=?", (example_id,)
            )
            assert example_row is not None and example_row["split"] == "dev"
            example = json.loads(example_row["payload_json"])
            metadata = json.loads(prediction["metadata_json"])
            usage = json.loads(prediction["usage_json"])
            assert raw_predictions[example_id]["output"] == json.loads(prediction["output_json"])
            row = {
                "condition": condition,
                "strategy": strategy,
                "run_id": run_id,
                "example_id": example_id,
                "task_score": scores[example_id],
                "task": example["task_type"],
                "difficulty": example["metadata"]["difficulty"],
                "reasoning_family": example["metadata"]["reasoning_family"],
                **usage,
                "wall_clock_strategy_latency_ms": metadata["wall_clock_strategy_latency_ms"],
                "metadata": metadata,
                "output": json.loads(prediction["output_json"]),
                "candidates": raw_predictions[example_id]["candidates"],
                "example": BenchmarkExample.model_validate(example),
            }
            if provenance is not None:
                row["source"] = provenance[run_id]
            rows.append(row)
            if strategy == "adaptive_router":
                routes.append(
                    {
                        k: row[k]
                        for k in (
                            "condition",
                            "strategy",
                            "run_id",
                            "example_id",
                            "task_score",
                            "model_calls",
                        )
                    }
                    | reconstruct_route(metadata, call_groups[example_id])
                )
            if strategy == "single_agent":
                candidates.append(
                    {
                        "condition": condition,
                        "strategy": strategy,
                        "run_id": run_id,
                        "example_id": example_id,
                        "candidate_only": True,
                        "features_available_at_inference_time": {
                            "confidence": prediction["confidence"],
                            **usage,
                        },
                        "retrospective_training_targets": {
                            "initial_quality": scores[example_id],
                            "beneficial_escalation_label": None,
                        },
                    }
                )
        assert len(rows) == 32 and len({r["example_id"] for r in rows}) == 32
        observations[condition][strategy] = rows
        metrics = {
            key: summary_stats([float(r[key]) for r in rows], seed=20261003).to_dict()
            for key in (
                "task_score",
                "model_calls",
                "total_tokens",
                "latency_ms",
                "wall_clock_strategy_latency_ms",
            )
        }
        breakdown: dict[str, Any] = {}
        for key in ("task", "difficulty", "reasoning_family"):
            breakdown[key] = {}
            for value in sorted({str(r[key]) for r in rows}):
                values = [float(r["task_score"]) for r in rows if r[key] == value]
                breakdown[key][value] = summary_stats(values, seed=20261003).to_dict() | {
                    "tiny_cell": len(values) < 5
                }
        failures = store.get_run_failures(run_id)
        failure_counts: dict[str, Counter[str]] = defaultdict(Counter)
        by_id = {r["example_id"]: r for r in rows}
        for failure in failures:
            failure_counts[by_id[failure["example_id"]]["task"]][failure["failure_type"]] += 1
        summaries[condition][strategy] = {
            "run_id": run_id,
            "condition": condition,
            "strategy": strategy,
            "metrics": metrics,
            "per_call_latency_ms": summary_stats(
                [float(c["latency_ms"]) for c in calls], seed=20261003
            ).to_dict(),
            "run_elapsed_seconds": (
                datetime.fromisoformat(run["end_ts"]) - datetime.fromisoformat(run["start_ts"])
            ).total_seconds(),
            "total_strategy_wall_latency_ms": sum(
                r["wall_clock_strategy_latency_ms"] for r in rows
            ),
            "breakdown": breakdown,
            "quality_per_1k_tokens": 1000
            * float(metrics["task_score"]["mean"])
            / float(metrics["total_tokens"]["mean"]),
            "api_cost_usd": 0,
            "quality_per_dollar": None,
            "failure_taxonomy_by_task": {k: dict(v) for k, v in failure_counts.items()},
            "parse_status_counts": dict(
                Counter(
                    json.loads(c["metadata_json"]).get("parse_status", "unrecorded") for c in calls
                )
            ),
            "mechanism": {
                "pre_call_budget_rejections": sum(
                    event.get("event_type") == "PRE_CALL_BUDGET_REJECTION"
                    for r in rows
                    for event in r["metadata"].get("budget_events", [])
                ),
                "post_call_budget_overruns": sum(
                    bool(r["metadata"].get("post_call_budget_overrun")) for r in rows
                ),
                "revised": sum(bool(r["metadata"].get("revised")) for r in rows),
                "not_revised_explicit": sum(r["metadata"].get("revised") is False for r in rows),
                "repair_retry": sum(bool(r["metadata"].get("repair_retry")) for r in rows),
                "budget_exhausted": sum(bool(r["metadata"].get("budget_exhausted")) for r in rows),
                **mechanism_analysis(rows, strategy),
                "revision_rate": sum(bool(r["metadata"].get("revised")) for r in rows) / 32,
                "no_revision_rate": sum(r["metadata"].get("revised") is False for r in rows) / 32,
            },
        }
        if provenance is not None:
            summaries[condition][strategy]["source"] = provenance[run_id]
    paired: dict[str, Any] = {}
    pareto: dict[str, Any] = {}
    compatibility: dict[str, Any] = {}
    accounts: dict[str, Any] = {}
    for condition, strategies in observations.items():
        baseline = {r["example_id"]: r for r in strategies["single_agent"]}
        paired[condition] = {}
        pareto[condition] = {}
        for strategy, rows in strategies.items():
            paired[condition][strategy] = {
                key: paired_bootstrap_comparison(
                    baseline={k: float(v[key]) for k, v in baseline.items()},
                    contender={r["example_id"]: float(r[key]) for r in rows},
                    metric=key,
                    seed=20261003,
                ).to_dict()
                for key in ("task_score", "total_tokens", "model_calls", "latency_ms")
            }
            pareto[condition][strategy] = {}
            own = summaries[condition][strategy]["metrics"]
            for resource in ("total_tokens", "model_calls", "wall_clock_strategy_latency_ms"):
                dominated = [
                    other
                    for other, data in summaries[condition].items()
                    if other != strategy
                    and data["metrics"]["task_score"]["mean"] >= own["task_score"]["mean"]
                    and data["metrics"][resource]["mean"] <= own[resource]["mean"]
                    and (
                        data["metrics"]["task_score"]["mean"] > own["task_score"]["mean"]
                        or data["metrics"][resource]["mean"] < own[resource]["mean"]
                    )
                ]
                pareto[condition][strategy][resource] = {"dominated_by": dominated}
            ranked = sorted(
                rows, key=lambda r: r["task_score"] - baseline[r["example_id"]]["task_score"]
            )
            selected = {r["example_id"]: r for r in [*ranked[:2], *ranked[-2:]]}
            if strategy == "single_agent":
                ordered = sorted(rows, key=lambda r: r["task_score"])
                selected = {r["example_id"]: r for r in [*ordered[:2], *ordered[-2:]]}
            if strategy == "critic_reviser":
                changes = summaries[condition][strategy]["mechanism"]["examples"]
                for sign in (-1, 1):
                    qualifying = [
                        r for r in changes if sign * r.get("final_minus_generator", 0) > 0
                    ]
                    if qualifying:
                        chosen = max(qualifying, key=lambda r: sign * r["final_minus_generator"])
                        selected[chosen["example_id"]] = next(
                            r for r in rows if r["example_id"] == chosen["example_id"]
                        )
            if strategy == "adaptive_router":
                for route in ("accept", "critic", "debate"):
                    qualifying_rows = [r for r in rows if r["metadata"].get("route") == route]
                    if qualifying_rows:
                        selected[qualifying_rows[0]["example_id"]] = qualifying_rows[0]
                selected.update(
                    {
                        r["example_id"]: r
                        for r in rows
                        if r["metadata"].get("route") not in {"accept", "critic", "debate"}
                    }
                )
            for row in selected.values():
                example = store._fetch_one(
                    "SELECT payload_json FROM examples WHERE example_id=?", (row["example_id"],)
                )
                assert example is not None
                payload = json.loads(example["payload_json"])
                review.extend(
                    [
                        "",
                        f"## {condition} / {strategy} / {row['example_id']}",
                        f"Run: {row['run_id']}; score delta vs SingleAgent: "
                        f"{row['task_score'] - baseline[row['example_id']]['task_score']:.6f}",
                        "```json",
                        json.dumps(
                            {
                                "input": payload["input"],
                                "gold": payload["gold"],
                                "prediction": row["output"],
                                "metadata": row["metadata"],
                                "baseline_output": baseline[row["example_id"]]["output"],
                                "candidate_outputs": row["candidates"],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        "```",
                    ]
                )
        compatibility[condition] = validate_matched_budget_compatibility(store, groups[condition])
        accounts[condition] = accounting(
            [c for c in all_calls if c["run_id"] in groups[condition]],
            workload[f"{condition}_condition"],
        )
    combined_estimate = {
        key: sum(accounts[c]["pre_run_estimate"][key] for c in groups)
        for key in ("estimated_model_calls", "estimated_total_tokens")
    }
    accounts["combined"] = accounting(all_calls, combined_estimate)
    router_summary = {}
    for condition in groups:
        subset = [r for r in routes if r["condition"] == condition]
        router_summary[condition] = {
            "route_counts": dict(Counter(r["route"] for r in subset)),
            "reconstructed": [
                r for r in subset if r["route_source"] != "explicit_prediction_metadata"
            ],
            "accepted_vs_escalated": {
                label: {
                    "n": len(route_rows := [r for r in subset if r["accepted"] == accepted]),
                    "mean_calls": sum(r["model_calls"] for r in route_rows) / len(route_rows)
                    if route_rows
                    else None,
                    "mean_task_score": sum(r["task_score"] for r in route_rows) / len(route_rows)
                    if route_rows
                    else None,
                }
                for label, accepted in (("accepted", True), ("escalated", False))
            },
            "accept_rate": sum(r["accepted"] for r in subset) / len(subset),
            "escalation_rate": sum(r["escalated"] for r in subset) / len(subset),
            "critic_route_rate": sum(r["route"] == "critic" for r in subset) / len(subset),
            "debate_route_rate": sum(r["route"] == "debate" for r in subset) / len(subset),
        }
        route_ids = {r["example_id"]: r for r in subset}
        single = {r["example_id"]: r for r in observations[condition]["single_agent"]}
        critics = {r["example_id"]: r for r in observations[condition]["critic_reviser"]}
        debates = {r["example_id"]: r for r in observations[condition]["multi_agent_debate"]}
        router_summary[condition]["retrospective_diagnostics"] = {
            "accepted_with_higher_standalone_critic_or_debate_score": [
                example_id
                for example_id, route in route_ids.items()
                if route["accepted"]
                and max(critics[example_id]["task_score"], debates[example_id]["task_score"])
                > single[example_id]["task_score"]
            ],
            "escalated_with_lower_final_score_than_initial": [
                r["example_id"]
                for r in observations[condition]["adaptive_router"]
                if route_ids[r["example_id"]]["escalated"]
                and r["candidates"]
                and r["task_score"]
                < score_prediction(r["example"], r["candidates"][0]["output"])["task_score"]
            ],
            "caveat": "Standalone strategy comparison is not a routing counterfactual. "
            "Missing candidates in budget-replaced results prevent initial-score diagnosis. "
            "These are retrospective gold-based diagnostics, never router input features.",
        }
    benchmark_dir = Path("data/benchmark_v3_1")
    from collectiveeval.pilot import verify_phase8_dev_freeze

    freeze = verify_phase8_dev_freeze()
    integrity = {
        "dev_freeze": freeze,
        "test_model_evaluation_occurred": False,
        "raw_run_count": len(run_ids),
        "examples_per_run": 32,
        "all_runs_completed": True,
        "condition_separation": dict(groups),
        "compatibility": compatibility,
        "provider_model": sorted(
            {
                (c["provider"], c["model"], json.loads(c["metadata_json"]).get("model_digest"))
                for c in all_calls
            }
        ),
        "accounting": accounts["combined"],
        "raw_evidence_sha256_before": before,
        "benchmark_directory_hint": str(benchmark_dir),
        "route_reconstruction": {
            c: {
                "reconstructed": len(router_summary[c]["reconstructed"]),
                "unresolved": sum(
                    r["route_source"] == "unresolved" for r in routes if r["condition"] == c
                ),
            }
            for c in groups
        },
        "no_test_evaluation_evidence": "All selected scientific predictions join to DEV examples. "
        "Offline script opens no TEST file and constructs no provider. Full pytest includes "
        "structural split/hash validation, with no real TEST model evaluation.",
        "historical_budget_admission": "Estimated serialized ProviderRequest included gold and "
        "local model metadata; future real-provider estimates now use rendered prompts. "
        "Compatibility passing establishes shared protocol, not absence of implementation defects.",
    }
    limitations = [
        "32 selected DEV examples, one quantized local model; no held-out claims.",
        "Bootstrap intervals are descriptive; tiny cells (<5) are flagged.",
        "Call records retain parse status but not raw responses. Critic text and intermediate "
        "candidates are absent where top-level budget replacement discarded strategy metadata.",
        "Mechanism help/hurt against SingleAgent is retrospective association, "
        "not causal evidence.",
        "No learned router training; gold scores appear only in retrospective targets.",
        "Local marginal API cost is zero; compute and electricity are not universally free.",
        "Historical SelfConsistency repeated the same prompt and requested seed; natural K=2 "
        "candidates were identical on 32/32 examples. Future sampling uses distinct seed offsets. "
        "Real APIs still cannot guarantee deterministic replay or independent draws.",
        "Historical real-provider admission estimated serialized request data including gold and "
        "model metadata, causing conservative premature stopping. Future estimates use prompts. "
        "Do not interpret matched-token losses as orchestration-only effects.",
        "Historical ledger rejects over-budget completed responses before persistence; recorded "
        "calls alone cannot prove no unrecorded provider attempts. "
        "No such attempt is asserted here.",
    ]
    result = {
        "analysis_version": "phase8.1.offline.v1",
        "bootstrap_seed": 20261003,
        "strategy_results": dict(summaries),
        "paired_deltas": paired,
        "pareto": pareto,
        "provider_accounting": accounts,
        "router_summary": router_summary,
        "router_examples": routes,
        "limitations": limitations,
    }
    if provenance is not None:
        result["analysis_version"] = "phase8.2.corrected-dev.v1"
        result["bootstrap_replicates"] = 1000
        result["observations"] = {
            condition: {
                strategy: [{k: v for k, v in r.items() if k != "example"} for r in rows]
                for strategy, rows in strategies.items()
            }
            for condition, strategies in observations.items()
        }
        result["limitations"] = [
            *limitations[:6],
            "Distinct requested seeds are not a guarantee of independent generations or "
            "deterministic API replay. Historical same-seed SelfConsistency is replaced here.",
            "Five reused historical runs retain their original accounting semantics. "
            "The single-call matched baseline is reused by the approved corrective protocol.",
            "Historical disappearing-call impact remains indeterminate without server logs.",
        ]
    (directory / "phase8_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    for filename, title, payload in (
        ("provider_accounting.md", "Corrected Provider Accounting", accounts),
        ("router_analysis.md", "Router Analysis", {"summary": router_summary, "examples": routes}),
    ):
        write_report(directory / filename, title, payload)
    for name in ("phase8_analysis.md", "pilot_report.md"):
        text = readable_analysis(result)
        if provenance is not None:
            text = text.replace("Phase 8.1 DEV Pilot Analysis", "Phase 8.2 Corrected DEV Analysis")
            text = text.replace(
                "No new inference or learned-router training.",
                "Five reused historical runs plus five corrective runs; no learned router.",
            )
        (directory / name).write_text(text, encoding="utf-8")
    router_lines = [
        "# Phase 8.1 Router Analysis",
        "",
        "Routes reconstructed from persisted call roles; historical predictions remain unchanged.",
        "",
        "| Condition | Example | Route | Accepted/escalated | Nested strategy | Calls | "
        "Budget exhausted | Score | Source |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in routes:
        router_lines.append(
            f"| {r['condition']} | {r['example_id']} | {r['route']} | "
            f"{'accepted' if r['accepted'] else 'escalated' if r['escalated'] else 'unresolved'} | "
            f"{r['nested_strategy']} | {r['model_calls']} | {r['budget_exhausted']} | "
            f"{r['task_score']:.9f} | {r['route_source']} |"
        )
    router_lines += [
        "",
        "## Runtime And Retrospective Diagnostics",
        "",
        "```json",
        json.dumps(router_summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (directory / "router_analysis.md").write_text("\n".join(router_lines), encoding="utf-8")
    (directory / "manual_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    (directory / "router_training_candidate.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in candidates),
        encoding="utf-8",
    )
    integrity["raw_evidence_unchanged"] = before == evidence_hashes(evidence_directory)
    assert integrity["raw_evidence_unchanged"]
    write_report(directory / "phase8_integrity_report.md", "Phase 8 Integrity", integrity)
    return result
