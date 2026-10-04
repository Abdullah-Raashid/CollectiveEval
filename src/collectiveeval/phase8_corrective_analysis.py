"""Read-only provenance-aware analysis of the approved Phase 8.2 DEV correction."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from collectiveeval.aggregation import canonical_answer
from collectiveeval.budget import BUDGET_SEMANTICS_VERSION, SCIENTIFIC_ATTEMPT_POLICY
from collectiveeval.core import BenchmarkExample
from collectiveeval.datasets import file_sha256
from collectiveeval.phase8_corrective import CORRECTIVE, ORIGINAL, recursive_model_audit, write_json
from collectiveeval.pilot import verify_phase8_dev_freeze
from collectiveeval.pilot_analysis import (
    ReadOnlyPilotStore,
    analyze,
    evidence_hashes,
    reconstruct_route,
    write_report,
)
from collectiveeval.reporting import attempt_accounting_report
from collectiveeval.statistics import paired_bootstrap_comparison
from collectiveeval.storage import SQLiteStore


def ensure_writable_analysis_namespace(corrective: Path) -> None:
    if corrective.resolve() in {ORIGINAL.resolve(), (ORIGINAL / "phase8_2").resolve()}:
        raise ValueError("Historical/blocked namespaces are immutable; use phase8_2_attempts_v2")


class CorrectedPilotStore(SQLiteStore):
    """Dispatch reads by immutable run ID; never merge or rewrite scientific rows."""

    def __init__(self, sources: dict[str, ReadOnlyPilotStore]) -> None:
        self.sources = sources

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.sources[run_id].get_run(run_id)

    def get_run_experiment(self, run_id: str) -> dict[str, Any] | None:
        return self.sources[run_id].get_run_experiment(run_id)

    def get_run_predictions(self, run_id: str) -> list[dict[str, Any]]:
        return self.sources[run_id].get_run_predictions(run_id)

    def get_run_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self.sources[run_id].get_run_metrics(run_id)

    def get_run_failures(self, run_id: str) -> list[dict[str, Any]]:
        return self.sources[run_id].get_run_failures(run_id)

    def get_run_model_calls(self, run_id: str) -> list[dict[str, Any]]:
        return self.sources[run_id].get_run_model_calls(run_id)

    def get_run_logical_calls(self, run_id: str) -> list[dict[str, Any]]:
        return self.sources[run_id].get_run_logical_calls(run_id)

    def example_metric_values(self, run_id: str, metric_name: str) -> dict[str, float]:
        return self.sources[run_id].example_metric_values(run_id, metric_name)

    def run_task_set(self, run_id: str) -> dict[str, Any]:
        return self.sources[run_id].run_task_set(run_id)

    def _fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        if not query.startswith("SELECT payload_json") or "FROM examples" not in query:
            raise ValueError("Only DEV example reads are supported by the corrected view")
        values = [source._fetch_one(query, params) for source in self.sources.values()]
        present = [value for value in values if value is not None]
        if not present or any(value != present[0] for value in present):
            raise ValueError("Missing or inconsistent DEV example across source stores")
        payload = BenchmarkExample.model_validate_json(present[0]["payload_json"])
        if payload.metadata["split"] != "dev":
            raise ValueError("Corrected analysis only accepts DEV examples")
        return present[0]


def completed_scope(
    original: Path = ORIGINAL, corrective: Path = CORRECTIVE
) -> tuple[CorrectedPilotStore, dict[str, str], list[dict[str, Any]]]:
    stop_path = corrective / "methodology_stop.json"
    if stop_path.exists() and json.loads(stop_path.read_text()).get("unresolved"):
        raise ValueError(
            "Unresolved live methodology blocker prevents corrected scientific analysis"
        )
    scope = json.loads((corrective / "rerun_scope.json").read_text())["entries"]
    status = json.loads((corrective / "phase8_2_status.json").read_text())
    if len(status.get("run_ids", [])) != 5:
        raise ValueError("All five approved corrective runs must finish before analysis")
    old = ReadOnlyPilotStore(original / "phase8.sqlite3")
    new = ReadOnlyPilotStore(corrective / "phase8_2.sqlite3")
    new_runs = {r["id"]: r for r in new._fetch_all("SELECT * FROM runs")}
    if set(new_runs) != set(status["run_ids"]) or any(
        r["status"] != "COMPLETED" for r in new_runs.values()
    ):
        raise ValueError("Corrective DB contains extra or incomplete scientific runs")
    identities: dict[tuple[str, str], str] = {}
    for run_id in new_runs:
        experiment = new.get_run_experiment(run_id)
        assert experiment is not None
        config = json.loads(experiment["config_json"])
        if config["phase8_2"]["observation_kind"] != "SCIENTIFIC":
            raise ValueError("Diagnostic observations must not enter the scientific view")
        identity = (config["phase8"]["condition"], new_runs[run_id]["strategy"])
        if identity in identities:
            raise ValueError("Duplicate condition/strategy corrective result")
        identities[identity] = run_id
    sources, provenance, entries = {}, {}, []
    for entry in scope:
        corrected = entry["action"] == "RERUN"
        run_id = (
            identities[(entry["condition"], entry["strategy"])]
            if corrected
            else entry["historical_run_id"]
        )
        sources[run_id] = new if corrected else old
        provenance[run_id] = "phase8_2_corrective" if corrected else "phase8_original_reused"
        entries.append({**entry, "run_id": run_id, "source": provenance[run_id]})
    if len(entries) != 10 or len(sources) != 10:
        raise ValueError("Corrected view requires exactly ten unique approved runs")
    return CorrectedPilotStore(sources), provenance, entries


def run_rows(store: SQLiteStore, run_id: str) -> list[dict[str, Any]]:
    scores = store.example_metric_values(run_id, "task_score")
    calls = store.get_run_model_calls(run_id)
    rows = []
    for prediction in store.get_run_predictions(run_id):
        example_id = prediction["example_id"]
        metadata = json.loads(prediction["metadata_json"])
        rows.append(
            {
                "example_id": example_id,
                "task_score": scores[example_id],
                "output": json.loads(prediction["output_json"]),
                "metadata": metadata,
                "candidates": metadata.get("persisted_candidates", []),
                "calls": [c for c in calls if c["example_id"] == example_id],
                **json.loads(prediction["usage_json"]),
            }
        )
    # Older SQLite predictions do not contain candidates; their artifacts are authoritative.
    run = store.get_run(run_id)
    assert run is not None
    path = Path(run["artifact_dir"]) / "predictions.jsonl"
    artifacts = {
        r["example_id"]: r for line in path.read_text().splitlines() if (r := json.loads(line))
    }
    for row in rows:
        raw = artifacts[row["example_id"]]
        if raw["output"] != row["output"]:
            raise ValueError("SQLite/artifact output mismatch")
        row["candidates"] = raw["candidates"]
    return rows


def trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    roles = [[c["role"] for c in r["calls"]] for r in rows]
    routes = [reconstruct_route(r["metadata"], r["calls"]) for r in rows]
    return {
        "n": len(rows),
        "mean_task_score": sum(r["task_score"] for r in rows) / len(rows),
        "calls_per_example": sum(r["model_calls"] for r in rows) / len(rows),
        "logical_model_calls_per_example": sum(
            r.get("logical_model_calls", r["model_calls"]) for r in rows
        )
        / len(rows),
        "provider_attempts_per_example": sum(
            len(r["calls"]) + sum(c.get("retry_count", 0) for c in r["calls"]) for r in rows
        )
        / len(rows),
        "tokens_per_example": sum(r["total_tokens"] for r in rows) / len(rows),
        "revision_rate": sum("reviser" in rs for rs in roles) / len(rows),
        "no_revision_rate": sum("reviser" not in rs for rs in roles) / len(rows),
        "repair_rate": sum("repair" in rs for rs in roles) / len(rows),
        "pre_call_budget_rejections": sum(
            e["event_type"] == "PRE_CALL_BUDGET_REJECTION"
            for r in rows
            for e in r["metadata"].get("budget_events", [])
        ),
        "historical_budget_exhaustion_flags": sum(
            bool(r["metadata"].get("budget_exhausted")) for r in rows
        ),
        "post_call_overruns": sum(
            bool(json.loads(c["metadata_json"]).get("post_call_budget_overrun"))
            for r in rows
            for c in r["calls"]
        ),
        "completed_debate_revision_rounds": sum(
            sum(json.loads(c["metadata_json"]).get("round_index") == 1 for c in r["calls"]) == 2
            for r in rows
        ),
        "incomplete_four_call_debate_trajectories": sum(len(r["calls"]) < 4 for r in rows),
        "route_counts": dict(Counter(r["route"] for r in routes)),
        "explicit_route_preserved": sum(
            r["route_source"] == "explicit_prediction_metadata" for r in routes
        ),
        "budget_event_comparability": "Historical pre/post-call events were not recorded; zero "
        "historical event counts mean unavailable, not proof of no rejections/overruns.",
    }


def compare_correction(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {r["example_id"]: r for r in old}
    if set(baseline) != {r["example_id"] for r in new}:
        raise ValueError("Correction comparison must be exactly paired")
    return {
        "old": trajectory_summary(old),
        "corrected": trajectory_summary(new),
        "paired_score_delta": paired_bootstrap_comparison(
            baseline={k: r["task_score"] for k, r in baseline.items()},
            contender={r["example_id"]: r["task_score"] for r in new},
            metric="task_score",
            seed=20261003,
        ).to_dict(),
        "changed_final_predictions": sum(
            r["output"] != baseline[r["example_id"]]["output"] for r in new
        ),
        "examples": [
            {
                "example_id": r["example_id"],
                "old_score": baseline[r["example_id"]]["task_score"],
                "corrected_score": r["task_score"],
                "output_changed": r["output"] != baseline[r["example_id"]]["output"],
                "old_calls": baseline[r["example_id"]]["model_calls"],
                "corrected_calls": r["model_calls"],
                "old_tokens": baseline[r["example_id"]]["total_tokens"],
                "corrected_tokens": r["total_tokens"],
            }
            for r in new
        ],
    }


def diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details = []
    for row in rows:
        outputs = [c["output"] for c in row["candidates"]]
        pair = [c["requested_seed"] for c in row["calls"] if c["role"] == "sample"]
        details.append(
            {
                "example_id": row["example_id"],
                "requested_seeds": pair,
                "two_candidates": len(outputs) == 2,
                "canonical_answer_agreement": len(outputs) == 2
                and canonical_answer(outputs[0]) == canonical_answer(outputs[1]),
                "exact_candidate_identity": len(outputs) == 2 and outputs[0] == outputs[1],
            }
        )
    paired = [r for r in details if r["two_candidates"]]
    return {
        "n": len(rows),
        "paired_candidates": len(paired),
        "agreement_rate": sum(r["canonical_answer_agreement"] for r in paired) / len(paired)
        if paired
        else None,
        "disagreement_rate": sum(not r["canonical_answer_agreement"] for r in paired) / len(paired)
        if paired
        else None,
        "exact_identity_rate": sum(r["exact_candidate_identity"] for r in paired) / len(paired)
        if paired
        else None,
        "examples": details,
        "tie_break": "Majority of canonical answer (QA/robustness: answer+abstain; otherwise "
        "full output); tied vote count uses average candidate confidence, then earliest sample. "
        "Returns the earliest candidate in the winning answer group.",
        "seed_limitation": "Different requested seeds do not guarantee different "
        "or independent generations.",
    }


def diagnostic_report(corrective: Path = CORRECTIVE) -> dict[str, Any]:
    ensure_writable_analysis_namespace(corrective)
    status = json.loads((corrective / "diagnostics" / "status.json").read_text())
    if status["status"] != "PASSED" or len(status["run_ids"]) != 3:
        raise ValueError("Three passed diagnostics are required")
    store = ReadOnlyPilotStore(corrective / "diagnostics" / "diagnostics.sqlite3")
    diagnostics = []
    for run_id in status["run_ids"]:
        run = store.get_run(run_id)
        assert run is not None and run["status"] == "COMPLETED"
        rows = run_rows(store, run_id)
        if len(rows) != 1 or store.run_task_set(run_id)["splits"] != ["dev"]:
            raise ValueError("Diagnostic is not a single DEV example")
        row = rows[0]
        summary = {
            "run_id": run_id,
            "strategy": run["strategy"],
            "example_id": row["example_id"],
            "observation_kind": "DIAGNOSTIC",
            "scientific_results_excluded": True,
            "attempt_accounting": attempt_accounting_report(store, run_id),
            "route": row["metadata"].get("route"),
            "provider_usage_retained": row["total_tokens"]
            == sum(c["input_tokens"] + c["output_tokens"] for c in row["calls"]),
            "calls": [
                {
                    **{
                        k: c[k]
                        for k in (
                            "role",
                            "provider",
                            "model",
                            "requested_seed",
                            "input_tokens",
                            "output_tokens",
                            "usage_source",
                            "normalized_error",
                        )
                    },
                    **{
                        k: json.loads(c["metadata_json"]).get(k)
                        for k in (
                            "model_digest",
                            "admission_estimated_input_tokens",
                            "configured_max_tokens",
                            "effective_max_tokens",
                            "budget_remaining_before_call",
                            "post_call_budget_overrun",
                            "budget_semantics_version",
                            "parse_status",
                        )
                    },
                }
                for c in row["calls"]
            ],
        }
        if run["strategy"] == "self_consistency":
            summary["diversity"] = diversity(rows)
        diagnostics.append(summary)
    report = {
        "status": "PASSED",
        "diagnostic_model_calls": sum(len(d["calls"]) for d in diagnostics),
        "logical_model_calls": sum(
            d["attempt_accounting"]["logical_model_calls"] for d in diagnostics
        ),
        "provider_attempts": sum(d["attempt_accounting"]["provider_attempts"] for d in diagnostics),
        "provider_reported_tokens": sum(
            c["input_tokens"] + c["output_tokens"] for d in diagnostics for c in d["calls"]
        ),
        "diagnostics": diagnostics,
        "gold_exclusion_evidence": "Provider-facing rendering and admission use prompt only; "
        "gold/hidden-metadata mutation regression is in test_phase8_2_budget.py. "
        "No raw request log is claimed.",
    }
    write_json(corrective / "diagnostics" / "diagnostic_report.json", report)
    write_report(
        corrective / "diagnostics" / "diagnostic_report.md", "Phase 8.2 DEV Diagnostics", report
    )
    return report


def validate_attempt_records(
    calls: list[dict[str, Any]], logical_calls: list[dict[str, Any]]
) -> list[str]:
    """Validate the repaired scientific policy, not legacy rows inferred after the fact."""
    issues = []
    attempt_ids = [c.get("attempt_id") for c in calls]
    call_ids = [r.get("logical_call_id") for r in logical_calls]
    if None in attempt_ids or len(set(attempt_ids)) != len(attempt_ids):
        issues.append("Missing or duplicate attempt identity")
    if None in call_ids or len(set(call_ids)) != len(call_ids):
        issues.append("Missing or duplicate logical-call identity")
    if set(call_ids) != {c.get("logical_call_id") for c in calls}:
        issues.append("Attempt/logical-call relationship mismatch")
    for logical in logical_calls:
        if logical.get("outcome") != "SUCCESS" or not logical.get("end_timestamp"):
            issues.append(f"Unfinished logical lifecycle: {logical.get('logical_call_id')}")
    for call in calls:
        metadata = json.loads(call["metadata_json"])
        if (
            call.get("attempt_index") != 0
            or call.get("outcome") != "SUCCESS"
            or call.get("usage_status") != "PROVIDER_REPORTED"
            or call["input_tokens"] is None
            or call["output_tokens"] is None
            or metadata.get("unknown_usage_policy") != SCIENTIFIC_ATTEMPT_POLICY
        ):
            issues.append(f"Scientific attempt policy/usage violation: {call.get('attempt_id')}")
        if not call.get("end_timestamp"):
            issues.append(f"Unfinished attempt: {call.get('attempt_id')}")
        elif datetime.fromisoformat(call["end_timestamp"]) < datetime.fromisoformat(
            call["start_timestamp"]
        ):
            issues.append(f"Invalid attempt timestamp order: {call.get('attempt_id')}")
    return issues


def accounting_markdown(accounts: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 8.2 Attempt Accounting",
        "",
        "Logical generations and sent provider attempts are separate. Tokens are exact only "
        "when every sent attempt returned authoritative usage. Historical reused observations "
        "retain legacy records; absence of recorded retries is not independent server evidence.",
        "",
        "| Condition / strategy | Logical calls | Attempts | Failed | Unknown usage | "
        "Known token lower bound | Exact tokens | Attempt latency (s) | "
        "Logical lifecycle (s) | Strategy wall latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in entries:
        account = accounts[entry["run_id"]]
        logical_latency = account["logical_call_lifecycle_latency_ms"]
        lines.append(
            f"| {entry['condition']} / {entry['strategy']} | "
            f"{account['logical_model_calls']} | {account['provider_attempts']} | "
            f"{account['failed_attempts']} | {account['unknown_usage_attempts']} | "
            f"{account['known_token_lower_bound']} | {account['total_provider_tokens']} | "
            f"{account['total_attempt_latency_ms'] / 1000:.3f} | "
            f"{logical_latency / 1000 if logical_latency is not None else 'unavailable'} | "
            f"{account['strategy_wall_clock_latency_ms'] / 1000:.3f} |"
        )
    return "\n".join(lines) + "\n"


def record_live_blocker(original: Path = ORIGINAL, corrective: Path = CORRECTIVE) -> dict[str, Any]:
    """Document interrupted scope without presenting it as a corrected scientific result."""
    diagnostics = diagnostic_report(corrective)
    store = ReadOnlyPilotStore(corrective / "phase8_2.sqlite3")
    runs = store._fetch_all("SELECT * FROM runs")
    if len(runs) != 1 or runs[0]["status"] != "COMPLETED":
        raise ValueError("Wait for the current run checkpoint before recording the safety stop")
    run_id = runs[0]["id"]
    rows = run_rows(store, run_id)
    calls = store.get_run_model_calls(run_id)
    retry_evidence = []
    for call in calls:
        if not call["retry_count"]:
            continue
        elapsed_ms = (
            datetime.fromisoformat(call["end_timestamp"])
            - datetime.fromisoformat(call["start_timestamp"])
        ).total_seconds() * 1000
        retry_evidence.append(
            {
                **{
                    k: call[k]
                    for k in (
                        "id",
                        "example_id",
                        "role",
                        "requested_seed",
                        "start_timestamp",
                        "end_timestamp",
                        "latency_ms",
                        "input_tokens",
                        "output_tokens",
                        "retry_count",
                        "normalized_error",
                        "usage_source",
                    )
                },
                "elapsed_request_lifecycle_ms": elapsed_ms,
                "elapsed_not_in_recorded_call_latency_ms": elapsed_ms - call["latency_ms"],
                "original_retryable_error_type": "NOT_PERSISTED",
                "failed_attempt_usage": "UNKNOWN_NOT_RETURNED",
            }
        )
    baseline = json.loads(
        (original / "phase8_integrity_report.md")
        .read_text()
        .split("```json\n")[1]
        .split("\n```")[0]
    )["raw_evidence_sha256_before"]
    freeze = verify_phase8_dev_freeze()
    old = ReadOnlyPilotStore(original / "phase8.sqlite3")
    scope = json.loads((corrective / "rerun_scope.json").read_text())["entries"]
    old_id = next(
        e["historical_run_id"]
        for e in scope
        if e["condition"] == "natural" and e["strategy"] == "self_consistency"
    )
    comparison = compare_correction(run_rows(old, old_id), rows)
    report = {
        "status": "BLOCKED_METHODOLOGY",
        "blocker": "RETRY_ATTEMPT_ACCOUNTING",
        "scientific_run_ids": [run_id],
        "scientific_runs_completed": 1,
        "approved_scientific_runs": 5,
        "matched_corrective_runs_started": 0,
        "completed_examples": len(rows),
        "recorded_scientific_model_calls": len(calls),
        "known_retry_attempts": sum(c["retry_count"] for c in calls),
        "known_provider_request_attempts": len(calls) + sum(c["retry_count"] for c in calls),
        "returned_input_tokens": sum(c["input_tokens"] for c in calls),
        "returned_output_tokens": sum(c["output_tokens"] for c in calls),
        "returned_total_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in calls),
        "true_total_provider_tokens": None,
        "true_total_provider_tokens_note": "Recorded totals are a lower bound. The two retryable "
        "failed attempts returned no usage; server-side generation/consumption is unknown.",
        "usage_source_counts_for_recorded_calls": dict(Counter(c["usage_source"] for c in calls)),
        "retry_evidence": retry_evidence,
        "diagnostics": diagnostics,
        "self_consistency_natural_observation_only": {
            **comparison,
            "diversity": diversity(rows),
            "scientifically_clean": False,
            "note": "Outputs retained for diagnosis; this is not the complete corrected DEV view.",
        },
        "dev_freeze": freeze,
        "exact_dev_ids_preserved": [r["example_id"] for r in rows]
        == json.loads((original / "example_ids.json").read_text())["example_ids"],
        "historical_raw_evidence_unchanged": evidence_hashes(original) == baseline,
        "historical_db_sha256": file_sha256(original / "phase8.sqlite3"),
        "test_evaluated": False,
        "phase9_started": False,
        "protocol_frozen": False,
        "next_required_repair": [
            "Persist one attempt record for every sent request, including retryable failures.",
            "Admit/reserve each retry centrally and count it against the call ceiling.",
            "Retain failed-attempt elapsed latency and mark unreturned usage as unknown.",
            "Choose and document a conservative stop policy when failed-attempt usage is unknown.",
            "Handle local timeouts so retries cannot silently overlap unfinished server work.",
            "Revalidate/resume only under a new versioned correction; never rewrite this evidence.",
        ],
    }
    write_json(corrective / "retry_accounting_blocker.json", report)
    write_report(
        corrective / "retry_accounting_blocker.md", "Phase 8.2 Live Methodology Blocker", report
    )
    write_report(
        corrective / "phase8_2_integrity_report.md", "Phase 8.2 Integrity: BLOCKED", report
    )
    write_json(
        corrective / "phase8_2_status.json",
        {
            "status": "BLOCKED_METHODOLOGY",
            "run_ids": [run_id],
            "diagnostic_run_ids": json.loads(
                (corrective / "diagnostics" / "status.json").read_text()
            )["run_ids"],
            "recorded_scientific_model_calls": len(calls),
            "diagnostic_real_model_calls": diagnostics["diagnostic_model_calls"],
            "phase9_started": False,
            "historical_evidence_modified": False,
            "blocker_report": str(corrective / "retry_accounting_blocker.json"),
        },
    )
    return report


def analyze_corrective(original: Path = ORIGINAL, corrective: Path = CORRECTIVE) -> dict[str, Any]:
    ensure_writable_analysis_namespace(corrective)
    before = evidence_hashes(original)
    store, provenance, entries = completed_scope(original, corrective)
    diagnostics = diagnostic_report(corrective)
    result = analyze(
        corrective,
        store=store,
        run_ids=list(provenance),
        evidence_directory=original,
        provenance=provenance,
    )
    attempt_accounts = {e["run_id"]: attempt_accounting_report(store, e["run_id"]) for e in entries}
    result["analysis_version"] = "phase8.2.corrected-dev.attempts.v2"
    result["attempt_accounting"] = attempt_accounts
    write_json(corrective / "attempt_accounting.json", attempt_accounts)
    account_text = accounting_markdown(attempt_accounts, entries)
    (corrective / "attempt_accounting.md").write_text(account_text, encoding="utf-8")
    write_json(corrective / "phase8_analysis.json", result)
    for filename in ("phase8_analysis.md", "pilot_report.md"):
        path = corrective / filename
        path.write_text(path.read_text() + "\n" + account_text, encoding="utf-8")
    old_store = ReadOnlyPilotStore(original / "phase8.sqlite3")
    corrections: dict[str, Any] = {}
    for entry in entries:
        if entry["action"] != "RERUN":
            continue
        old = run_rows(old_store, entry["historical_run_id"])
        new = run_rows(store, entry["run_id"])
        comparison = compare_correction(old, new)
        if entry["strategy"] == "self_consistency":
            comparison["old_diversity"] = diversity(old)
            comparison["corrected_diversity"] = diversity(new)
        corrections[f"{entry['condition']}/{entry['strategy']}"] = comparison
        if entry["strategy"] == "adaptive_router":
            old_by_id = {r["example_id"]: r for r in old}
            corrections["six_formerly_truncated_router_examples"] = [
                {
                    "example_id": r["example_id"],
                    "old_calls": old_by_id[r["example_id"]]["model_calls"],
                    "corrected_calls": r["model_calls"],
                    "old_score": old_by_id[r["example_id"]]["task_score"],
                    "corrected_score": r["task_score"],
                    "corrected_route": r["metadata"].get("route"),
                    "critic_completed": any(
                        c["role"] == "critic" and not c["normalized_error"] for c in r["calls"]
                    ),
                    "reviser_completed": any(
                        c["role"] == "reviser" and not c["normalized_error"] for c in r["calls"]
                    ),
                    "provider_tokens": sum(
                        c["input_tokens"] + c["output_tokens"] for c in r["calls"]
                    ),
                    "budget_still_stops_trajectory": bool(
                        r["metadata"].get("budget_exhausted")
                        or r["metadata"].get("stopped") == "budget"
                        or r["metadata"].get("budget_events")
                    ),
                    "budget_events": r["metadata"].get("budget_events", []),
                }
                for r in new
                if r["example_id"]
                in {
                    "v3_1-ext-03-09",
                    "v3_1-ext-08-07",
                    "v3_1-sum-02-06",
                    "v3_1-sum-03-08",
                    "v3_1-sum-08-07",
                    "v3_1-sum-10-06",
                }
            ]
    view = {
        "analysis_version": result["analysis_version"],
        "runs": entries,
        "observations": result["observations"],
        "strategy_results": result["strategy_results"],
        "paired_deltas": result["paired_deltas"],
        "attempt_accounting": attempt_accounts,
        "diagnostics_excluded": True,
        "historical_raw_data_overwritten": False,
    }
    write_json(corrective / "corrected_phase8_view.json", view)
    write_json(corrective / "correction_analysis.json", corrections)
    write_report(
        corrective / "correction_analysis.md", "Phase 8.2 Correction Analyses", corrections
    )
    freeze = verify_phase8_dev_freeze()
    expected_ids = json.loads((original / "example_ids.json").read_text())["example_ids"]
    new_entries = [e for e in entries if e["action"] == "RERUN"]
    calls = [c for e in new_entries for c in store.get_run_model_calls(e["run_id"])]
    expected_digest = json.loads(
        (corrective / "configs" / "natural_self_consistency.json").read_text()
    )["model"]["provider_options"]["model_digest"]
    issues = []
    logical_calls = [r for e in new_entries for r in store.get_run_logical_calls(e["run_id"])]
    issues.extend(validate_attempt_records(calls, logical_calls))
    for entry in entries:
        run_id = entry["run_id"]
        experiment = store.get_run_experiment(run_id)
        assert experiment is not None
        config = json.loads(experiment["config_json"])
        recursive_model_audit(config)
        rows = run_rows(store, run_id)
        if [r["example_id"] for r in rows] != expected_ids:
            issues.append(f"DEV ID/order mismatch: {run_id}")
        for row in rows:
            for key in ("model_calls", "input_tokens", "output_tokens"):
                observed = (
                    len(row["calls"]) if key == "model_calls" else sum(c[key] for c in row["calls"])
                )
                if row[key] != observed:
                    issues.append(f"Accounting mismatch: {run_id}/{row['example_id']}/{key}")
            if entry["action"] == "RERUN" and (
                row["provider_attempts"] != len(row["calls"])
                or row["logical_model_calls"] != len({c["logical_call_id"] for c in row["calls"]})
            ):
                issues.append(f"Attempt/logical checkpoint mismatch: {run_id}/{row['example_id']}")
        if (
            entry["action"] == "RERUN"
            and entry["strategy"] == "self_consistency"
            and any(
                len(p := r["requested_seeds"]) != 2 or p[0] == p[1]
                for r in diversity(rows)["examples"]
            )
        ):
            issues.append(f"SelfConsistency requested seeds missing or repeated: {run_id}")
    for call in calls:
        m = json.loads(call["metadata_json"])
        if (
            call["provider"] != "ollama"
            or call["model"] != "gemma3:4b"
            or m.get("model_digest") != expected_digest
        ):
            issues.append(f"Provider/model/digest mismatch: {call['id']}")
        if (
            call["usage_source"] != "PROVIDER_REPORTED"
            or m.get("budget_semantics_version") != BUDGET_SEMANTICS_VERSION
        ):
            issues.append(f"Usage/semantics mismatch: {call['id']}")
        if call["normalized_error"]:
            issues.append(f"Provider/parse failure: {call['id']}")
    old_integrity = json.loads(
        (corrective / "phase8_integrity_report.md")
        .read_text()
        .split("```json\n")[1]
        .split("\n```")[0]
    )
    compatibility = old_integrity["compatibility"]
    if not all(c["ok"] for c in compatibility.values()):
        issues.append("Per-condition budget compatibility failed")
    if before != evidence_hashes(original) or not freeze["ok"]:
        issues.append("Historical evidence or DEV freeze changed")
    budget_events = [
        event
        for rows in result["observations"].values()
        for group in rows.values()
        for r in group
        if r["source"] == "phase8_2_corrective"
        for event in r["metadata"].get("budget_events", [])
    ]
    actuals = {
        "model_calls": len(calls),
        "model_calls_semantics": "legacy alias for provider_attempts",
        "logical_model_calls": len(logical_calls),
        "provider_attempts": len(calls),
        "successful_attempts": sum(c["outcome"] == "SUCCESS" for c in calls),
        "failed_attempts": sum(c["outcome"] != "SUCCESS" for c in calls),
        "unknown_usage_attempts": sum(c["usage_status"] == "UNKNOWN_NOT_RETURNED" for c in calls),
        "total_provider_tokens_exact": all(
            attempt_accounts[e["run_id"]]["total_provider_tokens_exact"] for e in new_entries
        ),
        "known_token_lower_bound": sum(
            attempt_accounts[e["run_id"]]["known_token_lower_bound"] for e in new_entries
        ),
        "total_attempt_latency_ms": sum(c["latency_ms"] for c in calls),
        "logical_call_lifecycle_latency_ms": sum(r["lifecycle_latency_ms"] for r in logical_calls),
        "strategy_wall_clock_latency_ms": sum(
            attempt_accounts[e["run_id"]]["strategy_wall_clock_latency_ms"] for e in new_entries
        ),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "total_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in calls),
        "total_provider_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in calls)
        if all(attempt_accounts[e["run_id"]]["total_provider_tokens_exact"] for e in new_entries)
        else None,
        "pre_call_rejections": sum(
            e["event_type"] == "PRE_CALL_BUDGET_REJECTION" for e in budget_events
        ),
        "post_call_overruns": sum(
            bool(json.loads(c["metadata_json"]).get("post_call_budget_overrun")) for c in calls
        ),
        "parse_errors": sum(c["normalized_error"] == "PARSE_ERROR" for c in calls),
        "retries": sum(c["retry_count"] for c in calls),
        "usage_source_counts": dict(Counter(c["usage_source"] for c in calls)),
        "marginal_api_cost_usd": sum(c["estimated_cost_usd"] for c in calls),
    }
    integrity = {
        "status": "PASSED" if not issues else "BLOCKED",
        "issues": issues,
        "scientific_actuals": actuals,
        "scientific_run_ids": [e["run_id"] for e in new_entries],
        "diagnostics": diagnostics,
        "diagnostics_excluded": True,
        "dev_freeze": freeze,
        "historical_evidence_unchanged": before == evidence_hashes(original),
        "historical_db_sha256": file_sha256(original / "phase8.sqlite3"),
        "condition_compatibility": compatibility,
        "test_evaluated": False,
        "model_digest": expected_digest,
        "accounting_limit": "No persisted call/prediction discrepancy. No independent server "
        "log exists to audit unreturned usage or historical disappearing calls.",
        "scientific_attempt_policy": SCIENTIFIC_ATTEMPT_POLICY,
        "attempt_accounting": attempt_accounts,
    }
    write_json(corrective / "integrity.json", integrity)
    write_report(corrective / "phase8_2_integrity_report.md", "Phase 8.2 Integrity Gate", integrity)
    if issues:
        raise ValueError("Phase 8.2 integrity gate failed: " + "; ".join(issues))
    return {"analysis": result, "corrections": corrections, "integrity": integrity}


def main() -> int:
    from collectiveeval.phase8_attempt_repair import analyze_repaired

    print(json.dumps(analyze_repaired(), indent=2))
    return 0
