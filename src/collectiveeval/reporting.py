"""Run evaluation, comparison, and reproduction reports."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from collectiveeval.statistics import paired_bootstrap_comparison, summary_stats
from collectiveeval.storage import SQLiteStore

REJECTION = {
    "dataset": "DATASET_HASH_MISMATCH",
    "examples": "EXAMPLE_SET_MISMATCH",
    "split": "SPLIT_MISMATCH",
    "task": "TASK_FAMILY_MISMATCH",
    "mode": "BUDGET_MODE_MISMATCH",
    "tokens": "TOKEN_BUDGET_MISMATCH",
    "calls": "CALL_BUDGET_MISMATCH",
    "cost": "COST_BUDGET_MISMATCH",
    "model": "MODEL_CONSTRAINT_MISMATCH",
    "prompt": "PROMPT_VERSION_MISMATCH",
    "benchmark": "BENCHMARK_VERSION_MISMATCH",
}


def evaluate_run_report(
    store: SQLiteStore,
    run_id: str,
    *,
    metric: str = "task_score",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a structured report for one run."""

    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    aggregate_metrics = _aggregate_metrics(store.get_run_metrics(run_id))
    example_metrics = store.example_metric_values(run_id, metric)
    failures = store.get_run_failures(run_id)
    failure_counts = Counter(row["failure_type"] for row in failures)
    budget_events = [
        event
        for prediction in store.get_run_predictions(run_id)
        for event in json.loads(prediction["metadata_json"]).get("budget_events", [])
    ]
    return {
        "run": run,
        "metric": metric,
        "summary": summary_stats(
            list(example_metrics.values()),
            n_bootstrap=n_bootstrap,
            seed=seed,
        ).to_dict(),
        "aggregate_metrics": aggregate_metrics,
        "failure_counts": dict(sorted(failure_counts.items())),
        "failures": failures,
        "budget_event_counts": dict(Counter(event["event_type"] for event in budget_events)),
        "budget_events": budget_events,
        "attempt_accounting": attempt_accounting_report(store, run_id),
        "component_coverage": component_coverage(store, run_id),
    }


def component_coverage(store: SQLiteStore, run_id: str) -> dict[str, Any]:
    predictions = store._fetch_all(
        "SELECT example_id,json_extract(metadata_json,'$.result_kind') AS result_kind "
        "FROM predictions WHERE run_id=?",
        (run_id,),
    )
    failed = sum(row["result_kind"] == "failed_prediction" for row in predictions)
    rows = store._fetch_all(
        "SELECT m.metric_name,COUNT(*) AS n_applicable,COUNT(m.metric_value) AS n_scored, "
        "SUM(CASE WHEN json_extract(p.metadata_json,'$.result_kind')='failed_prediction' "
        "THEN 1 ELSE 0 END) AS failed_output_count FROM metrics m JOIN predictions p "
        "ON m.run_id=p.run_id AND m.example_id=p.example_id "
        "WHERE m.run_id=? GROUP BY m.metric_name",
        (run_id,),
    )
    return {
        "n_total": len(predictions),
        "failed_output_count": failed,
        "components": {
            row["metric_name"]: {
                "n_total": len(predictions),
                "n_scored": row["n_scored"],
                "n_applicable": row["n_applicable"],
                "failed_output_count": row["failed_output_count"],
                "n_unavailable": row["n_applicable"] - row["n_scored"],
                "n_not_applicable": len(predictions) - row["n_applicable"],
            }
            for row in rows
        },
    }


def attempt_accounting_report(store: SQLiteStore, run_id: str) -> dict[str, Any]:
    """Keep unknown usage out of exact totals, including legacy retry-incomplete runs."""
    calls = store.get_run_model_calls(run_id)
    modern = bool(calls) and all("attempt_id" in c for c in calls)
    known = sum(
        (c["input_tokens"] or 0) + (c["output_tokens"] or 0)
        for c in calls
        if c["usage_source"] == "PROVIDER_REPORTED"
    )
    unknown = sum(c.get("usage_status") == "UNKNOWN_NOT_RETURNED" for c in calls)
    historical_retries = sum(c["retry_count"] for c in calls) if not modern else 0
    exact = (
        bool(calls)
        and not unknown
        and not historical_retries
        and all(c["usage_source"] == "PROVIDER_REPORTED" for c in calls)
    )
    predictions = store.get_run_predictions(run_id)
    return {
        "schema": "provider-attempts.v2" if modern else "legacy-model-calls",
        "logical_model_calls": len({c["logical_call_id"] for c in calls}) if modern else len(calls),
        "provider_attempts": len(calls) + historical_retries,
        "successful_attempts": sum(
            c.get("outcome", "SUCCESS") in {"SUCCESS", "PARSE_ERROR_AFTER_RESPONSE"} for c in calls
        ),
        "failed_attempts": sum(
            c.get("outcome", "SUCCESS") not in {"SUCCESS", "PARSE_ERROR_AFTER_RESPONSE", "STARTED"}
            for c in calls
        )
        + historical_retries,
        "unknown_usage_attempts": unknown + historical_retries,
        "provider_reported_input_tokens": sum(
            c["input_tokens"] or 0 for c in calls if c["usage_source"] == "PROVIDER_REPORTED"
        ),
        "provider_reported_output_tokens": sum(
            c["output_tokens"] or 0 for c in calls if c["usage_source"] == "PROVIDER_REPORTED"
        ),
        "known_token_lower_bound": known,
        "total_provider_tokens": known if exact else None,
        "total_provider_tokens_exact": exact,
        "total_attempt_latency_ms": sum(c["latency_ms"] for c in calls),
        "total_attempt_latency_exact": modern and not any(c["outcome"] == "STARTED" for c in calls),
        "logical_call_lifecycle_latency_ms": sum(
            r["lifecycle_latency_ms"] for r in store.get_run_logical_calls(run_id)
        )
        if modern
        else None,
        "strategy_wall_clock_latency_ms": sum(
            json.loads(r["metadata_json"]).get("wall_clock_strategy_latency_ms", 0)
            for r in predictions
        ),
        "retries": sum(c.get("attempt_index", 0) > 0 for c in calls)
        if modern
        else historical_retries,
        "accounting_status": "ACCOUNTING_INCOMPLETE_RETRY_ATTEMPTS"
        if historical_retries
        else ("UNKNOWN_USAGE_STOP" if unknown else "EXACT_REPORTED" if exact else "ESTIMATED"),
    }


def compare_runs_report(
    store: SQLiteStore,
    run_ids: list[str],
    *,
    metric: str = "task_score",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare each contender run against the first run as baseline."""

    if len(run_ids) < 2:
        raise ValueError("compare requires at least two run ids")
    compatibility = validate_matched_budget_compatibility(store, run_ids)
    if not compatibility["ok"]:
        return {
            "metric": metric,
            "matched_budget": compatibility,
            "comparisons": [],
            "runs": [
                evaluate_run_report(store, run_id, metric=metric, n_bootstrap=0)
                for run_id in run_ids
                if store.get_run(run_id) is not None
            ],
        }
    baseline_id = run_ids[0]
    baseline_values = store.example_metric_values(baseline_id, metric)
    comparisons = []
    for contender_id in run_ids[1:]:
        contender_values = store.example_metric_values(contender_id, metric)
        comparisons.append(
            {
                "baseline_run_id": baseline_id,
                "contender_run_id": contender_id,
                **paired_bootstrap_comparison(
                    baseline=baseline_values,
                    contender=contender_values,
                    metric=metric,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                ).to_dict(),
            }
        )
    return {
        "metric": metric,
        "matched_budget": compatibility,
        "baseline_run_id": baseline_id,
        "runs": [
            evaluate_run_report(
                store,
                run_id,
                metric=metric,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            for run_id in run_ids
        ],
        "comparisons": comparisons,
        "note": (
            "Intervals are bootstrap confidence intervals; no automatic significance claim is made."
        ),
    }


def validate_matched_budget_compatibility(
    store: SQLiteStore,
    run_ids: list[str],
) -> dict[str, Any]:
    """Validate that runs are compatible for matched-budget analysis."""

    reasons: list[str] = []
    reason_codes: list[str] = []
    experiments = []
    task_sets = []
    for run_id in run_ids:
        calls = store.get_run_model_calls(run_id)
        if any(
            c.get("usage_status") == "UNKNOWN_NOT_RETURNED"
            or c.get("input_tokens") is None
            or c.get("output_tokens") is None
            for c in calls
        ):
            reasons.append(f"run {run_id} has unknown provider consumption")
            reason_codes.append("UNKNOWN_USAGE_ACCOUNTING")
        if any("attempt_id" not in c and c.get("retry_count", 0) for c in calls):
            reasons.append(f"run {run_id} has unpersisted legacy retry attempts")
            reason_codes.append("INCOMPLETE_RETRY_ATTEMPT_ACCOUNTING")
        run = store.get_run(run_id)
        if run is not None and run.get("status", "COMPLETED") != "COMPLETED":
            reasons.append(f"run {run_id} is incomplete")
            reason_codes.append("INCOMPLETE_RUN")
        experiment = store.get_run_experiment(run_id)
        if experiment is None:
            reasons.append(f"missing experiment for run {run_id}")
            reason_codes.append("MISSING_EXPERIMENT")
            continue
        experiments.append((run_id, experiment, json.loads(experiment["config_json"])))
        task_sets.append((run_id, store.run_task_set(run_id)))
    if len(experiments) != len(run_ids):
        return {"ok": False, "reasons": reasons}

    dataset_hashes = {experiment["dataset_hash"] for _, experiment, _ in experiments}
    if len(dataset_hashes) != 1:
        reasons.append("runs use different dataset hashes")
        reason_codes.append(REJECTION["dataset"])

    benchmark_versions = {_benchmark_version(config) for _, _, config in experiments}
    if len(benchmark_versions) != 1:
        reasons.append("runs use different benchmark versions")
        reason_codes.append(REJECTION["benchmark"])

    example_sets = {tuple(task_set["example_ids"]) for _, task_set in task_sets}
    if len(example_sets) != 1:
        reasons.append("runs use different example sets")
        reason_codes.append(REJECTION["examples"])
    split_sets = {tuple(task_set["splits"]) for _, task_set in task_sets}
    if len(split_sets) != 1:
        reasons.append("runs use different splits")
        reason_codes.append(REJECTION["split"])
    task_type_sets = {tuple(task_set["task_types"]) for _, task_set in task_sets}
    if len(task_type_sets) != 1:
        reasons.append("runs use different task type sets")
        reason_codes.append(REJECTION["task"])

    modes = {_budget_mode(config) for _, _, config in experiments}
    if len(modes) != 1:
        reasons.append("runs use different scientific budget modes")
        reason_codes.append(REJECTION["mode"])
    token_budgets = {_budget_value(config, "max_total_tokens") for _, _, config in experiments}
    if len(token_budgets) != 1:
        reasons.append("runs use different matched token budgets")
        reason_codes.append(REJECTION["tokens"])
    call_budgets = {_budget_value(config, "max_calls") for _, _, config in experiments}
    if len(call_budgets) != 1:
        reasons.append("runs use different matched call budgets")
        reason_codes.append(REJECTION["calls"])
    cost_budgets = {_budget_value(config, "max_cost_usd") for _, _, config in experiments}
    if len(cost_budgets) != 1:
        reasons.append("runs use different matched cost budgets")
        reason_codes.append(REJECTION["cost"])

    policies = {_budget_scope(config) for _, _, config in experiments}
    if policies != {"PER_EXAMPLE"}:
        reasons.append("runs do not share per-example scientific budget policy")
        reason_codes.append("BUDGET_SCOPE_MISMATCH")

    prompt_versions = {
        tuple(
            sorted(
                {
                    str(row["prompt_version"]).removesuffix(".critic-contract.v1")
                    if row["role"] == "critic"
                    else str(row["prompt_version"])
                    for row in store.get_run_model_calls(run_id)
                }
            )
        )
        for run_id in run_ids
    }
    if len(prompt_versions) != 1:
        reasons.append("runs use different prompt/task contract versions")
        reason_codes.append(REJECTION["prompt"])

    if any(
        dict(config.get("comparison", {})).get("require_same_model_group", False)
        for _, _, config in experiments
    ):
        model_groups = {_model_group(config) for _, _, config in experiments}
        if len(model_groups) != 1:
            reasons.append("runs use incompatible model groups")
            reason_codes.append(REJECTION["model"])

    return {
        "ok": not reasons,
        "reasons": reasons,
        "reason_codes": sorted(set(reason_codes)),
        "policy": "per_example",
        "run_ids": run_ids,
    }


def experiment_report(
    store: SQLiteStore,
    experiment_id: str,
    *,
    metric: str = "task_score",
) -> dict[str, Any]:
    """Summarize one experiment and all runs attached to it."""

    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"experiment not found: {experiment_id}")
    runs = store.get_runs_for_experiment(experiment_id)
    return {
        "experiment": experiment,
        "runs": [evaluate_run_report(store, run["id"], metric=metric) for run in runs],
        "artifacts": [store.get_run_artifacts(run["id"]) for run in runs],
    }


def reproduction_plan(store: SQLiteStore, experiment_id: str) -> dict[str, Any]:
    """Return resolved config and a reproducible local command skeleton."""

    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"experiment not found: {experiment_id}")
    config = json.loads(experiment["config_json"])
    return {
        "experiment_id": experiment_id,
        "name": experiment["name"],
        "config_hash": experiment["config_hash"],
        "dataset_hash": experiment["dataset_hash"],
        "config": config,
        "command": (
            "PYTHONPATH=src python3 -m collectiveeval.cli run "
            "--config <saved-config.yaml> --db <output.sqlite3> --output-dir <runs-dir>"
        ),
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        if row["example_id"] is None:
            metrics[str(row["metric_name"])] = float(row["metric_value"])
    return metrics


def _budget_mode(config: dict[str, Any]) -> str:
    scientific = dict(dict(config.get("budget_policy", {})).get("scientific_budget", {}))
    return str(scientific.get("mode") or dict(config.get("budget_policy", {})).get("mode") or "")


def _budget_scope(config: dict[str, Any]) -> str:
    scientific = dict(dict(config.get("budget_policy", {})).get("scientific_budget", {}))
    return str(scientific.get("scope") or "PER_EXAMPLE").upper()


def _budget_value(config: dict[str, Any], key: str) -> Any:
    scientific = dict(dict(config.get("budget_policy", {})).get("scientific_budget", {}))
    budget = dict(config.get("budget", {}))
    if key == "max_cost_usd":
        return scientific.get("max_estimated_cost_usd", budget.get("max_cost_usd"))
    return scientific.get(key, budget.get(key))


def _benchmark_version(config: dict[str, Any]) -> str:
    return str(config.get("benchmark_version", ""))


def _model_group(config: dict[str, Any]) -> tuple[Any, ...]:
    model = dict(config.get("model", {}))
    return model.get("provider"), model.get("name") or model.get("model")
