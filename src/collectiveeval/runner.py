"""Experiment execution for mock-provider benchmark runs."""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field

from collectiveeval.artifacts import write_run_artifacts
from collectiveeval.budget import (
    BUDGET_SEMANTICS_VERSION,
    SCIENTIFIC_ATTEMPT_POLICY,
    BudgetExceeded,
    BudgetLedger,
    ModelCallRecord,
    UnknownUsageStop,
)
from collectiveeval.budget_policy import normalized_budget_config, resolve_budget_policy
from collectiveeval.config import load_yaml_config, stable_config_hash
from collectiveeval.context import StrategyContext
from collectiveeval.core import BenchmarkExample, ProviderErrorType, StrategyResult
from collectiveeval.datasets import BENCHMARK_VERSION, file_sha256, load_jsonl
from collectiveeval.failed_outputs import (
    PENDING,
    FailedOutput,
    FailedPrediction,
    PredictionResult,
    evaluate_output,
    failed_prediction,
    failure_annotations,
)
from collectiveeval.failures import annotate_failures
from collectiveeval.metrics import abstention_scores, derived_quality_metrics
from collectiveeval.providers import ProviderError, build_provider_registry
from collectiveeval.storage import SQLiteStore
from collectiveeval.strategy_factory import strategy_from_config
from collectiveeval.task_contracts import CONTRACT_VERSION


class ExperimentRunResult(BaseModel):
    """Summary returned by a completed experiment run."""

    experiment_id: str
    run_id: str
    status: str
    artifact_dir: str
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    examples: int = 0
    config_hash: str
    dataset_hash: str


def run_experiment_from_path(
    config_path: str | Path,
    *,
    db_path: str | Path = "collectiveeval.sqlite3",
    output_dir: str | Path = "runs",
    max_examples: int | None = None,
    max_cost_usd: float | None = None,
    dry_run: bool = False,
    failed_output_policy: bool = False,
) -> ExperimentRunResult | dict[str, Any]:
    config = load_yaml_config(config_path)
    if dry_run:
        return dry_run_summary(config, max_examples=max_examples, max_cost_usd=max_cost_usd)
    return asyncio.run(
        run_experiment(
            config,
            db_path=db_path,
            output_dir=output_dir,
            max_examples=max_examples,
            max_cost_usd=max_cost_usd,
            failed_output_policy=failed_output_policy,
        )
    )


def dry_run_summary(
    config: dict[str, Any],
    *,
    max_examples: int | None = None,
    max_cost_usd: float | None = None,
) -> dict[str, Any]:
    dataset_path = Path(config["dataset"]["path"])
    examples = load_jsonl(dataset_path)
    selected = examples[:max_examples] if max_examples is not None else examples
    strategy = strategy_from_config(config)
    budget_policy = resolve_budget_policy(config, run_level_max_cost_usd=max_cost_usd)
    resolved_config = _config_with_resolved_budget(
        config,
        budget_policy,
        max_examples=max_examples,
    )
    return {
        "dry_run": True,
        "examples": len(selected),
        "strategy": strategy.name,
        "config_hash": stable_config_hash(resolved_config),
        "dataset_hash": file_sha256(dataset_path),
        "budget_policy": budget_policy.manifest_dict(),
    }


async def run_experiment(
    config: dict[str, Any],
    *,
    db_path: str | Path = "collectiveeval.sqlite3",
    output_dir: str | Path = "runs",
    max_examples: int | None = None,
    max_cost_usd: float | None = None,
    failed_output_policy: bool = False,
) -> ExperimentRunResult:
    """Run one experiment config against a JSONL dataset with persisted outputs."""

    dataset_path = Path(config["dataset"]["path"])
    examples = load_jsonl(dataset_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    budget_policy = resolve_budget_policy(config, run_level_max_cost_usd=max_cost_usd)
    resolved_config = _config_with_resolved_budget(
        config,
        budget_policy,
        max_examples=max_examples,
    )
    strategy = strategy_from_config(config)
    budget = budget_policy.scientific_budget.to_inference_budget()
    providers = build_provider_registry(config)
    now = datetime.now(UTC).isoformat()
    experiment_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    experiment_name = str(config.get("experiment", {}).get("name", "unnamed"))
    config_hash = stable_config_hash(resolved_config)
    dataset_hash = file_sha256(dataset_path)
    git_commit = _git_commit()
    artifact_dir = Path(output_dir) / experiment_name / run_id
    store = SQLiteStore(db_path)
    scientific_mode = (
        dict(config.get("provider_retry", {})).get("unknown_usage_policy") != "GENERAL_RETRY"
    )
    if scientific_mode and store.has_unknown_usage_for_config(config_hash):
        raise UnknownUsageStop(
            "UNKNOWN_USAGE_STOP: previous sent attempt under this config has "
            "unknown usage; automatic resume/relaunch is forbidden"
        )
    log_lines = [f"{now} starting run {run_id}"]
    resume_enabled = bool(dict(config.get("checkpoint", {})).get("resume_completed_examples"))
    completed_example_ids: set[str] = set()

    if resume_enabled:
        resumable = store.find_resumable_run_by_config_hash(config_hash)
        if resumable is not None:
            experiment_id = str(resumable["experiment_id"])
            run_id = str(resumable["id"])
            artifact_dir = Path(output_dir) / experiment_name / run_id
            completed_example_ids = {
                str(row["example_id"]) for row in store.get_run_predictions(run_id)
            }
            log_lines.append(
                f"{datetime.now(UTC).isoformat()} resuming run {run_id} "
                f"with {len(completed_example_ids)} completed examples"
            )

    store.insert_experiment(
        experiment_id=experiment_id,
        name=experiment_name,
        config_hash=config_hash,
        dataset_hash=dataset_hash,
        git_commit=git_commit,
        python_version=platform.python_version(),
        created_at=now,
        config=resolved_config,
    )
    store.insert_run(
        run_id=run_id,
        experiment_id=experiment_id,
        strategy=strategy.name,
        status="RUNNING",
        start_ts=now,
    )

    predictions: list[PredictionResult] = _prediction_rows_to_results(
        store.get_run_predictions(run_id)
    )
    metric_rows: list[dict[str, Any]] = _example_metric_rows(store, run_id, strategy.name)
    failure_rows: list[dict[str, Any]] = [
        {
            "run_id": row["run_id"],
            "example_id": row["example_id"],
            "failure_type": row["failure_type"],
            "source": row["source"],
            "rationale": row["rationale"],
            "metadata": json.loads(str(row["metadata_json"])),
        }
        for row in store.get_run_failures(run_id)
    ]
    abstention_predicted: list[bool] = []
    abstention_expected: list[bool] = []
    try:
        by_id = {example.id: example for example in examples}
        for prediction in predictions:
            if (
                isinstance(prediction, FailedPrediction)
                and prediction.metadata.get("evaluation_status") == PENDING
            ):
                if not failed_output_policy:
                    raise ValueError("Failed-output evaluation requires explicit v3 policy")
                scores = evaluate_output(by_id[prediction.example_id], prediction.output)
                scores.update(_usage_scores(prediction))
                scores.update(
                    derived_quality_metrics(
                        task_score=float(scores["task_score"] or 0),
                        total_tokens=prediction.total_tokens,
                        estimated_cost_usd=prediction.estimated_cost_usd,
                        latency_ms=prediction.latency_ms,
                    )
                )
                store.evaluate_pending_failure(prediction, scores)
                metric_rows.append(
                    {"example_id": prediction.example_id, "strategy": prediction.strategy, **scores}
                )
        spent_cost = 0.0
        for example in examples:
            store.upsert_example(example)
            if example.id in completed_example_ids:
                log_lines.append(
                    f"{datetime.now(UTC).isoformat()} skipped completed example {example.id}"
                )
                continue
            ledger = BudgetLedger(budget=budget, run_id=run_id)
            context = StrategyContext(
                providers=providers,
                ledger=ledger,
                max_retries=int(config.get("provider_retry", {}).get("max_retries", 2)),
                concurrency_limit=int(config.get("concurrency", {}).get("limit", 1)),
                scientific_mode=scientific_mode,
                attempt_sink=store.insert_model_call,
                logical_call_sink=store.insert_logical_call,
            )
            started = time.perf_counter()
            result: PredictionResult
            try:
                previous_attempts = [
                    c for c in store.get_run_model_calls(run_id) if c["example_id"] == example.id
                ]
                if previous_attempts:
                    for row in previous_attempts:
                        record = ModelCallRecord.model_validate(
                            {**row, "metadata": json.loads(row["metadata_json"])}
                        )
                        ledger.record_call(record)
                    result = _budget_exhausted_result(
                        example, strategy.name, ledger, "INTERRUPTED_TRAJECTORY_STOP"
                    )
                    result.metadata["interrupted_trajectory_stop"] = True
                    budget_exhausted = "INTERRUPTED_TRAJECTORY_STOP"
                else:
                    result = await strategy.run(example, context)
                    budget_exhausted = None
            except BudgetExceeded as exc:
                result = _budget_exhausted_result(example, strategy.name, ledger, str(exc))
                budget_exhausted = str(exc)
            except ProviderError as exc:
                if not failed_output_policy or exc.error_type != ProviderErrorType.PARSE_ERROR:
                    raise
                result = failed_prediction(ledger, example.id, strategy.name)
                budget_exhausted = None
            finally:
                # Persist completed usage even when the strategy fails on parsing/provider errors.
                for record in ledger.records:
                    store.insert_model_call(record)
            wall_clock_latency_ms = (time.perf_counter() - started) * 1000
            result = result.model_copy(
                update={
                    "logical_model_calls": ledger.logical_model_calls,
                    "provider_attempts": ledger.provider_attempts,
                    "metadata": {
                        **result.metadata,
                        **ledger.metadata_dict(),
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "wall_clock_strategy_latency_ms": wall_clock_latency_ms,
                        "summed_model_call_latency_ms": result.latency_ms,
                        "budget_exhaustion_reason": budget_exhausted
                        or result.metadata.get("budget_exhaustion_reason"),
                    },
                }
            )
            predictions.append(result)

            scores = evaluate_output(example, result.output)
            if isinstance(result, FailedPrediction):
                result.metadata["evaluation_status"] = "EVALUATED"
            if (
                scores.get("predicted_abstain") is not None
                and scores.get("should_abstain") is not None
            ):
                abstention_predicted.append(bool(scores["predicted_abstain"]))
                abstention_expected.append(bool(scores["should_abstain"]))
            scores.update(
                {
                    "input_tokens": float(result.input_tokens),
                    "output_tokens": float(result.output_tokens),
                    "total_tokens": float(result.total_tokens),
                    "model_calls": float(result.model_calls),
                    "latency_ms": float(result.latency_ms),
                    "wall_clock_strategy_latency_ms": float(wall_clock_latency_ms),
                    "estimated_cost_usd": float(result.estimated_cost_usd),
                    **{
                        key: float(result.metadata[key])
                        for key in (
                            "logical_model_calls",
                            "provider_attempts",
                            "successful_attempts",
                            "failed_attempts",
                            "unknown_usage_attempts",
                            "known_token_lower_bound",
                            "logical_call_latency_ms",
                        )
                        if key in result.metadata
                    },
                }
            )
            scores.update(
                derived_quality_metrics(
                    task_score=float(scores["task_score"] or 0.0),
                    total_tokens=result.total_tokens,
                    estimated_cost_usd=result.estimated_cost_usd,
                    latency_ms=result.latency_ms,
                )
            )
            row = {"example_id": example.id, "strategy": result.strategy, **scores}
            metric_rows.append(row)
            annotations = (
                []
                if isinstance(result, FailedPrediction)
                else annotate_failures(example, result, cast(dict[str, float], scores))
            )
            checkpoint_failures = []
            if isinstance(result, FailedPrediction):
                checkpoint_failures.extend(failure_annotations(result))
                failure_rows.extend(checkpoint_failures)
            for annotation in annotations:
                row = {
                    "run_id": run_id,
                    "example_id": annotation.example_id,
                    "failure_type": str(annotation.failure_type),
                    "source": annotation.source,
                    "rationale": annotation.rationale,
                    "metadata": annotation.metadata,
                }
                failure_rows.append(row)
                checkpoint_failures.append(row)
            store.checkpoint_example(result, scores, checkpoint_failures)
            if scientific_mode and ledger.unknown_usage_stop:
                raise UnknownUsageStop(
                    "UNKNOWN_USAGE_STOP: checkpoint retained; stop the run "
                    "to avoid overlapping unknown local server work"
                )

            spent_cost += result.estimated_cost_usd
            if max_cost_usd is not None and spent_cost > max_cost_usd:
                raise RuntimeError(f"max_cost_usd exceeded: {spent_cost:.6f} > {max_cost_usd:.6f}")

        aggregate_metrics = aggregate_numeric_metrics(metric_rows)
        all_abstention = _abstention_pairs_from_rows(metric_rows)
        if all_abstention is not None:
            aggregate_metrics.update(abstention_scores(*all_abstention))
        elif abstention_predicted:
            aggregate_metrics.update(abstention_scores(abstention_predicted, abstention_expected))
        for metric_name, metric_value in aggregate_metrics.items():
            store.insert_metric(
                run_id=run_id,
                example_id=None,
                metric_name=metric_name,
                metric_value=metric_value,
            )

        artifacts = write_run_artifacts(
            artifact_dir=artifact_dir,
            config=config,
            predictions=predictions,
            metric_rows=metric_rows,
            aggregate_metrics=aggregate_metrics,
            failure_rows=failure_rows,
            manifest={
                "experiment_id": experiment_id,
                "run_id": run_id,
                "experiment_name": experiment_name,
                "config_hash": config_hash,
                "dataset_hash": dataset_hash,
                "git_commit": git_commit,
                "python_version": platform.python_version(),
                "strategy": strategy.name,
                "examples": len(examples),
                "matrix": config.get("matrix", {}),
                "prompt_contract_version": CONTRACT_VERSION,
                "benchmark_version": config.get("benchmark_version", BENCHMARK_VERSION),
                "benchmark_hash": dataset_hash,
                "example_ids": [example.id for example in examples],
                "budget_policy": budget_policy.manifest_dict(),
                "budget_semantics_version": BUDGET_SEMANTICS_VERSION,
                "scientific_budget": budget_policy.scientific_budget.model_dump(mode="json"),
                "execution_safety_budget": budget_policy.execution_safety_budget.model_dump(
                    mode="json"
                ),
                "strategy_config": config.get("strategy", {}),
                "model": config.get("model", {}),
                "seed": config.get("seed"),
                "router_artifact_hash": _router_artifact_hash(config),
                "command_invocation": " ".join(sys.argv),
                "start_timestamp": now,
                "end_timestamp": datetime.now(UTC).isoformat(),
            },
            environment={
                "git_commit": git_commit,
                "git_dirty": _git_dirty(),
                "packages": _package_versions(),
                "operating_system": platform.platform(),
            },
            log_lines=[*log_lines, f"{datetime.now(UTC).isoformat()} completed run {run_id}"],
        )
        for kind, path in artifacts.items():
            store.insert_artifact(experiment_id=experiment_id, run_id=run_id, kind=kind, path=path)

        end_ts = datetime.now(UTC).isoformat()
        store.finish_run(
            run_id=run_id,
            status="COMPLETED",
            end_ts=end_ts,
            artifact_dir=str(artifact_dir),
        )
        return ExperimentRunResult(
            experiment_id=experiment_id,
            run_id=run_id,
            status="COMPLETED",
            artifact_dir=str(artifact_dir),
            aggregate_metrics=aggregate_metrics,
            examples=len(examples),
            config_hash=config_hash,
            dataset_hash=dataset_hash,
        )
    except Exception as exc:
        end_ts = datetime.now(UTC).isoformat()
        store.finish_run(
            run_id=run_id,
            status="STOPPED_UNKNOWN_USAGE" if isinstance(exc, UnknownUsageStop) else "FAILED",
            end_ts=end_ts,
            error=str(exc),
        )
        raise


def aggregate_numeric_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean every numeric metric across examples."""

    buckets: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key in {"example_id", "strategy"}:
                continue
            if isinstance(value, int | float):
                buckets.setdefault(key, []).append(float(value))
    return {f"mean_{key}": sum(values) / len(values) for key, values in sorted(buckets.items())}


def _usage_scores(result: PredictionResult) -> dict[str, float | None]:
    scores = {
        key: float(getattr(result, key))
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "model_calls",
            "latency_ms",
            "estimated_cost_usd",
        )
    }
    scores["wall_clock_strategy_latency_ms"] = float(
        result.metadata.get("wall_clock_strategy_latency_ms", result.latency_ms)
    )
    scores.update(
        {
            key: float(result.metadata[key])
            for key in (
                "logical_model_calls",
                "provider_attempts",
                "successful_attempts",
                "failed_attempts",
                "unknown_usage_attempts",
                "known_token_lower_bound",
                "logical_call_latency_ms",
            )
            if key in result.metadata
        }
    )
    return dict(scores)


def _prediction_rows_to_results(rows: list[dict[str, Any]]) -> list[PredictionResult]:
    results = []
    for row in rows:
        usage = json.loads(str(row["usage_json"]))
        metadata_payload = json.loads(str(row["metadata_json"]))
        candidates = metadata_payload.pop("candidates", [])
        results.append(
            (
                FailedPrediction
                if metadata_payload.get("result_kind") == "failed_prediction"
                else StrategyResult
            )(
                example_id=str(row["example_id"]),
                strategy=str(row["strategy"]),
                output=(
                    FailedOutput.model_validate(json.loads(str(row["output_json"])))
                    if metadata_payload.get("result_kind") == "failed_prediction"
                    else json.loads(str(row["output_json"]))
                ),
                **(
                    {"failed_outputs": metadata_payload.get("failed_outputs", [])}
                    if metadata_payload.get("result_kind") == "failed_prediction"
                    else {}
                ),
                confidence=float(row["confidence"]),
                candidates=candidates,
                model_calls=int(usage.get("model_calls", 0)),
                logical_model_calls=int(
                    usage.get("logical_model_calls", usage.get("model_calls", 0))
                ),
                provider_attempts=int(usage.get("provider_attempts", usage.get("model_calls", 0))),
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                latency_ms=float(usage.get("latency_ms", 0.0)),
                estimated_cost_usd=float(usage.get("estimated_cost_usd", 0.0)),
                metadata=metadata_payload,
            )
        )
    return results


def _example_metric_rows(
    store: SQLiteStore,
    run_id: str,
    strategy_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in store.get_run_metrics(run_id):
        example_id = row["example_id"]
        if example_id is None:
            continue
        metric_row = grouped.setdefault(
            str(example_id),
            {"example_id": str(example_id), "strategy": strategy_name},
        )
        metric_row[str(row["metric_name"])] = (
            None if row["metric_value"] is None else float(row["metric_value"])
        )
    return [grouped[key] for key in sorted(grouped)]


def _abstention_pairs_from_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[bool], list[bool]] | None:
    pairs = [
        (bool(row["predicted_abstain"]), bool(row["should_abstain"]))
        for row in rows
        if row.get("predicted_abstain") is not None and row.get("should_abstain") is not None
    ]
    if not pairs:
        return None
    predicted, expected = zip(*pairs, strict=True)
    return list(predicted), list(expected)


def _budget_exhausted_result(
    example: BenchmarkExample,
    strategy_name: str,
    ledger: BudgetLedger,
    reason: str,
) -> StrategyResult:
    completed = [
        r
        for r in ledger.records
        if r.role != "critic" and not r.normalized_error and r.metadata.get("structured_output")
    ]
    return StrategyResult(
        example_id=example.id,
        strategy=strategy_name,
        output=completed[-1].metadata["structured_output"] if completed else _empty_output(example),
        confidence=0.0,
        model_calls=ledger.model_calls,
        input_tokens=ledger.input_tokens,
        output_tokens=ledger.output_tokens,
        total_tokens=ledger.total_tokens,
        latency_ms=ledger.latency_ms,
        estimated_cost_usd=ledger.estimated_cost_usd,
        metadata={
            "budget_exhausted": not reason.startswith("UNKNOWN_USAGE_STOP"),
            "budget_exhaustion_reason": reason,
            **ledger.metadata_dict(),
        },
    )


def _empty_output(example: BenchmarkExample) -> dict[str, Any]:
    if str(example.task_type) in {"grounded_qa", "robustness"}:
        return {"answer": "", "citations": [], "abstain": True, "confidence": 0.0}
    if str(example.task_type) == "business_summarization":
        return {"summary": "", "decisions": [], "action_items": [], "risks": []}
    schema = example.gold.get("json_schema", {})
    required = schema.get("required", example.gold.get("schema_required", []))
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if isinstance(required, list) and isinstance(properties, dict):
        return {str(field): _empty_value(properties.get(field, {})) for field in required}
    return {}


def _empty_value(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return None
    type_spec = schema.get("type")
    allowed = type_spec if isinstance(type_spec, list) else [type_spec]
    if "string" in allowed:
        return ""
    if "integer" in allowed:
        return 0
    if "number" in allowed:
        return 0.0
    if "boolean" in allowed:
        return False
    if "array" in allowed:
        return []
    if "object" in allowed:
        return {}
    return None


def _config_with_resolved_budget(
    config: dict[str, Any],
    budget_policy: Any,
    *,
    max_examples: int | None = None,
) -> dict[str, Any]:
    resolved = cast(
        dict[str, Any],
        json.loads(json.dumps(config, ensure_ascii=False, default=str)),
    )
    resolved["budget"] = normalized_budget_config(budget_policy)
    resolved["budget_semantics_version"] = BUDGET_SEMANTICS_VERSION
    retry = resolved.setdefault("provider_retry", {})
    retry.setdefault("unknown_usage_policy", SCIENTIFIC_ATTEMPT_POLICY)
    resolved["budget_policy"] = budget_policy.manifest_dict()
    if max_examples is not None:
        resolved.setdefault("execution_overrides", {})
        resolved["execution_overrides"]["max_examples"] = max_examples
    return resolved


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return True
    return bool(result.stdout.strip())


def _package_versions() -> dict[str, str]:
    packages = ["pydantic", "numpy", "scikit-learn", "PyYAML", "ruff", "mypy", "pytest"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _router_artifact_hash(config: dict[str, Any]) -> str | None:
    artifact = dict(config.get("strategy", {})).get("router_artifact")
    if not artifact:
        return None
    path = Path(str(artifact))
    if path.is_file():
        return file_sha256(path)
    manifest = path / "training_manifest.json"
    return file_sha256(manifest) if manifest.exists() else None


def result_to_json(result: ExperimentRunResult | dict[str, Any]) -> str:
    if isinstance(result, ExperimentRunResult):
        return result.model_dump_json()
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
