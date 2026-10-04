"""Command line interface for benchmark validation and experiment runs."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from collectiveeval.artifact_verify import verify_run_artifacts
from collectiveeval.config import load_yaml_config, stable_config_hash
from collectiveeval.core import BenchmarkExample, ModelSpec, TaskType
from collectiveeval.costing import estimate_cost_for_config
from collectiveeval.datasets import (
    freeze_benchmark,
    inspect_benchmark_manifest,
    validate_benchmark_dir,
    validate_benchmark_file,
    verify_benchmark_manifest,
)
from collectiveeval.matrix import (
    generate_final_matrix,
    run_matrix,
    validate_manifest,
    write_matrix_configs,
)
from collectiveeval.providers import ProviderError, build_provider_registry
from collectiveeval.reporting import (
    compare_runs_report,
    evaluate_run_report,
    experiment_report,
    reproduction_plan,
)
from collectiveeval.router_training import train_learned_router_dev
from collectiveeval.runner import result_to_json, run_experiment_from_path
from collectiveeval.storage import SQLiteStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="collectiveeval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    validate = benchmark_sub.add_parser("validate")
    validate.add_argument("path", type=Path)
    freeze = benchmark_sub.add_parser("freeze")
    freeze.add_argument("--benchmark-dir", default=Path("data/benchmark_v1"), type=Path)
    freeze.add_argument("--version", default="collectiveeval-benchmark-v1")
    inspect = benchmark_sub.add_parser("inspect")
    inspect.add_argument("--manifest", default=Path("data/benchmark_v1/manifest.json"), type=Path)
    stats = benchmark_sub.add_parser("stats")
    stats.add_argument("path", default=Path("data/benchmark_v1"), nargs="?", type=Path)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--db", default="collectiveeval.sqlite3", type=Path)
    run.add_argument("--output-dir", default="runs", type=Path)
    run.add_argument("--max-cost-usd", type=float)
    run.add_argument("--max-examples", type=int)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("run_id")
    evaluate.add_argument("--db", default="collectiveeval.sqlite3", type=Path)
    evaluate.add_argument("--metric", default="task_score")

    compare = subparsers.add_parser("compare")
    compare.add_argument("run_ids", nargs="+")
    compare.add_argument("--db", default="collectiveeval.sqlite3", type=Path)
    compare.add_argument("--metric", default="task_score")

    report = subparsers.add_parser("report")
    report.add_argument("experiment_id")
    report.add_argument("--db", default="collectiveeval.sqlite3", type=Path)
    report.add_argument("--metric", default="task_score")

    reproduce = subparsers.add_parser("reproduce")
    reproduce.add_argument("experiment_id")
    reproduce.add_argument("--db", default="collectiveeval.sqlite3", type=Path)

    estimate = subparsers.add_parser("estimate-cost")
    estimate.add_argument("--config", required=True, type=Path)
    estimate.add_argument("--examples", type=int)

    matrix = subparsers.add_parser("matrix")
    matrix_sub = matrix.add_subparsers(dest="matrix_command", required=True)
    matrix_generate = matrix_sub.add_parser("generate")
    matrix_generate.add_argument("--base-config", required=True, type=Path)
    matrix_generate.add_argument("--output-dir", required=True, type=Path)
    matrix_generate.add_argument("--token-budget", action="append", type=int)
    matrix_generate.add_argument("--include-ablations", action="store_true")
    matrix_generate.add_argument("--ready-only", action="store_true")
    matrix_validate = matrix_sub.add_parser("validate")
    matrix_validate.add_argument("--manifest", required=True, type=Path)
    matrix_run = matrix_sub.add_parser("run")
    matrix_run.add_argument("--matrix", required=True, type=Path)
    matrix_run.add_argument("--db", default="collectiveeval.sqlite3", type=Path)
    matrix_run.add_argument("--output-dir", default="runs", type=Path)
    matrix_run.add_argument("--strategy")
    matrix_run.add_argument("--task-family")
    matrix_run.add_argument("--budget-tier")
    matrix_run.add_argument("--seed", type=int)
    matrix_run.add_argument("--max-examples", type=int)
    matrix_run.add_argument("--dry-run", action="store_true")
    matrix_run.add_argument("--resume", action="store_true")
    matrix_run.add_argument("--max-cost-usd", type=float)

    provider = subparsers.add_parser("provider")
    provider_sub = provider.add_subparsers(dest="provider_command", required=True)
    smoke = provider_sub.add_parser("smoke-test")
    smoke.add_argument("--provider", required=True)
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--base-url")
    smoke.add_argument("--api-key-env")
    smoke.add_argument("--mock-mode", default="fixture")

    artifacts = subparsers.add_parser("artifacts")
    artifacts_sub = artifacts.add_subparsers(dest="artifacts_command", required=True)
    artifacts_verify = artifacts_sub.add_parser("verify")
    artifacts_verify.add_argument("run_id")
    artifacts_verify.add_argument("--db", default="collectiveeval.sqlite3", type=Path)

    router = subparsers.add_parser("router")
    router_sub = router.add_subparsers(dest="router_command", required=True)
    router_train = router_sub.add_parser("train-dev")
    router_train.add_argument("--dataset", required=True, type=Path)
    router_train.add_argument("--artifact-dir", required=True, type=Path)
    router_train.add_argument("--max-examples", type=int)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "benchmark" and args.benchmark_command == "validate":
        path = Path(args.path)
        if path.is_dir():
            payload = validate_benchmark_dir(path)
            manifest_path = path / "manifest.json"
            if manifest_path.exists():
                payload["manifest"] = verify_benchmark_manifest(manifest_path)
        else:
            payload = validate_benchmark_file(path)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "benchmark" and args.benchmark_command == "freeze":
        print(
            json.dumps(
                freeze_benchmark(args.benchmark_dir, version=args.version),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "benchmark" and args.benchmark_command == "inspect":
        print(
            json.dumps(
                inspect_benchmark_manifest(args.manifest),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "benchmark" and args.benchmark_command == "stats":
        path = Path(args.path)
        payload = validate_benchmark_dir(path) if path.is_dir() else validate_benchmark_file(path)
        print(json.dumps(payload["counts"], ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "run":
        result = run_experiment_from_path(
            args.config,
            db_path=args.db,
            output_dir=args.output_dir,
            max_examples=args.max_examples,
            max_cost_usd=args.max_cost_usd,
            dry_run=args.dry_run,
        )
        print(result_to_json(result))
        return 0

    if args.command == "evaluate":
        store = SQLiteStore(args.db)
        print(
            json.dumps(
                evaluate_run_report(store, args.run_id, metric=args.metric),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "compare":
        store = SQLiteStore(args.db)
        print(
            json.dumps(
                compare_runs_report(store, args.run_ids, metric=args.metric),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "report":
        store = SQLiteStore(args.db)
        print(
            json.dumps(
                experiment_report(store, args.experiment_id, metric=args.metric),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "reproduce":
        store = SQLiteStore(args.db)
        print(
            json.dumps(
                reproduction_plan(store, args.experiment_id),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "estimate-cost":
        config = load_yaml_config(args.config)
        payload = {
            "config_hash": stable_config_hash(config),
            **estimate_cost_for_config(config, examples=args.examples),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "matrix" and args.matrix_command == "generate":
        base_config = load_yaml_config(args.base_config)
        entries = generate_final_matrix(
            base_config,
            token_budgets=args.token_budget or [8000, 12000],
            include_ablations=args.include_ablations,
            ready_only=args.ready_only,
        )
        manifest = write_matrix_configs(entries, args.output_dir)
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "matrix" and args.matrix_command == "validate":
        print(json.dumps(validate_manifest(args.manifest), ensure_ascii=False, sort_keys=True))
        return 0

    if args.command == "matrix" and args.matrix_command == "run":
        print(
            json.dumps(
                run_matrix(
                    args.matrix,
                    db_path=args.db,
                    output_dir=args.output_dir,
                    strategy=args.strategy,
                    task_family=args.task_family,
                    budget_tier=args.budget_tier,
                    seed=args.seed,
                    max_examples=args.max_examples,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    max_cost_usd=args.max_cost_usd,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "provider" and args.provider_command == "smoke-test":
        result = _provider_smoke_test(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            mock_mode=args.mock_mode,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["ok"] else 2

    if args.command == "artifacts" and args.artifacts_command == "verify":
        result = verify_run_artifacts(SQLiteStore(args.db), args.run_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result["ok"] else 2

    if args.command == "router" and args.router_command == "train-dev":
        from collectiveeval.datasets import load_jsonl

        examples = load_jsonl(args.dataset)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        result = train_learned_router_dev(examples, artifact_dir=args.artifact_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


def _provider_smoke_test(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    api_key_env: str | None,
    mock_mode: str,
) -> dict[str, object]:
    async def run() -> dict[str, object]:
        from collectiveeval.budget import BudgetLedger, InferenceBudget
        from collectiveeval.context import StrategyContext

        config = {
            "model": {
                "provider": provider,
                "name": model,
                "base_url": base_url,
                "api_key_env": api_key_env,
                "mock_mode": mock_mode,
                "max_tokens": 32,
            },
            "providers": {"mock": {"mock_mode": mock_mode}},
        }
        example = BenchmarkExample(
            id="provider-smoke",
            task_type=TaskType.GROUNDED_QA,
            input={
                "source_text": "契約書: 通知は30日前。",
                "question": "通知は何日前ですか。",
                "evidence_units": [{"id": "smoke-1", "text": "契約書: 通知は30日前。"}],
            },
            gold={"answerable": True, "answer": "30日前", "evidence": ["smoke-1"]},
            metadata={
                "source": "synthetic_provider_smoke",
                "difficulty": "easy",
                "tags": ["smoke"],
                "split": "mock",
            },
        )
        registry = build_provider_registry(config)
        context = StrategyContext(registry, BudgetLedger(InferenceBudget(max_calls=1)))
        spec = ModelSpec(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            mock_mode=mock_mode,
            max_tokens=32,
        )
        try:
            response = await context.call_model(
                example=example,
                model=spec,
                strategy="provider_smoke_test",
                role="generator",
            )
        except ProviderError as exc:
            return {"ok": False, "error_type": str(exc.error_type), "message": str(exc)}
        except Exception as exc:
            return {"ok": False, "error_type": "UNKNOWN_PROVIDER_ERROR", "message": str(exc)}
        return {
            "ok": True,
            "provider": provider,
            "model": model,
            "usage_source": str(response.usage_source),
            "finish_reason": response.finish_reason,
        }

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
