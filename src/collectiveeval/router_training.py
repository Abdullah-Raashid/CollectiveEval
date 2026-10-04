"""DEV-only learned router dataset construction and training."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from collectiveeval.budget import BudgetLedger, InferenceBudget
from collectiveeval.context import StrategyContext
from collectiveeval.core import BenchmarkExample, ModelSpec
from collectiveeval.metrics import score_prediction
from collectiveeval.providers import MockProvider, ModelProvider
from collectiveeval.router import (
    HeuristicRouter,
    LearnedRouter,
    RouterFeatureRow,
    RouterTrainingTarget,
    RouterUtilityConfig,
    construct_router_target,
    router_feature_row_from_initial,
    stable_json_hash,
    validate_feature_schema,
)
from collectiveeval.strategies import CriticReviser, MultiAgentDebate, SingleAgent


def deterministic_router_split(example_id: str) -> str:
    bucket = int(stable_json_hash({"example_id": example_id})[:8], 16) % 10
    return "router_validation" if bucket >= 7 else "router_train"


async def build_router_training_dataset(
    examples: list[BenchmarkExample],
    *,
    utility_config: RouterUtilityConfig | None = None,
) -> tuple[list[RouterFeatureRow], list[RouterTrainingTarget], dict[str, str]]:
    validate_feature_schema()
    if any(example.metadata.get("split") == "test" for example in examples):
        raise ValueError("router training dataset construction cannot use test examples")

    features: list[RouterFeatureRow] = []
    targets: list[RouterTrainingTarget] = []
    assignments: dict[str, str] = {}
    initial_strategy = SingleAgent(model=ModelSpec(model="mock-weak"))
    escalation_strategy = CriticReviser(
        generator=ModelSpec(model="mock-weak"),
        reviser=ModelSpec(model="mock-reviser"),
    )
    debate_strategy = MultiAgentDebate(
        agents=2,
        rounds=1,
        model=ModelSpec(model="mock-weak"),
    )
    providers: dict[str, ModelProvider] = {"mock": MockProvider(mock_mode="gold_fixture")}
    budget = InferenceBudget(max_calls=6, max_total_tokens=12000)

    for example in examples:
        split = deterministic_router_split(example.id)
        assignments[example.id] = split
        initial_context = StrategyContext(
            providers,
            BudgetLedger(budget=budget, run_id=f"router-initial-{example.id}"),
        )
        escalation_context = StrategyContext(
            providers,
            BudgetLedger(budget=budget, run_id=f"router-escalated-{example.id}"),
        )
        debate_context = StrategyContext(
            providers,
            BudgetLedger(
                budget=InferenceBudget(max_calls=8, max_total_tokens=16000),
                run_id=f"router-debate-{example.id}",
            ),
        )
        initial = await initial_strategy.run(example, initial_context)
        escalated = await escalation_strategy.run(example, escalation_context)
        debate = await debate_strategy.run(example, debate_context)
        initial_quality = score_prediction(example, initial.output)["task_score"]
        escalated_quality = score_prediction(example, escalated.output)["task_score"]
        debate_quality = score_prediction(example, debate.output)["task_score"]
        features.append(
            router_feature_row_from_initial(
                example,
                initial.output,
                confidence=initial.confidence,
                model_id="mock/mock-weak",
                split=split,
            )
        )
        targets.append(
            construct_router_target(
                example_id=example.id,
                initial_quality=initial_quality,
                escalated_quality=escalated_quality,
                initial_tokens=initial.total_tokens,
                escalated_tokens=escalated.total_tokens,
                initial_calls=initial.model_calls,
                escalated_calls=escalated.model_calls,
                initial_cost=initial.estimated_cost_usd,
                escalated_cost=escalated.estimated_cost_usd,
                debate_quality=debate_quality,
                debate_tokens=debate.total_tokens,
                debate_calls=debate.model_calls,
                debate_cost=debate.estimated_cost_usd,
                config=utility_config,
            )
        )
    return features, targets, assignments


def train_learned_router_dev(
    examples: list[BenchmarkExample],
    *,
    artifact_dir: str | Path,
    utility_config: RouterUtilityConfig | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Train learned router only on DEV/mock rows and persist a local artifact bundle."""

    if any(example.metadata.get("split") == "test" for example in examples):
        raise ValueError("learned router training cannot use test examples")
    features, targets, assignments = asyncio.run(
        build_router_training_dataset(examples, utility_config=utility_config)
    )
    train_rows = [row for row in features if row.split == "router_train"]
    target_by_id = {target.example_id: target for target in targets}
    train_targets = [target_by_id[row.example_id] for row in train_rows]
    router = LearnedRouter(threshold=threshold)
    router.train_feature_rows(train_rows, train_targets)

    artifact_path = Path(artifact_dir)
    training_manifest = {
        "router_split_policy": "stable_hash_mod_10_validation_buckets_7_9",
        "assignments": assignments,
        "feature_rows_hash": stable_json_hash([row.model_dump(mode="json") for row in features]),
        "target_rows_hash": stable_json_hash(
            [target.model_dump(mode="json") for target in targets]
        ),
        "utility_config": (utility_config or RouterUtilityConfig()).model_dump(mode="json"),
        "threshold": threshold,
        "examples": len(examples),
        "train_examples": len(train_rows),
        "validation_examples": sum(split == "router_validation" for split in assignments.values()),
    }
    router.save_artifact(artifact_path, training_manifest=training_manifest)
    metrics = evaluate_router(router, features, targets)
    (artifact_path / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (artifact_path / "subsplits.json").write_text(
        json.dumps(assignments, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_router_report(artifact_path, training_manifest, metrics, router)
    return {
        "artifact_dir": str(artifact_path),
        "training_manifest": training_manifest,
        "metrics": metrics,
    }


def evaluate_router(
    router: LearnedRouter,
    features: list[RouterFeatureRow],
    targets: list[RouterTrainingTarget],
) -> dict[str, Any]:
    target_by_id = {target.example_id: target for target in targets}
    validation = [row for row in features if row.split == "router_validation"]
    if not validation:
        validation = features
    tp = fp = tn = fn = 0
    probabilities: dict[str, float] = {}
    for row in validation:
        probability = router.escalation_probability_from_features(row.features)
        probabilities[row.example_id] = probability
        predicted = probability >= router.threshold
        actual = bool(target_by_id[row.example_id].label)
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    total = len(validation)
    return {
        "validation_examples": total,
        "escalation_precision": precision,
        "escalation_recall": recall,
        "escalation_f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "escalation_rate": (tp + fp) / total if total else 0.0,
        "unnecessary_escalation_rate": fp / total if total else 0.0,
        "missed_beneficial_escalation_rate": fn / total if total else 0.0,
        "per_example_probability": probabilities,
        "system_behavior": _system_behavior_metrics(router, validation, target_by_id),
    }


def _system_behavior_metrics(
    router: LearnedRouter,
    rows: list[RouterFeatureRow],
    target_by_id: dict[str, RouterTrainingTarget],
) -> dict[str, Any]:
    policies: dict[str, list[dict[str, float]]] = {
        "always_single": [],
        "always_critic_reviser": [],
        "heuristic_router": [],
        "learned_router": [],
    }
    if all(target_by_id[row.example_id].debate_quality is not None for row in rows):
        policies["always_debate"] = []

    for row in rows:
        target = target_by_id[row.example_id]
        policies["always_single"].append(_outcome(target, escalation="single"))
        policies["always_critic_reviser"].append(_outcome(target, escalation="critic"))
        if "always_debate" in policies:
            policies["always_debate"].append(_outcome(target, escalation="debate"))
        heuristic = HeuristicRouter()
        policies["heuristic_router"].append(
            _outcome(
                target,
                escalation="critic" if _heuristic_escalates(row.features, heuristic) else "single",
            )
        )
        learned_escalates = (
            router.escalation_probability_from_features(row.features) >= router.threshold
        )
        policies["learned_router"].append(
            _outcome(target, escalation="critic" if learned_escalates else "single")
        )

    summary = {name: _summarize_policy(outcomes) for name, outcomes in policies.items()}
    expensive = summary["always_critic_reviser"]
    for metrics in summary.values():
        metrics["quality_retained_vs_always_critic_reviser"] = _ratio(
            metrics["average_quality"],
            expensive["average_quality"],
        )
        metrics["token_savings_vs_always_critic_reviser"] = _savings_ratio(
            metrics["average_tokens"],
            expensive["average_tokens"],
        )
        metrics["call_savings_vs_always_critic_reviser"] = _savings_ratio(
            metrics["average_calls"],
            expensive["average_calls"],
        )
        metrics["utility_retained_vs_always_critic_reviser"] = _ratio(
            metrics["average_utility_gain"],
            expensive["average_utility_gain"],
        )
    return summary


def _outcome(target: RouterTrainingTarget, *, escalation: str) -> dict[str, float]:
    if escalation == "critic":
        return {
            "quality": target.escalated_quality,
            "tokens": float(target.escalated_tokens),
            "calls": float(target.escalated_calls),
            "cost": target.escalated_cost,
            "utility_gain": target.utility_gain,
        }
    if escalation == "debate" and target.debate_quality is not None:
        return {
            "quality": target.debate_quality,
            "tokens": float(target.debate_tokens or 0),
            "calls": float(target.debate_calls or 0),
            "cost": float(target.debate_cost or 0.0),
            "utility_gain": target.debate_quality - target.initial_quality,
        }
    return {
        "quality": target.initial_quality,
        "tokens": float(target.initial_tokens),
        "calls": float(target.initial_calls),
        "cost": target.initial_cost,
        "utility_gain": 0.0,
    }


def _summarize_policy(outcomes: list[dict[str, float]]) -> dict[str, float]:
    if not outcomes:
        return {
            "average_quality": 0.0,
            "average_tokens": 0.0,
            "average_calls": 0.0,
            "average_estimated_cost": 0.0,
            "average_utility_gain": 0.0,
        }
    return {
        "average_quality": _mean(row["quality"] for row in outcomes),
        "average_tokens": _mean(row["tokens"] for row in outcomes),
        "average_calls": _mean(row["calls"] for row in outcomes),
        "average_estimated_cost": _mean(row["cost"] for row in outcomes),
        "average_utility_gain": _mean(row["utility_gain"] for row in outcomes),
    }


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _savings_ratio(actual: float, baseline: float) -> float:
    return 0.0 if baseline == 0 else (baseline - actual) / baseline


def _heuristic_escalates(features: dict[str, float], router: HeuristicRouter) -> bool:
    score = 0.0
    score += 0.30 if not bool(features["parser_success"]) else 0.0
    score += 0.35 if not bool(features["schema_valid"]) else 0.0
    score += (
        0.25 if bool(features["task_grounded_qa"]) and int(features["citation_count"]) == 0 else 0.0
    )
    score += 0.20 * (1.0 - features["citation_coverage"])
    score += 0.30 * (1.0 - features["initial_confidence"])
    score += (
        0.10 if bool(features["abstain_flag"]) and features["initial_confidence"] < 0.7 else 0.0
    )
    score += 0.10 if features["input_length"] > 2000 else 0.0
    score += 0.15 if bool(features["difficulty_hard"]) else 0.0
    score += 0.10 * min(3.0, features["validator_failures"])
    score += 0.30 * features["sample_disagreement"]
    return min(1.0, score) >= router.critic_threshold


def write_router_report(
    artifact_dir: Path,
    training_manifest: dict[str, Any],
    metrics: dict[str, Any],
    router: LearnedRouter,
) -> None:
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    coefficient_rows = router.coefficients()
    lines = [
        "# Router DEV Report",
        "",
        "This is MOCK / INFRASTRUCTURE VALIDATION ONLY. It is not scientific evidence.",
        "",
        "## Training Setup",
        "",
        f"- Artifact: `{artifact_dir}`",
        f"- Examples: {training_manifest['examples']}",
        f"- Train examples: {training_manifest['train_examples']}",
        f"- Validation examples: {training_manifest['validation_examples']}",
        f"- DEV split policy: `{training_manifest['router_split_policy']}`",
        "- Feature construction excludes gold-derived evaluation features.",
        "",
        "## Feature Schema",
        "",
        "The learned router uses only inference-time fields available after the initial "
        "cheap answer and before escalation.",
        "",
        "| Feature |",
        "|---|",
    ]
    for feature in coefficient_rows:
        lines.append(f"| {feature['feature']} |")
    if not coefficient_rows:
        lines.extend(["| No trained coefficients available |"])
    lines.extend(
        [
            "",
            "Gold-derived labels and scores are stored only in training targets.",
            "",
            "## Target Definition",
            "",
            "Utility gain = quality_gain - lambda_tokens * normalized_extra_tokens "
            "- lambda_calls * normalized_extra_calls - lambda_cost * normalized_extra_cost.",
            "",
            "Targets are derived from DEV gold labels after observing initial and escalation "
            "outcomes. They are not router input features.",
            "",
            "## Classification Metrics",
            "",
            "```json",
            json.dumps(
                {key: value for key, value in metrics.items() if key != "system_behavior"},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## System Behavior",
            "",
            "```json",
            json.dumps(
                metrics.get("system_behavior", {}),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Coefficients",
            "",
            "| Feature | Coefficient | Normalized importance |",
            "|---|---:|---:|",
        ]
    )
    for row in coefficient_rows:
        lines.append(
            f"| {row['feature']} | {row['coefficient']:.6f} | {row['normalized_importance']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Failure Analysis",
            "",
            "- Router over-escalation is counted as validation false positives.",
            "- Router under-escalation is counted as validation false negatives.",
            "- Detailed qualitative review remains required before real-model claims.",
            "",
            "## Limitations",
            "",
            "- DEV-only mock-provider training is for pipeline validation.",
            "- AlwaysDebate uses the configured Phase 7 mock debate baseline, not a final "
            "held-out experiment.",
            "- Coefficients are interpretable but not causal.",
            "- Test split is not used.",
        ]
    )
    (report_dir / "router_dev_report.md").write_text("\n".join(lines), encoding="utf-8")
