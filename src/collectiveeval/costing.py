"""Deterministic cost estimation for experiment configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from collectiveeval.datasets import load_jsonl
from collectiveeval.providers import estimate_tokens
from collectiveeval.strategy_factory import budget_from_config, model_from_config


def estimate_cost_for_config(
    config: dict[str, Any],
    *,
    examples: int | None = None,
) -> dict[str, Any]:
    """Estimate worst-case calls, tokens, and cost without making provider calls."""

    dataset_path = Path(config["dataset"]["path"])
    benchmark = load_jsonl(dataset_path)
    selected = benchmark[:examples] if examples is not None else benchmark
    calls_per_example = _configured_calls(config)
    budget = budget_from_config(config.get("budget"))
    if budget.max_calls is not None:
        calls_per_example = min(calls_per_example, budget.max_calls)

    model = model_from_config(config.get("model"))
    strategy_config = dict(config.get("strategy", {}))
    output_tokens_per_call = int(
        config.get("estimate", {}).get(
            "output_tokens_per_call",
            _default_output_tokens(calls_per_example, budget.max_output_tokens),
        )
    )

    input_tokens = sum(estimate_tokens(example.model_dump(mode="json")) for example in selected)
    estimated_input_tokens = input_tokens * calls_per_example
    estimated_output_tokens = len(selected) * calls_per_example * output_tokens_per_call
    estimated_cost = (
        estimated_input_tokens * model.input_cost_per_1k
        + estimated_output_tokens * model.output_cost_per_1k
    ) / 1000

    return {
        "examples": len(selected),
        "strategy_type": str(strategy_config.get("type", "single_agent")),
        "calls_per_example": calls_per_example,
        "estimated_model_calls": len(selected) * calls_per_example,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
        "estimated_cost_usd": estimated_cost,
        "input_cost_per_1k": model.input_cost_per_1k,
        "output_cost_per_1k": model.output_cost_per_1k,
    }


def _configured_calls(config: dict[str, Any]) -> int:
    strategy = dict(config.get("strategy", {}))
    strategy_type = str(strategy.get("type", "single_agent")).lower()
    if strategy_type in {"single", "single_agent"}:
        return 1
    if strategy_type in {"self_consistency", "self-consistency"}:
        return int(strategy.get("k", 4))
    if strategy_type in {"debate", "multi_agent_debate", "multi-agent-debate"}:
        return int(strategy.get("agents", 3)) * int(strategy.get("rounds", 2))
    if strategy_type in {"critic", "critic_reviser", "critic-reviser"}:
        return 3
    if strategy_type in {"panel", "heterogeneous_panel", "heterogeneous-panel"}:
        models = strategy.get("models")
        return len(models) if isinstance(models, list) and models else 2
    if strategy_type in {"adaptive", "adaptive_router", "adaptive-router"}:
        return 1 + int(strategy.get("escalation_calls", 4))
    return 1


def _default_output_tokens(calls_per_example: int, max_output_tokens: int | None) -> int:
    if max_output_tokens is None:
        return 256
    return max(1, max_output_tokens // max(1, calls_per_example))
