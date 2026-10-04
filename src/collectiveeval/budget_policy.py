"""Scientific budget policy resolution for matched-budget experiments."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from collectiveeval.budget import InferenceBudget


class ScientificBudgetMode(StrEnum):
    MATCHED_TOKENS = "MATCHED_TOKENS"
    MATCHED_CALLS = "MATCHED_CALLS"
    MATCHED_ESTIMATED_COST = "MATCHED_ESTIMATED_COST"


class ScientificBudgetScope(StrEnum):
    PER_EXAMPLE = "PER_EXAMPLE"


class BudgetAccountingSource(StrEnum):
    PROVIDER_REPORTED_OR_ESTIMATED = "PROVIDER_REPORTED_OR_ESTIMATED"


class BudgetTier(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


CANONICAL_BUDGET_TIERS: dict[BudgetTier, dict[str, int | float | None]] = {
    BudgetTier.SMALL: {
        "max_calls": 2,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "max_total_tokens": 4000,
        "max_cost_usd": None,
    },
    BudgetTier.MEDIUM: {
        "max_calls": 4,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "max_total_tokens": 8000,
        "max_cost_usd": None,
    },
    BudgetTier.LARGE: {
        "max_calls": 8,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "max_total_tokens": 16000,
        "max_cost_usd": None,
    },
}


class ScientificBudget(BaseModel):
    """Resolved per-example scientific treatment budget."""

    scope: ScientificBudgetScope = ScientificBudgetScope.PER_EXAMPLE
    mode: ScientificBudgetMode = ScientificBudgetMode.MATCHED_TOKENS
    tier: BudgetTier = BudgetTier.MEDIUM
    max_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_estimated_cost_usd: float | None = None
    accounting_source: BudgetAccountingSource = (
        BudgetAccountingSource.PROVIDER_REPORTED_OR_ESTIMATED
    )

    @model_validator(mode="after")
    def validate_mode_has_ceiling(self) -> ScientificBudget:
        if self.mode == ScientificBudgetMode.MATCHED_TOKENS and self.max_total_tokens is None:
            raise ValueError("MATCHED_TOKENS requires max_total_tokens")
        if self.mode == ScientificBudgetMode.MATCHED_CALLS and self.max_calls is None:
            raise ValueError("MATCHED_CALLS requires max_calls")
        if (
            self.mode == ScientificBudgetMode.MATCHED_ESTIMATED_COST
            and self.max_estimated_cost_usd is None
        ):
            raise ValueError("MATCHED_ESTIMATED_COST requires max_estimated_cost_usd")
        return self

    def to_inference_budget(self) -> InferenceBudget:
        return InferenceBudget(
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            max_total_tokens=self.max_total_tokens,
            max_calls=self.max_calls,
            max_cost_usd=self.max_estimated_cost_usd,
        )


class ExecutionSafetyBudget(BaseModel):
    """Run-level execution safety controls, not scientific treatment variables."""

    max_cost_usd: float | None = None


class BudgetPolicy(BaseModel):
    scientific_budget: ScientificBudget
    execution_safety_budget: ExecutionSafetyBudget = Field(default_factory=ExecutionSafetyBudget)

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "scientific_budget": self.scientific_budget.model_dump(mode="json"),
            "execution_safety_budget": self.execution_safety_budget.model_dump(mode="json"),
        }


def resolve_budget_policy(
    config: dict[str, Any],
    *,
    run_level_max_cost_usd: float | None = None,
) -> BudgetPolicy:
    """Resolve legacy and Phase 7 budget config into a typed policy."""

    raw_budget = dict(config.get("budget", {}))
    raw_policy = dict(config.get("budget_policy", {}))
    tier = BudgetTier(str(config.get("budget_tier") or raw_policy.get("budget_tier") or "medium"))
    tier_values = dict(CANONICAL_BUDGET_TIERS[tier])
    tier_values.update({key: value for key, value in raw_budget.items() if value is not None})
    mode = ScientificBudgetMode(
        str(raw_policy.get("mode") or raw_policy.get("scientific_budget_mode") or "MATCHED_TOKENS")
    )
    if "max_estimated_cost_usd" not in tier_values and "max_cost_usd" in tier_values:
        tier_values["max_estimated_cost_usd"] = tier_values["max_cost_usd"]
    scientific_budget = ScientificBudget(
        mode=mode,
        tier=tier,
        max_calls=_optional_int(tier_values.get("max_calls")),
        max_input_tokens=_optional_int(tier_values.get("max_input_tokens")),
        max_output_tokens=_optional_int(tier_values.get("max_output_tokens")),
        max_total_tokens=_optional_int(tier_values.get("max_total_tokens")),
        max_estimated_cost_usd=_optional_float(tier_values.get("max_estimated_cost_usd")),
    )
    return BudgetPolicy(
        scientific_budget=scientific_budget,
        execution_safety_budget=ExecutionSafetyBudget(max_cost_usd=run_level_max_cost_usd),
    )


def normalized_budget_config(policy: BudgetPolicy) -> dict[str, Any]:
    budget = policy.scientific_budget
    return {
        "max_calls": budget.max_calls,
        "max_input_tokens": budget.max_input_tokens,
        "max_output_tokens": budget.max_output_tokens,
        "max_total_tokens": budget.max_total_tokens,
        "max_cost_usd": budget.max_estimated_cost_usd,
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
