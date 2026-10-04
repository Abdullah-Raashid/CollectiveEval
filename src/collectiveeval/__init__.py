"""CollectiveEval public API."""

from collectiveeval.budget import BudgetExceeded, BudgetLedger, InferenceBudget
from collectiveeval.core import BenchmarkExample, ModelSpec, StrategyResult, TaskType
from collectiveeval.matrix import generate_final_matrix, validate_matched_budgets
from collectiveeval.providers import MockProvider, ProviderRegistry
from collectiveeval.reporting import compare_runs_report, evaluate_run_report, experiment_report
from collectiveeval.runner import ExperimentRunResult, run_experiment, run_experiment_from_path
from collectiveeval.strategies import (
    AdaptiveRouterStrategy,
    CriticReviser,
    HeterogeneousPanel,
    MultiAgentDebate,
    SelfConsistency,
    SingleAgent,
)

__all__ = [
    "AdaptiveRouterStrategy",
    "BenchmarkExample",
    "BudgetExceeded",
    "BudgetLedger",
    "CriticReviser",
    "HeterogeneousPanel",
    "InferenceBudget",
    "MockProvider",
    "ModelSpec",
    "MultiAgentDebate",
    "ProviderRegistry",
    "ExperimentRunResult",
    "generate_final_matrix",
    "validate_matched_budgets",
    "compare_runs_report",
    "evaluate_run_report",
    "experiment_report",
    "SelfConsistency",
    "SingleAgent",
    "StrategyResult",
    "TaskType",
    "run_experiment",
    "run_experiment_from_path",
]
