"""Bootstrap summaries and paired comparisons for experiment analysis."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryStats:
    """Descriptive statistics with a bootstrap confidence interval."""

    n: int
    mean: float
    median: float
    std: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }


@dataclass(frozen=True)
class PairedComparison:
    """Paired bootstrap result for contender minus baseline."""

    metric: str
    common_examples: int
    baseline_mean: float
    contender_mean: float
    mean_difference: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "metric": self.metric,
            "common_examples": self.common_examples,
            "baseline_mean": self.baseline_mean,
            "contender_mean": self.contender_mean,
            "mean_difference": self.mean_difference,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }


def summary_stats(
    values: list[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> SummaryStats:
    """Compute mean/median/std and a nonparametric bootstrap CI for the mean."""

    if not values:
        return SummaryStats(n=0, mean=0.0, median=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0)

    sorted_values = sorted(values)
    mean = sum(values) / len(values)
    median = _median(sorted_values)
    std = _std(values, mean)
    if len(values) == 1 or n_bootstrap <= 0:
        return SummaryStats(
            n=len(values),
            mean=mean,
            median=median,
            std=std,
            ci_lower=mean,
            ci_upper=mean,
        )

    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot_means.append(sum(sample) / len(sample))
    lower, upper = percentile_interval(boot_means, confidence=confidence)
    return SummaryStats(
        n=len(values),
        mean=mean,
        median=median,
        std=std,
        ci_lower=lower,
        ci_upper=upper,
    )


def paired_bootstrap_comparison(
    *,
    baseline: dict[str, float],
    contender: dict[str, float],
    metric: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Compare two runs on shared examples using contender - baseline."""

    common_ids = sorted(set(baseline) & set(contender))
    if not common_ids:
        return PairedComparison(
            metric=metric,
            common_examples=0,
            baseline_mean=0.0,
            contender_mean=0.0,
            mean_difference=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
        )

    differences = [contender[example_id] - baseline[example_id] for example_id in common_ids]
    baseline_values = [baseline[example_id] for example_id in common_ids]
    contender_values = [contender[example_id] for example_id in common_ids]
    mean_difference = sum(differences) / len(differences)

    if len(differences) == 1 or n_bootstrap <= 0:
        lower = upper = mean_difference
    else:
        rng = random.Random(seed)
        boot_diffs = []
        for _ in range(n_bootstrap):
            sample = [differences[rng.randrange(len(differences))] for _ in differences]
            boot_diffs.append(sum(sample) / len(sample))
        lower, upper = percentile_interval(boot_diffs, confidence=confidence)

    return PairedComparison(
        metric=metric,
        common_examples=len(common_ids),
        baseline_mean=sum(baseline_values) / len(baseline_values),
        contender_mean=sum(contender_values) / len(contender_values),
        mean_difference=mean_difference,
        ci_lower=lower,
        ci_upper=upper,
    )


def percentile_interval(values: list[float], *, confidence: float) -> tuple[float, float]:
    """Return a percentile interval without claiming significance."""

    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    alpha = 1.0 - confidence
    lower_index = max(0, math.floor((alpha / 2) * (len(ordered) - 1)))
    upper_index = min(len(ordered) - 1, math.ceil((1 - alpha / 2) * (len(ordered) - 1)))
    return ordered[lower_index], ordered[upper_index]


def _median(sorted_values: list[float]) -> float:
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
