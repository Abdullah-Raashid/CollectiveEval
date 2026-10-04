"""Deterministic failure taxonomy annotations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from collectiveeval.core import BenchmarkExample, StrategyResult, TaskType
from collectiveeval.metrics import score_prediction
from collectiveeval.validation import validate_output_schema


class FailureType(StrEnum):
    UNKNOWN_PROVIDER_USAGE = "UNKNOWN_PROVIDER_USAGE"
    INTERRUPTED_TRAJECTORY = "INTERRUPTED_TRAJECTORY"
    HALLUCINATION = "HALLUCINATION"
    MISSED_EVIDENCE = "MISSED_EVIDENCE"
    WRONG_CITATION = "WRONG_CITATION"
    OVER_ABSTENTION = "OVER_ABSTENTION"
    UNDER_ABSTENTION = "UNDER_ABSTENTION"
    SCHEMA_FAILURE = "SCHEMA_FAILURE"
    NUMERIC_ERROR = "NUMERIC_ERROR"
    DATE_NORMALIZATION_ERROR = "DATE_NORMALIZATION_ERROR"
    NEGATION_ERROR = "NEGATION_ERROR"
    PEER_ERROR_PROPAGATION = "PEER_ERROR_PROPAGATION"
    MAJORITY_WRONG = "MAJORITY_WRONG"
    CRITIC_REGRESSION = "CRITIC_REGRESSION"
    ROUTER_UNDER_ESCALATION = "ROUTER_UNDER_ESCALATION"
    ROUTER_OVER_ESCALATION = "ROUTER_OVER_ESCALATION"


class FailureAnnotation(BaseModel):
    """One automatic or manual failure label."""

    example_id: str
    failure_type: FailureType
    source: str = "automatic"
    rationale: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def annotate_failures(
    example: BenchmarkExample,
    result: StrategyResult,
    scores: dict[str, float],
) -> list[FailureAnnotation]:
    """Annotate deterministic failures without storing hidden reasoning."""

    annotations: list[FailureAnnotation] = []
    output = result.output
    schema_valid, schema_issues = validate_output_schema(example, output)
    task_score = scores.get("task_score", 0.0)

    def add(failure_type: FailureType, rationale: str, **metadata: Any) -> None:
        annotations.append(
            FailureAnnotation(
                example_id=example.id,
                failure_type=failure_type,
                rationale=rationale,
                metadata=metadata,
            )
        )

    if result.metadata.get("unknown_usage_stop"):
        add(
            FailureType.UNKNOWN_PROVIDER_USAGE,
            "Sent provider attempt returned no authoritative usage; "
            "trajectory stopped conservatively.",
        )
    if result.metadata.get("interrupted_trajectory_stop"):
        add(
            FailureType.INTERRUPTED_TRAJECTORY,
            "Stored incomplete trajectory was retained without replay.",
        )

    if not schema_valid:
        add(
            FailureType.SCHEMA_FAILURE,
            "Output failed deterministic schema validation.",
            issues=schema_issues,
        )

    if example.task_type in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
        answerable = bool(example.gold.get("answerable", True))
        abstain = bool(output.get("abstain", False))
        if answerable and abstain:
            add(FailureType.OVER_ABSTENTION, "Model abstained on an answerable example.")
        if not answerable and not abstain:
            add(FailureType.UNDER_ABSTENTION, "Model answered an unanswerable example.")
        if scores.get("citation_precision", 1.0) < 1.0:
            add(FailureType.WRONG_CITATION, "Predicted citations include unsupported evidence ids.")
        if answerable and scores.get("citation_recall", 1.0) < 1.0 and not abstain:
            add(FailureType.MISSED_EVIDENCE, "Predicted citations missed required gold evidence.")
        if answerable and not abstain and task_score < 1.0:
            add(
                FailureType.HALLUCINATION,
                "Answer was not fully supported by the reference scoring.",
            )

    if example.task_type == TaskType.STRUCTURED_EXTRACTION and task_score < 1.0:
        expected = example.gold.get("expected", example.gold)
        if _has_numeric_mismatch(expected, output):
            add(FailureType.NUMERIC_ERROR, "Numeric extracted value differs from reference.")

    tags = {str(tag).lower() for tag in example.metadata.get("tags", [])}
    if task_score < 1.0 and "negation" in tags:
        add(
            FailureType.NEGATION_ERROR,
            "Example is tagged as negation-sensitive and score is imperfect.",
        )
    if task_score < 1.0 and ({"date", "date_normalization", "era"} & tags):
        add(FailureType.DATE_NORMALIZATION_ERROR, "Date-sensitive example scored imperfectly.")

    if result.strategy == "multi_agent_debate" and task_score < 1.0:
        candidate_scores = [
            score_prediction(example, candidate.output)["task_score"]
            for candidate in result.candidates
        ]
        if any(score >= 1.0 for score in candidate_scores):
            add(
                FailureType.MAJORITY_WRONG,
                "Final debate answer lost to an available correct candidate.",
            )
        if len({_answer_key(candidate.output) for candidate in result.candidates}) == 1:
            add(
                FailureType.PEER_ERROR_PROPAGATION,
                "Debate candidates converged on the same wrong answer.",
            )

    if result.strategy == "critic_reviser" and len(result.candidates) >= 2:
        first_score = score_prediction(example, result.candidates[0].output)["task_score"]
        if task_score < first_score:
            add(FailureType.CRITIC_REGRESSION, "Revision scored worse than the generator output.")

    if result.strategy == "adaptive_router":
        route = str(result.metadata.get("route", "accept"))
        if route == "accept" and task_score < 1.0:
            add(FailureType.ROUTER_UNDER_ESCALATION, "Router accepted an imperfect initial answer.")
        if route != "accept" and result.candidates:
            initial_score = score_prediction(example, result.candidates[0].output)["task_score"]
            if initial_score >= task_score:
                add(FailureType.ROUTER_OVER_ESCALATION, "Escalation did not improve task score.")

    return annotations


def _has_numeric_mismatch(expected: dict[str, Any], output: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if isinstance(expected_value, int | float) and output.get(key) != expected_value:
            return True
    return False


def _answer_key(output: dict[str, Any]) -> str:
    return str(output.get("answer", output))
