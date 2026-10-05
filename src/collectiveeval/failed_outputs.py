"""V3 returned-failure representation; valid scoring remains in metrics.py."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from collectiveeval.budget import BudgetLedger, UnknownUsageStop
from collectiveeval.core import BenchmarkExample, Candidate, StrategyResult
from collectiveeval.parsing import _extract_json_object

CONTRACT = "failed-output.v3"
UNSCORABLE = "UNSCORABLE_FAILED_OUTPUT"
PENDING = "PENDING_AUTHORIZED_EXECUTION"


class FailedOutput(BaseModel):
    kind: Literal["failed_output"] = "failed_output"
    raw_content: Any
    parsed_json: Any = None
    json_syntax_parsed: bool
    raw_json_syntax_parsed: bool
    json_object: bool
    schema_valid: Literal[False] = False
    decode_source: str
    normalized_error: Literal["PARSE_ERROR"] = "PARSE_ERROR"
    failure_type: Literal["SCHEMA_FAILURE"] = "SCHEMA_FAILURE"
    attempt_id: str
    logical_call_id: str
    example_id: str
    strategy: str
    role: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    usage_source: Literal["PROVIDER_REPORTED"] = "PROVIDER_REPORTED"
    validation_error: str


class FailedPrediction(BaseModel):
    """A failed result is not a fabricated structured answer."""

    kind: Literal["failed_prediction"] = "failed_prediction"
    example_id: str
    strategy: str
    output: FailedOutput
    failed_outputs: list[FailedOutput]
    confidence: float = 0.0
    candidates: list[Candidate] = Field(default_factory=list)
    model_calls: int = 0
    logical_model_calls: int = 0
    provider_attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


PredictionResult = StrategyResult | FailedPrediction


def decode_failure(raw: Any, *, historical_projection: bool = False) -> tuple[Any, bool, str]:
    if not isinstance(raw, str):
        return raw, True, "RETURNED_STRUCTURED_CONTENT"
    try:
        return json.loads(raw), True, "STRICT_JSON"
    except json.JSONDecodeError:
        # Match the pre-existing parser's object projection, preserving all fields
        # and the complete raw response. This is not a new value/field repair.
        projected = _extract_json_object(raw) if historical_projection else None
        if projected:
            try:
                return json.loads(projected), True, "V2_OBJECT_PROJECTION_RAW_RETAINED"
            except json.JSONDecodeError:
                pass
    return None, False, "UNPARSEABLE_RAW_RETAINED"


def failed_prediction(
    ledger: BudgetLedger, example_id: str, strategy: str, *, historical_projection: bool = False
) -> FailedPrediction:
    records = ledger.records
    allowed_strategies = {strategy}
    if strategy == "adaptive_router":
        # Nested requests retain their existing strategy identity in the ledger.
        allowed_strategies.update({"critic_reviser", "multi_agent_debate"})
    if (
        ledger.unknown_usage_stop
        or not records
        or any(
            r.usage_status != "PROVIDER_REPORTED"
            or r.usage_source != "PROVIDER_REPORTED"
            or r.input_tokens is None
            or r.output_tokens is None
            for r in records
        )
    ):
        raise UnknownUsageStop("Failed output has non-authoritative consumption; no continuation")
    if any(
        r.outcome not in {"SUCCESS", "PARSE_ERROR_AFTER_RESPONSE"}
        or not r.end_timestamp
        or r.attempt_index != 0
        or r.example_id != example_id
        or r.strategy not in allowed_strategies
        for r in records
    ):
        raise ValueError("Incomplete/corrupted failed-output accounting")
    if (
        len(records) != len(ledger.logical_calls)
        or ({r.logical_call_id for r in records} != set(ledger.logical_calls))
        or any(
            not r.end_timestamp or r.outcome not in {"SUCCESS", "PARSE_ERROR_AFTER_RESPONSE"}
            for r in ledger.logical_calls.values()
        )
    ):
        raise ValueError("Incomplete failed-output logical accounting")
    outputs = []
    for record in records:
        if record.outcome != "PARSE_ERROR_AFTER_RESPONSE":
            continue
        if record.normalized_error != "PARSE_ERROR":
            raise ValueError("Returned failure classification mismatch")
        raw = record.metadata.get("raw_output")
        if raw is None:
            raw = record.metadata.get("structured_output")
        if raw is None:
            raise ValueError("Raw returned failure evidence is missing")
        parsed, syntax, source = decode_failure(raw, historical_projection=historical_projection)
        retained_object = record.metadata.get("failed_parsed_output")
        if isinstance(retained_object, dict):
            parsed, syntax, source = retained_object, True, "PRESERVED_PARSED_OBJECT"
        assert record.input_tokens is not None and record.output_tokens is not None
        outputs.append(
            FailedOutput(
                raw_content=raw,
                parsed_json=parsed,
                json_syntax_parsed=syntax,
                raw_json_syntax_parsed=decode_failure(raw)[1],
                json_object=syntax and isinstance(parsed, dict),
                decode_source=source,
                attempt_id=record.attempt_id,
                logical_call_id=record.logical_call_id,
                example_id=example_id,
                strategy=record.strategy,
                role=record.role,
                provider=record.provider,
                model=record.model,
                input_tokens=int(record.input_tokens),
                output_tokens=int(record.output_tokens),
                validation_error=str(record.metadata.get("parse_error", "PARSE_ERROR")),
            )
        )
    if not outputs:
        raise ValueError("No persisted returned parse failure")
    return FailedPrediction(
        example_id=example_id,
        strategy=strategy,
        output=outputs[0],
        failed_outputs=outputs,
        model_calls=ledger.model_calls,
        logical_model_calls=ledger.logical_model_calls,
        provider_attempts=ledger.provider_attempts,
        input_tokens=ledger.input_tokens,
        output_tokens=ledger.output_tokens,
        total_tokens=ledger.total_tokens,
        latency_ms=ledger.latency_ms,
        estimated_cost_usd=ledger.estimated_cost_usd,
        metadata={
            **ledger.metadata_dict(),
            "result_kind": "failed_prediction",
            "failed_output_contract": CONTRACT,
            "failed_outputs": [r.model_dump(mode="json") for r in outputs],
            "evaluation_status": PENDING,
        },
    )


def component_names(example: BenchmarkExample) -> set[str]:
    task = str(example.task_type)
    if task == "structured_extraction":
        return {
            "field_exact_match_mean",
            "numeric_field_accuracy",
            "date_field_accuracy",
            "hallucinated_field_count",
            "exact_match",
        }
    if task == "business_summarization":
        return {
            "decision_extraction_correctness",
            "action_item_correctness",
            "owner_correctness",
            "deadline_correctness",
            "risk_extraction_correctness",
            "supported_fact_coverage",
            "exact_match",
            "token_f1",
        }
    names = {
        "exact_match",
        "token_f1",
        "citation_precision",
        "citation_recall",
        "citation_f1",
        "citation_correctness",
        "evidence_coverage",
        "unsupported_answer_claims_heuristic",
        "unsupported_claim_rate_heuristic",
        "predicted_abstain",
        "should_abstain",
        "abstention_correct",
        "over_abstention_rate",
        "under_abstention_rate",
    }
    if task == "robustness":
        from collectiveeval.metrics import _metric_slug

        names |= {
            f"robustness_error_tag_{_metric_slug(str(tag))}"
            for tag in example.metadata.get("tags", [])
        }
    return names


def evaluate_output(
    example: BenchmarkExample, output: dict[str, Any] | FailedOutput
) -> dict[str, float | None]:
    from collectiveeval.metrics import score_prediction

    if isinstance(output, dict):
        return dict(score_prediction(example, output))
    if output.json_object:
        try:
            scores: dict[str, float | None] = dict(score_prediction(example, output.parsed_json))
        except (TypeError, ValueError, OverflowError):
            scores = dict.fromkeys(component_names(example))
            scores["task_score"] = 0.0
    else:
        scores = dict.fromkeys(component_names(example))
        scores["task_score"] = 0.0
    scores.update(
        json_parse_success=float(output.json_syntax_parsed),
        schema_compliance=0.0,
        schema_validity=0.0,
        task_output_validity=0.0,
    )
    if str(example.task_type) == "structured_extraction":
        scores["json_schema_validity"] = 0.0
    return scores


def failure_annotations(result: FailedPrediction) -> list[dict[str, Any]]:
    return [
        {
            "run_id": result.metadata["run_id"],
            "example_id": result.example_id,
            "failure_type": "SCHEMA_FAILURE",
            "source": CONTRACT,
            "rationale": "Returned output failed validation; no replay or field repair.",
            "metadata": {
                "normalized_error": "PARSE_ERROR",
                "attempt_ids": [output.attempt_id for output in result.failed_outputs],
                "contract": CONTRACT,
            },
        }
    ]
