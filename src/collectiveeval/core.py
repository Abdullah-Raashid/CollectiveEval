"""Core typed objects shared by providers, strategies, and evaluators."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskType(StrEnum):
    """Canonical task families supported by the benchmark interface."""

    GROUNDED_QA = "grounded_qa"
    STRUCTURED_EXTRACTION = "structured_extraction"
    BUSINESS_SUMMARIZATION = "business_summarization"
    ROBUSTNESS = "robustness"


class BenchmarkExample(BaseModel):
    """One benchmark item, independent of strategy or model provider."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    task_type: TaskType
    language: str = Field(default="ja", min_length=2)
    input: dict[str, Any]
    gold: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def require_mapping_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        split = value.get("split")
        if split is not None and split not in {"train", "dev", "test", "mock"}:
            raise ValueError("metadata.split must be one of train, dev, test, or mock")
        return value

    @model_validator(mode="after")
    def validate_task_payload(self) -> BenchmarkExample:
        _validate_metadata(self.metadata)
        if self.task_type in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
            _validate_grounded_qa_payload(self.input, self.gold)
        elif self.task_type == TaskType.STRUCTURED_EXTRACTION:
            _validate_structured_extraction_payload(self.input, self.gold)
        elif self.task_type == TaskType.BUSINESS_SUMMARIZATION:
            _validate_summarization_payload(self.input, self.gold)
        return self


class ModelSpec(BaseModel):
    """Provider/model selection plus static cost metadata."""

    provider: str = "mock"
    model: str = "mock-accurate"
    role: str | None = None
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    timeout_s: float = 30.0
    base_url: str | None = None
    api_key_env: str | None = None
    mock_mode: str | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    @property
    def model_id(self) -> str:
        return f"{self.provider}/{self.model}"


class TokenUsage(BaseModel):
    """Token usage for a single model call."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class UsageSource(StrEnum):
    """Where persisted token usage came from."""

    PROVIDER_REPORTED = "PROVIDER_REPORTED"
    ESTIMATED = "ESTIMATED"


class ProviderErrorType(StrEnum):
    """Normalized provider and parsing errors."""

    AUTH_ERROR = "AUTH_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"
    CONTEXT_LENGTH = "CONTEXT_LENGTH"
    INVALID_REQUEST = "INVALID_REQUEST"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PARSE_ERROR = "PARSE_ERROR"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    UNKNOWN_PROVIDER_ERROR = "UNKNOWN_PROVIDER_ERROR"


class ParseStatus(StrEnum):
    """Structured output parsing state."""

    OK = "OK"
    REPAIRED = "REPAIRED"
    PARSE_ERROR = "PARSE_ERROR"


class ProviderRequest(BaseModel):
    """Structured request sent from strategies to providers."""

    example: BenchmarkExample
    model: ModelSpec
    strategy: str
    role: str
    round_index: int = 0
    candidate: dict[str, Any] | None = None
    peer_summaries: list[dict[str, Any]] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""
    prompt_version: str = ""
    expected_schema: dict[str, Any] = Field(default_factory=dict)


class ModelOutput(BaseModel):
    """Structured provider response.

    The `content` field contains task answers only. It must not contain hidden
    chain-of-thought or raw private reasoning.
    """

    content: dict[str, Any] = Field(default_factory=dict)
    raw_output: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    usage_source: UsageSource | None = None
    parse_status: ParseStatus = ParseStatus.OK
    parse_error: str | None = None
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    finish_reason: str = "stop"
    provider: str = "mock"
    model: str = "mock"
    retry_count: int = 0


class Candidate(BaseModel):
    """A candidate answer considered by an aggregation step."""

    output: dict[str, Any]
    confidence: float
    model_id: str
    role: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyResult(BaseModel):
    """Final result emitted by an inference strategy."""

    example_id: str
    strategy: str
    output: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
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


def finish_result(
    *,
    example_id: str,
    strategy: str,
    output: dict[str, Any],
    confidence: float,
    candidates: list[Candidate],
    context_usage: dict[str, int | float],
    metadata: dict[str, Any] | None = None,
) -> StrategyResult:
    """Build a result with budget usage copied from the strategy context."""

    return StrategyResult(
        example_id=example_id,
        strategy=strategy,
        output=output,
        confidence=confidence,
        candidates=candidates,
        model_calls=int(context_usage["model_calls"]),
        logical_model_calls=int(
            context_usage.get("logical_model_calls", context_usage["model_calls"])
        ),
        provider_attempts=int(context_usage.get("provider_attempts", context_usage["model_calls"])),
        input_tokens=int(context_usage["input_tokens"]),
        output_tokens=int(context_usage["output_tokens"]),
        total_tokens=int(context_usage["total_tokens"]),
        latency_ms=float(context_usage["latency_ms"]),
        estimated_cost_usd=float(context_usage["estimated_cost_usd"]),
        metadata=metadata or {},
    )


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if not metadata.get("source"):
        raise ValueError("metadata.source is required")
    difficulty = metadata.get("difficulty")
    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("metadata.difficulty must be easy, medium, or hard")
    tags = metadata.get("tags")
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) for tag in tags):
        raise ValueError("metadata.tags must be a non-empty list of strings")
    split = metadata.get("split")
    if split not in {"train", "dev", "test", "mock"}:
        raise ValueError("metadata.split is required")


def _validate_grounded_qa_payload(input_payload: dict[str, Any], gold: dict[str, Any]) -> None:
    if not isinstance(input_payload.get("question"), str) or not input_payload["question"].strip():
        raise ValueError("grounded QA requires input.question")
    if not (
        isinstance(input_payload.get("source_text"), str)
        or isinstance(input_payload.get("evidence_units"), list)
    ):
        raise ValueError("grounded QA requires source_text or evidence_units")
    answerable = gold.get("answerable")
    if not isinstance(answerable, bool):
        raise ValueError("grounded QA requires boolean gold.answerable")
    answer = gold.get("answer", "")
    evidence = gold.get("evidence") or gold.get("gold_evidence") or []
    if answerable:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answerable QA requires non-empty gold.answer")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("answerable QA requires non-empty gold.evidence")
    else:
        if answer not in {"", None}:
            raise ValueError("unanswerable QA must not include a gold answer")
        if evidence:
            raise ValueError("unanswerable QA must not include gold evidence")
    variants = gold.get("acceptable_answers", [])
    if variants and not (
        isinstance(variants, list) and all(isinstance(item, str) for item in variants)
    ):
        raise ValueError("gold.acceptable_answers must be a list of strings")


def _validate_structured_extraction_payload(
    input_payload: dict[str, Any],
    gold: dict[str, Any],
) -> None:
    if not isinstance(input_payload.get("text"), str) or not input_payload["text"].strip():
        raise ValueError("structured extraction requires input.text")
    expected = gold.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("structured extraction requires gold.expected object")
    schema = gold.get("json_schema")
    if not isinstance(schema, dict):
        raise ValueError("structured extraction requires gold.json_schema")
    _validate_json_schema_definition(schema)


def _validate_summarization_payload(input_payload: dict[str, Any], gold: dict[str, Any]) -> None:
    if not any(isinstance(input_payload.get(field), str) for field in ("text", "thread", "report")):
        raise ValueError("summarization requires input text/thread/report")
    if not isinstance(gold.get("summary"), str) or not gold["summary"].strip():
        raise ValueError("summarization requires gold.summary")
    for field in ("decisions", "action_items", "risks"):
        if not isinstance(gold.get(field), list):
            raise ValueError(f"summarization requires gold.{field} list")
    for item in gold.get("action_items", []):
        if not isinstance(item, dict):
            raise ValueError("gold.action_items entries must be objects")
        for field in ("owner", "action", "deadline"):
            if field not in item:
                raise ValueError(f"gold.action_items entries require {field}")


def _validate_json_schema_definition(schema: dict[str, Any]) -> None:
    if schema.get("type") != "object":
        raise ValueError("json_schema.type must be object")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ValueError("json_schema.properties must be a non-empty object")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError("json_schema.required must be a list of strings")
    missing = [field for field in required if field not in properties]
    if missing:
        raise ValueError(f"json_schema.required references missing properties: {missing}")
    allowed_types = {"string", "number", "integer", "boolean", "array", "object", "null"}
    for field, spec in properties.items():
        if not isinstance(field, str) or not isinstance(spec, dict):
            raise ValueError("json_schema.properties must map strings to schema objects")
        type_spec = spec.get("type")
        allowed = type_spec if isinstance(type_spec, list) else [type_spec]
        if not allowed or any(type_name not in allowed_types for type_name in allowed):
            raise ValueError(f"json_schema field {field} has unsupported type")
