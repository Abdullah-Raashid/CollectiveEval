"""Task-specific model output parsing and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from collectiveeval.core import BenchmarkExample, ParseStatus, ProviderErrorType, TaskType
from collectiveeval.task_contracts import TaskContract, contract_for_example
from collectiveeval.validation import validate_output_schema


class OutputParseError(ValueError):
    """Raised when provider text cannot be parsed into a task answer."""

    def __init__(self, message: str, *, parsed_output: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.parsed_output = parsed_output


@dataclass(frozen=True)
class ParsedProviderOutput:
    content: dict[str, Any]
    raw_output: str
    parse_status: ParseStatus
    repair_attempted: bool = False


def parse_provider_output(
    example: BenchmarkExample,
    raw_output: str,
    *,
    contract: TaskContract | None = None,
    allow_repair: bool = True,
) -> ParsedProviderOutput:
    """Strictly parse task output with one bounded JSON-object extraction repair."""

    task_contract = contract or contract_for_example(example)
    try:
        parsed = _loads_object(raw_output)
        status = ParseStatus.OK
        repaired = False
    except OutputParseError:
        if not allow_repair:
            raise
        repaired_text = _extract_json_object(raw_output)
        if repaired_text is None:
            raise OutputParseError(f"{ProviderErrorType.PARSE_ERROR}: malformed JSON") from None
        parsed = _loads_object(repaired_text)
        status = ParseStatus.REPAIRED
        repaired = True

    try:
        validate_task_output(example, parsed, task_contract)
    except OutputParseError as exc:
        exc.parsed_output = parsed
        raise
    return ParsedProviderOutput(
        content=parsed,
        raw_output=raw_output,
        parse_status=status,
        repair_attempted=repaired,
    )


def validate_task_output(
    example: BenchmarkExample,
    output: dict[str, Any],
    contract: TaskContract | None = None,
) -> None:
    """Validate parsed output against task-specific expectations."""

    task_contract = contract or contract_for_example(example)
    schema_valid, issues = validate_output_schema(example, output)
    if not schema_valid:
        raise OutputParseError(f"{ProviderErrorType.PARSE_ERROR}: {issues}")

    if example.task_type in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
        confidence = output.get("confidence")
        if not isinstance(confidence, int | float) or not 0.0 <= float(confidence) <= 1.0:
            raise OutputParseError(f"{ProviderErrorType.PARSE_ERROR}: invalid confidence")
    if example.task_type == TaskType.STRUCTURED_EXTRACTION:
        _validate_json_schema_subset(output, task_contract.output_schema)


def _loads_object(raw_output: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise OutputParseError(f"{ProviderErrorType.PARSE_ERROR}: malformed JSON") from exc
    if not isinstance(payload, dict):
        raise OutputParseError(f"{ProviderErrorType.PARSE_ERROR}: expected JSON object")
    return payload


def _extract_json_object(raw_output: str) -> str | None:
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return raw_output[start : end + 1]


def _validate_json_schema_subset(output: dict[str, Any], schema: dict[str, Any]) -> None:
    """Small JSON Schema subset validator for tests and keyless execution."""

    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [field for field in required if field not in output]
        if missing:
            raise OutputParseError(f"{ProviderErrorType.PARSE_ERROR}: missing fields {missing}")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return
    for field, spec in properties.items():
        if field not in output or not isinstance(spec, dict):
            continue
        expected_type = spec.get("type")
        if expected_type is not None and not _matches_json_type(output[field], expected_type):
            raise OutputParseError(
                f"{ProviderErrorType.PARSE_ERROR}: field {field} expected {expected_type}"
            )


def _matches_json_type(value: Any, expected_type: Any) -> bool:
    allowed = expected_type if isinstance(expected_type, list) else [expected_type]
    for type_name in allowed:
        if type_name == "string" and isinstance(value, str):
            return True
        if type_name == "number" and isinstance(value, int | float) and not isinstance(value, bool):
            return True
        if type_name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if type_name == "boolean" and isinstance(value, bool):
            return True
        if type_name == "array" and isinstance(value, list):
            return True
        if type_name == "object" and isinstance(value, dict):
            return True
        if type_name == "null" and value is None:
            return True
    return False
