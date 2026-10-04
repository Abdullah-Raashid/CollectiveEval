"""Deterministic task-output validation."""

from __future__ import annotations

from typing import Any

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.datasets import validate_against_json_schema


def validate_output_schema(
    example: BenchmarkExample, output: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Validate the expected answer shape for each task family."""

    issues: list[str] = []
    if example.task_type in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
        if not isinstance(output.get("answer"), str):
            issues.append("answer_must_be_string")
        if not isinstance(output.get("citations", []), list):
            issues.append("citations_must_be_list")
        if not isinstance(output.get("abstain", False), bool):
            issues.append("abstain_must_be_boolean")
        confidence = output.get("confidence")
        if not isinstance(confidence, int | float) or not 0.0 <= float(confidence) <= 1.0:
            issues.append("confidence_must_be_number_between_0_and_1")
    elif example.task_type == TaskType.BUSINESS_SUMMARIZATION:
        for field in ("summary", "decisions", "action_items", "risks"):
            if field not in output:
                issues.append(f"missing_{field}")
        if "summary" in output and not isinstance(output["summary"], str):
            issues.append("summary_must_be_string")
        for field in ("decisions", "action_items", "risks"):
            if field in output and not isinstance(output[field], list):
                issues.append(f"{field}_must_be_list")
        if isinstance(output.get("action_items"), list):
            for index, item in enumerate(output["action_items"]):
                if not isinstance(item, dict):
                    issues.append(f"action_items_{index}_must_be_object")
                    continue
                for field in ("owner", "action", "deadline"):
                    if field not in item:
                        issues.append(f"action_items_{index}_missing_{field}")
                if "deadline" in item and not (
                    item["deadline"] is None or isinstance(item["deadline"], str)
                ):
                    issues.append(f"action_items_{index}_deadline_must_be_string_or_null")
    elif example.task_type == TaskType.STRUCTURED_EXTRACTION:
        schema = example.gold.get("json_schema")
        if isinstance(schema, dict):
            issues.extend(validate_against_json_schema(output, schema, prefix="output"))
        else:
            required = example.gold.get("schema_required", [])
            for field in required:
                if field not in output:
                    issues.append(f"missing_{field}")
    return not issues, issues
