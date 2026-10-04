"""JSONL dataset loading, validation, and frozen-file hashing."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from collectiveeval.core import BenchmarkExample, TaskType

BENCHMARK_V1_VERSION = "collectiveeval-benchmark-v1"
BENCHMARK_V2_VERSION = "collectiveeval-benchmark-v2"
BENCHMARK_V3_VERSION = "collectiveeval-benchmark-v3"
BENCHMARK_V3_1_VERSION = "collectiveeval-benchmark-v3.1"
BENCHMARK_VERSION = BENCHMARK_V1_VERSION

QUALITY_BENCHMARK_VERSIONS = {BENCHMARK_V2_VERSION, BENCHMARK_V3_VERSION, BENCHMARK_V3_1_VERSION}
HARD_REASONING_OPERATIONS = {
    "conflict_resolution_same_entity",
    "current_version_resolution",
    "exception_rule_resolution",
    "cross_sentence_composition",
    "conditional_logic",
    "negation_affects_answer",
    "multi_constraint_resolution",
    "insufficient_evidence_abstention",
    "japanese_era_date_conversion",
    "arithmetic_or_normalization",
    "long_context_same_entity_retrieval",
    "referential_ambiguity_resolution",
}


def load_jsonl(path: str | Path) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            examples.append(BenchmarkExample.model_validate(payload))
    return examples


def write_jsonl(path: str | Path, examples: Iterable[BenchmarkExample]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.model_dump_json() + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_benchmark_file(path: str | Path) -> dict[str, Any]:
    examples = load_jsonl(path)
    issues = validate_examples(examples)
    if issues:
        raise ValueError("; ".join(issues))
    split_values = {str(example.metadata.get("split", "")) for example in examples}
    return {
        "path": str(path),
        "examples": len(examples),
        "sha256": file_sha256(path),
        "splits": ",".join(sorted(split_values)),
        "counts": benchmark_counts(examples),
    }


def validate_examples(examples: list[BenchmarkExample]) -> list[str]:
    """Return validation issues for a loaded benchmark collection."""

    issues: list[str] = []
    ids = [example.id for example in examples]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        issues.append(f"duplicate ids: {duplicates[:5]}")

    for example in examples:
        issues.extend(_validate_example_consistency(example))
        if _is_quality_benchmark_example(example):
            issues.extend(_validate_v2_metadata(example))
            issues.extend(_validate_v2_source_gold_consistency(example))
        if _is_v3_example(example):
            issues.extend(_validate_v3_metadata(example))
            issues.extend(_validate_v3_quality_invariants(example))
        if _is_v3_1_example(example):
            issues.extend(operation_witness_issues(example))
    return issues


def benchmark_counts(examples: list[BenchmarkExample]) -> dict[str, dict[str, int]]:
    by_task = Counter(str(example.task_type) for example in examples)
    by_difficulty = Counter(str(example.metadata.get("difficulty")) for example in examples)
    by_split = Counter(str(example.metadata.get("split")) for example in examples)
    return {
        "task_type": dict(sorted(by_task.items())),
        "difficulty": dict(sorted(by_difficulty.items())),
        "split": dict(sorted(by_split.items())),
    }


def validate_benchmark_dir(path: str | Path) -> dict[str, Any]:
    benchmark_dir = Path(path)
    dev_path = benchmark_dir / "dev.jsonl"
    test_path = benchmark_dir / "test.jsonl"
    dev = load_jsonl(dev_path)
    test = load_jsonl(test_path)
    issues = [*validate_examples(dev), *validate_examples(test)]
    dev_ids = {example.id for example in dev}
    test_ids = {example.id for example in test}
    leakage = sorted(dev_ids & test_ids)
    if leakage:
        issues.append(f"dev/test id leakage: {leakage[:5]}")
    all_examples = [*dev, *test]
    if any(_is_quality_benchmark_example(example) for example in all_examples):
        issues.extend(_validate_grouped_split(all_examples))
        near_duplicates = near_duplicate_leakage_report(dev, test, threshold=0.97)
        if near_duplicates["high_similarity_pairs"]:
            issues.append(
                f"near-duplicate dev/test leakage: {near_duplicates['high_similarity_pairs'][:3]}"
            )
    if any(_is_v3_example(example) for example in all_examples):
        issues.extend(_validate_v3_collection_quality(all_examples))
    if issues:
        raise ValueError("; ".join(issues))
    manifest_path = benchmark_dir / "manifest.json"
    version = BENCHMARK_VERSION
    if manifest_path.exists():
        try:
            version = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("version"))
        except json.JSONDecodeError:
            version = BENCHMARK_VERSION
    return {
        "version": version,
        "path": str(benchmark_dir),
        "examples": len(all_examples),
        "dev_examples": len(dev),
        "test_examples": len(test),
        "dev_sha256": file_sha256(dev_path),
        "test_sha256": file_sha256(test_path),
        "counts": benchmark_counts(all_examples),
    }


def freeze_benchmark(
    benchmark_dir: str | Path,
    *,
    version: str = BENCHMARK_VERSION,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_dir)
    validation = validate_benchmark_dir(benchmark_path)
    manifest_payload: dict[str, Any] = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "files": {
            "dev.jsonl": {
                "sha256": validation["dev_sha256"],
                "examples": validation["dev_examples"],
            },
            "test.jsonl": {
                "sha256": validation["test_sha256"],
                "examples": validation["test_examples"],
                "held_out": True,
            },
        },
        "counts": validation["counts"],
        "held_out_policy": "test split must not be used for training, tuning, or router fitting",
    }
    manifest_payload["manifest_payload_sha256"] = stable_json_sha256(
        {key: value for key, value in manifest_payload.items() if key != "manifest_payload_sha256"}
    )
    manifest_path = benchmark_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_payload["manifest_file_sha256"] = file_sha256(manifest_path)
    return manifest_payload


def inspect_benchmark_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        **manifest,
        "manifest_file_sha256": file_sha256(manifest_path),
        "ok": verify_benchmark_manifest(manifest_path)["ok"],
    }


def verify_benchmark_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_dir = manifest_path.parent
    reasons: list[str] = []
    payload_hash = manifest.get("manifest_payload_sha256")
    actual_payload_hash = stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    )
    if payload_hash != actual_payload_hash:
        reasons.append("manifest payload hash mismatch")
    for filename, file_info in dict(manifest.get("files", {})).items():
        actual_hash = file_sha256(benchmark_dir / filename)
        if actual_hash != file_info.get("sha256"):
            reasons.append(f"{filename} sha256 mismatch")
    return {"ok": not reasons, "reasons": reasons}


def stable_json_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_example_consistency(example: BenchmarkExample) -> list[str]:
    if example.task_type in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
        return _validate_grounded_evidence(example)
    if example.task_type == TaskType.STRUCTURED_EXTRACTION:
        return _validate_structured_expected_values(example)
    return []


def _validate_grounded_evidence(example: BenchmarkExample) -> list[str]:
    evidence_units = example.input.get("evidence_units", [])
    if not isinstance(evidence_units, list) or not evidence_units:
        return [f"{example.id}: missing evidence_units"]
    evidence_ids = {
        str(unit.get("id"))
        for unit in evidence_units
        if isinstance(unit, dict) and unit.get("id") is not None
    }
    if len(evidence_ids) != len(evidence_units):
        return [f"{example.id}: evidence_units need unique ids"]
    gold_evidence = example.gold.get("evidence") or example.gold.get("gold_evidence") or []
    missing = sorted(str(item) for item in gold_evidence if str(item) not in evidence_ids)
    if missing:
        return [f"{example.id}: gold evidence ids missing from evidence_units: {missing}"]
    return []


def _validate_structured_expected_values(example: BenchmarkExample) -> list[str]:
    schema = example.gold.get("json_schema", {})
    expected = example.gold.get("expected", {})
    return validate_against_json_schema(expected, schema, prefix=f"{example.id}.gold.expected")


def validate_against_json_schema(
    value: dict[str, Any],
    schema: dict[str, Any],
    *,
    prefix: str = "value",
) -> list[str]:
    issues: list[str] = []
    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [field for field in required if field not in value]
        issues.extend(f"{prefix}: missing required field {field}" for field in missing)
    properties = schema.get("properties", {})
    additional_allowed = bool(schema.get("additionalProperties", True))
    if isinstance(properties, dict) and not additional_allowed:
        extra = sorted(set(value) - set(properties))
        issues.extend(f"{prefix}: unexpected field {field}" for field in extra)
    if not isinstance(properties, dict):
        return [f"{prefix}: schema properties must be an object"]
    for field, spec in properties.items():
        if field not in value:
            continue
        if not isinstance(spec, dict):
            issues.append(f"{prefix}: schema for {field} must be an object")
            continue
        issues.extend(_validate_json_value(value[field], spec, f"{prefix}.{field}"))
    return issues


def _validate_json_value(value: Any, spec: dict[str, Any], path: str) -> list[str]:
    expected_type = spec.get("type")
    allowed = expected_type if isinstance(expected_type, list) else [expected_type]
    if not _matches_json_type(value, allowed):
        return [f"{path}: expected {expected_type}"]
    if value is None:
        return []
    if spec.get("format") == "date" and not _is_iso_date(str(value)):
        return [f"{path}: malformed date"]
    return []


def _matches_json_type(value: Any, allowed: list[Any]) -> bool:
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


def _is_iso_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def near_duplicate_leakage_report(
    dev: list[BenchmarkExample],
    test: list[BenchmarkExample],
    *,
    threshold: float = 0.92,
) -> dict[str, Any]:
    """Detect near duplicates across splits by metadata and normalized source text."""

    template_overlap = sorted(
        {
            str(dev_example.metadata.get("template_family"))
            for dev_example in dev
            for test_example in test
            if dev_example.metadata.get("template_family")
            and dev_example.metadata.get("template_family")
            == test_example.metadata.get("template_family")
            and dev_example.metadata.get("split_policy") == "heldout_template_family"
        }
    )
    scenario_overlap = sorted(
        {
            str(dev_example.metadata.get("scenario_family"))
            for dev_example in dev
            for test_example in test
            if dev_example.metadata.get("scenario_family")
            and dev_example.metadata.get("scenario_family")
            == test_example.metadata.get("scenario_family")
        }
    )
    high_similarity_pairs = []
    for dev_example in dev:
        dev_text = _normalized_source_text(dev_example)
        for test_example in test:
            if str(dev_example.task_type) != str(test_example.task_type):
                continue
            ratio = SequenceMatcher(None, dev_text, _normalized_source_text(test_example)).ratio()
            if ratio >= threshold:
                high_similarity_pairs.append(
                    {
                        "dev_id": dev_example.id,
                        "test_id": test_example.id,
                        "similarity": round(ratio, 4),
                    }
                )
    return {
        "template_family_overlap": template_overlap,
        "scenario_family_overlap": scenario_overlap,
        "high_similarity_pairs": high_similarity_pairs,
    }


def _is_quality_benchmark_example(example: BenchmarkExample) -> bool:
    return str(example.metadata.get("generator_version")) in QUALITY_BENCHMARK_VERSIONS


def _is_v2_example(example: BenchmarkExample) -> bool:
    return str(example.metadata.get("generator_version")) == BENCHMARK_V2_VERSION


def _is_v3_example(example: BenchmarkExample) -> bool:
    return str(example.metadata.get("generator_version")) in {
        BENCHMARK_V3_VERSION,
        BENCHMARK_V3_1_VERSION,
    }


def _is_v3_1_example(example: BenchmarkExample) -> bool:
    return str(example.metadata.get("generator_version")) == BENCHMARK_V3_1_VERSION


def _validate_v2_metadata(example: BenchmarkExample) -> list[str]:
    issues = []
    for field in ("template_family", "scenario_family", "generator_version", "difficulty_factors"):
        if field not in example.metadata:
            issues.append(f"{example.id}: v2 metadata missing {field}")
    factors = example.metadata.get("difficulty_factors")
    if not isinstance(factors, list) or not factors:
        issues.append(f"{example.id}: difficulty_factors must be a non-empty list")
    return issues


def _validate_v3_metadata(example: BenchmarkExample) -> list[str]:
    issues = []
    required = (
        "surface_form_family",
        "reasoning_family",
        "generator_operations",
        "structural_signature",
    )
    for field in required:
        if field not in example.metadata:
            issues.append(f"{example.id}: v3 metadata missing {field}")
    tags = example.metadata.get("tags", [])
    if isinstance(tags, list):
        canonical_tags = [_canonical_tag(str(tag)) for tag in tags]
        if canonical_tags != tags:
            issues.append(f"{example.id}: tags must be canonicalized")
        if len(canonical_tags) != len(set(canonical_tags)):
            issues.append(f"{example.id}: duplicate metadata tags")
    operations = example.metadata.get("generator_operations", [])
    factors = example.metadata.get("difficulty_factors", [])
    if not isinstance(operations, list) or not operations:
        issues.append(f"{example.id}: generator_operations must be a non-empty list")
    elif not all(isinstance(operation, str) for operation in operations):
        issues.append(f"{example.id}: generator_operations must contain strings")
    if isinstance(factors, list) and isinstance(operations, list):
        unsupported = sorted(set(factors) - set(operations))
        if unsupported:
            issues.append(f"{example.id}: unsupported difficulty factors: {unsupported}")
    difficulty = str(example.metadata.get("difficulty"))
    operation_set = set(operations) if isinstance(operations, list) else set()
    if difficulty == "hard" and not operation_set.intersection(HARD_REASONING_OPERATIONS):
        issues.append(f"{example.id}: hard example lacks a genuine hard reasoning operation")
    if difficulty == "easy" and operation_set.intersection(HARD_REASONING_OPERATIONS):
        issues.append(f"{example.id}: easy example includes hard reasoning operation")
    return issues


def _validate_grouped_split(examples: list[BenchmarkExample]) -> list[str]:
    issues = []
    splits_by_template: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        if example.metadata.get("split_policy") == "heldout_template_family":
            template = str(example.metadata.get("template_family"))
            splits_by_template[template].add(str(example.metadata.get("split")))
    leaked = {
        template: sorted(splits)
        for template, splits in splits_by_template.items()
        if len(splits) > 1
    }
    if leaked:
        issues.append(f"template family split leakage: {leaked}")
    return issues


def _validate_v2_source_gold_consistency(example: BenchmarkExample) -> list[str]:
    if example.task_type in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
        return _validate_v2_grounded_support(example)
    if example.task_type == TaskType.STRUCTURED_EXTRACTION:
        return _validate_v2_extraction_support(example)
    if example.task_type == TaskType.BUSINESS_SUMMARIZATION:
        return _validate_v2_summary_support(example)
    return []


def _validate_v2_grounded_support(example: BenchmarkExample) -> list[str]:
    if not bool(example.gold.get("answerable", True)):
        return []
    evidence_by_id = {
        str(unit.get("id")): str(unit.get("text", ""))
        for unit in example.input.get("evidence_units", [])
        if isinstance(unit, dict)
    }
    support_text = " ".join(
        evidence_by_id.get(str(evidence_id), "") for evidence_id in example.gold.get("evidence", [])
    )
    acceptable = [example.gold.get("answer", ""), *example.gold.get("acceptable_answers", [])]
    if not any(_value_supported_by_text(answer, support_text) for answer in acceptable):
        return [f"{example.id}: gold evidence does not entail answer literal/normalized value"]
    return []


def _validate_v2_extraction_support(example: BenchmarkExample) -> list[str]:
    text = str(example.input.get("text", ""))
    expected = example.gold.get("expected", {})
    if not isinstance(expected, dict):
        return [f"{example.id}: expected payload must be an object"]
    issues = []
    for field, value in expected.items():
        if value is None:
            if field == "due_date" and _contains_deadline(text):
                issues.append(f"{example.id}: due_date is null despite explicit deadline")
            continue
        if not _value_supported_by_text(value, text):
            issues.append(f"{example.id}: expected field {field} unsupported by source")
    return issues


def _validate_v2_summary_support(example: BenchmarkExample) -> list[str]:
    text = str(example.input.get("text", ""))
    issues = []
    for action in example.gold.get("action_items", []):
        if not isinstance(action, dict):
            continue
        owner = str(action.get("owner", ""))
        deadline = action.get("deadline")
        if not _value_supported_by_text(owner, text):
            issues.append(f"{example.id}: action owner unsupported: {owner}")
        if deadline is not None and not _owner_deadline_supported(text, owner, str(deadline)):
            issues.append(f"{example.id}: action deadline unsupported for {owner}")
        if deadline is None and _owner_has_explicit_deadline(text, owner):
            issues.append(f"{example.id}: action deadline null despite explicit owner deadline")
    for decision in example.gold.get("decisions", []):
        if not _value_supported_by_text(decision, text):
            issues.append(f"{example.id}: decision unsupported by source")
    for risk in example.gold.get("risks", []):
        if not _value_supported_by_text(risk, text):
            issues.append(f"{example.id}: risk unsupported by source")
    return issues


def _validate_v3_quality_invariants(example: BenchmarkExample) -> list[str]:
    issues = []
    issues.extend(_validate_v3_chronology(example))
    issues.extend(_validate_v3_target_alignment(example))
    if example.task_type == TaskType.BUSINESS_SUMMARIZATION:
        issues.extend(_validate_v3_action_support(example))
        issues.extend(_validate_v3_supported_facts(example))
    return issues


def _validate_v3_chronology(example: BenchmarkExample) -> list[str]:
    issues = []
    if example.task_type == TaskType.STRUCTURED_EXTRACTION:
        expected = example.gold.get("expected", {})
        if isinstance(expected, dict):
            issue_date = _parse_iso_date_or_none(expected.get("issue_date"))
            due_date = _parse_iso_date_or_none(expected.get("due_date"))
            if issue_date and due_date and due_date < issue_date:
                issues.append(f"{example.id}: due_date precedes issue_date")
    document_date = _parse_iso_date_or_none(example.metadata.get("document_date"))
    if document_date and example.task_type == TaskType.BUSINESS_SUMMARIZATION:
        for action in example.gold.get("action_items", []):
            if isinstance(action, dict):
                deadline = _parse_iso_date_or_none(action.get("deadline"))
                if deadline and deadline < document_date:
                    issues.append(f"{example.id}: action deadline precedes document date")
    chronology = example.metadata.get("chronology", {})
    if isinstance(chronology, dict):
        for pair in chronology.get("ordered_pairs", []):
            if not isinstance(pair, dict):
                continue
            earlier = _parse_iso_date_or_none(pair.get("earlier"))
            later = _parse_iso_date_or_none(pair.get("later"))
            label = str(pair.get("label", "date pair"))
            if earlier and later and later < earlier:
                issues.append(f"{example.id}: chronology violation for {label}")
    return issues


def _validate_v3_target_alignment(example: BenchmarkExample) -> list[str]:
    if example.task_type not in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
        return []
    entity = str(example.metadata.get("target_entity", ""))
    if not entity:
        return []
    question = str(example.input.get("question", ""))
    evidence_by_id = {
        str(unit.get("id")): str(unit.get("text", ""))
        for unit in example.input.get("evidence_units", [])
        if isinstance(unit, dict)
    }
    support_text = " ".join(
        evidence_by_id.get(str(evidence_id), "") for evidence_id in example.gold.get("evidence", [])
    )
    if bool(example.gold.get("answerable", True)) and (
        entity not in question or entity not in support_text
    ):
        return [f"{example.id}: question/evidence entity mismatch for {entity}"]
    return []


def _validate_v3_supported_facts(example: BenchmarkExample) -> list[str]:
    supported = example.gold.get("supported_facts", [])
    if not isinstance(supported, list):
        return [f"{example.id}: supported_facts must be exhaustive list"]
    normalized_supported = {_normalize_scalar(item) for item in supported}
    required: list[Any] = []
    required.extend(example.gold.get("decisions", []))
    required.extend(example.gold.get("risks", []))
    for action in example.gold.get("action_items", []):
        if not isinstance(action, dict):
            continue
        required.extend([action.get("owner"), action.get("action")])
        if action.get("deadline") is not None:
            required.append(action.get("deadline"))
    missing = [
        str(item)
        for item in required
        if item is not None and _normalize_scalar(item) not in normalized_supported
    ]
    if missing:
        return [f"{example.id}: supported_facts missing exhaustive gold facts: {missing[:5]}"]
    return []


def _validate_v3_action_support(example: BenchmarkExample) -> list[str]:
    text = str(example.input.get("text", ""))
    issues = []
    for action in example.gold.get("action_items", []):
        if not isinstance(action, dict):
            continue
        action_text = str(action.get("action", ""))
        if action_text and not _value_supported_by_text(action_text, text):
            issues.append(f"{example.id}: action description unsupported: {action_text}")
    return issues


def operation_witness_issues(example: BenchmarkExample) -> list[str]:
    """Return v3.1 operation labels that are not witnessed by the rendered example."""

    operations = example.metadata.get("generator_operations", [])
    if not isinstance(operations, list):
        return [f"{example.id}: generator_operations must be a list for witness audit"]
    issues = [
        f"{example.id}: operation not witnessed: {operation}"
        for operation in operations
        if isinstance(operation, str) and not _operation_witnessed(example, operation)
    ]
    if str(example.metadata.get("difficulty")) == "hard":
        witnessed_hard = {
            operation
            for operation in operations
            if isinstance(operation, str)
            and operation in HARD_REASONING_OPERATIONS
            and _operation_witnessed(example, operation)
        }
        if not witnessed_hard:
            issues.append(f"{example.id}: hard example lacks witnessed hard operation")
    return issues


def operation_witness_report(examples: list[BenchmarkExample]) -> dict[str, Any]:
    """Summarize operation-witness violations for a benchmark corpus."""

    violations_by_operation: dict[str, list[str]] = defaultdict(list)
    affected_examples: set[str] = set()
    for example in examples:
        for issue in operation_witness_issues(example):
            affected_examples.add(example.id)
            match = re.search(r"operation not witnessed: ([a-z0-9_]+)", issue)
            operation = match.group(1) if match else "hard_without_witnessed_operation"
            violations_by_operation[operation].append(example.id)
    return {
        "examples_checked": len(examples),
        "affected_example_count": len(affected_examples),
        "affected_example_ids": sorted(affected_examples),
        "violations_by_operation": {
            operation: sorted(example_ids)
            for operation, example_ids in sorted(violations_by_operation.items())
        },
    }


def _operation_witnessed(example: BenchmarkExample, operation: str) -> bool:
    text = _example_rendered_text(example)
    if operation in {"direct_lookup", "one_relevant_fact", "minimal_noise"}:
        if example.task_type == TaskType.BUSINESS_SUMMARIZATION:
            return not _validate_v2_summary_support(example)
        return _has_answer_literal_or_expected_value(example, text)
    if operation == "nested_negation":
        return "ないわけではない" in text or len(re.findall("ない", text)) >= 2
    if operation in {"same_entity_distractor", "relevant_distractor_same_entity"}:
        entity = str(example.metadata.get("target_entity", ""))
        return bool(entity and text.count(entity) >= 2)
    if operation in {"paraphrase_normalization", "normalization"}:
        return _has_answer_literal_or_expected_value(example, text)
    if operation in {"nullable_missing_field", "missing_field_handling"}:
        if example.task_type == TaskType.BUSINESS_SUMMARIZATION:
            return _has_null_or_absent_gold(example)
        return _has_null_or_absent_gold(example) and any(
            phrase in text for phrase in ("明記されていない", "未設定", "確認できない")
        )
    if operation in {"conflict_resolution_same_entity", "current_version_resolution"}:
        return _has_distinct_old_current_values(text)
    if operation in {"exception_rule_resolution", "conditional_logic"}:
        return bool(re.search(r"通常|原則|一般", text)) and bool(
            re.search(r"緊急|例外|ただし|今回区分", text)
        )
    if operation == "cross_sentence_composition":
        return _requires_multiple_evidence_units(example) or _has_cross_sentence_summary_fact(text)
    if operation == "multi_constraint_resolution":
        return (
            _requires_multiple_evidence_units(example)
            or _operation_witnessed(example, "exception_rule_resolution")
            or _operation_witnessed(example, "referential_ambiguity_resolution")
            or _operation_witnessed(example, "cross_sentence_composition")
        )
    if operation == "insufficient_evidence_abstention":
        return _has_insufficient_gold(example) and any(
            phrase in text for phrase in ("明記されていない", "確認できない", "記載されていない")
        )
    if operation == "long_context_same_entity_retrieval":
        entity = str(example.metadata.get("target_entity", ""))
        return bool(entity and text.count(entity) >= 8 and _distractor_count(text) >= 8)
    if operation == "referential_ambiguity_resolution":
        return bool(
            re.search(r"A案件|B案件|前者|後者|対象は", text)
        ) and _requires_multiple_evidence_units(example)
    if operation == "japanese_era_date_conversion":
        return bool(re.search(r"(令和|平成|昭和)\d+年", text)) and _gold_has_iso_date(example)
    if operation == "arithmetic_or_normalization":
        return bool(re.search(r"[０-９]+|万円|[0-9]{1,3}(,[0-9]{3})+円", text))
    return True


def _example_rendered_text(example: BenchmarkExample) -> str:
    return str(
        example.input.get("source_text")
        or example.input.get("text")
        or example.input.get("thread")
        or example.input
    )


def _has_answer_literal_or_expected_value(example: BenchmarkExample, text: str) -> bool:
    answer = example.gold.get("answer")
    if isinstance(answer, str) and answer:
        return _value_supported_by_text(answer, text) or any(
            _value_supported_by_text(value, text)
            for value in example.gold.get("acceptable_answers", [])
        )
    expected = example.gold.get("expected", {})
    if isinstance(expected, dict):
        return any(
            value is not None and _value_supported_by_text(value, text)
            for value in expected.values()
        )
    return True


def _has_null_or_absent_gold(example: BenchmarkExample) -> bool:
    expected = example.gold.get("expected", {})
    if isinstance(expected, dict) and any(value is None for value in expected.values()):
        return True
    if isinstance(example.gold.get("answer"), str) and "明記されていない" in str(
        example.gold["answer"]
    ):
        return True
    return any(
        isinstance(action, dict) and action.get("deadline") is None
        for action in example.gold.get("action_items", [])
    )


def _has_insufficient_gold(example: BenchmarkExample) -> bool:
    if example.task_type in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
        return not bool(example.gold.get("answerable", True))
    expected = example.gold.get("expected", {})
    return isinstance(expected, dict) and any(value is None for value in expected.values())


def _has_distinct_old_current_values(text: str) -> bool:
    old_sentence = next(
        (
            sentence
            for sentence in re.split(r"[。．\n]", text)
            if re.search(r"旧|前回|改訂前", sentence)
        ),
        "",
    )
    current_sentence = next(
        (
            sentence
            for sentence in re.split(r"[。．\n]", text)
            if re.search(r"最新版|改訂版|現行|今回|改訂後", sentence)
        ),
        "",
    )
    if not old_sentence or not current_sentence:
        return False
    old_values = set(_salient_values(old_sentence))
    current_values = set(_salient_values(current_sentence))
    return bool(old_values and current_values and old_values != current_values)


def _salient_values(text: str) -> list[str]:
    normalized = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return re.findall(r"\d{4}-\d{2}-\d{2}|\d+日前|\d+日|\d+(?:,\d{3})*円|\d+万円", normalized)


def _requires_multiple_evidence_units(example: BenchmarkExample) -> bool:
    evidence = example.gold.get("evidence") or example.gold.get("gold_evidence") or []
    return isinstance(evidence, list) and len(evidence) >= 2


def _has_cross_sentence_summary_fact(text: str) -> bool:
    sentences = [sentence for sentence in re.split(r"[。．\n]", text) if sentence.strip()]
    return len(sentences) >= 2 and any("要する" in sentence for sentence in sentences)


def _gold_has_iso_date(example: BenchmarkExample) -> bool:
    if re.search(r"20\d{2}-\d{2}-\d{2}", str(example.gold.get("answer", ""))):
        return True
    expected = example.gold.get("expected", {})
    return isinstance(expected, dict) and any(
        isinstance(value, str) and re.fullmatch(r"20\d{2}-\d{2}-\d{2}", value)
        for value in expected.values()
    )


def _distractor_count(text: str) -> int:
    return len(re.findall(r"対象外|参考|旧|過去|別件", text))


def _validate_v3_collection_quality(examples: list[BenchmarkExample]) -> list[str]:
    v3_examples = [example for example in examples if _is_v3_example(example)]
    if not v3_examples:
        return []
    issues = []
    surface_counts = Counter(
        str(example.metadata.get("surface_form_family")) for example in v3_examples
    )
    reasoning_counts = Counter(
        str(example.metadata.get("reasoning_family")) for example in v3_examples
    )
    if len(surface_counts) < 8:
        issues.append(f"v3 structural diversity too low: {dict(surface_counts)}")
    if len(reasoning_counts) < 8:
        issues.append(f"v3 reasoning diversity too low: {dict(reasoning_counts)}")
    nominal_only = [
        example.id
        for example in v3_examples
        if str(example.metadata.get("template_family"))
        == str(example.metadata.get("document_type"))
    ]
    if nominal_only:
        issues.append(f"template families differ only by nominal document type: {nominal_only[:5]}")
    return issues


def _normalized_source_text(example: BenchmarkExample) -> str:
    return _normalize_text(
        str(
            example.input.get("source_text")
            or example.input.get("text")
            or example.input.get("thread")
            or example.input
        )
    )


def _normalize_text(value: Any) -> str:
    text = str(value).lower()
    text = text.translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
    text = text.replace(",", "")
    return re.sub(r"\s+", "", text)


def _canonical_tag(tag: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", tag.strip().lower()).strip("_")


def _value_supported_by_text(value: Any, text: str) -> bool:
    if isinstance(value, bool):
        if value:
            return any(phrase in text for phrase in ("有効", "自動更新する", "適用する"))
        return any(phrase in text for phrase in ("無効", "有効ではない", "対象外"))
    normalized_value = _normalize_scalar(value)
    normalized_text = _normalize_text(text)
    if normalized_value in normalized_text:
        return True
    if isinstance(value, int):
        man = f"{value // 10000}万円" if value % 10000 == 0 else ""
        return bool(man and man in normalized_text)
    return False


def _normalize_scalar(value: Any) -> str:
    normalized = _normalize_text(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return normalized.replace(",", "")


def _contains_deadline(text: str) -> bool:
    return bool(
        re.search(r"(期限|締切)(は|:|：)?20\d{2}-\d{2}-\d{2}", text)
        or re.search(r"20\d{2}-\d{2}-\d{2}まで", text)
    )


def _owner_deadline_supported(text: str, owner: str, deadline: str) -> bool:
    return any(
        owner in sentence and deadline in sentence for sentence in re.split(r"[。．\n]", text)
    )


def _owner_has_explicit_deadline(text: str, owner: str) -> bool:
    return any(
        owner in sentence and re.search(r"20\d{2}-\d{2}-\d{2}まで", sentence)
        for sentence in re.split(r"[。．\n]", text)
    )


def _parse_iso_date_or_none(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
