"""Deterministic reference-based metrics for Phase 1."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Any, cast

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.validation import validate_output_schema


def normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value).strip().lower())
    return re.sub(r"\s+", " ", normalized)


def exact_match(prediction: Any, reference: Any) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def _tokens(text: Any) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    pieces = normalized.split()
    if len(pieces) > 1:
        return pieces
    return list(normalized)


def token_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    matches = sum(overlap.values())
    if matches == 0:
        return 0.0
    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def citation_scores(predicted: list[Any], gold: list[Any]) -> dict[str, float]:
    pred_set = {str(item) for item in predicted}
    gold_set = {str(item) for item in gold}
    if not pred_set and not gold_set:
        return {"citation_precision": 1.0, "citation_recall": 1.0, "citation_f1": 1.0}
    if not pred_set:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_f1": 0.0}
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set)
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"citation_precision": precision, "citation_recall": recall, "citation_f1": f1}


def json_parse_success(value: Any) -> float:
    if isinstance(value, dict):
        return 1.0
    if not isinstance(value, str):
        return 0.0
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return 0.0
    return 1.0


def score_prediction(example: BenchmarkExample, output: dict[str, Any]) -> dict[str, float]:
    schema_valid, _ = validate_output_schema(example, output)
    scores: dict[str, float] = {
        "json_parse_success": json_parse_success(output),
        "schema_compliance": float(schema_valid),
    }

    if example.task_type in {TaskType.GROUNDED_QA, TaskType.ROBUSTNESS}:
        gold_answer = str(example.gold.get("answer", ""))
        acceptable_answers = [gold_answer, *list(example.gold.get("acceptable_answers", []))]
        gold_citations = (
            example.gold.get("evidence")
            or example.gold.get("gold_evidence")
            or example.gold.get("citations")
            or []
        )
        answerable = bool(example.gold.get("answerable", True))
        abstain = bool(output.get("abstain", False))
        should_abstain = not answerable
        predicted_answer = output.get("answer", "")
        scores["exact_match"] = max(
            exact_match(predicted_answer, reference) for reference in acceptable_answers
        )
        scores["token_f1"] = max(
            token_f1(predicted_answer, reference) for reference in acceptable_answers
        )
        scores.update(citation_scores(list(output.get("citations", [])), list(gold_citations)))
        scores["citation_correctness"] = scores["citation_precision"]
        scores["evidence_coverage"] = scores["citation_recall"]
        scores["unsupported_answer_claims_heuristic"] = unsupported_answer_claims_heuristic(
            example,
            output,
        )
        scores["unsupported_claim_rate_heuristic"] = scores["unsupported_answer_claims_heuristic"]
        scores["predicted_abstain"] = float(abstain)
        scores["should_abstain"] = float(should_abstain)
        scores["abstention_correct"] = float(abstain == should_abstain)
        scores["over_abstention_rate"] = float(abstain and answerable)
        scores["under_abstention_rate"] = float((not abstain) and should_abstain)
        if not answerable:
            scores["task_score"] = float(abstain and not normalize_text(predicted_answer))
        elif abstain:
            scores["task_score"] = 0.0
        else:
            scores["task_score"] = (
                scores["exact_match"]
                + scores["token_f1"]
                + scores["citation_f1"]
                + (1.0 - scores["unsupported_answer_claims_heuristic"])
            ) / 4
        if example.task_type == TaskType.ROBUSTNESS:
            error = 1.0 - scores["task_score"]
            for tag in example.metadata.get("tags", []):
                scores[f"robustness_error_tag_{_metric_slug(str(tag))}"] = error
        return scores

    if example.task_type == TaskType.STRUCTURED_EXTRACTION:
        expected = example.gold.get("expected", example.gold)
        schema = example.gold.get("json_schema", {})
        field_scores = extraction_field_scores(output, expected, schema)
        scores.update(field_scores)
        scores["exact_match"] = float(output == expected)
        scores["json_schema_validity"] = scores["schema_compliance"]
        scores["task_score"] = (
            scores["json_schema_validity"]
            + scores["field_exact_match_mean"]
            + scores["numeric_field_accuracy"]
            + scores["date_field_accuracy"]
        ) / 4
        return scores

    scores.update(summarization_scores(output, example.gold))
    scores["exact_match"] = exact_match(output.get("summary", ""), example.gold.get("summary", ""))
    scores["token_f1"] = token_f1(output.get("summary", ""), example.gold.get("summary", ""))
    scores["task_score"] = (
        scores["schema_compliance"]
        + scores["decision_extraction_correctness"]
        + scores["action_item_correctness"]
        + scores["risk_extraction_correctness"]
        + scores["supported_fact_coverage"]
    ) / 5
    return scores


def abstention_scores(predicted: list[bool], should_abstain: list[bool]) -> dict[str, float]:
    tp = sum(pred and gold for pred, gold in zip(predicted, should_abstain, strict=True))
    fp = sum(pred and not gold for pred, gold in zip(predicted, should_abstain, strict=True))
    fn = sum((not pred) and gold for pred, gold in zip(predicted, should_abstain, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": f1,
        "over_abstention_rate": fp / len(predicted) if predicted else 0.0,
        "under_abstention_rate": fn / len(predicted) if predicted else 0.0,
    }


def derived_quality_metrics(
    *, task_score: float, total_tokens: int, estimated_cost_usd: float, latency_ms: float
) -> dict[str, float]:
    return {
        "quality_per_1k_tokens": task_score / (total_tokens / 1000) if total_tokens else 0.0,
        "quality_per_dollar": task_score / estimated_cost_usd if estimated_cost_usd else 0.0,
        "quality_per_second": task_score / (latency_ms / 1000) if latency_ms else 0.0,
    }


def unsupported_answer_claims_heuristic(
    example: BenchmarkExample,
    output: dict[str, Any],
) -> float:
    """Heuristic answer-support check; this is not semantic entailment."""

    answer = normalize_text(output.get("answer", ""))
    if not answer or bool(output.get("abstain", False)):
        return 0.0
    acceptable = [
        normalize_text(item)
        for item in [example.gold.get("answer", ""), *example.gold.get("acceptable_answers", [])]
        if normalize_text(item)
    ]
    if answer in acceptable:
        return 0.0
    evidence_text = " ".join(_gold_evidence_texts(example, output.get("citations", [])))
    if answer and answer in normalize_text(evidence_text):
        return 0.0
    return 1.0


def extraction_field_scores(
    output: dict[str, Any],
    expected: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, float]:
    properties = schema.get("properties", {})
    fields = list(properties) if isinstance(properties, dict) and properties else list(expected)
    exact_scores: list[float] = []
    numeric_scores: list[float] = []
    date_scores: list[float] = []
    for field in fields:
        if field not in expected:
            continue
        field_schema = schema_for_field(schema, field)
        score = _field_match(output.get(field), expected.get(field), field_schema)
        exact_scores.append(score)
        field_format = field_schema.get("format")
        if _schema_allows_type(field_schema, {"number", "integer"}):
            numeric_scores.append(
                float(normalize_number(output.get(field)) == normalize_number(expected[field]))
            )
        if field_format == "date" or str(field).endswith("_date"):
            date_scores.append(
                float(normalize_date(output.get(field)) == normalize_date(expected[field]))
            )
    hallucinated = sorted(set(output) - set(fields))
    return {
        "field_exact_match_mean": _mean(exact_scores),
        "numeric_field_accuracy": _mean(numeric_scores, default=1.0),
        "date_field_accuracy": _mean(date_scores, default=1.0),
        "hallucinated_field_count": float(len(hallucinated)),
    }


def summarization_scores(output: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    predicted_actions = _action_items(output.get("action_items", []))
    gold_actions = _action_items(gold.get("action_items", []))
    return {
        "decision_extraction_correctness": _set_f1(
            _string_items(output.get("decisions", [])),
            _string_items(gold.get("decisions", [])),
        ),
        "action_item_correctness": _set_f1(
            [_action_key(item) for item in predicted_actions],
            [_action_key(item) for item in gold_actions],
        ),
        "owner_correctness": _set_f1(
            [item.get("owner", "") for item in predicted_actions],
            [item.get("owner", "") for item in gold_actions],
        ),
        "deadline_correctness": _set_f1(
            [normalize_date(item.get("deadline", "")) for item in predicted_actions],
            [normalize_date(item.get("deadline", "")) for item in gold_actions],
        ),
        "risk_extraction_correctness": _set_f1(
            _string_items(output.get("risks", [])),
            _string_items(gold.get("risks", [])),
        ),
        "supported_fact_coverage": supported_fact_coverage(output, gold),
    }


def supported_fact_coverage(output: dict[str, Any], gold: dict[str, Any]) -> float:
    facts = [
        normalize_text(item) for item in gold.get("supported_facts", []) if normalize_text(item)
    ]
    if not facts:
        return 1.0
    output_text = normalize_text(json.dumps(output, ensure_ascii=False, sort_keys=True))
    covered = sum(1 for fact in facts if fact in output_text)
    return covered / len(facts)


def schema_for_field(schema: dict[str, Any], field: str) -> dict[str, Any]:
    properties = schema.get("properties", {})
    if isinstance(properties, dict) and isinstance(properties.get(field), dict):
        return cast(dict[str, Any], properties[field])
    return {}


def normalize_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    text = normalize_text(value)
    text = text.replace(",", "").replace("¥", "")
    multiplier = 1.0
    if "万円" in text:
        multiplier = 10000.0
        text = text.replace("万円", "")
    if "千円" in text:
        multiplier = 1000.0
        text = text.replace("千円", "")
    text = text.replace("円", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) * multiplier if match else None


def normalize_date(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    era_match = re.search(r"令和(\d+)年(\d+)月(\d+)日", text)
    if era_match:
        year = 2018 + int(era_match.group(1))
        return f"{year:04d}-{int(era_match.group(2)):02d}-{int(era_match.group(3)):02d}"
    for pattern in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, pattern).strftime("%Y-%m-%d")
        except ValueError:
            pass
    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if match:
        return f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    return text


def _field_match(predicted: Any, expected: Any, schema: dict[str, Any]) -> float:
    field_format = schema.get("format")
    if field_format == "date":
        return float(normalize_date(predicted) == normalize_date(expected))
    if _schema_allows_type(schema, {"number", "integer"}):
        return float(normalize_number(predicted) == normalize_number(expected))
    return exact_match(predicted, expected)


def _schema_allows_type(schema: dict[str, Any], expected_types: set[str]) -> bool:
    type_spec = schema.get("type")
    allowed = type_spec if isinstance(type_spec, list) else [type_spec]
    return bool(expected_types & {str(item) for item in allowed})


def _set_f1(predicted: list[str], gold: list[str]) -> float:
    predicted_set = {normalize_text(item) for item in predicted if normalize_text(item)}
    gold_set = {normalize_text(item) for item in gold if normalize_text(item)}
    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0
    tp = len(predicted_set & gold_set)
    precision = tp / len(predicted_set)
    recall = tp / len(gold_set)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _gold_evidence_texts(example: BenchmarkExample, citations: Any) -> list[str]:
    cited = {str(item) for item in citations} if isinstance(citations, list) else set()
    evidence_units = example.input.get("evidence_units", [])
    if not isinstance(evidence_units, list):
        return []
    texts = []
    for unit in evidence_units:
        if isinstance(unit, dict) and str(unit.get("id")) in cited:
            texts.append(str(unit.get("text", "")))
    return texts


def _string_items(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _action_items(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {str(key): str(item.get(key, "")) for key in ("owner", "action", "deadline")}
        for item in value
        if isinstance(item, dict)
    ]


def _action_key(item: dict[str, str]) -> str:
    return "|".join(normalize_text(item.get(key, "")) for key in ("owner", "action", "deadline"))


def _mean(values: list[float], *, default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _metric_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", normalize_text(value)).strip("_")
