"""Uncertainty features and adaptive routing policies."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import sklearn
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.validation import validate_output_schema

ROUTER_VERSION = "learned-router-v1"
ROUTER_FEATURE_NAMES = [
    "task_grounded_qa",
    "task_structured_extraction",
    "task_business_summarization",
    "task_robustness",
    "difficulty_easy",
    "difficulty_medium",
    "difficulty_hard",
    "input_length",
    "source_evidence_length",
    "output_length",
    "initial_confidence",
    "schema_valid",
    "parser_success",
    "citations_present",
    "citation_count",
    "citation_coverage",
    "abstain_flag",
    "validator_failures",
    "sample_disagreement",
    "model_is_weak_or_cheap",
]
FORBIDDEN_ROUTER_FEATURE_SUBSTRINGS = ("gold", "score", "task_score", "quality")


class UncertaintySignals(BaseModel):
    """Structured signals used by adaptive routers."""

    schema_valid: bool
    missing_citation: bool = False
    citation_coverage: float = 0.0
    citation_count: int = 0
    parser_success: bool = True
    output_length: int = 0
    source_evidence_length: int = 0
    abstain: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    input_length: int = 0
    difficulty: str = "unknown"
    validator_failures: int = 0
    sample_disagreement: float = 0.0
    model_id: str = "unknown"
    task_type: TaskType


def extract_uncertainty_signals(
    example: BenchmarkExample,
    output: dict[str, Any],
    *,
    confidence: float,
    model_id: str,
    sample_disagreement: float = 0.0,
) -> UncertaintySignals:
    schema_valid, issues = validate_output_schema(example, output)
    citations = output.get("citations", [])
    citation_count = len(citations) if isinstance(citations, list) else 0
    evidence_units = example.input.get("evidence_units", [])
    evidence_count = len(evidence_units) if isinstance(evidence_units, list) else 0
    citation_coverage = (
        citation_count / evidence_count if evidence_count else float(citation_count > 0)
    )
    parser_success = not bool(output.get("_parse_error"))
    return UncertaintySignals(
        schema_valid=schema_valid,
        missing_citation=example.task_type == TaskType.GROUNDED_QA and citation_count == 0,
        citation_coverage=min(1.0, citation_coverage),
        citation_count=citation_count,
        parser_success=parser_success,
        output_length=len(str(output)),
        source_evidence_length=len(str(example.input.get("source_text", ""))),
        abstain=bool(output.get("abstain", False)),
        confidence=confidence,
        input_length=len(str(example.input)),
        difficulty=str(example.metadata.get("difficulty", "unknown")),
        validator_failures=len(issues),
        sample_disagreement=sample_disagreement,
        model_id=model_id,
        task_type=example.task_type,
    )


class HeuristicRouter:
    """Transparent threshold router for Phase 1 adaptive inference."""

    def __init__(self, critic_threshold: float = 0.45, debate_threshold: float = 0.75) -> None:
        self.critic_threshold = critic_threshold
        self.debate_threshold = debate_threshold

    def uncertainty_score(self, signals: UncertaintySignals) -> float:
        score = 0.0
        score += 0.30 if not signals.parser_success else 0.0
        score += 0.35 if not signals.schema_valid else 0.0
        score += 0.25 if signals.missing_citation else 0.0
        score += 0.20 * (1.0 - signals.citation_coverage)
        score += 0.30 * (1.0 - signals.confidence)
        score += 0.10 if signals.abstain and signals.confidence < 0.7 else 0.0
        score += 0.10 if signals.input_length > 2000 else 0.0
        score += 0.15 if signals.difficulty in {"hard", "high"} else 0.0
        score += 0.10 * min(3, signals.validator_failures)
        score += 0.30 * signals.sample_disagreement
        return min(1.0, score)

    def route(self, signals: UncertaintySignals) -> str:
        score = self.uncertainty_score(signals)
        if score >= self.debate_threshold:
            return "debate"
        if score >= self.critic_threshold:
            return "critic"
        return "accept"


@dataclass(frozen=True)
class RouterTrainingExample:
    """One dev-only row for learned escalation training."""

    signals: UncertaintySignals
    escalation_helped: bool
    split: str = "dev"


class RouterFeatureRow(BaseModel):
    """Inference-time router features only; no gold-derived evaluation fields."""

    example_id: str
    split: str
    features: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouterTrainingTarget(BaseModel):
    """Gold-derived training target kept separate from router features."""

    example_id: str
    initial_quality: float = 0.0
    escalated_quality: float = 0.0
    initial_tokens: int = 0
    escalated_tokens: int = 0
    initial_calls: int = 0
    escalated_calls: int = 0
    initial_cost: float = 0.0
    escalated_cost: float = 0.0
    debate_quality: float | None = None
    debate_tokens: int | None = None
    debate_calls: int | None = None
    debate_cost: float | None = None
    quality_gain: float
    extra_tokens: int
    extra_calls: int
    extra_cost: float
    utility_gain: float
    label: int


class RouterUtilityConfig(BaseModel):
    lambda_tokens: float = 0.0
    lambda_calls: float = 0.0
    lambda_cost: float = 0.0
    min_utility_gain: float = 0.0
    normalizer_tokens: float = 1000.0
    normalizer_calls: float = 1.0
    normalizer_cost: float = 1.0


class LearnedRouter:
    """Logistic regression router trained only on dev rows."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.model = LogisticRegression(random_state=0)
        self.scaler = StandardScaler()
        self.threshold = threshold
        self._is_trained = False

    def train(self, rows: Iterable[RouterTrainingExample]) -> None:
        materialized = list(rows)
        if not materialized:
            raise ValueError("learned router needs at least one training row")
        invalid_splits = sorted({row.split for row in materialized if row.split != "dev"})
        if invalid_splits:
            raise ValueError(f"learned router can train only on dev data, got {invalid_splits}")
        x = np.array([feature_vector(row.signals) for row in materialized], dtype=float)
        y = np.array([int(row.escalation_helped) for row in materialized], dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("learned router needs both positive and negative examples")
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self._is_trained = True

    def train_feature_rows(
        self,
        rows: list[RouterFeatureRow],
        targets: list[RouterTrainingTarget],
    ) -> None:
        validate_feature_schema()
        if not rows:
            raise ValueError("learned router needs feature rows")
        target_by_id = {target.example_id: target for target in targets}
        missing = [row.example_id for row in rows if row.example_id not in target_by_id]
        if missing:
            raise ValueError(f"missing router targets for examples: {missing[:5]}")
        invalid_splits = sorted({row.split for row in rows if row.split != "router_train"})
        if invalid_splits:
            raise ValueError(
                f"learned router can train only on router_train rows, got {invalid_splits}"
            )
        x = np.array([ordered_feature_vector(row.features) for row in rows], dtype=float)
        y = np.array([target_by_id[row.example_id].label for row in rows], dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("learned router needs both positive and negative targets")
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self._is_trained = True

    def escalation_probability(self, signals: UncertaintySignals) -> float:
        if not self._is_trained:
            raise RuntimeError("learned router has not been trained")
        x = np.array([feature_vector(signals)], dtype=float)
        return float(self.model.predict_proba(self.scaler.transform(x))[0, 1])

    def escalation_probability_from_features(self, features: dict[str, float]) -> float:
        if not self._is_trained:
            raise RuntimeError("learned router has not been trained")
        x = np.array([ordered_feature_vector(features)], dtype=float)
        return float(self.model.predict_proba(self.scaler.transform(x))[0, 1])

    def route(self, signals: UncertaintySignals, threshold: float | None = None) -> str:
        probability = self.escalation_probability(signals)
        threshold_value = self.threshold if threshold is None else threshold
        if probability < threshold_value:
            return "accept"
        if probability < 0.75:
            return "critic"
        return "debate"

    def uncertainty_score(self, signals: UncertaintySignals) -> float:
        return self.escalation_probability(signals)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix:
            with path.open("wb") as handle:
                pickle.dump(self, handle)
            return
        self.save_artifact(path, training_manifest={})

    def save_artifact(self, path: str | Path, *, training_manifest: dict[str, Any]) -> None:
        if not self._is_trained:
            raise RuntimeError("cannot save an untrained learned router")
        artifact_dir = Path(path)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        feature_schema = {
            "router_version": ROUTER_VERSION,
            "feature_names": ROUTER_FEATURE_NAMES,
            "forbidden_feature_substrings": FORBIDDEN_ROUTER_FEATURE_SUBSTRINGS,
        }
        (artifact_dir / "feature_schema.json").write_text(
            json.dumps(feature_schema, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "threshold": self.threshold,
                "feature_schema_hash": stable_json_hash(feature_schema),
            },
            artifact_dir / "model.joblib",
        )
        manifest = {
            "router_version": ROUTER_VERSION,
            "model_hash": file_sha256(artifact_dir / "model.joblib"),
            "feature_schema_hash": stable_json_hash(feature_schema),
            "training_dataset_hash": stable_json_hash(training_manifest),
            "training_configuration": training_manifest,
            "sklearn_version": sklearn.__version__,
            "trust_assumption": (
                "Load router artifacts only from trusted local sources; joblib executes "
                "Python object deserialization."
            ),
        }
        (artifact_dir / "training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metrics = {"threshold": self.threshold, "coefficients": self.coefficients()}
        (artifact_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> LearnedRouter:
        path = Path(path)
        if path.is_dir():
            return cls.load_artifact(path)
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, cls):
            raise ValueError("router artifact is not a LearnedRouter")
        if not loaded._is_trained:
            raise ValueError("learned router artifact is not trained")
        return loaded

    @classmethod
    def load_artifact(cls, path: str | Path) -> LearnedRouter:
        artifact_dir = Path(path)
        feature_schema = json.loads(
            (artifact_dir / "feature_schema.json").read_text(encoding="utf-8")
        )
        if feature_schema.get("feature_names") != ROUTER_FEATURE_NAMES:
            raise ValueError("learned router feature schema mismatch")
        bundle = joblib.load(artifact_dir / "model.joblib")
        if bundle.get("feature_schema_hash") != stable_json_hash(feature_schema):
            raise ValueError("learned router feature schema hash mismatch")
        router = cls(threshold=float(bundle["threshold"]))
        router.model = bundle["model"]
        router.scaler = bundle["scaler"]
        router._is_trained = True
        return router

    def coefficients(self) -> list[dict[str, float | str]]:
        if not self._is_trained:
            return []
        coefs = self.model.coef_[0].tolist()
        max_abs = max(abs(value) for value in coefs) or 1.0
        return [
            {
                "feature": feature,
                "coefficient": float(coef),
                "normalized_importance": abs(float(coef)) / max_abs,
            }
            for feature, coef in zip(ROUTER_FEATURE_NAMES, coefs, strict=True)
        ]


def feature_vector(signals: UncertaintySignals) -> list[float]:
    return ordered_feature_vector(features_from_signals(signals))


def features_from_signals(signals: UncertaintySignals) -> dict[str, float]:
    return {
        "task_grounded_qa": float(signals.task_type == TaskType.GROUNDED_QA),
        "task_structured_extraction": float(signals.task_type == TaskType.STRUCTURED_EXTRACTION),
        "task_business_summarization": float(signals.task_type == TaskType.BUSINESS_SUMMARIZATION),
        "task_robustness": float(signals.task_type == TaskType.ROBUSTNESS),
        "difficulty_easy": float(signals.difficulty == "easy"),
        "difficulty_medium": float(signals.difficulty == "medium"),
        "difficulty_hard": float(signals.difficulty in {"hard", "high"}),
        "input_length": float(signals.input_length),
        "source_evidence_length": float(signals.source_evidence_length),
        "output_length": float(signals.output_length),
        "initial_confidence": float(signals.confidence),
        "schema_valid": float(signals.schema_valid),
        "parser_success": float(signals.parser_success),
        "citations_present": float(signals.citation_count > 0),
        "citation_count": float(signals.citation_count),
        "citation_coverage": float(signals.citation_coverage),
        "abstain_flag": float(signals.abstain),
        "validator_failures": float(signals.validator_failures),
        "sample_disagreement": float(signals.sample_disagreement),
        "model_is_weak_or_cheap": float("weak" in signals.model_id or "cheap" in signals.model_id),
    }


def router_feature_row_from_initial(
    example: BenchmarkExample,
    output: dict[str, Any],
    *,
    confidence: float,
    model_id: str,
    split: str,
) -> RouterFeatureRow:
    signals = extract_uncertainty_signals(
        example,
        output,
        confidence=confidence,
        model_id=model_id,
    )
    return RouterFeatureRow(
        example_id=example.id,
        split=split,
        features=features_from_signals(signals),
        metadata={"model_id": model_id},
    )


def construct_router_target(
    *,
    example_id: str,
    initial_quality: float,
    escalated_quality: float,
    initial_tokens: int,
    escalated_tokens: int,
    initial_calls: int,
    escalated_calls: int,
    initial_cost: float,
    escalated_cost: float,
    debate_quality: float | None = None,
    debate_tokens: int | None = None,
    debate_calls: int | None = None,
    debate_cost: float | None = None,
    config: RouterUtilityConfig | None = None,
) -> RouterTrainingTarget:
    utility = config or RouterUtilityConfig()
    quality_gain = escalated_quality - initial_quality
    extra_tokens = escalated_tokens - initial_tokens
    extra_calls = escalated_calls - initial_calls
    extra_cost = escalated_cost - initial_cost
    utility_gain = (
        quality_gain
        - utility.lambda_tokens * (extra_tokens / utility.normalizer_tokens)
        - utility.lambda_calls * (extra_calls / utility.normalizer_calls)
        - utility.lambda_cost * (extra_cost / utility.normalizer_cost)
    )
    return RouterTrainingTarget(
        example_id=example_id,
        initial_quality=initial_quality,
        escalated_quality=escalated_quality,
        initial_tokens=initial_tokens,
        escalated_tokens=escalated_tokens,
        initial_calls=initial_calls,
        escalated_calls=escalated_calls,
        initial_cost=initial_cost,
        escalated_cost=escalated_cost,
        debate_quality=debate_quality,
        debate_tokens=debate_tokens,
        debate_calls=debate_calls,
        debate_cost=debate_cost,
        quality_gain=quality_gain,
        extra_tokens=extra_tokens,
        extra_calls=extra_calls,
        extra_cost=extra_cost,
        utility_gain=utility_gain,
        label=int(utility_gain > utility.min_utility_gain),
    )


def ordered_feature_vector(features: dict[str, float]) -> list[float]:
    validate_feature_schema()
    missing = [name for name in ROUTER_FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"missing router features: {missing}")
    extras = sorted(set(features) - set(ROUTER_FEATURE_NAMES))
    if extras:
        raise ValueError(f"unknown router features: {extras}")
    return [float(features[name]) for name in ROUTER_FEATURE_NAMES]


def validate_feature_schema() -> None:
    forbidden = [
        name
        for name in ROUTER_FEATURE_NAMES
        if any(token in name.lower() for token in FORBIDDEN_ROUTER_FEATURE_SUBSTRINGS)
    ]
    if forbidden:
        raise ValueError(f"router feature schema contains gold-derived feature names: {forbidden}")


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
