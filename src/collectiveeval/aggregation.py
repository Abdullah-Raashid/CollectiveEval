"""Deterministic candidate aggregation utilities."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from collectiveeval.core import Candidate


def canonical_answer(output: dict[str, object]) -> str:
    """Stable key for deterministic voting."""

    if "answer" in output or "abstain" in output:
        payload = {
            "answer": output.get("answer", ""),
            "abstain": bool(output.get("abstain", False)),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


def majority_vote(candidates: list[Candidate]) -> Candidate:
    """Select the majority answer, breaking ties by confidence then order."""

    if not candidates:
        raise ValueError("majority_vote requires at least one candidate")

    counts: Counter[str] = Counter()
    confidences: dict[str, list[float]] = defaultdict(list)
    first_index: dict[str, int] = {}

    for index, candidate in enumerate(candidates):
        key = canonical_answer(candidate.output)
        counts[key] += 1
        confidences[key].append(candidate.confidence)
        first_index.setdefault(key, index)

    def rank(key: str) -> tuple[int, float, int]:
        avg_confidence = sum(confidences[key]) / len(confidences[key])
        return counts[key], avg_confidence, -first_index[key]

    selected_key = max(counts, key=rank)
    return candidates[first_index[selected_key]]


def summarize_peer(candidate: Candidate) -> dict[str, object]:
    """Concise debate peer summary without raw conversation history."""

    return {
        "answer": candidate.output.get("answer", candidate.output),
        "citations": candidate.output.get("citations", []),
        "abstain": candidate.output.get("abstain", False),
        "confidence": candidate.confidence,
        "model_id": candidate.model_id,
        "role": candidate.role,
    }
