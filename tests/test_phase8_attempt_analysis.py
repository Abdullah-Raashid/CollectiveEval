"""Offline validation of the repaired scientific report, without provider or TEST access."""

import json
from unittest.mock import patch

import pytest

from collectiveeval.budget import SCIENTIFIC_ATTEMPT_POLICY
from collectiveeval.phase8_corrective_analysis import validate_attempt_records


def attempt() -> dict:
    return {
        "attempt_id": "attempt-1",
        "logical_call_id": "logical-1",
        "attempt_index": 0,
        "outcome": "SUCCESS",
        "usage_status": "PROVIDER_REPORTED",
        "input_tokens": 30,
        "output_tokens": 17,
        "start_timestamp": "2026-10-04T00:00:00+00:00",
        "end_timestamp": "2026-10-04T00:00:01+00:00",
        "metadata_json": json.dumps({"unknown_usage_policy": SCIENTIFIC_ATTEMPT_POLICY}),
    }


def logical() -> dict:
    return {
        "logical_call_id": "logical-1",
        "outcome": "SUCCESS",
        "end_timestamp": "2026-10-04T00:00:01+00:00",
    }


def test_clean_attempt_relationship_passes() -> None:
    assert validate_attempt_records([attempt()], [logical()]) == []


@pytest.mark.parametrize(
    "change",
    [
        {"attempt_index": 1},
        {"outcome": "TIMEOUT", "input_tokens": None, "usage_status": "UNKNOWN_NOT_RETURNED"},
        {"usage_status": "ESTIMATED"},
        {"metadata_json": json.dumps({"unknown_usage_policy": "GENERAL_RETRY"})},
        {"end_timestamp": None},
        {"end_timestamp": "2026-10-03T00:00:01+00:00"},
    ],
)
def test_scientific_accounting_integrity_rejects_policy_or_usage_drift(change: dict) -> None:
    assert validate_attempt_records([attempt() | change], [logical()])


def test_duplicate_attempts_and_missing_logical_records_are_rejected() -> None:
    assert validate_attempt_records([attempt(), attempt()], []) == [
        "Missing or duplicate attempt identity",
        "Attempt/logical-call relationship mismatch",
    ]


def test_unfinished_logical_lifecycle_is_not_clean_efficiency_evidence() -> None:
    assert validate_attempt_records(
        [attempt()], [logical() | {"outcome": "STARTED", "end_timestamp": None}]
    ) == ["Unfinished logical lifecycle: logical-1"]


def test_legacy_analysis_default_refuses_all_writes_before_reading_evidence() -> None:
    from collectiveeval.phase8_corrective_analysis import analyze_corrective

    with (
        patch("collectiveeval.phase8_corrective_analysis.evidence_hashes") as read,
        pytest.raises(ValueError, match="immutable"),
    ):
        analyze_corrective()
    read.assert_not_called()
