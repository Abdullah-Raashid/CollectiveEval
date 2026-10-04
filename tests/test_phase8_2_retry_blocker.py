"""Known live blocker pinned offline; no Ollama or TEST access."""

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import pytest

from collectiveeval import phase8_corrective
from collectiveeval.budget import BudgetExceeded, BudgetLedger, InferenceBudget
from collectiveeval.context import StrategyContext
from collectiveeval.core import (
    BenchmarkExample,
    ModelOutput,
    ModelSpec,
    ProviderErrorType,
    ProviderRequest,
    TaskType,
    TokenUsage,
    UsageSource,
)
from collectiveeval.providers import ModelProvider, ProviderCapabilities, ProviderError


class RetryOnceProvider(ModelProvider):
    provider_id = "offline-retry-fixture"
    capabilities = ProviderCapabilities(supports_usage=True)

    def __init__(self) -> None:
        self.attempts = 0

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        self.attempts += 1
        if self.attempts == 1:
            raise ProviderError(ProviderErrorType.TIMEOUT, "offline timeout", retryable=True)
        return ModelOutput(
            content={
                "answer": "30 days",
                "citations": ["doc"],
                "confidence": 0.8,
                "abstain": False,
            },
            usage=TokenUsage(input_tokens=30, output_tokens=17),
            usage_source=UsageSource.PROVIDER_REPORTED,
        )


def test_live_methodology_stop_blocks_before_provider_identity_lookup() -> None:
    with patch("collectiveeval.phase8_corrective.discover_local_model_identity") as discover:
        with pytest.raises(ValueError, match="Unresolved live methodology blocker"):
            phase8_corrective.execute({"methodology_stop": {"unresolved": True}})
        discover.assert_not_called()


def test_blocked_launcher_preserves_saved_run_identity(tmp_path: Path) -> None:
    status_path = tmp_path / "phase8_2_status.json"
    status_path.write_text(json.dumps({"run_ids": ["saved-run"], "status": "BLOCKED_METHODOLOGY"}))
    preflight = {"methodology_stop": {"unresolved": True}}
    with (
        patch.object(phase8_corrective, "CORRECTIVE", tmp_path),
        patch.object(phase8_corrective, "prepare", return_value=preflight),
        patch("sys.argv", ["run_phase8_2.py", "--execute"]),
        pytest.raises(ValueError, match="Unresolved live methodology blocker"),
    ):
        phase8_corrective.main()
    status = json.loads(status_path.read_text())
    assert status["run_ids"] == ["saved-run"]
    assert status["status"] == "BLOCKED_METHODOLOGY"


def test_retry_must_not_bypass_one_call_scientific_ceiling() -> None:
    example = BenchmarkExample(
        id="dev-retry-fixture",
        task_type=TaskType.GROUNDED_QA,
        input={
            "question": "deadline?",
            "source_text": "30 days",
            "evidence_units": [{"id": "doc", "text": "30 days"}],
        },
        gold={"answer": "30 days", "answerable": True, "evidence": ["doc"]},
        metadata={"source": "unit-fixture", "difficulty": "hard", "split": "dev", "tags": ["qa"]},
    )
    provider = RetryOnceProvider()
    ledger = BudgetLedger(InferenceBudget(max_calls=1, max_total_tokens=4000), run_id="offline")
    context = StrategyContext({"ollama": provider}, ledger, max_retries=1, scientific_mode=True)
    with suppress(BudgetExceeded):
        asyncio.run(
            context.call_model(
                example=example,
                model=ModelSpec(provider="ollama", model="gemma3:4b", max_tokens=700),
                strategy="single_agent",
                role="generator",
            )
        )
    assert provider.attempts == 1
    assert ledger.model_calls == 1
    assert ledger.records[0].normalized_error == "TIMEOUT"
    assert ledger.records[0].metadata["usage_unavailable"]
    assert ledger.provider_attempts == ledger.logical_model_calls == 1
    assert ledger.unknown_usage_stop
    assert ledger.records[0].input_tokens is None and ledger.records[0].output_tokens is None
