"""Central inference budget accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

BUDGET_SEMANTICS_VERSION = "provider-attempts-unknown-stop.v2"
SCIENTIFIC_ATTEMPT_POLICY = "UNKNOWN_USAGE_STOP.v1"


@dataclass(frozen=True)
class CallReservation:
    """Admission bounds held while a call is in flight, not consumed usage."""

    reservation_id: int
    estimated_input_tokens: int
    configured_max_tokens: int | None
    effective_max_tokens: int | None
    remaining_total_tokens: int | None


class BudgetExceeded(RuntimeError):
    """Raised when a provider call would exceed an inference budget."""


class UnknownUsageStop(BudgetExceeded):
    """Sent request has unknown consumption; scientific inference must stop."""


class LogicalCallRecord(BaseModel):
    logical_call_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    example_id: str
    strategy: str
    role: str
    start_timestamp: str
    end_timestamp: str | None = None
    lifecycle_latency_ms: float = 0.0
    outcome: str = "STARTED"


class InferenceBudget(BaseModel):
    """Hard ceilings applied to a strategy run."""

    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_calls: int | None = None
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None


class ModelCallRecord(BaseModel):
    """One provider attempt; the name is retained for the legacy read interface."""

    attempt_id: str = Field(default_factory=lambda: str(uuid4()))
    logical_call_id: str = Field(default_factory=lambda: str(uuid4()))
    attempt_index: int = 0
    outcome: str = "SUCCESS"
    usage_status: str = "ESTIMATED"
    run_id: str
    example_id: str
    strategy: str
    provider: str
    model: str
    role: str
    prompt_version: str = ""
    requested_seed: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    input_tokens: int | None
    output_tokens: int | None
    usage_source: str | None = "ESTIMATED"
    latency_ms: float
    estimated_cost_usd: float
    retry_count: int = 0
    finish_reason: str
    normalized_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BudgetLedger:
    """Mutable central ledger used by strategies through `StrategyContext`."""

    def __init__(self, budget: InferenceBudget | None = None, run_id: str = "local") -> None:
        self.budget = budget or InferenceBudget()
        self.run_id = run_id
        self.records: list[ModelCallRecord] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_ms = 0.0
        self.estimated_cost_usd = 0.0
        self._reservations: dict[int, CallReservation] = {}
        self._next_reservation_id = 0
        self.budget_events: list[dict[str, Any]] = []
        self.overrun_details: list[dict[str, Any]] = []
        self.logical_calls: dict[str, LogicalCallRecord] = {}
        self._sent_reservations: set[int] = set()
        self.unknown_usage_stop = False

    @property
    def reserved_calls(self) -> int:
        return len(self._reservations)

    @property
    def model_calls(self) -> int:
        """Legacy alias for provider_attempts, never logical generations."""
        return len(self.records)

    @property
    def provider_attempts(self) -> int:
        return len(self.records)

    @property
    def logical_model_calls(self) -> int:
        return len({record.logical_call_id for record in self.records})

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def usage_dict(self) -> dict[str, int | float]:
        return {
            "model_calls": self.model_calls,
            "provider_attempts": self.provider_attempts,
            "logical_model_calls": self.logical_model_calls,
            "successful_attempts": sum(
                r.outcome in {"SUCCESS", "PARSE_ERROR_AFTER_RESPONSE"} for r in self.records
            ),
            "failed_attempts": sum(
                r.outcome not in {"SUCCESS", "PARSE_ERROR_AFTER_RESPONSE", "STARTED"}
                for r in self.records
            ),
            "unknown_usage_attempts": sum(
                r.usage_status == "UNKNOWN_NOT_RETURNED" for r in self.records
            ),
            "known_token_lower_bound": sum(
                (r.input_tokens or 0) + (r.output_tokens or 0)
                for r in self.records
                if r.usage_source == "PROVIDER_REPORTED"
            ),
            "logical_call_latency_ms": sum(
                r.lifecycle_latency_ms for r in self.logical_calls.values()
            ),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    def remaining_dict(self) -> dict[str, int | float | None]:
        return {
            "remaining_calls": (
                None
                if self.budget.max_calls is None
                else max(0, self.budget.max_calls - self.model_calls)
            ),
            "remaining_input_tokens": (
                None
                if self.budget.max_input_tokens is None
                else max(0, self.budget.max_input_tokens - self.input_tokens)
            ),
            "remaining_output_tokens": (
                None
                if self.budget.max_output_tokens is None
                else max(0, self.budget.max_output_tokens - self.output_tokens)
            ),
            "remaining_total_tokens": (
                None
                if self.budget.max_total_tokens is None
                else max(0, self.budget.max_total_tokens - self.total_tokens)
            ),
            "remaining_estimated_cost_usd": (
                None
                if self.budget.max_cost_usd is None
                else max(0.0, self.budget.max_cost_usd - self.estimated_cost_usd)
            ),
        }

    def has_remaining_call(self) -> bool:
        return (
            not self.unknown_usage_stop
            and not self.overrun_details
            and (
                self.budget.max_calls is None
                or self.model_calls + self.reserved_calls - len(self._sent_reservations)
                < self.budget.max_calls
            )
            and (
                self.budget.max_total_tokens is None
                or self.total_tokens < self.budget.max_total_tokens
            )
            and (
                self.budget.max_output_tokens is None
                or self.output_tokens < self.budget.max_output_tokens
            )
        )

    def _call_allowance(
        self, estimated_input_tokens: int, configured_max_output: int | None
    ) -> tuple[int | None, int | None]:
        if estimated_input_tokens < 0:
            raise ValueError("estimated input tokens must be nonnegative")
        if self.unknown_usage_stop:
            raise UnknownUsageStop("UNKNOWN_USAGE_STOP: previous attempt consumption is unknown")
        if self.overrun_details:
            raise BudgetExceeded("post-call budget overrun blocks further calls")
        reserved_input = sum(r.estimated_input_tokens for r in self._reservations.values())
        reserved_output = sum(r.effective_max_tokens or 0 for r in self._reservations.values())
        remaining: int | None = None
        allowances = [] if configured_max_output is None else [configured_max_output]
        if self.budget.max_input_tokens is not None and (
            self.input_tokens + reserved_input + estimated_input_tokens
            > self.budget.max_input_tokens
        ):
            raise BudgetExceeded("input token budget exhausted")
        if self.budget.max_total_tokens is not None:
            remaining = self.budget.max_total_tokens - self.total_tokens
            available = remaining - reserved_input - reserved_output - estimated_input_tokens
            if available <= 0:
                raise BudgetExceeded("total token budget exhausted")
            allowances.append(available)
        if self.budget.max_output_tokens is not None:
            allowances.append(self.budget.max_output_tokens - self.output_tokens - reserved_output)
        effective = min(allowances) if allowances else None
        if effective is not None and effective <= 0:
            raise BudgetExceeded("output token budget exhausted")
        if self.budget.max_latency_ms is not None and self.latency_ms >= self.budget.max_latency_ms:
            raise BudgetExceeded("latency budget exhausted")
        if (
            self.budget.max_cost_usd is not None
            and self.estimated_cost_usd > self.budget.max_cost_usd
        ):
            raise BudgetExceeded("cost budget exhausted")
        return effective, remaining

    def ensure_can_start_call(self, estimated_input_tokens: int = 0) -> None:
        if self.unknown_usage_stop:
            raise UnknownUsageStop("UNKNOWN_USAGE_STOP: previous attempt consumption is unknown")
        if (
            self.budget.max_calls is not None
            and self.model_calls + self.reserved_calls - len(self._sent_reservations) + 1
            > self.budget.max_calls
        ):
            raise BudgetExceeded("model call budget exhausted")
        self._call_allowance(estimated_input_tokens, None)

    def reserve_call_start(
        self, estimated_input_tokens: int = 0, configured_max_output: int | None = None
    ) -> CallReservation:
        try:
            self.ensure_can_start_call(estimated_input_tokens)
            effective, remaining = self._call_allowance(
                estimated_input_tokens, configured_max_output
            )
        except UnknownUsageStop:
            raise
        except BudgetExceeded as exc:
            self.budget_events.append(
                {
                    "event_type": "PRE_CALL_BUDGET_REJECTION",
                    "reason": str(exc),
                    "pre_call_budget_rejected": True,
                    "admission_estimated_input_tokens": estimated_input_tokens,
                    "configured_max_tokens": configured_max_output,
                    "budget_remaining_before_call": self.remaining_dict(),
                }
            )
            raise
        reservation = CallReservation(
            self._next_reservation_id,
            estimated_input_tokens,
            configured_max_output,
            effective,
            remaining,
        )
        self._next_reservation_id += 1
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def release_reserved_call(self, reservation: CallReservation | None = None) -> None:
        if reservation is not None:
            self._reservations.pop(reservation.reservation_id, None)
            self._sent_reservations.discard(reservation.reservation_id)
        elif self._reservations:
            self._reservations.pop(next(iter(self._reservations)))

    def start_attempt(self, record: ModelCallRecord, reservation: CallReservation) -> None:
        self.records.append(record)
        self._sent_reservations.add(reservation.reservation_id)

    def stop_unknown_usage(self, attempt_id: str) -> None:
        self.unknown_usage_stop = True
        self.budget_events.append(
            {
                "event_type": "UNKNOWN_USAGE_STOP",
                "attempt_id": attempt_id,
                "reason": "sent_attempt_usage_not_returned",
            }
        )

    def record_call(self, record: ModelCallRecord) -> None:
        before = self.usage_dict()
        existing_index = next(
            (i for i, r in enumerate(self.records) if r.attempt_id == record.attempt_id), None
        )
        projected_input = self.input_tokens + (record.input_tokens or 0)
        projected_output = self.output_tokens + (record.output_tokens or 0)
        projected_latency = self.latency_ms + record.latency_ms
        projected_cost = self.estimated_cost_usd + record.estimated_cost_usd
        projected_total = projected_input + projected_output

        checks: list[tuple[str, float | int, float | int | None]] = [
            (
                "provider attempt",
                self.provider_attempts + (existing_index is None),
                self.budget.max_calls,
            ),
            ("input token", projected_input, self.budget.max_input_tokens),
            ("output token", projected_output, self.budget.max_output_tokens),
            ("total token", projected_total, self.budget.max_total_tokens),
            ("latency", projected_latency, self.budget.max_latency_ms),
            ("cost", projected_cost, self.budget.max_cost_usd),
        ]
        # A completed call is evidence even if actual usage exceeds its admission estimate.
        self.input_tokens = projected_input
        self.output_tokens = projected_output
        self.latency_ms = projected_latency
        self.estimated_cost_usd = projected_cost
        if existing_index is None:
            self.records.append(record)
        else:
            self.records[existing_index] = record
        violations = [
            {
                "resource": label,
                "actual": projected,
                "ceiling": ceiling,
                "overrun": projected - ceiling,
            }
            for label, projected, ceiling in checks
            if ceiling is not None and projected > ceiling
        ]
        record.metadata["post_call_budget_overrun"] = bool(violations)
        if violations:
            overrun_tokens = (
                max(0, projected_total - self.budget.max_total_tokens)
                if self.budget.max_total_tokens is not None
                else 0
            )
            detail = {
                "event_type": "POST_CALL_BUDGET_OVERRUN",
                "budget_overrun": True,
                "budget_overrun_tokens": overrun_tokens,
                "budget_overrun_reason": "actual_provider_usage_exceeded_admission_estimate",
                "violations": violations,
            }
            self.overrun_details.append(detail)
            self.budget_events.append(detail)
            record.metadata.update(detail)
        record.metadata.setdefault(
            "budget_accounting",
            {
                "requested_budget": self.budget.model_dump(mode="json"),
                "consumed_before_call": before,
                "consumed_after_call": self.usage_dict(),
                "remaining_after_call": self.remaining_dict(),
                "usage_source": record.usage_source,
            },
        )

    def metadata_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "budget_semantics_version": BUDGET_SEMANTICS_VERSION,
            "budget_events": self.budget_events,
            "post_call_budget_overrun": bool(self.overrun_details),
            "model_calls_semantics": "legacy_alias_for_provider_attempts",
            "unknown_usage_stop": self.unknown_usage_stop,
            "stop_reason": "UNKNOWN_USAGE_STOP" if self.unknown_usage_stop else None,
            "total_provider_tokens": self.total_tokens
            if all(r.usage_source == "PROVIDER_REPORTED" for r in self.records)
            else None,
            "total_provider_tokens_exact": all(
                r.usage_source == "PROVIDER_REPORTED" for r in self.records
            ),
            **{
                k: v
                for k, v in self.usage_dict().items()
                if k
                in {
                    "logical_model_calls",
                    "provider_attempts",
                    "successful_attempts",
                    "failed_attempts",
                    "unknown_usage_attempts",
                    "known_token_lower_bound",
                    "logical_call_latency_ms",
                }
            },
        }
        if self.overrun_details:
            payload.update(
                {
                    "budget_overrun": True,
                    "budget_overrun_tokens": max(
                        0, self.total_tokens - self.budget.max_total_tokens
                    )
                    if self.budget.max_total_tokens is not None
                    else 0,
                    "budget_overrun_reason": "actual_provider_usage_exceeded_admission_estimate",
                }
            )
        return payload
