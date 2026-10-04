"""Strategy execution context that enforces central provider accounting."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from collectiveeval.budget import (
    BUDGET_SEMANTICS_VERSION,
    BudgetExceeded,
    BudgetLedger,
    CallReservation,
    LogicalCallRecord,
    ModelCallRecord,
    UnknownUsageStop,
)
from collectiveeval.core import (
    BenchmarkExample,
    ModelOutput,
    ModelSpec,
    ProviderErrorType,
    ProviderRequest,
    UsageSource,
)
from collectiveeval.datasets import validate_against_json_schema
from collectiveeval.parsing import OutputParseError, parse_provider_output, validate_task_output
from collectiveeval.providers import (
    ProviderError,
    ProviderRegistry,
    base_url_class,
    estimate_tokens,
)
from collectiveeval.task_contracts import (
    CRITIC_CONTRACT_VERSION,
    CRITIC_OUTPUT_SCHEMA,
    contract_for_example,
)


class StrategyContext:
    """Provider registry plus budget ledger for one strategy run."""

    def __init__(
        self,
        providers: ProviderRegistry,
        ledger: BudgetLedger | None = None,
        *,
        max_retries: int = 2,
        concurrency_limit: int = 1,
        scientific_mode: bool = False,
        attempt_sink: Callable[[ModelCallRecord], None] | None = None,
        logical_call_sink: Callable[[LogicalCallRecord], None] | None = None,
    ) -> None:
        self.providers = providers
        self.ledger = ledger or BudgetLedger()
        self.max_retries = max_retries
        self.concurrency_limit = max(1, concurrency_limit)
        self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        self._ledger_lock = asyncio.Lock()
        self.scientific_mode = scientific_mode
        self.attempt_sink = attempt_sink
        self.logical_call_sink = logical_call_sink

    def has_remaining_call(self) -> bool:
        return self.ledger.has_remaining_call()

    async def call_model(
        self,
        *,
        example: BenchmarkExample,
        model: ModelSpec,
        strategy: str,
        role: str,
        round_index: int = 0,
        candidate: dict[str, Any] | None = None,
        peer_summaries: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ModelOutput:
        async with self._semaphore:
            return await self._call_model_locked(
                example=example,
                model=model,
                strategy=strategy,
                role=role,
                round_index=round_index,
                candidate=candidate,
                peer_summaries=peer_summaries,
                extra=extra,
            )

    async def _call_model_locked(
        self,
        *,
        example: BenchmarkExample,
        model: ModelSpec,
        strategy: str,
        role: str,
        round_index: int = 0,
        candidate: dict[str, Any] | None = None,
        peer_summaries: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ModelOutput:
        contract = contract_for_example(example)
        is_critic = role == "critic"
        expected_schema = CRITIC_OUTPUT_SCHEMA if is_critic else contract.output_schema
        prompt_version = (
            f"{contract.prompt_version}.{CRITIC_CONTRACT_VERSION}"
            if is_critic
            else contract.prompt_version
        )

        request = ProviderRequest(
            example=example,
            model=model,
            strategy=strategy,
            role=role,
            round_index=round_index,
            candidate=candidate,
            peer_summaries=peer_summaries or [],
            extra=extra or {},
            prompt=contract.build_prompt(
                example,
                role=role,
                candidate=candidate,
                peer_summaries=peer_summaries or [],
                extra=extra or {},
            ),
            prompt_version=prompt_version,
            expected_schema=expected_schema,
        )
        estimated_input_tokens = estimate_tokens(request.prompt)
        if model.provider not in self.providers:
            raise ValueError(f"provider is not registered: {model.provider}")
        provider = self.providers[model.provider]
        logical = LogicalCallRecord(
            run_id=self.ledger.run_id,
            example_id=example.id,
            strategy=strategy,
            role=role,
            start_timestamp=datetime.now(UTC).isoformat(),
        )
        logical_started = time.perf_counter()
        attempt_index = 0
        try:
            while True:
                async with self._ledger_lock:
                    try:
                        reservation = self.ledger.reserve_call_start(
                            estimated_input_tokens, model.max_tokens
                        )
                    except UnknownUsageStop:
                        raise
                    except BudgetExceeded:
                        self.ledger.budget_events[-1].update(
                            {"example_id": example.id, "strategy": strategy, "role": role}
                        )
                        raise
                    remaining_before_call = self.ledger.remaining_dict()
                    if logical.logical_call_id not in self.ledger.logical_calls:
                        self.ledger.logical_calls[logical.logical_call_id] = logical
                        if self.logical_call_sink:
                            self.logical_call_sink(logical)
                    attempt = ModelCallRecord(
                        logical_call_id=logical.logical_call_id,
                        attempt_index=attempt_index,
                        run_id=self.ledger.run_id,
                        example_id=example.id,
                        strategy=strategy,
                        provider=model.provider,
                        model=model.model,
                        role=role,
                        prompt_version=prompt_version,
                        requested_seed=model.seed,
                        temperature=model.temperature,
                        max_tokens=reservation.effective_max_tokens,
                        start_timestamp=datetime.now(UTC).isoformat(),
                        input_tokens=None,
                        output_tokens=None,
                        usage_source=None,
                        usage_status="UNKNOWN_NOT_RETURNED",
                        outcome="STARTED",
                        latency_ms=0.0,
                        estimated_cost_usd=0.0,
                        finish_reason="pending",
                        metadata={
                            **self._admission_metadata(reservation, remaining_before_call),
                            "base_url_class": base_url_class(model.base_url),
                            "model_digest": model.provider_options.get("model_digest"),
                            "model_identity": model.provider_options.get("model_identity"),
                            "generation_settings": {
                                "temperature": model.temperature,
                                "top_p": model.top_p,
                                "max_tokens": reservation.effective_max_tokens,
                                "seed": model.seed,
                            },
                            "round_index": round_index,
                            "unknown_usage_policy": "UNKNOWN_USAGE_STOP.v1"
                            if self.scientific_mode
                            else "GENERAL_RETRY",
                        },
                    )
                    self.ledger.start_attempt(attempt, reservation)
                    if self.attempt_sink:
                        self.attempt_sink(attempt)
                outgoing = request.model_copy(
                    update={
                        "model": model.model_copy(
                            update={"max_tokens": reservation.effective_max_tokens}
                        )
                    }
                )
                started = time.perf_counter()
                try:
                    response = await provider.complete(outgoing)
                except BaseException as exc:
                    retryable = isinstance(exc, ProviderError) and exc.retryable
                    error_type = (
                        exc.error_type
                        if isinstance(exc, ProviderError)
                        else ProviderErrorType.UNKNOWN_PROVIDER_ERROR
                    )
                    outcome = (
                        "TIMEOUT"
                        if error_type == ProviderErrorType.TIMEOUT
                        else (
                            "INTERRUPTED"
                            if isinstance(exc, asyncio.CancelledError)
                            else "RETRYABLE_ERROR"
                            if retryable
                            else "NONRETRYABLE_ERROR"
                        )
                    )
                    failed = attempt.model_copy(
                        update={
                            "end_timestamp": datetime.now(UTC).isoformat(),
                            "latency_ms": (time.perf_counter() - started) * 1000,
                            "outcome": outcome,
                            "finish_reason": "error",
                            "normalized_error": str(error_type),
                            "metadata": {
                                **attempt.metadata,
                                "usage_unavailable": True,
                                "failed_before_response": True,
                                "error": str(exc),
                            },
                        }
                    )
                    async with self._ledger_lock:
                        self.ledger.record_call(failed)
                        self.ledger.release_reserved_call(reservation)
                        if self.scientific_mode:
                            self.ledger.stop_unknown_usage(failed.attempt_id)
                        if self.attempt_sink:
                            self.attempt_sink(failed)
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    if self.scientific_mode:
                        raise UnknownUsageStop(
                            "UNKNOWN_USAGE_STOP: sent attempt returned no usage"
                        ) from exc
                    if not retryable or attempt_index >= self.max_retries:
                        raise
                    attempt_index += 1
                    await asyncio.sleep(min(0.05 * 2 ** (attempt_index - 1), 0.5))
                    continue

                # Downstream parse failure never erases the completed inference or its usage.
                parse_error: OutputParseError | None = None
                original_response = response
                try:
                    response = self._parse_response(request=outgoing, response=response)
                except OutputParseError as exc:
                    parse_error = exc
                    response = original_response.model_copy(
                        update={"parse_status": "PARSE_ERROR", "parse_error": str(exc)}
                    )
                response = self._account_response(
                    request=outgoing,
                    response=response,
                    estimated_input_tokens=estimated_input_tokens,
                )
                record = self._record_from_response(
                    request=outgoing,
                    response=response,
                    start_timestamp=attempt.start_timestamp or "",
                    retry_count=0,
                ).model_copy(
                    update={
                        "attempt_id": attempt.attempt_id,
                        "logical_call_id": logical.logical_call_id,
                        "attempt_index": attempt_index,
                        "usage_status": str(response.usage_source),
                        "latency_ms": max(
                            response.latency_ms, (time.perf_counter() - started) * 1000
                        ),
                        "outcome": "PARSE_ERROR_AFTER_RESPONSE" if parse_error else "SUCCESS",
                    }
                )
                record.metadata.update(attempt.metadata)
                record.metadata["provider_response_latency_ms"] = response.latency_ms
                if parse_error:
                    record.normalized_error = str(ProviderErrorType.PARSE_ERROR)
                    record.metadata.update(
                        {
                            "parse_error": str(parse_error),
                            "raw_output": original_response.raw_output,
                        }
                    )
                async with self._ledger_lock:
                    self.ledger.record_call(record)
                    self.ledger.release_reserved_call(reservation)
                    if self.attempt_sink:
                        self.attempt_sink(record)
                if parse_error:
                    raise ProviderError(
                        ProviderErrorType.PARSE_ERROR, str(parse_error), retryable=False
                    ) from parse_error
                return response
        finally:
            if logical.logical_call_id in self.ledger.logical_calls:
                logical.end_timestamp = datetime.now(UTC).isoformat()
                logical.lifecycle_latency_ms = (time.perf_counter() - logical_started) * 1000
                attempts = [
                    r for r in self.ledger.records if r.logical_call_id == logical.logical_call_id
                ]
                logical.outcome = (
                    "UNKNOWN_USAGE_STOP"
                    if self.ledger.unknown_usage_stop
                    else (attempts[-1].outcome if attempts else "REJECTED")
                )
                if self.logical_call_sink:
                    self.logical_call_sink(logical)

    def _parse_response(
        self,
        *,
        request: ProviderRequest,
        response: ModelOutput,
    ) -> ModelOutput:
        contract = contract_for_example(request.example)
        if request.role == "critic":
            if not response.content:
                if not response.raw_output:
                    raise OutputParseError("PARSE_ERROR: empty critic output")

                raw = response.raw_output.strip()
                cleaned = raw

                # Local instruction models commonly wrap otherwise-valid JSON
                # in Markdown fences.
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines and lines[0].strip().startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()

                try:
                    critique = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Also tolerate a short natural-language prefix/suffix
                    # around one valid JSON object.
                    start = cleaned.find("{")
                    if start < 0:
                        raise OutputParseError(
                            "PARSE_ERROR: malformed critic JSON; no JSON object found"
                        ) from None
                    try:
                        critique, _ = json.JSONDecoder().raw_decode(cleaned[start:])
                    except json.JSONDecodeError as exc:
                        raise OutputParseError("PARSE_ERROR: malformed critic JSON") from exc

                if not isinstance(critique, dict):
                    raise OutputParseError("PARSE_ERROR: critic output must be an object")

                response = response.model_copy(update={"content": critique})

            if "needs_revision" not in response.content or not isinstance(
                response.content["needs_revision"], bool
            ):
                raise OutputParseError("PARSE_ERROR: critic output requires boolean needs_revision")

            issues = response.content.get("issues")
            if not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
                raise OutputParseError(
                    "PARSE_ERROR: critic output requires issues as an array of strings"
                )

            if response.raw_output is None:
                response = response.model_copy(
                    update={"raw_output": json.dumps(response.content, ensure_ascii=False)}
                )
            schema_issues = validate_against_json_schema(
                response.content, CRITIC_OUTPUT_SCHEMA, prefix="critic"
            )
            if schema_issues:
                raise OutputParseError("PARSE_ERROR: " + "; ".join(schema_issues))
        elif response.raw_output and not response.content:
            parsed = parse_provider_output(
                request.example,
                response.raw_output,
                contract=contract,
                allow_repair=True,
            )
            response = response.model_copy(
                update={
                    "content": parsed.content,
                    "parse_status": parsed.parse_status,
                    "parse_error": None,
                }
            )
        else:
            validate_task_output(request.example, response.content, contract)
            if response.raw_output is None:
                response = response.model_copy(update={"raw_output": str(response.content)})

        return response

    def _account_response(
        self, *, request: ProviderRequest, response: ModelOutput, estimated_input_tokens: int
    ) -> ModelOutput:
        usage_source = response.usage_source
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        if (
            usage_source != UsageSource.PROVIDER_REPORTED
            or input_tokens is None
            or output_tokens is None
        ):
            input_tokens = estimated_input_tokens
            output_tokens = estimate_tokens(response.raw_output or response.content)
            usage_source = UsageSource.ESTIMATED

        estimated_cost = response.estimated_cost_usd
        if usage_source == UsageSource.ESTIMATED or estimated_cost == 0.0:
            estimated_cost = (
                input_tokens * request.model.input_cost_per_1k
                + output_tokens * request.model.output_cost_per_1k
            ) / 1000

        return response.model_copy(
            update={
                "usage": response.usage.model_copy(
                    update={"input_tokens": input_tokens, "output_tokens": output_tokens}
                ),
                "usage_source": usage_source,
                "estimated_cost_usd": estimated_cost,
                "provider": request.model.provider,
                "model": request.model.model,
            }
        )

    def _admission_metadata(
        self, reservation: CallReservation, remaining: dict[str, int | float | None]
    ) -> dict[str, Any]:
        return {
            "budget_semantics_version": BUDGET_SEMANTICS_VERSION,
            "configured_max_tokens": reservation.configured_max_tokens,
            "effective_max_tokens": reservation.effective_max_tokens,
            "admission_estimated_input_tokens": reservation.estimated_input_tokens,
            "budget_remaining_before_call": remaining,
            "pre_call_budget_rejected": False,
        }

    def _record_from_response(
        self,
        *,
        request: ProviderRequest,
        response: ModelOutput,
        start_timestamp: str,
        retry_count: int,
    ) -> ModelCallRecord:
        return ModelCallRecord(
            run_id=self.ledger.run_id,
            example_id=request.example.id,
            strategy=request.strategy,
            provider=request.model.provider,
            model=request.model.model,
            role=request.role,
            prompt_version=request.prompt_version,
            requested_seed=request.model.seed,
            temperature=request.model.temperature,
            max_tokens=request.model.max_tokens,
            start_timestamp=start_timestamp,
            end_timestamp=datetime.now(UTC).isoformat(),
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
            usage_source=str(response.usage_source or UsageSource.ESTIMATED),
            latency_ms=response.latency_ms,
            estimated_cost_usd=response.estimated_cost_usd,
            retry_count=retry_count,
            finish_reason=response.finish_reason,
            metadata={
                "actual_input_tokens": response.usage.input_tokens,
                "actual_output_tokens": response.usage.output_tokens,
                "actual_total_tokens": response.usage.total_tokens,
                "round_index": request.round_index,
                "parse_status": str(response.parse_status),
                "raw_output_present": response.raw_output is not None,
                "structured_output": response.content,
                "base_url_class": base_url_class(request.model.base_url),
                "model_identity": request.model.provider_options.get("model_identity"),
                "model_digest": request.model.provider_options.get("model_digest"),
                "generation_settings": {
                    "temperature": request.model.temperature,
                    "top_p": request.model.top_p,
                    "max_tokens": request.model.max_tokens,
                    "seed": request.model.seed,
                },
            },
        )
