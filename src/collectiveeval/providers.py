"""Model provider interfaces and deterministic test provider."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

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


def estimate_tokens(value: Any) -> int:
    """Small deterministic token estimator suitable for budget tests."""

    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return max(1, (len(text) + 3) // 4)


class ModelProvider(ABC):
    """Abstract provider called only through a strategy context."""

    provider_id: str
    capabilities: ProviderCapabilities

    @abstractmethod
    async def complete(self, request: ProviderRequest) -> ModelOutput:
        """Return one structured model output."""


ProviderRegistry = dict[str, ModelProvider]


@dataclass(frozen=True)
class ProviderCapabilities:
    """Provider feature metadata strategies must not assume implicitly."""

    supports_seed: bool = False
    supports_usage: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = False
    supports_logprobs: bool = False
    supports_local_execution: bool = False


class ProviderError(RuntimeError):
    """Normalized provider error with retry metadata."""

    def __init__(
        self,
        error_type: ProviderErrorType,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable
        self.status_code = status_code


class MockProvider(ModelProvider):
    """Deterministic provider for tests and dry runs.

    The provider can replay `example.metadata["mock_outputs"]` entries. Each
    entry may either be a raw task output dict or an object with `content`,
    `confidence`, and related fields.
    """

    provider_id = "mock"
    capabilities = ProviderCapabilities(
        supports_seed=True,
        supports_usage=True,
        supports_json_mode=True,
    )

    def __init__(self, mock_mode: str = "fixture") -> None:
        mode_aliases = {"adversarial": "noisy", "mock-noisy": "noisy"}
        normalized_mode = mode_aliases.get(mock_mode, mock_mode)
        if normalized_mode not in {"fixture", "noisy", "gold_fixture"}:
            raise ValueError(f"unknown mock_mode: {mock_mode}")
        self.mock_mode = normalized_mode
        self._counters: dict[tuple[str, str], int] = defaultdict(int)

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        started = time.perf_counter()
        await asyncio.sleep(0)

        content, confidence, finish_reason = self._select_response(request)
        input_tokens = estimate_tokens(request.model_dump(mode="json", exclude={"example"}))
        input_tokens += estimate_tokens(request.example.model_dump(mode="json"))
        output_tokens = estimate_tokens(content)
        latency_ms = (time.perf_counter() - started) * 1000
        estimated_cost = (
            input_tokens * request.model.input_cost_per_1k
            + output_tokens * request.model.output_cost_per_1k
        ) / 1000

        return ModelOutput(
            content=content,
            raw_output=json.dumps(content, ensure_ascii=False, sort_keys=True),
            confidence=confidence,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            usage_source=UsageSource.PROVIDER_REPORTED,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost,
            finish_reason=finish_reason,
            provider=request.model.provider,
            model=request.model.model,
        )

    def _select_response(self, request: ProviderRequest) -> tuple[dict[str, Any], float, str]:
        key = (request.example.id, request.strategy)
        call_index = self._counters[key]
        self._counters[key] += 1

        replay = request.example.metadata.get("mock_outputs")
        if isinstance(replay, list) and replay:
            raw = replay[min(call_index, len(replay) - 1)]
            if isinstance(raw, dict) and "content" in raw:
                return (
                    dict(raw["content"]),
                    float(raw.get("confidence", 0.5)),
                    str(raw.get("finish_reason", "stop")),
                )
            if isinstance(raw, dict):
                return dict(raw), float(raw.get("confidence", 0.5)), "stop"

        if request.role == "critic":
            return self._critic_response(request), 0.8, "stop"
        if request.role == "reviser":
            if self.mock_mode == "gold_fixture":
                return self._gold_response(request.example), 0.88, "stop"
            return self._fixture_response(request.example), 0.62, "stop"

        if "weak" in request.model.model or "wrong" in request.model.model:
            return self._weak_response(request.example), 0.24, "stop"
        if "uncertain" in request.model.model:
            output = self._fixture_response(request.example)
            output["citations"] = []
            return output, 0.35, "stop"

        if self.mock_mode == "gold_fixture":
            return self._gold_response(request.example), 0.9, "stop"
        if self.mock_mode == "noisy":
            return self._weak_response(request.example), 0.31, "stop"
        return self._fixture_response(request.example), 0.7, "stop"

    def _critic_response(self, request: ProviderRequest) -> dict[str, Any]:
        candidate = request.candidate or {}
        issues = []
        valid = True
        if "answer" in candidate and not isinstance(candidate.get("answer"), str):
            valid = False
            issues.append("answer_must_be_string")
        if request.example.task_type == TaskType.GROUNDED_QA and not candidate.get("citations"):
            valid = False
            issues.append("missing_citation")
        return {"valid": valid, "needs_revision": not valid, "issues": issues}

    def _weak_response(self, example: BenchmarkExample) -> dict[str, Any]:
        if example.task_type == TaskType.GROUNDED_QA:
            return {"answer": "不明", "citations": [], "abstain": False, "confidence": 0.24}
        if example.task_type == TaskType.STRUCTURED_EXTRACTION:
            return _schema_placeholder_response(example, weak=True)
        if example.task_type == TaskType.BUSINESS_SUMMARIZATION:
            return {"summary": "要確認", "decisions": [], "action_items": [], "risks": []}
        return {"answer": "不明", "citations": [], "abstain": False, "confidence": 0.24}

    def _fixture_response(self, example: BenchmarkExample) -> dict[str, Any]:
        if example.task_type == TaskType.GROUNDED_QA:
            return {"answer": "不明", "citations": [], "abstain": True, "confidence": 0.7}
        if example.task_type == TaskType.STRUCTURED_EXTRACTION:
            return _schema_placeholder_response(example)
        if example.task_type == TaskType.BUSINESS_SUMMARIZATION:
            return {"summary": "要確認", "decisions": [], "action_items": [], "risks": []}
        return {"answer": "不明", "citations": [], "abstain": True, "confidence": 0.7}

    def _gold_response(self, example: BenchmarkExample) -> dict[str, Any]:
        if example.task_type == TaskType.GROUNDED_QA:
            answerable = bool(example.gold.get("answerable", True))
            if not answerable:
                return {"answer": "", "citations": [], "abstain": True, "confidence": 0.9}
            return {
                "answer": str(example.gold.get("answer", "")),
                "citations": list(_gold_citations(example)),
                "abstain": False,
                "confidence": 0.9,
            }
        if example.task_type == TaskType.STRUCTURED_EXTRACTION:
            expected = example.gold.get("expected", example.gold)
            return dict(expected)
        if example.task_type == TaskType.BUSINESS_SUMMARIZATION:
            return {
                "summary": str(example.gold.get("summary", "")),
                "decisions": list(example.gold.get("decisions", [])),
                "action_items": list(example.gold.get("action_items", [])),
                "risks": list(example.gold.get("risks", [])),
            }
        return {
            "answer": str(example.gold.get("answer", "")),
            "citations": list(_gold_citations(example)),
            "abstain": bool(example.gold.get("abstain", False)),
            "confidence": 0.9,
        }


class OpenAICompatibleProvider(ModelProvider):
    """Generic OpenAI-compatible chat/completions provider."""

    provider_id = "openai"
    capabilities = ProviderCapabilities(
        supports_seed=True,
        supports_usage=True,
        supports_json_mode=True,
    )

    def __init__(self, *, provider_id: str = "openai", base_url: str | None = None) -> None:
        self.provider_id = provider_id
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.capabilities = ProviderCapabilities(
            supports_seed=True,
            supports_usage=True,
            supports_json_mode=True,
            supports_local_execution=base_url_class(self.base_url) == "local",
        )

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        started = time.perf_counter()
        payload = _openai_payload(request)
        headers = self._headers(request.model)
        response = await self._post_json(
            f"{request.model.base_url or self.base_url}/chat/completions",
            payload,
            headers=headers,
            timeout_s=request.model.timeout_s,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return normalize_openai_response(
            response,
            request=request,
            latency_ms=latency_ms,
            provider_id=self.provider_id,
        )

    async def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
        timeout_s: float,
    ) -> dict[str, Any]:
        def send() -> dict[str, Any]:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as response:
                    return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
            except TimeoutError as exc:
                raise ProviderError(
                    ProviderErrorType.TIMEOUT,
                    "provider request timed out",
                    retryable=True,
                ) from exc
            except urllib.error.HTTPError as exc:
                raise _http_error_to_provider_error(exc) from exc
            except urllib.error.URLError as exc:
                raise ProviderError(
                    ProviderErrorType.PROVIDER_UNAVAILABLE,
                    "provider endpoint unavailable",
                    retryable=True,
                ) from exc

        return await asyncio.to_thread(send)

    def _headers(self, model: ModelSpec) -> dict[str, str]:
        effective_base_url = model.base_url or self.base_url
        if base_url_class(effective_base_url) == "local":
            return {
                "Authorization": "Bearer collectiveeval-local-placeholder",
                "Content-Type": "application/json",
            }
        api_key_env = model.api_key_env or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderError(
                ProviderErrorType.AUTH_ERROR,
                f"missing API key environment variable {api_key_env}",
                retryable=False,
            )
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


class VLLMCompatibleProvider(OpenAICompatibleProvider):
    """vLLM OpenAI-compatible endpoint provider."""

    provider_id = "vllm"
    capabilities = ProviderCapabilities(
        supports_seed=True,
        supports_usage=True,
        supports_json_mode=True,
    )

    def __init__(self, *, base_url: str | None = None) -> None:
        super().__init__(provider_id="vllm", base_url=base_url or "http://localhost:8000/v1")


class HuggingFaceLocalProvider(ModelProvider):
    """Local Hugging Face causal language model provider with lazy loading."""

    provider_id = "huggingface"
    capabilities = ProviderCapabilities(
        supports_seed=True,
        supports_usage=True,
        supports_local_execution=True,
    )

    def __init__(self) -> None:
        self._loaded: dict[str, Any] | None = None

    async def complete(self, request: ProviderRequest) -> ModelOutput:
        started = time.perf_counter()
        tokenizer, model = await asyncio.to_thread(self._load, request.model)
        raw_output = await asyncio.to_thread(self._generate, tokenizer, model, request)
        input_tokens = len(tokenizer.encode(request.prompt))
        output_tokens = len(tokenizer.encode(raw_output))
        latency_ms = (time.perf_counter() - started) * 1000
        return ModelOutput(
            raw_output=raw_output,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            usage_source=UsageSource.PROVIDER_REPORTED,
            latency_ms=latency_ms,
            provider=self.provider_id,
            model=request.model.model,
        )

    def _load(self, model_spec: ModelSpec) -> tuple[Any, Any]:
        if self._loaded is not None:
            return self._loaded["tokenizer"], self._loaded["model"]
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ProviderError(
                ProviderErrorType.PROVIDER_UNAVAILABLE,
                "transformers is not installed; install provider extras to use Hugging Face",
                retryable=False,
            ) from exc
        options = model_spec.provider_options
        tokenizer = AutoTokenizer.from_pretrained(model_spec.model)
        model = AutoModelForCausalLM.from_pretrained(
            model_spec.model,
            torch_dtype=options.get("dtype"),
            device_map=options.get("device"),
        )
        self._loaded = {"tokenizer": tokenizer, "model": model}
        return tokenizer, model

    def _generate(self, tokenizer: Any, model: Any, request: ProviderRequest) -> str:
        if request.model.seed is not None:
            random.seed(request.model.seed)
        inputs = tokenizer(request.prompt, return_tensors="pt")
        generated = model.generate(
            **inputs,
            max_new_tokens=request.model.max_tokens or 256,
            do_sample=request.model.temperature > 0,
            temperature=max(request.model.temperature, 1e-6),
            top_p=request.model.top_p or 1.0,
        )
        return str(tokenizer.decode(generated[0], skip_special_tokens=True))


def normalize_openai_response(
    response: dict[str, Any],
    *,
    request: ProviderRequest,
    latency_ms: float,
    provider_id: str,
) -> ModelOutput:
    choices = response.get("choices") or []
    if not choices:
        raise ProviderError(
            ProviderErrorType.UNKNOWN_PROVIDER_ERROR,
            "provider response did not include choices",
            retryable=False,
        )
    message = choices[0].get("message", {})
    raw_output = str(message.get("content", ""))
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if provider_id == "ollama" and (prompt_tokens is None or completion_tokens is None):
        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
    token_usage = TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )
    usage_source = (
        UsageSource.PROVIDER_REPORTED
        if token_usage.input_tokens is not None and token_usage.output_tokens is not None
        else None
    )
    estimated_cost = 0.0
    if usage_source == UsageSource.PROVIDER_REPORTED:
        input_tokens = token_usage.input_tokens
        output_tokens = token_usage.output_tokens
        assert input_tokens is not None
        assert output_tokens is not None
        estimated_cost = (
            input_tokens * request.model.input_cost_per_1k
            + output_tokens * request.model.output_cost_per_1k
        ) / 1000
    return ModelOutput(
        raw_output=raw_output,
        usage=token_usage,
        usage_source=usage_source,
        latency_ms=latency_ms,
        estimated_cost_usd=estimated_cost,
        finish_reason=str(choices[0].get("finish_reason", "stop")),
        provider=provider_id,
        model=request.model.model,
    )


def build_provider_registry(config: dict[str, Any]) -> ProviderRegistry:
    """Build providers required by an experiment config."""

    provider_names = _provider_names_from_config(config)
    registry: ProviderRegistry = {}
    for provider_name in provider_names:
        if provider_name == "mock":
            registry[provider_name] = MockProvider(mock_mode=_mock_mode_from_config(config))
        elif provider_name in {"openai", "openai_compatible", "ollama"}:
            registry[provider_name] = OpenAICompatibleProvider(
                provider_id=provider_name, base_url=_provider_base_url(config, provider_name)
            )
        elif provider_name == "vllm":
            registry[provider_name] = VLLMCompatibleProvider(
                base_url=_provider_base_url(config, provider_name)
            )
        elif provider_name in {"huggingface", "hf"}:
            registry[provider_name] = HuggingFaceLocalProvider()
        else:
            raise ValueError(f"unknown provider: {provider_name}")
    return registry


def _openai_payload(request: ProviderRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": request.model.temperature,
    }
    if request.model.max_tokens is not None:
        payload["max_tokens"] = request.model.max_tokens
    if request.model.seed is not None:
        payload["seed"] = request.model.seed
    if request.model.top_p is not None:
        payload["top_p"] = request.model.top_p
    return payload


def _http_error_to_provider_error(exc: urllib.error.HTTPError) -> ProviderError:
    status = exc.code
    if status in {401, 403}:
        return ProviderError(
            ProviderErrorType.AUTH_ERROR,
            "provider authentication failed",
            status_code=status,
        )
    if status == 429:
        return ProviderError(
            ProviderErrorType.RATE_LIMIT,
            "provider rate limit",
            retryable=True,
            status_code=status,
        )
    if status == 400:
        return ProviderError(
            ProviderErrorType.INVALID_REQUEST,
            "provider rejected request",
            status_code=status,
        )
    if status == 413:
        return ProviderError(
            ProviderErrorType.CONTEXT_LENGTH,
            "context length exceeded",
            status_code=status,
        )
    if 500 <= status < 600:
        return ProviderError(
            ProviderErrorType.PROVIDER_UNAVAILABLE,
            "provider unavailable",
            retryable=True,
            status_code=status,
        )
    return ProviderError(
        ProviderErrorType.UNKNOWN_PROVIDER_ERROR,
        "provider request failed",
        status_code=status,
    )


def _provider_names_from_config(config: dict[str, Any]) -> set[str]:
    names = {str(dict(config.get("model", {})).get("provider", "mock"))}
    strategy = dict(config.get("strategy", {}))
    for key in ("cheap_model",):
        value = strategy.get(key)
        if isinstance(value, dict):
            names.add(str(value.get("provider", "mock")))
    models = strategy.get("models")
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict):
                names.add(str(model.get("provider", "mock")))
    return names


def _mock_mode_from_config(config: dict[str, Any]) -> str:
    model = dict(config.get("model", {}))
    provider_config = dict(config.get("providers", {}).get("mock", {}))
    return str(model.get("mock_mode") or provider_config.get("mock_mode") or "fixture")


def _provider_base_url(config: dict[str, Any], provider_name: str) -> str | None:
    model = dict(config.get("model", {}))
    provider_config = dict(config.get("providers", {}).get(provider_name, {}))
    return model.get("base_url") or provider_config.get("base_url")


def base_url_class(base_url: str | None) -> str:
    """Classify endpoints for local-vs-remote accounting metadata."""

    if not base_url:
        return "unspecified"
    parsed = urllib.parse.urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return "local"
    return "remote"


def _gold_citations(example: BenchmarkExample) -> list[str]:
    citations = (
        example.gold.get("evidence")
        or example.gold.get("gold_evidence")
        or example.gold.get("citations")
        or []
    )
    return [str(item) for item in citations]


def _schema_placeholder_response(
    example: BenchmarkExample,
    *,
    weak: bool = False,
) -> dict[str, Any]:
    schema = example.gold.get("json_schema", {})
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", example.gold.get("schema_required", []))
    if not isinstance(properties, dict) or not isinstance(required, list):
        return {}
    return {
        str(field): _placeholder_for_schema(properties.get(field, {}), weak=weak)
        for field in required
    }


def _placeholder_for_schema(spec: Any, *, weak: bool) -> Any:
    if not isinstance(spec, dict):
        return None
    if spec.get("format") == "date":
        return "1970-01-01" if weak else "2000-01-01"
    type_spec = spec.get("type")
    allowed = type_spec if isinstance(type_spec, list) else [type_spec]
    if "string" in allowed:
        return "不明" if weak else ""
    if "integer" in allowed:
        return -1 if weak else 0
    if "number" in allowed:
        return -1.0 if weak else 0.0
    if "boolean" in allowed:
        return False
    if "array" in allowed:
        return []
    if "object" in allowed:
        return {}
    return None


def _comparable(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
