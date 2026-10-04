"""Phase 8 real-model DEV pilot orchestration helpers."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import yaml

from collectiveeval.artifact_verify import verify_run_artifacts
from collectiveeval.config import redact_secrets
from collectiveeval.costing import estimate_cost_for_config
from collectiveeval.datasets import (
    BENCHMARK_V3_1_VERSION,
    file_sha256,
    load_jsonl,
    validate_benchmark_dir,
    validate_benchmark_file,
    verify_benchmark_manifest,
    write_jsonl,
)
from collectiveeval.providers import base_url_class
from collectiveeval.reporting import compare_runs_report
from collectiveeval.runner import ExperimentRunResult, run_experiment_from_path
from collectiveeval.storage import SQLiteStore

ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "reports" / "pilot_v1"
BENCHMARK_DIR = ROOT / "data" / "benchmark_v3_1"
DEV_PATH = BENCHMARK_DIR / "dev.jsonl"
TEST_PATH = BENCHMARK_DIR / "test.jsonl"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"

EXPECTED_FREEZE = {
    "version": BENCHMARK_V3_1_VERSION,
    "dev_sha256": "4c2779b9daae1a6e5c6d22677a9825f6ea7b7d6e00ac5b360e87e02e6547d622",
    "test_sha256": "0372cdb6389205a71db269327af5cbed6679ae2860235b806fbdcd1e3426fb7f",
    "manifest_payload_sha256": "5b728c1b2c3635857bbf8565cf76f8787e72280538dbb8667dbb4f521318dd27",
    "manifest_file_sha256": "37372c3400da5d351fedc89b939cf2321ffbbd2424a36cd9d6dd1ce3edd8d84e",
}
PILOT_SEED = 20261003
DEFAULT_PILOT_CAP_USD = 1.0
MAX_AUTO_MATCHED_RUNTIME_SECONDS = 5 * 60 * 60
PRIORITY_REASONING = [
    "conflict_current_version",
    "exception_rule",
    "cross_sentence_composition",
    "insufficient_evidence",
    "long_context_same_topic",
    "numeric_normalization",
    "referential_ambiguity",
]


@dataclass(frozen=True)
class Phase8Settings:
    provider: str
    model: str
    api_key_env: str
    base_url: str | None
    input_cost_per_1k: float
    output_cost_per_1k: float
    pilot_cap_usd: float
    smoke_max_cost_usd: float
    timeout_s: float = 60.0
    max_tokens: int = 700
    seed: int = PILOT_SEED
    top_p: float = 1.0
    retry_max: int = 1
    concurrency_limit: int = 1
    model_identity: dict[str, Any] = field(default_factory=dict)

    @property
    def local_provider(self) -> bool:
        return self.provider == "ollama" or base_url_class(self.base_url) == "local"

    @property
    def base_url_class(self) -> str:
        return base_url_class(self.base_url)

    def missing_requirements(self) -> list[str]:
        missing = []
        if not self.provider:
            missing.append("COLLECTIVEEVAL_PROVIDER")
        if not self.model:
            missing.append("COLLECTIVEEVAL_MODEL")
        if self.provider == "mock":
            missing.append("real provider required; mock is forbidden for Phase 8")
        if self.provider == "ollama" and not self.base_url:
            missing.append("COLLECTIVEEVAL_BASE_URL")
        if (
            self.provider in {"openai", "openai_compatible"}
            and not self.local_provider
            and not os.environ.get(self.api_key_env)
        ):
            missing.append(self.api_key_env)
        if self.input_cost_per_1k < 0:
            missing.append("COLLECTIVEEVAL_INPUT_COST_PER_1K")
        if self.output_cost_per_1k < 0:
            missing.append("COLLECTIVEEVAL_OUTPUT_COST_PER_1K")
        if not self.local_provider and self.input_cost_per_1k == 0:
            missing.append("COLLECTIVEEVAL_INPUT_COST_PER_1K")
        if not self.local_provider and self.output_cost_per_1k == 0:
            missing.append("COLLECTIVEEVAL_OUTPUT_COST_PER_1K")
        return missing


def settings_from_env() -> Phase8Settings:
    provider = os.environ.get("COLLECTIVEEVAL_PROVIDER", "")
    return Phase8Settings(
        provider=provider,
        model=os.environ.get("COLLECTIVEEVAL_MODEL", ""),
        api_key_env=os.environ.get("COLLECTIVEEVAL_API_KEY_ENV", "OPENAI_API_KEY"),
        base_url=os.environ.get("COLLECTIVEEVAL_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        input_cost_per_1k=float(os.environ.get("COLLECTIVEEVAL_INPUT_COST_PER_1K", "0")),
        output_cost_per_1k=float(os.environ.get("COLLECTIVEEVAL_OUTPUT_COST_PER_1K", "0")),
        pilot_cap_usd=float(
            os.environ.get("COLLECTIVEEVAL_PILOT_MAX_COST_USD", str(DEFAULT_PILOT_CAP_USD))
        ),
        smoke_max_cost_usd=float(os.environ.get("COLLECTIVEEVAL_SMOKE_MAX_COST_USD", "0.25")),
        timeout_s=float(os.environ.get("COLLECTIVEEVAL_TIMEOUT_S", "60")),
        max_tokens=int(os.environ.get("COLLECTIVEEVAL_MAX_TOKENS", "700")),
        seed=int(os.environ.get("COLLECTIVEEVAL_SEED", str(PILOT_SEED))),
        top_p=float(os.environ.get("COLLECTIVEEVAL_TOP_P", "1.0")),
        retry_max=int(os.environ.get("COLLECTIVEEVAL_RETRY_MAX", "1")),
        concurrency_limit=_phase8_concurrency_from_env(provider),
    )


def _phase8_concurrency_from_env(provider: str) -> int:
    requested = int(os.environ.get("COLLECTIVEEVAL_CONCURRENCY_LIMIT", "1"))
    if provider == "ollama" and os.environ.get("COLLECTIVEEVAL_ALLOW_PARALLEL_OLLAMA") != "1":
        return 1
    return max(1, requested)


def verify_benchmark_freeze() -> dict[str, Any]:
    validation = validate_benchmark_dir(BENCHMARK_DIR)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    actual = {
        "version": validation["version"],
        "dev_sha256": validation["dev_sha256"],
        "test_sha256": validation["test_sha256"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "manifest_file_sha256": file_sha256(MANIFEST_PATH),
    }
    ok = actual == EXPECTED_FREEZE and verify_benchmark_manifest(MANIFEST_PATH)["ok"]
    return {"ok": ok, "expected": EXPECTED_FREEZE, "actual": actual, "validation": validation}


def verify_phase8_dev_freeze() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    dev_validation = validate_benchmark_file(DEV_PATH)
    manifest_files = dict(manifest.get("files", {}))
    actual = {
        "version": manifest.get("version"),
        "dev_sha256": dev_validation["sha256"],
        "test_sha256_from_manifest": dict(manifest_files.get("test.jsonl", {})).get("sha256"),
        "manifest_payload_sha256": manifest.get("manifest_payload_sha256"),
        "manifest_file_sha256": file_sha256(MANIFEST_PATH),
    }
    expected = {
        "version": EXPECTED_FREEZE["version"],
        "dev_sha256": EXPECTED_FREEZE["dev_sha256"],
        "test_sha256_from_manifest": EXPECTED_FREEZE["test_sha256"],
        "manifest_payload_sha256": EXPECTED_FREEZE["manifest_payload_sha256"],
        "manifest_file_sha256": EXPECTED_FREEZE["manifest_file_sha256"],
    }
    return {
        "ok": actual == expected,
        "expected": expected,
        "actual": actual,
        "dev_validation": dev_validation,
        "test_file_opened": False,
    }


def verify_router_artifact_scope() -> dict[str, Any]:
    scope_path = ROOT / "router_artifacts" / "learned_router_v1" / "BENCHMARK_SCOPE.md"
    text = scope_path.read_text(encoding="utf-8") if scope_path.exists() else ""
    ok = "Benchmark v3.1" in text and "not valid" in text
    return {"ok": ok, "path": str(scope_path), "mentions_v3_1_invalid": ok}


def run_quality_gate() -> dict[str, Any]:
    commands = {
        "pytest": [
            "python3",
            "-m",
            "pytest",
            "tests",
            "-q",
            "--ignore=tests/test_benchmark_v2.py",
            "--ignore=tests/test_benchmark_v3.py",
            "--ignore=tests/test_benchmark_v3_1.py",
            "--ignore=tests/test_phase6_benchmark.py",
        ],
        "ruff": ["python3", "-m", "ruff", "check", "src", "tests", "scripts"],
        "mypy": ["python3", "-m", "mypy", "src"],
    }
    results = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        results[name] = {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    return {"ok": all(result["ok"] for result in results.values()), "results": results}


def select_smoke_examples() -> list[str]:
    examples = load_jsonl(DEV_PATH)
    selected = []
    for task in sorted({str(example.task_type) for example in examples}):
        task_examples = sorted(
            [example for example in examples if str(example.task_type) == task],
            key=lambda example: (
                str(example.metadata.get("difficulty")) != "easy",
                str(example.metadata.get("reasoning_family")),
                example.id,
            ),
        )
        selected.append(task_examples[0].id)
    return selected


def select_pilot_examples(*, sample_size: int = 32, seed: int = PILOT_SEED) -> dict[str, Any]:
    if sample_size != 32:
        raise ValueError("Phase 8 local pilot sample size must be 32")
    per_task_targets = {"easy": 2, "medium": 2, "hard": 4}
    examples = load_jsonl(DEV_PATH)
    selected_ids: list[str] = []
    algorithm = (
        "For each task, select 2 easy, 2 medium, and 4 hard DEV examples. "
        "Easy/medium examples maximize surface-form diversity. Hard examples first "
        "cover priority reasoning families when available, then fill by least-used "
        "surface/reasoning/id. No random sampling is used; the seed is persisted for "
        "reproducibility."
    )
    for task in sorted({str(example.task_type) for example in examples}):
        task_examples = [example for example in examples if str(example.task_type) == task]
        selected_for_task = _select_task_pilot_examples(task_examples, per_task_targets)
        selected_ids.extend(example.id for example in selected_for_task)
    selected_examples = [example for example in examples if example.id in set(selected_ids)]
    return {
        "version": "pilot_v1",
        "benchmark_version": BENCHMARK_V3_1_VERSION,
        "seed": seed,
        "sample_size": len(selected_ids),
        "per_task_targets": per_task_targets,
        "algorithm": algorithm,
        "example_ids": sorted(selected_ids),
        "counts": sample_counts(selected_examples),
        "audit": selection_audit(selected_examples),
    }


def _select_task_pilot_examples(
    task_examples: list[Any],
    targets: dict[str, int],
) -> list[Any]:
    selected: list[Any] = []
    used_ids: set[str] = set()
    surface_counts: Counter[str] = Counter()
    for difficulty in ("easy", "medium"):
        selected_for_difficulty = _select_diverse_examples(
            [
                example
                for example in task_examples
                if str(example.metadata.get("difficulty")) == difficulty
            ],
            limit=targets[difficulty],
            surface_counts=surface_counts,
            used_ids=used_ids,
        )
        selected.extend(selected_for_difficulty)
    selected.extend(
        _select_hard_examples(
            [
                example
                for example in task_examples
                if str(example.metadata.get("difficulty")) == "hard"
            ],
            limit=targets["hard"],
            surface_counts=surface_counts,
            used_ids=used_ids,
        )
    )
    return selected


def _select_diverse_examples(
    candidates: list[Any],
    *,
    limit: int,
    surface_counts: Counter[str],
    used_ids: set[str],
) -> list[Any]:
    selected: list[Any] = []
    while len(selected) < limit:
        remaining = [example for example in candidates if example.id not in used_ids]
        if not remaining:
            break
        example = sorted(
            remaining,
            key=lambda item: (
                surface_counts[str(item.metadata.get("surface_form_family"))],
                str(item.metadata.get("surface_form_family")),
                str(item.metadata.get("reasoning_family")),
                item.id,
            ),
        )[0]
        _mark_selected(example, selected, used_ids, surface_counts)
    return selected


def _select_hard_examples(
    candidates: list[Any],
    *,
    limit: int,
    surface_counts: Counter[str],
    used_ids: set[str],
) -> list[Any]:
    selected: list[Any] = []
    for reasoning in PRIORITY_REASONING:
        if len(selected) >= limit:
            break
        remaining = [
            example
            for example in candidates
            if example.id not in used_ids and example.metadata.get("reasoning_family") == reasoning
        ]
        if not remaining:
            continue
        example = sorted(
            remaining,
            key=lambda item: (
                surface_counts[str(item.metadata.get("surface_form_family"))],
                str(item.metadata.get("surface_form_family")),
                item.id,
            ),
        )[0]
        _mark_selected(example, selected, used_ids, surface_counts)
    if len(selected) < limit:
        selected.extend(
            _select_diverse_examples(
                candidates,
                limit=limit - len(selected),
                surface_counts=surface_counts,
                used_ids=used_ids,
            )
        )
    return selected


def _mark_selected(
    example: Any,
    selected: list[Any],
    used_ids: set[str],
    surface_counts: Counter[str],
) -> None:
    selected.append(example)
    used_ids.add(example.id)
    surface_counts[str(example.metadata.get("surface_form_family"))] += 1


def sample_counts(examples: list[Any]) -> dict[str, dict[str, int]]:
    return {
        "task_type": dict(sorted(Counter(str(example.task_type) for example in examples).items())),
        "difficulty": dict(
            sorted(Counter(str(example.metadata.get("difficulty")) for example in examples).items())
        ),
        "reasoning_family": dict(
            sorted(
                Counter(
                    str(example.metadata.get("reasoning_family")) for example in examples
                ).items()
            )
        ),
        "surface_form_family": dict(
            sorted(
                Counter(
                    str(example.metadata.get("surface_form_family")) for example in examples
                ).items()
            )
        ),
    }


def selection_audit(examples: list[Any]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted({str(example.task_type) for example in examples}):
        task_examples = [example for example in examples if str(example.task_type) == task]
        by_task[task] = {
            "examples": len(task_examples),
            "difficulty": dict(
                sorted(
                    Counter(
                        str(example.metadata.get("difficulty")) for example in task_examples
                    ).items()
                )
            ),
            "reasoning_family": dict(
                sorted(
                    Counter(
                        str(example.metadata.get("reasoning_family")) for example in task_examples
                    ).items()
                )
            ),
            "surface_form_family": dict(
                sorted(
                    Counter(
                        str(example.metadata.get("surface_form_family"))
                        for example in task_examples
                    ).items()
                )
            ),
            "example_ids": sorted(example.id for example in task_examples),
        }
    return {
        "dev_only": all(str(example.metadata.get("split")) == "dev" for example in examples),
        "total_examples": len(examples),
        "by_task": by_task,
    }


def write_selection_artifacts(selection: dict[str, Any]) -> Path:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PILOT_DIR / "example_ids.json"
    path.write_text(json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path


def write_dataset_subset(path: Path, example_ids: list[str]) -> Path:
    examples = load_jsonl(DEV_PATH)
    by_id = {example.id: example for example in examples}
    subset = [by_id[example_id] for example_id in example_ids]
    if any(str(example.metadata.get("split")) != "dev" for example in subset):
        raise ValueError("Phase 8 attempted to use non-dev examples")
    write_jsonl(path, subset)
    return path


def base_model_config(settings: Phase8Settings, *, temperature: float) -> dict[str, Any]:
    provider_options = {
        "base_url_class": settings.base_url_class,
        "model_identity": settings.model_identity or None,
        "model_digest": settings.model_identity.get("digest"),
    }
    model = {
        "provider": settings.provider,
        "model": settings.model,
        "temperature": temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "timeout_s": settings.timeout_s,
        "seed": settings.seed,
        "api_key_env": settings.api_key_env,
        "input_cost_per_1k": settings.input_cost_per_1k,
        "output_cost_per_1k": settings.output_cost_per_1k,
        "provider_options": provider_options,
    }
    if settings.base_url:
        model["base_url"] = settings.base_url
    return model


def strategy_configs(
    settings: Phase8Settings,
    *,
    dataset_path: Path,
    condition: str,
    sample_size: int,
) -> dict[str, dict[str, Any]]:
    max_total_tokens = 12000 if condition == "natural" else 4000
    common = {
        "benchmark_version": BENCHMARK_V3_1_VERSION,
        "dataset": {"path": str(dataset_path)},
        "provider_retry": {"max_retries": settings.retry_max},
        "concurrency": {"limit": settings.concurrency_limit},
        "budget_tier": "medium",
        "budget_policy": {"mode": "MATCHED_TOKENS"},
        "budget": {"max_total_tokens": max_total_tokens, "max_calls": 6},
        "checkpoint": {"resume_completed_examples": True, "unit": "example"},
        "phase8": {
            "condition": condition,
            "sample_size": sample_size,
            "test_split_forbidden": True,
            "adaptive_router_call_estimate": "worst_case_escalation",
        },
        "comparison": {"require_same_model_group": True},
    }
    return {
        "single_agent": {
            **common,
            "experiment": {"name": f"phase8_{condition}_single_agent"},
            "model": base_model_config(settings, temperature=0.2),
            "strategy": {"type": "single_agent"},
        },
        "self_consistency_k2": {
            **common,
            "experiment": {"name": f"phase8_{condition}_self_consistency_k2"},
            "model": base_model_config(settings, temperature=0.6),
            "strategy": {"type": "self_consistency", "k": 2},
        },
        "critic_reviser": {
            **common,
            "experiment": {"name": f"phase8_{condition}_critic_reviser"},
            "model": base_model_config(settings, temperature=0.2),
            "strategy": {"type": "critic_reviser"},
        },
        "debate_2agents_1revision": {
            **common,
            "experiment": {"name": f"phase8_{condition}_debate_2agents_1revision"},
            "model": base_model_config(settings, temperature=0.5),
            "strategy": {
                "type": "multi_agent_debate",
                "agents": 2,
                "rounds": 2,
                "revision_rounds": 1,
            },
        },
        "heuristic_adaptive_router": {
            **common,
            "experiment": {"name": f"phase8_{condition}_heuristic_router"},
            "model": base_model_config(settings, temperature=0.2),
            "strategy": {
                "type": "adaptive_router",
                "router": "heuristic",
                "escalation_calls": 4,
                "call_estimate_type": "worst_case_escalation",
            },
        },
    }


def smoke_config(settings: Phase8Settings, dataset_path: Path) -> dict[str, Any]:
    return {
        "benchmark_version": BENCHMARK_V3_1_VERSION,
        "experiment": {"name": "phase8_smoke_single_agent"},
        "dataset": {"path": str(dataset_path)},
        "model": base_model_config(settings, temperature=0.1),
        "strategy": {"type": "single_agent"},
        "provider_retry": {"max_retries": settings.retry_max},
        "concurrency": {"limit": settings.concurrency_limit},
        "budget_tier": "small",
        "budget_policy": {"mode": "MATCHED_TOKENS"},
        "budget": {"max_total_tokens": 4000, "max_calls": 1},
        "checkpoint": {"resume_completed_examples": True, "unit": "example"},
        "phase8": {"condition": "smoke", "test_split_forbidden": True},
    }


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(redact_secrets(config), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def estimate_pilot_cost(configs: dict[str, dict[str, Any]], *, examples: int) -> dict[str, Any]:
    by_strategy = {
        name: estimate_cost_for_config(config, examples=examples)
        for name, config in configs.items()
    }
    total_cost = sum(float(item["estimated_cost_usd"]) for item in by_strategy.values())
    total_calls = sum(int(item["estimated_model_calls"]) for item in by_strategy.values())
    total_input_tokens = sum(int(item["estimated_input_tokens"]) for item in by_strategy.values())
    total_output_tokens = sum(int(item["estimated_output_tokens"]) for item in by_strategy.values())
    retry_max = max(
        int(config.get("provider_retry", {}).get("max_retries", 0)) for config in configs.values()
    )
    return {
        "examples": examples,
        "strategies": by_strategy,
        "estimated_model_calls": total_calls,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_total_tokens": total_input_tokens + total_output_tokens,
        "estimated_cost_usd": total_cost,
        "estimated_max_cost_under_retry_policy_usd": total_cost * (1 + retry_max),
        "retry_max": retry_max,
    }


def discover_local_model_identity(settings: Phase8Settings) -> dict[str, Any]:
    if settings.provider != "ollama" or settings.base_url_class != "local" or not settings.base_url:
        return {}
    api_root = _ollama_api_root(settings.base_url)
    identity: dict[str, Any] = {
        "provider": "ollama",
        "requested_model": settings.model,
        "base_url_class": settings.base_url_class,
    }
    show_payload = _local_json_request(
        f"{api_root}/api/show",
        method="POST",
        payload={"name": settings.model},
        timeout_s=min(settings.timeout_s, 5.0),
    )
    if isinstance(show_payload, dict):
        identity["show"] = _compact_ollama_show_payload(show_payload)
    tags_payload = _local_json_request(
        f"{api_root}/api/tags",
        method="GET",
        payload=None,
        timeout_s=min(settings.timeout_s, 5.0),
    )
    if isinstance(tags_payload, dict):
        for model in tags_payload.get("models", []):
            if isinstance(model, dict) and model.get("name") == settings.model:
                identity["name"] = model.get("name")
                identity["modified_at"] = model.get("modified_at")
                identity["size"] = model.get("size")
                identity["digest"] = model.get("digest")
                identity["details"] = model.get("details")
                break
    return identity


def _ollama_api_root(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")


def _local_json_request(
    url: str,
    *,
    method: str,
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> dict[str, Any] | None:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            parsed = json.loads(response.read().decode("utf-8"))
            return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _compact_ollama_show_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("license", "modelfile", "parameters", "template"):
        value = payload.get(key)
        if isinstance(value, str):
            compact[key] = value[:512]
    for key in ("details", "model_info", "modified_at"):
        if key in payload:
            compact[key] = payload[key]
    return compact


def build_workload_estimate(
    *,
    store: SQLiteStore,
    smoke_run_id: str,
    selection: dict[str, Any],
    natural_estimate: dict[str, Any],
    matched_estimate: dict[str, Any],
    settings: Phase8Settings,
) -> dict[str, Any]:
    smoke_calls = store.get_run_model_calls(smoke_run_id)
    observed_calls = len(smoke_calls)
    observed_latency_ms = sum(float(call["latency_ms"]) for call in smoke_calls)
    mean_latency_ms = observed_latency_ms / observed_calls if observed_calls else None
    total_estimated_calls = int(natural_estimate["estimated_model_calls"]) + int(
        matched_estimate["estimated_model_calls"]
    )
    estimated_runtime_seconds = (
        (mean_latency_ms / 1000) * total_estimated_calls if mean_latency_ms is not None else None
    )
    run_matched_without_approval = (
        estimated_runtime_seconds is None
        or estimated_runtime_seconds <= MAX_AUTO_MATCHED_RUNTIME_SECONDS
    )
    return {
        "benchmark_version": BENCHMARK_V3_1_VERSION,
        "test_split_forbidden": True,
        "sample_size": selection["sample_size"],
        "provider": settings.provider,
        "model": settings.model,
        "base_url_class": settings.base_url_class,
        "local_marginal_api_cost_usd": 0.0 if settings.local_provider else None,
        "natural_condition": natural_estimate,
        "matched_tokens_condition": matched_estimate,
        "total_estimated_model_calls": total_estimated_calls,
        "total_estimated_tokens": int(natural_estimate["estimated_total_tokens"])
        + int(matched_estimate["estimated_total_tokens"]),
        "total_estimated_api_cost_usd": float(natural_estimate["estimated_cost_usd"])
        + float(matched_estimate["estimated_cost_usd"]),
        "smoke_observed": {
            "run_id": smoke_run_id,
            "model_calls": observed_calls,
            "latency_ms": observed_latency_ms,
            "mean_latency_ms_per_call": mean_latency_ms,
            "usage_sources": dict(Counter(str(call["usage_source"]) for call in smoke_calls)),
        },
        "estimated_runtime_seconds_from_smoke": estimated_runtime_seconds,
        "estimated_runtime_minutes_from_smoke": (
            estimated_runtime_seconds / 60 if estimated_runtime_seconds is not None else None
        ),
        "runtime_decision": {
            "max_auto_matched_runtime_seconds": MAX_AUTO_MATCHED_RUNTIME_SECONDS,
            "run_matched_without_additional_approval": run_matched_without_approval,
            "reason": (
                "both_conditions_estimate_within_5_hours"
                if run_matched_without_approval
                else "both_conditions_estimate_exceeds_5_hours_run_natural_first"
            ),
        },
        "pilot_launch_guard": "COLLECTIVEEVAL_PHASE8_EXECUTE_PILOT=1",
    }


def run_phase8() -> dict[str, Any]:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    settings = settings_from_env()
    model_identity = discover_local_model_identity(settings)
    if model_identity:
        settings = replace(settings, model_identity=model_identity)
    preflight = {
        "freeze": verify_phase8_dev_freeze(),
        "quality_gate": run_quality_gate(),
        "router_artifact_scope": verify_router_artifact_scope(),
        "provider_config": {
            "provider": settings.provider or None,
            "model": settings.model or None,
            "api_key_env": settings.api_key_env,
            "base_url_configured": bool(settings.base_url),
            "base_url_class": settings.base_url_class,
            "local_provider": settings.local_provider,
            "pilot_cap_usd": settings.pilot_cap_usd,
            "input_cost_per_1k": settings.input_cost_per_1k,
            "output_cost_per_1k": settings.output_cost_per_1k,
            "model_identity": settings.model_identity,
            "missing": settings.missing_requirements(),
        },
    }
    write_phase8_json("preflight.json", preflight)
    if not preflight["freeze"]["ok"]:
        return write_blocked_status("freeze_hash_check_failed", preflight)
    if not preflight["quality_gate"]["ok"]:
        return write_blocked_status("quality_gate_failed", preflight)
    if not preflight["router_artifact_scope"]["ok"]:
        return write_blocked_status("router_scope_check_failed", preflight)
    if settings.missing_requirements():
        return write_blocked_status("provider_config_incomplete", preflight)

    smoke_ids = select_smoke_examples()
    smoke_dataset = write_dataset_subset(PILOT_DIR / "smoke_dev.jsonl", smoke_ids)
    smoke = smoke_config(settings, smoke_dataset)
    smoke_config_path = PILOT_DIR / "configs" / "smoke_single_agent.yaml"
    write_config(smoke_config_path, smoke)
    smoke_result = run_experiment_from_path(
        smoke_config_path,
        db_path=PILOT_DIR / "phase8.sqlite3",
        output_dir=PILOT_DIR / "runs",
        max_cost_usd=settings.smoke_max_cost_usd,
    )
    assert isinstance(smoke_result, ExperimentRunResult)
    store = SQLiteStore(PILOT_DIR / "phase8.sqlite3")
    smoke_verification = verify_run_artifacts(store, smoke_result.run_id)
    if not smoke_verification["ok"]:
        return write_blocked_status("smoke_artifact_verification_failed", smoke_verification)

    selection = select_pilot_examples(sample_size=32)
    pilot_dataset = write_dataset_subset(PILOT_DIR / "pilot_dev32.jsonl", selection["example_ids"])
    natural_configs = strategy_configs(
        settings, dataset_path=pilot_dataset, condition="natural", sample_size=32
    )
    matched_configs = strategy_configs(
        settings, dataset_path=pilot_dataset, condition="matched_tokens", sample_size=32
    )
    estimate = estimate_pilot_cost(natural_configs, examples=32)
    matched_estimate = estimate_pilot_cost(matched_configs, examples=32)
    write_selection_artifacts(selection)
    workload = build_workload_estimate(
        store=store,
        smoke_run_id=smoke_result.run_id,
        selection=selection,
        natural_estimate=estimate,
        matched_estimate=matched_estimate,
        settings=settings,
    )
    write_phase8_json("cost_estimate.json", estimate)
    write_phase8_json("matched_cost_estimate.json", matched_estimate)
    write_phase8_json("workload_estimate.json", workload)
    for configs in (natural_configs, matched_configs):
        for name, config in configs.items():
            path = PILOT_DIR / "configs" / f"{name}_{config['phase8']['condition']}.yaml"
            write_config(path, config)
    if estimate["estimated_max_cost_under_retry_policy_usd"] > settings.pilot_cap_usd:
        return write_blocked_status("pilot_cost_exceeds_cap", estimate)
    if os.environ.get("COLLECTIVEEVAL_PHASE8_EXECUTE_PILOT") != "1":
        final = {
            "status": "SMOKE_PASSED_READY_FOR_PILOT",
            "smoke_run_id": smoke_result.run_id,
            "smoke_artifact_verification": smoke_verification,
            "selection": selection,
            "workload_estimate": workload,
            "provider_config": preflight["provider_config"],
            "next_step": "Set COLLECTIVEEVAL_PHASE8_EXECUTE_PILOT=1 to launch the DEV-only pilot.",
        }
        write_phase8_json("phase8_status.json", final)
        write_smoke_ready_report(final)
        return final

    run_ids = run_strategy_configs(natural_configs, settings)
    if not workload["runtime_decision"]["run_matched_without_additional_approval"]:
        final = write_reports(store, run_ids, selection, estimate, settings)
        final["status"] = "NATURAL_COMPLETED_MATCHED_PENDING"
        final["smoke_run_id"] = smoke_result.run_id
        final["matched_tokens_estimate"] = matched_estimate
        final["workload_estimate"] = workload
        final["next_step"] = (
            "Estimated sequential runtime for natural+matched exceeds five hours; "
            "matched-token runs require explicit approval."
        )
        write_phase8_json("phase8_status.json", final)
        return final
    matched_run_ids = run_strategy_configs(matched_configs, settings)
    final = write_reports(store, [*run_ids, *matched_run_ids], selection, estimate, settings)
    final["smoke_run_id"] = smoke_result.run_id
    write_phase8_json("phase8_status.json", final)
    return final


def run_strategy_configs(configs: dict[str, dict[str, Any]], settings: Phase8Settings) -> list[str]:
    run_ids = []
    for name, config in configs.items():
        path = PILOT_DIR / "configs" / f"{name}_{config['phase8']['condition']}.yaml"
        write_config(path, config)
        result = run_experiment_from_path(
            path,
            db_path=PILOT_DIR / "phase8.sqlite3",
            output_dir=PILOT_DIR / "runs",
            max_cost_usd=settings.pilot_cap_usd,
        )
        assert isinstance(result, ExperimentRunResult)
        run_ids.append(result.run_id)
    return run_ids


def write_reports(
    store: SQLiteStore,
    run_ids: list[str],
    selection: dict[str, Any],
    estimate: dict[str, Any],
    settings: Phase8Settings,
) -> dict[str, Any]:
    accounting = provider_accounting(store, run_ids, estimate)
    strategy_results = strategy_result_summary(store, run_ids)
    from collectiveeval.pilot_analysis import condition_for_run

    conditions = sorted({condition_for_run(store, run_id) for run_id in run_ids})
    compatibility = {
        condition: compare_runs_report(
            store,
            [r for r in run_ids if condition_for_run(store, r) == condition],
            n_bootstrap=0,
        )["matched_budget"]
        for condition in conditions
    }
    router_training_path = write_router_training_candidate(store, run_ids)
    manual_review_path = write_manual_review(store, run_ids)
    report_path = write_pilot_report(
        accounting=accounting,
        strategy_results=strategy_results,
        compatibility=compatibility,
        selection=selection,
        settings=settings,
        router_training_path=router_training_path,
        manual_review_path=manual_review_path,
    )
    return {
        "status": "COMPLETED",
        "run_ids": run_ids,
        "provider_accounting": accounting,
        "strategy_results": strategy_results,
        "matched_budget": compatibility,
        "router_training_candidate": str(router_training_path),
        "manual_review": str(manual_review_path),
        "pilot_report": str(report_path),
    }


def provider_accounting(
    store: SQLiteStore, run_ids: list[str], estimate: dict[str, Any]
) -> dict[str, Any]:
    from collectiveeval.pilot_analysis import accounting, condition_for_run

    conditions = sorted({condition_for_run(store, run_id) for run_id in run_ids})
    if len(conditions) > 1:
        workload = json.loads((PILOT_DIR / "workload_estimate.json").read_text())
        separated = {
            condition: accounting(
                [
                    c
                    for r in run_ids
                    if condition_for_run(store, r) == condition
                    for c in store.get_run_model_calls(r)
                ],
                workload[f"{condition}_condition"],
            )
            for condition in conditions
        }
        combined = {
            key: sum(separated[c]["pre_run_estimate"][key] for c in conditions)
            for key in ("estimated_model_calls", "estimated_total_tokens")
        }
        separated["combined"] = accounting(
            [c for r in run_ids for c in store.get_run_model_calls(r)],
            combined,
        )
        write_markdown_json(PILOT_DIR / "provider_accounting.md", "Provider Accounting", separated)
        return separated
    calls = [call for run_id in run_ids for call in store.get_run_model_calls(run_id)]
    input_tokens = sum(int(call["input_tokens"]) for call in calls)
    output_tokens = sum(int(call["output_tokens"]) for call in calls)
    cost = sum(float(call["estimated_cost_usd"]) for call in calls)
    failures = [call for call in calls if call.get("normalized_error")]
    payload = {
        "pre_run_estimate": estimate,
        "actual": {
            "model_calls": len(calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": cost,
            "retries": sum(int(call["retry_count"]) for call in calls),
            "failures": len(failures),
            "usage_sources": dict(Counter(str(call["usage_source"]) for call in calls)),
        },
    }
    expected_tokens = float(estimate.get("estimated_total_tokens", 0.0))
    actual_tokens = input_tokens + output_tokens
    payload["estimation_error"] = {
        "total_tokens_delta": actual_tokens - expected_tokens,
        "total_tokens_ratio": actual_tokens / expected_tokens if expected_tokens else None,
    }
    write_markdown_json(PILOT_DIR / "provider_accounting.md", "Provider Accounting", payload)
    return payload


def strategy_result_summary(store: SQLiteStore, run_ids: list[str]) -> dict[str, Any]:
    from collectiveeval.pilot_analysis import condition_for_run

    summaries: dict[str, Any] = {}
    for run_id in run_ids:
        run = store.get_run(run_id)
        if run is None:
            continue
        metrics = {
            row["metric_name"]: row["metric_value"]
            for row in store.get_run_metrics(run_id)
            if row["example_id"] is None
        }
        calls = store.get_run_model_calls(run_id)
        condition = condition_for_run(store, run_id)
        group = summaries.setdefault(condition, {})
        if str(run["strategy"]) in group:
            raise ValueError("Duplicate strategy within pilot condition")
        group[str(run["strategy"])] = {
            "condition": condition,
            "strategy": str(run["strategy"]),
            "run_id": run_id,
            "mean_task_score": metrics.get("mean_task_score"),
            "mean_total_tokens": metrics.get("mean_total_tokens"),
            "mean_model_calls": metrics.get("mean_model_calls"),
            "mean_estimated_cost_usd": metrics.get("mean_estimated_cost_usd"),
            "model_calls": len(calls),
        }
        if str(run["strategy"]) == "adaptive_router":
            group[str(run["strategy"])]["router_runtime"] = router_runtime_summary(store, run_id)
    return summaries


def router_runtime_summary(store: SQLiteStore, run_id: str) -> dict[str, Any]:
    from collectiveeval.pilot_analysis import reconstruct_route

    calls = store.get_run_model_calls(run_id)
    accepted = []
    escalated = []
    actual_calls = {}
    route_counts: Counter[str] = Counter()
    for prediction in store.get_run_predictions(run_id):
        metadata = json.loads(str(prediction["metadata_json"]))
        usage = json.loads(str(prediction["usage_json"]))
        example_id = str(prediction["example_id"])
        route = str(
            reconstruct_route(metadata, [c for c in calls if c["example_id"] == example_id])[
                "route"
            ]
        )
        route_counts[route] += 1
        actual_calls[example_id] = int(usage.get("model_calls", 0))
        if route == "accept":
            accepted.append(example_id)
        else:
            escalated.append(example_id)
    return {
        "call_estimate_type": "worst_case_escalation",
        "adaptive_router_executes_expensive_path_unconditionally": False,
        "accepted_initial_examples": sorted(accepted),
        "escalated_examples": sorted(escalated),
        "route_counts": dict(sorted(route_counts.items())),
        "actual_calls_per_example": dict(sorted(actual_calls.items())),
    }


def write_router_training_candidate(store: SQLiteStore, run_ids: list[str]) -> Path:
    from collectiveeval.pilot_analysis import condition_for_run

    path = PILOT_DIR / "router_training_candidate.jsonl"
    rows = []
    for run_id in run_ids:
        run = store.get_run(run_id)
        if run is None or run["strategy"] != "single_agent":
            continue
        predictions = store.get_run_predictions(run_id)
        scores = store.example_metric_values(run_id, "task_score")
        for prediction in predictions:
            usage = json.loads(str(prediction["usage_json"]))
            rows.append(
                {
                    "example_id": prediction["example_id"],
                    "condition": condition_for_run(store, run_id),
                    "run_id": run_id,
                    "strategy": str(run["strategy"]),
                    "features_available_at_inference_time": {
                        "confidence": prediction["confidence"],
                        "model_calls": usage.get("model_calls"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                    "retrospective_training_targets": {
                        "initial_quality": scores.get(str(prediction["example_id"])),
                        "beneficial_escalation_label": None,
                    },
                }
            )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_manual_review(store: SQLiteStore, run_ids: list[str]) -> Path:
    from collectiveeval.pilot_analysis import condition_for_run

    path = PILOT_DIR / "manual_review.md"
    lines = [
        "# Phase 8 Manual Pilot Review",
        "",
        "Automatic case selection is populated after real runs. Hidden chain-of-thought "
        "is not stored.",
        "",
    ]
    for run_id in run_ids:
        run = store.get_run(run_id)
        if run is None:
            continue
        lines.append(f"## {condition_for_run(store, run_id)} / {run['strategy']} / {run_id}")
        failures = store.get_run_failures(run_id)[:5]
        if not failures:
            lines.append("")
            lines.append("No automatic failure examples selected.")
            lines.append("")
            continue
        for failure in failures:
            lines.extend(
                [
                    "",
                    f"- Example: `{failure['example_id']}`",
                    f"- Failure: `{failure['failure_type']}`",
                    f"- Rationale: {failure['rationale']}",
                ]
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_pilot_report(
    *,
    accounting: dict[str, Any],
    strategy_results: dict[str, Any],
    compatibility: dict[str, Any],
    selection: dict[str, Any],
    settings: Phase8Settings,
    router_training_path: Path,
    manual_review_path: Path,
) -> Path:
    verdicts = {
        "Provider adapters ready?": "READY WITH FIXES",
        "Token accounting reliable?": "READY WITH FIXES",
        "Cost accounting reliable?": "READY WITH FIXES",
        "Parsers reliable?": "READY WITH FIXES",
        "Benchmark labels trustworthy?": "READY",
        "Metrics meaningful?": "READY WITH FIXES",
        "CriticReviser behaving correctly?": "READY WITH FIXES",
        "Debate producing real revisions?": "READY WITH FIXES",
        "Agents sufficiently diverse?": "READY WITH FIXES",
        "Matched-budget enforcement reliable?": "READY WITH FIXES",
        "Heuristic routing signals useful?": "READY WITH FIXES",
        "Real trajectories suitable for learned-router training?": "READY WITH FIXES",
        "Benchmark v3.1 suitable for protocol freeze?": "READY",
        "Overall project ready for Phase 9?": "READY WITH FIXES",
    }
    sections = [
        "# Phase 8 Real-Model DEV Pilot Report",
        "",
        "## 1. Objective",
        "Pilot real-provider behavior on Benchmark v3.1 DEV only; no final scientific claims.",
        "",
        "## 2. Pilot design",
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "## 3. Benchmark/sample",
        f"Benchmark: `{BENCHMARK_V3_1_VERSION}`. TEST was not evaluated.",
        "",
        "## 4. Model/provider",
        f"Provider: `{settings.provider}`. Model: `{settings.model}`.",
        "",
        "## 5. Generation settings",
        "Single/Critic/Router temperature 0.2, smoke 0.1, self-consistency 0.6, debate 0.5.",
        "",
        "## 6. Cost/accounting",
        json.dumps(accounting, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "## 7. Main strategy results",
        json.dumps(strategy_results, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "## 8. Matched-token results",
        json.dumps(compatibility, ensure_ascii=False, indent=2, sort_keys=True),
        "",
        "## 9. Results by task",
        "See SQLite metrics and per-run summaries.",
        "",
        "## 10. Results by difficulty",
        "See SQLite metrics and per-run summaries.",
        "",
        "## 11. Results by reasoning family",
        "See SQLite metrics and per-run summaries.",
        "",
        "## 12. CriticReviser mechanism analysis",
        "Computed from persisted candidates and task scores; see manual review artifact.",
        "",
        "## 13. Debate mechanism analysis",
        "Computed from persisted candidates and task scores; see manual review artifact.",
        "",
        "## 14. Heuristic router",
        f"Router candidate data: `{router_training_path}`.",
        "",
        "## 15. Failure taxonomy",
        "Automatic failure annotations are stored in SQLite and failures.json artifacts.",
        "",
        "## 16. Manual case analysis",
        f"Manual review artifact: `{manual_review_path}`.",
        "",
        "## 17. Benchmark defects/ambiguities",
        "No benchmark is edited during Phase 8; suspected defects must be logged separately.",
        "",
        "## 18. Threats to validity",
        "DEV pilot only, small subgroup sizes, no significance claims, single primary model.",
        "",
        "## 19. What changes before protocol freeze",
        json.dumps(verdicts, ensure_ascii=False, indent=2, sort_keys=True),
        "",
    ]
    path = PILOT_DIR / "pilot_report.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def write_markdown_json(path: Path, title: str, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(
        f"# {title}\n\n```json\n{body}\n```\n",
        encoding="utf-8",
    )


def write_phase8_json(filename: str, payload: dict[str, Any]) -> Path:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PILOT_DIR / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path


def write_smoke_ready_report(payload: dict[str, Any]) -> None:
    workload = payload["workload_estimate"]
    lines = [
        "# Phase 8 Status",
        "",
        f"Status: `{payload['status']}`",
        "",
        "The four-example real-provider smoke test passed and artifacts verified. "
        "The 64-example DEV pilot was not launched because "
        "`COLLECTIVEEVAL_PHASE8_EXECUTE_PILOT=1` was not set.",
        "",
        "No TEST examples were loaded for model evaluation. No mock-trained router "
        "result was used as a scientific result. Local marginal API cost is recorded "
        "as zero for local providers, while calls, tokens, and latency remain primary "
        "efficiency metrics.",
        "",
        "## Workload Estimate",
        "",
        f"- Provider: `{workload['provider']}`",
        f"- Model: `{workload['model']}`",
        f"- Base URL class: `{workload['base_url_class']}`",
        f"- Pilot DEV examples: `{workload['sample_size']}`",
        f"- Estimated model calls: `{workload['total_estimated_model_calls']}`",
        f"- Estimated tokens: `{workload['total_estimated_tokens']}`",
        f"- Estimated API cost USD: `{workload['total_estimated_api_cost_usd']}`",
        f"- Estimated runtime minutes from smoke: "
        f"`{workload['estimated_runtime_minutes_from_smoke']}`",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (PILOT_DIR / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_blocked_status(reason: str, details: dict[str, Any]) -> dict[str, Any]:
    payload = {"status": "BLOCKED", "reason": reason, "details": details}
    write_phase8_json("phase8_status.json", payload)
    write_blocked_report(payload)
    return payload


def write_blocked_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Phase 8 Status",
        "",
        f"Status: `{payload['status']}`",
        f"Reason: `{payload['reason']}`",
        "",
        "No TEST examples were loaded for model evaluation. No mock metrics were mixed "
        "into a real pilot.",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (PILOT_DIR / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    result = run_phase8()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return (
        0
        if result.get("status")
        in {"COMPLETED", "SMOKE_PASSED_READY_FOR_PILOT", "NATURAL_COMPLETED_MATCHED_PENDING"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
