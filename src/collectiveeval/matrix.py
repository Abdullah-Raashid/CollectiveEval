"""Final experiment matrix generation and matched-budget validation."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from collectiveeval.budget import BUDGET_SEMANTICS_VERSION
from collectiveeval.budget_policy import normalized_budget_config, resolve_budget_policy
from collectiveeval.config import stable_config_hash
from collectiveeval.datasets import load_jsonl, write_jsonl
from collectiveeval.runner import ExperimentRunResult, run_experiment_from_path
from collectiveeval.storage import SQLiteStore


@dataclass(frozen=True)
class MatrixEntry:
    """One generated experiment config plus matrix metadata."""

    name: str
    family: str
    budget_label: str
    config: dict[str, Any]
    execution_ready: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def config_hash(self) -> str:
        return stable_config_hash(self.config)

    def manifest_row(self, config_path: str | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "budget_label": self.budget_label,
            "config_hash": self.config_hash,
            "execution_ready": self.execution_ready,
            "notes": self.notes,
        }
        if config_path is not None:
            row["config_path"] = config_path
        return row


@dataclass(frozen=True)
class MatrixValidationReport:
    """Result of validating matrix budget comparability."""

    ok: bool
    entries: int
    ready_entries: int
    budget_groups: dict[str, dict[str, Any]]
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "entries": self.entries,
            "ready_entries": self.ready_entries,
            "budget_groups": self.budget_groups,
            "issues": self.issues,
        }


def generate_final_matrix(
    base_config: dict[str, Any],
    *,
    token_budgets: list[int],
    include_ablations: bool = False,
    ready_only: bool = False,
) -> list[MatrixEntry]:
    """Generate final matrix configs across fixed token budgets."""

    entries: list[MatrixEntry] = []
    for token_budget in token_budgets:
        budget_label = f"tokens_{token_budget}"
        specs = _final_strategy_specs(base_config)
        if include_ablations:
            specs = [*specs, *_ablation_specs()]
        for spec in specs:
            if ready_only and not spec["execution_ready"]:
                continue
            entries.append(
                _entry_from_spec(
                    base_config,
                    spec=spec,
                    token_budget=token_budget,
                    budget_label=budget_label,
                )
            )
    return entries


def validate_matched_budgets(entries: list[MatrixEntry]) -> MatrixValidationReport:
    """Ensure all configs in a fixed-budget group share budget ceilings."""

    issues: list[str] = []
    budget_groups: dict[str, dict[str, Any]] = {}
    seen_hashes: set[str] = set()
    for entry in entries:
        if entry.config_hash in seen_hashes:
            issues.append(f"duplicate config hash: {entry.name}")
        seen_hashes.add(entry.config_hash)

        budget = dict(entry.config.get("budget", {}))
        signature = {
            "max_total_tokens": budget.get("max_total_tokens"),
            "max_cost_usd": budget.get("max_cost_usd"),
            "max_latency_ms": budget.get("max_latency_ms"),
        }
        group = budget_groups.setdefault(
            entry.budget_label,
            {
                "signature": signature,
                "entries": 0,
                "ready_entries": 0,
                "planned_entries": 0,
            },
        )
        group["entries"] += 1
        if entry.execution_ready:
            group["ready_entries"] += 1
        else:
            group["planned_entries"] += 1
        if group["signature"] != signature:
            issues.append(
                f"budget mismatch in {entry.budget_label}: {entry.name} has {signature}, "
                f"expected {group['signature']}"
            )

    return MatrixValidationReport(
        ok=not issues,
        entries=len(entries),
        ready_entries=sum(1 for entry in entries if entry.execution_ready),
        budget_groups=budget_groups,
        issues=issues,
    )


def write_matrix_configs(entries: list[MatrixEntry], output_dir: str | Path) -> dict[str, Any]:
    """Write generated matrix configs and a manifest."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for entry in entries:
        file_name = f"{_safe_name(entry.name)}.yaml"
        config_path = output_path / file_name
        config_path.write_text(
            yaml.safe_dump(entry.config, allow_unicode=True, sort_keys=True),
            encoding="utf-8",
        )
        manifest_entries.append(entry.manifest_row(config_path=file_name))

    report = validate_matched_budgets(entries)
    manifest = {
        "version": 1,
        "description": "CollectiveEval generated experiment matrix.",
        "entries": manifest_entries,
        "validation": report.to_dict(),
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_matrix_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "entries" not in payload:
        raise ValueError("matrix manifest must contain entries")
    return payload


def validate_manifest(path: str | Path) -> dict[str, Any]:
    """Validate an already-written manifest."""

    manifest_path = Path(path)
    manifest = load_matrix_manifest(manifest_path)
    issues: list[str] = []
    seen_paths: set[str] = set()
    for raw_entry in manifest.get("entries", []):
        if not isinstance(raw_entry, dict):
            issues.append("manifest entry must be an object")
            continue
        config_path = raw_entry.get("config_path")
        if not isinstance(config_path, str):
            issues.append(f"missing config_path for {raw_entry.get('name', '<unknown>')}")
            continue
        if config_path in seen_paths:
            issues.append(f"duplicate config_path: {config_path}")
        seen_paths.add(config_path)
        full_path = manifest_path.parent / config_path
        if not full_path.exists():
            issues.append(f"missing config file: {config_path}")
            continue
        loaded = yaml.safe_load(full_path.read_text(encoding="utf-8")) or {}
        if stable_config_hash(loaded) != raw_entry.get("config_hash"):
            issues.append(f"config hash mismatch: {config_path}")

    embedded = manifest.get("validation", {})
    if isinstance(embedded, dict) and embedded.get("issues"):
        issues.extend(str(issue) for issue in embedded["issues"])
    return {
        "ok": not issues,
        "entries": len(manifest.get("entries", [])),
        "issues": issues,
    }


def run_matrix(
    manifest_path: str | Path,
    *,
    db_path: str | Path,
    output_dir: str | Path,
    strategy: str | None = None,
    task_family: str | None = None,
    budget_tier: str | None = None,
    seed: int | None = None,
    max_examples: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    max_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Execute ready matrix entries on DEV only, with duplicate-skip semantics."""

    manifest_file = Path(manifest_path)
    manifest = load_matrix_manifest(manifest_file)
    store = SQLiteStore(db_path)
    results: list[dict[str, Any]] = []
    for raw_entry in manifest.get("entries", []):
        if not isinstance(raw_entry, dict) or not raw_entry.get("execution_ready", True):
            continue
        if strategy and raw_entry.get("family") != strategy and raw_entry.get("name") != strategy:
            continue
        if budget_tier and raw_entry.get("budget_label") != budget_tier:
            continue
        config_path = manifest_file.parent / str(raw_entry["config_path"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        needs_temp_config = seed is not None
        if seed is not None:
            config["seed"] = seed
        examples = load_jsonl(config["dataset"]["path"])
        if task_family:
            examples = [example for example in examples if str(example.task_type) == task_family]
            filtered_dataset_path = (
                manifest_file.parent / f".phase7_{raw_entry['name']}_{task_family}_dataset.jsonl"
            )
            write_jsonl(filtered_dataset_path, examples)
            config = deepcopy(config)
            config["dataset"] = dict(config.get("dataset", {}))
            config["dataset"]["path"] = str(filtered_dataset_path)
            needs_temp_config = True
        splits = {example.metadata.get("split") for example in examples}
        if "test" in splits:
            raise ValueError("Phase 7 matrix runner refuses to execute test split configs")
        if splits - {"dev", "mock"}:
            raise ValueError(f"Phase 7 matrix runner accepts only dev/mock splits, got {splits}")
        if needs_temp_config:
            temp_config_path = manifest_file.parent / (
                f".phase7_{raw_entry['name']}_seed_{seed or 'default'}.yaml"
            )
            temp_config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
                encoding="utf-8",
            )
            config_path = temp_config_path
        config_hash = _resolved_config_hash(
            config,
            max_cost_usd=max_cost_usd,
            max_examples=max_examples,
        )
        existing = store.find_completed_run_by_config_hash(config_hash)
        if existing and resume:
            results.append(
                {
                    "entry": raw_entry.get("name"),
                    "status": "SKIPPED_COMPLETED",
                    "run_id": existing["id"],
                    "config_hash": config_hash,
                }
            )
            continue
        if existing and not resume:
            results.append(
                {
                    "entry": raw_entry.get("name"),
                    "status": "DUPLICATE_COMPLETED",
                    "run_id": existing["id"],
                    "config_hash": config_hash,
                }
            )
            continue
        if dry_run:
            results.append(
                {
                    "entry": raw_entry.get("name"),
                    "status": "DRY_RUN",
                    "config_hash": config_hash,
                    "examples": len(examples[:max_examples] if max_examples else examples),
                }
            )
            continue
        result = cast(
            ExperimentRunResult,
            run_experiment_from_path(
                config_path,
                db_path=db_path,
                output_dir=output_dir,
                max_examples=max_examples,
                max_cost_usd=max_cost_usd,
            ),
        )
        results.append(
            {
                "entry": raw_entry.get("name"),
                "status": result.status,
                "run_id": result.run_id,
                "experiment_id": result.experiment_id,
                "config_hash": result.config_hash,
            }
        )
    return {"matrix": str(manifest_file), "results": results}


def _resolved_config_hash(
    config: dict[str, Any],
    *,
    max_cost_usd: float | None,
    max_examples: int | None,
) -> str:
    policy = resolve_budget_policy(config, run_level_max_cost_usd=max_cost_usd)
    resolved = json.loads(json.dumps(config, ensure_ascii=False, default=str))
    resolved["budget"] = normalized_budget_config(policy)
    resolved["budget_semantics_version"] = BUDGET_SEMANTICS_VERSION
    resolved.setdefault("provider_retry", {}).setdefault(
        "unknown_usage_policy", "UNKNOWN_USAGE_STOP.v1"
    )
    resolved["budget_policy"] = policy.manifest_dict()
    if max_examples is not None:
        resolved.setdefault("execution_overrides", {})
        resolved["execution_overrides"]["max_examples"] = max_examples
    return stable_config_hash(resolved)


def _entry_from_spec(
    base_config: dict[str, Any],
    *,
    spec: dict[str, Any],
    token_budget: int,
    budget_label: str,
) -> MatrixEntry:
    config = deepcopy(base_config)
    config.setdefault("experiment", {})
    config["experiment"]["name"] = f"{spec['name']}_{budget_label}"
    config["strategy"] = deepcopy(spec["strategy"])
    config.setdefault("budget", {})
    config["budget"]["max_total_tokens"] = token_budget
    config.setdefault("matrix", {})
    config["matrix"].update(
        {
            "entry_name": spec["name"],
            "family": spec["family"],
            "budget_label": budget_label,
            "execution_ready": spec["execution_ready"],
            "notes": spec["notes"],
        }
    )
    return MatrixEntry(
        name=config["experiment"]["name"],
        family=str(spec["family"]),
        budget_label=budget_label,
        config=config,
        execution_ready=bool(spec["execution_ready"]),
        notes=list(spec["notes"]),
    )


def _final_strategy_specs(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    base_model = dict(base_config.get("model", {"provider": "mock", "name": "mock-accurate"}))
    panel_two = [
        {**base_model, "role": "qa_specialist"},
        {**base_model, "role": "business_specialist"},
    ]
    panel_three = [
        *panel_two,
        {**base_model, "role": "robustness_specialist"},
    ]
    return [
        _spec("single_agent", "single_agent", {"type": "single_agent"}),
        _spec("self_consistency_k2", "self_consistency", {"type": "self_consistency", "k": 2}),
        _spec("self_consistency_k4", "self_consistency", {"type": "self_consistency", "k": 4}),
        _spec("critic_reviser", "critic_reviser", {"type": "critic_reviser"}),
        _spec("debate_2x1", "multi_agent_debate", {"type": "debate", "agents": 2, "rounds": 1}),
        _spec("debate_3x1", "multi_agent_debate", {"type": "debate", "agents": 3, "rounds": 1}),
        _spec("debate_3x2", "multi_agent_debate", {"type": "debate", "agents": 3, "rounds": 2}),
        _spec("debate_4x2", "multi_agent_debate", {"type": "debate", "agents": 4, "rounds": 2}),
        _spec(
            "heterogeneous_panel_2",
            "heterogeneous_panel",
            {"type": "panel", "models": panel_two},
        ),
        _spec(
            "heterogeneous_panel_3",
            "heterogeneous_panel",
            {"type": "panel", "models": panel_three},
        ),
        _spec(
            "adaptive_router_heuristic",
            "adaptive_router",
            {"type": "adaptive_router", "router": "heuristic"},
        ),
        _spec(
            "adaptive_router_learned",
            "adaptive_router",
            {
                "type": "adaptive_router",
                "router": "learned",
                "requires_trained_router": True,
            },
            execution_ready=False,
            notes=["Requires a trained router artifact before execution."],
        ),
    ]


def _ablation_specs() -> list[dict[str, Any]]:
    ablation_note = ["Executable Phase 7 ablation control."]
    return [
        _spec(
            "ablation_debate_without_peer_evidence",
            "ablation",
            {"type": "debate", "agents": 3, "rounds": 2, "peer_evidence": False},
            notes=ablation_note,
        ),
        _spec(
            "ablation_debate_without_specialist_roles",
            "ablation",
            {"type": "debate", "agents": 3, "rounds": 2, "specialist_roles": False},
            notes=ablation_note,
        ),
        _spec(
            "ablation_duplicated_homogeneous_agents",
            "ablation",
            {"type": "debate", "agents": 3, "rounds": 2, "homogeneous_agents": True},
            notes=ablation_note,
        ),
        _spec(
            "ablation_critic_without_revision",
            "ablation",
            {"type": "critic_reviser", "revision_enabled": False},
            notes=ablation_note,
        ),
        _spec(
            "ablation_revision_without_critic",
            "ablation",
            {"type": "critic_reviser", "critic_enabled": False},
            notes=ablation_note,
        ),
        _spec(
            "ablation_router_without_uncertainty_signals",
            "ablation",
            {"type": "adaptive_router", "uncertainty_signals": False},
            notes=ablation_note,
        ),
    ]


def _spec(
    name: str,
    family: str,
    strategy: dict[str, Any],
    *,
    execution_ready: bool = True,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "strategy": strategy,
        "execution_ready": execution_ready,
        "notes": notes or [],
    }


def _safe_name(name: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_" for character in name
    )
