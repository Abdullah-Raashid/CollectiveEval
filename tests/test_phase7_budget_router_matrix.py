import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from collectiveeval.artifact_verify import verify_run_artifacts
from collectiveeval.budget import BudgetLedger, InferenceBudget, ModelCallRecord
from collectiveeval.budget_policy import ScientificBudgetMode, resolve_budget_policy
from collectiveeval.config import stable_config_hash
from collectiveeval.context import StrategyContext
from collectiveeval.core import BenchmarkExample, ModelSpec, TaskType
from collectiveeval.datasets import write_jsonl
from collectiveeval.matrix import generate_final_matrix, run_matrix
from collectiveeval.providers import MockProvider
from collectiveeval.router import (
    FORBIDDEN_ROUTER_FEATURE_SUBSTRINGS,
    ROUTER_FEATURE_NAMES,
    LearnedRouter,
    RouterFeatureRow,
    RouterTrainingTarget,
    RouterUtilityConfig,
    construct_router_target,
    extract_uncertainty_signals,
    ordered_feature_vector,
    validate_feature_schema,
)
from collectiveeval.router_training import build_router_training_dataset
from collectiveeval.runner import run_experiment_from_path
from collectiveeval.storage import SQLiteStore
from collectiveeval.strategies import SelfConsistency


def qa_example(example_id: str = "phase7-dev", *, split: str = "dev") -> BenchmarkExample:
    return BenchmarkExample(
        id=example_id,
        task_type=TaskType.GROUNDED_QA,
        input={
            "source_text": "A: renewal requires 30 days notice. B: support replies in 2 days.",
            "question": "How much notice is required?",
            "evidence_units": [
                {"id": "a", "text": "renewal requires 30 days notice"},
                {"id": "b", "text": "support replies in 2 days"},
            ],
        },
        gold={"answerable": True, "answer": "30 days", "evidence": ["a"]},
        metadata={
            "source": "synthetic_unit_test",
            "difficulty": "hard",
            "tags": ["phase7"],
            "split": split,
        },
    )


def write_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")


def base_config(dataset_path: Path, *, name: str = "phase7") -> dict[str, object]:
    return {
        "experiment": {"name": name},
        "dataset": {"path": str(dataset_path)},
        "model": {"provider": "mock", "name": "mock-accurate", "mock_mode": "gold_fixture"},
        "strategy": {"type": "single_agent"},
        "budget": {"max_calls": 2, "max_total_tokens": 4000},
        "budget_policy": {"mode": "MATCHED_TOKENS"},
    }


class Phase7BudgetRouterMatrixTests(unittest.TestCase):
    def test_budget_policy_separates_scientific_and_execution_safety_budgets(self) -> None:
        policy = resolve_budget_policy(
            {"budget_tier": "small", "budget_policy": {"mode": "MATCHED_CALLS"}},
            run_level_max_cost_usd=1.25,
        )

        self.assertEqual(policy.scientific_budget.mode, ScientificBudgetMode.MATCHED_CALLS)
        self.assertEqual(policy.scientific_budget.max_calls, 2)
        self.assertIsNone(policy.scientific_budget.max_estimated_cost_usd)
        self.assertEqual(policy.execution_safety_budget.max_cost_usd, 1.25)

    def test_budget_ledger_records_before_after_and_remaining_budget(self) -> None:
        ledger = BudgetLedger(
            InferenceBudget(max_calls=2, max_total_tokens=100),
            run_id="phase7-budget",
        )
        ledger.record_call(
            ModelCallRecord(
                run_id="phase7-budget",
                example_id="example",
                strategy="single_agent",
                provider="mock",
                model="mock-accurate",
                role="generator",
                input_tokens=10,
                output_tokens=5,
                latency_ms=3.0,
                estimated_cost_usd=0.0,
                finish_reason="stop",
            )
        )

        accounting = ledger.records[0].metadata["budget_accounting"]
        self.assertEqual(accounting["consumed_before_call"]["model_calls"], 0)
        self.assertEqual(accounting["consumed_after_call"]["model_calls"], 1)
        self.assertEqual(accounting["remaining_after_call"]["remaining_calls"], 1)
        self.assertEqual(accounting["remaining_after_call"]["remaining_total_tokens"], 85)

    def test_bounded_concurrency_does_not_bypass_call_budget(self) -> None:
        context = StrategyContext(
            {"mock": MockProvider(mock_mode="gold_fixture")},
            BudgetLedger(InferenceBudget(max_calls=2, max_total_tokens=4000), run_id="concurrent"),
            concurrency_limit=4,
        )
        result = asyncio.run(
            SelfConsistency(k=4, model=ModelSpec(max_tokens=256)).run(qa_example(), context)
        )

        self.assertEqual(result.model_calls, 2)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(len(context.ledger.records), 2)

    def test_router_features_exclude_gold_derived_names_and_use_input_evidence(self) -> None:
        validate_feature_schema()
        for name in ROUTER_FEATURE_NAMES:
            lowered = name.lower()
            self.assertFalse(
                any(token in lowered for token in FORBIDDEN_ROUTER_FEATURE_SUBSTRINGS),
                msg=name,
            )

        signals = extract_uncertainty_signals(
            qa_example(),
            {"answer": "30 days", "citations": ["a"], "abstain": False, "confidence": 0.8},
            confidence=0.8,
            model_id="mock/mock-weak",
        )

        self.assertEqual(signals.citation_count, 1)
        self.assertEqual(signals.citation_coverage, 0.5)
        self.assertGreater(signals.source_evidence_length, 0)

    def test_router_target_keeps_gold_derived_utility_out_of_feature_rows(self) -> None:
        target = construct_router_target(
            example_id="example",
            initial_quality=0.4,
            escalated_quality=0.8,
            initial_tokens=100,
            escalated_tokens=300,
            initial_calls=1,
            escalated_calls=2,
            initial_cost=0.0,
            escalated_cost=0.0,
            config=RouterUtilityConfig(lambda_tokens=0.5, min_utility_gain=0.2),
        )

        self.assertAlmostEqual(target.quality_gain, 0.4)
        self.assertAlmostEqual(target.utility_gain, 0.3)
        self.assertAlmostEqual(target.initial_quality, 0.4)
        self.assertAlmostEqual(target.escalated_quality, 0.8)
        self.assertEqual(target.label, 1)

    def test_learned_router_artifact_round_trips_with_strict_feature_schema(self) -> None:
        positive_features = dict.fromkeys(ROUTER_FEATURE_NAMES, 0.0)
        negative_features = dict.fromkeys(ROUTER_FEATURE_NAMES, 0.0)
        positive_features.update({"initial_confidence": 0.1, "validator_failures": 3.0})
        negative_features.update({"initial_confidence": 0.95, "schema_valid": 1.0})
        rows = [
            RouterFeatureRow(example_id="pos", split="router_train", features=positive_features),
            RouterFeatureRow(example_id="neg", split="router_train", features=negative_features),
        ]
        targets = [
            RouterTrainingTarget(
                example_id="pos",
                quality_gain=0.5,
                extra_tokens=100,
                extra_calls=1,
                extra_cost=0.0,
                utility_gain=0.5,
                label=1,
            ),
            RouterTrainingTarget(
                example_id="neg",
                quality_gain=0.0,
                extra_tokens=100,
                extra_calls=1,
                extra_cost=0.0,
                utility_gain=0.0,
                label=0,
            ),
        ]
        router = LearnedRouter()
        router.train_feature_rows(rows, targets)

        with tempfile.TemporaryDirectory() as raw_tmpdir:
            artifact_dir = Path(raw_tmpdir) / "router"
            router.save_artifact(artifact_dir, training_manifest={"examples": 2})
            loaded = LearnedRouter.load(artifact_dir)
            self.assertAlmostEqual(
                router.escalation_probability_from_features(positive_features),
                loaded.escalation_probability_from_features(positive_features),
            )
            self.assertIn("joblib", (artifact_dir / "training_manifest.json").read_text())

        with self.assertRaisesRegex(ValueError, "unknown router features"):
            ordered_feature_vector({**positive_features, "initial_task_score": 1.0})

    def test_learned_router_training_rejects_validation_rows_and_test_examples(self) -> None:
        row = RouterFeatureRow(
            example_id="dev-only",
            split="router_validation",
            features=dict.fromkeys(ROUTER_FEATURE_NAMES, 0.0),
        )
        router = LearnedRouter()
        with self.assertRaisesRegex(ValueError, "router_train"):
            router.train_feature_rows(
                [row],
                [
                    RouterTrainingTarget(
                        example_id="dev-only",
                        quality_gain=0.0,
                        extra_tokens=0,
                        extra_calls=0,
                        extra_cost=0.0,
                        utility_gain=0.0,
                        label=0,
                    )
                ],
            )

        with self.assertRaisesRegex(ValueError, "test examples"):
            asyncio.run(build_router_training_dataset([qa_example("test-only", split="test")]))

    def test_budget_exhaustion_is_recorded_as_an_example_result(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            dataset_path = tmpdir / "dev.jsonl"
            write_jsonl(dataset_path, [qa_example()])
            config = base_config(dataset_path, name="phase7-budget-exhaustion")
            config["budget"] = {"max_calls": 0, "max_total_tokens": 4000}
            config_path = tmpdir / "config.yaml"
            write_config(config_path, config)

            result = run_experiment_from_path(
                config_path,
                db_path=tmpdir / "runs.sqlite3",
                output_dir=tmpdir / "runs",
            )

            self.assertEqual(result.status, "COMPLETED")
            predictions = (
                Path(result.artifact_dir).joinpath("predictions.jsonl").read_text().splitlines()
            )
            payload = json.loads(predictions[0])
            self.assertEqual(
                payload["metadata"]["budget_exhaustion_reason"],
                "model call budget exhausted",
            )
            self.assertEqual(payload["model_calls"], 0)

    def test_matrix_runner_refuses_test_split_even_for_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            dataset_path = tmpdir / "test.jsonl"
            write_jsonl(dataset_path, [qa_example("held-out", split="test")])
            config = base_config(dataset_path, name="phase7-test-refusal")
            config_path = tmpdir / "config.yaml"
            write_config(config_path, config)
            manifest = {
                "entries": [
                    {
                        "name": "single",
                        "family": "single_agent",
                        "budget_label": "small",
                        "execution_ready": True,
                        "config_path": config_path.name,
                        "config_hash": stable_config_hash(config),
                    }
                ]
            }
            (tmpdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "test split"):
                run_matrix(
                    tmpdir / "manifest.json",
                    db_path=tmpdir / "runs.sqlite3",
                    output_dir=tmpdir / "runs",
                    dry_run=True,
                )

    def test_ablation_matrix_entries_are_executable(self) -> None:
        entries = generate_final_matrix(
            base_config(Path("data/benchmark_v1/dev.jsonl")),
            token_budgets=[4000],
            include_ablations=True,
        )

        ablations = [entry for entry in entries if entry.family == "ablation"]
        self.assertEqual(len(ablations), 6)
        self.assertTrue(all(entry.execution_ready for entry in ablations))

    def test_matrix_resume_uses_resolved_budget_hash_and_artifacts_verify(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            dataset_path = tmpdir / "dev.jsonl"
            write_jsonl(dataset_path, [qa_example()])
            config = base_config(dataset_path, name="phase7-matrix")
            config_path = tmpdir / "config.yaml"
            write_config(config_path, config)
            manifest = {
                "entries": [
                    {
                        "name": "single",
                        "family": "single_agent",
                        "budget_label": "small",
                        "execution_ready": True,
                        "config_path": config_path.name,
                        "config_hash": stable_config_hash(config),
                    }
                ]
            }
            manifest_path = tmpdir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            db_path = tmpdir / "runs.sqlite3"

            first = run_matrix(
                manifest_path,
                db_path=db_path,
                output_dir=tmpdir / "runs",
                max_examples=1,
            )
            self.assertEqual(first["results"][0]["status"], "COMPLETED")
            run_id = first["results"][0]["run_id"]
            self.assertTrue(verify_run_artifacts(SQLiteStore(db_path), run_id)["ok"])

            second = run_matrix(
                manifest_path,
                db_path=db_path,
                output_dir=tmpdir / "runs",
                max_examples=1,
                resume=True,
            )
            self.assertEqual(second["results"][0]["status"], "SKIPPED_COMPLETED")
            self.assertEqual(second["results"][0]["run_id"], run_id)


if __name__ == "__main__":
    unittest.main()
