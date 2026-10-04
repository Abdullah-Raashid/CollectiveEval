import json
import unittest
from pathlib import Path

from collectiveeval.core import BenchmarkExample
from collectiveeval.datasets import (
    BENCHMARK_V3_1_VERSION,
    HARD_REASONING_OPERATIONS,
    load_jsonl,
    near_duplicate_leakage_report,
    operation_witness_report,
    validate_benchmark_dir,
    validate_examples,
)

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkV31Tests(unittest.TestCase):
    def test_benchmark_v3_1_files_validate_and_are_frozen(self) -> None:
        benchmark_dir = ROOT / "data" / "benchmark_v3_1"
        report = validate_benchmark_dir(benchmark_dir)
        manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(report["version"], BENCHMARK_V3_1_VERSION)
        self.assertEqual(report["examples"], 480)
        self.assertEqual(report["dev_examples"], 280)
        self.assertEqual(report["test_examples"], 200)
        self.assertEqual(report["dev_sha256"], manifest["files"]["dev.jsonl"]["sha256"])
        self.assertEqual(report["test_sha256"], manifest["files"]["test.jsonl"]["sha256"])

    def test_operation_witness_audit_is_clean(self) -> None:
        examples = load_jsonl(ROOT / "data" / "benchmark_v3_1" / "dev.jsonl") + load_jsonl(
            ROOT / "data" / "benchmark_v3_1" / "test.jsonl"
        )
        report = operation_witness_report(examples)

        self.assertEqual(report["affected_example_count"], 0)
        for example in examples:
            if example.metadata["difficulty"] == "hard":
                witnessed = (
                    set(example.metadata["generator_operations"]) & HARD_REASONING_OPERATIONS
                )
                self.assertTrue(witnessed)

    def test_v3_1_grouped_split_and_near_duplicate_audit_are_clean(self) -> None:
        dev = load_jsonl(ROOT / "data" / "benchmark_v3_1" / "dev.jsonl")
        test = load_jsonl(ROOT / "data" / "benchmark_v3_1" / "test.jsonl")
        dev_templates = {example.metadata["template_family"] for example in dev}
        test_templates = {example.metadata["template_family"] for example in test}
        leakage = near_duplicate_leakage_report(dev, test, threshold=0.97)

        self.assertFalse(dev_templates & test_templates)
        self.assertEqual(leakage["template_family_overlap"], [])
        self.assertEqual(leakage["scenario_family_overlap"], [])
        self.assertEqual(leakage["high_similarity_pairs"], [])

    def test_v3_1_review_sample_is_dev_only_and_shows_operations(self) -> None:
        review = (ROOT / "reports" / "benchmark_v3_1_review_sample.md").read_text(encoding="utf-8")

        self.assertIn("DEV ONLY", review)
        self.assertIn("Generator operations:", review)
        self.assertNotIn("Split: `test`", review)

    def test_v3_1_rejects_unwitnessed_operation_claims(self) -> None:
        example = next(
            item
            for item in load_jsonl(ROOT / "data" / "benchmark_v3_1" / "dev.jsonl")
            if item.metadata["reasoning_family"] == "numeric_normalization"
        )
        payload = example.model_dump(mode="json")
        payload["metadata"]["generator_operations"] = ["japanese_era_date_conversion"]
        payload["metadata"]["difficulty_factors"] = ["japanese_era_date_conversion"]
        defective = BenchmarkExample.model_validate(payload)

        self.assertTrue(
            any("operation not witnessed" in issue for issue in validate_examples([defective]))
        )


if __name__ == "__main__":
    unittest.main()
