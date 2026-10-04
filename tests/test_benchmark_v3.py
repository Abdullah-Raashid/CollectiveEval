import json
import unittest
from pathlib import Path

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.datasets import (
    BENCHMARK_V3_VERSION,
    HARD_REASONING_OPERATIONS,
    load_jsonl,
    near_duplicate_leakage_report,
    validate_benchmark_dir,
    validate_examples,
)

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkV3Tests(unittest.TestCase):
    def test_benchmark_v3_files_validate_and_are_frozen(self) -> None:
        benchmark_dir = ROOT / "data" / "benchmark_v3"
        report = validate_benchmark_dir(benchmark_dir)
        manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(report["version"], BENCHMARK_V3_VERSION)
        self.assertEqual(report["examples"], 480)
        self.assertEqual(report["dev_examples"], 280)
        self.assertEqual(report["test_examples"], 200)
        self.assertEqual(
            report["counts"]["task_type"],
            {
                "business_summarization": 120,
                "grounded_qa": 120,
                "robustness": 120,
                "structured_extraction": 120,
            },
        )
        self.assertEqual(report["dev_sha256"], manifest["files"]["dev.jsonl"]["sha256"])
        self.assertEqual(report["test_sha256"], manifest["files"]["test.jsonl"]["sha256"])

    def test_v3_metadata_is_truthful_and_canonical(self) -> None:
        examples = load_jsonl(ROOT / "data" / "benchmark_v3" / "dev.jsonl")
        for example in examples:
            self.assertEqual(example.metadata["generator_version"], BENCHMARK_V3_VERSION)
            self.assertIn("surface_form_family", example.metadata)
            self.assertIn("reasoning_family", example.metadata)
            self.assertIn("generator_operations", example.metadata)
            self.assertEqual(len(example.metadata["tags"]), len(set(example.metadata["tags"])))
            self.assertTrue(
                set(example.metadata["difficulty_factors"]).issubset(
                    set(example.metadata["generator_operations"])
                )
            )
            if example.metadata["difficulty"] == "hard":
                self.assertTrue(
                    set(example.metadata["generator_operations"]) & HARD_REASONING_OPERATIONS
                )

    def test_v3_review_sample_is_dev_only(self) -> None:
        review = (ROOT / "reports" / "benchmark_v3_review_sample.md").read_text(encoding="utf-8")
        self.assertIn("DEV ONLY", review)
        self.assertNotIn("Split: `test`", review)

    def test_v3_grouped_split_and_near_duplicate_audit_are_clean(self) -> None:
        dev = load_jsonl(ROOT / "data" / "benchmark_v3" / "dev.jsonl")
        test = load_jsonl(ROOT / "data" / "benchmark_v3" / "test.jsonl")
        dev_templates = {example.metadata["template_family"] for example in dev}
        test_templates = {example.metadata["template_family"] for example in test}
        leakage = near_duplicate_leakage_report(dev, test, threshold=0.97)

        self.assertFalse(dev_templates & test_templates)
        self.assertEqual(leakage["template_family_overlap"], [])
        self.assertEqual(leakage["scenario_family_overlap"], [])
        self.assertEqual(leakage["high_similarity_pairs"], [])

    def test_v3_structural_and_reasoning_diversity_are_reported(self) -> None:
        examples = load_jsonl(ROOT / "data" / "benchmark_v3" / "dev.jsonl") + load_jsonl(
            ROOT / "data" / "benchmark_v3" / "test.jsonl"
        )
        surface_families = {example.metadata["surface_form_family"] for example in examples}
        reasoning_families = {example.metadata["reasoning_family"] for example in examples}
        audit = (ROOT / "reports" / "benchmark_v3_quality_audit.md").read_text(encoding="utf-8")

        self.assertGreaterEqual(len(surface_families), 8)
        self.assertGreaterEqual(len(reasoning_families), 8)
        self.assertIn("structural_diversity_counts", audit)

    def test_v3_audits_reject_reviewed_defect_patterns(self) -> None:
        extraction = next(
            example
            for example in load_jsonl(ROOT / "data" / "benchmark_v3" / "dev.jsonl")
            if example.task_type == TaskType.STRUCTURED_EXTRACTION
            and example.gold["expected"]["due_date"] is not None
        )
        payload = extraction.model_dump(mode="json")
        payload["gold"]["expected"]["due_date"] = "2026-01-01"
        invalid_chronology = BenchmarkExample.model_validate(payload)
        self.assertTrue(
            any(
                "due_date precedes issue_date" in issue
                for issue in validate_examples([invalid_chronology])
            )
        )

        tagged = extraction.model_dump(mode="json")
        tagged["metadata"]["tags"].append(tagged["metadata"]["tags"][0])
        duplicate_tag = BenchmarkExample.model_validate(tagged)
        self.assertTrue(
            any("duplicate metadata tags" in issue for issue in validate_examples([duplicate_tag]))
        )

        hard = next(
            example
            for example in load_jsonl(ROOT / "data" / "benchmark_v3" / "dev.jsonl")
            if example.metadata["difficulty"] == "hard"
        )
        payload = hard.model_dump(mode="json")
        payload["metadata"]["generator_operations"] = ["direct_lookup"]
        payload["metadata"]["difficulty_factors"] = ["direct_lookup"]
        unsupported_hard = BenchmarkExample.model_validate(payload)
        self.assertTrue(
            any("hard example lacks" in issue for issue in validate_examples([unsupported_hard]))
        )

        summary = next(
            example
            for example in load_jsonl(ROOT / "data" / "benchmark_v3" / "dev.jsonl")
            if example.task_type == TaskType.BUSINESS_SUMMARIZATION
        )
        payload = summary.model_dump(mode="json")
        payload["gold"]["supported_facts"] = payload["gold"]["supported_facts"][:1]
        incomplete_summary = BenchmarkExample.model_validate(payload)
        self.assertTrue(
            any(
                "supported_facts missing" in issue
                for issue in validate_examples([incomplete_summary])
            )
        )


if __name__ == "__main__":
    unittest.main()
