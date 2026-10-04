import json
import tempfile
import unittest
from pathlib import Path

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.datasets import (
    BENCHMARK_V2_VERSION,
    freeze_benchmark,
    load_jsonl,
    near_duplicate_leakage_report,
    validate_benchmark_dir,
    validate_examples,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkV2Tests(unittest.TestCase):
    def test_benchmark_v2_files_validate_and_are_frozen(self) -> None:
        benchmark_dir = ROOT / "data" / "benchmark_v2"
        report = validate_benchmark_dir(benchmark_dir)
        manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(report["version"], BENCHMARK_V2_VERSION)
        self.assertEqual(report["examples"], 480)
        self.assertEqual(report["dev_examples"], 286)
        self.assertEqual(report["test_examples"], 194)
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

    def test_v2_examples_have_required_provenance_and_difficulty_factors(self) -> None:
        examples = load_jsonl(ROOT / "data" / "benchmark_v2" / "dev.jsonl")
        for example in examples:
            self.assertEqual(example.metadata["generator_version"], BENCHMARK_V2_VERSION)
            self.assertIn("template_family", example.metadata)
            self.assertIn("scenario_family", example.metadata)
            self.assertEqual(example.metadata["split_policy"], "heldout_template_family")
            self.assertTrue(example.metadata["difficulty_factors"])

    def test_v2_review_sample_is_dev_only(self) -> None:
        review = (ROOT / "reports" / "benchmark_v2_review_sample.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("DEV ONLY", review)
        self.assertNotIn("Split: `test`", review)

    def test_v2_grouped_split_and_near_duplicate_audit_are_clean(self) -> None:
        dev = load_jsonl(ROOT / "data" / "benchmark_v2" / "dev.jsonl")
        test = load_jsonl(ROOT / "data" / "benchmark_v2" / "test.jsonl")
        dev_templates = {example.metadata["template_family"] for example in dev}
        test_templates = {example.metadata["template_family"] for example in test}
        leakage = near_duplicate_leakage_report(dev, test, threshold=0.97)

        self.assertFalse(dev_templates & test_templates)
        self.assertEqual(leakage["template_family_overlap"], [])
        self.assertEqual(leakage["scenario_family_overlap"], [])
        self.assertEqual(leakage["high_similarity_pairs"], [])

    def test_v2_summarization_null_deadlines_are_not_inferred(self) -> None:
        examples = [
            example
            for example in load_jsonl(ROOT / "data" / "benchmark_v2" / "dev.jsonl")
            if example.task_type == TaskType.BUSINESS_SUMMARIZATION
        ]
        null_deadline_items = [
            (example, item)
            for example in examples
            for item in example.gold["action_items"]
            if item["deadline"] is None
        ]

        self.assertTrue(null_deadline_items)
        self.assertEqual(validate_examples(examples), [])

    def test_v2_audits_reject_reviewed_defect_patterns(self) -> None:
        summary = next(
            example
            for example in load_jsonl(ROOT / "data" / "benchmark_v2" / "dev.jsonl")
            if example.task_type == TaskType.BUSINESS_SUMMARIZATION
            and any(item["deadline"] is None for item in example.gold["action_items"])
        )
        payload = summary.model_dump(mode="json")
        for item in payload["gold"]["action_items"]:
            if item["deadline"] is None:
                item["deadline"] = payload["gold"]["action_items"][0]["deadline"]
                break
        defective_summary = BenchmarkExample.model_validate(payload)
        self.assertTrue(
            any(
                "action deadline unsupported" in issue
                for issue in validate_examples([defective_summary])
            )
        )

        extraction = next(
            example
            for example in load_jsonl(ROOT / "data" / "benchmark_v2" / "dev.jsonl")
            if example.task_type == TaskType.STRUCTURED_EXTRACTION
            and example.gold["expected"]["due_date"] is not None
        )
        payload = extraction.model_dump(mode="json")
        payload["gold"]["expected"]["due_date"] = None
        defective_extraction = BenchmarkExample.model_validate(payload)
        self.assertTrue(
            any("due_date is null" in issue for issue in validate_examples([defective_extraction]))
        )

    def test_v2_freeze_uses_requested_version(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            dev = load_jsonl(ROOT / "data" / "benchmark_v2" / "dev.jsonl")[:1]
            test = load_jsonl(ROOT / "data" / "benchmark_v2" / "test.jsonl")[:1]
            write_jsonl(tmpdir / "dev.jsonl", dev)
            write_jsonl(tmpdir / "test.jsonl", test)
            manifest = freeze_benchmark(tmpdir, version=BENCHMARK_V2_VERSION)

            self.assertEqual(manifest["version"], BENCHMARK_V2_VERSION)
            self.assertEqual(validate_benchmark_dir(tmpdir)["version"], BENCHMARK_V2_VERSION)


if __name__ == "__main__":
    unittest.main()
