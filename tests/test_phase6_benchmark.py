import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from collectiveeval.core import BenchmarkExample, TaskType
from collectiveeval.datasets import (
    BENCHMARK_VERSION,
    freeze_benchmark,
    load_jsonl,
    validate_benchmark_dir,
    validate_examples,
    verify_benchmark_manifest,
    write_jsonl,
)
from collectiveeval.metrics import abstention_scores, score_prediction
from collectiveeval.router import LearnedRouter, RouterTrainingExample, extract_uncertainty_signals
from collectiveeval.runner import run_experiment_from_path
from collectiveeval.storage import SQLiteStore
from collectiveeval.validation import validate_output_schema

ROOT = Path(__file__).resolve().parents[1]


def qa_example(
    example_id: str = "qa",
    *,
    split: str = "dev",
    answerable: bool = True,
) -> BenchmarkExample:
    return BenchmarkExample(
        id=example_id,
        task_type=TaskType.GROUNDED_QA,
        input={
            "question": "通知期限は何日前ですか。",
            "source_text": "契約第3条: 通知期限は30日前です。",
            "evidence_units": [
                {"id": "e1", "text": "契約第3条: 通知期限は30日前です。"},
                {"id": "d1", "text": "更新期限は10日前です。"},
            ],
        },
        gold={
            "answerable": answerable,
            "answer": "30日前" if answerable else "",
            "evidence": ["e1"] if answerable else [],
            "acceptable_answers": ["30日"] if answerable else [],
        },
        metadata={
            "source": "synthetic_unit_test",
            "difficulty": "easy",
            "tags": ["grounded_qa", "numeric_fact"],
            "split": split,
        },
    )


def extraction_example(example_id: str = "ext", *, split: str = "dev") -> BenchmarkExample:
    schema = {
        "type": "object",
        "required": ["company", "amount_jpy", "contract_date", "auto_renewal"],
        "additionalProperties": False,
        "properties": {
            "company": {"type": "string"},
            "amount_jpy": {"type": "integer"},
            "contract_date": {"type": "string", "format": "date"},
            "auto_renewal": {"type": "boolean"},
            "owner": {"type": ["string", "null"]},
        },
    }
    return BenchmarkExample(
        id=example_id,
        task_type=TaskType.STRUCTURED_EXTRACTION,
        input={
            "text": (
                "株式会社青空は2026年4月1日に150万円の契約を締結した。"
                "自動更新は無効で担当者は佐藤。"
            )
        },
        gold={
            "expected": {
                "company": "株式会社青空",
                "amount_jpy": 1500000,
                "contract_date": "2026-04-01",
                "auto_renewal": False,
                "owner": "佐藤",
            },
            "json_schema": schema,
            "schema_required": schema["required"],
        },
        metadata={
            "source": "synthetic_unit_test",
            "difficulty": "medium",
            "tags": ["structured_extraction", "date", "currency", "boolean"],
            "split": split,
        },
    )


class Phase6BenchmarkTests(unittest.TestCase):
    def test_benchmark_v1_files_validate_and_match_manifest(self) -> None:
        benchmark_dir = ROOT / "data" / "benchmark_v1"
        report = validate_benchmark_dir(benchmark_dir)
        verification = verify_benchmark_manifest(benchmark_dir / "manifest.json")

        self.assertEqual(report["version"], BENCHMARK_VERSION)
        self.assertEqual(report["examples"], 480)
        self.assertEqual(report["dev_examples"], 288)
        self.assertEqual(report["test_examples"], 192)
        self.assertEqual(
            report["counts"]["task_type"],
            {
                "business_summarization": 120,
                "grounded_qa": 120,
                "robustness": 120,
                "structured_extraction": 120,
            },
        )
        self.assertTrue(verification["ok"])

    def test_duplicate_ids_and_dev_test_leakage_are_rejected(self) -> None:
        duplicate_issues = validate_examples([qa_example("dup"), qa_example("dup")])
        self.assertTrue(any("duplicate ids" in issue for issue in duplicate_issues))

        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            write_jsonl(tmpdir / "dev.jsonl", [qa_example("leak", split="dev")])
            write_jsonl(tmpdir / "test.jsonl", [qa_example("leak", split="test")])

            with self.assertRaisesRegex(ValueError, "dev/test id leakage"):
                validate_benchmark_dir(tmpdir)

    def test_missing_metadata_and_invalid_task_payloads_are_rejected(self) -> None:
        payload = qa_example().model_dump(mode="json")
        payload["metadata"].pop("source")
        with self.assertRaises(ValidationError):
            BenchmarkExample.model_validate(payload)

        payload = qa_example().model_dump(mode="json")
        payload["gold"]["answer"] = ""
        with self.assertRaises(ValidationError):
            BenchmarkExample.model_validate(payload)

        payload = qa_example(answerable=False).model_dump(mode="json")
        payload["gold"]["answer"] = "30日前"
        with self.assertRaises(ValidationError):
            BenchmarkExample.model_validate(payload)

    def test_invalid_evidence_refs_schema_and_dates_are_detected(self) -> None:
        example = qa_example()
        payload = example.model_dump(mode="json")
        payload["gold"]["evidence"] = ["missing-id"]
        invalid_reference = BenchmarkExample.model_validate(payload)
        self.assertTrue(validate_examples([invalid_reference]))

        payload = extraction_example().model_dump(mode="json")
        payload["gold"]["json_schema"]["required"] = ["missing"]
        with self.assertRaises(ValidationError):
            BenchmarkExample.model_validate(payload)

        payload = extraction_example().model_dump(mode="json")
        payload["gold"]["expected"]["contract_date"] = "2026年4月1日"
        malformed_date = BenchmarkExample.model_validate(payload)
        self.assertTrue(
            any("malformed date" in issue for issue in validate_examples([malformed_date]))
        )

    def test_frozen_manifest_detects_file_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            write_jsonl(tmpdir / "dev.jsonl", [qa_example("dev", split="dev")])
            write_jsonl(tmpdir / "test.jsonl", [qa_example("test", split="test")])
            freeze_benchmark(tmpdir)

            self.assertTrue(verify_benchmark_manifest(tmpdir / "manifest.json")["ok"])
            with (tmpdir / "dev.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(qa_example("extra", split="dev").model_dump_json() + "\n")

            verification = verify_benchmark_manifest(tmpdir / "manifest.json")
            self.assertFalse(verification["ok"])
            self.assertIn("dev.jsonl sha256 mismatch", verification["reasons"])

    def test_learned_router_training_rejects_test_rows(self) -> None:
        signals = extract_uncertainty_signals(
            qa_example(split="test"),
            {"answer": "30日前", "citations": ["e1"], "abstain": False, "confidence": 0.9},
            confidence=0.9,
            model_id="mock/test",
        )
        with self.assertRaisesRegex(ValueError, "only on dev data"):
            LearnedRouter().train([RouterTrainingExample(signals, False, split="test")])

    def test_extraction_uses_json_schema_validation_and_field_metrics(self) -> None:
        example = extraction_example()
        output = {
            "company": "株式会社青空",
            "amount_jpy": "150万円",
            "contract_date": "令和8年4月1日",
            "auto_renewal": False,
            "owner": "佐藤",
            "unexpected": "hallucinated",
        }
        schema_valid, issues = validate_output_schema(example, output)
        scores = score_prediction(example, output)

        self.assertFalse(schema_valid)
        self.assertTrue(any("unexpected field" in issue for issue in issues))
        self.assertEqual(scores["json_schema_validity"], 0.0)
        self.assertEqual(scores["numeric_field_accuracy"], 1.0)
        self.assertEqual(scores["date_field_accuracy"], 1.0)
        self.assertEqual(scores["hallucinated_field_count"], 1.0)

    def test_groundedness_and_summarization_metrics_are_structured(self) -> None:
        qa_scores = score_prediction(
            qa_example(),
            {"answer": "10日前", "citations": ["d1"], "abstain": False, "confidence": 0.4},
        )
        self.assertNotIn("unsupported_claim_rate", qa_scores)
        self.assertIn("citation_correctness", qa_scores)
        self.assertIn("evidence_coverage", qa_scores)
        self.assertIn("unsupported_answer_claims_heuristic", qa_scores)

        summary_example = next(
            example
            for example in load_jsonl(ROOT / "data" / "benchmark_v1" / "dev.jsonl")
            if example.task_type == TaskType.BUSINESS_SUMMARIZATION
        )
        scores = score_prediction(summary_example, summary_example.gold)
        self.assertEqual(scores["decision_extraction_correctness"], 1.0)
        self.assertEqual(scores["action_item_correctness"], 1.0)
        self.assertEqual(scores["risk_extraction_correctness"], 1.0)

    def test_abstention_metrics_are_aggregated_in_normal_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            dataset = tmpdir / "qa.jsonl"
            write_jsonl(
                dataset,
                [
                    qa_example("answerable", split="mock"),
                    qa_example("unanswerable", split="mock", answerable=False),
                ],
            )
            config = {
                "experiment": {"name": "abstention_aggregation"},
                "dataset": {"path": str(dataset)},
                "strategy": {"type": "single_agent"},
                "model": {
                    "provider": "mock",
                    "name": "mock-accurate",
                    "temperature": 0.0,
                    "mock_mode": "gold_fixture",
                },
                "budget": {"max_calls": 2, "max_total_tokens": 8000},
            }
            config_path = tmpdir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            result = run_experiment_from_path(
                config_path,
                db_path=tmpdir / "runs.sqlite3",
                output_dir=tmpdir / "runs",
            )
            metrics = {
                row["metric_name"]: row["metric_value"]
                for row in SQLiteStore(tmpdir / "runs.sqlite3").get_run_metrics(result.run_id)
                if row["example_id"] is None
            }

        self.assertIn("abstention_precision", metrics)
        self.assertIn("abstention_recall", metrics)
        self.assertIn("abstention_f1", metrics)
        self.assertEqual(abstention_scores([False, True], [False, True])["abstention_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
