import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from collectiveeval.core import BenchmarkExample, ModelSpec, StrategyResult, TaskType
from collectiveeval.failures import FailureType, annotate_failures
from collectiveeval.metrics import score_prediction
from collectiveeval.statistics import paired_bootstrap_comparison, summary_stats

ROOT = Path(__file__).resolve().parents[1]


def write_config(tmpdir: Path, *, name: str, model_name: str) -> Path:
    config = {
        "experiment": {"name": name},
        "dataset": {"path": str(ROOT / "data" / "splits" / "dev.mock.jsonl")},
        "strategy": {"type": "single_agent"},
        "model": {
            "provider": "mock",
            "name": model_name,
            "temperature": 0.0,
            "mock_mode": "gold_fixture" if "accurate" in model_name else "fixture",
            "input_cost_per_1k": 0.001,
            "output_cost_per_1k": 0.002,
        },
        "budget": {"max_calls": 4, "max_total_tokens": 8000},
        "estimate": {"output_tokens_per_call": 32},
        "seed": 42,
    }
    path = tmpdir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


class Phase3AnalysisTests(unittest.TestCase):
    def test_bootstrap_and_paired_comparison(self) -> None:
        stats = summary_stats([0.0, 1.0, 1.0], n_bootstrap=100, seed=1)
        comparison = paired_bootstrap_comparison(
            baseline={"a": 0.0, "b": 0.5},
            contender={"a": 1.0, "b": 0.5},
            metric="task_score",
            n_bootstrap=100,
            seed=1,
        )

        self.assertEqual(stats.n, 3)
        self.assertGreater(stats.mean, 0.0)
        self.assertEqual(comparison.common_examples, 2)
        self.assertGreater(comparison.mean_difference, 0.0)

    def test_failure_taxonomy_annotations(self) -> None:
        example = BenchmarkExample(
            id="qa",
            task_type=TaskType.GROUNDED_QA,
            input={
                "question": "何日前?",
                "source_text": "第3条: 解約通知は30日前まで。",
                "evidence_units": [{"id": "doc-3", "text": "第3条: 解約通知は30日前まで。"}],
            },
            gold={"answerable": True, "answer": "30日前", "evidence": ["doc-3"]},
            metadata={
                "source": "synthetic_unit_test",
                "difficulty": "medium",
                "split": "mock",
                "tags": ["negation"],
            },
        )
        result = StrategyResult(
            example_id="qa",
            strategy="single_agent",
            output={"answer": "不明", "citations": [], "abstain": False, "confidence": 0.2},
            confidence=0.2,
            metadata={"model": ModelSpec(model="mock-weak").model_id},
        )
        scores = score_prediction(example, result.output)
        labels = {
            annotation.failure_type
            for annotation in annotate_failures(example, result, scores)
        }

        self.assertIn(FailureType.MISSED_EVIDENCE, labels)
        self.assertIn(FailureType.HALLUCINATION, labels)
        self.assertIn(FailureType.NEGATION_ERROR, labels)

    def test_cli_reports_compare_and_estimate_cost(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            db_path = tmpdir / "collectiveeval.sqlite3"
            output_dir = tmpdir / "runs"
            weak_config = write_config(tmpdir, name="weak", model_name="mock-weak")
            strong_config = write_config(tmpdir, name="strong", model_name="mock-accurate")

            weak = self._cli_json(
                "run",
                "--config",
                str(weak_config),
                "--db",
                str(db_path),
                "--output-dir",
                str(output_dir),
            )
            strong = self._cli_json(
                "run",
                "--config",
                str(strong_config),
                "--db",
                str(db_path),
                "--output-dir",
                str(output_dir),
            )

            evaluated = self._cli_json("evaluate", weak["run_id"], "--db", str(db_path))
            compared = self._cli_json(
                "compare",
                weak["run_id"],
                strong["run_id"],
                "--db",
                str(db_path),
            )
            report = self._cli_json("report", strong["experiment_id"], "--db", str(db_path))
            reproduced = self._cli_json("reproduce", strong["experiment_id"], "--db", str(db_path))
            estimated = self._cli_json(
                "estimate-cost",
                "--config",
                str(strong_config),
                "--examples",
                "1",
            )

            self.assertIn("failure_counts", evaluated)
            self.assertGreater(evaluated["failure_counts"].get("HALLUCINATION", 0), 0)
            self.assertGreater(compared["comparisons"][0]["mean_difference"], 0.0)
            self.assertEqual(report["experiment"]["id"], strong["experiment_id"])
            self.assertEqual(reproduced["config_hash"], strong["config_hash"])
            self.assertGreater(estimated["estimated_cost_usd"], 0.0)

    def _cli_json(self, *args: str) -> dict:
        completed = subprocess.run(
            [sys.executable, "-m", "collectiveeval.cli", *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src")},
        )
        return json.loads(completed.stdout)


if __name__ == "__main__":
    unittest.main()
