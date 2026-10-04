import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from collectiveeval.runner import run_experiment_from_path
from collectiveeval.storage import SQLiteStore

ROOT = Path(__file__).resolve().parents[1]


def write_config(tmpdir: Path, strategy_type: str = "single_agent") -> Path:
    config = {
        "experiment": {"name": f"phase2_{strategy_type}"},
        "dataset": {"path": str(ROOT / "data" / "splits" / "dev.mock.jsonl")},
        "strategy": {"type": strategy_type},
        "model": {
            "provider": "mock",
            "name": "mock-accurate",
            "temperature": 0.0,
            "mock_mode": "gold_fixture",
        },
        "budget": {"max_calls": 4, "max_total_tokens": 8000},
        "seed": 42,
    }
    path = tmpdir / "config.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return path


class RunnerStorageCliTests(unittest.TestCase):
    def test_run_experiment_persists_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            config_path = write_config(tmpdir)
            db_path = tmpdir / "collectiveeval.sqlite3"
            output_dir = tmpdir / "runs"

            result = run_experiment_from_path(
                config_path,
                db_path=db_path,
                output_dir=output_dir,
                max_examples=2,
            )

            self.assertEqual(result.status, "COMPLETED")
            self.assertEqual(result.examples, 2)
            self.assertGreaterEqual(result.aggregate_metrics["mean_task_score"], 0.5)
            artifact_dir = Path(result.artifact_dir)
            for name in (
                "config.yaml",
                "predictions.jsonl",
                "metrics.json",
                "summary.csv",
                "environment.json",
                "manifest.json",
                "run.log",
            ):
                self.assertTrue((artifact_dir / name).exists(), name)

            store = SQLiteStore(db_path)
            self.assertIsNotNone(store.get_experiment(result.experiment_id))
            self.assertEqual(store.get_run(result.run_id)["status"], "COMPLETED")
            self.assertEqual(len(store.get_run_predictions(result.run_id)), 2)
            self.assertTrue(store.get_run_metrics(result.run_id))

    def test_cli_run_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            config_path = write_config(tmpdir, "self_consistency")
            db_path = tmpdir / "collectiveeval.sqlite3"
            output_dir = tmpdir / "runs"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "collectiveeval.cli",
                    "run",
                    "--config",
                    str(config_path),
                    "--db",
                    str(db_path),
                    "--output-dir",
                    str(output_dir),
                    "--max-examples",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "COMPLETED")

            evaluated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "collectiveeval.cli",
                    "evaluate",
                    payload["run_id"],
                    "--db",
                    str(db_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
            )
            self.assertTrue(json.loads(evaluated.stdout))

    def test_dry_run_does_not_create_database(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            config_path = write_config(tmpdir)
            db_path = tmpdir / "collectiveeval.sqlite3"

            result = run_experiment_from_path(config_path, db_path=db_path, dry_run=True)

            self.assertEqual(result["dry_run"], True)
            self.assertFalse(db_path.exists())


if __name__ == "__main__":
    unittest.main()
