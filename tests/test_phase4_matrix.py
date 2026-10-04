import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from collectiveeval.matrix import (
    generate_final_matrix,
    validate_manifest,
    validate_matched_budgets,
    write_matrix_configs,
)
from collectiveeval.runner import run_experiment_from_path
from collectiveeval.strategy_factory import strategy_from_config

ROOT = Path(__file__).resolve().parents[1]


def base_config() -> dict:
    return {
        "experiment": {"name": "matrix_base"},
        "dataset": {"path": str(ROOT / "data" / "splits" / "dev.mock.jsonl")},
        "strategy": {"type": "single_agent"},
        "model": {"provider": "mock", "name": "mock-accurate", "temperature": 0.0},
        "budget": {"max_calls": 8, "max_total_tokens": 8000},
        "seed": 42,
    }


class Phase4MatrixTests(unittest.TestCase):
    def test_generate_final_matrix_and_validate_budgets(self) -> None:
        entries = generate_final_matrix(base_config(), token_budgets=[4000, 8000])
        report = validate_matched_budgets(entries)

        self.assertEqual(len(entries), 24)
        self.assertEqual(report.ready_entries, 22)
        self.assertTrue(report.ok)
        self.assertIn("tokens_4000", report.budget_groups)
        self.assertEqual(report.budget_groups["tokens_4000"]["planned_entries"], 1)
        self.assertIn(
            "adaptive_router_learned",
            {entry.config["matrix"]["entry_name"] for entry in entries},
        )

    def test_ready_only_matrix_writes_and_runs_a_config(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            entries = generate_final_matrix(base_config(), token_budgets=[5000], ready_only=True)
            manifest = write_matrix_configs(entries, tmpdir / "matrix")
            validation = validate_manifest(tmpdir / "matrix" / "manifest.json")

            self.assertEqual(len(entries), 11)
            self.assertEqual(len(manifest["entries"]), 11)
            self.assertTrue(validation["ok"])

            first_config = tmpdir / "matrix" / manifest["entries"][0]["config_path"]
            result = run_experiment_from_path(
                first_config,
                db_path=tmpdir / "matrix.sqlite3",
                output_dir=tmpdir / "runs",
                max_examples=1,
            )
            self.assertEqual(result.status, "COMPLETED")

    def test_planned_matrix_entry_refuses_execution(self) -> None:
        entries = generate_final_matrix(base_config(), token_budgets=[5000])
        planned = next(entry for entry in entries if not entry.execution_ready)

        with self.assertRaises(ValueError):
            strategy_from_config(planned.config)

    def test_cli_matrix_generate_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            base_path = tmpdir / "base.yaml"
            base_path.write_text(
                yaml.safe_dump(base_config(), allow_unicode=True),
                encoding="utf-8",
            )
            output_dir = tmpdir / "matrix"

            generated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "collectiveeval.cli",
                    "matrix",
                    "generate",
                    "--base-config",
                    str(base_path),
                    "--output-dir",
                    str(output_dir),
                    "--token-budget",
                    "6000",
                    "--ready-only",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
            )
            payload = json.loads(generated.stdout)
            self.assertEqual(len(payload["entries"]), 11)
            self.assertTrue(payload["validation"]["ok"])

            validated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "collectiveeval.cli",
                    "matrix",
                    "validate",
                    "--manifest",
                    str(output_dir / "manifest.json"),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
            )
            self.assertTrue(json.loads(validated.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
