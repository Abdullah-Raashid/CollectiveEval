import json
import os
import tempfile
import unittest
from pathlib import Path

from collectiveeval.config import load_yaml_config, redact_secrets, stable_config_hash
from collectiveeval.datasets import file_sha256, load_jsonl, validate_benchmark_file

ROOT = Path(__file__).resolve().parents[1]


class ConfigDatasetTests(unittest.TestCase):
    def test_mock_jsonl_validates_and_hashes(self) -> None:
        path = ROOT / "data" / "splits" / "dev.mock.jsonl"
        examples = load_jsonl(path)
        info = validate_benchmark_file(path)

        self.assertEqual(len(examples), 2)
        self.assertEqual(info["examples"], 2)
        self.assertEqual(info["sha256"], file_sha256(path))

    def test_config_env_resolution_and_hash_are_stable(self) -> None:
        os.environ["MODEL_NAME"] = "mock-accurate"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "model:\n  name: ${MODEL_NAME}\n  api_key: secret-value\n",
                encoding="utf-8",
            )
            config = load_yaml_config(config_path)

        self.assertEqual(config["model"]["name"], "mock-accurate")
        self.assertEqual(
            stable_config_hash(config),
            stable_config_hash(json.loads(json.dumps(config))),
        )
        self.assertEqual(redact_secrets(config)["model"]["api_key"], "***REDACTED***")

    def test_redaction_preserves_non_secret_token_budget_fields(self) -> None:
        config = {
            "model": {"api_key": "secret-value", "access_token": "secret-token"},
            "budget": {"max_total_tokens": 4000, "max_output_tokens": 700},
        }

        redacted = redact_secrets(config)

        self.assertEqual(redacted["model"]["api_key"], "***REDACTED***")
        self.assertEqual(redacted["model"]["access_token"], "***REDACTED***")
        self.assertEqual(redacted["budget"]["max_total_tokens"], 4000)
        self.assertEqual(redacted["budget"]["max_output_tokens"], 700)


if __name__ == "__main__":
    unittest.main()
