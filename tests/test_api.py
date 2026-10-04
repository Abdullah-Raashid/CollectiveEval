import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None

from collectiveeval.api import create_app

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(TestClient is None, "FastAPI is not installed")
class ApiTests(unittest.TestCase):
    def test_submit_and_inspect_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmpdir:
            tmpdir = Path(raw_tmpdir)
            app = create_app(db_path=tmpdir / "api.sqlite3", output_dir=tmpdir / "runs")
            client = TestClient(app)
            config = {
                "experiment": {"name": "api_phase2"},
                "dataset": {"path": str(ROOT / "data" / "splits" / "dev.mock.jsonl")},
                "strategy": {"type": "single_agent"},
                "model": {"provider": "mock", "name": "mock-accurate"},
                "budget": {"max_calls": 3, "max_total_tokens": 8000},
            }

            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            response = client.post("/experiments", json={"config": config})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "submitted")

            experiments = client.get("/experiments").json()
            self.assertEqual(len(experiments), 1)
            experiment_id = experiments[0]["id"]
            self.assertEqual(client.get(f"/experiments/{experiment_id}").status_code, 200)


if __name__ == "__main__":
    unittest.main()
