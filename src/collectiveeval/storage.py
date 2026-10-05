"""SQLite persistence for experiments, runs, predictions, metrics, and calls."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from collectiveeval.budget import LogicalCallRecord, ModelCallRecord
from collectiveeval.core import BenchmarkExample
from collectiveeval.failed_outputs import UNSCORABLE, FailedPrediction, PredictionResult

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  dataset_hash TEXT NOT NULL,
  git_commit TEXT NOT NULL,
  python_version TEXT NOT NULL,
  created_at TEXT NOT NULL,
  config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL,
  strategy TEXT NOT NULL,
  status TEXT NOT NULL,
  start_ts TEXT NOT NULL,
  end_ts TEXT,
  artifact_dir TEXT,
  error TEXT,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS examples (
  example_id TEXT PRIMARY KEY,
  task_type TEXT NOT NULL,
  language TEXT NOT NULL,
  split TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS logical_model_calls (
  logical_call_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  example_id TEXT NOT NULL,
  strategy TEXT NOT NULL,
  role TEXT NOT NULL,
  start_timestamp TEXT NOT NULL,
  end_timestamp TEXT,
  lifecycle_latency_ms REAL NOT NULL,
  outcome TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS provider_attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  attempt_id TEXT NOT NULL UNIQUE,
  logical_call_id TEXT NOT NULL,
  attempt_index INTEGER NOT NULL,
  outcome TEXT NOT NULL,
  usage_status TEXT NOT NULL,
  run_id TEXT NOT NULL,
  example_id TEXT NOT NULL,
  strategy TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  role TEXT NOT NULL,
  prompt_version TEXT NOT NULL DEFAULT '',
  requested_seed INTEGER,
  temperature REAL,
  max_tokens INTEGER,
  timestamp TEXT NOT NULL,
  start_timestamp TEXT,
  end_timestamp TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  usage_source TEXT,
  latency_ms REAL NOT NULL,
  estimated_cost_usd REAL NOT NULL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  finish_reason TEXT NOT NULL,
  normalized_error TEXT,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE VIEW IF NOT EXISTS model_calls AS SELECT * FROM provider_attempts;

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  example_id TEXT NOT NULL,
  strategy TEXT NOT NULL,
  output_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  usage_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  example_id TEXT,
  metric_name TEXT NOT NULL,
  metric_value REAL,
  metric_status TEXT NOT NULL DEFAULT 'SCORED',
  FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS prediction_checkpoint ON predictions(run_id, example_id);
CREATE UNIQUE INDEX IF NOT EXISTS example_metric_checkpoint
ON metrics(run_id, example_id, metric_name)
WHERE example_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  experiment_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id),
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS failure_annotations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  example_id TEXT NOT NULL,
  failure_type TEXT NOT NULL,
  source TEXT NOT NULL,
  rationale TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);
PRAGMA user_version = 3;
"""


class SQLiteStore:
    """Small sqlite3-backed repository for Phase 2 persistence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            legacy = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_calls'"
            ).fetchone()
            if legacy:
                raise ValueError(
                    "Legacy scientific DB is immutable; use ReadOnlyPilotStore "
                    "for reading and a new DB for attempt accounting"
                )
            metric_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(metrics)")
            }
            if metric_columns and "metric_status" not in metric_columns:
                raise ValueError("Historical metric storage is immutable; use a separate v3 DB")
            connection.executescript(SCHEMA)

    def insert_experiment(
        self,
        *,
        experiment_id: str,
        name: str,
        config_hash: str,
        dataset_hash: str,
        git_commit: str,
        python_version: str,
        created_at: str,
        config: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experiments
                (
                    id, name, config_hash, dataset_hash, git_commit, python_version,
                    created_at, config_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    name,
                    config_hash,
                    dataset_hash,
                    git_commit,
                    python_version,
                    created_at,
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                ),
            )

    def insert_run(
        self,
        *,
        run_id: str,
        experiment_id: str,
        strategy: str,
        status: str,
        start_ts: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs
                (id, experiment_id, strategy, status, start_ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, experiment_id, strategy, status, start_ts),
            )

    def finish_run(
        self,
        *,
        run_id: str,
        status: str,
        end_ts: str,
        artifact_dir: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, end_ts = ?, artifact_dir = ?, error = ?
                WHERE id = ?
                """,
                (status, end_ts, artifact_dir, error, run_id),
            )

    def upsert_example(self, example: BenchmarkExample) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO examples
                (example_id, task_type, language, split, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    example.id,
                    str(example.task_type),
                    example.language,
                    str(example.metadata.get("split", "")),
                    example.model_dump_json(),
                ),
            )

    def insert_model_call(self, record: ModelCallRecord) -> None:
        """Durably start/finalize one attempt; replay of a finished row is idempotent."""
        payload = record.model_dump(mode="json", exclude={"metadata"})
        payload["metadata_json"] = json.dumps(record.metadata, ensure_ascii=False, sort_keys=True)
        columns = list(payload)
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM provider_attempts WHERE attempt_id=?", (record.attempt_id,)
            ).fetchone()
            if existing and existing["outcome"] != "STARTED":
                if any(existing[k] != v for k, v in payload.items()):
                    raise ValueError("Finished provider attempt is immutable")
                return
            connection.execute(
                f"INSERT INTO provider_attempts ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)}) "
                "ON CONFLICT(attempt_id) DO UPDATE SET "
                + ", ".join(f"{k}=excluded.{k}" for k in columns if k != "attempt_id"),
                tuple(payload[k] for k in columns),
            )

    def insert_logical_call(self, record: LogicalCallRecord) -> None:
        payload = record.model_dump(mode="json")
        columns = list(payload)
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO logical_model_calls ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)}) "
                "ON CONFLICT(logical_call_id) DO UPDATE SET "
                "end_timestamp=excluded.end_timestamp, "
                "lifecycle_latency_ms=excluded.lifecycle_latency_ms, outcome=excluded.outcome",
                tuple(payload[k] for k in columns),
            )

    def insert_prediction(self, result: PredictionResult) -> None:
        with self.connect() as connection:
            self._insert_prediction(connection, result)

    def _insert_prediction(self, connection: sqlite3.Connection, result: PredictionResult) -> None:
        connection.execute(
            """
                INSERT INTO predictions
                (run_id, example_id, strategy, output_json, confidence, usage_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
            (
                result.metadata["run_id"],
                result.example_id,
                result.strategy,
                json.dumps(
                    result.output.model_dump(mode="json")
                    if isinstance(result, FailedPrediction)
                    else result.output,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                result.confidence,
                json.dumps(
                    {
                        "model_calls": result.model_calls,
                        "provider_attempts": result.metadata.get(
                            "provider_attempts", result.model_calls
                        ),
                        "logical_model_calls": result.metadata.get(
                            "logical_model_calls", result.model_calls
                        ),
                        **{
                            key: result.metadata[key]
                            for key in (
                                "successful_attempts",
                                "failed_attempts",
                                "unknown_usage_attempts",
                                "known_token_lower_bound",
                                "total_provider_tokens",
                                "total_provider_tokens_exact",
                                "logical_call_latency_ms",
                            )
                            if key in result.metadata
                        },
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "total_tokens": result.total_tokens,
                        "latency_ms": result.latency_ms,
                        "estimated_cost_usd": result.estimated_cost_usd,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        **result.metadata,
                        "candidates": [c.model_dump(mode="json") for c in result.candidates],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )

    def checkpoint_example(
        self,
        result: PredictionResult,
        scores: dict[str, float | None],
        failures: list[dict[str, Any]],
    ) -> None:
        """Prediction, metrics and annotations commit together; attempts are already durable."""
        run_id = result.metadata["run_id"]
        with self.connect() as connection:
            self._insert_prediction(connection, result)
            connection.executemany(
                "INSERT INTO metrics (run_id,example_id,metric_name,metric_value,metric_status) "
                "VALUES (?,?,?,?,?)",
                [
                    (
                        run_id,
                        result.example_id,
                        name,
                        value,
                        UNSCORABLE if value is None else "SCORED",
                    )
                    for name, value in scores.items()
                ],
            )
            connection.executemany(
                "INSERT INTO failure_annotations "
                "(run_id,example_id,failure_type,source,rationale,metadata_json) "
                "VALUES (?,?,?,?,?,?)",
                [
                    (
                        run_id,
                        result.example_id,
                        r["failure_type"],
                        r["source"],
                        r["rationale"],
                        json.dumps(r["metadata"], ensure_ascii=False, sort_keys=True),
                    )
                    for r in failures
                ],
            )

    def evaluate_pending_failure(
        self, result: FailedPrediction, scores: dict[str, float | None]
    ) -> None:
        """Evaluate a preserved failure only during explicitly authorized execution."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT output_json,metadata_json FROM predictions WHERE run_id=? AND example_id=?",
                (result.metadata["run_id"], result.example_id),
            ).fetchone()
            if (
                row is None
                or json.loads(row["metadata_json"]).get("evaluation_status")
                != ("PENDING_AUTHORIZED_EXECUTION")
                or json.loads(row["output_json"]) != result.output.model_dump(mode="json")
            ):
                raise ValueError("Preserved failed prediction changed or already evaluated")
            connection.executemany(
                "INSERT INTO metrics (run_id,example_id,metric_name,metric_value,metric_status) "
                "VALUES (?,?,?,?,?)",
                [
                    (
                        result.metadata["run_id"],
                        result.example_id,
                        name,
                        value,
                        UNSCORABLE if value is None else "SCORED",
                    )
                    for name, value in scores.items()
                ],
            )
            result.metadata["evaluation_status"] = "EVALUATED"
            payload = {**result.metadata, "candidates": []}
            connection.execute(
                "UPDATE predictions SET metadata_json=? WHERE run_id=? AND example_id=?",
                (
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    result.metadata["run_id"],
                    result.example_id,
                ),
            )

    def has_unknown_usage_for_config(self, config_hash: str) -> bool:
        return (
            self._fetch_one(
                "SELECT 1 FROM provider_attempts a JOIN runs r ON a.run_id=r.id "
                "JOIN experiments e ON r.experiment_id=e.id "
                "WHERE e.config_hash=? AND a.usage_status='UNKNOWN_NOT_RETURNED' LIMIT 1",
                (config_hash,),
            )
            is not None
        )

    def insert_metric(
        self, *, run_id: str, metric_name: str, metric_value: float, example_id: str | None = None
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO metrics (run_id, example_id, metric_name, metric_value)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, example_id, metric_name, metric_value),
            )

    def insert_artifact(
        self, *, experiment_id: str, run_id: str, kind: str, path: str | Path
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (experiment_id, run_id, kind, path)
                VALUES (?, ?, ?, ?)
                """,
                (experiment_id, run_id, kind, str(path)),
            )

    def insert_failure_annotation(
        self,
        *,
        run_id: str,
        example_id: str,
        failure_type: str,
        source: str,
        rationale: str,
        metadata: dict[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO failure_annotations
                (run_id, example_id, failure_type, source, rationale, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    example_id,
                    failure_type,
                    source,
                    rationale,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                ),
            )

    def list_experiments(self) -> list[dict[str, Any]]:
        return self._fetch_all("SELECT * FROM experiments ORDER BY created_at DESC")

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM experiments WHERE id = ?", (experiment_id,))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def get_run_experiment(self, run_id: str) -> dict[str, Any] | None:
        return self._fetch_one(
            """
            SELECT experiments.*
            FROM experiments
            JOIN runs ON runs.experiment_id = experiments.id
            WHERE runs.id = ?
            """,
            (run_id,),
        )

    def get_run_metrics(self, run_id: str) -> list[dict[str, Any]]:
        return self._fetch_all("SELECT * FROM metrics WHERE run_id = ? ORDER BY id", (run_id,))

    def get_run_predictions(self, run_id: str) -> list[dict[str, Any]]:
        return self._fetch_all("SELECT * FROM predictions WHERE run_id = ? ORDER BY id", (run_id,))

    def get_runs_for_experiment(self, experiment_id: str) -> list[dict[str, Any]]:
        return self._fetch_all(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY start_ts",
            (experiment_id,),
        )

    def get_run_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        return self._fetch_all("SELECT * FROM artifacts WHERE run_id = ? ORDER BY id", (run_id,))

    def get_run_failures(self, run_id: str) -> list[dict[str, Any]]:
        return self._fetch_all(
            "SELECT * FROM failure_annotations WHERE run_id = ? ORDER BY id",
            (run_id,),
        )

    def get_run_model_calls(self, run_id: str) -> list[dict[str, Any]]:
        return self._fetch_all(
            "SELECT * FROM model_calls WHERE run_id = ? ORDER BY id",
            (run_id,),
        )

    def get_run_logical_calls(self, run_id: str) -> list[dict[str, Any]]:
        return self._fetch_all(
            "SELECT * FROM logical_model_calls WHERE run_id=? ORDER BY start_timestamp",
            (run_id,),
        )

    def find_completed_run_by_config_hash(self, config_hash: str) -> dict[str, Any] | None:
        return self._fetch_one(
            """
            SELECT runs.*, experiments.config_hash
            FROM runs
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.config_hash = ? AND runs.status = 'COMPLETED'
            ORDER BY runs.end_ts DESC
            LIMIT 1
            """,
            (config_hash,),
        )

    def find_resumable_run_by_config_hash(self, config_hash: str) -> dict[str, Any] | None:
        return self._fetch_one(
            """
            SELECT runs.*, experiments.config_hash
            FROM runs
            JOIN experiments ON experiments.id = runs.experiment_id
            WHERE experiments.config_hash = ? AND runs.status IN ('RUNNING', 'FAILED')
            ORDER BY runs.start_ts DESC
            LIMIT 1
            """,
            (config_hash,),
        )

    def example_metric_values(self, run_id: str, metric_name: str) -> dict[str, float]:
        rows = self._fetch_all(
            """
            SELECT example_id, metric_value
            FROM metrics
            WHERE run_id = ? AND metric_name = ? AND example_id IS NOT NULL
            AND metric_value IS NOT NULL
            ORDER BY example_id
            """,
            (run_id, metric_name),
        )
        return {str(row["example_id"]): float(row["metric_value"]) for row in rows}

    def run_task_set(self, run_id: str) -> dict[str, Any]:
        rows = self._fetch_all(
            """
            SELECT examples.example_id, examples.task_type, examples.split
            FROM predictions
            JOIN examples ON examples.example_id = predictions.example_id
            WHERE predictions.run_id = ?
            ORDER BY examples.example_id
            """,
            (run_id,),
        )
        return {
            "example_ids": [row["example_id"] for row in rows],
            "task_types": sorted({row["task_type"] for row in rows}),
            "splits": sorted({row["split"] for row in rows}),
        }

    def compare_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        if not run_ids:
            return []
        placeholders = ",".join("?" for _ in run_ids)
        query = f"""
            SELECT run_id, metric_name, metric_value
            FROM metrics
            WHERE example_id IS NULL AND run_id IN ({placeholders})
            ORDER BY run_id, metric_name
        """
        return self._fetch_all(query, tuple(run_ids))

    def _fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(query, params).fetchone()
        return dict(row) if row is not None else None

    def _fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]
