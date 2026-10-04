"""FastAPI application for submitting and inspecting experiment runs."""

import tempfile
from pathlib import Path
from typing import Any

import yaml

from collectiveeval.reporting import compare_runs_report, evaluate_run_report
from collectiveeval.runner import run_experiment
from collectiveeval.storage import SQLiteStore


def create_app(
    *,
    db_path: str | Path = "collectiveeval.sqlite3",
    output_dir: str | Path = "runs",
) -> Any:
    """Create the Phase 2 API app.

    FastAPI is imported lazily so the core library and tests can still run in
    keyless/minimal environments where only the research pipeline is installed.
    """

    try:
        from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
    except ModuleNotFoundError as exc:  # pragma: no cover - minimal environments only
        raise RuntimeError("FastAPI is not installed") from exc

    app = FastAPI(title="CollectiveEval", version="0.1.0")
    store = SQLiteStore(db_path)

    @app.get("/health")  # type: ignore[misc]
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/experiments")  # type: ignore[misc]
    def submit_experiment(
        payload: dict[str, Any],
        background_tasks: BackgroundTasks,
    ) -> dict[str, str]:
        config = payload.get("config", payload)
        if not isinstance(config, dict):
            raise HTTPException(status_code=400, detail="config must be an object")
        background_tasks.add_task(_run_background, config, db_path, output_dir)
        return {"status": "submitted"}

    @app.get("/experiments")  # type: ignore[misc]
    def list_experiments() -> list[dict[str, Any]]:
        return store.list_experiments()

    @app.get("/experiments/{experiment_id}")  # type: ignore[misc]
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        experiment = store.get_experiment(experiment_id)
        if experiment is None:
            raise HTTPException(status_code=404, detail="experiment not found")
        return experiment

    @app.get("/runs/{run_id}")  # type: ignore[misc]
    def get_run(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @app.get("/runs/{run_id}/metrics")  # type: ignore[misc]
    def get_metrics(run_id: str) -> dict[str, Any]:
        return evaluate_run_report(store, run_id)

    @app.get("/runs/{run_id}/predictions")  # type: ignore[misc]
    def get_predictions(run_id: str) -> list[dict[str, Any]]:
        return store.get_run_predictions(run_id)

    @app.get("/compare")  # type: ignore[misc]
    def compare(request: Request) -> dict[str, Any]:
        run_ids = list(request.query_params.getlist("run_id"))
        if len(run_ids) < 2:
            raise HTTPException(status_code=400, detail="at least two run_id query params required")
        return compare_runs_report(store, run_ids)

    return app


def _run_background(config: dict[str, Any], db_path: str | Path, output_dir: str | Path) -> None:
    # BackgroundTasks accepts sync callables; use asyncio.run inside this worker boundary.
    import asyncio

    asyncio.run(run_experiment(config, db_path=db_path, output_dir=output_dir))


def write_temp_config(config: dict[str, Any]) -> Path:
    """Utility for tests and manual API experiments."""

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=True)
        return Path(handle.name)
