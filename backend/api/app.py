"""
api/app.py
----------
Minimal HTTP bridge between the React/Vite Command Center (Module 3) and
the CodeHeal Python pipeline (Modules 1 & 2).

Endpoints
---------
POST  /api/run
    Start a pipeline run for a bundled sample scenario.
    Returns immediately with a run_id.

GET   /api/run/{run_id}
    Return the latest PipelineState for the given run as JSON.

GET   /api/run/{run_id}/events
    Server-Sent Events stream.  One event per pipeline callback invocation
    plus a terminal "complete" or "failed" event.

Design notes
------------
* Runs are isolated by UUID run_id; a per-run queue carries SSE payloads.
* The pipeline executes in a daemon background thread.
* Only the three bundled sample scenarios are accepted; arbitrary paths are
  rejected with 422.
* Credentials (WATSONX_API_KEY, WATSONX_PROJECT_ID) are never serialised.
* CORS is enabled for localhost Vite dev servers on ports 3000 and 5173.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import queue
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.models import PipelineState, PipelineStage
from core.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
_ALLOWED_SCENARIOS = frozenset({"none_bug", "broken_api", "off_by_one"})
_SENTINEL = object()  # signals the SSE generator that the run is finished

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="CodeHeal API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
      "https://codeheal-frontend.vercel.app",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory run store
# ---------------------------------------------------------------------------


class _RunContext:
    """All mutable state for a single pipeline run."""

    def __init__(self, run_id: str, repo_path: str) -> None:
        self.run_id = run_id
        self.repo_path = repo_path
        self.latest_state: PipelineState | None = None
        self.event_queue: queue.Queue = queue.Queue()
        self.finished = threading.Event()


# Maps run_id → _RunContext.  Only written at run-creation time (POST /api/run),
# never mutated by the SSE generator or GET handler.
_runs: dict[str, _RunContext] = {}


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _serialise(obj: Any) -> Any:
    """Recursively convert pipeline dataclasses / enums to JSON-safe types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialise(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, PipelineStage):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, list):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    return obj


def _state_to_dict(state: PipelineState) -> dict:
    """Return a credential-safe, JSON-serialisable dict for a PipelineState."""
    d = _serialise(state)
    # Drop source_files contents to keep payloads small and avoid leaking
    # accidental secrets embedded in fixture files.
    if "source_files" in d:
        d["source_files"] = list(d["source_files"].keys())
    return d


# ---------------------------------------------------------------------------
# SSE event helpers
# ---------------------------------------------------------------------------


def _sse_event(event_type: str, data: Any) -> str:
    """Format a single SSE message."""
    payload = json.dumps({"type": event_type, "data": data})
    return f"event: {event_type}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Background pipeline thread
# ---------------------------------------------------------------------------


def _run_in_thread(ctx: _RunContext) -> None:
    """
    Execute run_pipeline in a background thread.

    The state_callback converts each PipelineState snapshot into an SSE
    event and pushes it to ctx.event_queue.  A sentinel is pushed when the
    pipeline returns (success or failure).
    """

    def _callback(state: PipelineState) -> None:
        ctx.latest_state = state
        data = _state_to_dict(state)
        event_type = "pipeline_state"
        if state.stage == PipelineStage.COMPLETE:
            event_type = "complete"
        elif state.stage == PipelineStage.FAILED:
            event_type = "failed"
        ctx.event_queue.put(_sse_event(event_type, data))

    try:
        final_state = run_pipeline(ctx.repo_path, state_callback=_callback)
        ctx.latest_state = final_state
    except Exception as exc:  # noqa: BLE001
        # Surface unexpected exceptions as a failed SSE event.
        ctx.event_queue.put(
            _sse_event("failed", {"error": str(exc), "run_id": ctx.run_id})
        )
    finally:
        ctx.event_queue.put(_SENTINEL)
        ctx.finished.set()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    scenario: str


class RunResponse(BaseModel):
    run_id: str
    status: str


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@app.post("/api/run", response_model=RunResponse)
def create_run(body: RunRequest) -> RunResponse:
    """
    Start a CodeHeal pipeline run for a bundled sample scenario.

    Returns immediately; the pipeline executes in the background.
    Poll ``GET /api/run/{run_id}`` or stream ``GET /api/run/{run_id}/events``.
    """
    scenario = body.scenario
    if scenario not in _ALLOWED_SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown scenario '{scenario}'. "
                f"Allowed values: {sorted(_ALLOWED_SCENARIOS)}"
            ),
        )

    repo_path = str(_SAMPLES_DIR / scenario)
    run_id = str(uuid.uuid4())
    ctx = _RunContext(run_id=run_id, repo_path=repo_path)
    _runs[run_id] = ctx

    thread = threading.Thread(target=_run_in_thread, args=(ctx,), daemon=True)
    thread.start()

    return RunResponse(run_id=run_id, status="started")


@app.get("/api/run/{run_id}")
def get_run(run_id: str) -> dict:
    """Return the latest PipelineState for *run_id* as JSON."""
    ctx = _runs.get(run_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    if ctx.latest_state is None:
        return {"run_id": run_id, "status": "pending"}
    return {"run_id": run_id, **_state_to_dict(ctx.latest_state)}


@app.get("/api/run/{run_id}/events")
def run_events(run_id: str) -> StreamingResponse:
    """
    Stream pipeline events for *run_id* as Server-Sent Events.

    The stream terminates automatically when the pipeline reaches COMPLETE or
    FAILED (signalled by a sentinel value pushed into the run's queue).
    """
    ctx = _runs.get(run_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    def _generate() -> Generator[str, None, None]:
        while True:
            try:
                item = ctx.event_queue.get(timeout=30)
            except queue.Empty:
                # Heartbeat to keep the connection alive.
                yield ": heartbeat\n\n"
                continue
            if item is _SENTINEL:
                yield _sse_event("stream_end", {"run_id": run_id})
                break
            yield item

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )



