"""
tests/test_api.py
-----------------
Tests for the FastAPI HTTP bridge (api/app.py).

All pipeline execution is mocked — no real watsonx calls are made and no
real git/pytest subprocesses are started.  The test client from FastAPI's
testing utilities (httpx-based) is used throughout.

Test classes
------------
TestRunCreation          — POST /api/run basics
TestInvalidScenario      — 422 on bad/missing scenario names
TestRunStatus            — GET /api/run/{run_id}
TestSSEEvents            — SSE event delivery and content
TestMockedPipelineSuccess — full pipeline success through the API
TestPipelineFailure      — pipeline failure propagated via SSE
TestNoCredentialLeakage  — secrets never appear in API responses
TestConcurrentRunIsolation — parallel runs get independent queues/state
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.models import (
    DiagnosisResult,
    GeneratedTest,
    PatchResult,
    PipelineStage,
    PipelineState,
    TestResult,
    TimelineEvent,
)

# ---------------------------------------------------------------------------
# Import app under test — must happen AFTER the import patch in conftest so
# watsonx imports don't blow up during collection.
# ---------------------------------------------------------------------------

from api.app import app, _runs

# ---------------------------------------------------------------------------
# Shared fixture JSON (mirrors none_bug scenario)
# ---------------------------------------------------------------------------

_DIAGNOSIS_JSON = json.dumps(
    {
        "root_cause": "divide() has no None-guard",
        "affected_file": "src/calculator.py",
        "affected_lines": [2],
        "explanation": "Needs a None-check",
        "confidence": 0.9,
    }
)

_GENERATED_TEST_JSON = json.dumps(
    {
        "filename": "tests/test_generated.py",
        "code": "def test_bug(): raise AssertionError('bug present')",
    }
)

_PATCH_JSON = json.dumps(
    {
        "diff": (
            "diff --git a/src/calculator.py b/src/calculator.py\n"
            "index abc..def 100644\n"
            "--- a/src/calculator.py\n"
            "+++ b/src/calculator.py\n"
            "@@ -1,2 +1,4 @@\n"
            "+def divide(a, b):\n"
            "+    if a is None: raise ValueError()\n"
            "     return a / b\n"
        )
    }
)

_FAILING = TestResult(stdout="FAILED", stderr="", exit_code=1)
_PASSING = TestResult(stdout="1 passed", stderr="", exit_code=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_mock(
    *,
    stage: PipelineStage = PipelineStage.COMPLETE,
    error: str | None = None,
) -> MagicMock:
    """
    Return a mock for ``run_pipeline`` that immediately calls state_callback
    with a synthetic final state and returns it.
    """

    def _fake_run_pipeline(repo_path: str, state_callback=None):
        state = PipelineState(
            repo_path=repo_path,
            stage=stage,
            error=error,
            source_files={"src/calculator.py": "code"},
            initial_test_result=_FAILING,
            diagnosis=DiagnosisResult(
                root_cause="bug",
                affected_file="src/calculator.py",
                affected_lines=[1],
                explanation="ex",
                confidence=0.9,
            ),
            generated_test=GeneratedTest(
                filename="tests/test_gen.py",
                code="def test_bug(): pass",
                confirmed_failing=True,
            ),
            patch=PatchResult(diff="diff --git a/x b/x\n@@\n", validated=True),
            final_test_result=_PASSING if stage == PipelineStage.COMPLETE else TestResult(),
            timeline=[TimelineEvent(stage="Complete", status="done")],
        )
        if state_callback:
            state_callback(state)
        return state

    mock = MagicMock(side_effect=_fake_run_pipeline)
    return mock


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_runs():
    """Ensure the global run store is empty before every test."""
    _runs.clear()
    yield
    _runs.clear()


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def success_mock():
    """Patch run_pipeline with the successful fake."""
    with patch("api.app.run_pipeline", _make_pipeline_mock()):
        yield


# ===========================================================================
# 1. Run creation
# ===========================================================================


class TestRunCreation:
    def test_post_returns_200(self, client, success_mock):
        resp = client.post("/api/run", json={"scenario": "none_bug"})
        assert resp.status_code == 200

    def test_post_returns_run_id(self, client, success_mock):
        resp = client.post("/api/run", json={"scenario": "none_bug"})
        data = resp.json()
        assert "run_id" in data
        assert len(data["run_id"]) == 36  # UUID4 format

    def test_post_returns_started_status(self, client, success_mock):
        resp = client.post("/api/run", json={"scenario": "none_bug"})
        assert resp.json()["status"] == "started"

    def test_all_valid_scenarios_accepted(self, client, success_mock):
        for scenario in ("none_bug", "broken_api", "off_by_one"):
            resp = client.post("/api/run", json={"scenario": scenario})
            assert resp.status_code == 200, f"scenario {scenario!r} rejected"

    def test_run_id_unique_per_call(self, client, success_mock):
        ids = {
            client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            for _ in range(5)
        }
        assert len(ids) == 5


# ===========================================================================
# 2. Invalid scenario
# ===========================================================================


class TestInvalidScenario:
    def test_unknown_scenario_returns_422(self, client):
        resp = client.post("/api/run", json={"scenario": "evil_scenario"})
        assert resp.status_code == 422

    def test_empty_scenario_returns_422(self, client):
        resp = client.post("/api/run", json={"scenario": ""})
        assert resp.status_code == 422

    def test_path_traversal_rejected(self, client):
        resp = client.post("/api/run", json={"scenario": "../../etc/passwd"})
        assert resp.status_code == 422

    def test_missing_body_returns_422(self, client):
        resp = client.post("/api/run", json={})
        assert resp.status_code == 422

    def test_error_detail_mentions_scenario(self, client):
        resp = client.post("/api/run", json={"scenario": "unknown_x"})
        assert "unknown_x" in resp.json()["detail"]


# ===========================================================================
# 3. Run status (GET /api/run/{run_id})
# ===========================================================================


class TestRunStatus:
    def test_unknown_run_id_returns_404(self, client):
        resp = client.get("/api/run/does-not-exist")
        assert resp.status_code == 404

    def test_returns_run_id_in_payload(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        # Allow background thread to finish.
        time.sleep(0.3)
        resp = client.get(f"/api/run/{run_id}")
        assert resp.json()["run_id"] == run_id

    def test_returns_stage_after_completion(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        time.sleep(0.3)
        resp = client.get(f"/api/run/{run_id}")
        assert resp.json()["stage"] == PipelineStage.COMPLETE.value

    def test_source_files_serialised_as_list_of_keys(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        time.sleep(0.3)
        resp = client.get(f"/api/run/{run_id}")
        src = resp.json().get("source_files")
        assert isinstance(src, list), "source_files should be a list of keys"


# ===========================================================================
# 4. SSE event delivery
# ===========================================================================


class TestSSEEvents:
    def _collect_events(self, client, run_id: str) -> list[dict]:
        """Read the full SSE stream synchronously and return parsed events."""
        events = []
        with client.stream("GET", f"/api/run/{run_id}/events") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    try:
                        events.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
        return events

    def test_events_endpoint_returns_200(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        with client.stream("GET", f"/api/run/{run_id}/events") as resp:
            assert resp.status_code == 200

    def test_content_type_is_event_stream(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        with client.stream("GET", f"/api/run/{run_id}/events") as resp:
            assert "text/event-stream" in resp.headers["content-type"]

    def test_unknown_run_id_returns_404_for_events(self, client):
        resp = client.get("/api/run/no-such-id/events")
        assert resp.status_code == 404

    def test_stream_ends_with_stream_end_event(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        events = self._collect_events(client, run_id)
        types = [e.get("type") for e in events]
        assert "stream_end" in types

    def test_at_least_one_pipeline_state_event(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        events = self._collect_events(client, run_id)
        types = [e.get("type") for e in events]
        assert any(t in ("pipeline_state", "complete", "failed") for t in types)

    def test_complete_event_appears_in_stream(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        events = self._collect_events(client, run_id)
        types = [e.get("type") for e in events]
        assert "complete" in types

    def test_complete_event_data_contains_stage(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        events = self._collect_events(client, run_id)
        complete_events = [e for e in events if e.get("type") == "complete"]
        assert complete_events
        assert complete_events[0]["data"]["stage"] == PipelineStage.COMPLETE.value


# ===========================================================================
# 5. Mocked pipeline success
# ===========================================================================


class TestMockedPipelineSuccess:
    def test_run_reaches_complete_stage(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        time.sleep(0.3)
        state = client.get(f"/api/run/{run_id}").json()
        assert state["stage"] == PipelineStage.COMPLETE.value

    def test_diagnosis_populated_in_state(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        time.sleep(0.3)
        state = client.get(f"/api/run/{run_id}").json()
        assert state["diagnosis"]["root_cause"] == "bug"

    def test_generated_test_populated_in_state(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        time.sleep(0.3)
        state = client.get(f"/api/run/{run_id}").json()
        assert state["generated_test"]["filename"] == "tests/test_gen.py"

    def test_patch_populated_in_state(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        time.sleep(0.3)
        state = client.get(f"/api/run/{run_id}").json()
        assert state["patch"]["validated"] is True

    def test_timeline_present_in_state(self, client, success_mock):
        run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        time.sleep(0.3)
        state = client.get(f"/api/run/{run_id}").json()
        assert isinstance(state["timeline"], list)
        assert len(state["timeline"]) > 0


# ===========================================================================
# 6. Pipeline failure
# ===========================================================================


class TestPipelineFailure:
    def test_failed_stage_reflected_in_status(self, client):
        with patch(
            "api.app.run_pipeline",
            _make_pipeline_mock(stage=PipelineStage.FAILED, error="something broke"),
        ):
            run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            time.sleep(0.3)
            state = client.get(f"/api/run/{run_id}").json()
            assert state["stage"] == PipelineStage.FAILED.value

    def test_error_field_populated_on_failure(self, client):
        with patch(
            "api.app.run_pipeline",
            _make_pipeline_mock(stage=PipelineStage.FAILED, error="something broke"),
        ):
            run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            time.sleep(0.3)
            state = client.get(f"/api/run/{run_id}").json()
            assert state["error"] == "something broke"

    def test_failed_event_in_sse_stream(self, client):
        with patch(
            "api.app.run_pipeline",
            _make_pipeline_mock(stage=PipelineStage.FAILED, error="something broke"),
        ):
            run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            events: list[dict] = []
            with client.stream("GET", f"/api/run/{run_id}/events") as resp:
                for line in resp.iter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        raw = line[len("data:"):].strip()
                        try:
                            events.append(json.loads(raw))
                        except json.JSONDecodeError:
                            pass
            types = [e.get("type") for e in events]
            assert "failed" in types

    def test_exception_in_pipeline_produces_failed_event(self, client):
        def _crash(repo_path, state_callback=None):
            raise RuntimeError("unexpected crash")

        with patch("api.app.run_pipeline", side_effect=_crash):
            run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            events: list[dict] = []
            with client.stream("GET", f"/api/run/{run_id}/events") as resp:
                for line in resp.iter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        raw = line[len("data:"):].strip()
                        try:
                            events.append(json.loads(raw))
                        except json.JSONDecodeError:
                            pass
            types = [e.get("type") for e in events]
            assert "failed" in types


# ===========================================================================
# 7. No credential leakage
# ===========================================================================


class TestNoCredentialLeakage:
    """
    Ensure WATSONX_API_KEY and WATSONX_PROJECT_ID never appear in API
    responses, even if accidentally embedded in state or error messages.
    """

    _FAKE_KEY = "sk-super-secret-api-key-12345"
    _FAKE_PROJECT = "my-secret-project-id-99"

    def _all_text(self, resp) -> str:
        try:
            return json.dumps(resp.json())
        except Exception:
            return resp.text

    def test_api_key_not_in_run_response(self, client, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", self._FAKE_KEY)
        with patch("api.app.run_pipeline", _make_pipeline_mock()):
            resp = client.post("/api/run", json={"scenario": "none_bug"})
            assert self._FAKE_KEY not in self._all_text(resp)

    def test_project_id_not_in_run_response(self, client, monkeypatch):
        monkeypatch.setenv("WATSONX_PROJECT_ID", self._FAKE_PROJECT)
        with patch("api.app.run_pipeline", _make_pipeline_mock()):
            resp = client.post("/api/run", json={"scenario": "none_bug"})
            assert self._FAKE_PROJECT not in self._all_text(resp)

    def test_api_key_not_in_status_response(self, client, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", self._FAKE_KEY)
        with patch("api.app.run_pipeline", _make_pipeline_mock()):
            run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            time.sleep(0.3)
            resp = client.get(f"/api/run/{run_id}")
            assert self._FAKE_KEY not in self._all_text(resp)

    def test_project_id_not_in_status_response(self, client, monkeypatch):
        monkeypatch.setenv("WATSONX_PROJECT_ID", self._FAKE_PROJECT)
        with patch("api.app.run_pipeline", _make_pipeline_mock()):
            run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            time.sleep(0.3)
            resp = client.get(f"/api/run/{run_id}")
            assert self._FAKE_PROJECT not in self._all_text(resp)

    def test_api_key_not_in_sse_stream(self, client, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", self._FAKE_KEY)
        with patch("api.app.run_pipeline", _make_pipeline_mock()):
            run_id = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            full_text = ""
            with client.stream("GET", f"/api/run/{run_id}/events") as resp:
                for line in resp.iter_lines():
                    full_text += line
            assert self._FAKE_KEY not in full_text


# ===========================================================================
# 8. Concurrent run isolation
# ===========================================================================


class TestConcurrentRunIsolation:
    def test_two_runs_have_different_run_ids(self, client, success_mock):
        r1 = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
        r2 = client.post("/api/run", json={"scenario": "broken_api"}).json()["run_id"]
        assert r1 != r2

    def test_run_ids_are_independent_in_store(self, client):
        """
        Each run_id maps to its own _RunContext; modifying one does not affect
        the other.
        """
        barriers: dict[str, threading.Barrier] = {}
        results: dict[str, PipelineStage] = {}
        lock = threading.Lock()

        def _staged_pipeline(scenario_stage):
            def _fake(repo_path, state_callback=None):
                state = PipelineState(
                    repo_path=repo_path,
                    stage=scenario_stage,
                    source_files={},
                )
                if state_callback:
                    state_callback(state)
                return state
            return _fake

        with (
            patch("api.app.run_pipeline", _staged_pipeline(PipelineStage.COMPLETE)),
        ):
            r1 = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            r2 = client.post("/api/run", json={"scenario": "broken_api"}).json()["run_id"]
            assert r1 in _runs
            assert r2 in _runs
            assert _runs[r1] is not _runs[r2]

    def test_concurrent_runs_get_independent_states(self, client):
        """Two runs completing at different times have separate latest_state."""
        run_ids = []
        with patch("api.app.run_pipeline", _make_pipeline_mock()):
            for scenario in ("none_bug", "off_by_one"):
                run_ids.append(
                    client.post("/api/run", json={"scenario": scenario}).json()["run_id"]
                )

        time.sleep(0.3)

        states = [client.get(f"/api/run/{rid}").json() for rid in run_ids]
        # Both reach COMPLETE independently.
        for s in states:
            assert s["stage"] == PipelineStage.COMPLETE.value

        # repo_path differs between the two runs.
        assert states[0].get("repo_path") != states[1].get("repo_path")

    def test_event_queues_are_independent(self, client):
        """Events pushed to run A's queue must not appear in run B's stream."""
        with patch("api.app.run_pipeline", _make_pipeline_mock()):
            r1 = client.post("/api/run", json={"scenario": "none_bug"}).json()["run_id"]
            r2 = client.post("/api/run", json={"scenario": "broken_api"}).json()["run_id"]

        assert _runs[r1].event_queue is not _runs[r2].event_queue
