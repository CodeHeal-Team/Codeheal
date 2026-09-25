"""
tests/test_pipeline.py
-----------------------
Focused tests for core/pipeline.py.

Strategy
--------
Every test patches at the boundary of the pipeline's dependencies:
  * core.pipeline.load_repo         — avoids touching the filesystem
  * core.pipeline.run_tests         — avoids spinning up pytest subprocesses
  * core.pipeline.DiagnosticAgent   — avoids needing watsonx credentials
  * core.pipeline.TestGeneratorAgent
  * core.pipeline.RefactoringAgent
  * subprocess.run                  — avoids real git invocations

This makes all tests fast, deterministic, and credential-free.

Real-agent integration (end-to-end against samples/) is covered in Sub-Task 9.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from core.models import (
    DiagnosisResult,
    GeneratedTest,
    PatchResult,
    PipelineStage,
    PipelineState,
    TestResult,
    TimelineEvent,
)
from core.pipeline import _emit, _fail, _transition, reset_repo, run_pipeline

# ---------------------------------------------------------------------------
# Shared fixture data (mirrors the none_bug scenario)
# ---------------------------------------------------------------------------

_SOURCE_FILES = {
    "src/calculator.py": "def divide(a, b):\n    return a / b\n",
    "tests/test_calculator.py": "def test_divide(): pass\n",
}

_FAILING_TEST_RESULT = TestResult(stdout="FAILED", stderr="", exit_code=1)
_PASSING_TEST_RESULT = TestResult(stdout="passed", stderr="", exit_code=0)

_DIAGNOSIS_JSON = """{
  "root_cause": "divide() has no None-guard",
  "affected_file": "src/calculator.py",
  "affected_lines": [2],
  "explanation": "Needs a None-check",
  "confidence": 0.9
}"""

_GENERATED_TEST_JSON = """{
  "filename": "tests/test_generated.py",
  "code": "def test_bug(): raise AssertionError('bug present')"
}"""

_PATCH_JSON = """{
  "diff": "diff --git a/src/calculator.py b/src/calculator.py\\nindex abc..def 100644\\n--- a/src/calculator.py\\n+++ b/src/calculator.py\\n@@ -1,2 +1,4 @@\\n+def divide(a, b):\\n+    if a is None: raise ValueError()\\n     return a / b\\n"
}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop(_state: PipelineState) -> None:
    pass


def _make_mock_diagnostic_agent(raw: str = _DIAGNOSIS_JSON) -> MagicMock:
    agent = MagicMock()
    agent.return_value.run.return_value = raw
    return agent


def _make_mock_test_gen_agent(raw: str = _GENERATED_TEST_JSON) -> MagicMock:
    agent = MagicMock()
    agent.return_value.run.return_value = raw
    return agent


def _make_mock_refactor_agent(raw: str = _PATCH_JSON) -> MagicMock:
    agent = MagicMock()
    agent.return_value.run.return_value = raw
    return agent


def _mock_subprocess_run_success() -> MagicMock:
    """subprocess.run mock that simulates successful git apply."""
    m = MagicMock()
    m.return_value.returncode = 0
    m.return_value.stderr = ""
    m.return_value.stdout = ""
    return m


def _mock_subprocess_run_failure() -> MagicMock:
    """subprocess.run mock that simulates failed git apply."""
    m = MagicMock()
    m.return_value.returncode = 1
    m.return_value.stderr = "patch does not apply"
    m.return_value.stdout = ""
    return m


def _full_patch(
    *,
    tmp_path=None,
    load_repo_files: dict = None,
    initial_run_result: TestResult = None,
    final_run_result: TestResult = None,
    confirm_run_result: TestResult = None,
    diag_raw: str = _DIAGNOSIS_JSON,
    test_raw: str = _GENERATED_TEST_JSON,
    patch_raw: str = _PATCH_JSON,
    git_apply_rc: int = 0,
):
    """
    Context manager that patches all external dependencies for a full
    successful pipeline run.  Override individual defaults as needed.

    When *tmp_path* is provided, ``src/calculator.py`` is created there so
    that ``response_parser.parse_patch`` can validate the diff's target file.
    """
    from unittest.mock import patch as _patch, MagicMock

    load_files = load_repo_files if load_repo_files is not None else _SOURCE_FILES
    init_result = initial_run_result if initial_run_result is not None else _FAILING_TEST_RESULT
    fin_result = final_run_result if final_run_result is not None else _PASSING_TEST_RESULT
    confirm_result = confirm_run_result if confirm_run_result is not None else _FAILING_TEST_RESULT

    # run_tests is called three times: initial, confirm, final
    run_tests_mock = MagicMock(side_effect=[init_result, confirm_result, fin_result])

    diag_agent = MagicMock()
    diag_agent.return_value.run.return_value = diag_raw

    test_agent = MagicMock()
    test_agent.return_value.run.return_value = test_raw

    refactor_agent = MagicMock()
    refactor_agent.return_value.run.return_value = patch_raw

    git_mock = MagicMock()
    git_mock.return_value.returncode = git_apply_rc
    git_mock.return_value.stderr = "" if git_apply_rc == 0 else "patch failed"
    git_mock.return_value.stdout = ""

    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        # Create the target file so parse_patch's file-existence check passes.
        if tmp_path is not None:
            src_dir = tmp_path / "src"
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / "calculator.py").write_text("def divide(a, b): return a/b\n")

        with (
            _patch("core.pipeline.load_repo", return_value=load_files),
            _patch("core.pipeline.run_tests", run_tests_mock),
            _patch("core.pipeline.DiagnosticAgent", diag_agent),
            _patch("core.pipeline.TestGeneratorAgent", test_agent),
            _patch("core.pipeline.RefactoringAgent", refactor_agent),
            _patch("core.pipeline.subprocess.run", git_mock),
        ):
            yield {
                "load_repo": load_files,
                "run_tests": run_tests_mock,
                "DiagnosticAgent": diag_agent,
                "TestGeneratorAgent": test_agent,
                "RefactoringAgent": refactor_agent,
                "subprocess_run": git_mock,
            }

    return _ctx()


# ===========================================================================
# Module-level helper tests
# ===========================================================================

class TestEmitHelper:
    def test_emit_appends_timeline_event(self):
        state = PipelineState()
        _emit(state, "Load", "in_progress")
        assert len(state.timeline) == 1

    def test_emit_correct_label(self):
        state = PipelineState()
        _emit(state, "Diagnose", "done")
        assert state.timeline[0].stage == "Diagnose"

    def test_emit_correct_status(self):
        state = PipelineState()
        _emit(state, "Verify", "failed")
        assert state.timeline[0].status == "failed"

    def test_emit_multiple_accumulates(self):
        state = PipelineState()
        _emit(state, "A", "in_progress")
        _emit(state, "A", "done")
        assert len(state.timeline) == 2


class TestTransitionHelper:
    def test_transition_sets_stage(self):
        state = PipelineState()
        _transition(state, PipelineStage.LOADING, _noop, "Load", "in_progress")
        assert state.stage == PipelineStage.LOADING

    def test_transition_appends_timeline(self):
        state = PipelineState()
        _transition(state, PipelineStage.LOADING, _noop, "Load", "in_progress")
        assert len(state.timeline) == 1

    def test_transition_fires_callback(self):
        state = PipelineState()
        received = []
        _transition(state, PipelineStage.LOADING, received.append, "Load", "in_progress")
        assert len(received) == 1
        assert received[0].stage == PipelineStage.LOADING


class TestFailHelper:
    def test_fail_sets_failed_stage(self):
        state = PipelineState()
        result = _fail(state, _noop, "Load", "boom")
        assert result.stage == PipelineStage.FAILED

    def test_fail_sets_error_message(self):
        state = PipelineState()
        _fail(state, _noop, "Load", "something went wrong")
        assert state.error == "something went wrong"

    def test_fail_appends_failed_timeline_event(self):
        state = PipelineState()
        _fail(state, _noop, "Load", "boom")
        assert state.timeline[-1].status == "failed"

    def test_fail_fires_callback(self):
        state = PipelineState()
        received = []
        _fail(state, received.append, "Load", "boom")
        assert len(received) == 1


# ===========================================================================
# reset_repo tests
# ===========================================================================

class TestResetRepo:
    def test_calls_git_checkout(self, tmp_path):
        with patch("core.pipeline.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            reset_repo(str(tmp_path))
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args == ["git", "checkout", "."]

    def test_uses_correct_cwd(self, tmp_path):
        with patch("core.pipeline.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            reset_repo(str(tmp_path))
            kwargs = mock_run.call_args[1]
            assert kwargs["cwd"] == str(tmp_path.resolve())

    def test_does_not_raise_on_git_failure(self, tmp_path, capsys):
        with patch(
            "core.pipeline.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "git", stderr=b"err"),
        ):
            # Should not propagate the exception.
            reset_repo(str(tmp_path))
        captured = capsys.readouterr()
        assert "reset_repo" in captured.err

    def test_does_not_raise_when_git_missing(self, tmp_path, capsys):
        with patch("core.pipeline.subprocess.run", side_effect=FileNotFoundError):
            reset_repo(str(tmp_path))
        captured = capsys.readouterr()
        assert "reset_repo" in captured.err


# ===========================================================================
# run_pipeline — state initialisation
# ===========================================================================

class TestRunPipelineInit:
    def test_returns_pipeline_state(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert isinstance(state, PipelineState)

    def test_repo_path_stored_in_state(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert state.repo_path == str(tmp_path.resolve())

    def test_callback_receives_state_objects(self, tmp_path):
        received = []
        with _full_patch(tmp_path=tmp_path) as _:
            run_pipeline(str(tmp_path), state_callback=received.append)
        assert all(isinstance(s, PipelineState) for s in received)

    def test_no_callback_does_not_raise(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path), state_callback=None)
        assert state.stage == PipelineStage.COMPLETE


# ===========================================================================
# run_pipeline — successful full run
# ===========================================================================

class TestRunPipelineSuccess:
    def test_final_stage_is_complete(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.COMPLETE

    def test_source_files_populated(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert state.source_files == _SOURCE_FILES

    def test_initial_test_result_stored(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert state.initial_test_result.exit_code == 1

    def test_diagnosis_populated(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert isinstance(state.diagnosis, DiagnosisResult)
        assert state.diagnosis.root_cause != ""

    def test_generated_test_populated(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert isinstance(state.generated_test, GeneratedTest)
        assert state.generated_test.filename == "tests/test_generated.py"

    def test_patch_populated(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert isinstance(state.patch, PatchResult)
        assert state.patch.validated is True

    def test_final_test_result_stored(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert state.final_test_result.exit_code == 0

    def test_error_is_none_on_success(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert state.error is None

    def test_confirmed_failing_set_by_confirm_run(self, tmp_path):
        with _full_patch(tmp_path=tmp_path, confirm_run_result=_FAILING_TEST_RESULT) as _:
            state = run_pipeline(str(tmp_path))
        assert state.generated_test.confirmed_failing is True

    def test_confirmed_failing_false_when_test_passes(self, tmp_path):
        with _full_patch(tmp_path=tmp_path, confirm_run_result=_PASSING_TEST_RESULT) as _:
            state = run_pipeline(str(tmp_path))
        # Test passed during confirm step — confirmed_failing should be False
        assert state.generated_test.confirmed_failing is False


# ===========================================================================
# run_pipeline — timeline events
# ===========================================================================

class TestRunPipelineTimeline:
    def test_timeline_is_not_empty(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        assert len(state.timeline) > 0

    def test_timeline_contains_load_done(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        labels = [e.stage for e in state.timeline]
        assert "Load" in labels

    def test_timeline_contains_detect_done(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        statuses = {e.stage: e.status for e in state.timeline}
        assert "Detect" in statuses

    def test_timeline_contains_diagnose_done(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        statuses = {e.stage: e.status for e in state.timeline}
        assert "Diagnose" in statuses

    def test_timeline_contains_generate_test_done(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        stages = [e.stage for e in state.timeline]
        assert "Generate Test" in stages

    def test_timeline_contains_fix_done(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        stages = [e.stage for e in state.timeline]
        assert "Fix" in stages

    def test_timeline_contains_verify_done(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        stages = [e.stage for e in state.timeline]
        assert "Verify" in stages

    def test_timeline_contains_complete_done(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        statuses = [(e.stage, e.status) for e in state.timeline]
        assert ("Complete", "done") in statuses

    def test_timeline_events_have_timestamps(self, tmp_path):
        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))
        for event in state.timeline:
            assert event.timestamp is not None

    def test_callback_called_for_each_stage(self, tmp_path):
        calls = []
        with _full_patch(tmp_path=tmp_path) as _:
            run_pipeline(str(tmp_path), state_callback=calls.append)
        # At minimum: LOADING, after-load, RUNNING_TESTS, after-detect,
        # DIAGNOSING, after-diagnose, GENERATING_TEST, after-gen,
        # APPLYING_FIX, after-fix, VERIFYING, after-verify, COMPLETE
        assert len(calls) >= 7


# ===========================================================================
# run_pipeline — LOADING failure
# ===========================================================================

class TestRunPipelineLoadingFailure:
    def test_stage_is_failed_on_load_error(self, tmp_path):
        with (
            patch("core.pipeline.load_repo", side_effect=FileNotFoundError("nope")),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED

    def test_error_message_contains_stage_name(self, tmp_path):
        with patch("core.pipeline.load_repo", side_effect=RuntimeError("disk error")):
            state = run_pipeline(str(tmp_path))
        assert "LOADING" in state.error

    def test_timeline_has_failed_status(self, tmp_path):
        with patch("core.pipeline.load_repo", side_effect=RuntimeError("x")):
            state = run_pipeline(str(tmp_path))
        assert any(e.status == "failed" for e in state.timeline)


# ===========================================================================
# run_pipeline — RUNNING_TESTS failure (all pass — early stop)
# ===========================================================================

class TestRunPipelineAllTestsPass:
    def test_stage_is_failed_when_no_failing_tests(self, tmp_path):
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", return_value=_PASSING_TEST_RESULT),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED

    def test_error_mentions_clean_repo(self, tmp_path):
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", return_value=_PASSING_TEST_RESULT),
        ):
            state = run_pipeline(str(tmp_path))
        assert "No failing tests found" in state.error

    def test_source_files_still_populated(self, tmp_path):
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", return_value=_PASSING_TEST_RESULT),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.source_files == _SOURCE_FILES

    def test_run_tests_exception_sets_failed(self, tmp_path):
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", side_effect=RuntimeError("pytest crash")),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED
        assert "RUNNING_TESTS" in state.error


# ===========================================================================
# run_pipeline — DIAGNOSING failure
# ===========================================================================

class TestRunPipelineDiagnosingFailure:
    def test_stage_failed_on_agent_error(self, tmp_path):
        diag_mock = MagicMock(side_effect=RuntimeError("model down"))
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", return_value=_FAILING_TEST_RESULT),
            patch("core.pipeline.DiagnosticAgent", diag_mock),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED

    def test_error_mentions_stage(self, tmp_path):
        diag_mock = MagicMock(side_effect=RuntimeError("bad"))
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", return_value=_FAILING_TEST_RESULT),
            patch("core.pipeline.DiagnosticAgent", diag_mock),
        ):
            state = run_pipeline(str(tmp_path))
        assert "DIAGNOSING" in state.error

    def test_stage_failed_on_parse_error(self, tmp_path):
        diag_agent = MagicMock()
        diag_agent.return_value.run.return_value = "not valid json"
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", return_value=_FAILING_TEST_RESULT),
            patch("core.pipeline.DiagnosticAgent", diag_agent),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED


# ===========================================================================
# run_pipeline — GENERATING_TEST failure
# ===========================================================================

class TestRunPipelineGeneratingTestFailure:
    def _base_patches(self):
        diag_agent = MagicMock()
        diag_agent.return_value.run.return_value = _DIAGNOSIS_JSON
        return diag_agent

    def test_stage_failed_on_agent_error(self, tmp_path):
        test_agent = MagicMock(side_effect=RuntimeError("fail"))
        diag_agent = self._base_patches()
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", return_value=_FAILING_TEST_RESULT),
            patch("core.pipeline.DiagnosticAgent", diag_agent),
            patch("core.pipeline.TestGeneratorAgent", test_agent),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED
        assert "GENERATING_TEST" in state.error

    def test_stage_failed_on_bad_json(self, tmp_path):
        test_agent = MagicMock()
        test_agent.return_value.run.return_value = "garbage"
        diag_agent = self._base_patches()
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", return_value=_FAILING_TEST_RESULT),
            patch("core.pipeline.DiagnosticAgent", diag_agent),
            patch("core.pipeline.TestGeneratorAgent", test_agent),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED


# ===========================================================================
# run_pipeline — APPLYING_FIX failure
# ===========================================================================

class TestRunPipelineApplyingFixFailure:
    def test_stage_failed_on_git_apply_nonzero(self, tmp_path):
        with _full_patch(tmp_path=tmp_path, git_apply_rc=1) as _:
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED

    def test_error_mentions_git_apply(self, tmp_path):
        with _full_patch(tmp_path=tmp_path, git_apply_rc=1) as _:
            state = run_pipeline(str(tmp_path))
        assert "git apply failed" in state.error

    def test_stage_failed_on_agent_error(self, tmp_path):
        refactor_mock = MagicMock(side_effect=RuntimeError("refactor broke"))
        diag_agent = MagicMock()
        diag_agent.return_value.run.return_value = _DIAGNOSIS_JSON
        test_agent = MagicMock()
        test_agent.return_value.run.return_value = _GENERATED_TEST_JSON
        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch(
                "core.pipeline.run_tests",
                side_effect=[_FAILING_TEST_RESULT, _FAILING_TEST_RESULT],
            ),
            patch("core.pipeline.DiagnosticAgent", diag_agent),
            patch("core.pipeline.TestGeneratorAgent", test_agent),
            patch("core.pipeline.RefactoringAgent", refactor_mock),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED
        assert "APPLYING_FIX" in state.error


# ===========================================================================
# run_pipeline — VERIFYING failure
# ===========================================================================

class TestRunPipelineVerifyingFailure:
    def _create_src_file(self, tmp_path) -> None:
        """Create the calculator source file parse_patch needs to validate."""
        src = tmp_path / "src"
        src.mkdir(parents=True, exist_ok=True)
        (src / "calculator.py").write_text("def divide(a, b): return a/b\n")

    def test_stage_failed_on_runner_error(self, tmp_path):
        from core.runner import RunnerTimeoutError

        self._create_src_file(tmp_path)
        run_side_effects = [
            _FAILING_TEST_RESULT,          # initial
            _FAILING_TEST_RESULT,          # confirm
            RunnerTimeoutError("timeout"), # final
        ]
        git_mock = _mock_subprocess_run_success()
        diag_agent = MagicMock()
        diag_agent.return_value.run.return_value = _DIAGNOSIS_JSON
        test_agent = MagicMock()
        test_agent.return_value.run.return_value = _GENERATED_TEST_JSON
        refactor_agent = MagicMock()
        refactor_agent.return_value.run.return_value = _PATCH_JSON

        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch("core.pipeline.run_tests", side_effect=run_side_effects),
            patch("core.pipeline.DiagnosticAgent", diag_agent),
            patch("core.pipeline.TestGeneratorAgent", test_agent),
            patch("core.pipeline.RefactoringAgent", refactor_agent),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            state = run_pipeline(str(tmp_path))
        assert state.stage == PipelineStage.FAILED
        assert "VERIFYING" in state.error


# ===========================================================================
# run_pipeline — agent request objects are correctly constructed
# ===========================================================================

class TestRunPipelineRequestObjects:
    def _setup_src(self, tmp_path) -> None:
        """Create src/calculator.py so parse_patch's file-existence check passes."""
        src = tmp_path / "src"
        src.mkdir(parents=True, exist_ok=True)
        (src / "calculator.py").write_text("def divide(a, b): return a/b\n")

    def _make_patches(self, tmp_path):
        """Return the common mock objects and a partial `patch` context setup."""
        diag_agent = MagicMock()
        diag_agent.return_value.run.return_value = _DIAGNOSIS_JSON
        test_agent = MagicMock()
        test_agent.return_value.run.return_value = _GENERATED_TEST_JSON
        refactor_agent = MagicMock()
        refactor_agent.return_value.run.return_value = _PATCH_JSON
        git_mock = _mock_subprocess_run_success()
        return diag_agent, test_agent, refactor_agent, git_mock

    def test_diagnosis_request_uses_source_files(self, tmp_path):
        self._setup_src(tmp_path)
        diag_agent, test_agent, refactor_agent, git_mock = self._make_patches(tmp_path)

        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch(
                "core.pipeline.run_tests",
                side_effect=[_FAILING_TEST_RESULT, _FAILING_TEST_RESULT, _PASSING_TEST_RESULT],
            ),
            patch("core.pipeline.DiagnosticAgent", diag_agent),
            patch("core.pipeline.TestGeneratorAgent", test_agent),
            patch("core.pipeline.RefactoringAgent", refactor_agent),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            run_pipeline(str(tmp_path))

        called_request = diag_agent.return_value.run.call_args[0][0]
        assert called_request.source_files == _SOURCE_FILES

    def test_test_gen_request_includes_diagnosis(self, tmp_path):
        self._setup_src(tmp_path)
        diag_agent, test_agent, refactor_agent, git_mock = self._make_patches(tmp_path)

        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch(
                "core.pipeline.run_tests",
                side_effect=[_FAILING_TEST_RESULT, _FAILING_TEST_RESULT, _PASSING_TEST_RESULT],
            ),
            patch("core.pipeline.DiagnosticAgent", diag_agent),
            patch("core.pipeline.TestGeneratorAgent", test_agent),
            patch("core.pipeline.RefactoringAgent", refactor_agent),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            run_pipeline(str(tmp_path))

        called_request = test_agent.return_value.run.call_args[0][0]
        assert isinstance(called_request.diagnosis, DiagnosisResult)

    def test_refactor_request_includes_generated_test(self, tmp_path):
        self._setup_src(tmp_path)
        diag_agent, test_agent, refactor_agent, git_mock = self._make_patches(tmp_path)

        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch(
                "core.pipeline.run_tests",
                side_effect=[_FAILING_TEST_RESULT, _FAILING_TEST_RESULT, _PASSING_TEST_RESULT],
            ),
            patch("core.pipeline.DiagnosticAgent", diag_agent),
            patch("core.pipeline.TestGeneratorAgent", test_agent),
            patch("core.pipeline.RefactoringAgent", refactor_agent),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            run_pipeline(str(tmp_path))

        called_request = refactor_agent.return_value.run.call_args[0][0]
        assert isinstance(called_request.generated_test, GeneratedTest)

    def test_git_apply_called_with_patch_file(self, tmp_path):
        self._setup_src(tmp_path)
        diag_agent, test_agent, refactor_agent, git_mock = self._make_patches(tmp_path)

        with (
            patch("core.pipeline.load_repo", return_value=_SOURCE_FILES),
            patch(
                "core.pipeline.run_tests",
                side_effect=[_FAILING_TEST_RESULT, _FAILING_TEST_RESULT, _PASSING_TEST_RESULT],
            ),
            patch("core.pipeline.DiagnosticAgent", diag_agent),
            patch("core.pipeline.TestGeneratorAgent", test_agent),
            patch("core.pipeline.RefactoringAgent", refactor_agent),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            run_pipeline(str(tmp_path))

        call_args = git_mock.call_args[0][0]
        assert call_args[0] == "git"
        assert call_args[1] == "apply"


# ===========================================================================
# run_pipeline — generated test temp file is cleaned up
# ===========================================================================

class TestRunPipelineTempFileCleanup:
    def test_temp_test_file_removed_after_confirm(self, tmp_path):
        """The generated test file must not persist after GENERATING_TEST stage."""
        gen_test_abs = tmp_path / "tests" / "test_generated.py"
        gen_test_abs.parent.mkdir(parents=True, exist_ok=True)

        with _full_patch(tmp_path=tmp_path) as _:
            state = run_pipeline(str(tmp_path))

        # File should be gone even on success.
        assert not gen_test_abs.exists()
        assert state.stage == PipelineStage.COMPLETE
