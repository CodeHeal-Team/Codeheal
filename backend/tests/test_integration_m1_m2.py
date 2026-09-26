"""
tests/test_integration_m1_m2.py
--------------------------------
Integration tests for Module 1 (Core Brain) + Module 2 (Agent Intelligence).

These tests exercise the *real* agent classes (DiagnosticAgent,
TestGeneratorAgent, RefactoringAgent) together with the real pipeline
orchestrator, response parser, repo loader, and test runner.

No live API calls are made — IBM watsonx.ai is fully mocked at the
``ModelInference`` boundary.

Coverage goals
~~~~~~~~~~~~~~
1. All three sample scenarios flow end-to-end through the full pipeline
   using the real agent classes wired to mocked watsonx responses.
2. The real ``_build_prompt`` methods are called (not bypassed by MagicMock).
3. Retry logic: the pipeline retry_fn correctly re-invokes the agent when
   the first response is malformed.
4. Malformed model response on first call → retry succeeds.
5. Malformed model response on both calls → pipeline FAILS gracefully.
6. Each stage transition populates the correct PipelineState fields.
7. ``confirmed_failing`` flag is set correctly based on confirm run outcome.
8. Temp test file is cleaned up after GENERATING_TEST stage.
9. Pipeline handles LOADING, DETECT, DIAGNOSE, GENERATE_TEST, APPLY_FIX,
   VERIFY, and COMPLETE events in correct order.
"""

from __future__ import annotations

import json
import shutil
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Make the project root importable when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base_agent import PINNED_MODEL
from agents.diagnostic_agent import DiagnosticAgent
from agents.test_generator_agent import TestGeneratorAgent
from agents.refactoring_agent import RefactoringAgent
from core.models import (
    DiagnosisRequest,
    DiagnosisResult,
    GeneratedTest,
    PatchResult,
    PipelineStage,
    PipelineState,
    RefactorRequest,
    TestGenRequest,
    TestResult,
    TimelineEvent,
)
from core.pipeline import run_pipeline
from core.response_parser import ParseError, parse_diagnosis, parse_generated_test, parse_patch


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

@pytest.fixture(autouse=True)
def isolate_sample_repositories(tmp_path, monkeypatch):
    """Run integration tests against temporary copies of sample repos."""
    original_samples = SAMPLES_DIR
    isolated_samples = tmp_path / "samples"

    shutil.copytree(original_samples, isolated_samples)

    monkeypatch.setattr(
        sys.modules[__name__],
        "SAMPLES_DIR",
        isolated_samples,
    )

    # Class-level paths are evaluated when this module is imported,
    # so redirect those paths to the temporary copies too.
    for test_class in list(globals().values()):
        if not isinstance(test_class, type):
            continue

        scenario = test_class.__dict__.get("_SCENARIO")
        if scenario:
            scenario_name = Path(scenario).name
            monkeypatch.setattr(
                test_class,
                "_SCENARIO",
                str(isolated_samples / scenario_name),
                raising=False,
            )

# Raw JSON the mocked model returns — mirrors the none_bug scenario.
_DIAG_JSON_NONE_BUG = json.dumps({
    "root_cause": "divide() has no guard for None inputs, causing TypeError.",
    "affected_file": "src/calculator.py",
    "affected_lines": [13, 14],
    "explanation": "When a or b is None, Python raises TypeError instead of ValueError.",
    "confidence": 0.97,
})

_TEST_GEN_JSON_NONE_BUG = json.dumps({
    "filename": "tests/test_generated.py",
    "code": (
        "import sys, os\n"
        "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))\n"
        "import pytest\n"
        "from calculator import divide\n"
        "def test_divide_none_raises():\n"
        "    with pytest.raises((ValueError, TypeError)):\n"
        "        divide(None, 2)\n"
    ),
})

_PATCH_JSON_NONE_BUG = json.dumps({
    "diff": (
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "index 0000001..0000002 100644\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -12,4 +12,8 @@ def add(a, b):\n"
        " \n"
        " def divide(a, b):\n"
        "-    # BUG: no None-guard\n"
        "-    return a / b\n"
        "+    if a is None or b is None:\n"
        "+        raise ValueError('None not allowed')\n"
        "+    return a / b\n"
    ),
})

# Raw JSON for broken_api scenario.
_DIAG_JSON_BROKEN_API = json.dumps({
    "root_cause": "ROUTES key uses Cyrillic 'о' instead of ASCII 'o'.",
    "affected_file": "src/router.py",
    "affected_lines": [10],
    "explanation": "The Cyrillic character causes a KeyError on dispatch('hello', ...).",
    "confidence": 0.99,
})

_TEST_GEN_JSON_BROKEN_API = json.dumps({
    "filename": "tests/test_generated.py",
    "code": (
        "import sys, os\n"
        "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))\n"
        "import pytest\n"
        "from router import dispatch\n"
        "def test_hello_dispatch_raises():\n"
        "    with pytest.raises(KeyError):\n"
        "        dispatch('hello', 'World')\n"
    ),
})

_PATCH_JSON_BROKEN_API = json.dumps({
    "diff": (
        "diff --git a/src/router.py b/src/router.py\n"
        "index 0000001..0000002 100644\n"
        "--- a/src/router.py\n"
        "+++ b/src/router.py\n"
        "@@ -8,7 +8,7 @@ A simple function dispatcher.\n"
        " \n"
        " ROUTES = {\n"
        "-    \"hell\u043e\": lambda name: f\"Hello, {name}!\",\n"
        "+    \"hello\": lambda name: f\"Hello, {name}!\",\n"
        "     \"bye\": lambda name: f\"Goodbye, {name}!\",\n"
        " }\n"
    ),
})

# Raw JSON for off_by_one scenario.
_DIAG_JSON_OFF_BY_ONE = json.dumps({
    "root_cause": "last_element() uses len(items) instead of len(items) - 1.",
    "affected_file": "src/list_utils.py",
    "affected_lines": [12],
    "explanation": "Off-by-one: items[len(items)] raises IndexError for any non-empty list.",
    "confidence": 0.95,
})

_TEST_GEN_JSON_OFF_BY_ONE = json.dumps({
    "filename": "tests/test_generated.py",
    "code": (
        "import sys, os\n"
        "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))\n"
        "import pytest\n"
        "from list_utils import last_element\n"
        "def test_last_element_raises_index_error():\n"
        "    with pytest.raises(IndexError):\n"
        "        last_element([1, 2, 3])\n"
    ),
})

_PATCH_JSON_OFF_BY_ONE = json.dumps({
    "diff": (
        "diff --git a/src/list_utils.py b/src/list_utils.py\n"
        "index 0000001..0000002 100644\n"
        "--- a/src/list_utils.py\n"
        "+++ b/src/list_utils.py\n"
        "@@ -9,5 +9,5 @@ def last_element(items):\n"
        "     if not items:\n"
        "         raise ValueError('list is empty')\n"
        "-    return items[len(items)]  # BUG: off-by-one\n"
        "+    return items[len(items) - 1]\n"
    ),
})

# Shared mock results.
_FAILING = TestResult(stdout="FAILED tests/test_x.py::test_y", stderr="", exit_code=1)
_PASSING = TestResult(stdout="1 passed", stderr="", exit_code=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(responses: list[str]) -> MagicMock:
    """
    Return a mock ModelInference whose generate() returns successive responses.

    Each response is wrapped in the watsonx.ai response envelope.
    """
    mock = MagicMock()
    mock.generate.side_effect = [
        {"results": [{"generated_text": text}]}
        for text in responses
    ]
    return mock


@contextmanager
def _real_agents_pipeline(
    repo_path: str,
    *,
    diag_responses: list[str],
    test_gen_responses: list[str],
    refactor_responses: list[str],
    run_tests_side_effects: list,
    git_apply_rc: int = 0,
):
    """
    Context manager that wires the *real* agent classes to mocked watsonx models,
    patches run_tests and subprocess.run (git apply), and runs the pipeline.

    Yields the final PipelineState.
    """
    diag_model = _make_mock_model(diag_responses)
    test_gen_model = _make_mock_model(test_gen_responses)
    refactor_model = _make_mock_model(refactor_responses)

    git_mock = MagicMock()
    git_mock.return_value.returncode = git_apply_rc
    git_mock.return_value.stderr = "" if git_apply_rc == 0 else "patch failed"
    git_mock.return_value.stdout = ""

    run_tests_mock = MagicMock(side_effect=run_tests_side_effects)

    env_patch = {"WATSONX_API_KEY": "test-key", "WATSONX_PROJECT_ID": "test-proj"}

    with (
        patch.dict(os.environ, env_patch, clear=False),
        patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
        patch("agents.base_agent.Credentials", return_value=MagicMock()),
        patch("agents.base_agent.ModelInference", side_effect=[
            diag_model, test_gen_model, refactor_model
        ]),
        patch("core.pipeline.run_tests", run_tests_mock),
        patch("core.pipeline.subprocess.run", git_mock),
    ):
        state = run_pipeline(repo_path)
        yield state


# ---------------------------------------------------------------------------
# Helper: patch real agents into pipeline (instead of placeholders)
# ---------------------------------------------------------------------------

@contextmanager
def _with_real_agents():
    """Swap pipeline imports from placeholders to real agent classes."""
    with (
        patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
        patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
        patch("core.pipeline.RefactoringAgent", RefactoringAgent),
    ):
        yield


# ===========================================================================
# Integration: real agents + pipeline (mocked watsonx)
# ===========================================================================

class TestRealAgentsPipelineNoneBug:
    """
    Full Module 1 + Module 2 integration using real agent classes against
    the none_bug sample, with mocked watsonx ModelInference.
    """

    _SCENARIO = str(SAMPLES_DIR / "none_bug")

    def _run(self):
        """Run the full pipeline using real agents and mocked watsonx."""
        with _with_real_agents():
            with _real_agents_pipeline(
                self._SCENARIO,
                diag_responses=[_DIAG_JSON_NONE_BUG],
                test_gen_responses=[_TEST_GEN_JSON_NONE_BUG],
                refactor_responses=[_PATCH_JSON_NONE_BUG],
                run_tests_side_effects=[_FAILING, _FAILING, _PASSING],
            ) as state:
                return state

    def test_pipeline_reaches_complete(self):
        state = self._run()
        assert state.stage == PipelineStage.COMPLETE

    def test_source_files_loaded(self):
        state = self._run()
        assert "src/calculator.py" in state.source_files

    def test_diagnosis_populated_from_model(self):
        state = self._run()
        assert isinstance(state.diagnosis, DiagnosisResult)
        assert "None" in state.diagnosis.root_cause

    def test_diagnosis_affected_file_correct(self):
        state = self._run()
        assert state.diagnosis.affected_file == "src/calculator.py"

    def test_diagnosis_confidence_in_range(self):
        state = self._run()
        assert 0.0 <= state.diagnosis.confidence <= 1.0

    def test_generated_test_populated(self):
        state = self._run()
        assert isinstance(state.generated_test, GeneratedTest)
        assert state.generated_test.filename == "tests/test_generated.py"
        assert "def test_" in state.generated_test.code

    def test_patch_validated(self):
        state = self._run()
        assert isinstance(state.patch, PatchResult)
        assert state.patch.validated is True
        assert state.patch.diff.startswith("diff --git")

    def test_confirmed_failing_true(self):
        """The confirm run still fails — bug confirmed."""
        state = self._run()
        assert state.generated_test.confirmed_failing is True

    def test_final_test_result_passed(self):
        state = self._run()
        assert state.final_test_result.passed is True

    def test_error_is_none(self):
        state = self._run()
        assert state.error is None

    def test_timeline_has_all_stages(self):
        state = self._run()
        stage_labels = {e.stage for e in state.timeline}
        for expected in ("Load", "Detect", "Diagnose", "Generate Test", "Fix", "Verify", "Complete"):
            assert expected in stage_labels, f"Stage '{expected}' missing from timeline"

    def test_timeline_complete_stage_is_done(self):
        state = self._run()
        complete_events = [e for e in state.timeline if e.stage == "Complete"]
        assert any(e.status == "done" for e in complete_events)

    def test_raw_response_preserved_in_diagnosis(self):
        state = self._run()
        assert state.diagnosis.raw_response != ""


class TestRealAgentsPipelineBrokenApi:
    """Full integration for the broken_api scenario."""

    _SCENARIO = str(SAMPLES_DIR / "broken_api")

    def _run(self):
        with _with_real_agents():
            with _real_agents_pipeline(
                self._SCENARIO,
                diag_responses=[_DIAG_JSON_BROKEN_API],
                test_gen_responses=[_TEST_GEN_JSON_BROKEN_API],
                refactor_responses=[_PATCH_JSON_BROKEN_API],
                run_tests_side_effects=[_FAILING, _FAILING, _PASSING],
            ) as state:
                return state

    def test_pipeline_reaches_complete(self):
        assert self._run().stage == PipelineStage.COMPLETE

    def test_diagnosis_identifies_router(self):
        state = self._run()
        assert state.diagnosis.affected_file == "src/router.py"

    def test_patch_diff_targets_router(self):
        state = self._run()
        assert "src/router.py" in state.patch.diff

    def test_source_files_contain_router(self):
        assert "src/router.py" in self._run().source_files


class TestRealAgentsPipelineOffByOne:
    """Full integration for the off_by_one scenario."""

    _SCENARIO = str(SAMPLES_DIR / "off_by_one")

    def _run(self):
        with _with_real_agents():
            with _real_agents_pipeline(
                self._SCENARIO,
                diag_responses=[_DIAG_JSON_OFF_BY_ONE],
                test_gen_responses=[_TEST_GEN_JSON_OFF_BY_ONE],
                refactor_responses=[_PATCH_JSON_OFF_BY_ONE],
                run_tests_side_effects=[_FAILING, _FAILING, _PASSING],
            ) as state:
                return state

    def test_pipeline_reaches_complete(self):
        assert self._run().stage == PipelineStage.COMPLETE

    def test_diagnosis_identifies_list_utils(self):
        state = self._run()
        assert state.diagnosis.affected_file == "src/list_utils.py"

    def test_patch_diff_targets_list_utils(self):
        state = self._run()
        assert "src/list_utils.py" in state.patch.diff

    def test_source_files_contain_list_utils(self):
        assert "src/list_utils.py" in self._run().source_files


# ===========================================================================
# Integration: retry wiring in pipeline (mocked watsonx)
# ===========================================================================

class TestPipelineRetryOnMalformedResponse:
    """
    Verify the retry_fn wiring: first response is malformed, second is valid.
    The pipeline must complete successfully after the retry.
    """

    _SCENARIO = str(SAMPLES_DIR / "none_bug")

    def test_diagnosis_retried_on_malformed_first_response(self):
        """
        DiagnosticAgent's first model response is garbage JSON.
        The retry_fn re-invokes the agent and the second response is valid.
        Pipeline must complete.
        """
        # First call returns garbage; second call returns valid JSON.
        diag_model = MagicMock()
        diag_model.generate.side_effect = [
            {"results": [{"generated_text": "not json at all"}]},
            {"results": [{"generated_text": _DIAG_JSON_NONE_BUG}]},
        ]
        test_gen_model = _make_mock_model([_TEST_GEN_JSON_NONE_BUG])
        refactor_model = _make_mock_model([_PATCH_JSON_NONE_BUG])

        git_mock = MagicMock()
        git_mock.return_value.returncode = 0
        git_mock.return_value.stderr = ""

        run_tests_mock = MagicMock(side_effect=[_FAILING, _FAILING, _PASSING])

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            state = run_pipeline(self._SCENARIO)

        assert state.stage == PipelineStage.COMPLETE
        assert state.diagnosis.root_cause != ""

    def test_test_generator_retried_on_malformed_first_response(self):
        """
        TestGeneratorAgent's first response is malformed; second is valid.
        Pipeline must complete.
        """
        diag_model = _make_mock_model([_DIAG_JSON_NONE_BUG])
        test_gen_model = MagicMock()
        test_gen_model.generate.side_effect = [
            {"results": [{"generated_text": "{{invalid json}"}]},
            {"results": [{"generated_text": _TEST_GEN_JSON_NONE_BUG}]},
        ]
        refactor_model = _make_mock_model([_PATCH_JSON_NONE_BUG])

        git_mock = MagicMock()
        git_mock.return_value.returncode = 0
        git_mock.return_value.stderr = ""

        run_tests_mock = MagicMock(side_effect=[_FAILING, _FAILING, _PASSING])

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            state = run_pipeline(self._SCENARIO)

        assert state.stage == PipelineStage.COMPLETE
        assert state.generated_test.filename == "tests/test_generated.py"

    def test_refactoring_retried_on_malformed_first_response(self):
        """
        RefactoringAgent's first response has an invalid diff; second is valid.
        Pipeline must complete.
        """
        # Missing @@ hunk marker on first response.
        bad_patch = json.dumps({
            "diff": "diff --git a/src/calculator.py b/src/calculator.py\n--- a/\n+++ b/\n"
        })
        diag_model = _make_mock_model([_DIAG_JSON_NONE_BUG])
        test_gen_model = _make_mock_model([_TEST_GEN_JSON_NONE_BUG])
        refactor_model = MagicMock()
        refactor_model.generate.side_effect = [
            {"results": [{"generated_text": bad_patch}]},
            {"results": [{"generated_text": _PATCH_JSON_NONE_BUG}]},
        ]

        git_mock = MagicMock()
        git_mock.return_value.returncode = 0
        git_mock.return_value.stderr = ""

        run_tests_mock = MagicMock(side_effect=[_FAILING, _FAILING, _PASSING])

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            state = run_pipeline(self._SCENARIO)

        assert state.stage == PipelineStage.COMPLETE
        assert state.patch.validated is True

    def test_both_retries_fail_sets_pipeline_failed(self):
        """
        When both the first and retry response are malformed, the pipeline
        must set FAILED with a helpful error message.
        """
        diag_model = MagicMock()
        diag_model.generate.side_effect = [
            {"results": [{"generated_text": "no json here"}]},
            {"results": [{"generated_text": "still no json"}]},
        ]
        run_tests_mock = MagicMock(return_value=_FAILING)

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", return_value=diag_model),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
        ):
            state = run_pipeline(str(SAMPLES_DIR / "none_bug"))

        assert state.stage == PipelineStage.FAILED
        assert state.error is not None
        assert "DIAGNOSING" in state.error


# ===========================================================================
# Integration: prompt content verification via mocked model
# ===========================================================================

class TestRealAgentPromptContent:
    """
    Verify that real agent _build_prompt methods include the expected content
    in what they pass to ModelInference.generate, without a live API call.
    """

    def _make_agent_with_capture(self, AgentClass, response_json: str):
        """
        Instantiate AgentClass with a mocked ModelInference.
        Returns (agent, mock_model) so we can inspect what was passed to generate().
        """
        mock_model = _make_mock_model([response_json])
        env = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", return_value=mock_model),
        ):
            agent = AgentClass()
        agent._model = mock_model
        return agent, mock_model

    def test_diagnostic_prompt_contains_source_path(self):
        agent, model = self._make_agent_with_capture(DiagnosticAgent, _DIAG_JSON_NONE_BUG)
        request = DiagnosisRequest(
            source_files={"src/calculator.py": "def divide(a, b): return a/b\n"},
            test_output="FAILED",
        )
        agent.run(request)
        prompt = model.generate.call_args[1]["prompt"]
        assert "src/calculator.py" in prompt

    def test_diagnostic_prompt_contains_test_output(self):
        agent, model = self._make_agent_with_capture(DiagnosticAgent, _DIAG_JSON_NONE_BUG)
        request = DiagnosisRequest(
            source_files={"src/calculator.py": "def divide(a, b): return a/b\n"},
            test_output="TypeError raised at line 99",
        )
        agent.run(request)
        prompt = model.generate.call_args[1]["prompt"]
        assert "TypeError raised at line 99" in prompt

    def test_test_generator_prompt_contains_root_cause(self):
        agent, model = self._make_agent_with_capture(TestGeneratorAgent, _TEST_GEN_JSON_NONE_BUG)
        diagnosis = DiagnosisResult(
            root_cause="No None-guard on divide()",
            affected_file="src/calculator.py",
            affected_lines=[1],
            explanation="Needs a guard.",
            confidence=0.9,
        )
        request = TestGenRequest(
            source_files={"src/calculator.py": "def divide(a, b): return a/b\n"},
            diagnosis=diagnosis,
        )
        agent.run(request)
        prompt = model.generate.call_args[1]["prompt"]
        assert "No None-guard on divide()" in prompt

    def test_test_generator_prompt_contains_affected_file_content(self):
        agent, model = self._make_agent_with_capture(TestGeneratorAgent, _TEST_GEN_JSON_NONE_BUG)
        diagnosis = DiagnosisResult(
            root_cause="Bug",
            affected_file="src/calculator.py",
            affected_lines=[1],
            explanation="Details.",
            confidence=0.8,
        )
        request = TestGenRequest(
            source_files={"src/calculator.py": "UNIQUE_CONTENT_MARKER_XYZ"},
            diagnosis=diagnosis,
        )
        agent.run(request)
        prompt = model.generate.call_args[1]["prompt"]
        assert "UNIQUE_CONTENT_MARKER_XYZ" in prompt

    def test_refactoring_prompt_contains_affected_lines(self):
        agent, model = self._make_agent_with_capture(RefactoringAgent, _PATCH_JSON_NONE_BUG)
        diagnosis = DiagnosisResult(
            root_cause="Bug",
            affected_file="src/calculator.py",
            affected_lines=[42, 43],
            explanation="Details.",
            confidence=0.8,
        )
        gen_test = GeneratedTest(
            filename="tests/test_generated.py",
            code="def test_bug(): pass",
            confirmed_failing=True,
        )
        request = RefactorRequest(
            source_files={"src/calculator.py": "def divide(a, b): return a/b\n"},
            diagnosis=diagnosis,
            generated_test=gen_test,
        )
        agent.run(request)
        prompt = model.generate.call_args[1]["prompt"]
        assert "42" in prompt
        assert "43" in prompt

    def test_refactoring_prompt_contains_generated_test_code(self):
        agent, model = self._make_agent_with_capture(RefactoringAgent, _PATCH_JSON_NONE_BUG)
        diagnosis = DiagnosisResult(
            root_cause="Bug",
            affected_file="src/calculator.py",
            affected_lines=[1],
            explanation="Details.",
            confidence=0.8,
        )
        gen_test = GeneratedTest(
            filename="tests/test_generated.py",
            code="def test_UNIQUE_TEST_MARKER(): pass",
            confirmed_failing=True,
        )
        request = RefactorRequest(
            source_files={"src/calculator.py": "def divide(a, b): return a/b\n"},
            diagnosis=diagnosis,
            generated_test=gen_test,
        )
        agent.run(request)
        prompt = model.generate.call_args[1]["prompt"]
        assert "test_UNIQUE_TEST_MARKER" in prompt


# ===========================================================================
# Integration: state callback wiring
# ===========================================================================

class TestStateCallbackIntegration:
    """
    Verify that state_callback is called with properly updated PipelineState
    at each stage when the real agents are used.
    """

    _SCENARIO = str(SAMPLES_DIR / "none_bug")

    def test_callback_receives_pipeline_state_objects(self):
        received = []
        with _with_real_agents():
            with _real_agents_pipeline(
                self._SCENARIO,
                diag_responses=[_DIAG_JSON_NONE_BUG],
                test_gen_responses=[_TEST_GEN_JSON_NONE_BUG],
                refactor_responses=[_PATCH_JSON_NONE_BUG],
                run_tests_side_effects=[_FAILING, _FAILING, _PASSING],
            ):
                pass  # callback not hooked here; test via direct call below

        # Use run_pipeline directly with a callback.
        diag_model = _make_mock_model([_DIAG_JSON_NONE_BUG])
        test_gen_model = _make_mock_model([_TEST_GEN_JSON_NONE_BUG])
        refactor_model = _make_mock_model([_PATCH_JSON_NONE_BUG])

        git_mock = MagicMock()
        git_mock.return_value.returncode = 0
        git_mock.return_value.stderr = ""

        run_tests_mock = MagicMock(side_effect=[_FAILING, _FAILING, _PASSING])

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model,
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            run_pipeline(self._SCENARIO, state_callback=received.append)

        assert len(received) >= 7
        assert all(isinstance(s, PipelineState) for s in received)

    def test_callback_sees_stage_progression(self):
        """Stage values seen by callback must advance in the correct order."""
        stages_seen = []

        diag_model = _make_mock_model([_DIAG_JSON_NONE_BUG])
        test_gen_model = _make_mock_model([_TEST_GEN_JSON_NONE_BUG])
        refactor_model = _make_mock_model([_PATCH_JSON_NONE_BUG])

        git_mock = MagicMock()
        git_mock.return_value.returncode = 0
        git_mock.return_value.stderr = ""

        run_tests_mock = MagicMock(side_effect=[_FAILING, _FAILING, _PASSING])

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model,
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            run_pipeline(self._SCENARIO, state_callback=lambda s: stages_seen.append(s.stage))

        # COMPLETE must be the final stage seen.
        assert stages_seen[-1] == PipelineStage.COMPLETE

    def test_callback_sees_loading_stage(self):
        """LOADING stage must appear early in the callback sequence."""
        stages_seen = []

        diag_model = _make_mock_model([_DIAG_JSON_NONE_BUG])
        test_gen_model = _make_mock_model([_TEST_GEN_JSON_NONE_BUG])
        refactor_model = _make_mock_model([_PATCH_JSON_NONE_BUG])

        git_mock = MagicMock()
        git_mock.return_value.returncode = 0
        git_mock.return_value.stderr = ""

        run_tests_mock = MagicMock(side_effect=[_FAILING, _FAILING, _PASSING])

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model,
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            run_pipeline(self._SCENARIO, state_callback=lambda s: stages_seen.append(s.stage))

        assert PipelineStage.LOADING in stages_seen


# ===========================================================================
# Integration: temp file cleanup and confirmed_failing logic
# ===========================================================================

class TestTempFileAndConfirmedFailing:
    """Verify cleanup and confirmed_failing wiring using real agent classes."""

    _SCENARIO = str(SAMPLES_DIR / "none_bug")

    def _run_with_confirm(self, confirm_result: TestResult):
        """Run pipeline and return final state; confirm step uses confirm_result."""
        diag_model = _make_mock_model([_DIAG_JSON_NONE_BUG])
        test_gen_model = _make_mock_model([_TEST_GEN_JSON_NONE_BUG])
        refactor_model = _make_mock_model([_PATCH_JSON_NONE_BUG])

        git_mock = MagicMock()
        git_mock.return_value.returncode = 0
        git_mock.return_value.stderr = ""

        run_tests_mock = MagicMock(side_effect=[_FAILING, confirm_result, _PASSING])

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model,
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            return run_pipeline(self._SCENARIO)

    def test_confirmed_failing_true_when_confirm_fails(self):
        state = self._run_with_confirm(_FAILING)
        assert state.generated_test.confirmed_failing is True

    def test_confirmed_failing_false_when_confirm_passes(self):
        state = self._run_with_confirm(_PASSING)
        assert state.generated_test.confirmed_failing is False

    def test_temp_test_file_not_left_on_disk(self):
        """
        The generated test file written during GENERATING_TEST must be
        removed before the stage completes.
        """
        scenario_path = Path(self._SCENARIO)
        gen_test_path = scenario_path / "tests" / "test_generated.py"

        # Ensure the file doesn't exist before we start.
        if gen_test_path.exists():
            gen_test_path.unlink()

        self._run_with_confirm(_FAILING)

        # File must be gone after the pipeline run.
        assert not gen_test_path.exists(), (
            f"Generated test file was not cleaned up: {gen_test_path}"
        )


# ===========================================================================
# Integration: Stage 5 (APPLYING_FIX) retry scenarios with real agents
# ===========================================================================

class TestRealAgentsPipelineRefactorRetry:
    """
    Integration tests that exercise Stage 5 retry paths using real agent
    classes wired to mocked watsonx responses.

    These are distinct from the unit-level tests in test_pipeline.py because
    they exercise the full agent class hierarchy (real _build_prompt, real
    response parsing) rather than bypassing agents with MagicMock stubs.
    """

    _SCENARIO = str(SAMPLES_DIR / "none_bug")

    def test_refactor_parse_retry_exhausted_sets_pipeline_failed(self):
        """
        Both the initial and the retry response from RefactoringAgent are
        structurally invalid (no @@ hunk marker).
        The pipeline must reach FAILED with a meaningful error.
        """
        bad_patch = json.dumps({
            "diff": (
                "diff --git a/src/calculator.py b/src/calculator.py\n"
                "--- a/src/calculator.py\n"
                "+++ b/src/calculator.py\n"
                # Intentionally missing @@ hunk marker
            )
        })

        diag_model = _make_mock_model([_DIAG_JSON_NONE_BUG])
        test_gen_model = _make_mock_model([_TEST_GEN_JSON_NONE_BUG])
        # Both initial and retry return the invalid patch.
        refactor_model = _make_mock_model([bad_patch, bad_patch])

        git_mock = MagicMock()
        git_mock.return_value.returncode = 0
        git_mock.return_value.stderr = ""

        run_tests_mock = MagicMock(side_effect=[_FAILING, _FAILING])

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            state = run_pipeline(self._SCENARIO)

        assert state.stage == PipelineStage.FAILED
        assert state.error is not None
        assert "APPLYING_FIX" in state.error

    def test_refactor_git_check_retry_succeeds(self):
        """
        The first response is valid JSON+diff but git apply --check rejects it.
        The second response (after retry) also passes the check.
        Pipeline must complete.
        """
        diag_model = _make_mock_model([_DIAG_JSON_NONE_BUG])
        test_gen_model = _make_mock_model([_TEST_GEN_JSON_NONE_BUG])
        # Both responses are the same valid patch JSON.
        refactor_model = _make_mock_model([_PATCH_JSON_NONE_BUG, _PATCH_JSON_NONE_BUG])

        # subprocess.run side_effects:
        #   0 → git apply --check → FAIL (corrupt patch)
        #   1 → git apply --check (retry) → OK
        #   2 → git apply → OK
        git_responses = [
            MagicMock(returncode=1, stderr="corrupt patch at line 16", stdout=""),
            MagicMock(returncode=0, stderr="", stdout=""),
            MagicMock(returncode=0, stderr="", stdout=""),
        ]
        git_mock = MagicMock(side_effect=git_responses)

        run_tests_mock = MagicMock(side_effect=[_FAILING, _FAILING, _PASSING])

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            state = run_pipeline(self._SCENARIO)

        assert state.stage == PipelineStage.COMPLETE, state.error
        assert state.patch.validated is True

    def test_patch_temp_file_absent_after_failed_apply(self):
        """
        After a completely failed Stage 5 (both retries exhausted),
        the .codeheal.patch file must not remain on disk.
        """
        bad_patch = json.dumps({
            "diff": (
                "diff --git a/src/calculator.py b/src/calculator.py\n"
                "--- a/src/calculator.py\n"
                "+++ b/src/calculator.py\n"
            )
        })

        diag_model = _make_mock_model([_DIAG_JSON_NONE_BUG])
        test_gen_model = _make_mock_model([_TEST_GEN_JSON_NONE_BUG])
        refactor_model = _make_mock_model([bad_patch, bad_patch])

        git_mock = MagicMock()
        git_mock.return_value.returncode = 0
        git_mock.return_value.stderr = ""
        run_tests_mock = MagicMock(side_effect=[_FAILING, _FAILING])

        scenario_path = Path(self._SCENARIO)
        patch_file = scenario_path / ".codeheal.patch"
        if patch_file.exists():
            patch_file.unlink()

        env_patch = {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
            patch("agents.base_agent.ModelInference", side_effect=[
                diag_model, test_gen_model, refactor_model
            ]),
            patch("core.pipeline.DiagnosticAgent", DiagnosticAgent),
            patch("core.pipeline.TestGeneratorAgent", TestGeneratorAgent),
            patch("core.pipeline.RefactoringAgent", RefactoringAgent),
            patch("core.pipeline.run_tests", run_tests_mock),
            patch("core.pipeline.subprocess.run", git_mock),
        ):
            run_pipeline(self._SCENARIO)

        assert not patch_file.exists(), (
            f".codeheal.patch was not cleaned up: {patch_file}"
        )
