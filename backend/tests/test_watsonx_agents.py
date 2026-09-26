"""
tests/test_watsonx_agents.py
-----------------------------
Focused unit tests for the real IBM watsonx.ai agents:
    * ``DiagnosticAgent``
    * ``TestGeneratorAgent``
    * ``RefactoringAgent``

All tests use ``unittest.mock`` to patch ``ibm_watsonx_ai`` so **no live
API key is required**.

Test strategy
~~~~~~~~~~~~~
1. ``_resolve_model_id`` — startup model availability check and fallback.
2. ``WatsonxAgent.__init__`` — credential validation, client construction.
3. ``WatsonxAgent.generate`` — response parsing, error handling.
4. Each agent's ``_build_prompt`` — content inspection without touching the model.
5. Each agent's ``run`` — verifies the full generate→return pipeline with a mocked
   ``ModelInference.generate`` response.
6. Integration with ``response_parser`` — raw mocked output parses cleanly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base_agent import (
    BaseAgent,
    WatsonxAgent,
    _resolve_model_id,
    PINNED_MODEL,
    _DEFAULT_URL,
)
from agents.diagnostic_agent import DiagnosticAgent
from agents.test_generator_agent import TestGeneratorAgent
from agents.refactoring_agent import RefactoringAgent
from core.models import (
    DiagnosisRequest,
    DiagnosisResult,
    GeneratedTest,
    TestGenRequest,
    RefactorRequest,
)
from core.response_parser import parse_diagnosis, parse_generated_test, parse_patch


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_SOURCE_FILES = {
    "src/calculator.py": (
        "def divide(a, b):\n"
        "    return a / b\n"
    )
}

_TEST_OUTPUT = (
    "FAILED tests/test_calculator.py::test_divide_with_none_raises_value_error\n"
    "TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'\n"
)

_DIAGNOSIS_RESULT = DiagnosisResult(
    root_cause="divide() does not guard against None inputs",
    affected_file="src/calculator.py",
    affected_lines=[1, 2],
    explanation="When a or b is None, the / operator raises TypeError.",
    confidence=0.95,
)

_GENERATED_TEST = GeneratedTest(
    filename="tests/test_generated.py",
    code=(
        "import pytest\n"
        "from src.calculator import divide\n\n"
        "def test_divide_with_none_raises():\n"
        "    with pytest.raises(TypeError):\n"
        "        divide(None, 1)\n"
    ),
    confirmed_failing=True,
)

# Valid raw model responses (what the model would return)
_VALID_DIAGNOSIS_RAW = json.dumps({
    "root_cause": "divide() does not guard against None inputs",
    "affected_file": "src/calculator.py",
    "affected_lines": [1, 2],
    "explanation": "When a or b is None, the / operator raises TypeError.",
    "confidence": 0.95,
})

_VALID_TEST_GEN_RAW = json.dumps({
    "filename": "tests/test_generated.py",
    "code": (
        "import pytest\n"
        "from src.calculator import divide\n\n"
        "def test_divide_with_none_raises():\n"
        "    with pytest.raises(TypeError):\n"
        "        divide(None, 1)\n"
    ),
})

_VALID_PATCH_RAW = json.dumps({
    "diff": (
        "diff --git a/src/calculator.py b/src/calculator.py\n"
        "--- a/src/calculator.py\n"
        "+++ b/src/calculator.py\n"
        "@@ -1,2 +1,5 @@\n"
        " def divide(a, b):\n"
        "+    if a is None or b is None:\n"
        "+        raise ValueError('Arguments must not be None')\n"
        "     return a / b\n"
    ),
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(generated_text: str) -> MagicMock:
    """Return a mock ModelInference whose generate() returns the expected structure."""
    mock = MagicMock()
    mock.generate.return_value = {
        "results": [{"generated_text": generated_text}]
    }
    return mock


def _make_watsonx_agent_with_mock(mock_model: MagicMock, env: dict | None = None) -> WatsonxAgent:
    """
    Construct a concrete WatsonxAgent subclass with a pre-built mock model.

    Uses monkeypatching-by-injection pattern: patch the SDK imports and skip
    ``_resolve_model_id`` so no network call is made.
    """
    env = env or {"WATSONX_API_KEY": "test-key", "WATSONX_PROJECT_ID": "test-proj"}

    class _ConcreteAgent(WatsonxAgent):
        def run(self, request):  # type: ignore[override]
            return self.generate("test prompt")

    with (
        patch.dict(os.environ, env, clear=False),
        patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
        patch("agents.base_agent.ModelInference", return_value=mock_model),
        patch("agents.base_agent.Credentials", return_value=MagicMock()),
    ):
        agent = _ConcreteAgent()

    # Inject mock directly so it's used after construction.
    agent._model = mock_model
    return agent


# ---------------------------------------------------------------------------
# _resolve_model_id — model availability check
# ---------------------------------------------------------------------------

class TestResolveModelId:
    """Unit tests for the startup model availability check."""

    def test_returns_pinned_model_when_available(self):
        """When pinned model is in the list, it is returned unchanged."""
        mock_specs = {
            "resources": [
                {"model_id": PINNED_MODEL},
                {"model_id": "ibm/granite-13b-instruct-v2"},
            ]
        }
        with patch("agents.base_agent.get_model_specs", return_value=mock_specs):
            result = _resolve_model_id(_DEFAULT_URL)
        assert result == PINNED_MODEL

    def test_falls_back_to_first_granite_instruct(self):
        """When pinned model is absent, selects the first granite instruct alphabetically."""
        mock_specs = {
            "resources": [
                {"model_id": "ibm/granite-13b-instruct-v2"},
                {"model_id": "ibm/granite-3-3-8b-chat"},  # not instruct — should not match
                {"model_id": "ibm/granite-7b-instruct"},
            ]
        }
        with patch("agents.base_agent.get_model_specs", return_value=mock_specs):
            result = _resolve_model_id(_DEFAULT_URL)
        # Pinned model absent; first alphabetical granite instruct is granite-13b-instruct-v2
        assert "granite" in result.lower()
        assert "instruct" in result.lower()
        assert result != PINNED_MODEL

    def test_falls_back_to_pinned_when_no_granite_instruct_found(self):
        """When no granite instruct model is available, returns PINNED_MODEL with a warning."""
        mock_specs = {
            "resources": [
                {"model_id": "ibm/granite-7b-chat"},
                {"model_id": "meta-llama/llama-3-8b"},
            ]
        }
        with patch("agents.base_agent.get_model_specs", return_value=mock_specs):
            result = _resolve_model_id(_DEFAULT_URL)
        assert result == PINNED_MODEL

    def test_falls_back_to_pinned_when_get_model_specs_raises(self):
        """Network errors in model check are caught; PINNED_MODEL is used."""
        with patch(
            "agents.base_agent.get_model_specs",
            side_effect=Exception("network error"),
        ):
            result = _resolve_model_id(_DEFAULT_URL)
        assert result == PINNED_MODEL

    def test_falls_back_when_resources_key_missing(self):
        """Empty or malformed specs dict returns PINNED_MODEL."""
        with patch("agents.base_agent.get_model_specs", return_value={}):
            result = _resolve_model_id(_DEFAULT_URL)
        assert result == PINNED_MODEL

    def test_fallback_logs_warning(self, caplog):
        """A warning is logged when falling back to an alternative model."""
        import logging
        mock_specs = {
            "resources": [
                {"model_id": "ibm/granite-13b-instruct-v2"},
            ]
        }
        with patch("agents.base_agent.get_model_specs", return_value=mock_specs):
            with caplog.at_level(logging.WARNING, logger="agents.base_agent"):
                result = _resolve_model_id(_DEFAULT_URL)
        assert result != PINNED_MODEL
        assert any("Falling back" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# WatsonxAgent construction
# ---------------------------------------------------------------------------

class TestWatsonxAgentConstruction:
    """Tests for credential validation and client setup."""

    def test_raises_environment_error_without_api_key(self, monkeypatch):
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj-123")

        class _Agent(WatsonxAgent):
            def run(self, request):
                pass

        with pytest.raises(EnvironmentError, match="WATSONX_API_KEY"):
            _Agent()

    def test_raises_environment_error_without_project_id(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "key-abc")
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)

        class _Agent(WatsonxAgent):
            def run(self, request):
                pass

        with pytest.raises(EnvironmentError, match="WATSONX_PROJECT_ID"):
            _Agent()

    def test_raises_environment_error_when_both_missing(self, monkeypatch):
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)

        class _Agent(WatsonxAgent):
            def run(self, request):
                pass

        with pytest.raises(EnvironmentError):
            _Agent()

    def test_uses_default_url_when_not_set(self, monkeypatch):
        """When WATSONX_URL is absent, _DEFAULT_URL is used."""
        monkeypatch.setenv("WATSONX_API_KEY", "key-abc")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj-123")
        monkeypatch.delenv("WATSONX_URL", raising=False)

        captured_url = {}

        def fake_resolve(url):
            captured_url["url"] = url
            return PINNED_MODEL

        class _Agent(WatsonxAgent):
            def run(self, request):
                pass

        with (
            patch("agents.base_agent._resolve_model_id", side_effect=fake_resolve),
            patch("agents.base_agent.ModelInference", return_value=MagicMock()),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            _Agent()

        assert captured_url["url"] == _DEFAULT_URL

    def test_uses_custom_url_from_env(self, monkeypatch):
        """WATSONX_URL env var is passed to the model initialiser."""
        custom_url = "https://eu-gb.ml.cloud.ibm.com"
        monkeypatch.setenv("WATSONX_API_KEY", "key-abc")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj-123")
        monkeypatch.setenv("WATSONX_URL", custom_url)

        captured_url = {}

        def fake_resolve(url):
            captured_url["url"] = url
            return PINNED_MODEL

        class _Agent(WatsonxAgent):
            def run(self, request):
                pass

        with (
            patch("agents.base_agent._resolve_model_id", side_effect=fake_resolve),
            patch("agents.base_agent.ModelInference", return_value=MagicMock()),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            _Agent()

        assert captured_url["url"] == custom_url

    def test_model_id_stored_on_agent(self, monkeypatch):
        """After construction, _model_id reflects the resolved model."""
        monkeypatch.setenv("WATSONX_API_KEY", "key-abc")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj-123")

        class _Agent(WatsonxAgent):
            def run(self, request):
                pass

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=MagicMock()),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = _Agent()

        assert agent._model_id == PINNED_MODEL


# ---------------------------------------------------------------------------
# WatsonxAgent.generate — response parsing
# ---------------------------------------------------------------------------

class TestWatsonxAgentGenerate:
    """Tests for the generate() helper."""

    def _make_agent(self, model_mock: MagicMock) -> WatsonxAgent:
        """Construct a WatsonxAgent with an injected mock model."""
        with (
            patch.dict(os.environ, {"WATSONX_API_KEY": "k", "WATSONX_PROJECT_ID": "p"}),
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=model_mock),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            class _Agent(WatsonxAgent):
                def run(self, request):
                    return self.generate("prompt")

            agent = _Agent()
        agent._model = model_mock
        return agent

    def test_generate_returns_generated_text(self):
        mock = _make_mock_model("hello world")
        agent = self._make_agent(mock)
        assert agent.generate("test") == "hello world"

    def test_generate_calls_model_with_prompt(self):
        mock = _make_mock_model("output")
        agent = self._make_agent(mock)
        agent.generate("my prompt")
        mock.generate.assert_called_once_with(prompt="my prompt")

    def test_generate_raises_on_missing_results_key(self):
        mock = MagicMock()
        mock.generate.return_value = {"unexpected": "structure"}
        agent = self._make_agent(mock)
        with pytest.raises(RuntimeError, match="Unexpected response structure"):
            agent.generate("prompt")

    def test_generate_raises_on_empty_results(self):
        mock = MagicMock()
        mock.generate.return_value = {"results": []}
        agent = self._make_agent(mock)
        with pytest.raises(RuntimeError, match="Unexpected response structure"):
            agent.generate("prompt")

    def test_generate_raises_on_none_response(self):
        mock = MagicMock()
        mock.generate.return_value = None
        agent = self._make_agent(mock)
        with pytest.raises(RuntimeError, match="Unexpected response structure"):
            agent.generate("prompt")


# ---------------------------------------------------------------------------
# DiagnosticAgent
# ---------------------------------------------------------------------------

class TestDiagnosticAgent:
    """Tests for DiagnosticAgent._build_prompt and run()."""

    # -- prompt builder ---------------------------------------------------

    def test_prompt_includes_source_file_path(self):
        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        prompt = DiagnosticAgent._build_prompt(req)
        assert "src/calculator.py" in prompt

    def test_prompt_includes_source_file_content(self):
        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        prompt = DiagnosticAgent._build_prompt(req)
        assert "def divide" in prompt

    def test_prompt_includes_test_output(self):
        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        prompt = DiagnosticAgent._build_prompt(req)
        assert "FAILED" in prompt
        assert "TypeError" in prompt

    def test_prompt_instructs_json_only(self):
        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        prompt = DiagnosticAgent._build_prompt(req)
        assert "Return ONLY" in prompt
        assert "no prose" in prompt.lower() or "no Markdown" in prompt

    def test_prompt_lists_all_required_keys(self):
        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        prompt = DiagnosticAgent._build_prompt(req)
        for key in ("root_cause", "affected_file", "affected_lines", "explanation", "confidence"):
            assert key in prompt, f"Expected key '{key}' in prompt"

    def test_prompt_starts_with_opening_brace(self):
        """Prompt includes the required diagnostic JSON structure."""
        req = DiagnosisRequest(
            source_files=_SOURCE_FILES,
            test_output=_TEST_OUTPUT,
        )
        prompt = DiagnosticAgent._build_prompt(req)

        assert "Return exactly this JSON structure:" in prompt
        assert '"root_cause":' in prompt
        assert '"affected_file":' in prompt
        assert '"affected_lines":' in prompt
        assert '"confidence":' in prompt

    # -- run() with mocked model ------------------------------------------

    def test_run_returns_string(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_DIAGNOSIS_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = DiagnosticAgent()

        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        result = agent.run(req)
        assert isinstance(result, str)

    def test_run_returns_model_output_verbatim(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_DIAGNOSIS_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = DiagnosticAgent()

        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        result = agent.run(req)
        assert result == _VALID_DIAGNOSIS_RAW

    def test_run_output_parses_to_diagnosis_result(self, monkeypatch):
        """run() output can be passed directly to parse_diagnosis."""
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_DIAGNOSIS_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = DiagnosticAgent()

        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        raw = agent.run(req)
        diagnosis = parse_diagnosis(raw)
        assert diagnosis.root_cause == "divide() does not guard against None inputs"
        assert diagnosis.affected_file == "src/calculator.py"
        assert diagnosis.affected_lines == [1, 2]
        assert diagnosis.confidence == 0.95

    def test_run_output_with_markdown_fences_still_parses(self, monkeypatch):
        """Model output wrapped in ```json fences is handled by response_parser."""
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        fenced = f"```json\n{_VALID_DIAGNOSIS_RAW}\n```"
        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(fenced)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = DiagnosticAgent()

        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        raw = agent.run(req)
        diagnosis = parse_diagnosis(raw)
        assert diagnosis.root_cause == "divide() does not guard against None inputs"

    def test_run_calls_generate_with_built_prompt(self, monkeypatch):
        """run() passes the prompt from _build_prompt to generate()."""
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        mock_model = _make_mock_model(_VALID_DIAGNOSIS_RAW)
        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=mock_model),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = DiagnosticAgent()

        req = DiagnosisRequest(source_files=_SOURCE_FILES, test_output=_TEST_OUTPUT)
        expected_prompt = DiagnosticAgent._build_prompt(req)
        agent.run(req)
        mock_model.generate.assert_called_once_with(prompt=expected_prompt)


# ---------------------------------------------------------------------------
# TestGeneratorAgent
# ---------------------------------------------------------------------------

class TestTestGeneratorAgent:
    """Tests for TestGeneratorAgent._build_prompt and run()."""

    # -- prompt builder ---------------------------------------------------

    def _make_request(self) -> TestGenRequest:
        return TestGenRequest(
            source_files=_SOURCE_FILES,
            diagnosis=_DIAGNOSIS_RESULT,
        )

    def test_prompt_includes_root_cause(self):
        req = self._make_request()
        prompt = TestGeneratorAgent._build_prompt(req)
        assert "divide() does not guard against None inputs" in prompt

    def test_prompt_includes_affected_file_path(self):
        req = self._make_request()
        prompt = TestGeneratorAgent._build_prompt(req)
        assert "src/calculator.py" in prompt

    def test_prompt_includes_affected_file_content(self):
        req = self._make_request()
        prompt = TestGeneratorAgent._build_prompt(req)
        assert "def divide" in prompt

    def test_prompt_instructs_json_only(self):
        req = self._make_request()
        prompt = TestGeneratorAgent._build_prompt(req)
        assert "Return ONLY" in prompt

    def test_prompt_includes_filename_key(self):
        req = self._make_request()
        prompt = TestGeneratorAgent._build_prompt(req)
        assert '"filename"' in prompt

    def test_prompt_includes_code_key(self):
        req = self._make_request()
        prompt = TestGeneratorAgent._build_prompt(req)
        assert '"code"' in prompt

    def test_prompt_starts_with_opening_brace(self):
        """Prompt includes the required test-generation JSON structure."""
        req = self._make_request()
        prompt = TestGeneratorAgent._build_prompt(req)

        assert "exactly these two fields:" in prompt
        assert '"filename": "tests/test_generated.py"' in prompt
        assert '"code": "complete pytest source code"' in prompt

    # -- run() with mocked model ------------------------------------------

    def test_run_returns_string(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_TEST_GEN_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = TestGeneratorAgent()

        result = agent.run(self._make_request())
        assert isinstance(result, str)

    def test_run_returns_model_output_verbatim(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_TEST_GEN_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = TestGeneratorAgent()

        result = agent.run(self._make_request())
        assert result == _VALID_TEST_GEN_RAW

    def test_run_output_parses_to_generated_test(self, monkeypatch):
        """run() output can be passed directly to parse_generated_test."""
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_TEST_GEN_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = TestGeneratorAgent()

        raw = agent.run(self._make_request())
        test = parse_generated_test(raw)
        assert test.filename == "tests/test_generated.py"
        assert "def test_divide_with_none_raises" in test.code
        assert "from src.calculator import divide" in test.code

    def test_run_calls_generate_with_built_prompt(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        mock_model = _make_mock_model(_VALID_TEST_GEN_RAW)
        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=mock_model),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = TestGeneratorAgent()

        req = self._make_request()
        expected_prompt = TestGeneratorAgent._build_prompt(req)
        agent.run(req)
        mock_model.generate.assert_called_once_with(prompt=expected_prompt)


# ---------------------------------------------------------------------------
# RefactoringAgent
# ---------------------------------------------------------------------------

class TestRefactoringAgent:
    """Tests for RefactoringAgent._build_prompt and run()."""

    def _make_request(self) -> RefactorRequest:
        return RefactorRequest(
            source_files=_SOURCE_FILES,
            diagnosis=_DIAGNOSIS_RESULT,
            generated_test=_GENERATED_TEST,
        )

    # -- prompt builder ---------------------------------------------------

    def test_prompt_includes_root_cause(self):
        req = self._make_request()
        prompt = RefactoringAgent._build_prompt(req)
        assert "divide() does not guard against None inputs" in prompt

    def test_prompt_includes_affected_file_path(self):
        req = self._make_request()
        prompt = RefactoringAgent._build_prompt(req)
        assert "src/calculator.py" in prompt

    def test_prompt_includes_affected_file_content(self):
        req = self._make_request()
        prompt = RefactoringAgent._build_prompt(req)
        assert "def divide" in prompt

    def test_prompt_includes_generated_test_code(self):
        req = self._make_request()
        prompt = RefactoringAgent._build_prompt(req)
        assert "def test_divide_with_none_raises" in prompt

    def test_prompt_includes_diff_git_instruction(self):
        req = self._make_request()
        prompt = RefactoringAgent._build_prompt(req)
        assert "diff --git" in prompt

    def test_prompt_instructs_json_only(self):
        req = self._make_request()
        prompt = RefactoringAgent._build_prompt(req)
        assert "Return ONLY" in prompt

    def test_prompt_starts_with_opening_brace(self):
        """Prompt includes the required patch JSON instructions."""
        req = self._make_request()
        prompt = RefactoringAgent._build_prompt(req)

        assert "valid JSON object with one key: diff" in prompt
        assert 'The value of "diff" must be a Git diff string.' in prompt
        assert 'The diff must start with "diff --git".' in prompt

    # -- run() with mocked model ------------------------------------------

    def test_run_returns_string(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_PATCH_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = RefactoringAgent()

        result = agent.run(self._make_request())
        assert isinstance(result, str)

    def test_run_returns_model_output_verbatim(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_PATCH_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = RefactoringAgent()

        result = agent.run(self._make_request())
        assert result == _VALID_PATCH_RAW

    def test_run_output_parses_to_patch_result(self, monkeypatch, tmp_path):
        """run() output can be passed directly to parse_patch (no repo_path check)."""
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_PATCH_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = RefactoringAgent()

        raw = agent.run(self._make_request())
        # parse_patch without repo_path skips file existence check
        patch_result = parse_patch(raw)
        assert patch_result.validated is True
        assert patch_result.diff.startswith("diff --git")
        assert "@@" in patch_result.diff

    def test_run_output_parses_with_repo_path_check(self, monkeypatch, tmp_path):
        """parse_patch with repo_path validates that the target file exists."""
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        # Create the target file in tmp_path so the path check passes
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "calculator.py").write_text("def divide(a, b):\n    return a / b\n")

        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=_make_mock_model(_VALID_PATCH_RAW)),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = RefactoringAgent()

        raw = agent.run(self._make_request())
        patch_result = parse_patch(raw, repo_path=str(tmp_path))
        assert patch_result.validated is True

    def test_run_calls_generate_with_built_prompt(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")

        mock_model = _make_mock_model(_VALID_PATCH_RAW)
        with (
            patch("agents.base_agent._resolve_model_id", return_value=PINNED_MODEL),
            patch("agents.base_agent.ModelInference", return_value=mock_model),
            patch("agents.base_agent.Credentials", return_value=MagicMock()),
        ):
            agent = RefactoringAgent()

        req = self._make_request()
        expected_prompt = RefactoringAgent._build_prompt(req)
        agent.run(req)
        mock_model.generate.assert_called_once_with(prompt=expected_prompt)


# ---------------------------------------------------------------------------
# Response parser integration — confidence clamping and retry
# ---------------------------------------------------------------------------

class TestResponseParserIntegration:
    """Tests that mocked model output flows correctly through response_parser."""

    def test_confidence_above_one_is_clamped(self):
        raw = json.dumps({
            "root_cause": "bug",
            "affected_file": "src/x.py",
            "affected_lines": [1],
            "explanation": "explains",
            "confidence": 1.5,
        })
        result = parse_diagnosis(raw)
        assert result.confidence == 1.0

    def test_confidence_below_zero_is_clamped(self):
        raw = json.dumps({
            "root_cause": "bug",
            "affected_file": "src/x.py",
            "affected_lines": [1],
            "explanation": "explains",
            "confidence": -0.3,
        })
        result = parse_diagnosis(raw)
        assert result.confidence == 0.0

    def test_retry_fn_is_called_on_malformed_diagnosis(self):
        """When initial parse fails, retry_fn is called once."""
        malformed = '{"root_cause": "bug"}'  # missing required keys
        called = []

        def retry_fn(error_msg: str) -> str:
            called.append(error_msg)
            return _VALID_DIAGNOSIS_RAW

        result = parse_diagnosis(malformed, retry_fn=retry_fn)
        assert len(called) == 1
        assert result.root_cause == "divide() does not guard against None inputs"

    def test_parse_generated_test_accepts_mocked_output(self):
        result = parse_generated_test(_VALID_TEST_GEN_RAW)
        assert result.filename == "tests/test_generated.py"
        assert "pytest" in result.code

    def test_parse_patch_rejects_diff_without_hunk_marker(self):
        from core.response_parser import ParseError
        raw = json.dumps({"diff": "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n"})
        with pytest.raises(ParseError, match="hunk"):
            parse_patch(raw)

    def test_parse_patch_rejects_diff_without_diff_git_header(self):
        from core.response_parser import ParseError
        raw = json.dumps({"diff": "--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n"})
        with pytest.raises(ParseError, match="diff --git"):
            parse_patch(raw)

    def test_retry_fn_called_on_malformed_patch(self):
        """Retry is triggered once when the patch is invalid."""
        from core.response_parser import ParseError
        malformed = '{"diff": "not a real diff"}'
        called = []

        def retry_fn(error_msg: str) -> str:
            called.append(error_msg)
            return _VALID_PATCH_RAW

        result = parse_patch(malformed, retry_fn=retry_fn)
        assert len(called) == 1
        assert result.validated is True
