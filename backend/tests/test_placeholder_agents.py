"""
tests/test_placeholder_agents.py
---------------------------------
Focused unit tests for ``agents/placeholders.py`` and the common
``BaseAgent`` interface.

Coverage targets
----------------
BaseAgent contract
    * ``BaseAgent`` is abstract — cannot be instantiated directly
    * All three real agent classes (``DiagnosticAgent``, ``TestGeneratorAgent``,
      ``RefactoringAgent``) inherit ``BaseAgent`` and raise ``EnvironmentError``
      when instantiated without credentials (they now require watsonx credentials)

Placeholder agents — interface
    * All three placeholder classes inherit ``BaseAgent``
    * ``run()`` accepts the correct request type and returns a ``str``
    * Return value is valid JSON

Placeholder agents — JSON content
    * ``PlaceholderDiagnosticAgent`` returns JSON with all required
      ``DiagnosisResult`` keys and correct types
    * ``PlaceholderTestGeneratorAgent`` returns JSON with ``filename`` and
      ``code`` keys
    * ``PlaceholderRefactoringAgent`` returns JSON with a ``diff`` key starting
      with ``diff --git``

Scenario inference
    * Correct fixture is loaded for ``none_bug``, ``broken_api``, and
      ``off_by_one`` based on ``source_files`` content

Fixture fallback
    * Unknown source files fall back to the ``none_bug`` fixture

Error handling
    * Missing fixture file raises ``FileNotFoundError``

Prompt builder helpers
    * ``DiagnosticAgent._build_prompt`` includes source file paths and the
      required JSON schema description
    * ``TestGeneratorAgent._build_prompt`` includes diagnosis info
    * ``RefactoringAgent._build_prompt`` includes the affected file path
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the project root importable when running pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base_agent import BaseAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.test_generator_agent import TestGeneratorAgent
from agents.refactoring_agent import RefactoringAgent
from agents.placeholders import (
    PlaceholderDiagnosticAgent,
    PlaceholderTestGeneratorAgent,
    PlaceholderRefactoringAgent,
    _infer_scenario,
    _read_fixture,
    _FALLBACK_SCENARIO,
)
from core.models import (
    DiagnosisRequest,
    DiagnosisResult,
    TestGenRequest,
    GeneratedTest,
    RefactorRequest,
)


# ---------------------------------------------------------------------------
# Test helpers / shared fixtures
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _none_bug_source_files() -> dict[str, str]:
    """Minimal source_files dict that identifies the none_bug scenario."""
    return {"src/calculator.py": "def divide(a, b):\n    return a / b\n"}


def _broken_api_source_files() -> dict[str, str]:
    """Minimal source_files dict that identifies the broken_api scenario."""
    return {"src/router.py": "ROUTES = {'hello': lambda n: n}\n"}


def _off_by_one_source_files() -> dict[str, str]:
    """Minimal source_files dict that identifies the off_by_one scenario."""
    return {"src/list_utils.py": "def last_element(items):\n    return items[len(items)]\n"}


def _sample_diagnosis() -> DiagnosisResult:
    return DiagnosisResult(
        root_cause="No None-guard in divide()",
        affected_file="src/calculator.py",
        affected_lines=[13, 14],
        explanation="divide() crashes on None input.",
        confidence=0.95,
    )


def _make_diagnosis_request(source_files: dict[str, str] | None = None) -> DiagnosisRequest:
    return DiagnosisRequest(
        source_files=source_files or _none_bug_source_files(),
        test_output="FAILED tests/test_calculator.py::test_divide_with_none_raises_value_error",
    )


def _make_test_gen_request(source_files: dict[str, str] | None = None) -> TestGenRequest:
    return TestGenRequest(
        source_files=source_files or _none_bug_source_files(),
        diagnosis=_sample_diagnosis(),
    )


def _make_refactor_request(source_files: dict[str, str] | None = None) -> RefactorRequest:
    return RefactorRequest(
        source_files=source_files or _none_bug_source_files(),
        diagnosis=_sample_diagnosis(),
        generated_test=GeneratedTest(
            filename="tests/test_generated.py",
            code="def test_placeholder(): pass",
            confirmed_failing=True,
        ),
    )


# ---------------------------------------------------------------------------
# BaseAgent contract
# ---------------------------------------------------------------------------

class TestBaseAgentContract:
    """Verify the ABC cannot be instantiated and that subclasses must implement run."""

    def test_base_agent_is_abstract(self):
        """Directly instantiating BaseAgent must raise TypeError."""
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_diagnostic_agent_is_subclass(self):
        assert issubclass(DiagnosticAgent, BaseAgent)

    def test_test_generator_agent_is_subclass(self):
        assert issubclass(TestGeneratorAgent, BaseAgent)

    def test_refactoring_agent_is_subclass(self):
        assert issubclass(RefactoringAgent, BaseAgent)

    def test_diagnostic_agent_requires_credentials(self, monkeypatch):
        """Instantiating DiagnosticAgent without env vars raises EnvironmentError."""
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        with pytest.raises(EnvironmentError):
            DiagnosticAgent()

    def test_test_generator_agent_requires_credentials(self, monkeypatch):
        """Instantiating TestGeneratorAgent without env vars raises EnvironmentError."""
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        with pytest.raises(EnvironmentError):
            TestGeneratorAgent()

    def test_refactoring_agent_requires_credentials(self, monkeypatch):
        """Instantiating RefactoringAgent without env vars raises EnvironmentError."""
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        with pytest.raises(EnvironmentError):
            RefactoringAgent()


# ---------------------------------------------------------------------------
# Placeholder agents — BaseAgent inheritance
# ---------------------------------------------------------------------------

class TestPlaceholderAgentInheritance:
    """All placeholder classes must be concrete BaseAgent subclasses."""

    def test_placeholder_diagnostic_is_subclass(self):
        assert issubclass(PlaceholderDiagnosticAgent, BaseAgent)

    def test_placeholder_test_generator_is_subclass(self):
        assert issubclass(PlaceholderTestGeneratorAgent, BaseAgent)

    def test_placeholder_refactoring_is_subclass(self):
        assert issubclass(PlaceholderRefactoringAgent, BaseAgent)

    def test_placeholder_diagnostic_instantiates(self):
        agent = PlaceholderDiagnosticAgent()
        assert agent is not None

    def test_placeholder_test_generator_instantiates(self):
        agent = PlaceholderTestGeneratorAgent()
        assert agent is not None

    def test_placeholder_refactoring_instantiates(self):
        agent = PlaceholderRefactoringAgent()
        assert agent is not None


# ---------------------------------------------------------------------------
# Placeholder agents — run() returns a string
# ---------------------------------------------------------------------------

class TestPlaceholderAgentRunReturnsString:
    """run() must return str for every placeholder and every scenario."""

    @pytest.mark.parametrize("scenario_files,expected_scenario", [
        (_none_bug_source_files(), "none_bug"),
        (_broken_api_source_files(), "broken_api"),
        (_off_by_one_source_files(), "off_by_one"),
    ])
    def test_diagnostic_run_returns_str(self, scenario_files, expected_scenario):
        agent = PlaceholderDiagnosticAgent()
        result = agent.run(_make_diagnosis_request(scenario_files))
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0

    @pytest.mark.parametrize("scenario_files,expected_scenario", [
        (_none_bug_source_files(), "none_bug"),
        (_broken_api_source_files(), "broken_api"),
        (_off_by_one_source_files(), "off_by_one"),
    ])
    def test_test_generator_run_returns_str(self, scenario_files, expected_scenario):
        agent = PlaceholderTestGeneratorAgent()
        result = agent.run(_make_test_gen_request(scenario_files))
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0

    @pytest.mark.parametrize("scenario_files,expected_scenario", [
        (_none_bug_source_files(), "none_bug"),
        (_broken_api_source_files(), "broken_api"),
        (_off_by_one_source_files(), "off_by_one"),
    ])
    def test_refactoring_run_returns_str(self, scenario_files, expected_scenario):
        agent = PlaceholderRefactoringAgent()
        result = agent.run(_make_refactor_request(scenario_files))
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Placeholder agents — JSON validity
# ---------------------------------------------------------------------------

class TestPlaceholderAgentReturnsValidJson:
    """The raw string returned by run() must be parseable as JSON."""

    def test_diagnostic_returns_valid_json(self):
        agent = PlaceholderDiagnosticAgent()
        raw = agent.run(_make_diagnosis_request())
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_test_generator_returns_valid_json(self):
        agent = PlaceholderTestGeneratorAgent()
        raw = agent.run(_make_test_gen_request())
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_refactoring_returns_valid_json(self):
        agent = PlaceholderRefactoringAgent()
        raw = agent.run(_make_refactor_request())
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Placeholder agents — DiagnosisResult schema compliance
# ---------------------------------------------------------------------------

class TestPlaceholderDiagnosticSchema:
    """Verify the JSON from PlaceholderDiagnosticAgent matches DiagnosisResult."""

    @pytest.fixture(autouse=True)
    def _parse(self):
        agent = PlaceholderDiagnosticAgent()
        raw = agent.run(_make_diagnosis_request())
        self.parsed = json.loads(raw)

    def test_has_root_cause_key(self):
        assert "root_cause" in self.parsed

    def test_root_cause_is_string(self):
        assert isinstance(self.parsed["root_cause"], str)
        assert len(self.parsed["root_cause"]) > 0

    def test_has_affected_file_key(self):
        assert "affected_file" in self.parsed

    def test_affected_file_is_string(self):
        assert isinstance(self.parsed["affected_file"], str)

    def test_has_affected_lines_key(self):
        assert "affected_lines" in self.parsed

    def test_affected_lines_is_list(self):
        assert isinstance(self.parsed["affected_lines"], list)

    def test_affected_lines_contains_ints(self):
        for item in self.parsed["affected_lines"]:
            assert isinstance(item, int), f"Expected int, got {type(item)}: {item}"

    def test_has_explanation_key(self):
        assert "explanation" in self.parsed

    def test_explanation_is_string(self):
        assert isinstance(self.parsed["explanation"], str)
        assert len(self.parsed["explanation"]) > 0

    def test_has_confidence_key(self):
        assert "confidence" in self.parsed

    def test_confidence_is_float(self):
        assert isinstance(self.parsed["confidence"], float)

    def test_confidence_in_range(self):
        assert 0.0 <= self.parsed["confidence"] <= 1.0

    def test_affected_file_is_src_calculator(self):
        """none_bug fixture must point at the correct file."""
        assert self.parsed["affected_file"] == "src/calculator.py"


# ---------------------------------------------------------------------------
# Placeholder agents — GeneratedTest schema compliance
# ---------------------------------------------------------------------------

class TestPlaceholderTestGeneratorSchema:
    """Verify the JSON from PlaceholderTestGeneratorAgent matches GeneratedTest."""

    @pytest.fixture(autouse=True)
    def _parse(self):
        agent = PlaceholderTestGeneratorAgent()
        raw = agent.run(_make_test_gen_request())
        self.parsed = json.loads(raw)

    def test_has_filename_key(self):
        assert "filename" in self.parsed

    def test_filename_is_string(self):
        assert isinstance(self.parsed["filename"], str)
        assert len(self.parsed["filename"]) > 0

    def test_filename_ends_with_py(self):
        assert self.parsed["filename"].endswith(".py")

    def test_has_code_key(self):
        assert "code" in self.parsed

    def test_code_is_string(self):
        assert isinstance(self.parsed["code"], str)
        assert len(self.parsed["code"]) > 0

    def test_code_contains_def_test(self):
        """Generated test must contain at least one test function."""
        assert "def test_" in self.parsed["code"]

    def test_code_contains_import(self):
        """Generated test must have an import statement."""
        assert "import" in self.parsed["code"]


# ---------------------------------------------------------------------------
# Placeholder agents — PatchResult schema compliance
# ---------------------------------------------------------------------------

class TestPlaceholderRefactoringSchema:
    """Verify the JSON from PlaceholderRefactoringAgent matches PatchResult."""

    @pytest.fixture(autouse=True)
    def _parse(self):
        agent = PlaceholderRefactoringAgent()
        raw = agent.run(_make_refactor_request())
        self.parsed = json.loads(raw)

    def test_has_diff_key(self):
        assert "diff" in self.parsed

    def test_diff_is_string(self):
        assert isinstance(self.parsed["diff"], str)
        assert len(self.parsed["diff"]) > 0

    def test_diff_starts_with_diff_git(self):
        assert self.parsed["diff"].startswith("diff --git")

    def test_diff_contains_hunk_marker(self):
        assert "@@" in self.parsed["diff"]

    def test_diff_contains_minus_line(self):
        """Diff must remove at least one line."""
        assert "\n-" in self.parsed["diff"]

    def test_diff_contains_plus_line(self):
        """Diff must add at least one line."""
        assert "\n+" in self.parsed["diff"]


# ---------------------------------------------------------------------------
# Scenario inference
# ---------------------------------------------------------------------------

class TestScenarioInference:
    """_infer_scenario() must select the right scenario from source_files keys."""

    def test_none_bug_inferred(self):
        assert _infer_scenario(_none_bug_source_files()) == "none_bug"

    def test_broken_api_inferred(self):
        assert _infer_scenario(_broken_api_source_files()) == "broken_api"

    def test_off_by_one_inferred(self):
        assert _infer_scenario(_off_by_one_source_files()) == "off_by_one"

    def test_unknown_falls_back_to_default(self):
        result = _infer_scenario({"src/unknown_module.py": "x = 1"})
        assert result == _FALLBACK_SCENARIO

    def test_empty_source_files_falls_back(self):
        result = _infer_scenario({})
        assert result == _FALLBACK_SCENARIO

    def test_correct_fixture_loaded_for_broken_api(self):
        """PlaceholderDiagnosticAgent picks up the broken_api diagnosis fixture."""
        agent = PlaceholderDiagnosticAgent()
        raw = agent.run(_make_diagnosis_request(_broken_api_source_files()))
        parsed = json.loads(raw)
        assert parsed["affected_file"] == "src/router.py"

    def test_correct_fixture_loaded_for_off_by_one(self):
        """PlaceholderDiagnosticAgent picks up the off_by_one diagnosis fixture."""
        agent = PlaceholderDiagnosticAgent()
        raw = agent.run(_make_diagnosis_request(_off_by_one_source_files()))
        parsed = json.loads(raw)
        assert parsed["affected_file"] == "src/list_utils.py"


# ---------------------------------------------------------------------------
# Fixture error handling
# ---------------------------------------------------------------------------

class TestFixtureErrorHandling:
    """_read_fixture() must raise FileNotFoundError for missing files."""

    def test_missing_fixture_raises_file_not_found(self, tmp_path):
        """A non-existent scenario/fixture combination raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _read_fixture("nonexistent_scenario", "diagnosis.json")

    def test_missing_fixture_file_in_known_scenario(self, tmp_path, monkeypatch):
        """A known scenario with a missing fixture file raises FileNotFoundError."""
        # Temporarily point SAMPLES_DIR at a tmp directory so none_bug/fixtures is absent
        import agents.placeholders as ph
        original = ph._SAMPLES_DIR
        monkeypatch.setattr(ph, "_SAMPLES_DIR", tmp_path)
        try:
            with pytest.raises(FileNotFoundError):
                _read_fixture("none_bug", "diagnosis.json")
        finally:
            monkeypatch.setattr(ph, "_SAMPLES_DIR", original)


# ---------------------------------------------------------------------------
# Prompt builder helpers
# ---------------------------------------------------------------------------

class TestPromptBuilders:
    """Static _build_prompt methods produce correct content."""

    def test_diagnostic_prompt_includes_source_paths(self):
        request = _make_diagnosis_request()
        prompt = DiagnosticAgent._build_prompt(request)
        assert "src/calculator.py" in prompt

    def test_diagnostic_prompt_includes_test_output(self):
        request = _make_diagnosis_request()
        prompt = DiagnosticAgent._build_prompt(request)
        assert request.test_output in prompt

    def test_diagnostic_prompt_includes_schema_keys(self):
        request = _make_diagnosis_request()
        prompt = DiagnosticAgent._build_prompt(request)
        for key in ("root_cause", "affected_file", "affected_lines", "explanation", "confidence"):
            assert key in prompt, f"Schema key missing from prompt: {key!r}"

    def test_diagnostic_prompt_instructs_json_only(self):
        request = _make_diagnosis_request()
        prompt = DiagnosticAgent._build_prompt(request)
        assert "JSON" in prompt

    def test_test_generator_prompt_includes_root_cause(self):
        request = _make_test_gen_request()
        prompt = TestGeneratorAgent._build_prompt(request)
        assert request.diagnosis.root_cause in prompt

    def test_test_generator_prompt_includes_affected_file(self):
        request = _make_test_gen_request()
        prompt = TestGeneratorAgent._build_prompt(request)
        assert request.diagnosis.affected_file in prompt

    def test_test_generator_prompt_includes_source_content(self):
        request = _make_test_gen_request()
        prompt = TestGeneratorAgent._build_prompt(request)
        # The affected file content must appear in the prompt.
        assert "def divide" in prompt

    def test_test_generator_prompt_instructs_json_only(self):
        request = _make_test_gen_request()
        prompt = TestGeneratorAgent._build_prompt(request)
        assert "JSON" in prompt

    def test_refactoring_prompt_includes_affected_file_path(self):
        request = _make_refactor_request()
        prompt = RefactoringAgent._build_prompt(request)
        assert request.diagnosis.affected_file in prompt

    def test_refactoring_prompt_includes_generated_test_code(self):
        request = _make_refactor_request()
        prompt = RefactoringAgent._build_prompt(request)
        assert request.generated_test.code in prompt

    def test_refactoring_prompt_instructs_json_only(self):
        request = _make_refactor_request()
        prompt = RefactoringAgent._build_prompt(request)
        assert "JSON" in prompt

    def test_refactoring_prompt_mentions_diff_git(self):
        request = _make_refactor_request()
        prompt = RefactoringAgent._build_prompt(request)
        assert "diff --git" in prompt
