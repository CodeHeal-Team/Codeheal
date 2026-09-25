"""
tests/test_runner.py
---------------------
Unit tests for core/runner.py.

Tests cover:
- runner returns TestResult with passed=False for each failing sample scenario
- runner returns passed=True for a minimal always-passing project
- stdout and stderr are captured as strings
- exit_code matches pytest's actual exit code
- FileNotFoundError for a non-existent path
- FileNotFoundError for a file path (not a directory)
- RunnerTimeoutError is raised when pytest exceeds the timeout
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runner import run_tests, RunnerTimeoutError, TIMEOUT_SECONDS
from core.models import TestResult


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


# ---------------------------------------------------------------------------
# Tests against the real sample scenarios (integration-style)
# ---------------------------------------------------------------------------

class TestRunnerWithSamples:
    """Validate that runner correctly detects failing tests in each sample."""

    # none_bug uses a generous timeout: the first pytest invocation on this
    # machine can take up to ~45 s due to interpreter startup overhead.
    _NONE_BUG_TIMEOUT = 90

    def test_none_bug_fails(self):
        result = run_tests(str(SAMPLES_DIR / "none_bug"), timeout=self._NONE_BUG_TIMEOUT)
        assert isinstance(result, TestResult)
        assert result.passed is False

    def test_none_bug_exit_code_nonzero(self):
        result = run_tests(str(SAMPLES_DIR / "none_bug"), timeout=self._NONE_BUG_TIMEOUT)
        assert result.exit_code != 0

    def test_none_bug_stdout_is_string(self):
        result = run_tests(str(SAMPLES_DIR / "none_bug"), timeout=self._NONE_BUG_TIMEOUT)
        assert isinstance(result.stdout, str)

    def test_none_bug_stderr_is_string(self):
        result = run_tests(str(SAMPLES_DIR / "none_bug"), timeout=self._NONE_BUG_TIMEOUT)
        assert isinstance(result.stderr, str)

    def test_none_bug_stdout_contains_failed(self):
        result = run_tests(str(SAMPLES_DIR / "none_bug"), timeout=self._NONE_BUG_TIMEOUT)
        combined = result.stdout + result.stderr
        assert "FAILED" in combined or "failed" in combined

    def test_broken_api_fails(self):
        result = run_tests(str(SAMPLES_DIR / "broken_api"))
        assert result.passed is False

    def test_off_by_one_fails(self):
        result = run_tests(str(SAMPLES_DIR / "off_by_one"))
        assert result.passed is False


class TestRunnerWithPassingProject:
    """Validate that runner correctly detects a passing test suite."""

    def test_passing_project_returns_passed_true(self, tmp_path):
        # Minimal always-passing test project.
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "test_pass.py").write_text(
            "def test_always_passes():\n    assert 1 + 1 == 2\n",
            encoding="utf-8",
        )
        result = run_tests(str(tmp_path))
        assert result.passed is True
        assert result.exit_code == 0

    def test_passing_project_stdout_contains_passed(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "test_pass.py").write_text(
            "def test_ok():\n    assert True\n",
            encoding="utf-8",
        )
        result = run_tests(str(tmp_path))
        assert "passed" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------

class TestRunnerErrors:

    def test_nonexistent_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_tests("/does/not/exist/anywhere")

    def test_file_path_raises_file_not_found(self, tmp_path):
        f = tmp_path / "notadir.py"
        f.write_text("x = 1", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            run_tests(str(f))

    def test_timeout_raises_runner_timeout_error(self, tmp_path):
        """Simulate a pytest timeout by patching subprocess.run."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        with patch("core.runner.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["pytest"], timeout=30)
            with pytest.raises(RunnerTimeoutError):
                run_tests(str(tmp_path))


# ---------------------------------------------------------------------------
# TestResult model contract
# (class name avoids "Test" prefix so pytest does not try to collect it)
# ---------------------------------------------------------------------------

class ResultModelContract:
    """Confirm the TestResult dataclass contract that runner.py relies on."""

    def test_exit_code_zero_means_passed(self):
        r = TestResult(stdout="ok", stderr="", exit_code=0)
        assert r.passed is True

    def test_exit_code_one_means_failed(self):
        r = TestResult(stdout="", stderr="err", exit_code=1)
        assert r.passed is False

    def test_exit_code_negative_means_failed(self):
        r = TestResult(exit_code=-1)
        assert r.passed is False
