"""
tests/test_repo_loader.py
--------------------------
Unit tests for core/repo_loader.py.

Tests cover:
- Basic loading of .py files from a real sample directory
- Skipping of __pycache__ and .git directories
- Skipping of non-.py files
- Correct relative paths returned as POSIX strings
- FileNotFoundError for a non-existent path
- FileNotFoundError for a file path (not a directory)
- Empty directory returns an empty dict
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Ensure the project root is importable when running pytest from anywhere.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.repo_loader import load_repo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


# ---------------------------------------------------------------------------
# Tests using the real sample scenarios
# ---------------------------------------------------------------------------

class TestLoadRepoWithSamples:
    """Validate load_repo against the bundled sample scenarios."""

    def test_none_bug_loads_py_files(self):
        result = load_repo(str(SAMPLES_DIR / "none_bug"))
        assert len(result) > 0, "Expected at least one .py file"

    def test_none_bug_contains_src_calculator(self):
        result = load_repo(str(SAMPLES_DIR / "none_bug"))
        assert "src/calculator.py" in result

    def test_none_bug_contains_test_file(self):
        result = load_repo(str(SAMPLES_DIR / "none_bug"))
        assert "tests/test_calculator.py" in result

    def test_none_bug_file_content_is_string(self):
        result = load_repo(str(SAMPLES_DIR / "none_bug"))
        for _path, content in result.items():
            assert isinstance(content, str)

    def test_none_bug_calculator_content_has_divide(self):
        result = load_repo(str(SAMPLES_DIR / "none_bug"))
        assert "def divide" in result["src/calculator.py"]

    def test_broken_api_contains_router(self):
        result = load_repo(str(SAMPLES_DIR / "broken_api"))
        assert "src/router.py" in result

    def test_off_by_one_contains_list_utils(self):
        result = load_repo(str(SAMPLES_DIR / "off_by_one"))
        assert "src/list_utils.py" in result

    def test_keys_are_posix_paths(self):
        """Keys must use forward slashes, not backslashes."""
        result = load_repo(str(SAMPLES_DIR / "none_bug"))
        for key in result:
            assert "\\" not in key, f"Key contains backslash: {key!r}"


class TestLoadRepoSkipsDirectories:
    """Ensure __pycache__ and .git contents are excluded."""

    def test_no_pycache_files(self):
        for scenario in ("none_bug", "broken_api", "off_by_one"):
            result = load_repo(str(SAMPLES_DIR / scenario))
            for key in result:
                assert "__pycache__" not in key, (
                    f"__pycache__ path leaked into results: {key!r}"
                )

    def test_no_git_files(self):
        for scenario in ("none_bug", "broken_api", "off_by_one"):
            result = load_repo(str(SAMPLES_DIR / scenario))
            for key in result:
                assert not key.startswith(".git"), (
                    f".git path leaked into results: {key!r}"
                )


class TestLoadRepoWithTempDir:
    """Isolated tests using temporary directories for precise control."""

    def test_only_py_files_included(self, tmp_path):
        (tmp_path / "module.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# README", encoding="utf-8")
        (tmp_path / "data.txt").write_text("hello", encoding="utf-8")
        result = load_repo(str(tmp_path))
        assert list(result.keys()) == ["module.py"]

    def test_nested_py_files_included(self, tmp_path):
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "a.py").write_text("a = 1", encoding="utf-8")
        (tmp_path / "top.py").write_text("b = 2", encoding="utf-8")
        result = load_repo(str(tmp_path))
        assert "top.py" in result
        assert "pkg/a.py" in result

    def test_pycache_skipped(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "cached.py").write_text("# cached", encoding="utf-8")
        (tmp_path / "real.py").write_text("x = 1", encoding="utf-8")
        result = load_repo(str(tmp_path))
        assert "real.py" in result
        assert "__pycache__/cached.py" not in result

    def test_dot_directories_skipped(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("secret = True", encoding="utf-8")
        (tmp_path / "visible.py").write_text("x = 1", encoding="utf-8")
        result = load_repo(str(tmp_path))
        assert "visible.py" in result
        for key in result:
            assert not key.startswith(".hidden"), f"Hidden dir leaked: {key!r}"

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        result = load_repo(str(tmp_path))
        assert result == {}

    def test_file_contents_match(self, tmp_path):
        source = "def hello():\n    return 'world'\n"
        (tmp_path / "hello.py").write_text(source, encoding="utf-8")
        result = load_repo(str(tmp_path))
        assert result["hello.py"] == source


class TestLoadRepoErrors:
    """Edge-case and error-path tests."""

    def test_nonexistent_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_repo("/does/not/exist/anywhere")

    def test_file_path_raises_file_not_found(self, tmp_path):
        f = tmp_path / "notadir.py"
        f.write_text("x = 1", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            load_repo(str(f))
