"""
tests/test_dashboard.py
-----------------------
Unit tests for the pure-Python helper functions in ``dashboard/app.py``.

These tests import only the logic functions that have no Streamlit dependency.
They do NOT launch a Streamlit server or call any ``st.*`` functions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ── ensure the package root is importable ───────────────────────────────────
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# We need to import only the pure-Python helpers without triggering Streamlit.
# Import the module but stub ``streamlit`` before it is loaded.
import types
import unittest.mock as mock

# Create a minimal stub for streamlit so the import doesn't fail in CI.
_st_stub = types.ModuleType("streamlit")
_st_stub.set_page_config = mock.MagicMock()
_st_stub.session_state = {}
sys.modules.setdefault("streamlit", _st_stub)

# Now import the helpers we want to test.
from dashboard.app import _parse_diff_sides, _count_files_changed  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_diff_sides
# ---------------------------------------------------------------------------

_SIMPLE_DIFF = """\
diff --git a/src/calculator.py b/src/calculator.py
index 0000001..0000002 100644
--- a/src/calculator.py
+++ b/src/calculator.py
@@ -12,4 +12,8 @@ def add(a, b):

 def divide(a, b):
-    # BUG: no None-guard
-    return a / b
+    if a is None or b is None:
+        raise ValueError("divide() does not accept None")
+    return a / b
"""


class TestParseDiffSides:
    """Tests for _parse_diff_sides."""

    def test_returns_two_lists(self):
        before, after = _parse_diff_sides(_SIMPLE_DIFF)
        assert isinstance(before, list)
        assert isinstance(after, list)

    def test_removed_lines_only_in_before(self):
        before, after = _parse_diff_sides(_SIMPLE_DIFF)
        assert "    # BUG: no None-guard" in before
        assert "    # BUG: no None-guard" not in after

    def test_added_lines_only_in_after(self):
        before, after = _parse_diff_sides(_SIMPLE_DIFF)
        assert '    if a is None or b is None:' in after
        assert '    if a is None or b is None:' not in before

    def test_context_lines_in_both(self):
        before, after = _parse_diff_sides(_SIMPLE_DIFF)
        # "def divide(a, b):" is a context line (leading space stripped) — must appear in both
        assert "def divide(a, b):" in before
        assert "def divide(a, b):" in after

    def test_diff_header_lines_excluded(self):
        """Lines starting with diff, index, ---, +++ must not appear in output."""
        before, after = _parse_diff_sides(_SIMPLE_DIFF)
        all_lines = before + after
        for line in all_lines:
            assert not line.startswith("diff --git")
            assert not line.startswith("index ")
            assert not line.startswith("---")
            assert not line.startswith("+++")

    def test_hunk_header_excluded(self):
        before, after = _parse_diff_sides(_SIMPLE_DIFF)
        all_lines = before + after
        assert not any(ln.startswith("@@") for ln in all_lines)

    def test_empty_diff_returns_empty_lists(self):
        before, after = _parse_diff_sides("")
        assert before == []
        assert after == []

    def test_diff_with_no_changes(self):
        """A diff with only context lines produces identical before and after."""
        diff = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,2 +1,2 @@\n"
            " line_one\n"
            " line_two\n"
        )
        before, after = _parse_diff_sides(diff)
        assert before == after

    def test_only_additions(self):
        diff = (
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,1 +1,2 @@\n"
            " existing\n"
            "+new_line\n"
        )
        before, after = _parse_diff_sides(diff)
        assert "new_line" in after
        assert "new_line" not in before

    def test_only_removals(self):
        diff = (
            "diff --git a/f.py b/f.py\n"
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,2 +1,1 @@\n"
            " existing\n"
            "-removed\n"
        )
        before, after = _parse_diff_sides(diff)
        assert "removed" in before
        assert "removed" not in after


# ---------------------------------------------------------------------------
# _count_files_changed
# ---------------------------------------------------------------------------

class TestCountFilesChanged:
    """Tests for _count_files_changed."""

    def test_single_file(self):
        diff = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n-old\n+new\n"
        assert _count_files_changed(diff) == 1

    def test_two_files(self):
        diff = (
            "diff --git a/x.py b/x.py\n-old\n+new\n"
            "diff --git a/y.py b/y.py\n-old2\n+new2\n"
        )
        assert _count_files_changed(diff) == 2

    def test_empty_diff(self):
        assert _count_files_changed("") == 0

    def test_no_diff_lines(self):
        assert _count_files_changed("some random text\nno diff headers") == 0

    def test_three_files(self):
        diff = (
            "diff --git a/a.py b/a.py\n"
            "diff --git a/b.py b/b.py\n"
            "diff --git a/c.py b/c.py\n"
        )
        assert _count_files_changed(diff) == 3


# ---------------------------------------------------------------------------
# Integration: parse_diff_sides with the real none_bug fixture
# ---------------------------------------------------------------------------

class TestParseDiffWithRealFixture:
    """Smoke-tests against the actual none_bug patch fixture."""

    @pytest.fixture
    def fixture_diff(self) -> str:
        import json
        fixture_path = _PKG_ROOT / "samples" / "none_bug" / "fixtures" / "patch.json"
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        return data["diff"]

    def test_before_contains_bug_comment(self, fixture_diff):
        before, _ = _parse_diff_sides(fixture_diff)
        joined = "\n".join(before)
        assert "BUG" in joined

    def test_after_contains_none_check(self, fixture_diff):
        _, after = _parse_diff_sides(fixture_diff)
        joined = "\n".join(after)
        assert "None" in joined

    def test_after_does_not_contain_bug_comment(self, fixture_diff):
        _, after = _parse_diff_sides(fixture_diff)
        joined = "\n".join(after)
        assert "BUG" not in joined

    def test_file_count_is_one(self, fixture_diff):
        assert _count_files_changed(fixture_diff) == 1


# ---------------------------------------------------------------------------
# _BUNDLED_SCENARIOS constant sanity checks
# ---------------------------------------------------------------------------

class TestBundledScenarios:
    """Verify the bundled scenario paths exist on disk."""

    def test_all_bundled_scenario_paths_exist(self):
        from dashboard.app import _BUNDLED_SCENARIOS
        for display_name, path in _BUNDLED_SCENARIOS.items():
            assert Path(path).exists(), (
                f"Bundled scenario '{display_name}' points to non-existent path: {path}"
            )

    def test_bundled_scenarios_non_empty(self):
        from dashboard.app import _BUNDLED_SCENARIOS
        assert len(_BUNDLED_SCENARIOS) >= 1

    def test_bundled_scenarios_have_three_entries(self):
        from dashboard.app import _BUNDLED_SCENARIOS
        assert len(_BUNDLED_SCENARIOS) == 3


# ---------------------------------------------------------------------------
# _STAGE_LABELS constant
# ---------------------------------------------------------------------------

class TestStageLabels:
    """Verify _STAGE_LABELS contains the expected pipeline stage names."""

    def test_detect_present(self):
        from dashboard.app import _STAGE_LABELS
        assert "Detect" in _STAGE_LABELS

    def test_diagnose_present(self):
        from dashboard.app import _STAGE_LABELS
        assert "Diagnose" in _STAGE_LABELS

    def test_generate_test_present(self):
        from dashboard.app import _STAGE_LABELS
        assert "Generate Test" in _STAGE_LABELS

    def test_fix_present(self):
        from dashboard.app import _STAGE_LABELS
        assert "Fix" in _STAGE_LABELS

    def test_verify_present(self):
        from dashboard.app import _STAGE_LABELS
        assert "Verify" in _STAGE_LABELS

    def test_five_stages(self):
        from dashboard.app import _STAGE_LABELS
        assert len(_STAGE_LABELS) == 5


# ---------------------------------------------------------------------------
# _STATUS_STYLE constant
# ---------------------------------------------------------------------------

class TestStatusStyle:
    """Verify _STATUS_STYLE covers all expected status values."""

    def test_pending_has_entry(self):
        from dashboard.app import _STATUS_STYLE
        assert "pending" in _STATUS_STYLE

    def test_in_progress_has_entry(self):
        from dashboard.app import _STATUS_STYLE
        assert "in_progress" in _STATUS_STYLE

    def test_done_has_entry(self):
        from dashboard.app import _STATUS_STYLE
        assert "done" in _STATUS_STYLE

    def test_failed_has_entry(self):
        from dashboard.app import _STATUS_STYLE
        assert "failed" in _STATUS_STYLE

    def test_each_entry_is_two_tuple(self):
        from dashboard.app import _STATUS_STYLE
        for key, value in _STATUS_STYLE.items():
            assert isinstance(value, tuple) and len(value) == 2, (
                f"_STATUS_STYLE['{key}'] should be a 2-tuple (emoji, colour)"
            )
