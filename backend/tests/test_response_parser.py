"""
tests/test_response_parser.py
------------------------------
Focused tests for core/response_parser.py.

Coverage matrix
---------------
+---------------------------+-------------------------------------------+
| Scenario                  | Tests                                     |
+---------------------------+-------------------------------------------+
| Valid bare JSON           | all three parsers accept clean JSON       |
| Fenced JSON (```json)     | fences stripped before parsing            |
| Fenced JSON (```)         | fences stripped before parsing            |
| Prose + JSON              | first JSON object extracted from prose    |
| Malformed output          | ParseError raised                         |
| Schema errors             | missing key, wrong type                   |
| Invalid confidence        | clamped low / high; non-numeric rejected  |
| Invalid diffs             | missing header, missing hunk              |
| File path validation      | target file missing in repo               |
| Retry behaviour           | retry_fn called once; success on retry    |
| Retry still fails         | ParseError raised after both attempts     |
| No retry_fn               | ParseError raised immediately             |
+---------------------------+-------------------------------------------+
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.response_parser import (
    ParseError,
    parse_diagnosis,
    parse_generated_test,
    parse_patch,
    _strip_fences,
    _extract_first_json,
)
from core.models import DiagnosisResult, GeneratedTest, PatchResult


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _valid_diagnosis_dict(**overrides) -> dict:
    base = {
        "root_cause": "divide() does not guard against None",
        "affected_file": "src/calculator.py",
        "affected_lines": [13, 14],
        "explanation": "A None value causes a TypeError.",
        "confidence": 0.92,
    }
    base.update(overrides)
    return base


def _valid_diagnosis_json(**overrides) -> str:
    return json.dumps(_valid_diagnosis_dict(**overrides))


def _valid_gen_test_dict(**overrides) -> dict:
    base = {
        "filename": "tests/test_generated.py",
        "code": "import pytest\ndef test_divide_none():\n    pass\n",
    }
    base.update(overrides)
    return base


def _valid_gen_test_json(**overrides) -> str:
    return json.dumps(_valid_gen_test_dict(**overrides))


_VALID_DIFF = (
    "diff --git a/src/calculator.py b/src/calculator.py\n"
    "index 0000001..0000002 100644\n"
    "--- a/src/calculator.py\n"
    "+++ b/src/calculator.py\n"
    "@@ -12,4 +12,6 @@ def add(a, b):\n"
    " \n"
    " def divide(a, b):\n"
    "-    return a / b\n"
    "+    if a is None or b is None:\n"
    "+        raise ValueError('None not allowed')\n"
    "+    return a / b\n"
)


def _valid_patch_json(**overrides) -> str:
    base = {"diff": _VALID_DIFF}
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# Internal helper unit tests
# ---------------------------------------------------------------------------

class TestStripFences:
    """Tests for the _strip_fences internal helper."""

    def test_no_fences_unchanged(self):
        raw = '{"a": 1}'
        assert _strip_fences(raw) == raw

    def test_json_fence_stripped(self):
        raw = "```json\n{\"a\": 1}\n```"
        assert _strip_fences(raw) == '{"a": 1}'

    def test_plain_fence_stripped(self):
        raw = "```\n{\"a\": 1}\n```"
        assert _strip_fences(raw) == '{"a": 1}'

    def test_uppercase_json_fence_stripped(self):
        raw = "```JSON\n{\"a\": 1}\n```"
        assert _strip_fences(raw) == '{"a": 1}'

    def test_surrounding_whitespace_stripped(self):
        raw = "  ```json\n  {\"a\": 1}  \n```  "
        result = _strip_fences(raw)
        assert result == '{"a": 1}'

    def test_empty_string_unchanged(self):
        assert _strip_fences("") == ""


class TestExtractFirstJson:
    """Tests for the _extract_first_json internal helper."""

    def test_bare_object_returned(self):
        assert _extract_first_json('{"x": 1}') == '{"x": 1}'

    def test_prose_before_object(self):
        result = _extract_first_json('Here is the result: {"x": 1}')
        assert result == '{"x": 1}'

    def test_prose_after_object(self):
        result = _extract_first_json('{"x": 1} and some trailing text')
        assert result == '{"x": 1}'

    def test_nested_objects(self):
        raw = '{"outer": {"inner": 42}}'
        assert _extract_first_json(raw) == raw

    def test_no_brace_raises(self):
        with pytest.raises(ParseError, match="No JSON object found"):
            _extract_first_json("no braces here")

    def test_unbalanced_raises(self):
        with pytest.raises(ParseError, match="Unbalanced"):
            _extract_first_json('{"unclosed": 1')

    def test_string_with_braces_not_confused(self):
        # Braces inside a string value should not confuse the counter.
        raw = '{"text": "contains { and }"}'
        result = _extract_first_json(raw)
        assert result == raw


# ---------------------------------------------------------------------------
# parse_diagnosis tests
# ---------------------------------------------------------------------------

class TestParseDiagnosis:

    # -- Valid JSON --

    def test_valid_bare_json_returns_diagnosis_result(self):
        result = parse_diagnosis(_valid_diagnosis_json())
        assert isinstance(result, DiagnosisResult)

    def test_valid_json_fields_mapped_correctly(self):
        result = parse_diagnosis(_valid_diagnosis_json())
        assert result.root_cause == "divide() does not guard against None"
        assert result.affected_file == "src/calculator.py"
        assert result.affected_lines == [13, 14]
        assert result.explanation == "A None value causes a TypeError."
        assert result.confidence == pytest.approx(0.92)

    def test_raw_response_preserved(self):
        raw = _valid_diagnosis_json()
        result = parse_diagnosis(raw)
        assert result.raw_response == raw

    # -- Fenced JSON --

    def test_json_fenced_json_accepted(self):
        raw = "```json\n" + _valid_diagnosis_json() + "\n```"
        result = parse_diagnosis(raw)
        assert isinstance(result, DiagnosisResult)
        assert result.affected_file == "src/calculator.py"

    def test_plain_fenced_json_accepted(self):
        raw = "```\n" + _valid_diagnosis_json() + "\n```"
        result = parse_diagnosis(raw)
        assert isinstance(result, DiagnosisResult)

    # -- Prose surrounding JSON --

    def test_prose_before_json_accepted(self):
        raw = "Here is the diagnosis result:\n" + _valid_diagnosis_json()
        result = parse_diagnosis(raw)
        assert isinstance(result, DiagnosisResult)

    def test_prose_after_json_accepted(self):
        raw = _valid_diagnosis_json() + "\nPlease let me know if you need more information."
        result = parse_diagnosis(raw)
        assert isinstance(result, DiagnosisResult)

    # -- Malformed output --

    def test_empty_string_raises(self):
        with pytest.raises(ParseError):
            parse_diagnosis("")

    def test_plain_text_raises(self):
        with pytest.raises(ParseError):
            parse_diagnosis("The bug is in calculator.py line 14.")

    def test_invalid_json_syntax_raises(self):
        with pytest.raises(ParseError):
            parse_diagnosis("{root_cause: not valid json}")

    # -- Schema validation --

    def test_missing_root_cause_raises(self):
        d = _valid_diagnosis_dict()
        del d["root_cause"]
        with pytest.raises(ParseError, match="root_cause"):
            parse_diagnosis(json.dumps(d))

    def test_missing_affected_file_raises(self):
        d = _valid_diagnosis_dict()
        del d["affected_file"]
        with pytest.raises(ParseError, match="affected_file"):
            parse_diagnosis(json.dumps(d))

    def test_missing_affected_lines_raises(self):
        d = _valid_diagnosis_dict()
        del d["affected_lines"]
        with pytest.raises(ParseError, match="affected_lines"):
            parse_diagnosis(json.dumps(d))

    def test_missing_explanation_raises(self):
        d = _valid_diagnosis_dict()
        del d["explanation"]
        with pytest.raises(ParseError, match="explanation"):
            parse_diagnosis(json.dumps(d))

    def test_missing_confidence_raises(self):
        d = _valid_diagnosis_dict()
        del d["confidence"]
        with pytest.raises(ParseError, match="confidence"):
            parse_diagnosis(json.dumps(d))

    def test_root_cause_wrong_type_raises(self):
        with pytest.raises(ParseError, match="root_cause"):
            parse_diagnosis(_valid_diagnosis_json(root_cause=123))

    def test_affected_file_wrong_type_raises(self):
        with pytest.raises(ParseError, match="affected_file"):
            parse_diagnosis(_valid_diagnosis_json(affected_file=["src/calc.py"]))

    def test_affected_lines_wrong_type_raises(self):
        with pytest.raises(ParseError, match="affected_lines"):
            parse_diagnosis(_valid_diagnosis_json(affected_lines="13,14"))

    def test_affected_lines_non_int_elements_raises(self):
        with pytest.raises(ParseError, match="affected_lines"):
            parse_diagnosis(_valid_diagnosis_json(affected_lines=[13.5, 14.0]))

    # -- Confidence clamping --

    def test_confidence_above_one_clamped(self):
        result = parse_diagnosis(_valid_diagnosis_json(confidence=1.5))
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_below_zero_clamped(self):
        result = parse_diagnosis(_valid_diagnosis_json(confidence=-0.3))
        assert result.confidence == pytest.approx(0.0)

    def test_confidence_exactly_zero_accepted(self):
        result = parse_diagnosis(_valid_diagnosis_json(confidence=0.0))
        assert result.confidence == pytest.approx(0.0)

    def test_confidence_exactly_one_accepted(self):
        result = parse_diagnosis(_valid_diagnosis_json(confidence=1.0))
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_as_int_accepted(self):
        # JSON ints (e.g. 1) are valid for confidence.
        result = parse_diagnosis(_valid_diagnosis_json(confidence=1))
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_non_numeric_raises(self):
        with pytest.raises(ParseError, match="confidence"):
            parse_diagnosis(_valid_diagnosis_json(confidence="high"))


# ---------------------------------------------------------------------------
# parse_generated_test tests
# ---------------------------------------------------------------------------

class TestParseGeneratedTest:

    # -- Valid JSON --

    def test_valid_bare_json_returns_generated_test(self):
        result = parse_generated_test(_valid_gen_test_json())
        assert isinstance(result, GeneratedTest)

    def test_valid_json_fields_mapped(self):
        result = parse_generated_test(_valid_gen_test_json())
        assert result.filename == "tests/test_generated.py"
        assert "def test_divide_none" in result.code

    def test_confirmed_failing_defaults_false(self):
        result = parse_generated_test(_valid_gen_test_json())
        assert result.confirmed_failing is False

    # -- Fenced JSON --

    def test_json_fenced_accepted(self):
        raw = "```json\n" + _valid_gen_test_json() + "\n```"
        result = parse_generated_test(raw)
        assert isinstance(result, GeneratedTest)

    def test_plain_fenced_accepted(self):
        raw = "```\n" + _valid_gen_test_json() + "\n```"
        result = parse_generated_test(raw)
        assert isinstance(result, GeneratedTest)

    # -- Prose surrounding JSON --

    def test_prose_surrounding_accepted(self):
        raw = "Here is the test I generated:\n" + _valid_gen_test_json() + "\nDone."
        result = parse_generated_test(raw)
        assert isinstance(result, GeneratedTest)

    # -- Malformed output --

    def test_empty_string_raises(self):
        with pytest.raises(ParseError):
            parse_generated_test("")

    def test_non_json_raises(self):
        with pytest.raises(ParseError):
            parse_generated_test("def test_foo(): pass")

    # -- Schema validation --

    def test_missing_filename_raises(self):
        d = _valid_gen_test_dict()
        del d["filename"]
        with pytest.raises(ParseError, match="filename"):
            parse_generated_test(json.dumps(d))

    def test_missing_code_raises(self):
        d = _valid_gen_test_dict()
        del d["code"]
        with pytest.raises(ParseError, match="code"):
            parse_generated_test(json.dumps(d))

    def test_filename_wrong_type_raises(self):
        with pytest.raises(ParseError, match="filename"):
            parse_generated_test(_valid_gen_test_json(filename=42))

    def test_code_wrong_type_raises(self):
        with pytest.raises(ParseError, match="code"):
            parse_generated_test(_valid_gen_test_json(code={"nested": "obj"}))


# ---------------------------------------------------------------------------
# parse_patch tests
# ---------------------------------------------------------------------------

class TestParsePatch:

    # -- Valid JSON --

    def test_valid_bare_json_returns_patch_result(self):
        result = parse_patch(_valid_patch_json())
        assert isinstance(result, PatchResult)

    def test_valid_json_diff_stored(self):
        result = parse_patch(_valid_patch_json())
        assert result.diff == _VALID_DIFF

    def test_validated_flag_set_true(self):
        result = parse_patch(_valid_patch_json())
        assert result.validated is True

    # -- Fenced JSON --

    def test_json_fenced_accepted(self):
        raw = "```json\n" + _valid_patch_json() + "\n```"
        result = parse_patch(raw)
        assert isinstance(result, PatchResult)

    def test_plain_fenced_accepted(self):
        raw = "```\n" + _valid_patch_json() + "\n```"
        result = parse_patch(raw)
        assert isinstance(result, PatchResult)

    # -- Malformed output --

    def test_empty_string_raises(self):
        with pytest.raises(ParseError):
            parse_patch("")

    def test_non_json_raises(self):
        with pytest.raises(ParseError):
            parse_patch("just a patch without JSON")

    # -- Schema validation --

    def test_missing_diff_key_raises(self):
        with pytest.raises(ParseError, match="diff"):
            parse_patch(json.dumps({}))

    def test_diff_wrong_type_raises(self):
        with pytest.raises(ParseError, match="diff"):
            parse_patch(json.dumps({"diff": 123}))

    # -- Invalid diffs --

    def test_diff_without_header_raises(self):
        bad_diff = _VALID_DIFF.replace("diff --git", "--- ")
        with pytest.raises(ParseError, match="diff --git"):
            parse_patch(json.dumps({"diff": bad_diff}))

    def test_diff_without_hunk_marker_raises(self):
        bad_diff = _VALID_DIFF.replace("@@", "##")
        with pytest.raises(ParseError, match="hunk markers"):
            parse_patch(json.dumps({"diff": bad_diff}))

    def test_diff_with_leading_whitespace_before_header_ok(self):
        # Some models emit a newline before the diff --git line.
        raw_diff = "\n" + _VALID_DIFF
        result = parse_patch(json.dumps({"diff": raw_diff}))
        assert result.validated is True

    # -- File path validation with repo_path --

    def test_valid_diff_with_existing_file_accepted(self, tmp_path):
        # Create the file the diff references.
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "calculator.py").write_text("def divide(a, b): return a/b\n")
        result = parse_patch(_valid_patch_json(), repo_path=str(tmp_path))
        assert result.validated is True

    def test_diff_target_missing_in_repo_raises(self, tmp_path):
        # Diff references src/calculator.py but that file does not exist.
        with pytest.raises(ParseError, match="does not exist"):
            parse_patch(_valid_patch_json(), repo_path=str(tmp_path))

    def test_no_repo_path_skips_file_check(self):
        # Without repo_path, the file-existence check is skipped.
        result = parse_patch(_valid_patch_json(), repo_path=None)
        assert result.validated is True


# ---------------------------------------------------------------------------
# Retry behaviour tests
# ---------------------------------------------------------------------------

class TestRetryBehaviour:
    """
    Verify that each parser calls retry_fn exactly once on failure and
    succeeds when retry_fn returns valid JSON.
    """

    # ---- parse_diagnosis ----

    def test_diagnosis_retry_fn_called_on_failure(self):
        retry_fn = MagicMock(return_value=_valid_diagnosis_json())
        parse_diagnosis("not json", retry_fn=retry_fn)
        retry_fn.assert_called_once()

    def test_diagnosis_retry_fn_receives_error_message(self):
        retry_fn = MagicMock(return_value=_valid_diagnosis_json())
        parse_diagnosis("no json here", retry_fn=retry_fn)
        error_arg = retry_fn.call_args[0][0]
        assert isinstance(error_arg, str) and len(error_arg) > 0

    def test_diagnosis_retry_success_returns_result(self):
        retry_fn = MagicMock(return_value=_valid_diagnosis_json())
        result = parse_diagnosis("bad input", retry_fn=retry_fn)
        assert isinstance(result, DiagnosisResult)

    def test_diagnosis_retry_not_called_on_success(self):
        retry_fn = MagicMock()
        parse_diagnosis(_valid_diagnosis_json(), retry_fn=retry_fn)
        retry_fn.assert_not_called()

    def test_diagnosis_retry_raises_if_second_attempt_also_fails(self):
        retry_fn = MagicMock(return_value="still not json")
        with pytest.raises(ParseError, match="Parsing failed after retry"):
            parse_diagnosis("bad input", retry_fn=retry_fn)

    def test_diagnosis_retry_fn_called_at_most_once(self):
        retry_fn = MagicMock(return_value="still not json")
        with pytest.raises(ParseError):
            parse_diagnosis("bad input", retry_fn=retry_fn)
        assert retry_fn.call_count == 1

    def test_diagnosis_no_retry_fn_raises_immediately(self):
        with pytest.raises(ParseError):
            parse_diagnosis("bad input", retry_fn=None)

    # ---- parse_generated_test ----

    def test_gen_test_retry_fn_called_on_failure(self):
        retry_fn = MagicMock(return_value=_valid_gen_test_json())
        parse_generated_test("not json", retry_fn=retry_fn)
        retry_fn.assert_called_once()

    def test_gen_test_retry_success_returns_result(self):
        retry_fn = MagicMock(return_value=_valid_gen_test_json())
        result = parse_generated_test("bad input", retry_fn=retry_fn)
        assert isinstance(result, GeneratedTest)

    def test_gen_test_retry_raises_after_second_failure(self):
        retry_fn = MagicMock(return_value="still not json")
        with pytest.raises(ParseError, match="Parsing failed after retry"):
            parse_generated_test("bad input", retry_fn=retry_fn)

    def test_gen_test_retry_fn_called_at_most_once(self):
        retry_fn = MagicMock(return_value="still not json")
        with pytest.raises(ParseError):
            parse_generated_test("bad input", retry_fn=retry_fn)
        assert retry_fn.call_count == 1

    def test_gen_test_no_retry_fn_raises_immediately(self):
        with pytest.raises(ParseError):
            parse_generated_test("bad input", retry_fn=None)

    # ---- parse_patch ----

    def test_patch_retry_fn_called_on_failure(self):
        retry_fn = MagicMock(return_value=_valid_patch_json())
        parse_patch("not json", retry_fn=retry_fn)
        retry_fn.assert_called_once()

    def test_patch_retry_success_returns_result(self):
        retry_fn = MagicMock(return_value=_valid_patch_json())
        result = parse_patch("bad input", retry_fn=retry_fn)
        assert isinstance(result, PatchResult)

    def test_patch_retry_raises_after_second_failure(self):
        retry_fn = MagicMock(return_value="still not json")
        with pytest.raises(ParseError, match="Parsing failed after retry"):
            parse_patch("bad input", retry_fn=retry_fn)

    def test_patch_retry_fn_called_at_most_once(self):
        retry_fn = MagicMock(return_value="still not json")
        with pytest.raises(ParseError):
            parse_patch("bad input", retry_fn=retry_fn)
        assert retry_fn.call_count == 1

    def test_patch_no_retry_fn_raises_immediately(self):
        with pytest.raises(ParseError):
            parse_patch("bad input", retry_fn=None)

    # -- Retry with schema error (not just JSON decode error) --

    def test_diagnosis_retry_called_on_schema_error(self):
        # The JSON parses fine but is missing required keys.
        bad_schema_json = json.dumps({"root_cause": "x"})
        retry_fn = MagicMock(return_value=_valid_diagnosis_json())
        result = parse_diagnosis(bad_schema_json, retry_fn=retry_fn)
        retry_fn.assert_called_once()
        assert isinstance(result, DiagnosisResult)

    def test_patch_retry_called_on_invalid_diff(self):
        bad_diff = json.dumps({"diff": "not a real diff"})
        retry_fn = MagicMock(return_value=_valid_patch_json())
        result = parse_patch(bad_diff, retry_fn=retry_fn)
        retry_fn.assert_called_once()
        assert isinstance(result, PatchResult)


# ---------------------------------------------------------------------------
# Real fixture round-trip tests (uses samples/ fixtures)
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


class TestFixtureRoundTrip:
    """Verify that the fixture files satisfy the parser without errors."""

    @pytest.mark.parametrize("scenario", ["none_bug", "broken_api", "off_by_one"])
    def test_diagnosis_fixture_parses(self, scenario):
        fixture = (SAMPLES_DIR / scenario / "fixtures" / "diagnosis.json").read_text()
        result = parse_diagnosis(fixture)
        assert isinstance(result, DiagnosisResult)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.parametrize("scenario", ["none_bug", "broken_api", "off_by_one"])
    def test_generated_test_fixture_parses(self, scenario):
        fixture = (SAMPLES_DIR / scenario / "fixtures" / "generated_test.json").read_text()
        result = parse_generated_test(fixture)
        assert isinstance(result, GeneratedTest)
        assert result.filename.endswith(".py")

    @pytest.mark.parametrize("scenario", ["none_bug", "broken_api", "off_by_one"])
    def test_patch_fixture_parses(self, scenario):
        fixture = (SAMPLES_DIR / scenario / "fixtures" / "patch.json").read_text()
        result = parse_patch(fixture)
        assert isinstance(result, PatchResult)
        assert result.validated is True

    @pytest.mark.parametrize("scenario", ["none_bug", "broken_api", "off_by_one"])
    def test_patch_fixture_diff_starts_with_diff_git(self, scenario):
        fixture = (SAMPLES_DIR / scenario / "fixtures" / "patch.json").read_text()
        result = parse_patch(fixture)
        assert result.diff.lstrip().startswith("diff --git")

    @pytest.mark.parametrize("scenario", ["none_bug", "broken_api", "off_by_one"])
    def test_patch_fixture_diff_contains_hunk(self, scenario):
        fixture = (SAMPLES_DIR / scenario / "fixtures" / "patch.json").read_text()
        result = parse_patch(fixture)
        assert "@@" in result.diff
