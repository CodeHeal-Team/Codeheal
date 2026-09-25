"""
core/response_parser.py
------------------------
Robust parser and validator for raw agent responses.

Every agent returns a raw string (possibly wrapped in Markdown fences or
surrounded by prose).  This module normalises those strings into validated
dataclass instances before any pipeline stage stores or uses them.

Public API
----------
parse_diagnosis(raw, retry_fn) -> DiagnosisResult
parse_generated_test(raw, retry_fn) -> GeneratedTest
parse_patch(raw, retry_fn, repo_path) -> PatchResult

Each function:
1. Strips Markdown code fences.
2. Extracts the first ``{...}`` JSON object from the text.
3. Validates the required schema (keys + types).
4. Applies domain-specific validation (confidence clamping, diff headers).
5. On failure, calls ``retry_fn(error_message) -> str`` once and retries.
6. On second failure, raises ``ParseError`` with a human-readable message.

``retry_fn`` signature
----------------------
    retry_fn(error_message: str) -> str

The pipeline is responsible for wiring ``retry_fn`` to a new agent call that
includes the validation error in its prompt.  The parser only ever calls
``retry_fn`` once; the pipeline is not involved in the retry decision.

If ``retry_fn`` is ``None`` the parser still attempts parsing but raises
``ParseError`` immediately on any failure (no retry).
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional

from core.models import DiagnosisResult, GeneratedTest, PatchResult


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when parsing and optional retry both fail."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL | re.IGNORECASE,
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove leading/trailing Markdown code fences, returning inner text."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_first_json(text: str) -> str:
    """
    Return the first ``{...}`` block found in *text*.

    Uses a simple brace-counting scan so nested objects are handled correctly
    without relying on the regex engine's greediness.

    Raises
    ------
    ParseError
        If no ``{`` is found.
    """
    start = text.find("{")
    if start == -1:
        raise ParseError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ParseError("Unbalanced braces — no complete JSON object found in model output.")


def _parse_json(raw: str) -> dict:
    """Strip fences, extract first JSON object, and parse to dict."""
    cleaned = _strip_fences(raw)
    json_str = _extract_first_json(cleaned)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ParseError(f"JSON decode error: {exc}") from exc


def _require_keys(data: dict, required: dict[str, type | tuple]) -> None:
    """
    Validate that *data* contains all *required* keys with the right types.

    Parameters
    ----------
    data:
        Parsed JSON dict.
    required:
        Mapping of key name → expected Python type(s).

    Raises
    ------
    ParseError
        On the first missing key or wrong type.
    """
    for key, expected_type in required.items():
        if key not in data:
            raise ParseError(f"Missing required key: '{key}'")
        value = data[key]
        if not isinstance(value, expected_type):
            got = type(value).__name__
            if isinstance(expected_type, tuple):
                exp = " or ".join(t.__name__ for t in expected_type)
            else:
                exp = expected_type.__name__
            raise ParseError(
                f"Key '{key}' has wrong type: expected {exp}, got {got}"
            )


def _with_retry(
    raw: str,
    parse_fn: Callable[[str], object],
    retry_fn: Optional[Callable[[str], str]],
) -> object:
    """
    Attempt *parse_fn(raw)*.  On ``ParseError`` call ``retry_fn`` once and
    retry.  On second failure raise the ``ParseError``.

    If *retry_fn* is ``None``, raises immediately on the first failure.
    """
    try:
        return parse_fn(raw)
    except ParseError as first_err:
        if retry_fn is None:
            raise
        try:
            retry_raw = retry_fn(str(first_err))
            return parse_fn(retry_raw)
        except ParseError as second_err:
            raise ParseError(
                f"Parsing failed after retry. "
                f"First error: {first_err}. "
                f"Second error: {second_err}"
            ) from second_err


# ---------------------------------------------------------------------------
# DiagnosisResult parser
# ---------------------------------------------------------------------------

_DIAGNOSIS_SCHEMA: dict[str, type | tuple] = {
    "root_cause":    str,
    "affected_file": str,
    "affected_lines": list,
    "explanation":   str,
    "confidence":    (int, float),
}


def _parse_diagnosis_once(raw: str) -> DiagnosisResult:
    data = _parse_json(raw)
    _require_keys(data, _DIAGNOSIS_SCHEMA)

    # affected_lines must be a list of ints
    lines = data["affected_lines"]
    if not all(isinstance(n, int) for n in lines):
        raise ParseError("'affected_lines' must be a list of integers")

    # Clamp confidence to [0.0, 1.0]
    confidence = float(data["confidence"])
    confidence = max(0.0, min(1.0, confidence))

    return DiagnosisResult(
        root_cause=data["root_cause"],
        affected_file=data["affected_file"],
        affected_lines=lines,
        explanation=data["explanation"],
        confidence=confidence,
        raw_response=raw,
    )


def parse_diagnosis(
    raw: str,
    retry_fn: Optional[Callable[[str], str]] = None,
) -> DiagnosisResult:
    """
    Parse and validate a raw agent string into a :class:`~core.models.DiagnosisResult`.

    Parameters
    ----------
    raw:
        Raw string returned by the Diagnostic Agent.
    retry_fn:
        Optional callable that accepts an error message and returns a new raw
        response string.  Called at most once.

    Returns
    -------
    DiagnosisResult
        Validated, confidence-clamped result.

    Raises
    ------
    ParseError
        If parsing fails even after the optional retry.
    """
    return _with_retry(raw, _parse_diagnosis_once, retry_fn)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# GeneratedTest parser
# ---------------------------------------------------------------------------

_GENERATED_TEST_SCHEMA: dict[str, type | tuple] = {
    "filename": str,
    "code":     str,
}


def _parse_generated_test_once(raw: str) -> GeneratedTest:
    data = _parse_json(raw)
    _require_keys(data, _GENERATED_TEST_SCHEMA)
    return GeneratedTest(
        filename=data["filename"],
        code=data["code"],
    )


def parse_generated_test(
    raw: str,
    retry_fn: Optional[Callable[[str], str]] = None,
) -> GeneratedTest:
    """
    Parse and validate a raw agent string into a :class:`~core.models.GeneratedTest`.

    Raises
    ------
    ParseError
        If parsing fails even after the optional retry.
    """
    return _with_retry(raw, _parse_generated_test_once, retry_fn)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# PatchResult parser
# ---------------------------------------------------------------------------

_PATCH_SCHEMA: dict[str, type | tuple] = {
    "diff": str,
}

_DIFF_GIT_PREFIX = "diff --git"
_HUNK_MARKER_RE = re.compile(r"^@@", re.MULTILINE)


def _validate_diff(diff: str, repo_path: Optional[str]) -> None:
    """
    Check structural validity of a unified diff string.

    Raises
    ------
    ParseError
        If the diff header, hunk markers, or file path are invalid.
    """
    if not diff.lstrip().startswith(_DIFF_GIT_PREFIX):
        raise ParseError(
            f"Diff must start with '{_DIFF_GIT_PREFIX}'. "
            f"Got: {diff[:60]!r}"
        )

    if not _HUNK_MARKER_RE.search(diff):
        raise ParseError("Diff contains no '@@' hunk markers.")

    # Validate that the target file referenced in the diff header exists.
    if repo_path is not None:
        # Extract file path from "diff --git a/... b/..." — use the b/ side.
        header_match = re.search(
            r"^diff --git a/(.+?) b/(.+?)$", diff, re.MULTILINE
        )
        if header_match:
            target_rel = header_match.group(2)
            target_abs = os.path.join(repo_path, target_rel)
            if not os.path.isfile(target_abs):
                raise ParseError(
                    f"Diff targets '{target_rel}' but that file does not "
                    f"exist in repo '{repo_path}'."
                )


def _parse_patch_once(raw: str, repo_path: Optional[str] = None) -> PatchResult:
    data = _parse_json(raw)
    _require_keys(data, _PATCH_SCHEMA)
    diff = data["diff"]
    _validate_diff(diff, repo_path)
    return PatchResult(diff=diff, validated=True)


def parse_patch(
    raw: str,
    retry_fn: Optional[Callable[[str], str]] = None,
    repo_path: Optional[str] = None,
) -> PatchResult:
    """
    Parse and validate a raw agent string into a :class:`~core.models.PatchResult`.

    Parameters
    ----------
    raw:
        Raw string returned by the Refactoring Agent.
    retry_fn:
        Optional callable for one retry on failure.
    repo_path:
        Absolute path to the repo root.  When provided, the diff's target
        file path is checked against the real filesystem.

    Raises
    ------
    ParseError
        If parsing fails even after the optional retry.
    """
    parse_fn = lambda r: _parse_patch_once(r, repo_path)  # noqa: E731
    return _with_retry(raw, parse_fn, retry_fn)  # type: ignore[return-value]
