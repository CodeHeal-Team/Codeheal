"""
agents/placeholders.py
-----------------------
Placeholder agent implementations for Sub-Tasks 7 and 8.

These classes expose **exactly the same** ``run(request) -> str`` interface
as the real agents, but instead of calling watsonx.ai they read pre-written
JSON fixtures from ``samples/<scenario>/fixtures/``.

Usage in ``pipeline.py`` during development
--------------------------------------------
Replace real agent imports with placeholder imports:

.. code-block:: python

    # Development (Sub-Tasks 7 & 8)
    from agents.placeholders import (
        PlaceholderDiagnosticAgent as DiagnosticAgent,
        PlaceholderTestGeneratorAgent as TestGeneratorAgent,
        PlaceholderRefactoringAgent as RefactoringAgent,
    )

    # Production (Sub-Tasks 4-6 complete)
    from agents.diagnostic_agent import DiagnosticAgent
    from agents.test_generator_agent import TestGeneratorAgent
    from agents.refactoring_agent import RefactoringAgent

No other code changes are required when swapping in the real agents.

Fixture layout
--------------
Each scenario directory under ``samples/`` must contain a ``fixtures/``
subdirectory with three JSON files:

* ``diagnosis.json``       — raw JSON matching the ``DiagnosisResult`` schema
* ``generated_test.json``  — raw JSON matching the ``GeneratedTest`` schema
* ``patch.json``           — raw JSON matching the ``PatchResult`` schema

The placeholder agents return the raw file content as a string, just as a
real model would.  ``response_parser`` in Module 1 validates these strings
before use.

Supported scenario keys
-----------------------
* ``"none_bug"``
* ``"broken_api"``
* ``"off_by_one"``

The scenario is inferred from ``request``'s ``source_files`` keys: the
placeholder looks for the first key that contains a known scenario name.
If no known scenario can be identified, ``"none_bug"`` is used as the
safe fallback so the pipeline always has a valid response.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

from agents.base_agent import BaseAgent
from core.models import DiagnosisRequest, TestGenRequest, RefactorRequest

# ---------------------------------------------------------------------------
# Fixture resolution
# ---------------------------------------------------------------------------

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
_KNOWN_SCENARIOS = ("none_bug", "broken_api", "off_by_one")
_FALLBACK_SCENARIO = "none_bug"


# Maps a substring that uniquely identifies a scenario to its directory name.
# Checked against the joined source_files keys (lower-cased, slashes normalised).
_SCENARIO_FINGERPRINTS: tuple[tuple[str, str], ...] = (
    ("router",     "broken_api"),   # broken_api uses src/router.py
    ("list_util",  "off_by_one"),   # off_by_one uses src/list_utils.py
    ("calculator", "none_bug"),     # none_bug uses src/calculator.py
    # Fall-through scenario names embedded in directory-style paths
    ("broken_api", "broken_api"),
    ("off_by_one", "off_by_one"),
    ("none_bug",   "none_bug"),
)


def _infer_scenario(source_files: dict[str, str]) -> str:
    """
    Guess the active scenario from the keys and content in *source_files*.

    Strategy:
    1. Join all file path keys into a lower-cased string.
    2. Scan for characteristic substrings defined in ``_SCENARIO_FINGERPRINTS``.
    3. Return the first matching scenario name, or ``_FALLBACK_SCENARIO`` if
       none matches.

    Returns the scenario name (e.g. ``"none_bug"``) or the fallback.
    """
    joined = " ".join(source_files.keys()).lower().replace("\\", "/")
    for fingerprint, scenario in _SCENARIO_FINGERPRINTS:
        if fingerprint in joined:
            return scenario
    return _FALLBACK_SCENARIO


def _read_fixture(scenario: str, fixture_name: str) -> str:
    """
    Read and return the raw text of ``samples/<scenario>/fixtures/<fixture_name>``.

    Parameters
    ----------
    scenario:
        One of the known scenario directory names.
    fixture_name:
        File name inside the ``fixtures/`` directory, e.g. ``"diagnosis.json"``.

    Raises
    ------
    FileNotFoundError
        If the fixture file does not exist.
    """
    fixture_path = _SAMPLES_DIR / scenario / "fixtures" / fixture_name
    if not fixture_path.is_file():
        raise FileNotFoundError(
            f"Fixture not found: {fixture_path}. "
            f"Expected fixture files at samples/{scenario}/fixtures/"
        )
    return fixture_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Placeholder agents
# ---------------------------------------------------------------------------


class PlaceholderDiagnosticAgent(BaseAgent):
    """
    Returns the pre-written ``diagnosis.json`` fixture for the active scenario.

    Drop-in replacement for :class:`~agents.diagnostic_agent.DiagnosticAgent`.
    """

    def run(self, request: DiagnosisRequest) -> str:  # type: ignore[override]
        """
        Return the raw fixture JSON string for the inferred scenario.

        Parameters
        ----------
        request:
            :class:`~core.models.DiagnosisRequest` — only ``source_files``
            is inspected (to infer the scenario).

        Returns
        -------
        str
            Raw JSON string matching the ``DiagnosisResult`` schema.
        """
        scenario = _infer_scenario(request.source_files)
        return _read_fixture(scenario, "diagnosis.json")


class PlaceholderTestGeneratorAgent(BaseAgent):
    """
    Returns the pre-written ``generated_test.json`` fixture for the active
    scenario.

    Drop-in replacement for
    :class:`~agents.test_generator_agent.TestGeneratorAgent`.
    """

    def run(self, request: TestGenRequest) -> str:  # type: ignore[override]
        """
        Return the raw fixture JSON string for the inferred scenario.

        Parameters
        ----------
        request:
            :class:`~core.models.TestGenRequest` — ``source_files`` is used
            to infer the scenario; ``diagnosis`` is accepted but not used.

        Returns
        -------
        str
            Raw JSON string matching the ``GeneratedTest`` schema envelope.
        """
        scenario = _infer_scenario(request.source_files)
        return _read_fixture(scenario, "generated_test.json")


class PlaceholderRefactoringAgent(BaseAgent):
    """
    Returns the pre-written ``patch.json`` fixture for the active scenario.

    Drop-in replacement for
    :class:`~agents.refactoring_agent.RefactoringAgent`.
    """

    def run(self, request: RefactorRequest) -> str:  # type: ignore[override]
        """
        Return the raw fixture JSON string for the inferred scenario.

        Parameters
        ----------
        request:
            :class:`~core.models.RefactorRequest` — ``source_files`` is used
            to infer the scenario; ``diagnosis`` and ``generated_test`` are
            accepted but not used.

        Returns
        -------
        str
            Raw JSON string matching the ``PatchResult`` schema envelope.
        """
        scenario = _infer_scenario(request.source_files)
        return _read_fixture(scenario, "patch.json")
