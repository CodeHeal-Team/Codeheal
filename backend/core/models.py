"""
core/models.py
--------------
All shared data types for the CodeHeal pipeline.
This is the single contract between Module 1 (Core Brain) and Module 2 (Agent Intelligence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Pipeline stage enum
# ---------------------------------------------------------------------------

class PipelineStage(Enum):
    """Ordered stages that the pipeline passes through."""
    IDLE = "idle"
    LOADING = "loading"
    RUNNING_TESTS = "running_tests"
    DIAGNOSING = "diagnosing"
    GENERATING_TEST = "generating_test"
    APPLYING_FIX = "applying_fix"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """One entry in the event timeline displayed by the dashboard."""
    stage: str          # human-readable label, e.g. "Detect"
    status: str         # "pending" | "in_progress" | "done" | "failed"
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Test execution result
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Raw output from running pytest in a subprocess."""
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


# ---------------------------------------------------------------------------
# Agent output types
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    """Structured output from the Diagnostic Agent."""
    root_cause: str = ""          # one-sentence summary
    affected_file: str = ""       # relative path from repo root
    affected_lines: list[int] = field(default_factory=list)
    explanation: str = ""         # detailed explanation for the dashboard
    confidence: float = 0.0       # 0.0–1.0 (display only — does not gate pipeline)
    raw_response: str = ""        # preserved for debugging


@dataclass
class GeneratedTest:
    """Structured output from the Test Generator Agent."""
    filename: str = ""            # e.g. "tests/test_generated.py"
    code: str = ""                # complete pytest function source
    confirmed_failing: bool = False  # set by pipeline after running the test


@dataclass
class PatchResult:
    """Structured output from the Refactoring Agent."""
    diff: str = ""                # complete unified diff starting with "diff --git"
    validated: bool = False       # set by response_parser after header/hunk checks


# ---------------------------------------------------------------------------
# Agent request types
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisRequest:
    """Input to the Diagnostic Agent."""
    source_files: dict[str, str]  # filename → contents
    test_output: str              # combined stdout + stderr from the initial test run


@dataclass
class TestGenRequest:
    """Input to the Test Generator Agent."""
    source_files: dict[str, str]
    diagnosis: DiagnosisResult


@dataclass
class RefactorRequest:
    """Input to the Refactoring Agent."""
    source_files: dict[str, str]
    diagnosis: DiagnosisResult
    generated_test: GeneratedTest


# ---------------------------------------------------------------------------
# Top-level pipeline state
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """
    Single shared state object passed between all pipeline stages.
    Module 3 (Streamlit) reads this after every stage transition.
    """
    repo_path: str = ""
    source_files: dict[str, str] = field(default_factory=dict)
    stage: PipelineStage = PipelineStage.IDLE
    timeline: list[TimelineEvent] = field(default_factory=list)
    initial_test_result: TestResult = field(default_factory=TestResult)
    diagnosis: DiagnosisResult = field(default_factory=DiagnosisResult)
    generated_test: GeneratedTest = field(default_factory=GeneratedTest)
    patch: PatchResult = field(default_factory=PatchResult)
    final_test_result: TestResult = field(default_factory=TestResult)
    error: Optional[str] = None
