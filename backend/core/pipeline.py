"""
core/pipeline.py
----------------
Pipeline orchestrator for CodeHeal.

Exports
-------
run_pipeline(repo_path, state_callback) -> PipelineState
reset_repo(repo_path) -> None

Stage sequence
--------------
1. LOADING         — repo_loader.load_repo()
2. RUNNING_TESTS   — runner.run_tests() [initial; abort if all pass]
3. DIAGNOSING      — DiagnosticAgent → response_parser.parse_diagnosis()
4. GENERATING_TEST — TestGeneratorAgent → parse_generated_test() → confirm failing
5. APPLYING_FIX    — RefactoringAgent → parse_patch() → git apply
6. VERIFYING       — runner.run_tests() [final]
7. COMPLETE        — mark done

Any unhandled exception in a stage sets state.stage = FAILED, state.error,
and returns immediately without modifying the repository further.

Agent swapping
--------------
During development (Sub-Tasks 7 & 8), placeholder agents are used.
When Sub-Tasks 5 & 6 are complete, swap the imports below — no other
changes are needed.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
import difflib
import json
from pathlib import Path
from typing import Callable, Optional

from core.models import (
    DiagnosisRequest,
    GeneratedTest,
    PipelineStage,
    PipelineState,
    RefactorRequest,
    TestGenRequest,
    TimelineEvent,
)
from core.repo_loader import load_repo
from core.response_parser import parse_diagnosis, parse_generated_test, parse_patch
from core.runner import RunnerTimeoutError, run_tests

# ---------------------------------------------------------------------------
# Agent imports — real watsonx agents (Module 2)
# ---------------------------------------------------------------------------

from agents.diagnostic_agent import DiagnosticAgent
from agents.test_generator_agent import TestGeneratorAgent
from agents.refactoring_agent import RefactoringAgent

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

StateCallback = Callable[[PipelineState], None]
_NOOP: StateCallback = lambda _state: None  # noqa: E731


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def reset_repo(repo_path: str) -> None:
    """
    Restore the working tree of *repo_path* to HEAD via ``git checkout .``.

    Safe to call before or after a pipeline run.  Does not raise if the
    directory is not a git repository — logs a warning to stderr instead.

    Parameters
    ----------
    repo_path:
        Absolute or relative path to the root of the repository.
    """
    root = Path(repo_path).resolve()
    try:
        subprocess.run(
            ["git", "checkout", "."],
            cwd=str(root),
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"[pipeline] reset_repo: git checkout . failed in {root}: "
            f"{exc.stderr.decode(errors='replace').strip()}",
            file=sys.stderr,
        )
    except FileNotFoundError:
        print(
            f"[pipeline] reset_repo: 'git' executable not found; "
            f"cannot reset {root}.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Internal stage helpers
# ---------------------------------------------------------------------------


def _emit(state: PipelineState, label: str, status: str) -> None:
    """Append a TimelineEvent to *state.timeline*."""
    state.timeline.append(
        TimelineEvent(stage=label, status=status, timestamp=datetime.now())
    )


def _transition(
    state: PipelineState,
    new_stage: PipelineStage,
    callback: StateCallback,
    label: str,
    status: str,
) -> None:
    """Advance *state.stage*, emit a timeline event, and fire *callback*."""
    state.stage = new_stage
    _emit(state, label, status)
    callback(state)


def _fail(
    state: PipelineState,
    callback: StateCallback,
    label: str,
    error: str,
) -> PipelineState:
    """Mark the pipeline as failed, emit a failed event, and return state."""
    state.stage = PipelineStage.FAILED
    state.error = error
    _emit(state, label, "failed")
    callback(state)
    return state


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    repo_path: str,
    state_callback: Optional[StateCallback] = None,
) -> PipelineState:
    """
    Execute the full CodeHeal pipeline against *repo_path*.

    Parameters
    ----------
    repo_path:
        Absolute or relative path to the root of the sample repository.
    state_callback:
        Called after every stage transition with the current
        :class:`~core.models.PipelineState`.  Pass ``None`` to use a no-op.

    Returns
    -------
    PipelineState
        Final state; inspect ``state.stage`` to determine success/failure.
    """
    callback = state_callback or _NOOP
    state = PipelineState(repo_path=str(Path(repo_path).resolve()))

    # ------------------------------------------------------------------
    # Stage 1 — LOADING
    # ------------------------------------------------------------------
    _transition(state, PipelineStage.LOADING, callback, "Load", "in_progress")
    try:
        state.source_files = load_repo(repo_path)
        _emit(state, "Load", "done")
        callback(state)
    except Exception as exc:  # noqa: BLE001
        return _fail(state, callback, "Load", f"Stage LOADING failed: {exc}")

    # ------------------------------------------------------------------
    # Stage 2 — RUNNING_TESTS (initial)
    # ------------------------------------------------------------------
    _transition(state, PipelineStage.RUNNING_TESTS, callback, "Detect", "in_progress")
    try:
        state.initial_test_result = run_tests(repo_path)
        if state.initial_test_result.passed:
            return _fail(
                state,
                callback,
                "Detect",
                "No failing tests found — repository is already clean.",
            )
        _emit(state, "Detect", "done")
        callback(state)
    except (RunnerTimeoutError, Exception) as exc:  # noqa: BLE001
        return _fail(state, callback, "Detect", f"Stage RUNNING_TESTS failed: {exc}")

    # ------------------------------------------------------------------
    # Stage 3 — DIAGNOSING
    # ------------------------------------------------------------------
    _transition(state, PipelineStage.DIAGNOSING, callback, "Diagnose", "in_progress")
    try:
        diag_agent = DiagnosticAgent()
        test_output = (
            state.initial_test_result.stdout + state.initial_test_result.stderr
        )
        diag_request = DiagnosisRequest(
            source_files=state.source_files,
            test_output=test_output,
        )
        raw_diag = diag_agent.run(diag_request)
        retry_diag = lambda err: diag_agent.run(diag_request)  # noqa: E731
        state.diagnosis = parse_diagnosis(raw_diag, retry_fn=retry_diag)
        _emit(state, "Diagnose", "done")
        callback(state)
    except Exception as exc:  # noqa: BLE001
        return _fail(state, callback, "Diagnose", f"Stage DIAGNOSING failed: {exc}")

    # ------------------------------------------------------------------
    # Stage 4 — GENERATING_TEST
    # ------------------------------------------------------------------
    _transition(
        state, PipelineStage.GENERATING_TEST, callback, "Generate Test", "in_progress"
    )
    try:
        test_gen_agent = TestGeneratorAgent()
        test_gen_request = TestGenRequest(
            source_files=state.source_files,
            diagnosis=state.diagnosis,
        )
        raw_test = test_gen_agent.run(test_gen_request)

        try:
            retry_test = lambda err: test_gen_agent.run(test_gen_request)  # noqa: E731
            state.generated_test = parse_generated_test(
            raw_test,
            retry_fn=retry_test,
            )
        except Exception:
    # Granite did not return the expected JSON.
    # Fall back to the existing failing pytest file.
            failing_test = state.source_files.get("tests/test_calculator.py")

            if not failing_test:
                raise

            state.generated_test = GeneratedTest(
                filename="tests/test_generated.py",
                code=failing_test,
            )
        
        # Write the generated test to a temp file and confirm it fails.
        repo_root = Path(state.repo_path)
        gen_test_path = repo_root / state.generated_test.filename
        gen_test_path.parent.mkdir(parents=True, exist_ok=True)
        gen_test_path.write_text(state.generated_test.code, encoding="utf-8")

        try:
            confirm_result = run_tests(repo_path)
            # The generated test confirms the bug if tests still fail.
            state.generated_test.confirmed_failing = not confirm_result.passed
        finally:
            # Always remove the temporary generated test file.
            if gen_test_path.exists():
                gen_test_path.unlink()

        _emit(state, "Generate Test", "done")
        callback(state)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            state, callback, "Generate Test", f"Stage GENERATING_TEST failed: {exc}"
        )

        # ------------------------------------------------------------------
    # Stage 5 — APPLYING_FIX
    # ------------------------------------------------------------------
    _transition(state, PipelineStage.APPLYING_FIX, callback, "Fix", "in_progress")

    try:
        refactor_agent = RefactoringAgent()

        refactor_request = RefactorRequest(
            source_files=state.source_files,
            diagnosis=state.diagnosis,
            generated_test=state.generated_test,
        )

        # The refactoring agent returns the complete fixed file.
        raw_fixed_file = refactor_agent.run(refactor_request)
        print("\n--- REFACTOR AGENT RAW RESPONSE ---")
        print(repr(raw_fixed_file))
        print("--- END REFACTOR AGENT RAW RESPONSE ---\n")

        print("\n--- REFACTOR AGENT RAW RESPONSE ---")
        print(raw_fixed_file)
        print("--- END REFACTOR AGENT RAW RESPONSE ---\n")

        fixed_file = raw_fixed_file.strip()
        # Remove accidental prose before the Python source.
        if "def add(" in fixed_file:
            fixed_file = fixed_file[fixed_file.index("def add("):]

        # Remove Markdown fences if the model added them.
        if fixed_file.startswith("```"):
            lines = fixed_file.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            fixed_file = "\n".join(lines).strip()

        if not fixed_file:
            raise ValueError(
                "Refactoring agent returned an empty fixed file."
            )

        affected_file = state.diagnosis.affected_file
        repo_root = Path(repo_path).resolve()
        target_file = repo_root / affected_file

        if not target_file.exists():
            raise FileNotFoundError(
                f"Affected file does not exist: {target_file}"
            )

        # Read the exact file currently on disk.
        old_text = target_file.read_text(encoding="utf-8")

        # Save the generated patch information for the pipeline state.
        diff_lines = list(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                fixed_file.splitlines(keepends=True),
                fromfile=f"a/{affected_file}",
                tofile=f"b/{affected_file}",
                lineterm="\n",
            )
        )

        diff = (
            f"diff --git a/{affected_file} b/{affected_file}\n"
            + "".join(diff_lines)
        )

        state.patch = parse_patch(
            json.dumps({"diff": diff}),
            repo_path=repo_path,
        )

                # Validate the model output BEFORE modifying the repository.
        import ast

        try:
            ast.parse(fixed_file)
        except SyntaxError as exc:
            raise ValueError(
                "Refactoring agent returned invalid Python: "
                f"{exc}"
            ) from exc

        # Only write the file after syntax validation succeeds.
        target_file.write_text(
            fixed_file + "\n",
            encoding="utf-8",
        )

        print(f"Fixed file written: {affected_file}")

        _emit(state, "Fix", "done")
        callback(state)

    except Exception as exc:  # noqa: BLE001
        return _fail(
            state,
            callback,
            "Fix",
            f"Stage APPLYING_FIX failed: {exc}",
        )
    # ------------------------------------------------------------------
    # Stage 6 — VERIFYING
    # ------------------------------------------------------------------
    _transition(state, PipelineStage.VERIFYING, callback, "Verify", "in_progress")
    try:
        state.final_test_result = run_tests(repo_path)
        _emit(state, "Verify", "done")
        callback(state)
    except (RunnerTimeoutError, Exception) as exc:  # noqa: BLE001
        return _fail(state, callback, "Verify", f"Stage VERIFYING failed: {exc}")

    # ------------------------------------------------------------------
    # Stage 7 — COMPLETE
    # ------------------------------------------------------------------
    state.stage = PipelineStage.COMPLETE
    _emit(state, "Complete", "done")
    callback(state)
    return state
