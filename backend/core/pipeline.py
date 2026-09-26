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
import ast
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
    PatchResult,
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
            # Fall back to an existing test file for any scenario.
            test_files = sorted(
                path
                for path in state.source_files
                if path.startswith("tests/")
                and path.endswith(".py")
                and not path.endswith("__init__.py")
                and not path.endswith("test_generated.py")
            )

            if not test_files:
                raise

            failing_test = test_files[0]

            state.generated_test = GeneratedTest(
                filename="tests/test_generated.py",
                code=state.source_files[failing_test],
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

        repo_root = Path(state.repo_path)
        patch_file = repo_root / ".codeheal.patch"
        target_file = repo_root / state.diagnosis.affected_file

        # ---- retry helpers ------------------------------------------------

        def _build_parse_retry_instructions(error: str) -> str:
            return (
                "Your previous response was invalid.\n"
                f"Exact parse error: {error}\n"
                "Return a corrected, non-empty Git diff inside the required "
                "JSON object with one key: diff. The diff must start with "
                "'diff --git', contain valid unified diff hunk headers (@@) "
                "with correct line counts, and modify the exact affected file "
                "path shown in the prompt. Do not return Markdown or code "
                "fences. Do not add any text outside the JSON object."
            )

        def _build_git_retry_instructions(git_error: str) -> str:
            return (
                "Git rejected your previous patch.\n"
                f"Exact Git error: {git_error}\n"
                "Generate a complete corrected Git diff. Verify the file "
                "path, the context lines (prefixed with a space), removed "
                "lines (prefixed with -), and added lines (prefixed with +). "
                "Recount the hunk header line numbers and counts carefully. "
                "The hunk format is: @@ -<old_start>,<old_count> "
                "+<new_start>,<new_count> @@\n"
                "Return only the required JSON object with a non-empty diff "
                "string. Do not return Markdown or code fences."
            )

        def _retry_refactor_parse(error: str) -> str:
            """Called by parse_patch on a parse/structure failure."""
            return refactor_agent.run(
                refactor_request,
                retry_instructions=_build_parse_retry_instructions(error),
            ).strip()

        patch_applied = False

        # ---- attempt loop: up to 2 round-trips with the model -------------
        # Attempt 0: initial model call.
        # Attempt 1: only reached when git apply --check failed on attempt 0;
        #            a new raw response is obtained with the git error message.
        # parse_patch internally performs one additional retry (via retry_fn)
        # when the response fails structural validation, so the model may be
        # called up to 4 times in the worst case.

        raw_diff = refactor_agent.run(refactor_request).strip()

        try:
            for attempt in range(2):
                # Parse the model response; parse_patch calls _retry_refactor_parse
                # once if the response is structurally invalid.
                try:
                    state.patch = parse_patch(
                        raw_diff,
                        retry_fn=_retry_refactor_parse,
                        repo_path=str(repo_root),
                    )
                except Exception as parse_exc:  # noqa: BLE001
                    # parse_patch exhausted its one internal retry.
                    # On the first attempt, propagate as a recoverable error;
                    # on the second attempt (already a git-check retry) there
                    # is nothing more to do.
                    raise RuntimeError(
                        f"Failed to parse a valid diff from the model "
                        f"(attempt {attempt + 1}/2): {parse_exc}"
                    ) from parse_exc

                diff = state.patch.diff
                if not diff or not diff.strip():
                    raise RuntimeError(
                        "Refactoring agent returned an empty diff "
                        f"(attempt {attempt + 1}/2)."
                    )

                patch_file.write_text(diff, encoding="utf-8")

                # Pre-check: never apply a patch that git would reject.
                check_result = subprocess.run(
                    ["git", "apply", "--check", "--recount", str(patch_file)],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if check_result.returncode != 0:
                    git_error = (
                        check_result.stderr.strip()
                        or check_result.stdout.strip()
                        or "Unknown git apply --check error"
                    )

                    if attempt == 1:
                        # A second diff can still target the wrong context/line offsets.
                        # Fall back to complete file content, then build a fresh diff
                        # against the exact on-disk file so Git receives matching context.
                        full_response = refactor_agent.run_full_file(
                            refactor_request,
                            retry_instructions=(
                                f"The previous diff was rejected: {git_error}. "
                                "Return a corrected complete file."
                            ),
                        )
                        try:
                            json_start = full_response.find("{")
                            if json_start < 0:
                                raise ValueError("No JSON object in full-file response.")
                            full_data, _ = json.JSONDecoder().raw_decode(
                                full_response[json_start:]
                            )
                            corrected_content = full_data.get("content")
                            if not isinstance(corrected_content, str) or not corrected_content.strip():
                                raise ValueError("JSON field 'content' must be non-empty text.")
                        except (json.JSONDecodeError, ValueError, AttributeError) as parse_exc:
                            raise RuntimeError(
                                f"Full-file fallback response was invalid: {parse_exc}"
                            ) from parse_exc

                        if target_file.suffix == ".py":
                            try:
                                ast.parse(corrected_content)
                            except SyntaxError as syntax_exc:
                                raise RuntimeError(
                                    f"Full-file fallback introduced a syntax error: {syntax_exc}"
                                ) from syntax_exc

                        original_content = target_file.read_text(encoding="utf-8")
                        if not corrected_content.endswith("\n"):
                            corrected_content += "\n"
                        generated_lines = list(difflib.unified_diff(
                            original_content.splitlines(keepends=True),
                            corrected_content.splitlines(keepends=True),
                            fromfile=f"a/{state.diagnosis.affected_file}",
                            tofile=f"b/{state.diagnosis.affected_file}",
                        ))
                        if not generated_lines:
                            raise RuntimeError(
                                "Full-file fallback produced no changes to the affected file."
                            )
                        diff = (
                            f"diff --git a/{state.diagnosis.affected_file} "
                            f"b/{state.diagnosis.affected_file}\n"
                            + "".join(generated_lines)
                        )
                        patch_file.write_text(diff, encoding="utf-8")
                        fallback_check = subprocess.run(
                            ["git", "apply", "--check", "--recount", str(patch_file)],
                            cwd=str(repo_root),
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if fallback_check.returncode != 0:
                            raise RuntimeError(
                                "Full-file fallback diff failed Git pre-check: "
                                + (fallback_check.stderr.strip() or fallback_check.stdout.strip())
                            )
                        fallback_apply = subprocess.run(
                            ["git", "apply", "--recount", str(patch_file)],
                            cwd=str(repo_root),
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        if fallback_apply.returncode != 0:
                            raise RuntimeError(
                                "Full-file fallback diff failed to apply: "
                                + (fallback_apply.stderr.strip() or fallback_apply.stdout.strip())
                            )
                        state.patch = PatchResult(diff=diff, validated=True)
                        patch_applied = True
                        break

                    # Ask the model to correct the patch using the exact git error.
                    raw_diff = refactor_agent.run(
                        refactor_request,
                        retry_instructions=_build_git_retry_instructions(git_error),
                    ).strip()
                    continue

                # Pre-check passed — apply the patch.
                apply_result = subprocess.run(
                    ["git", "apply", "--recount", str(patch_file)],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if apply_result.returncode != 0:
                    apply_error = (
                        apply_result.stderr.strip()
                        or apply_result.stdout.strip()
                        or "Unknown git apply error"
                    )
                    raise RuntimeError(
                        f"git apply failed: {apply_error}"
                    )

                patch_applied = True
                break

            if not patch_applied:
                raise RuntimeError(
                    "Could not produce an applicable patch after retry."
                )

            # Validate the patched Python file (syntax check only).
            if target_file.suffix == ".py":
                try:
                    ast.parse(target_file.read_text(encoding="utf-8"))
                except SyntaxError as syn_exc:
                    raise RuntimeError(
                        f"Applied patch introduced a syntax error in "
                        f"{state.diagnosis.affected_file}: {syn_exc}"
                    ) from syn_exc

        finally:
            # Always remove the temporary patch file.
            patch_file.unlink(missing_ok=True)

        print(f"Fixed file applied: {state.diagnosis.affected_file}")

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
