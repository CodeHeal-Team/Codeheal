"""
core/runner.py
--------------
Executes ``pytest`` as a subprocess inside a given directory and returns a
:class:`~core.models.TestResult` capturing stdout, stderr, exit code, and the
derived pass/fail status.

Design notes
~~~~~~~~~~~~
* Uses :func:`subprocess.run` with ``capture_output=True`` so both streams are
  captured without leaking to the terminal.
* Hard timeout of 30 s — raises :class:`RunnerTimeoutError` so callers can
  decide how to surface it rather than hanging indefinitely.
* The working directory is set to *repo_path* so relative imports inside sample
  projects resolve correctly.
* ``sys.executable`` is used instead of a bare ``"python"`` / ``"pytest"`` so
  the correct interpreter (and its site-packages) are always selected.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from core.models import TestResult


class RunnerTimeoutError(Exception):
    """Raised when pytest exceeds the allowed wall-clock timeout."""


# Maximum seconds pytest is allowed to run before we kill it.
TIMEOUT_SECONDS: int = 30


def run_tests(repo_path: str, timeout: int = TIMEOUT_SECONDS) -> TestResult:
    """Run ``pytest tests/`` inside *repo_path* and return a :class:`TestResult`.

    Parameters
    ----------
    repo_path:
        Absolute or relative path to the root of the repository.  pytest is
        invoked with this directory as its working directory so that
        ``tests/`` is a path relative to the repo root.
    timeout:
        Maximum seconds pytest is allowed to run.  Defaults to
        :data:`TIMEOUT_SECONDS` (30 s).  Pass a larger value in environments
        with slow interpreter startup.

    Returns
    -------
    TestResult
        ``passed`` is ``True`` only when pytest exits with code ``0``.

    Raises
    ------
    FileNotFoundError
        If *repo_path* does not exist or is not a directory.
    RunnerTimeoutError
        If pytest does not finish within *timeout* seconds.
    """
    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path is not a directory: {repo_path}")

    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunnerTimeoutError(
            f"pytest timed out after {timeout} s in {repo_path}"
        ) from exc

    return TestResult(
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
    )
