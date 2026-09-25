"""
core/repo_loader.py
-------------------
Reads all Python source files from a repository directory and returns them
as a dict mapping relative filename to file contents.

Skips: __pycache__, .git, and any directory whose name starts with a dot.
Only files with the .py extension are included.
"""

from __future__ import annotations

import os
from pathlib import Path

# Directories to skip entirely (by name, at any depth).
_SKIP_DIRS: frozenset[str] = frozenset({"__pycache__", ".git"})


def load_repo(repo_path: str) -> dict[str, str]:
    """Walk *repo_path* and return ``{relative_path: file_contents}`` for
    every ``.py`` file found, skipping cache and version-control directories.

    Parameters
    ----------
    repo_path:
        Absolute or relative path to the root of the repository.

    Returns
    -------
    dict[str, str]
        Keys are POSIX-style paths relative to *repo_path* (e.g.
        ``"src/calculator.py"``).  Values are the raw text contents of each
        file.  Files that cannot be decoded as UTF-8 are silently skipped.

    Raises
    ------
    FileNotFoundError
        If *repo_path* does not exist or is not a directory.
    """
    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path is not a directory: {repo_path}")

    files: dict[str, str] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip-dirs in-place so os.walk never descends into them.
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            abs_path = Path(dirpath) / filename
            rel_path = abs_path.relative_to(root).as_posix()

            try:
                contents = abs_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                # Skip files that cannot be read as UTF-8 text.
                continue

            files[rel_path] = contents

    return files
