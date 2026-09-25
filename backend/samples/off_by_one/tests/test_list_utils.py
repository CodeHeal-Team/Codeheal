"""
tests/test_list_utils.py
------------------------
Catches the off-by-one / wrap-around error in last_n().
This test FAILS before the bug is fixed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from list_utils import last_n


def test_last_n_when_n_exceeds_length():
    # Asking for the last 10 items of a 3-element list should return all 3,
    # not wrap around and still return 3 (the bug gives the right length here
    # via wrap, but the boundary case below catches the logic error).
    items = [1, 2, 3]
    result = last_n(items, 10)
    assert result == [1, 2, 3], f"Expected [1, 2, 3], got {result}"


def test_last_n_boundary_negative_start():
    # n == len(items) + 1 → start index -1 → wraps to the last element only.
    # The buggy code returns items[-1:] == [3] instead of all items.
    items = [10, 20, 30]
    result = last_n(items, 4)   # n > len → should return all 3 items
    assert len(result) == 3, f"Expected 3 items, got {len(result)}: {result}"
