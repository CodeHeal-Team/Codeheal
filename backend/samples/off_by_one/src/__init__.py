"""
src/list_utils.py
-----------------
Utility functions for lists.
Deliberate bug: last_n() uses `len(items) - n` as the slice start, but it
should use `max(0, len(items) - n)`.  When n >= len(items) the index goes
negative and Python silently wraps around, returning more items than expected.
"""


def last_n(items: list, n: int) -> list:
    """Return the last *n* elements of *items*.

    BUG: when n >= len(items), a negative slice start wraps around and the
    result is the full list instead of raising / clamping correctly.
    """
    start = len(items) - n          # BUG: should be max(0, len(items) - n)
    return items[start:]

