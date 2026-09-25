"""
tests/test_calculator.py
------------------------
Exposes the None-input crash in divide().
This test FAILS before the bug is fixed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from calculator import divide


def test_divide_with_none_raises_value_error():
    try:
        divide(None, 2)
    except ValueError:
        return
    except TypeError:
        raise
    assert False, "Expected ValueError for None input"
