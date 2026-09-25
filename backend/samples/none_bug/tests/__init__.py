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
    # Expect a ValueError for None inputs, but the buggy code raises TypeError
    try:
        result = divide(None, 2)
        assert False, "Expected an exception but got: " + str(result)
    except TypeError:
        # The bug: we get TypeError instead of the expected ValueError
        raise

