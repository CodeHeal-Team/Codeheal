"""
src/calculator.py
-----------------
Simple calculator with a deliberate bug: divide() does not guard against
None inputs, causing an unhandled TypeError at runtime.
"""


def add(a, b):
    return a + b


def divide(a, b):
    # BUG: no None-guard â€” crashes with TypeError when a or b is None
    return a / b

