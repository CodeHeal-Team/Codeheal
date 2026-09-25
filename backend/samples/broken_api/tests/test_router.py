"""
tests/test_router.py
--------------------
Calls dispatch() with the correct ASCII key "hello".
This test FAILS because the route is stored under a misspelled key.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from router import dispatch


def test_hello_route_returns_greeting():
    result = dispatch("hello", "World")
    assert result == "Hello, World!"
