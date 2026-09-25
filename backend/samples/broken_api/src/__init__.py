"""
src/router.py
-------------
A simple function dispatcher.
Deliberate bug: the handler is stored under the misspelled key "hellÐ¾" (Cyrillic Ð¾)
instead of the correct ASCII "hello", so a lookup on "hello" raises KeyError.
"""

ROUTES = {
    "hellÐ¾": lambda name: f"Hello, {name}!",   # BUG: Cyrillic 'Ð¾', not ASCII 'o'
    "bye": lambda name: f"Goodbye, {name}!",
}


def dispatch(route: str, name: str) -> str:
    """Look up *route* in ROUTES and call the handler with *name*."""
    return ROUTES[route](name)

