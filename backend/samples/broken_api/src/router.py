"""
src/router.py
-------------
A simple function dispatcher.
Deliberate bug: the handler is stored under the misspelled key "hellо" (Cyrillic о)
instead of the correct ASCII "hello", so a lookup on "hello" raises KeyError.
"""

ROUTES = {
    "hellо": lambda name: f"Hello, {name}!",   # BUG: Cyrillic 'о', not ASCII 'o'
    "bye": lambda name: f"Goodbye, {name}!",
}


def dispatch(route: str, name: str) -> str:
    """Look up *route* in ROUTES and call the handler with *name*."""
    return ROUTES[route](name)
