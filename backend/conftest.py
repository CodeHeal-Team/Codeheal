# conftest.py — root pytest configuration
import warnings
import pytest


def pytest_configure(config):
    # Suppress harmless PytestCollectionWarning for classes whose names start
    # with "Test" but cannot be collected (dataclasses / agent classes with
    # __init__).
    for class_name in (
        "TestResult",
        "TestGeneratorAgent",
        "TestGenRequest",
    ):
        warnings.filterwarnings(
            "ignore",
            message=f"cannot collect test class '{class_name}'",
            category=pytest.PytestCollectionWarning,
        )
