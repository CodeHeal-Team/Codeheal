"""
agents/__init__.py
------------------
Public API for the CodeHeal agents package.

Real agents (not-yet-watsonx-connected stubs):
    DiagnosticAgent, TestGeneratorAgent, RefactoringAgent

Placeholder agents (fixture-backed, for Sub-Tasks 7 & 8):
    PlaceholderDiagnosticAgent, PlaceholderTestGeneratorAgent,
    PlaceholderRefactoringAgent
"""

from agents.base_agent import BaseAgent
from agents.diagnostic_agent import DiagnosticAgent
from agents.test_generator_agent import TestGeneratorAgent
from agents.refactoring_agent import RefactoringAgent
from agents.placeholders import (
    PlaceholderDiagnosticAgent,
    PlaceholderTestGeneratorAgent,
    PlaceholderRefactoringAgent,
)

__all__ = [
    "BaseAgent",
    "DiagnosticAgent",
    "TestGeneratorAgent",
    "RefactoringAgent",
    "PlaceholderDiagnosticAgent",
    "PlaceholderTestGeneratorAgent",
    "PlaceholderRefactoringAgent",
]
