"""
Refactoring Agent for CodeHeal.
"""

from __future__ import annotations

from agents.base_agent import WatsonxAgent
from core.models import RefactorRequest


class RefactoringAgent(WatsonxAgent):
    """
    Produces the old and new file contents for a targeted fix.
    """

    def run(self, request: RefactorRequest) -> str:
        prompt = self._build_prompt(request)

        result = self.generate(prompt)

        print("\n=== REFACTOR RAW ===")
        print(repr(result))
        print("=== END REFACTOR RAW ===")

        if not result or not result.strip():
            raise RuntimeError(
                "Refactoring agent returned an empty response."
            )

        return result.strip()

    
    @staticmethod
    def _build_prompt(request: RefactorRequest) -> str:
        diagnosis = request.diagnosis
        affected_content = request.source_files.get(
            diagnosis.affected_file, ""
        )
        affected_lines = ", ".join(
            str(line) for line in diagnosis.affected_lines
        )

        return (
            "You are a Python software engineer fixing one specific bug.\n\n"
            "ROOT CAUSE:\n"
            f"{diagnosis.root_cause}\n\n"
            "BUG EXPLANATION:\n"
            f"{diagnosis.explanation}\n\n"
            "AFFECTED FILE:\n"
            f"{diagnosis.affected_file}\n\n"
            "AFFECTED LINES:\n"
            f"{affected_lines or 'Not specified'}\n\n"
            "CURRENT FILE CONTENT:\n"
            f"{affected_content}\n\n"
            "FAILING TEST:\n"
            f"{request.generated_test.code}\n\n"
            "TASK:\n"
            "Fix the bug with the smallest possible change.\n"
            "Preserve unrelated behavior and existing code.\n"
            "The fix must make the failing test pass.\n\n"
            "OUTPUT RULES:\n"
            "Return ONLY a valid JSON object with one key: diff.\n"
            'The value of "diff" must be a Git diff string.\n'
            'The diff must start with "diff --git".\n'
            "Include the correct file paths and unified diff hunks.\n"
            "Use the exact affected file path given above.\n"
            "Do not return Markdown or code fences.\n"
            "Do not add explanations or extra JSON keys.\n"
            "Escape newlines correctly inside the JSON string.\n"
            "Begin your JSON response now:\n"
            "{"
        )