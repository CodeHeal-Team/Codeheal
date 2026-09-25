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

        return (
            "You are a Python software engineer fixing one specific bug.\n\n"
            "BUG:\n"
            f"{diagnosis.explanation}\n\n"
            "AFFECTED FILE:\n"
            f"{diagnosis.affected_file}\n\n"
            "CURRENT FILE CONTENT:\n"
            f"{affected_content}\n\n"
            "FAILING TEST:\n"
            f"{request.generated_test.code}\n\n"
            "TASK:\n"
            "Fix the bug with the smallest possible change.\n"
            "The fix must make the failing test pass.\n"
            "Do not change unrelated behavior.\n\n"
            "OUTPUT RULES:\n"
            "Return ONLY the complete corrected Python source file.\n"
            "Do NOT return JSON.\n"
            "Do NOT return Markdown.\n"
            "Do NOT use ``` code fences.\n"
            "Do NOT add explanations.\n"
            "Do NOT describe the file.\n"
            "Do NOT add text before or after the Python source.\n"
            "The first character of your response must be the first "
            "character of the Python file.\n"
            "The last character must be the last character of the Python file.\n"
        )
