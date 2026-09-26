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
    def run(
        self,
        request: RefactorRequest,
        retry_instructions: str = "",
    ) -> str:
        prompt = self._build_prompt(request, retry_instructions)

        result = self.generate(prompt)

        print("\n=== REFACTOR RAW ===")
        print(repr(result))
        print("=== END REFACTOR RAW ===")

        if not result or not result.strip():
            raise RuntimeError(
                "Refactoring agent returned an empty response."
            )

        return result.strip()

    def run_full_file(
        self,
        request: RefactorRequest,
        retry_instructions: str = "",
    ) -> str:
        """Ask the model for the complete corrected file when a diff won't apply."""
        diagnosis = request.diagnosis
        affected_content = request.source_files.get(diagnosis.affected_file, "")
        prompt = (
            "You are repairing a Python file after a generated Git diff failed to apply.\n"
            "Return the COMPLETE corrected contents of the affected file, not a diff.\n"
            "Use the exact source shown below as your starting point. Preserve unrelated code.\n"
            "Fix only the diagnosed bug and ensure the failing test passes.\n\n"
            f"FILE PATH: {diagnosis.affected_file}\n"
            f"ROOT CAUSE: {diagnosis.root_cause}\n"
            f"EXPLANATION: {diagnosis.explanation}\n"
            f"FAILING TEST:\n{request.generated_test.code}\n\n"
            f"CURRENT FILE CONTENT:\n{affected_content}\n\n"
            f"RETRY INSTRUCTIONS: {retry_instructions}\n\n"
            'Return only one JSON object with exactly one key, "content", whose value is the complete file text. '
            "Do not use Markdown fences or add any text outside the JSON."
        )
        result = self.generate(prompt)
        if not result or not result.strip():
            raise RuntimeError("Refactoring agent returned an empty full-file response.")
        return result.strip()

    @staticmethod
    def _build_prompt(
        request: RefactorRequest,
        retry_instructions: str = "",
    ) -> str:
        diagnosis = request.diagnosis
        affected_content = request.source_files.get(
            diagnosis.affected_file, ""
        )
        affected_lines = ", ".join(
            str(line) for line in diagnosis.affected_lines
        )

        # Show the file with 1-based line numbers so Granite can compute
        # hunk offsets without guessing.
        numbered_lines = "\n".join(
            f"{i + 1}: {line}"
            for i, line in enumerate(affected_content.splitlines())
        )

        total_lines = len(affected_content.splitlines())

        return (
            "You are a Python software engineer fixing one specific bug.\n\n"
            "ROOT CAUSE:\n"
            f"{diagnosis.root_cause}\n\n"
            "BUG EXPLANATION:\n"
            f"{diagnosis.explanation}\n\n"
            "AFFECTED FILE PATH:\n"
            f"{diagnosis.affected_file}\n\n"
            "AFFECTED LINES (1-based):\n"
            f"{affected_lines or 'Not specified'}\n\n"
            f"CURRENT FILE CONTENT ({total_lines} lines, 1-based):\n"
            f"{numbered_lines}\n\n"
            "FAILING TEST (must pass after fix):\n"
            f"{request.generated_test.code}\n\n"
            "TASK:\n"
            "Fix the bug with the smallest possible change.\n"
            "Preserve all unrelated behavior and code.\n"
            "The fix must make the failing test pass.\n\n"
            f"RETRY INSTRUCTIONS:\n"
            f"{retry_instructions or 'No retry; this is the initial attempt.'}\n\n"
            "OUTPUT FORMAT (read carefully):\n"
            "Return ONLY a single valid JSON object — no text before or after it.\n"
            'The JSON object has exactly one key: "diff".\n'
            'The value of "diff" is a complete Git unified diff string.\n\n'
            "DIFF FORMAT REQUIREMENTS:\n"
            f'1. The diff must start with: diff --git a/{diagnosis.affected_file} b/{diagnosis.affected_file}\n'
            "2. Include the index and --- / +++ lines.\n"
            "3. Each hunk header looks like: @@ -<old_start>,<old_count> +<new_start>,<new_count> @@\n"
            "   old_count = number of context + removed lines in the hunk\n"
            "   new_count = number of context + added lines in the hunk\n"
            "4. Context lines (unchanged) must have a single space prefix.\n"
            "5. Removed lines must have a '-' prefix.\n"
            "6. Added lines must have a '+' prefix.\n"
            "7. The diff must contain at least one @@ hunk.\n"
            "8. Use the exact file path shown in AFFECTED FILE PATH.\n"
            "9. Escape all newlines as \\n inside the JSON string value.\n\n"
            "EXAMPLE (do not copy — adapt to the actual bug):\n"
            '{"diff": "diff --git a/src/foo.py b/src/foo.py\\n'
            'index 0000001..0000002 100644\\n'
            '--- a/src/foo.py\\n'
            '+++ b/src/foo.py\\n'
            '@@ -3,4 +3,5 @@\\n'
            ' def bar(x):\\n'
            '-    return x\\n'
            '+    if x is None:\\n'
            '+        raise ValueError()\\n'
            '+    return x\\n'
            '"}\n\n'
            "Do not return Markdown fences. Do not add explanations.\n"
        )