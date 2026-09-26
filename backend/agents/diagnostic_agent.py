"""
Diagnostic Agent for CodeHeal.
"""

from __future__ import annotations

from agents.base_agent import WatsonxAgent
from core.models import DiagnosisRequest


class DiagnosticAgent(WatsonxAgent):
    """
    Analyses source files and failing test output to produce a diagnosis.
    """

    def run(self, request: DiagnosisRequest) -> str:
        prompt = self._build_prompt(request)
        return self.generate(prompt)

    @staticmethod
    def _build_prompt(request: DiagnosisRequest) -> str:
        """
        Construct the prompt sent to the model.
        """

        parts: list[str] = [
            "You are a Python debugging expert.\n",
            "Below are the source files from the repository:\n",
        ]

        for path, content in request.source_files.items():
            parts.append(f"\n--- {path} ---\n{content}\n")

        parts.append("\n--- Test output ---\n")
        parts.append(request.test_output)

        parts.append(
            "\n\nIMPORTANT: Your entire response must be valid JSON. "
            "Return ONLY the JSON object, with no prose before or after it. "
            "Do not use Markdown code fences.\n"
            "Return exactly this JSON structure:\n"
            "{\n"
            '  "root_cause": "brief description of the bug",\n'
            '  "affected_file": "path to the buggy file",\n'
            '  "affected_lines": [1],\n'
            '  "explanation": "explain why the code fails",\n'
            '  "confidence": 0.9\n'
            "}\n"
        )
        parts.append("\nBegin your JSON response now:\n{")
        return "".join(parts)