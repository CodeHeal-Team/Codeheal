"""
Test Generator Agent for CodeHeal.
"""

from __future__ import annotations

from agents.base_agent import WatsonxAgent
from core.models import TestGenRequest


class TestGeneratorAgent(WatsonxAgent):
    """
    Generates a pytest test that reproduces the diagnosed bug.
    """

    def run(self, request: TestGenRequest) -> str:
        prompt = self._build_prompt(request)
        return self.generate(prompt)

    @staticmethod
    def _build_prompt(request: TestGenRequest) -> str:
        diagnosis = request.diagnosis
        affected_content = request.source_files.get(
            diagnosis.affected_file, ""
        )

        return (
            "You are a Python test engineer.\n\n"
            "Your task is to create ONE pytest test that reproduces "
            "the diagnosed bug.\n\n"
            "DIAGNOSIS:\n"
            f"Root cause: {diagnosis.root_cause}\n"
            f"Affected file: {diagnosis.affected_file}\n"
            f"Explanation: {diagnosis.explanation}\n\n"
            "CURRENT SOURCE FILE:\n"
            f"--- {diagnosis.affected_file} ---\n"
            f"{affected_content}\n\n"
            "REQUIREMENTS:\n"
            "1. Return a complete pytest source file.\n"
            "2. The test must reproduce the diagnosed bug.\n"
            "3. Use the correct import for the affected Python module.\n"
            "4. The generated test must initially FAIL against the buggy code.\n"
            "5. Do not modify production code.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY one JSON object, with no text before or after it.\n"
            "Do not output Markdown.\n"
            "Do not output ``` fences.\n"
            "Do not output explanations.\n"
            "Do not output any text before or after the JSON object.\n\n"
            "The JSON object MUST have exactly these two fields:\n"
            "{\n"
            '  "filename": "tests/test_generated.py",\n'
            '  "code": "complete pytest source code"\n'
            "}\n\n"
            "The value of code must be a JSON string. "
            "Escape newline characters as \\n and escape double quotes "
            "inside the code as required by JSON.\n\n"
            "Example of the required response shape:\n"
            '{"filename":"tests/test_generated.py",'
            '"code":"from src.calculator import divide\\n\\n'
            'def test_example():\\n    ..."}'
            "\n\nBegin your JSON response now:\n"
            "{"
        )