"""
agents/base_agent.py
--------------------
Shared IBM watsonx.ai client setup for all CodeHeal agents.

Every agent that calls the model inherits :class:`WatsonxAgent`, which
initialises a :class:`~ibm_watsonx_ai.foundation_models.ModelInference`
client from environment variables and exposes a single
:py:meth:`WatsonxAgent.generate` helper.

Architecture
~~~~~~~~~~~~
* :class:`BaseAgent` â€” minimal abstract interface (``run`` only).
  Both the real watsonx agents and the placeholder agents inherit from it
  so the pipeline can swap implementations without call-site changes.
* :class:`WatsonxAgent` â€” extends ``BaseAgent`` with the watsonx.ai client.
  Real agents (``DiagnosticAgent``, ``TestGeneratorAgent``,
  ``RefactoringAgent``) inherit from ``WatsonxAgent``.

Model availability check
~~~~~~~~~~~~~~~~~~~~~~~~
On construction :class:`WatsonxAgent` calls
:func:`_resolve_model_id`, which:

1. Queries the watsonx.ai ``get_model_specs`` endpoint.
2. Confirms ``ibm/granite-4-h-small`` is available in the project.
3. If not, logs a warning that lists the available IBM Granite instruct
   models and automatically selects the first one alphabetically.

The check is skipped (with a warning) when ``get_model_specs`` is
unavailable or throws an exception, so the agent still works in offline
test environments.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
``WATSONX_API_KEY``      â€” IBM Cloud API key (required)
``WATSONX_PROJECT_ID``   â€” watsonx.ai project ID (required)
``WATSONX_URL``          â€” service URL, default ``https://us-south.ml.cloud.ibm.com``
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Union

from dotenv import load_dotenv

from core.models import DiagnosisRequest, TestGenRequest, RefactorRequest

# Import watsonx SDK at module level so tests can patch these names directly.
# Wrapped in try/except so the module is importable in offline environments.
try:
    from ibm_watsonx_ai import Credentials  # type: ignore  # noqa: F401
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore  # noqa: F401
    from ibm_watsonx_ai.foundation_models import get_model_specs  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover
    Credentials = None  # type: ignore[assignment,misc]
    ModelInference = None  # type: ignore[assignment,misc]
    get_model_specs = None  # type: ignore[assignment,misc]

load_dotenv()

logger = logging.getLogger(__name__)

# Union of all possible request types understood by this agent hierarchy.
AgentRequest = Union[DiagnosisRequest, TestGenRequest, RefactorRequest]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PINNED_MODEL = "ibm/granite-4-h-small"
_DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"

# Generation parameters tuned for structured JSON output.
_GENERATE_PARAMS = {
    "max_new_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
}


# ---------------------------------------------------------------------------
# Minimal abstract interface (shared by real and placeholder agents)
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Minimal interface contract shared by every CodeHeal agent.

    Subclasses **must** implement :py:meth:`run`.  They may override
    ``__init__`` to accept configuration (e.g. model credentials) but must
    call ``super().__init__()`` so the base class can perform any future
    shared setup.
    """

    def __init__(self) -> None:  # noqa: D107
        pass

    @abstractmethod
    def run(self, request: AgentRequest) -> str:
        """
        Execute the agent for the given *request* and return the raw model
        (or fixture) response as a string.

        Parameters
        ----------
        request:
            One of :class:`~core.models.DiagnosisRequest`,
            :class:`~core.models.TestGenRequest`, or
            :class:`~core.models.RefactorRequest`.

        Returns
        -------
        str
            Raw response string.  The caller is responsible for parsing and
            validating this value via ``core.response_parser``.
        """


# ---------------------------------------------------------------------------
# Watsonx-backed agent base
# ---------------------------------------------------------------------------


def _resolve_model_id(url: str) -> str:
    """
    Confirm the pinned model is available; fall back to the first available
    IBM Granite instruct model if it is not.

    The function attempts to call :func:`ibm_watsonx_ai.foundation_models.get_model_specs`.
    Any exception is caught so offline / test environments are not broken.

    Parameters
    ----------
    url:
        The watsonx.ai service URL, used as the base for the models endpoint.

    Returns
    -------
    str
        Model ID to use (either ``PINNED_MODEL`` or a fallback).
    """
    try:
        specs = get_model_specs(url)
        resources = specs.get("resources", [])
        available_ids: list[str] = [
            r.get("model_id", "") for r in resources if r.get("model_id")
        ]

        if PINNED_MODEL in available_ids:
            return PINNED_MODEL

        # Pinned model missing â€” find first available IBM Granite instruct model.
        granite_instruct = sorted(
            mid for mid in available_ids
            if "granite" in mid.lower() and "instruct" in mid.lower()
        )

        if granite_instruct:
            fallback = granite_instruct[0]
            logger.warning(
                "Pinned model '%s' is not available in this project. "
                "Available IBM Granite instruct models: %s. "
                "Falling back to '%s'.",
                PINNED_MODEL,
                granite_instruct,
                fallback,
            )
            return fallback

        logger.warning(
            "Pinned model '%s' is not available and no IBM Granite instruct "
            "fallback was found. Using '%s' anyway â€” requests may fail.",
            PINNED_MODEL,
            PINNED_MODEL,
        )
        return PINNED_MODEL

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Model availability check skipped (%s). Using '%s'.",
            exc,
            PINNED_MODEL,
        )
        return PINNED_MODEL


class WatsonxAgent(BaseAgent):
    """
    Base class for agents that call IBM watsonx.ai.

    Reads credentials from environment variables, resolves the model ID,
    and exposes :py:meth:`generate`.
    """

    def __init__(self) -> None:
        super().__init__()

        api_key = os.environ.get("WATSONX_API_KEY", "")
        project_id = os.environ.get("WATSONX_PROJECT_ID", "")
        url = os.environ.get("WATSONX_URL", _DEFAULT_URL)

        if not api_key or not project_id:
            raise EnvironmentError(
                "WATSONX_API_KEY and WATSONX_PROJECT_ID must be set. "
                "Copy .env.example to .env and fill in your credentials."
            )

        model_id = _resolve_model_id(url)

        self._model = ModelInference(
            model_id=model_id,
            credentials=Credentials(url=url, api_key=api_key),
            project_id=project_id,
            params=_GENERATE_PARAMS,
        )
        self._model_id = model_id

    def generate(self, prompt: str) -> str:
        """
        Call the watsonx.ai model with *prompt* and return the generated text.

        Parameters
        ----------
        prompt:
            Full prompt string to send to the model.

        Returns
        -------
        str
            Raw generated text from the model.
        """
        response = self._model.generate(prompt=prompt)
        # response is a dict: {"results": [{"generated_text": "..."}]}
        try:
            return response["results"][0]["generated_text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected response structure from watsonx.ai: {response}"
            ) from exc

