"""Provider-agnostic LLM interface.

The copilot works WITHOUT an LLM (deterministic classifier, rule-based slot extraction,
extractive answers). An LLM, when configured, improves three things only: low-confidence
classification, free-form slot extraction, and fluent grounded answers. Any LLM failure
degrades to the deterministic path instead of failing the request.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMError(RuntimeError):
    """Raised for any provider error (auth, network, rate limit, malformed output)."""


class LLMClient(Protocol):
    name: str

    def complete(self, *, system: str, user: str, max_tokens: int = 600, temperature: float = 0.0) -> str: ...

    def call_tool(
        self, *, system: str, user: str, tool_name: str, tool_description: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Force a single function call and return its parsed arguments."""
        ...
