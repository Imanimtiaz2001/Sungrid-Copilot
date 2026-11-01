from __future__ import annotations

import pytest

from sungrid_copilot.config import Settings
from sungrid_copilot.embeddings import HashingEmbedder
from sungrid_copilot.factory import build


def offline_settings(**overrides) -> Settings:
    """Deterministic, key-free settings: hashing embedder, no LLM, ignore any local .env."""
    params = {"embedding_provider": "hashing", "llm_provider": "none", **overrides}
    return Settings(_env_file=None, **params)  # type: ignore[call-arg]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return offline_settings()


@pytest.fixture(scope="session")
def components(settings):
    return build(settings, in_memory=True, embedder=HashingEmbedder(), use_llm=False)


@pytest.fixture()
def copilot(components):
    c = components.copilot
    c.sessions = type(c.sessions)()  # fresh sessions per test
    return c


class FakeLLM:
    """Scriptable stand-in for an LLM provider (no network)."""

    name = "fake:test"

    def __init__(self, complete=None, tool=None, fail: bool = False):
        self._complete = complete or (lambda system, user: "I couldn't find that.")
        self._tool = tool or (lambda **kw: {"intent": "knowledge_question", "program": "unknown"})
        self.fail = fail
        self.calls: list[str] = []

    def complete(self, *, system, user, max_tokens=600, temperature=0.0):
        from sungrid_copilot.llm.base import LLMError

        self.calls.append("complete")
        if self.fail:
            raise LLMError("simulated outage")
        return self._complete(system, user)

    def call_tool(self, *, system, user, tool_name, tool_description, parameters):
        from sungrid_copilot.llm.base import LLMError

        self.calls.append("call_tool")
        if self.fail:
            raise LLMError("simulated outage")
        return self._tool(user=user, parameters=parameters)
