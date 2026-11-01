"""LLM adapters: OpenAI-compatible (OpenAI, Gemini, Groq, Ollama, HF router, ...) and Anthropic."""

from __future__ import annotations

import json
import logging
from typing import Any

from ..config import DEFAULT_BASE_URLS, Settings
from .base import LLMClient, LLMError

log = logging.getLogger(__name__)


class OpenAICompatibleClient:
    def __init__(self, *, api_key: str, model: str, base_url: str | None, timeout: float, max_retries: int, label: str):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or "not-needed", base_url=base_url or None, timeout=timeout, max_retries=max_retries
        )
        self.model = model
        self.name = f"{label}:{model}"
        self._legacy_params = True  # max_tokens + temperature; flipped for reasoning-style models

    def _create(self, **kwargs: Any):
        """Call chat.completions, adapting once to models that reject `max_tokens`/`temperature`
        (e.g. OpenAI reasoning models want `max_completion_tokens` and default sampling)."""
        if not self._legacy_params:
            kwargs = self._modern(kwargs)
        try:
            return self._client.chat.completions.create(model=self.model, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if self._legacy_params and ("max_tokens" in msg or "temperature" in msg) and "400" in msg:
                self._legacy_params = False
                return self._client.chat.completions.create(model=self.model, **self._modern(kwargs))
            raise

    @staticmethod
    def _modern(kwargs: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in kwargs.items() if k != "temperature"}
        if "max_tokens" in out:
            out["max_completion_tokens"] = out.pop("max_tokens")
        return out

    def complete(self, *, system: str, user: str, max_tokens: int = 600, temperature: float = 0.0) -> str:
        try:
            resp = self._create(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{self.name}: {type(exc).__name__}: {exc}") from exc

    def call_tool(
        self, *, system: str, user: str, tool_name: str, tool_description: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            resp = self._create(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": tool_name, "description": tool_description, "parameters": parameters},
                    }
                ],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                temperature=0.0,
            )
            calls = resp.choices[0].message.tool_calls or []
            if not calls:
                raise LLMError(f"{self.name}: model returned no tool call")
            return json.loads(calls[0].function.arguments or "{}")
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{self.name}: {type(exc).__name__}: {exc}") from exc


class AnthropicClient:
    def __init__(self, *, api_key: str, model: str, timeout: float, max_retries: int, base_url: str | None = None):
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries, base_url=base_url or None
        )
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, *, system: str, user: str, max_tokens: int = 600, temperature: float = 0.0) -> str:
        try:
            # Current Claude models removed sampling params (temperature/top_p), so none are sent.
            resp = self._client.messages.create(
                model=self.model,
                system=system,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{self.name}: {type(exc).__name__}: {exc}") from exc

    def call_tool(
        self, *, system: str, user: str, tool_name: str, tool_description: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            resp = self._client.messages.create(
                model=self.model,
                system=system,
                max_tokens=500,
                messages=[{"role": "user", "content": user}],
                tools=[{"name": tool_name, "description": tool_description, "input_schema": parameters}],
                tool_choice={"type": "tool", "name": tool_name},
            )
            for b in resp.content:
                if getattr(b, "type", "") == "tool_use":
                    return dict(b.input)
            raise LLMError(f"{self.name}: model returned no tool call")
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{self.name}: {type(exc).__name__}: {exc}") from exc


def build_llm(settings: Settings) -> LLMClient | None:
    provider = settings.resolved_llm_provider()
    model = settings.resolved_llm_model()
    common = {"timeout": settings.llm_timeout_s, "max_retries": settings.llm_max_retries}
    try:
        if provider == "none":
            return None
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                raise LLMError("ANTHROPIC_API_KEY is not set")
            return AnthropicClient(
                api_key=settings.anthropic_api_key, model=model, base_url=settings.llm_base_url, **common
            )
        key = {
            "openai": settings.openai_api_key,
            "gemini": settings.gemini_api_key or settings.google_api_key,
            "groq": settings.groq_api_key,
            "ollama": "ollama",
            "openai_compatible": settings.llm_api_key,
        }[provider]
        if not key:
            raise LLMError(f"No API key configured for provider '{provider}'")
        if not model:
            raise LLMError("LLM_MODEL must be set for openai_compatible providers")
        base_url = settings.llm_base_url or DEFAULT_BASE_URLS.get(provider)
        return OpenAICompatibleClient(api_key=key, model=model, base_url=base_url, label=provider, **common)
    except LLMError as exc:
        log.warning("LLM disabled: %s. Running in offline (extractive) mode.", exc)
        return None
