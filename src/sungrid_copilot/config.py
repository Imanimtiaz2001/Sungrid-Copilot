"""Runtime configuration, loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # src/sungrid_copilot -> repo root

LLMProvider = Literal["auto", "none", "openai", "anthropic", "gemini", "groq", "ollama", "openai_compatible"]
EmbeddingProvider = Literal["auto", "fastembed", "openai", "hashing"]

# Defaults are small, cheap models. Override with LLM_MODEL.
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1:8b",
    "openai_compatible": "",
}
DEFAULT_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- paths -------------------------------------------------------------
    sungrid_home: Path = Field(default=_PACKAGE_ROOT)
    kb_dir: Path | None = None
    index_dir: Path | None = None

    # --- LLM -----------------------------------------------------------------
    llm_provider: LLMProvider = "auto"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""  # generic key for openai_compatible providers
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""

    # --- embeddings ------------------------------------------------------------
    embedding_provider: EmbeddingProvider = "auto"
    embedding_model: str = ""  # fastembed: BAAI/bge-small-en-v1.5, openai: text-embedding-3-small

    # --- retrieval / classification -----------------------------------------
    top_k: int = 4
    rrf_k: int = 60
    candidate_pool: int = 12
    # classifier: widen the filter to the top-2 categories when the margin is small
    filter_widen_margin: float = 0.20
    # classifier: ask the LLM to adjudicate when the local margin is below this
    llm_adjudicate_margin: float = 0.15
    # retrieval evidence (IDF-weighted query-term coverage of the best chunk)
    min_evidence: float = 0.20  # below this the filtered result is "weak" -> unfiltered fallback
    abstain_evidence: float = 0.10  # below this nothing in the KB supports an answer...
    semantic_support: float = 0.70  # ...unless a semantic embedder finds a close match (bge cosine)
    off_topic_evidence: float = 0.45  # non_relevant queries need at least this to be answered

    # --- eligibility tool -----------------------------------------------------
    tool_timeout_s: float = 5.0

    # ------------------------------------------------------------------------
    @property
    def kb_path(self) -> Path:
        return Path(self.kb_dir) if self.kb_dir else self.sungrid_home / "data" / "kb"

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir) if self.index_dir else self.sungrid_home / ".index"

    @property
    def config_path(self) -> Path:
        return self.sungrid_home / "config"

    @property
    def data_path(self) -> Path:
        return self.sungrid_home / "data"

    def resolved_llm_provider(self) -> str:
        """`auto` picks the first provider whose API key is configured, else `none`."""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        if self.gemini_api_key or self.google_api_key:
            return "gemini"
        if self.groq_api_key:
            return "groq"
        return "none"

    def resolved_llm_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS.get(self.resolved_llm_provider(), "")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Allow running from any cwd: read .env from the repo root as well as cwd.
    env_files = [p for p in (Path.cwd() / ".env", _PACKAGE_ROOT / ".env") if p.exists()]
    if env_files:
        return Settings(_env_file=env_files[0])  # type: ignore[call-arg]
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


def env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}
