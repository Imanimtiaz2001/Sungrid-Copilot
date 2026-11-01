"""Embedding backends behind one small interface.

* ``fastembed`` (default) — BAAI/bge-small-en-v1.5 via ONNX on CPU. Local, free, no API key,
  ~67 MB one-time download. Uses bge's query/passage instructions.
* ``openai`` — text-embedding-3-small (needs OPENAI_API_KEY).
* ``hashing`` — deterministic hashed word + char n-gram vectors. Zero downloads; used in CI
  and as an automatic fallback when no model can be loaded (e.g. air-gapped machines).

``auto`` tries fastembed -> openai (if a key is set) -> hashing, logging what it picked.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    semantic: bool  # True for learned models; False for the lexical hashing fallback

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        n = np.linalg.norm(m)
        return m / n if n else m
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class FastEmbedEmbedder:
    semantic = True

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None):
        from fastembed import TextEmbedding  # imported lazily: heavy

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.name = f"fastembed:{model_name}"
        self._dim = int(len(next(iter(self._model.query_embed("probe")))))

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2(np.vstack(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(next(iter(self._model.query_embed(text))))


class OpenAIEmbedder:
    semantic = True

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url or None, max_retries=3, timeout=30)
        self._model = model
        self.name = f"openai:{model}"
        self._dim = len(self._embed(["probe"])[0])

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            resp = self._client.embeddings.create(model=self._model, input=texts[i : i + 256])
            out.extend(d.embedding for d in resp.data)
        return _l2(np.array(out))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class HashingEmbedder:
    """Hashed TF of stemmed words, word bigrams and char 3-5 grams, L2-normalised.

    Not semantic — it is a robust lexical/fuzzy matcher that needs no model download and
    never changes between runs, which makes it ideal for CI and offline demos.
    """

    semantic = False

    def __init__(self, n_features: int = 2**12):
        from sklearn.feature_extraction.text import HashingVectorizer

        from .text import tokenize

        self._word = HashingVectorizer(
            n_features=n_features,
            analyzer=lambda s: self._word_feats(s, tokenize),
            alternate_sign=False,
            norm="l2",
        )
        self._char = HashingVectorizer(
            n_features=n_features,
            analyzer="char_wb",
            ngram_range=(3, 5),
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self.name = f"hashing:{n_features}"
        self._dim = 2 * n_features

    @staticmethod
    def _word_feats(s: str, tokenize) -> list[str]:
        toks = tokenize(s)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        w = self._word.transform(texts).toarray()
        c = self._char.transform(texts).toarray()
        return _l2(np.hstack([w * 0.7, c * 0.3]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    cache_dir = str(settings.index_path / "models")

    if provider == "hashing":
        return HashingEmbedder()
    if provider == "openai":
        return OpenAIEmbedder(settings.openai_api_key, settings.embedding_model or "text-embedding-3-small")
    if provider == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model or "BAAI/bge-small-en-v1.5", cache_dir)

    # auto
    try:
        return FastEmbedEmbedder(settings.embedding_model or "BAAI/bge-small-en-v1.5", cache_dir)
    except Exception as exc:  # noqa: BLE001 - any load/download failure triggers fallback
        log.warning("fastembed unavailable (%s); trying next embedding backend", type(exc).__name__)
    if settings.openai_api_key:
        try:
            return OpenAIEmbedder(settings.openai_api_key, "text-embedding-3-small")
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenAI embeddings unavailable (%s)", type(exc).__name__)
    log.warning("Falling back to the offline hashing embedder (lexical, not semantic).")
    return HashingEmbedder()
