"""Hybrid retrieval (BM25 + dense) with a category filter.

* Both retrievers apply the SAME category filter (Qdrant payload filter / Python filter).
* Results are fused with Reciprocal Rank Fusion, which needs no score calibration between
  BM25 and cosine similarity.
* Every result carries an embedder-independent *evidence* score: the IDF-weighted share of
  the query's content terms found in the best chunk. Abstention and the unfiltered fallback
  are driven by evidence, so thresholds do not need re-tuning when the embedder changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .ingestion.chunker import Chunk
from .store import ChunkStore, VectorStore
from .text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    bm25: float = 0.0
    dense: float = 0.0
    evidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "citation": self.chunk.citation,
            "categories": self.chunk.categories,
            "rrf": round(self.score, 4),
            "bm25": round(self.bm25, 3),
            "dense": round(self.dense, 3),
            "evidence": round(self.evidence, 3),
        }


@dataclass
class RetrievalResult:
    hits: list[Hit]
    filter_categories: list[str] | None
    fallback_used: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def evidence(self) -> float:
        return max((h.evidence for h in self.hits), default=0.0)


class HybridRetriever:
    def __init__(self, chunks: ChunkStore, store: VectorStore, embedder: Embedder, *, rrf_k: int = 60, pool: int = 12):
        self.chunks = chunks
        self.store = store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.pool = pool
        self._ids = [c.chunk_id for c in chunks.chunks]
        self._pos = {cid: i for i, cid in enumerate(self._ids)}
        corpus = [tokenize(c.embed_text) for c in chunks.chunks]
        self._doc_terms = [set(t) for t in corpus]
        self._bm25 = BM25Okapi(corpus, k1=1.2, b=0.6)
        n = len(corpus)
        df: dict[str, int] = {}
        for terms in self._doc_terms:
            for t in terms:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._max_idf = math.log(1 + (n + 0.5) / 0.5)

    # ------------------------------------------------------------------ helpers
    def idf(self, term: str) -> float:
        return self._idf.get(term, self._max_idf)

    def evidence(self, query_terms: list[str], chunk_id: str) -> float:
        """IDF-weighted fraction of query terms present in the chunk (0..1)."""
        terms = set(query_terms)
        if not terms:
            return 0.0
        doc_terms = self._doc_terms[self._pos[chunk_id]]
        total = sum(self.idf(t) for t in terms)
        hit = sum(self.idf(t) for t in terms if t in doc_terms)
        return hit / total if total else 0.0

    def _allowed(self, categories: list[str] | None) -> np.ndarray:
        if not categories:
            return np.ones(len(self._ids), dtype=bool)
        cats = set(categories)
        return np.array([bool(cats.intersection(c.categories)) for c in self.chunks.chunks])

    # ------------------------------------------------------------------ search
    def search(self, query: str, categories: list[str] | None, k: int = 4) -> RetrievalResult:
        q_terms = tokenize(query)
        allowed = self._allowed(categories)

        bm25_scores = self._bm25.get_scores(q_terms) if q_terms else np.zeros(len(self._ids))
        bm25_scores = np.where(allowed, bm25_scores, -np.inf)
        bm25_order = [
            i for i in np.argsort(-bm25_scores)[: self.pool] if np.isfinite(bm25_scores[i]) and bm25_scores[i] > 0
        ]

        qv = self.embedder.embed_query(query)
        dense = self.store.search(qv, categories, self.pool)
        dense_rank = {cid: r for r, (cid, _) in enumerate(dense)}
        dense_score = dict(dense)

        fused: dict[str, float] = {}
        for r, i in enumerate(bm25_order):
            cid = self._ids[i]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + r + 1)
        for cid, r in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + r + 1)

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        hits = []
        for cid, s in ranked:
            idx = self._pos[cid]
            hits.append(
                Hit(
                    chunk=self.chunks.by_id[cid],
                    score=s,
                    bm25=float(bm25_scores[idx]) if np.isfinite(bm25_scores[idx]) else 0.0,
                    dense=dense_score.get(cid, 0.0),
                    evidence=self.evidence(q_terms, cid),
                )
            )
        return RetrievalResult(hits=hits, filter_categories=list(categories) if categories else None)

    def search_with_policy(
        self,
        query: str,
        categories: list[str] | None,
        *,
        k: int,
        min_evidence: float,
    ) -> RetrievalResult:
        """Filtered search; if the filtered evidence is weak, retry unfiltered and keep the
        better-supported result (logged as a fallback, so filter misses are measurable)."""
        res = self.search(query, categories, k)
        if not categories or res.evidence >= min_evidence:
            return res
        wide = self.search(query, None, k)
        if wide.evidence > res.evidence + 0.10:
            wide.fallback_used = True
            wide.notes.append(
                f"category filter {categories} gave weak evidence ({res.evidence:.2f}); "
                f"used unfiltered results ({wide.evidence:.2f})"
            )
            return wide
        res.notes.append(f"weak evidence ({res.evidence:.2f}); unfiltered search was not better")
        return res
