"""Vector store: Qdrant (embedded/local mode) + a JSON sidecar of chunk records.

Why Qdrant: native payload filtering with `MatchAny` over a *list* field is exactly the
"multi-label document, single-label query" filter this design needs, and the same client
code runs against a Qdrant server/cluster when volume grows (only the constructor changes).
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from .ingestion.chunker import Chunk


def _collection_name(embedder_name: str) -> str:
    return "sungrid_chunks__" + re.sub(r"[^a-zA-Z0-9]+", "_", embedder_name).strip("_").lower()


class VectorStore:
    def __init__(self, path: Path | None, embedder_name: str):
        # path=None -> in-memory (tests)
        self.client = QdrantClient(location=":memory:") if path is None else QdrantClient(path=str(path))
        self.collection = _collection_name(embedder_name)

    def reset(self, dim: int) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection).count if self.exists() else 0

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector=v.tolist(),
                payload={
                    "doc_id": c.doc_id,
                    "section": c.section,
                    "categories": c.categories,
                    "primary_category": c.primary_category,
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.client.delete(self.collection, points_selector=models.PointIdsList(points=chunk_ids))

    def all_ids(self) -> set[str]:
        ids: set[str] = set()
        offset = None
        while True:
            pts, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=False)
            ids.update(str(p.id) for p in pts)
            if offset is None:
                return ids

    def search(self, vector: np.ndarray, categories: list[str] | None, limit: int) -> list[tuple[str, float]]:
        flt = None
        if categories:
            flt = models.Filter(
                must=[models.FieldCondition(key="categories", match=models.MatchAny(any=list(categories)))]
            )
        res = self.client.query_points(
            self.collection, query=vector.tolist(), query_filter=flt, limit=limit, with_payload=False
        )
        return [(str(p.id), float(p.score)) for p in res.points]

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.client.close()


class ChunkStore:
    """Chunk records keyed by id (source of truth for text, BM25 and citations)."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = sorted(chunks, key=lambda c: (c.doc_id, c.order))
        self.by_id = {c.chunk_id: c for c in self.chunks}

    def section(self, ref: str) -> Chunk | None:
        """Look up "<doc_id>#<section title>"."""
        doc_id, _, section = ref.partition("#")
        for c in self.chunks:
            if c.doc_id == doc_id and c.section.split(" (part")[0] == section:
                return c
        return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ChunkStore:
        return cls([Chunk.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))])
