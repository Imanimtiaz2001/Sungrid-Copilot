"""Ingestion: load -> categorise (manifest / auto-tag) -> chunk -> embed -> upsert.

Incremental and idempotent: chunk ids are content hashes, so re-running only embeds new
or changed chunks and deletes chunks whose source disappeared.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..embeddings import Embedder
from ..store import ChunkStore, VectorStore
from ..taxonomy import DocEntry, Manifest
from .chunker import Chunk, chunk_markdown

log = logging.getLogger(__name__)

CHUNKER_VERSION = "md-section-v1"
AutoTagger = Callable[[str], tuple[str, float]]  # text -> (category, confidence)


@dataclass
class IngestReport:
    embedder: str
    n_docs: int
    n_chunks: int
    embedded: int
    deleted: int
    unchanged: int
    auto_tagged: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def load_documents(kb_dir: Path) -> dict[str, str]:
    docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(kb_dir.glob("*.md"))}
    if not docs:
        raise FileNotFoundError(f"No markdown documents found in {kb_dir}")
    return docs


def build_chunks(
    docs: dict[str, str], manifest: Manifest, auto_tagger: AutoTagger | None = None
) -> tuple[list[Chunk], list[str]]:
    chunks: list[Chunk] = []
    auto: list[str] = []
    for doc_id, md in docs.items():
        entry = manifest.documents.get(doc_id)
        if entry is None:
            if auto_tagger is None:
                raise ValueError(f"Document '{doc_id}' is not in kb_manifest.yaml and no auto-tagger given")
            cat, conf = auto_tagger(md)
            entry = DocEntry(doc_id=doc_id, primary=cat, categories=[cat], auto_tagged=True)
            manifest.documents[doc_id] = entry
            auto.append(doc_id)
            log.warning("Auto-tagged unmapped document %s as %s (confidence %.2f) - review it", doc_id, cat, conf)
        for c in chunk_markdown(doc_id, md):
            c.categories = list(entry.categories)
            c.primary_category = entry.primary
            c.ambiguous = entry.ambiguous
            chunks.append(c)
    return chunks, auto


def fingerprint(docs: dict[str, str], manifest_path: Path, embedder_name: str) -> str:
    h = hashlib.sha256()
    h.update(CHUNKER_VERSION.encode())
    h.update(embedder_name.encode())
    h.update(manifest_path.read_bytes())
    for doc_id in sorted(docs):
        h.update(doc_id.encode())
        h.update(docs[doc_id].encode())
    return h.hexdigest()[:16]


def ingest(
    *,
    kb_dir: Path,
    manifest: Manifest,
    manifest_path: Path,
    embedder: Embedder,
    store: VectorStore,
    index_dir: Path | None,
    auto_tagger: AutoTagger | None = None,
    force: bool = False,
) -> tuple[ChunkStore, IngestReport]:
    t0 = time.perf_counter()
    docs = load_documents(kb_dir)
    chunks, auto = build_chunks(docs, manifest, auto_tagger)
    # categories are part of identity: re-categorising a doc must refresh its payloads
    for c in chunks:
        c.chunk_id = _with_categories(c)

    if force or not store.exists():
        store.reset(embedder.dim)
        existing: set[str] = set()
    else:
        existing = store.all_ids()

    new_ids = {c.chunk_id for c in chunks}
    to_embed = [c for c in chunks if c.chunk_id not in existing]
    stale = sorted(existing - new_ids)

    if to_embed:
        vecs = embedder.embed_documents([c.embed_text for c in to_embed])
        store.upsert(to_embed, vecs)
    store.delete(stale)

    chunk_store = ChunkStore(chunks)
    report = IngestReport(
        embedder=embedder.name,
        n_docs=len(docs),
        n_chunks=len(chunks),
        embedded=len(to_embed),
        deleted=len(stale),
        unchanged=len(chunks) - len(to_embed),
        auto_tagged=auto,
        duration_s=round(time.perf_counter() - t0, 3),
    )
    if index_dir is not None:
        chunk_store.save(index_dir / "chunks.json")
        meta = {
            "fingerprint": fingerprint(docs, manifest_path, embedder.name),
            "report": report.to_dict(),
        }
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "Ingested %s docs -> %s chunks (%s embedded, %s deleted)", len(docs), len(chunks), len(to_embed), len(stale)
    )
    return chunk_store, report


def _with_categories(c: Chunk) -> str:
    import uuid

    from .chunker import _NAMESPACE

    return str(uuid.uuid5(_NAMESPACE, f"{c.chunk_id}|{','.join(sorted(c.categories))}"))


def index_is_fresh(index_dir: Path, kb_dir: Path, manifest_path: Path, embedder_name: str) -> bool:
    meta_p = index_dir / "meta.json"
    if not meta_p.exists() or not (index_dir / "chunks.json").exists():
        return False
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        return meta.get("fingerprint") == fingerprint(load_documents(kb_dir), manifest_path, embedder_name)
    except Exception:  # noqa: BLE001
        return False
