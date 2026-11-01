"""Wire all components together (one place, so the CLI, API, tests and eval share it)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .agent.copilot import Copilot
from .agent.tools import EligibilityTool
from .classifier import QueryClassifier, build_training_set, load_seed_examples
from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .ingestion.pipeline import IngestReport, index_is_fresh, ingest
from .llm import build_llm
from .retrieval import HybridRetriever
from .store import ChunkStore, VectorStore
from .taxonomy import Manifest, Taxonomy

log = logging.getLogger(__name__)


@dataclass
class Components:
    copilot: Copilot
    embedder: Embedder
    store: VectorStore
    ingest_report: IngestReport | None


def build(
    settings: Settings | None = None,
    *,
    in_memory: bool = False,
    force_reindex: bool = False,
    embedder: Embedder | None = None,
    use_llm: bool = True,
) -> Components:
    s = settings or get_settings()
    taxonomy = Taxonomy.load(s.config_path / "taxonomy.yaml")
    manifest_path = s.config_path / "kb_manifest.yaml"
    manifest = Manifest.load(manifest_path, taxonomy)
    seeds = load_seed_examples(s.data_path / "classifier" / "seed_queries.jsonl")

    llm = build_llm(s) if use_llm else None
    emb = embedder or build_embedder(s)
    log.info("Embedder: %s | LLM: %s", emb.name, llm.name if llm else "none (offline/extractive mode)")

    index_dir = None if in_memory else s.index_path
    if index_dir is not None:
        index_dir.mkdir(parents=True, exist_ok=True)
    store = VectorStore(None if in_memory else index_dir / "qdrant", emb.name)

    # auto-tagger for documents missing from the manifest: a seed-only classifier
    seed_clf = QueryClassifier(taxonomy).fit(build_training_set(taxonomy, seeds, None))

    def auto_tag(text: str) -> tuple[str, float]:
        c = seed_clf.predict_local(text[:2000])
        label = c.label if c.label != taxonomy.non_relevant_id else c.second
        return label, c.confidence

    report = None
    fresh = (
        not in_memory
        and not force_reindex
        and store.exists()
        and store.count() > 0
        and index_is_fresh(index_dir, s.kb_path, manifest_path, emb.name)
    )
    if fresh:
        chunks = ChunkStore.load(index_dir / "chunks.json")
    else:
        chunks, report = ingest(
            kb_dir=s.kb_path,
            manifest=manifest,
            manifest_path=manifest_path,
            embedder=emb,
            store=store,
            index_dir=index_dir,
            auto_tagger=auto_tag,
            force=force_reindex,
        )

    classifier = QueryClassifier(taxonomy, embedder=emb, llm=llm, adjudicate_margin=s.llm_adjudicate_margin)
    classifier.fit(build_training_set(taxonomy, seeds, chunks.chunks))
    retriever = HybridRetriever(chunks, store, emb, rrf_k=s.rrf_k, pool=s.candidate_pool)
    copilot = Copilot(
        settings=s,
        taxonomy=taxonomy,
        manifest=manifest,
        chunks=chunks,
        retriever=retriever,
        classifier=classifier,
        tool=EligibilityTool(timeout_s=s.tool_timeout_s),
        llm=llm,
    )
    return Components(copilot=copilot, embedder=emb, store=store, ingest_report=report)
