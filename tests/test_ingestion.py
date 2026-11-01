import shutil
from pathlib import Path

from sungrid_copilot.embeddings import HashingEmbedder
from sungrid_copilot.ingestion.pipeline import ingest
from sungrid_copilot.store import VectorStore
from sungrid_copilot.taxonomy import Manifest, Taxonomy

ROOT = Path(__file__).resolve().parents[1]


def _load():
    tax = Taxonomy.load(ROOT / "config" / "taxonomy.yaml")
    return tax, Manifest.load(ROOT / "config" / "kb_manifest.yaml", tax)


def test_every_kb_document_is_in_the_manifest():
    _, man = _load()
    docs = {p.stem for p in (ROOT / "data" / "kb").glob("*.md")}
    assert docs == set(man.documents)


def test_ambiguous_document_is_multilabel_with_rationale():
    _, man = _load()
    d = man.documents["06_ambiguous_rebate_billing_adjustments"]
    assert set(d.categories) == {"incentive_rebate", "billing_account"}
    assert d.ambiguous and d.primary == "incentive_rebate" and d.rationale


def test_ingest_is_incremental_and_idempotent(tmp_path):
    tax, man = _load()
    kb = tmp_path / "kb"
    shutil.copytree(ROOT / "data" / "kb", kb)
    emb = HashingEmbedder()
    store = VectorStore(None, emb.name)
    kwargs = dict(
        kb_dir=kb,
        manifest=man,
        manifest_path=ROOT / "config" / "kb_manifest.yaml",
        embedder=emb,
        store=store,
        index_dir=tmp_path,
    )
    _, r1 = ingest(**kwargs)
    assert r1.embedded == r1.n_chunks == store.count()
    _, r2 = ingest(**kwargs)
    assert r2.embedded == 0 and r2.deleted == 0

    # edit one document: only its changed section is re-embedded, the stale one deleted
    p = kb / "05_company_updates.md"
    p.write_text(p.read_text().replace("4,200", "4,350"))
    _, r3 = ingest(**kwargs)
    assert r3.embedded == 1 and r3.deleted == 1 and store.count() == r1.n_chunks

    # remove a document entirely
    (kb / "14_annual_impact_report.md").unlink()
    man.documents.pop("14_annual_impact_report")
    _, r4 = ingest(**kwargs)
    assert r4.deleted == 4 and store.count() == r1.n_chunks - 4


def test_unmapped_document_is_auto_tagged(tmp_path):
    tax, man = _load()
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "99_new.md").write_text(
        "# SunGrid — Paperless Billing\n\n## E-statements\nBills can be delivered by email each month instead of paper statements.\n"
    )
    emb = HashingEmbedder()
    chunks, rep = ingest(
        kb_dir=kb,
        manifest=man,
        manifest_path=ROOT / "config" / "kb_manifest.yaml",
        embedder=emb,
        store=VectorStore(None, emb.name),
        index_dir=None,
        auto_tagger=lambda text: ("billing_account", 0.9),
    )
    assert rep.auto_tagged == ["99_new"]
    assert chunks.chunks[0].categories == ["billing_account"]
