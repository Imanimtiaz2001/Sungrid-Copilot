import sys
import types

import numpy as np

from conftest import offline_settings
from sungrid_copilot.classifier import QueryClassifier, build_training_set
from sungrid_copilot.embeddings import FastEmbedEmbedder, HashingEmbedder, build_embedder


class _FakeTextEmbedding:
    """Mimics fastembed.TextEmbedding (generators of float32 vectors)."""

    def __init__(self, model_name, cache_dir=None):
        self.model_name = model_name

    def _vec(self, t):
        rng = np.random.default_rng(abs(hash(t)) % 2**32)
        return rng.normal(size=384).astype(np.float32)

    def query_embed(self, q):
        qs = [q] if isinstance(q, str) else q
        return (self._vec("q:" + x) for x in qs)

    def passage_embed(self, texts):
        return (self._vec("p:" + x) for x in texts)


def test_fastembed_adapter_shapes_and_normalisation(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding))
    e = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    docs = e.embed_documents(["a", "b", "c"])
    assert docs.shape == (3, 384) and e.dim == 384
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert np.isclose(np.linalg.norm(e.embed_query("x")), 1.0)
    assert e.semantic and e.name == "fastembed:BAAI/bge-small-en-v1.5"


def test_auto_falls_back_to_hashing_when_no_model_can_load(monkeypatch):
    def broken(*a, **k):
        raise OSError("no network")

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=broken))
    e = build_embedder(offline_settings(embedding_provider="auto", openai_api_key=""))
    assert isinstance(e, HashingEmbedder)


def test_hashing_embedder_is_deterministic_and_lexically_sensible():
    e = HashingEmbedder()
    a, b = e.embed_query("rebate appeal deadline"), e.embed_query("rebate appeal deadline")
    assert np.array_equal(a, b)
    d = e.embed_documents(["Members may appeal a rebate determination within 30 days.", "Bills are issued monthly."])
    assert d[0] @ a > d[1] @ a


def test_classifier_uses_dense_features_with_a_semantic_embedder(monkeypatch, components):
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding))
    tax = components.copilot.taxonomy
    clf = QueryClassifier(tax, embedder=FastEmbedEmbedder())
    clf.fit(
        build_training_set(
            tax,
            [
                ("When is my bill due?", "billing_account"),
                ("hi", "non_relevant"),
                ("What is fault E07?", "technical_installation"),
            ],
            components.copilot.chunks.chunks,
        )
    )
    assert clf.embedder is not None
    assert clf.predict("When is my bill due?").label in tax.all_labels
