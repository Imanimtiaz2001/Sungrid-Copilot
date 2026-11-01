"""Single-label query classifier (5 KB categories + non_relevant).

Local model first, LLM only when it is unsure:
  1. A logistic-regression model over TF-IDF word/char n-grams (+ dense embeddings when a
     semantic embedder is available). Trained in < 1 s at start-up on: seed queries,
     category descriptions, and the KB chunks themselves (labelled via the manifest).
     Deterministic, free, ~1 ms per query, and it gives calibrated-ish probabilities.
  2. If the top-2 margin is small AND an LLM is configured, the LLM adjudicates between
     categories using the taxonomy descriptions.

The output category is used ONLY to build the retrieval filter (see copilot.py) - it never
chooses between pipelines. Intent (eligibility vs knowledge) is a separate decision.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .embeddings import Embedder
from .ingestion.chunker import Chunk
from .llm.base import LLMClient, LLMError
from .taxonomy import Taxonomy
from .text import tokenize

log = logging.getLogger(__name__)


@dataclass
class Classification:
    label: str
    confidence: float
    second: str
    margin: float
    scores: dict[str, float] = field(default_factory=dict)
    method: str = "local"
    domain_coverage: float = 1.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "second": self.second,
            "margin": round(self.margin, 3),
            "method": self.method,
            "domain_coverage": round(self.domain_coverage, 3),
            "scores": {k: round(v, 3) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])},
        }


def load_seed_examples(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append((d["text"], d["label"]))
    return out


def build_training_set(
    taxonomy: Taxonomy, seeds: list[tuple[str, str]], chunks: list[Chunk] | None
) -> list[tuple[str, str]]:
    data = [(t, taxonomy.validate(lab)) for t, lab in seeds]
    data += [(c.description, c.id) for c in taxonomy.categories]
    for ch in chunks or []:
        for cat in ch.categories:  # multi-label docs teach the model that they span categories
            data.append((ch.embed_text, cat))
            data.append((f"{ch.doc_title} {ch.section}", cat))
    return data


def _word_analyzer(s: str) -> list[str]:
    toks = tokenize(s)
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class QueryClassifier:
    def __init__(
        self,
        taxonomy: Taxonomy,
        *,
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        adjudicate_margin: float = 0.15,
    ):
        self.taxonomy = taxonomy
        self.embedder = embedder if (embedder is not None and getattr(embedder, "semantic", False)) else None
        self.llm = llm
        self.adjudicate_margin = adjudicate_margin
        self._word = TfidfVectorizer(analyzer=_word_analyzer, sublinear_tf=True, min_df=1)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self._clf = LogisticRegression(C=8.0, class_weight="balanced", max_iter=5000)
        self._domain_vocab: set[str] = set()
        self.labels: list[str] = []

    # ------------------------------------------------------------------ training
    def _features(self, texts: list[str], fit: bool = False):
        if fit:
            w = self._word.fit_transform(texts)
            c = self._char.fit_transform(texts)
        else:
            w = self._word.transform(texts)
            c = self._char.transform(texts)
        blocks = [w, c * 0.5]
        if self.embedder is not None:
            blocks.append(csr_matrix(self.embedder.embed_documents(texts)))
        return hstack(blocks).tocsr()

    def fit(self, examples: list[tuple[str, str]]) -> QueryClassifier:
        texts = [t for t, _ in examples]
        labels = [lab for _, lab in examples]
        self._clf.fit(self._features(texts, fit=True), labels)
        self.labels = list(self._clf.classes_)
        self._domain_vocab = {tok for t, lab in examples if lab != self.taxonomy.non_relevant_id for tok in tokenize(t)}
        return self

    # ------------------------------------------------------------------ inference
    def domain_coverage(self, query: str) -> float:
        toks = tokenize(query)
        if not toks:
            return 0.0
        return sum(t in self._domain_vocab for t in toks) / len(toks)

    def predict_local(self, query: str) -> Classification:
        probs = self._clf.predict_proba(self._features([query]))[0]
        order = np.argsort(-probs)
        top, second = order[0], order[1]
        scores = {lab: float(p) for lab, p in zip(self.labels, probs)}
        label = self.labels[top]
        cov = self.domain_coverage(query)
        # Guard against "confident" in-domain labels for queries with no KB vocabulary at all
        # (e.g. "hello" or a single unknown word): those are non_relevant by construction.
        if cov == 0.0 and label != self.taxonomy.non_relevant_id:
            label = self.taxonomy.non_relevant_id
        return Classification(
            label=label,
            confidence=float(probs[top]),
            second=self.labels[second] if label == self.labels[top] else self.labels[top],
            margin=float(probs[top] - probs[second]),
            scores=scores,
            domain_coverage=cov,
        )

    def predict(self, query: str) -> Classification:
        res = self.predict_local(query)
        if self.llm is None or res.margin >= self.adjudicate_margin:
            return res
        try:
            label = self._llm_adjudicate(query)
        except LLMError as exc:
            log.warning("LLM adjudication failed (%s); keeping local label", exc)
            return res
        if label and label != res.label:
            res.second, res.label = res.label, label
        res.method = "local+llm"
        return res

    def _llm_adjudicate(self, query: str) -> str | None:
        cats = "\n".join(f"- {c.id}: {c.description}" for c in self.taxonomy.categories)
        system = (
            "You route member questions for SunGrid Cooperative, a solar energy co-op. "
            "Pick the ONE category whose documents would best answer the question.\n"
            f"Categories:\n{cats}\n- {self.taxonomy.non_relevant_id}: anything unrelated to SunGrid, "
            "its programs, billing, installations or news.\n"
            'Reply with JSON only: {"category": "<id>"}'
        )
        raw = self.llm.complete(system=system, user=query, max_tokens=30)
        try:
            label = json.loads(raw[raw.index("{") : raw.rindex("}") + 1]).get("category", "")
        except (ValueError, AttributeError):
            label = raw.strip().strip('"').split()[0] if raw.strip() else ""
        return label if label in self.taxonomy.all_labels else None
