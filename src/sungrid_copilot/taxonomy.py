"""Category taxonomy and the document manifest (both config-driven)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CategoryDef:
    id: str
    label: str
    description: str


@dataclass(frozen=True)
class Taxonomy:
    categories: tuple[CategoryDef, ...]
    non_relevant_id: str = "non_relevant"

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.categories]

    @property
    def all_labels(self) -> list[str]:
        """In-domain categories plus the non_relevant label."""
        return [*self.ids, self.non_relevant_id]

    def label(self, cat_id: str) -> str:
        for c in self.categories:
            if c.id == cat_id:
                return c.label
        return "Non-relevant" if cat_id == self.non_relevant_id else cat_id

    def validate(self, cat_id: str) -> str:
        if cat_id not in self.all_labels:
            raise ValueError(f"Unknown category '{cat_id}'. Known: {self.all_labels}")
        return cat_id

    @classmethod
    def load(cls, path: Path) -> Taxonomy:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cats = tuple(
            CategoryDef(id=c["id"], label=c["label"], description=" ".join(c["description"].split()))
            for c in raw["categories"]
        )
        ids = [c.id for c in cats]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate category ids in taxonomy")
        return cls(categories=cats, non_relevant_id=raw.get("non_relevant_id", "non_relevant"))


@dataclass
class DocEntry:
    doc_id: str
    primary: str
    categories: list[str]
    ambiguous: bool = False
    rationale: str = ""
    auto_tagged: bool = False


@dataclass
class Manifest:
    documents: dict[str, DocEntry]
    eligibility_followups: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, taxonomy: Taxonomy) -> Manifest:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        docs: dict[str, DocEntry] = {}
        for doc_id, spec in (raw.get("documents") or {}).items():
            cats = [taxonomy.validate(c) for c in spec["categories"]]
            primary = taxonomy.validate(spec.get("primary", cats[0]))
            if primary not in cats:
                raise ValueError(f"{doc_id}: primary '{primary}' must be one of its categories")
            docs[doc_id] = DocEntry(
                doc_id=doc_id,
                primary=primary,
                categories=cats,
                ambiguous=bool(spec.get("ambiguous", False)) or len(cats) > 1,
                rationale=" ".join(str(spec.get("rationale", "")).split()),
            )
        return cls(documents=docs, eligibility_followups=raw.get("eligibility_followups") or {})
