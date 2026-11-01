"""Markdown-aware chunking.

Strategy (chosen for small, well-structured policy documents):
  * one chunk per `##` section — sections are self-contained policy units, so a
    section boundary is a far better split point than a fixed token window;
  * every chunk is prefixed with "<doc title> > <section>" (a contextual header) before
    embedding, so a short section like "Appeals Process" still carries the fact that it
    belongs to the *rebate* document;
  * markdown tables are linearised into sentences ("Household tier: Monthly Membership
    Fee $8; ...") so each row is retrievable and quotable on its own;
  * sections longer than `max_words` fall back to sentence-packing with a one-sentence
    overlap — not triggered by the current corpus, but needed at 10x volume.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import asdict, dataclass, field

from ..text import split_sentences

_REF_RE = re.compile(r"(?:the|see)\s+([A-Z][\w&/\- ]+?)\s+(?:document|materials)", re.IGNORECASE)
_NAMESPACE = uuid.UUID("6f1c2a52-6a2c-4c8e-9a43-0b5d2f1e7a10")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str  # body text, tables linearised (what we quote and cite)
    order: int
    categories: list[str] = field(default_factory=list)
    primary_category: str = ""
    ambiguous: bool = False
    references: list[str] = field(default_factory=list)  # other docs this chunk points to

    @property
    def header(self) -> str:
        return f"{self.doc_title} > {self.section}"

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"

    @property
    def citation(self) -> str:
        return f"{self.doc_id} § {self.section}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


def _linearise_table(lines: list[str]) -> list[str]:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)]
    if len(rows) < 2:
        return [" ".join(" ".join(r) for r in rows)]
    header, body = rows[0], rows[1:]
    suffix = " tier" if header[0].strip().lower() == "tier" else ""
    out = []
    for r in body:
        parts = [f"{h}: {v}" for h, v in zip(header[1:], r[1:])]
        out.append(f"{r[0]}{suffix} — " + "; ".join(parts) + ".")
    return out


def _normalise_body(raw: str) -> str:
    """Strip markdown emphasis and linearise tables, keeping list items as lines."""
    lines = raw.splitlines()
    out: list[str] = []
    table: list[str] = []
    for ln in lines + [""]:
        if ln.strip().startswith("|"):
            table.append(ln)
            continue
        if table:
            out.extend(_linearise_table(table))
            table = []
        ln = re.sub(r"\*\*(.+?)\*\*", r"\1", ln).rstrip()
        if ln.strip():
            out.append(ln.strip())
    return "\n".join(out)


def _split_long(text: str, max_words: int) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    sents = split_sentences(text)
    parts: list[str] = []
    cur: list[str] = []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            parts.append(" ".join(cur))
            cur = cur[-1:]  # one-sentence overlap
        cur.append(s)
    if cur:
        parts.append(" ".join(cur))
    return parts


def chunk_markdown(doc_id: str, markdown: str, *, max_words: int = 220) -> list[Chunk]:
    title = doc_id
    sections: list[tuple[str, list[str]]] = []
    current_name, current_lines = "Overview", []
    for line in markdown.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            if " — " in title:  # "SunGrid Cooperative — Billing & Account FAQs"
                title = title.split(" — ", 1)[1].strip()
            continue
        if line.startswith("## "):
            sections.append((current_name, current_lines))
            current_name, current_lines = line[3:].strip(), []
            continue
        current_lines.append(line)
    sections.append((current_name, current_lines))

    chunks: list[Chunk] = []
    for name, body_lines in sections:
        body = _normalise_body("\n".join(body_lines))
        if not body.strip():
            continue
        for part_idx, part in enumerate(_split_long(body, max_words)):
            section = name if part_idx == 0 else f"{name} (part {part_idx + 1})"
            digest = hashlib.sha1(f"{doc_id}|{section}|{part}".encode()).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(_NAMESPACE, digest)),
                    doc_id=doc_id,
                    doc_title=title,
                    section=section,
                    text=part,
                    order=len(chunks),
                    references=sorted({m.strip() for m in _REF_RE.findall(part)}),
                )
            )
    return chunks
