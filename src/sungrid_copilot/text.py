"""Small, dependency-free text utilities shared by BM25, the classifier and the guards."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself
    yourselves get got im i'm ive i've dont don't does doesnt let lets please tell know want
    need much many also us via sungrid sungrid's""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


def stem(token: str) -> str:
    """A deliberately light suffix stripper (plural / -ing / -ed / -ation).

    Good enough to align "rebates"/"rebate", "installing"/"install", "approved"/"approve";
    avoids a heavy NLP dependency.
    """
    t = token
    if len(t) <= 3 or t[0].isdigit():
        return t
    if t.endswith(("ibility", "ability")):
        return t[:-5] + "l"  # eligibility -> eligibl (matches "eligible")
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith(("sses", "xes", "zes", "ches", "shes")):
        t = t[:-2]
    elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    for suf in ("ations", "ation", "ments", "ment", "ing", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            break
    if t.endswith("e") and len(t) > 3:
        t = t[:-1]
    return t


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if not keep_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line:
            continue
        out.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
    return out


_NUM_RE = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Canonical numeric tokens in a text: '$4,000.00' -> '4000', '0.40' -> '0.4'."""
    out: set[str] = set()
    for m in _NUM_RE.findall(text):
        out.add(canonical_number(m))
    return out


def canonical_number(raw: str) -> str:
    s = raw.replace("$", "").replace(",", "").strip()
    try:
        v = float(s)
    except ValueError:
        return s
    return f"{v:g}" if v != int(v) else str(int(v))
