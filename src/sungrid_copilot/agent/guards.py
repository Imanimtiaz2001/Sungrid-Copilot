"""Output guards. Cheap, deterministic checks that run on every answer."""

from __future__ import annotations

import re

from ..text import canonical_number, numbers_in

_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_ESTIMATE_PHRASES = re.compile(
    r"\b(?:estimated rebate|you(?:'d| would) (?:get|receive)|rebate of \$|would qualify for \$)", re.I
)


def money_amounts(text: str) -> set[str]:
    return {canonical_number(m) for m in _MONEY_RE.findall(text)}


def gate_violations(answer: str, *, eligible: bool | None, allowed_amounts: set[str]) -> list[str]:
    """Hard-gate check for eligibility answers.

    For a failed (or unavailable) eligibility decision, the answer must not contain ANY
    currency amount except values the member supplied (their income) - no estimates, no
    "you would have received", no partial amounts.
    """
    problems = []
    for amt in money_amounts(answer) - allowed_amounts:
        problems.append(f"unexpected currency amount ${amt}")
    if eligible is not True and _ESTIMATE_PHRASES.search(answer):
        problems.append("estimate language in a non-eligible answer")
    return problems


def citation_numbers(answer: str) -> list[int]:
    return [int(n) for n in _CITE_RE.findall(answer)]


def strip_invalid_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_sources else ""

    cleaned = _CITE_RE.sub(repl, answer)
    return cleaned, sorted(set(citation_numbers(cleaned)))


def ungrounded_numbers(answer: str, *, context: str, query: str) -> set[str]:
    """Numbers in the answer that appear neither in the retrieved context nor the question.

    Catches hallucinated figures AND computed figures (e.g. '6 kW x $0.40/W = $2,400'),
    which is how a free-text answer could sneak a rebate estimate past the eligibility gate.
    """
    body = _CITE_RE.sub("", answer)
    allowed = numbers_in(context) | numbers_in(query)
    return {n for n in numbers_in(body) if n not in allowed and n not in {"0", "1", "2", "3"}}
