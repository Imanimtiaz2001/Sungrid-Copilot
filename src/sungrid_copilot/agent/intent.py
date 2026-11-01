"""Intent detection: does this message need the eligibility SERVICE or the knowledge base?

This is the agent's routing decision. It is deliberately separate from the category
classifier (which only shapes the retrieval filter).

Routing bias: the rules are tuned to prefer the eligibility path whenever a member asks
about THEIR eligibility or THEIR rebate amount. A false positive costs one clarifying
question; a false negative could let a free-text answer estimate a rebate without the
service's hard checks - which is the failure the brief forbids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Program = Literal["rooftop", "battery", "low_income", "commercial", "other"]

_PERSONAL_ELIG = re.compile(
    r"\b(?:am\s+i|are\s+we|is\s+my|would\s+i|will\s+i|do\s+i|do\s+we|does\s+my\s+\w+|can\s+i|could\s+i|may\s+i|would\s+we|can\s+we|i'?m|i\s+am)\b"
    r"[^.?!]{0,70}?\b(?:eligib\w*|qualif\w*|(?:get|receive|claim)\s+(?:the\s+|a\s+|my\s+|any\s+|this\s+)?(?:[\w-]+\s+){0,2}?(?:rebate|incentive|add-?on))",
    re.I,
)
_CHECK_ELIG = re.compile(
    r"\b(?:check|verify|confirm|determine|assess|see\s+if|see\s+whether|find\s+out\s+(?:if|whether))\b[^.?!]{0,40}\b(?:eligib|qualif)",
    re.I,
)
_MY_ELIG = re.compile(r"\bmy\s+(?:rebate\s+)?eligibility\b|\beligibility\s+check\b|\bcheck\s+eligibility\b", re.I)
_ESTIMATE = re.compile(
    r"\bhow\s+much\b(?!\s+(?:notice|time|longer|warning|later|earlier|sooner))[^.?!]{0,60}\b(?:rebate|incentive|get\s+back|would\s+i\s+get|will\s+i\s+get|could\s+i\s+get|receive)"
    r"|\b(?:estimate|calculate|compute|work\s+out)\b[^.?!]{0,40}\b(?:rebate|incentive|amount|what\s+i'?d\s+get)"
    r"|\bwhat\s+(?:would|will|could|do)\s+(?:i|we)\s+(?:get|receive|qualify\s+for)\b",
    re.I,
)
_PERSONAL_MARKER = re.compile(r"\b(?:i|i'?d|i'?m|me|my|we|our|us)\b|\d", re.I)
# Questions ABOUT a past determination or about the rules in general are knowledge questions.
_ABOUT_DETERMINATION = re.compile(
    r"\b(?:appeal\w*|challenge\w*|contest\w*|dispute\w*|denied|denial|rejected|revers\w*|clawe?d?\s*back|claw-?back|refund\w*)\b"
    r"|\beligibility\s+(?:rules|requirements|criteria|conditions|guidance|rules?\s+for)\b|\brely\s+on\b",
    re.I,
)
_RECHECK = re.compile(
    r"\b(?:actually|instead|correction|changed|what\s+if|now\s+it'?s|update|re-?check|try\s+again)\b", re.I
)

_PROGRAMS: list[tuple[Program, re.Pattern]] = [
    ("battery", re.compile(r"\bbatter(?:y|ies)\b|\bstorage\b|\badd-?on\b", re.I)),
    ("low_income", re.compile(r"low[- ]?income|bill\s+credit", re.I)),
    ("commercial", re.compile(r"\bcommercial\b|performance[- ]based|\bbusiness(?:es)?\b|carbon\s+markets", re.I)),
    (
        "other",
        re.compile(
            r"\bvot(?:e|ing)\b|\bboard\b|\bproxy\b|\bannual\s+meeting\b|\brun\s+for\b|\bgrievance\b|\btier\s+(?:change|transfer)\b|\bmembership\s+tier\b|multi-?family|apartment",
            re.I,
        ),
    ),
    ("rooftop", re.compile(r"\brooftop\b|\brebate\b|\bsolar\s+(?:incentive|rebate)\b", re.I)),
]


@dataclass
class Intent:
    kind: Literal["eligibility", "knowledge"]
    program: Program | None = None
    estimate_request: bool = False
    tool_supported: bool = True  # False -> member asked about personal eligibility we cannot check
    reasons: list[str] = field(default_factory=list)
    method: str = "rules"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "program": self.program,
            "estimate_request": self.estimate_request,
            "tool_supported": self.tool_supported,
            "reasons": self.reasons,
            "method": self.method,
        }


def detect_program(text: str) -> Program | None:
    for name, rx in _PROGRAMS:
        if rx.search(text):
            return name
    return None


def detect_intent(
    text: str,
    *,
    n_slots_in_message: int = 0,
    awaiting_slots: bool = False,
    had_eligibility_check: bool = False,
    session_program: Program | None = None,
) -> Intent:
    reasons: list[str] = []
    program = detect_program(text)
    personal = bool(_PERSONAL_ELIG.search(text) or _CHECK_ELIG.search(text) or _MY_ELIG.search(text))
    estimate = bool(_ESTIMATE.search(text)) and bool(_PERSONAL_MARKER.search(text))
    if personal:
        reasons.append("personal eligibility question")
    if estimate:
        reasons.append("personal rebate amount/estimate requested")

    eligibility_like = personal or estimate
    if eligibility_like and n_slots_in_message == 0 and _ABOUT_DETERMINATION.search(text):
        eligibility_like = False
        reasons.append("question about a past determination / the rules in general")
    if awaiting_slots and (n_slots_in_message > 0 or eligibility_like):
        reasons.append("answering a pending eligibility question")
        return Intent(
            "eligibility",
            program if program in ("rooftop", "battery") else (session_program or "rooftop"),
            estimate,
            True,
            reasons,
        )
    if (
        had_eligibility_check
        and n_slots_in_message > 0
        and (_RECHECK.search(text) or eligibility_like or n_slots_in_message >= 2)
    ):
        reasons.append("updated details after a previous eligibility check")
        return Intent(
            "eligibility",
            program if program in ("rooftop", "battery") else (session_program or "rooftop"),
            estimate,
            True,
            reasons,
        )
    if not eligibility_like and n_slots_in_message >= 2 and program in ("rooftop", "battery"):
        eligibility_like = True
        reasons.append("message supplies eligibility details for a rebate program")

    if not eligibility_like:
        return Intent("knowledge", program, False, True, reasons or ["general question"])

    if program in ("low_income", "commercial"):
        reasons.append(f"no eligibility service exists for the {program} program; answering from documents")
        return Intent("knowledge", program, estimate, False, reasons)
    if program == "other":
        # e.g. "am I eligible to vote" - not a rebate question at all
        return Intent("knowledge", program, False, True, [*reasons, "eligibility question about a non-rebate topic"])
    return Intent("eligibility", program or "rooftop", estimate, True, reasons)
