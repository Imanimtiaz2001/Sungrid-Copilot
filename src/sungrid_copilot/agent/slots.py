"""Deterministic extraction + validation of the eligibility tool's arguments.

Design rules (they matter for the hard gate):
  * Never guess a missing argument and never default one. A missing/uncertain value means
    "ask the member", not "assume". Defaulting `installer_approved=False` would wrongly
    deny; defaulting it to True would wrongly approve.
  * Every value an LLM proposes must be *grounded* in the member's own words (see
    `is_grounded`), so a hallucinated income can never reach the tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SLOT_ORDER = ("household_zip", "annual_income_usd", "system_size_kw", "installer_approved")
SLOT_QUESTIONS = {
    "household_zip": "your household's 5-digit ZIP code",
    "annual_income_usd": "your annual household income (USD)",
    "system_size_kw": "the solar system size in kW (e.g. 6 kW)",
    "installer_approved": "whether your installer is on SunGrid's approved installer list (yes / no)",
}


@dataclass
class SlotValues:
    household_zip: str | None = None
    annual_income_usd: float | None = None
    system_size_kw: float | None = None
    installer_approved: bool | None = None
    notes: list[str] = field(default_factory=list)  # interpretation notes shown to the member

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in SLOT_ORDER}

    def found(self) -> dict:
        return {k: v for k, v in self.as_dict().items() if v is not None}

    def missing(self) -> list[str]:
        return [k for k in SLOT_ORDER if getattr(self, k) is None]

    def merge(self, newer: SlotValues) -> SlotValues:
        """Newer values win (supports corrections like 'actually my income is 130k')."""
        out = SlotValues(**{k: getattr(self, k) for k in SLOT_ORDER})
        for k in SLOT_ORDER:
            v = getattr(newer, k)
            if v is not None:
                setattr(out, k, v)
        out.notes = list(newer.notes)
        return out


# --------------------------------------------------------------------------- patterns
_NUM = r"\d[\d,]*(?:\.\d+)?"
_INCOME_WORDS = r"(?:income|earn(?:s|ing)?|make|making|makes|salary|salaries|agi|gross|wage|wages)"
_COST_WORDS = re.compile(
    r"(?:paid|pay|cost|costs|price|priced|spent|spend|quote|quoted|loan|financ|bill|fee|rebate|credit|budget)\W*(?:\w+\W+){0,2}$",
    re.I,
)
_NOT_INCOME_RIGHT = re.compile(
    r"^\W*(?:rebate|incentive|credit|cap|discount|refund|bill|fee|charge|system|install|battery|loan|deposit)", re.I
)
_PERIOD_MONTH = re.compile(r"(?:per|a|/|each)\s*(?:month|mo)\b|monthly", re.I)

_SIZE_RE = re.compile(rf"(?P<n>{_NUM})\s*-?\s*(?P<u>kw|kilowatts?|kilo-watts?|w|watts?)(?![a-z])", re.I)
_ZIP_KEY_RE = re.compile(r"\b(?:zip|zipcode|zip\s*code|postal\s*code|postcode)\b\D{0,12}?(\d{5})(?:-\d{4})?\b", re.I)
_ZIP_LOC_RE = re.compile(
    r"\b(?:live|living|located|based|reside|residing|home|house|property|address)\b[^.\d$]{0,20}?\b(?:in|at|is)\s+(\d{5})(?:-\d{4})?\b",
    re.I,
)
_ZIP_IN_RE = re.compile(r"\bin\s+(\d{5})(?:-\d{4})?\b(?!\s*(?:k\b|dollars|usd|per|a year|/yr|annually))", re.I)
_MONEY_RE = re.compile(
    rf"(?P<cur>\$|usd\s*)?(?P<n>{_NUM})\s*(?P<mult>k\b|thousand\b|grand\b|m\b|million\b|mn\b)?\s*(?P<usd>dollars|usd)?",
    re.I,
)
_BARE5_RE = re.compile(
    r"(?<![\d$.,])(\d{5})(?!\d|[,.]\d|\s*(?:k\b|kw|w\b|watts?|dollars|usd|thousand|per\b|a\s+year|/\s*(?:yr|year|mo|month)))",
    re.I,
)

_INSTALLER_CTX = re.compile(r"install|contractor|electrician|approved list|certif|approved\b", re.I)
_UNSURE = re.compile(
    r"\b(?:not\s+sure|unsure|don'?t\s+know|do\s+not\s+know|no\s+idea|unclear|not\s+certain|can'?t\s+remember|maybe|might\s+be|i\s+think)\b",
    re.I,
)
_NEG_INSTALLER = re.compile(
    r"\b(?:not|isn'?t|wasn'?t|aren'?t|weren'?t|never|no\s+longer)\s+(?:\w+\s+){0,4}?(?:sungrid[- ]?)?(?:approved|certified|licensed|on\s+(?:the|sungrid'?s|your)\s+(?:approved\s+)?(?:installer\s+)?list)"
    r"|\b(?:un|non[- ]?)(?:approved|certified|licensed)\b"
    r"|certification\s+(?:had\s+|has\s+|was\s+)?(?:lapsed|expired|revoked)"
    r"|(?:lapsed|expired|revoked)\s+(?:\w+\s+)?certification"
    r"|removed\s+from\s+the\s+(?:approved\s+)?(?:installer\s+)?list"
    r"|installer\s*(?:approved)?\s*[:=]\s*(?:no|false|n)\b",
    re.I,
)
_POS_INSTALLER = re.compile(
    r"\b(?:sungrid[- ]?approved|approved\s+installer|certified\s+installer|approved\s+contractor|on\s+(?:the|sungrid'?s|your)\s+approved\s+(?:installer\s+)?list"
    r"|installer\s+(?:is\s+|was\s+|'s\s+)?(?:\w+\s+){0,2}?(?:approved|certified)\b|(?:is|was|are)\s+(?:sungrid[- ])?approved|installer\s*(?:approved)?\s*[:=]\s*(?:yes|true|y)\b)",
    re.I,
)
_YES = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|correct|they\s+are|it\s+is|he\s+is|she\s+is|sure|affirmative|approved)\b", re.I
)
_NO = re.compile(r"^\s*(?:no|nope|nah|not\s+approved|they'?re\s+not|it'?s\s+not|negative)\b", re.I)


def _to_float(n: str) -> float:
    return float(n.replace(",", ""))


def _extract_size(text: str, notes: list[str]) -> tuple[float | None, list[tuple[int, int]]]:
    spans = []
    value = None
    for m in _SIZE_RE.finditer(text):
        n, unit = _to_float(m.group("n")), m.group("u").lower()
        # "kWh"/"Wh" are energy (battery capacity), not system size: the regex excludes a
        # trailing letter, so "13.5 kWh" never matches here.
        kw = n if unit.startswith("k") else n / 1000.0
        if unit.startswith("w") and n < 100:  # "6 w" is almost certainly a typo for kW
            continue
        if unit.startswith("w"):
            notes.append(f"interpreted {m.group(0).strip()} as {kw:g} kW")
        value = kw
        spans.append(m.span())
    return value, spans


def _extract_zip(text: str, taken: list[tuple[int, int]]) -> tuple[str | None, list[tuple[int, int]]]:
    for rx in (_ZIP_KEY_RE, _ZIP_LOC_RE, _ZIP_IN_RE):
        m = rx.search(text)
        if m:
            return m.group(1), [m.span(1)]
    # bare 5-digit token (e.g. "94103, $80k, 6kW") — only if it is not money/size and is not
    # sitting right after an income word ("income 85000").
    for m in _BARE5_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        left = text[max(0, m.start() - 25) : m.start()].lower()
        if re.search(_INCOME_WORDS + r"\W*(?:is|of|about|around|roughly|:)?\W*$", left):
            continue
        return m.group(1), [m.span(1)]
    return None, []


def _extract_income(text: str, taken: list[tuple[int, int]], notes: list[str]) -> float | None:
    best = None
    for m in _MONEY_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        raw_n = m.group("n")
        has_money_marker = bool(m.group("cur") or m.group("mult") or m.group("usd") or "," in raw_n)
        left = text[max(0, m.start() - 45) : m.start()].lower()
        right = text[m.end() : m.end() + 30].lower()
        income_ctx = re.search(_INCOME_WORDS, left) or re.search(
            r"^\W*(?:per|a|/)\s*(?:year|yr|annum)|^\W*(?:annual|yearly|household income|income)", right
        )
        explicit_money = bool(m.group("cur") or m.group("mult") or m.group("usd"))
        not_income = (
            _COST_WORDS.search(left)
            or _NOT_INCOME_RIGHT.search(right)
            or re.search(r"\b(?:max(?:imum)?|cap|up\s+to|full)\W*$", left)
        )
        if not income_ctx and not (explicit_money and not not_income):
            continue
        if not has_money_marker and len(raw_n.replace(",", "").split(".")[0]) < 4:
            continue  # "income tier 2" etc.
        val = _to_float(raw_n)
        mult = (m.group("mult") or "").lower()
        if mult in {"k", "thousand", "grand"}:
            val *= 1_000
        elif mult in {"m", "million", "mn"}:
            val *= 1_000_000
        if _PERIOD_MONTH.search(right[:20]):
            notes.append(f"converted monthly income ${val:,.0f} to ${val * 12:,.0f}/year")
            val *= 12
        best = val
        break
    return best


def _extract_installer(text: str) -> bool | None:
    if not _INSTALLER_CTX.search(text):
        return None
    # look at installer-related clauses only
    clauses = [c for c in re.split(r"(?<=[.!?;])\s+|,\s+(?:and|but)\s+|\s+but\s+", text) if _INSTALLER_CTX.search(c)]
    for clause in clauses:
        if _NEG_INSTALLER.search(clause):
            return False
        if _UNSURE.search(clause):
            return None
        if _POS_INSTALLER.search(clause):
            return True
    return None


def extract_slots(text: str, *, awaiting: list[str] | None = None) -> SlotValues:
    """Extract eligibility arguments from one member message.

    `awaiting` lists the slots the agent asked for last turn, which lets a bare "yes"/"no"
    answer the installer question, or a bare "94103" answer the ZIP question.
    """
    notes: list[str] = []
    size, size_spans = _extract_size(text, notes)
    income_taken = list(size_spans)
    zip_code, zip_spans = _extract_zip(text, size_spans)
    income = _extract_income(text, income_taken + zip_spans, notes)
    installer = _extract_installer(text)
    if installer is False and re.search(r"lapsed|expired", text, re.I):
        notes.append(
            "a lapsed certification counts as a non-approved installer for rebate purposes "
            "(07_installer_certification § Annual Renewal)"
        )

    awaiting = awaiting or []
    if installer is None and awaiting == ["installer_approved"]:
        if _NO.search(text):
            installer = False
        elif _YES.search(text):
            installer = True
    if income is None and awaiting == ["annual_income_usd"]:
        m = re.fullmatch(rf"\s*\$?\s*({_NUM})\s*(k|thousand)?\s*(?:/\s*yr|per year|a year)?\s*\.?\s*", text, re.I)
        if m:
            income = _to_float(m.group(1)) * (1000 if m.group(2) else 1)
            if zip_code and zip_code == m.group(1):
                zip_code = None
    if size is None and awaiting == ["system_size_kw"]:
        m = re.fullmatch(rf"\s*({_NUM})\s*\.?\s*", text)
        if m:
            size = _to_float(m.group(1))
            notes.append(f"interpreted {m.group(1)} as {size:g} kW")

    return SlotValues(
        household_zip=zip_code,
        annual_income_usd=income,
        system_size_kw=size,
        installer_approved=installer,
        notes=notes,
    )


# --------------------------------------------------------------------------- validation
def validate(slots: SlotValues) -> dict[str, str]:
    """Return {slot: problem} for values that are present but unusable."""
    problems: dict[str, str] = {}
    if slots.household_zip is not None and not re.fullmatch(r"\d{5}", slots.household_zip):
        problems["household_zip"] = "a ZIP code must be exactly 5 digits"
    if slots.annual_income_usd is not None and not (0 <= slots.annual_income_usd < 100_000_000):
        problems["annual_income_usd"] = "income must be a non-negative amount in USD"
    if slots.system_size_kw is not None and not (0 < slots.system_size_kw < 100_000):
        problems["system_size_kw"] = "system size must be a positive number of kW"
    return problems


def is_grounded(slot: str, value, text: str) -> bool:
    """True if an LLM-proposed value is literally supported by the member's text."""
    if value is None:
        return True
    t = text.lower().replace(",", "")
    if slot == "household_zip":
        return str(value) in t
    if slot == "installer_approved":
        return bool(_INSTALLER_CTX.search(text)) or bool(_YES.search(text) or _NO.search(text))
    v = float(value)
    candidates = {f"{v:g}", f"{int(v)}" if v == int(v) else f"{v:g}"}
    if slot == "annual_income_usd":
        candidates |= {f"{v / 1000:g}k", f"{v / 1000:g} k", f"{v / 1000:g} thousand", f"{v / 12:g}", f"{v / 12000:g}k"}
        candidates |= {f"{v / 1_000_000:g}m", f"{v / 1_000_000:g} million"}
    if slot == "system_size_kw":
        candidates |= {f"{v * 1000:g}"}
    return any(c in t for c in candidates)
