"""The SunGrid Copilot agent (orchestrator).

Every request runs the same explicit steps; only the middle branch differs:

    1. classify      category (single label + confidence)    -> retrieval filter only
    2. understand    intent + eligibility arguments            -> chooses the branch
    3a. eligibility  validate args -> [ask for missing]  |  call TOOL -> HARD GATE -> policy lookup -> templated answer
    3b. knowledge    retrieve (category-filtered, hybrid) -> generate (LLM | extractive) -> verify (citations, numbers)
    4. guard         final output checks; every step is recorded in `trace`

The eligibility tool call is its own step and component (agent/tools.py); retrieval never
calls it, and the tool never sees retrieved text.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..classifier import Classification, QueryClassifier
from ..config import Settings
from ..llm.base import LLMClient, LLMError
from ..retrieval import HybridRetriever, RetrievalResult
from ..store import ChunkStore
from ..taxonomy import Manifest, Taxonomy
from ..text import canonical_number
from . import compose
from .guards import gate_violations, money_amounts, strip_invalid_citations, ungrounded_numbers
from .intent import Intent, detect_intent, detect_program
from .slots import SLOT_ORDER, SlotValues, extract_slots, is_grounded, validate
from .tools import EligibilityTool, ToolCall

log = logging.getLogger(__name__)

_SMALLTALK = re.compile(
    r"^\s*(?:hi|hello|hey|hiya|good\s+(?:morning|afternoon|evening)|thanks?|thank\s+you|cheers|bye|goodbye)\b[\s!.?,]*(?:there|copilot|sungrid)?[\s!.?]*$",
    re.I,
)
_FOLLOWUP = re.compile(r"^\s*(?:and|what\s+about|how\s+about|same\s+for|what\s+if|also)\b", re.I)
_RESET = re.compile(r"^\s*(?:reset|start\s+over|new\s+question|clear)\s*[.!]?\s*$", re.I)

UNDERSTAND_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["eligibility_check", "knowledge_question"],
            "description": "eligibility_check if the member asks whether THEY qualify for a rebate/incentive or how much THEY would receive; knowledge_question for general questions about policies, rules, billing, installation or news.",
        },
        "program": {"type": "string", "enum": ["rooftop", "battery", "low_income", "commercial", "other", "unknown"]},
        "household_zip": {"type": "string", "description": "5-digit ZIP if stated, else empty string"},
        "annual_income_usd": {
            "type": "number",
            "description": "Annual household income in USD if stated (convert 85k -> 85000, monthly x12), else -1",
        },
        "system_size_kw": {"type": "number", "description": "System size in kW if stated (6500 W -> 6.5), else -1"},
        "installer_approved": {
            "type": "string",
            "enum": ["yes", "no", "unknown"],
            "description": "yes only if the member says the installer is SunGrid-approved/certified; no if not approved or certification lapsed; unknown otherwise",
        },
    },
    "required": ["intent", "program"],
}


# --------------------------------------------------------------------------- state
@dataclass
class Session:
    id: str
    slots: SlotValues = field(default_factory=SlotValues)
    awaiting: list[str] = field(default_factory=list)
    program: str | None = None
    last_tool: ToolCall | None = None
    last_knowledge_query: str | None = None
    turns: int = 0


class SessionStore:
    """In-memory LRU session store (swap for Redis in production)."""

    def __init__(self, max_sessions: int = 2000):
        self._data: OrderedDict[str, Session] = OrderedDict()
        self._max = max_sessions
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> Session:
        with self._lock:
            sid = session_id or uuid.uuid4().hex[:12]
            s = self._data.pop(sid, None) or Session(id=sid)
            self._data[sid] = s
            while len(self._data) > self._max:
                self._data.popitem(last=False)
            return s

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


@dataclass
class AgentResponse:
    answer: str
    route: str  # eligibility_tool | eligibility_clarify | knowledge | abstain | smalltalk
    session_id: str
    category: dict[str, Any]
    intent: dict[str, Any]
    citations: list[str] = field(default_factory=list)
    tool_call: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "route": self.route,
            "session_id": self.session_id,
            "category": self.category,
            "intent": self.intent,
            "citations": self.citations,
            "tool_call": self.tool_call,
            "retrieval": self.retrieval,
            "trace": self.trace,
            "latency_ms": round(self.latency_ms, 1),
        }


class _Trace:
    def __init__(self):
        self.steps: list[dict[str, Any]] = []
        self._t = time.perf_counter()

    def add(self, step: str, **detail: Any) -> None:
        now = time.perf_counter()
        self.steps.append({"step": step, "ms": round((now - self._t) * 1000, 2), **detail})
        self._t = now


# --------------------------------------------------------------------------- agent
class Copilot:
    def __init__(
        self,
        *,
        settings: Settings,
        taxonomy: Taxonomy,
        manifest: Manifest,
        chunks: ChunkStore,
        retriever: HybridRetriever,
        classifier: QueryClassifier,
        tool: EligibilityTool,
        llm: LLMClient | None = None,
    ):
        self.settings = settings
        self.taxonomy = taxonomy
        self.manifest = manifest
        self.chunks = chunks
        self.retriever = retriever
        self.classifier = classifier
        self.tool = tool
        self.llm = llm
        self.sessions = SessionStore()

    # ------------------------------------------------------------------ public
    def ask(self, query: str, session_id: str | None = None) -> AgentResponse:
        t0 = time.perf_counter()
        trace = _Trace()
        session = self.sessions.get(session_id)
        session.turns += 1
        query = (query or "").strip()[:2000]

        if _RESET.match(query):
            self.sessions.reset(session.id)
            session = self.sessions.get(session.id)
            return self._finish(
                AgentResponse(
                    "Okay — I've cleared the details from this conversation. What would you like to know?",
                    "smalltalk",
                    session.id,
                    {},
                    {},
                ),
                trace,
                t0,
            )

        if not query:
            return self._finish(AgentResponse("Please type a question.", "abstain", session.id, {}, {}), trace, t0)

        # short follow-ups ("what about commercial?") inherit the previous knowledge question
        effective = query
        if session.last_knowledge_query and _FOLLOWUP.match(query) and len(query.split()) <= 8 and not session.awaiting:
            effective = f"{session.last_knowledge_query} {query}"
            trace.add("condense_followup", effective_query=effective)

        # 1. classify ------------------------------------------------------------
        cls = self.classifier.predict(effective)
        trace.add("classify", **cls.to_dict())

        if _SMALLTALK.match(query):
            answer = (
                "Hi! I'm SunGrid's member-support copilot. I can answer questions about membership & governance, "
                "rebates and incentives, billing & your account, installation & technical guidance, and company news — "
                "and I can check your Rooftop Rebate eligibility with SunGrid's eligibility service."
            )
            return self._finish(AgentResponse(answer, "smalltalk", session.id, cls.to_dict(), {}), trace, t0)

        # 2. understand ----------------------------------------------------------
        msg_slots = extract_slots(query, awaiting=session.awaiting)
        intent = detect_intent(
            query,
            n_slots_in_message=len(msg_slots.found()),
            awaiting_slots=bool(session.awaiting),
            had_eligibility_check=session.last_tool is not None,
            session_program=session.program,  # type: ignore[arg-type]
        )
        if self.llm is not None and self._should_consult_llm(query, intent, session, msg_slots):
            intent, msg_slots = self._llm_understand(query, intent, msg_slots, trace)
        trace.add("understand", intent=intent.to_dict(), slots_in_message=msg_slots.found())

        # 3. branch ----------------------------------------------------------------
        if intent.kind == "eligibility":
            resp = self._eligibility_flow(query, cls, intent, msg_slots, session, trace)
        else:
            resp = self._knowledge_flow(query, effective, cls, intent, session, trace)
        return self._finish(resp, trace, t0)

    # ------------------------------------------------------------------ eligibility
    def _eligibility_flow(
        self, query: str, cls: Classification, intent: Intent, msg_slots: SlotValues, session: Session, trace: _Trace
    ) -> AgentResponse:
        slots = session.slots.merge(msg_slots)
        problems = validate(slots)
        for slot in problems:
            setattr(slots, slot, None)
        session.slots = slots
        session.program = intent.program or session.program or "rooftop"
        program = session.program
        missing = slots.missing()
        trace.add("validate_arguments", have=slots.found(), missing=missing, problems=problems)

        if missing:
            session.awaiting = missing
            answer = compose.compose_clarification(slots, missing, problems, program, intent.estimate_request)
            answer = self._gate(answer, eligible=None, slots=slots, allowed_extra=money_amounts(query), trace=trace)
            return AgentResponse(answer, "eligibility_clarify", session.id, cls.to_dict(), intent.to_dict())

        session.awaiting = []
        # ---- distinct tool step -------------------------------------------------
        call = self.tool.invoke(**{k: getattr(slots, k) for k in SLOT_ORDER})
        session.last_tool = call
        trace.add("tool_call", **call.to_dict())

        # ---- hard gate ------------------------------------------------------------
        eligible = bool(call.status == "ok" and call.result and call.result["eligible"])
        trace.add("hard_gate", decision="pass" if eligible else ("fail_closed" if call.status != "ok" else "fail"))

        # ---- policy lookup (grounded follow-ups, configured in kb_manifest.yaml) ----
        fu = self.manifest.eligibility_followups
        refs = fu.get("eligible" if eligible else "ineligible", []) if call.status == "ok" else []
        followups = [c for c in (self.chunks.section(r) for r in refs) if c is not None]
        battery = (
            [c for c in (self.chunks.section(r) for r in fu.get("battery_terms", [])) if c is not None]
            if program == "battery"
            else []
        )
        trace.add("policy_lookup", sections=[c.citation for c in followups + battery])

        answer, cites = compose.compose_eligibility(call, slots, program, followups, battery)
        allowed = money_amounts(query)  # amounts the member stated themselves may be echoed
        if eligible:
            allowed.add(canonical_number(str(call.result["estimated_rebate_usd"])))
            for c in followups + battery:
                allowed |= money_amounts(c.text)
        answer = self._gate(answer, eligible=eligible, slots=slots, allowed_extra=allowed, trace=trace)
        if cites:
            answer += "\n\nSources:\n" + "\n".join(f"[{i}] {c}" for i, c in enumerate(cites, 1))
        return AgentResponse(
            answer,
            "eligibility_tool",
            session.id,
            cls.to_dict(),
            intent.to_dict(),
            citations=cites,
            tool_call=call.to_dict(),
        )

    def _gate(
        self, answer: str, *, eligible: bool | None, slots: SlotValues, allowed_extra: set[str], trace: _Trace
    ) -> str:
        """Last line of defence for the hard gate. With templated answers this never fires
        (the test-suite asserts that); if a future template change ever leaked an amount,
        the member gets a safe, amount-free answer instead."""
        allowed = set(allowed_extra)
        if slots.annual_income_usd is not None:
            allowed.add(canonical_number(str(slots.annual_income_usd)))
        problems = gate_violations(answer, eligible=eligible, allowed_amounts=allowed)
        trace.add("gate_guard", ok=not problems, problems=problems)
        if not problems:
            return answer
        log.error("Hard-gate guard replaced an answer: %s", problems)
        return (
            "I can't give a rebate amount for this request: SunGrid's eligibility service did not approve it. "
            "Please contact the Incentive Program committee for details."
        )

    # ------------------------------------------------------------------ knowledge
    def is_off_topic(self, cls: Classification) -> bool:
        """Confidently non_relevant. A near-tie with an in-domain label is treated as in-domain."""
        return cls.label == self.taxonomy.non_relevant_id and not (
            cls.margin < self.settings.filter_widen_margin
            and cls.second != self.taxonomy.non_relevant_id
            and cls.domain_coverage > 0
        )

    def filter_for(self, cls: Classification) -> list[str] | None:
        if cls.label == self.taxonomy.non_relevant_id:
            return [cls.second] if not self.is_off_topic(cls) else None
        cats = [cls.label]
        if cls.margin < self.settings.filter_widen_margin and cls.second != self.taxonomy.non_relevant_id:
            cats.append(cls.second)
        return cats

    def _knowledge_flow(
        self, query: str, retrieval_query: str, cls: Classification, intent: Intent, session: Session, trace: _Trace
    ) -> AgentResponse:
        s = self.settings

        cats = self.filter_for(cls)
        res = self.retriever.search_with_policy(retrieval_query, cats, k=s.top_k, min_evidence=s.min_evidence)
        trace.add(
            "retrieve",
            filter=cats,
            fallback=res.fallback_used,
            evidence=round(res.evidence, 3),
            notes=res.notes,
            hits=[h.to_dict() for h in res.hits],
        )
        retrieval_info = {
            "filter": cats,
            "fallback_used": res.fallback_used,
            "evidence": round(res.evidence, 3),
            "hits": [h.to_dict() for h in res.hits],
        }

        # abstain: evidence-based, so a misclassified in-domain question is still answered
        off_topic = self.is_off_topic(cls)
        semantic_support = (
            self.retriever.embedder.semantic and max((h.dense for h in res.hits), default=0.0) >= s.semantic_support
        )
        weak = res.evidence < s.abstain_evidence and not semantic_support
        if not res.hits or weak or (off_topic and res.evidence < s.off_topic_evidence):
            trace.add("abstain", reason="off_topic" if off_topic else "no_supporting_evidence")
            answer = (
                "I'm SunGrid's member-support copilot, so I can only help with SunGrid topics — membership & governance, "
                "rebates and incentives, billing & account, installation & technical guidance, and company news."
                if off_topic
                else "I couldn't find this in SunGrid's knowledge base, so I'd rather not guess. "
                "Please contact SunGrid member support through the member portal."
            )
            return AgentResponse(
                answer, "abstain", session.id, cls.to_dict(), intent.to_dict(), retrieval=retrieval_info
            )

        note = None
        if not intent.tool_supported and intent.program in ("low_income", "commercial"):
            note = (
                f"I can't determine personal eligibility for the {compose.PROGRAM_NAMES[intent.program]} — SunGrid's "
                "eligibility service only covers the Rooftop Rebate (and the Battery Add-On that depends on it). "
                "Here's what the policy says:"
            )

        question = query if retrieval_query == query else f"{query} (follow-up to: {session.last_knowledge_query})"
        answer, used = self._generate(question, retrieval_query, res, note, trace)
        session.last_knowledge_query = retrieval_query
        answer = (note + "\n\n" if note else "") + answer + compose.sources_footer(res.hits, used)
        cites = [res.hits[n - 1].chunk.citation for n in used]
        return AgentResponse(
            answer, "knowledge", session.id, cls.to_dict(), intent.to_dict(), citations=cites, retrieval=retrieval_info
        )

    def _generate(
        self, query: str, retrieval_query: str, res: RetrievalResult, note: str | None, trace: _Trace
    ) -> tuple[str, list[int]]:
        hits = res.hits
        if self.llm is not None:
            try:
                raw = self.llm.complete(
                    system=compose.ANSWER_SYSTEM_PROMPT,
                    user=compose.build_user_prompt(query, hits, note),
                    max_tokens=500,
                )
                text, used = strip_invalid_citations(raw, len(hits))
                bad_numbers = ungrounded_numbers(text, context=compose.build_context(hits), query=query)
                trace.add("verify", citations=used, ungrounded_numbers=sorted(bad_numbers))
                if text and not bad_numbers:
                    if not used:  # uncited answer: attribute to the top source rather than none
                        used = [1]
                    trace.add("generate", mode="llm", model=self.llm.name)
                    return text, used
                trace.add("generate_rejected", reason="ungrounded numbers" if bad_numbers else "empty")
            except LLMError as exc:
                trace.add("generate_failed", error=str(exc)[:200])
        text, used = compose.extractive_answer(retrieval_query, hits, self.retriever)
        trace.add("generate", mode="extractive", sources=used)
        return "Here's what SunGrid's knowledge base says:\n" + text, used

    # ------------------------------------------------------------------ LLM understanding
    @staticmethod
    def _should_consult_llm(query: str, intent: Intent, session: Session, slots: SlotValues) -> bool:
        if intent.kind == "eligibility" or session.awaiting or slots.found():
            return True
        return bool(re.search(r"eligib|qualif|rebate|incentive|how much|estimate", query, re.I))

    def _llm_understand(
        self, query: str, intent: Intent, slots: SlotValues, trace: _Trace
    ) -> tuple[Intent, SlotValues]:
        try:
            out = self.llm.call_tool(
                system="Extract routing information from a SunGrid Cooperative member's message. Only use values the member explicitly states.",
                user=query,
                tool_name="route_member_message",
                tool_description="Record the member's intent and any eligibility details they stated.",
                parameters=UNDERSTAND_SCHEMA,
            )
        except LLMError as exc:
            trace.add("llm_understand_failed", error=str(exc)[:200])
            return intent, slots

        rules_program = detect_program(query)
        llm_elig = out.get("intent") == "eligibility_check" and out.get("program") in ("rooftop", "battery", "unknown")
        if intent.kind == "knowledge" and llm_elig and rules_program not in ("low_income", "commercial", "other"):
            intent = Intent(
                "eligibility",
                out.get("program") if out.get("program") in ("rooftop", "battery") else "rooftop",
                intent.estimate_request,
                True,
                [*intent.reasons, "LLM detected a personal eligibility request"],
                "rules+llm",
            )
        else:
            intent.method = "rules+llm"

        # Fill only slots the rules missed, and only with values grounded in the member's text.
        proposed = {
            "household_zip": (out.get("household_zip") or "").strip() or None,
            "annual_income_usd": out.get("annual_income_usd")
            if isinstance(out.get("annual_income_usd"), (int, float)) and out.get("annual_income_usd") >= 0
            else None,
            "system_size_kw": out.get("system_size_kw")
            if isinstance(out.get("system_size_kw"), (int, float)) and out.get("system_size_kw") > 0
            else None,
            "installer_approved": {"yes": True, "no": False}.get(str(out.get("installer_approved", "unknown")).lower()),
        }
        rejected = []
        for k, v in proposed.items():
            if getattr(slots, k) is None and v is not None:
                if is_grounded(k, v, query) and not (
                    k == "installer_approved" and re.search(r"not sure|unsure|don'?t know|no idea", query, re.I)
                ):
                    setattr(slots, k, v)
                else:
                    rejected.append(k)
        trace.add("llm_understand", intent=out.get("intent"), program=out.get("program"), rejected_ungrounded=rejected)
        return intent, slots

    # ------------------------------------------------------------------ util
    @staticmethod
    def _finish(resp: AgentResponse, trace: _Trace, t0: float) -> AgentResponse:
        resp.trace = trace.steps
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        return resp
