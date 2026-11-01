"""Offline evaluation harness.

Runs the golden set (data/eval/golden_set.jsonl) through the full agent and reports:
classification accuracy + confusion matrix, intent/tool-routing accuracy, retrieval hit@k
and MRR (with a filter ablation), answer fact recall, abstention quality, eligibility
decision accuracy, hard-gate violations, and latency. Also audits document categories
(how a seed-only classifier sees each document) - which surfaces the ambiguous document.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent.copilot import Copilot
from .agent.guards import gate_violations, money_amounts
from .classifier import QueryClassifier, build_training_set, load_seed_examples
from .text import canonical_number

# CI quality gates (offline mode). The hard-gate threshold is absolute.
THRESHOLDS = {
    "classification_accuracy": 0.85,
    "intent_accuracy": 0.95,
    "retrieval_hit_at_k": 0.90,
    "eligibility_decision_accuracy": 1.0,
    "hard_gate_violations": 0,
}


@dataclass
class CaseResult:
    id: str
    passed: bool
    failures: list[str]
    route: str
    category: str
    expected_category: list[str] | None
    latency_ms: float
    retrieved_docs: list[str]


def load_golden(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _contains_any(text: str, needles: list[str]) -> bool:
    t = text.lower().replace("‑", "-")
    return any(n.lower() in t for n in needles)


def run_case(copilot: Copilot, case: dict[str, Any]) -> tuple[CaseResult, dict[str, Any]]:
    exp = case["expect"]
    sid = f"eval-{case['id']}"
    copilot.sessions.reset(sid)
    responses = [copilot.ask(turn, sid) for turn in case["turns"]]
    first, last = responses[0], responses[-1]
    failures: list[str] = []

    cat = first.category.get("label", "")
    if exp.get("category") and cat not in exp["category"]:
        failures.append(f"category {cat} not in {exp['category']}")
    if exp.get("intent") and last.intent.get("kind") and last.intent["kind"] != exp["intent"]:
        failures.append(f"intent {last.intent.get('kind')} != {exp['intent']}")
    if exp.get("route") and last.route != exp["route"]:
        failures.append(f"route {last.route} != {exp['route']}")

    tool_called = last.tool_call is not None
    if "tool_called" in exp and tool_called != exp["tool_called"]:
        failures.append(f"tool_called {tool_called} != {exp['tool_called']}")

    result = (last.tool_call or {}).get("result") or {}
    if "eligible" in exp and result.get("eligible") != exp["eligible"]:
        failures.append(f"eligible {result.get('eligible')} != {exp['eligible']}")
    if "rebate" in exp and abs(float(result.get("estimated_rebate_usd", -1)) - exp["rebate"]) > 0.01:
        failures.append(f"rebate {result.get('estimated_rebate_usd')} != {exp['rebate']}")
    if "reason_contains" in exp and exp["reason_contains"].lower() not in str(result.get("reason", "")).lower():
        failures.append(f"reason '{result.get('reason')}' lacks '{exp['reason_contains']}'")
    if "asks_for" in exp:
        asked = next((s for s in reversed(last.trace) if s["step"] == "validate_arguments"), {}).get("missing", [])
        if sorted(asked) != sorted(exp["asks_for"]):
            failures.append(f"asked for {asked} != {exp['asks_for']}")
    for bad in exp.get("must_not_contain", []):
        if bad.lower() in last.answer.lower():
            failures.append(f"answer contains forbidden '{bad}'")
    if exp.get("contains_any") and not _contains_any(last.answer, exp["contains_any"]):
        failures.append(f"answer lacks any of {exp['contains_any']}")

    docs = [h["citation"].split(" § ")[0] for h in (last.retrieval or {}).get("hits", [])]
    if exp.get("docs") and not set(exp["docs"]) & set(docs):
        failures.append(f"retrieved {docs} lacks {exp['docs']}")

    # hard gate: any non-eligible eligibility answer must contain no amount except the member's income
    gate_problems: list[str] = []
    if last.route.startswith("eligibility") and result.get("eligible") is not True:
        income = (last.tool_call or {}).get("arguments", {}).get("annual_income_usd")
        allowed = {canonical_number(str(income))} if income is not None else set()
        allowed |= money_amounts(case["turns"][-1])
        for s in last.trace:
            if s["step"] == "validate_arguments" and s["have"].get("annual_income_usd") is not None:
                allowed.add(canonical_number(str(s["have"]["annual_income_usd"])))
        gate_problems = gate_violations(last.answer, eligible=result.get("eligible"), allowed_amounts=allowed)
        failures += [f"HARD GATE: {p}" for p in gate_problems]

    cr = CaseResult(
        id=case["id"],
        passed=not failures,
        failures=failures,
        route=last.route,
        category=cat,
        expected_category=exp.get("category"),
        latency_ms=sum(r.latency_ms for r in responses),
        retrieved_docs=docs,
    )
    extra = {
        "first": first,
        "last": last,
        "gate_problems": gate_problems,
        "tool_called": tool_called,
        "fallback": bool((last.retrieval or {}).get("fallback_used")),
        "widened": len((last.retrieval or {}).get("filter") or []) > 1,
    }
    return cr, extra


def retrieval_ablation(copilot: Copilot, cases: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """hit@k / MRR for knowledge cases under four filter strategies."""
    k = copilot.settings.top_k
    strategies = {
        "no_filter": lambda q, exp: None,
        "hard_filter_predicted": lambda q, exp: (
            [copilot.classifier.predict(q).label]
            if copilot.classifier.predict(q).label != copilot.taxonomy.non_relevant_id
            else None
        ),
        "system_policy (soft filter + fallback)": None,
        "oracle_filter": lambda q, exp: exp["category"][:1],
    }
    out: dict[str, dict[str, float]] = {}
    kb_cases = [c for c in cases if c["expect"].get("docs") and len(c["turns"]) == 1]
    for name, fn in strategies.items():
        hits, rr = 0, 0.0
        for c in kb_cases:
            q, exp = c["turns"][0], c["expect"]
            if fn is None:
                cls = copilot.classifier.predict(q)
                res = copilot.retriever.search_with_policy(
                    q, copilot.filter_for(cls), k=k, min_evidence=copilot.settings.min_evidence
                )
            else:
                res = copilot.retriever.search(q, fn(q, exp), k)
            docs = [h.chunk.doc_id for h in res.hits]
            ranks = [i for i, d in enumerate(docs, 1) if d in exp["docs"]]
            hits += bool(ranks)
            rr += 1.0 / ranks[0] if ranks else 0.0
        out[name] = {"hit_at_k": hits / len(kb_cases), "mrr": rr / len(kb_cases), "n": len(kb_cases)}
    return out


def category_audit(copilot: Copilot, seeds_path: Path) -> list[dict[str, Any]]:
    """Classify every chunk with a SEED-ONLY classifier (no KB text in training, so no
    circularity) and aggregate per document. High entropy => the document spans categories."""
    import math

    tax = copilot.taxonomy
    clf = QueryClassifier(tax).fit(build_training_set(tax, load_seed_examples(seeds_path), None))
    per_doc: dict[str, Counter] = defaultdict(Counter)
    for ch in copilot.chunks.chunks:
        scores = clf.predict_local(ch.embed_text).scores
        for lab, p in scores.items():
            if lab != tax.non_relevant_id:
                per_doc[ch.doc_id][lab] += p
    rows = []
    for doc_id, cnt in sorted(per_doc.items()):
        total = sum(cnt.values())
        dist = {k: v / total for k, v in cnt.most_common()}
        entropy = -sum(p * math.log2(p) for p in dist.values() if p > 0)
        entry = copilot.manifest.documents[doc_id]
        top2 = list(dist.items())[:2]
        rows.append(
            {
                "doc_id": doc_id,
                "manifest": entry.categories,
                "predicted_top": top2[0][0],
                "top2": [(k, round(v, 2)) for k, v in top2],
                "entropy": round(entropy, 2),
                "ambiguous_flag": entry.ambiguous,
                "agrees": top2[0][0] in entry.categories,
            }
        )
    return rows


def evaluate(copilot: Copilot, golden_path: Path, seeds_path: Path) -> dict[str, Any]:
    cases = load_golden(golden_path)
    t0 = time.perf_counter()
    results, extras = [], []
    for c in cases:
        r, e = run_case(copilot, c)
        results.append(r)
        extras.append(e)
    elapsed = time.perf_counter() - t0

    labelled = [(r, c) for r, c in zip(results, cases) if c["expect"].get("category")]
    cls_acc = sum(r.category in c["expect"]["category"] for r, c in labelled) / len(labelled)
    strict_acc = sum(r.category == c["expect"]["category"][0] for r, c in labelled) / len(labelled)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r, c in labelled:
        confusion[c["expect"]["category"][0]][r.category] += 1

    intent_cases = [(e, c) for e, c in zip(extras, cases) if c["expect"].get("intent")]
    intent_acc = sum(e["last"].intent.get("kind", "knowledge") == c["expect"]["intent"] for e, c in intent_cases) / len(
        intent_cases
    )

    tool_cases = [(e, c) for e, c in zip(extras, cases) if "tool_called" in c["expect"]]
    tp = sum(e["tool_called"] and c["expect"]["tool_called"] for e, c in tool_cases)
    fp = sum(e["tool_called"] and not c["expect"]["tool_called"] for e, c in tool_cases)
    fn = sum((not e["tool_called"]) and c["expect"]["tool_called"] for e, c in tool_cases)

    elig_cases = [(r, c) for r, c in zip(results, cases) if "eligible" in c["expect"] or "asks_for" in c["expect"]]
    elig_acc = sum(r.passed for r, _ in elig_cases) / len(elig_cases)

    kb_cases = [(r, c) for r, c in zip(results, cases) if c["expect"].get("docs")]
    hit_k = sum(bool(set(c["expect"]["docs"]) & set(r.retrieved_docs)) for r, c in kb_cases) / len(kb_cases)
    fact_cases = [
        (e, c)
        for e, c in zip(extras, cases)
        if c["expect"].get("contains_any") and c["expect"].get("intent") == "knowledge"
    ]
    fact_recall = sum(_contains_any(e["last"].answer, c["expect"]["contains_any"]) for e, c in fact_cases) / len(
        fact_cases
    )

    nr = [(r, c) for r, c in zip(results, cases) if c["expect"].get("route") == "abstain"]
    abstain_recall = sum(r.route == "abstain" for r, _ in nr) / len(nr)
    in_domain = [r for r, c in zip(results, cases) if c["expect"].get("route") != "abstain"]
    false_abstain = sum(r.route == "abstain" for r in in_domain)

    gate_viol = sum(bool(e["gate_problems"]) for e in extras)
    lat = sorted(r.latency_ms for r in results)
    knowledge_turns = [e for e in extras if e["last"].route == "knowledge"]

    metrics = {
        "n_cases": len(cases),
        "pass_rate": sum(r.passed for r in results) / len(results),
        "classification_accuracy": cls_acc,
        "classification_accuracy_strict_primary": strict_acc,
        "intent_accuracy": intent_acc,
        "tool_routing": {
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "eligibility_decision_accuracy": elig_acc,
        "retrieval_hit_at_k": hit_k,
        "answer_fact_recall": fact_recall,
        "abstain_recall_non_relevant": abstain_recall,
        "false_abstentions_in_domain": false_abstain,
        "hard_gate_violations": gate_viol,
        "filter_fallback_rate": sum(e["fallback"] for e in knowledge_turns) / max(1, len(knowledge_turns)),
        "filter_widened_rate": sum(e["widened"] for e in knowledge_turns) / max(1, len(knowledge_turns)),
        "latency_ms": {
            "p50": statistics.median(lat),
            "p95": lat[int(0.95 * (len(lat) - 1))],
            "total_s": round(elapsed, 2),
        },
    }
    gates = {k: (metrics[k] <= v if k == "hard_gate_violations" else metrics[k] >= v) for k, v in THRESHOLDS.items()}
    return {
        "config": {
            "embedder": copilot.retriever.embedder.name,
            "llm": copilot.llm.name if copilot.llm else "none (extractive)",
            "top_k": copilot.settings.top_k,
        },
        "metrics": metrics,
        "quality_gates": gates,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "ablation": retrieval_ablation(copilot, cases),
        "category_audit": category_audit(copilot, seeds_path),
        "failures": [{"id": r.id, "failures": r.failures} for r in results if not r.passed],
    }


def render_markdown(report: dict[str, Any]) -> str:
    m = report["metrics"]
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        "# Evaluation report",
        "",
        f"Embedder: `{report['config']['embedder']}` · LLM: `{report['config']['llm']}` · top_k={report['config']['top_k']} · "
        f"{m['n_cases']} golden cases (data/eval/golden_set.jsonl)",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value | CI gate |",
        "|---|---|---|",
    ]
    rows = [
        ("Case pass rate (all assertions)", pct(m["pass_rate"]), ""),
        (
            "Classification accuracy (label in accepted set)",
            pct(m["classification_accuracy"]),
            f"≥ {pct(THRESHOLDS['classification_accuracy'])}",
        ),
        ("Classification accuracy (strict primary label)", pct(m["classification_accuracy_strict_primary"]), ""),
        ("Intent / routing accuracy", pct(m["intent_accuracy"]), f"≥ {pct(THRESHOLDS['intent_accuracy'])}"),
        (
            "Tool routing precision / recall",
            f"{pct(m['tool_routing']['precision'])} / {pct(m['tool_routing']['recall'])}",
            "",
        ),
        ("Eligibility decision accuracy", pct(m["eligibility_decision_accuracy"]), "= 100%"),
        ("**Hard-gate violations**", f"**{m['hard_gate_violations']}**", "= 0"),
        (
            "Retrieval hit@k (expected doc retrieved)",
            pct(m["retrieval_hit_at_k"]),
            f"≥ {pct(THRESHOLDS['retrieval_hit_at_k'])}",
        ),
        ("Answer fact recall (expected fact in answer)", pct(m["answer_fact_recall"]), ""),
        ("Off-topic abstention recall", pct(m["abstain_recall_non_relevant"]), ""),
        ("False abstentions on in-domain questions", str(m["false_abstentions_in_domain"]), ""),
        ("Unfiltered-fallback rate (knowledge turns)", pct(m["filter_fallback_rate"]), ""),
        ("Filter widened to top-2 categories", pct(m["filter_widened_rate"]), ""),
        ("Latency p50 / p95 per case", f"{m['latency_ms']['p50']:.0f} ms / {m['latency_ms']['p95']:.0f} ms", ""),
    ]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    lines += [
        "",
        "Quality gates: " + ", ".join(f"{k} {'✅' if v else '❌'}" for k, v in report["quality_gates"].items()),
        "",
    ]

    lines += ["## Retrieval filter ablation (knowledge questions)", "", "| Strategy | hit@k | MRR |", "|---|---|---|"]
    for name, v in report["ablation"].items():
        lines.append(f"| {name} | {pct(v['hit_at_k'])} | {v['mrr']:.3f} |")

    cats = sorted({*report["confusion"], *(p for row in report["confusion"].values() for p in row)})
    lines += [
        "",
        "## Classification confusion (rows = expected primary, cols = predicted)",
        "",
        "| expected \\ predicted | " + " | ".join(cats) + " |",
        "|---" * (len(cats) + 1) + "|",
    ]
    for exp in cats:
        if exp in report["confusion"]:
            lines.append(f"| {exp} | " + " | ".join(str(report["confusion"][exp].get(p, "")) for p in cats) + " |")

    lines += [
        "",
        "## Document category audit (seed-only classifier over each document's chunks)",
        "",
        "| Document | Manifest categories | Classifier top-2 (share) | Entropy (bits) |",
        "|---|---|---|---|",
    ]
    for r in report["category_audit"]:
        flag = " ⚠️ multi-label" if r["ambiguous_flag"] else ("" if r["agrees"] else " ❗ disagrees")
        lines.append(
            f"| {r['doc_id']}{flag} | {', '.join(r['manifest'])} | "
            + ", ".join(f"{k} {v:.2f}" for k, v in r["top2"])
            + f" | {r['entropy']} |"
        )

    lines += ["", "## Failed cases", ""]
    lines += [f"- `{f['id']}`: " + "; ".join(f["failures"]) for f in report["failures"]] or ["None."]
    return "\n".join(lines) + "\n"
