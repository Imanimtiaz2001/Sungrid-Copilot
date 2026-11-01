"""CI quality gates: the golden-set evaluation must meet every threshold (offline mode)."""

from pathlib import Path

from sungrid_copilot.evaluation import THRESHOLDS, evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_quality_gates(components):
    rep = evaluate(
        components.copilot,
        ROOT / "data" / "eval" / "golden_set.jsonl",
        ROOT / "data" / "classifier" / "seed_queries.jsonl",
    )
    m = rep["metrics"]
    assert m["hard_gate_violations"] == 0
    assert m["eligibility_decision_accuracy"] == 1.0
    failed = {k: m[k] for k, ok in rep["quality_gates"].items() if not ok}
    assert not failed, f"quality gates failed: {failed} (thresholds {THRESHOLDS})"


def test_audit_flags_the_ambiguous_document(components):
    from sungrid_copilot.evaluation import category_audit

    rows = {
        r["doc_id"]: r for r in category_audit(components.copilot, ROOT / "data" / "classifier" / "seed_queries.jsonl")
    }
    amb = rows["06_ambiguous_rebate_billing_adjustments"]
    cats = {k for k, _ in amb["top2"]}
    assert cats == {"incentive_rebate", "billing_account"}  # the classifier independently sees both
