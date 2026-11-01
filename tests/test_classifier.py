import json
from pathlib import Path

import pytest

from conftest import FakeLLM
from sungrid_copilot.classifier import QueryClassifier, build_training_set, load_seed_examples

ROOT = Path(__file__).resolve().parents[1]


def _golden():
    for line in (ROOT / "data" / "eval" / "golden_set.jsonl").read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def test_no_leakage_between_seed_queries_and_golden_set():
    seeds = {t.lower().strip() for t, _ in load_seed_examples(ROOT / "data" / "classifier" / "seed_queries.jsonl")}
    golden = {turn.lower().strip() for case in _golden() for turn in case["turns"]}
    assert not seeds & golden


def test_golden_classification_accuracy(components):
    clf = components.copilot.classifier
    cases = [c for c in _golden() if c["expect"].get("category")]
    correct = sum(clf.predict(c["turns"][0]).label in c["expect"]["category"] for c in cases)
    assert correct / len(cases) >= 0.9


@pytest.mark.parametrize(
    "query,label",
    [
        ("How long until the utility approves my interconnection?", "technical_installation"),
        ("When is the autopay charge taken?", "billing_account"),
        ("How many votes does each household account get?", "program_policies"),
        ("What is the cap on the rooftop rebate?", "incentive_rebate"),
        ("How much CO2 did members offset?", "company_updates"),
        ("Write me a haiku about autumn", "non_relevant"),
        ("hello", "non_relevant"),
    ],
)
def test_single_primary_label(components, query, label):
    assert components.copilot.classifier.predict(query).label == label


def test_probabilities_are_a_distribution(components):
    c = components.copilot.classifier.predict("autopay failed")
    assert abs(sum(c.scores.values()) - 1.0) < 1e-6
    assert set(c.scores) == set(components.copilot.taxonomy.all_labels)


def test_llm_adjudicates_only_low_margin_queries(components):
    tax = components.copilot.taxonomy
    seeds = load_seed_examples(ROOT / "data" / "classifier" / "seed_queries.jsonl")
    llm = FakeLLM(complete=lambda s, u: '{"category": "billing_account"}')
    clf = QueryClassifier(tax, llm=llm, adjudicate_margin=1.01).fit(
        build_training_set(tax, seeds, components.copilot.chunks.chunks)
    )
    res = clf.predict("my rebate was reversed and it shows on my statement")
    assert res.method == "local+llm" and res.label == "billing_account"

    clf.adjudicate_margin = 0.0  # never unsure -> never calls the LLM
    llm.calls.clear()
    clf.predict("What does fault code E07 mean?")
    assert llm.calls == []


def test_invalid_llm_label_is_ignored(components):
    tax = components.copilot.taxonomy
    seeds = load_seed_examples(ROOT / "data" / "classifier" / "seed_queries.jsonl")
    clf = QueryClassifier(tax, llm=FakeLLM(complete=lambda s, u: '{"category": "made_up"}'), adjudicate_margin=1.01)
    clf.fit(build_training_set(tax, seeds, None))
    assert clf.predict("What does fault code E07 mean?").label in tax.all_labels


def test_llm_outage_keeps_local_label(components):
    tax = components.copilot.taxonomy
    seeds = load_seed_examples(ROOT / "data" / "classifier" / "seed_queries.jsonl")
    clf = QueryClassifier(tax, llm=FakeLLM(fail=True), adjudicate_margin=1.01).fit(build_training_set(tax, seeds, None))
    res = clf.predict("When is my bill due?")
    assert res.label == "billing_account" and res.method == "local"
