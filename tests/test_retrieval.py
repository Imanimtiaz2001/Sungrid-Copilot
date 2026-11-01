import pytest


@pytest.fixture()
def retriever(components):
    return components.copilot.retriever


@pytest.mark.parametrize(
    "cats", [["billing_account"], ["technical_installation"], ["incentive_rebate", "billing_account"]]
)
def test_category_filter_is_respected_by_both_retrievers(retriever, cats):
    res = retriever.search("rebate bill installer payment inverter", cats, k=8)
    assert res.hits
    for h in res.hits:
        assert set(h.chunk.categories) & set(cats)


@pytest.mark.parametrize("cat", ["incentive_rebate", "billing_account"])
def test_ambiguous_doc_is_reachable_from_both_of_its_categories(retriever, cat):
    res = retriever.search("rebate reversed billing adjustment spread over billing cycles", [cat], k=3)
    assert res.hits[0].chunk.doc_id == "06_ambiguous_rebate_billing_adjustments"


def test_ambiguous_doc_is_not_reachable_from_unrelated_category(retriever):
    res = retriever.search("rebate reversed billing adjustment", ["technical_installation"], k=4)
    assert all(h.chunk.doc_id != "06_ambiguous_rebate_billing_adjustments" for h in res.hits)


def test_wrong_filter_falls_back_to_unfiltered_search(retriever):
    q = "What does inverter fault code E07 mean?"
    res = retriever.search_with_policy(q, ["company_updates"], k=4, min_evidence=0.2)
    assert res.fallback_used
    assert res.hits[0].chunk.section == "Common Fault Codes"
    assert any("category filter" in n for n in res.notes)


def test_correct_filter_does_not_fall_back(retriever):
    res = retriever.search_with_policy(
        "What does inverter fault code E07 mean?", ["technical_installation"], k=4, min_evidence=0.2
    )
    assert not res.fallback_used
    assert res.hits[0].chunk.section == "Common Fault Codes"


def test_evidence_is_zero_for_off_topic_queries(retriever):
    assert retriever.search("banana bread recipe", None, k=4).evidence == 0.0


def test_section_lookup_by_reference(components):
    ch = components.copilot.chunks.section("02_incentive_rebate_programs#Appeals Process")
    assert ch is not None and "30 days" in ch.text
