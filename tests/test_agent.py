import pytest

from conftest import FakeLLM


def test_knowledge_answer_is_cited_and_category_filtered(copilot):
    r = copilot.ask("How do I appeal a rebate eligibility decision?")
    assert r.route == "knowledge" and r.tool_call is None
    assert r.category["label"] == "incentive_rebate"
    assert r.retrieval["filter"][0] == "incentive_rebate"
    assert "30 days" in r.answer and "02_incentive_rebate_programs § Appeals Process" in r.citations
    assert "Sources:" in r.answer


def test_every_step_is_traced(copilot):
    r = copilot.ask("How do I appeal a rebate eligibility decision?")
    steps = [s["step"] for s in r.trace]
    assert steps[:3] == ["classify", "understand", "retrieve"] and "generate" in steps


def test_eligibility_flow_is_a_distinct_tool_step(copilot):
    r = copilot.ask("Am I eligible for the rooftop rebate? ZIP 94103, income $85,000, 6 kW, approved installer.")
    steps = [s["step"] for s in r.trace]
    assert steps == [
        "classify",
        "understand",
        "validate_arguments",
        "tool_call",
        "hard_gate",
        "policy_lookup",
        "gate_guard",
    ]
    assert "retrieve" not in steps  # the tool is not folded into retrieval
    assert r.tool_call["result"]["estimated_rebate_usd"] == 2400.0 and "$2,400.00" in r.answer


def test_multi_turn_slot_filling_and_correction(copilot):
    sid = "multi"
    r1 = copilot.ask("Am I eligible for the rebate?", sid)
    assert r1.route == "eligibility_clarify" and r1.tool_call is None
    r2 = copilot.ask("94103, 6kW, income 80k", sid)
    assert r2.route == "eligibility_clarify" and "approved installer list" in r2.answer
    r3 = copilot.ask("yes", sid)
    assert r3.tool_call["result"]["eligible"] is True
    r4 = copilot.ask("Actually my income is 125k", sid)
    assert r4.tool_call["result"]["eligible"] is False and "income" in r4.tool_call["result"]["reason"].lower()


def test_reset_clears_session(copilot):
    copilot.ask("Am I eligible for the rebate? zip 94103", "r")
    copilot.ask("reset", "r")
    r = copilot.ask("Am I eligible for the rebate?", "r")
    assert "Already noted" not in r.answer


def test_invalid_value_is_asked_again(copilot):
    r = copilot.ask("Am I eligible for the rooftop rebate? ZIP 94103, income $85,000, 0 kW system, approved installer.")
    assert r.route == "eligibility_clarify" and "positive number" in r.answer


def test_unsupported_program_gets_documents_plus_disclaimer(copilot):
    r = copilot.ask("Do I qualify for the low-income bill credit? I make 30k a year.")
    assert r.route == "knowledge" and r.tool_call is None
    assert "can't determine personal eligibility" in r.answer and "200%" in r.answer


@pytest.mark.parametrize(
    "q", ["What's the capital of Australia?", "Give me a recipe for banana bread", "Tell me a joke about cats"]
)
def test_off_topic_abstains(copilot, q):
    assert copilot.ask(q).route == "abstain"


def test_smalltalk(copilot):
    assert copilot.ask("hello!").route == "smalltalk"


def test_followup_question_is_condensed(copilot):
    copilot.ask("What is the monthly membership fee for the household tier?", "f")
    r = copilot.ask("what about commercial?", "f")
    assert "$150" in r.answer


# ---------------------------------------------------------------- LLM-assisted paths
def test_llm_answer_with_valid_citations_is_used(copilot, monkeypatch):
    llm = FakeLLM(
        complete=lambda s, u: (
            "You can appeal within 30 days of the determination [1]. The Incentive Program committee reviews it [1]."
        )
    )
    monkeypatch.setattr(copilot, "llm", llm)
    r = copilot.ask("How do I appeal a rebate decision?")
    assert r.answer.startswith("You can appeal within 30 days")
    assert any(s["step"] == "generate" and s["mode"] == "llm" for s in r.trace)


def test_llm_answer_with_invented_numbers_is_rejected(copilot, monkeypatch):
    # e.g. the model "helpfully" computes 6 kW x $0.40/W = $2,400 - a rebate estimate outside the tool
    llm = FakeLLM(complete=lambda s, u: "At $0.40/W a 6 kW system gets $2,400 [1].")
    monkeypatch.setattr(copilot, "llm", llm)
    r = copilot.ask("What does the rooftop rebate pay per watt?")
    assert "2,400" not in r.answer
    assert any(s["step"] == "generate_rejected" for s in r.trace)
    assert any(s["step"] == "generate" and s["mode"] == "extractive" for s in r.trace)


def test_invalid_citation_numbers_are_stripped(copilot, monkeypatch):
    monkeypatch.setattr(copilot, "llm", FakeLLM(complete=lambda s, u: "Bills are due within 21 days [1][7]."))
    r = copilot.ask("When are bills due?")
    assert "[7]" not in r.answer and "[1]" in r.answer


def test_llm_outage_degrades_to_extractive(copilot, monkeypatch):
    monkeypatch.setattr(copilot, "llm", FakeLLM(fail=True))
    r = copilot.ask("When are bills issued?")
    assert r.route == "knowledge" and "21 days" in r.answer


def test_llm_cannot_inject_ungrounded_slot_values(copilot, monkeypatch):
    def tool(**_):
        return {
            "intent": "eligibility_check",
            "program": "rooftop",
            "household_zip": "94103",
            "annual_income_usd": 50000,
            "system_size_kw": 6,
            "installer_approved": "yes",
        }

    monkeypatch.setattr(copilot, "llm", FakeLLM(tool=tool))
    r = copilot.ask("Am I eligible for the rebate? I live in 94103.")
    assert r.tool_call is None and r.route == "eligibility_clarify"  # income/size/installer were invented
    step = next(s for s in r.trace if s["step"] == "llm_understand")
    assert set(step["rejected_ungrounded"]) == {"annual_income_usd", "system_size_kw", "installer_approved"}


def test_llm_can_fill_grounded_values_the_rules_missed(copilot, monkeypatch):
    def tool(**_):
        return {
            "intent": "eligibility_check",
            "program": "rooftop",
            "household_zip": "94104",
            "annual_income_usd": 72000,
            "system_size_kw": 5,
            "installer_approved": "yes",
        }

    monkeypatch.setattr(copilot, "llm", FakeLLM(tool=tool))
    r = copilot.ask(
        "Could my household qualify for the rooftop rebate? We're at 94104, bring in 72000 dollars yearly, 5 kilowatt array, approved installer"
    )
    assert r.tool_call is not None and r.tool_call["result"]["eligible"] is True


def test_llm_upgrades_missed_eligibility_intent(copilot, monkeypatch):
    monkeypatch.setattr(copilot, "llm", FakeLLM(tool=lambda **_: {"intent": "eligibility_check", "program": "rooftop"}))
    r = copilot.ask("Is the rebate something my family could get?")
    assert r.route == "eligibility_clarify"
