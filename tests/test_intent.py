import pytest

from sungrid_copilot.agent.intent import detect_intent


@pytest.mark.parametrize(
    "text,kind,program",
    [
        ("Am I eligible for the rooftop rebate?", "eligibility", "rooftop"),
        ("Do I qualify for the rebate?", "eligibility", "rooftop"),
        ("Can you check my eligibility?", "eligibility", "rooftop"),
        ("How much rebate would I get for a 6 kW system?", "eligibility", "rooftop"),
        ("Estimate my rebate please", "eligibility", "rooftop"),
        ("Can I get the battery storage incentive?", "eligibility", "battery"),
        ("What are the eligibility requirements for the rooftop rebate?", "knowledge", "rooftop"),
        ("How much is the rooftop rebate per watt?", "knowledge", "rooftop"),
        ("My rebate eligibility was denied — how do I appeal?", "knowledge", "rooftop"),
        ("How much notice do I get before a rebate reversal hits my bill?", "knowledge", "rooftop"),
        ("Am I eligible to vote at the annual meeting?", "knowledge", "other"),
        ("Do I qualify for the low-income bill credit?", "knowledge", "low_income"),
        ("Is my business eligible for the commercial incentive?", "knowledge", "commercial"),
        ("When is my bill due?", "knowledge", None),
    ],
)
def test_routing(text, kind, program):
    it = detect_intent(text)
    assert (it.kind, it.program) == (kind, program)


def test_unsupported_program_is_flagged():
    it = detect_intent("Do I qualify for the low-income bill credit?")
    assert it.kind == "knowledge" and not it.tool_supported


def test_slot_only_message_continues_a_pending_check():
    assert detect_intent("94103, 85k", n_slots_in_message=2, awaiting_slots=True).kind == "eligibility"


def test_correction_after_a_check_triggers_recheck():
    assert (
        detect_intent("Actually my income is 125k", n_slots_in_message=1, had_eligibility_check=True).kind
        == "eligibility"
    )


def test_slot_values_with_rebate_mention_route_to_tool():
    assert detect_intent("rebate: 94103, 85k, 6kW", n_slots_in_message=3).kind == "eligibility"
