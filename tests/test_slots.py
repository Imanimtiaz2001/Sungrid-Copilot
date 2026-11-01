import pytest

from sungrid_copilot.agent.slots import SlotValues, extract_slots, is_grounded, validate

CASES = [
    (
        "Am I eligible? ZIP 94103, household income $85,000, 6 kW system, my installer is SunGrid-approved.",
        ("94103", 85000, 6.0, True),
    ),
    ("94103, $80k, 6kW, approved installer", ("94103", 80000, 6.0, True)),
    (
        "I live in 94105 and make 95k a year. Planning a 6500 W system with a certified installer.",
        ("94105", 95000, 6.5, True),
    ),
    ("my income is 85000 and zip code is 94101, 7.5 kw, installer is not approved", ("94101", 85000, 7.5, False)),
    ("We earn $9,000 per month, zip 94102, 5kW, installer approved", ("94102", 108000, 5.0, True)),
    ("My installer's certification had lapsed. ZIP 94102, income $70k, 5kW.", ("94102", 70000, 5.0, False)),
    ("I'm not sure if my installer is approved. 94104, income 60000, 4 kw", ("94104", 60000, 4.0, None)),
    ("income 1.2 million, zip 94103-1234, 10kw, installer: yes", ("94103", 1_200_000, 10.0, True)),
    ("A 13.5 kWh battery with my 8 kW system in 94102, income 90k, approved installer", ("94102", 90000, 8.0, True)),
    (
        "Household makes $119,999, I'm in 90210, 3 kW, installed by a non-approved contractor",
        ("90210", 119999, 3.0, False),
    ),
    ("Do I qualify? I paid $18,000 for my 6kW system", (None, None, 6.0, None)),
    ("Am I eligible for the rebate?", (None, None, None, None)),
    ("Am I eligible? 94103 / 85k income / 6.2kw / approved installer: yes", ("94103", 85000, 6.2, True)),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_extraction(text, expected):
    s = extract_slots(text)
    assert (s.household_zip, s.annual_income_usd, s.system_size_kw, s.installer_approved) == expected


def test_kwh_is_never_read_as_system_size():
    assert extract_slots("I have a 13.5 kWh battery").system_size_kw is None


def test_monthly_income_is_annualised_with_a_visible_note():
    s = extract_slots("we make $5,000 a month")
    assert s.annual_income_usd == 60000 and any("monthly" in n for n in s.notes)


@pytest.mark.parametrize(
    "text,awaiting,field,value",
    [
        ("yes", ["installer_approved"], "installer_approved", True),
        ("nope", ["installer_approved"], "installer_approved", False),
        ("85000", ["annual_income_usd"], "annual_income_usd", 85000),
        ("94103", ["household_zip"], "household_zip", "94103"),
        ("6", ["system_size_kw"], "system_size_kw", 6.0),
    ],
)
def test_short_answers_to_a_pending_question(text, awaiting, field, value):
    assert getattr(extract_slots(text, awaiting=awaiting), field) == value


def test_bare_yes_without_pending_question_sets_nothing():
    assert extract_slots("yes").found() == {}


def test_merge_lets_corrections_win():
    a = SlotValues("94103", 85000, 6.0, True)
    b = a.merge(extract_slots("actually my income is 125k"))
    assert b.annual_income_usd == 125000 and b.household_zip == "94103"


def test_validation_rejects_unusable_values():
    probs = validate(SlotValues("9410", -5, 0, None))
    assert set(probs) == {"household_zip", "annual_income_usd", "system_size_kw"}


@pytest.mark.parametrize(
    "slot,value,text,ok",
    [
        ("annual_income_usd", 85000, "I make 85k", True),
        ("annual_income_usd", 85000, "I make a decent living", False),
        ("household_zip", "94103", "zip 94103", True),
        ("household_zip", "94103", "I live in San Francisco", False),
        ("system_size_kw", 6.5, "a 6500 W system", True),
        ("system_size_kw", 6.5, "a medium system", False),
    ],
)
def test_llm_values_must_be_grounded_in_member_text(slot, value, text, ok):
    assert is_grounded(slot, value, text) is ok
