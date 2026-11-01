import time

from sungrid_copilot.agent.tools import EligibilityTool

ARGS = dict(household_zip="94103", annual_income_usd=85000, system_size_kw=6, installer_approved=True)


def test_happy_path_passes_through_service_result():
    call = EligibilityTool().invoke(**ARGS)
    assert call.status == "ok"
    assert call.result == {"eligible": True, "reason": "All eligibility checks passed.", "estimated_rebate_usd": 2400.0}
    assert call.arguments["household_zip"] == "94103" and isinstance(call.arguments["annual_income_usd"], float)


def test_service_exception_fails_closed():
    def boom(**_):
        raise ConnectionError("service down")

    call = EligibilityTool(fn=boom).invoke(**ARGS)
    assert call.status == "error" and call.result is None and "ConnectionError" in call.error


def test_service_timeout_fails_closed():
    def slow(**_):
        time.sleep(0.5)
        return {"eligible": True, "reason": "x", "estimated_rebate_usd": 1.0}

    call = EligibilityTool(fn=slow, timeout_s=0.05).invoke(**ARGS)
    assert call.status == "error" and "timeout" in call.error


def test_contract_violation_is_rejected():
    call = EligibilityTool(fn=lambda **_: {"eligible": "yes"}).invoke(**ARGS)
    assert call.status == "invalid_response" and call.result is None
