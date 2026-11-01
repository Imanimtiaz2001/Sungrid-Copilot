"""
Stub tool(s) for the SunGrid Cooperative Copilot assessment.

This mimics a real-world situation you will hit often on the job: a downstream
service (e.g. a rebate-eligibility microservice, a pricing API, a registry
lookup) that isn't available yet, but whose interface is defined. You are
expected to treat this as a clearly-marked integration point, not something
to reimplement or hollow out.

Do not change the function signature or return shape. You may change the
internal mock logic if you want more realistic test cases, as long as the
interface and eligibility logic described in
docs/02_incentive_rebate_programs.md is respected.
"""

from typing import TypedDict


class EligibilityResult(TypedDict):
    eligible: bool
    reason: str
    estimated_rebate_usd: float


# Placeholder — in production this would call SunGrid's Incentive Program
# service. Treat this as a stand-in for a real API call.
APPROVED_ZIP_CODES = {"94101", "94102", "94103", "94104", "94105"}
INCOME_THRESHOLD_USD = 120_000


def check_rebate_eligibility(
    household_zip: str,
    annual_income_usd: float,
    system_size_kw: float,
    installer_approved: bool,
) -> EligibilityResult:
    """
    Mock eligibility check for the Rooftop Rebate Program.

    Mirrors the hard-eligibility-filter pattern described in
    docs/02_incentive_rebate_programs.md: a household that fails ANY hard
    check gets eligible=False, regardless of other factors. Only households
    that pass every hard check get a calculated rebate amount.
    """
    if not installer_approved:
        return {
            "eligible": False,
            "reason": "Installer is not on the SunGrid approved installer list.",
            "estimated_rebate_usd": 0.0,
        }

    if household_zip not in APPROVED_ZIP_CODES:
        return {
            "eligible": False,
            "reason": "Household ZIP code is outside the SunGrid service region.",
            "estimated_rebate_usd": 0.0,
        }

    if annual_income_usd >= INCOME_THRESHOLD_USD:
        return {
            "eligible": False,
            "reason": "Household income exceeds the regional threshold.",
            "estimated_rebate_usd": 0.0,
        }

    if not (3.0 <= system_size_kw <= 10.0):
        return {
            "eligible": False,
            "reason": "System size is outside the eligible 3kW-10kW range.",
            "estimated_rebate_usd": 0.0,
        }

    rebate = min(system_size_kw * 1000 * 0.40, 4000.0)
    return {
        "eligible": True,
        "reason": "All eligibility checks passed.",
        "estimated_rebate_usd": round(rebate, 2),
    }
