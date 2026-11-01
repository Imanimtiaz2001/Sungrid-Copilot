"""The eligibility stub is owned by another team: we wire into it, we never edit it."""

import hashlib
import inspect
from pathlib import Path

from sungrid_copilot.agent.tools import EligibilityTool
from sungrid_copilot.vendor import stub_tools

ORIGINAL_SHA256 = "94625eef007754b061bb167b6ba60c3e7f08e3c0529b29b7763eb9954fec2389"


def test_vendored_stub_is_byte_identical_to_the_starter_kit():
    data = Path(stub_tools.__file__).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ORIGINAL_SHA256


def test_signature_unchanged():
    sig = inspect.signature(stub_tools.check_rebate_eligibility)
    assert list(sig.parameters) == ["household_zip", "annual_income_usd", "system_size_kw", "installer_approved"]


def test_tool_schema_is_derived_from_the_service_signature():
    schema = EligibilityTool().parameters
    assert schema["required"] == ["household_zip", "annual_income_usd", "system_size_kw", "installer_approved"]
    assert {k: v["type"] for k, v in schema["properties"].items()} == {
        "household_zip": "string",
        "annual_income_usd": "number",
        "system_size_kw": "number",
        "installer_approved": "boolean",
    }
