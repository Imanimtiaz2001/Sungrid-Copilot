"""Tool layer: a thin, owned adapter around the externally-owned eligibility service.

`vendor/stub_tools.py` is treated as someone else's service: we never edit it and never
re-implement its rules. This adapter adds only what a client of a real service needs:
  * a JSON schema derived from the service's own signature (handed to the LLM),
  * input validation before the call and a response-contract check after it,
  * a timeout and FAIL-CLOSED error handling: if the service errors, the member gets
    "can't confirm right now", never a docs-based guess,
  * structured, PII-redacted audit logging.
"""

from __future__ import annotations

import concurrent.futures as cf
import inspect
import logging
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..vendor import stub_tools

log = logging.getLogger("sungrid.tools")

_PY_TO_JSON = {str: "string", float: "number", int: "integer", bool: "boolean"}

PARAM_DESCRIPTIONS = {
    "household_zip": "Household 5-digit US ZIP code, exactly as the member stated it.",
    "annual_income_usd": "Annual household (adjusted gross) income in USD, as stated by the member.",
    "system_size_kw": "Rooftop solar system size in kilowatts (convert watts to kW).",
    "installer_approved": "True only if the member says the installer is SunGrid-approved/certified; False if they say it is not or certification lapsed.",
}


def schema_from_signature(fn: Callable, descriptions: dict[str, str]) -> dict[str, Any]:
    hints = typing.get_type_hints(fn)
    props, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        props[name] = {"type": _PY_TO_JSON[hints[name]], "description": descriptions.get(name, "")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    status: str  # "ok" | "error" | "invalid_response"
    result: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


class EligibilityTool:
    name = "check_rebate_eligibility"
    description = (
        "SunGrid Incentive Program service: authoritative Rooftop Rebate eligibility decision and "
        "rebate estimate. Fails any household that does not pass every hard check."
    )

    def __init__(self, fn: Callable[..., dict] | None = None, timeout_s: float = 5.0):
        self._fn = fn or stub_tools.check_rebate_eligibility
        self.timeout_s = timeout_s
        self.parameters = schema_from_signature(stub_tools.check_rebate_eligibility, PARAM_DESCRIPTIONS)
        self._result_keys = set(typing.get_type_hints(stub_tools.EligibilityResult))
        self._pool = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="eligibility")

    def invoke(
        self, *, household_zip: str, annual_income_usd: float, system_size_kw: float, installer_approved: bool
    ) -> ToolCall:
        args = {
            "household_zip": str(household_zip),
            "annual_income_usd": float(annual_income_usd),
            "system_size_kw": float(system_size_kw),
            "installer_approved": bool(installer_approved),
        }
        t0 = time.perf_counter()
        try:
            fut = self._pool.submit(self._fn, **args)
            result = fut.result(timeout=self.timeout_s)
        except cf.TimeoutError:
            return self._done(ToolCall(self.name, args, "error", error=f"timeout after {self.timeout_s}s"), t0)
        except Exception as exc:  # noqa: BLE001 - any service failure must fail closed
            return self._done(ToolCall(self.name, args, "error", error=f"{type(exc).__name__}: {exc}"), t0)

        if (
            not isinstance(result, dict)
            or set(result) != self._result_keys
            or not isinstance(result.get("eligible"), bool)
        ):
            return self._done(
                ToolCall(
                    self.name,
                    args,
                    "invalid_response",
                    result=None,
                    error="response violates EligibilityResult contract",
                ),
                t0,
            )
        return self._done(ToolCall(self.name, args, "ok", result=dict(result)), t0)

    def _done(self, call: ToolCall, t0: float) -> ToolCall:
        call.latency_ms = (time.perf_counter() - t0) * 1000
        redacted = {**call.arguments, "annual_income_usd": "<redacted>"}
        log.info(
            "tool=%s status=%s eligible=%s latency_ms=%.1f args=%s",
            call.name,
            call.status,
            (call.result or {}).get("eligible"),
            call.latency_ms,
            redacted,
        )
        return call
