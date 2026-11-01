import pytest

from conftest import offline_settings
from fake_llm_server import FakeLLMServer
from sungrid_copilot.llm.providers import AnthropicClient, OpenAICompatibleClient, build_llm


@pytest.fixture(scope="module")
def server():
    s = FakeLLMServer()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def no_proxy_for_localhost(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)


def test_openai_compatible_complete_and_tool_call(server):
    c = OpenAICompatibleClient(
        api_key="k", model="m", base_url=f"http://127.0.0.1:{server.port}/v1", timeout=5, max_retries=0, label="openai"
    )
    assert c.complete(system="s", user="u") == "Bills are due within 21 days [1]."
    args = c.call_tool(
        system="s", user="u", tool_name="route", tool_description="d", parameters={"type": "object", "properties": {}}
    )
    assert args["household_zip"] == "94103"
    req = server.requests[-1]["body"]
    assert req["tool_choice"] == {"type": "function", "function": {"name": "route"}} and req["temperature"] == 0.0


def test_openai_adapter_adapts_to_models_rejecting_max_tokens(server):
    server.reject_legacy = True
    try:
        c = OpenAICompatibleClient(
            api_key="k",
            model="reasoning-model",
            base_url=f"http://127.0.0.1:{server.port}/v1",
            timeout=5,
            max_retries=0,
            label="openai",
        )
        assert c.complete(system="s", user="u", max_tokens=50).startswith("Bills")
        body = server.requests[-1]["body"]
        assert "max_completion_tokens" in body and "max_tokens" not in body and "temperature" not in body
    finally:
        server.reject_legacy = False


def test_anthropic_complete_and_tool_call(server):
    c = AnthropicClient(
        api_key="k", model="claude-test", base_url=f"http://127.0.0.1:{server.port}", timeout=5, max_retries=0
    )
    assert c.complete(system="s", user="u") == "Bills are due within 21 days [1]."
    args = c.call_tool(
        system="s", user="u", tool_name="route", tool_description="d", parameters={"type": "object", "properties": {}}
    )
    assert args["program"] == "rooftop"
    req = server.requests[-1]["body"]
    assert req["tool_choice"] == {"type": "tool", "name": "route"} and req["system"] == "s"


def test_provider_errors_become_llmerror():
    from sungrid_copilot.llm.base import LLMError

    c = OpenAICompatibleClient(
        api_key="k", model="m", base_url="http://127.0.0.1:9/v1", timeout=1, max_retries=0, label="openai"
    )
    with pytest.raises(LLMError):
        c.complete(system="s", user="u")


def test_auto_provider_resolution():
    assert offline_settings(llm_provider="auto").resolved_llm_provider() in {
        "none",
        "anthropic",
        "openai",
        "gemini",
        "groq",
    }
    s = offline_settings(
        llm_provider="auto",
        anthropic_api_key="",
        openai_api_key="sk-x",
        gemini_api_key="",
        google_api_key="",
        groq_api_key="",
    )
    assert s.resolved_llm_provider() == "openai" and s.resolved_llm_model() == "gpt-4o-mini"


def test_missing_key_disables_llm_instead_of_crashing():
    assert build_llm(offline_settings(llm_provider="openai", openai_api_key="")) is None


def test_end_to_end_agent_through_real_sdk(server, components, monkeypatch):
    """Full agent with an OpenAI-compatible provider: answer generated, cited and verified."""
    llm = OpenAICompatibleClient(
        api_key="k", model="m", base_url=f"http://127.0.0.1:{server.port}/v1", timeout=5, max_retries=0, label="openai"
    )
    copilot = components.copilot
    monkeypatch.setattr(copilot, "llm", llm)
    server.text_reply = "Bills are issued on the 1st and due within 21 days [1]."
    r = copilot.ask("When is my bill due?")
    assert r.answer.startswith("Bills are issued on the 1st and due within 21 days [1].")
    assert any(s.get("mode") == "llm" for s in r.trace)
