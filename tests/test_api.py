from fastapi.testclient import TestClient

import sungrid_copilot.api as api


def test_ask_endpoint(settings, monkeypatch):
    from sungrid_copilot.embeddings import HashingEmbedder
    from sungrid_copilot.factory import build

    # a private instance: the app's shutdown closes its vector store
    monkeypatch.setattr(
        api, "build", lambda *a, **k: build(settings, in_memory=True, embedder=HashingEmbedder(), use_llm=False)
    )
    with TestClient(api.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/ask", json={"query": "Am I eligible for the rebate?", "session_id": "api-1"}).json()
        assert r["route"] == "eligibility_clarify" and "trace" not in r
        r = client.post(
            "/ask", json={"query": "94103, 85k, 6 kW, approved installer", "session_id": "api-1", "include_trace": True}
        ).json()
        assert r["tool_call"]["result"]["eligible"] is True and r["trace"]
        assert client.post("/ask", json={"query": ""}).status_code == 422
