import os

from fastapi.testclient import TestClient

from api.main import app


def test_health_is_public():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_requires_key(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    client = TestClient(app)
    response = client.get("/graph/summary")
    assert response.status_code == 401


def test_api_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    client = TestClient(app)
    response = client.get(
        "/graph/summary",
        headers={"X-API-Key": "wrong-secret"},
    )
    assert response.status_code == 401


def test_api_accepts_configured_key_for_protected_route(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    # Exercise authentication without touching the real graph or storage.
    from knowledge_graph import graph_store
    class FakeGraph:
        def summary(self):
            return {"entities": 0, "videos": 0, "relationships": 0}
    monkeypatch.setattr(graph_store, "get_knowledge_graph", lambda: FakeGraph())
    client = TestClient(app)
    response = client.get(
        "/graph/summary",
        headers={"X-API-Key": "test-secret"},
    )
    assert response.status_code == 200
    assert response.json()["entities"] == 0
