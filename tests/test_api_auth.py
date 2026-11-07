import os
from fastapi.testclient import TestClient

from api.main import app, require_api_key


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


def test_session_id_header_propagates_client_identity(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    client_id = require_api_key(x_api_key="test-secret", x_session_id="user-session-1234")
    assert client_id == "session:user-session-1234"


def test_missing_session_id_defaults_to_service_identity(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-secret")
    client_id = require_api_key(x_api_key="test-secret", x_session_id=None)
    assert client_id == "streamlit-service"
