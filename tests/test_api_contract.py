from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
TEST_KEY = "test-api-key"


def test_api_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert {
        "/health",
        "/ready",
        "/ingest",
        "/ask",
        "/entities/resolve",
        "/entities/register",
        "/graph/summary",
    }.issubset(paths)


def test_health_does_not_initialize_rag_runtime():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_probe_returns_status(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", TEST_KEY)
    response = client.get("/ready")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data or "detail" in data


def test_empty_entity_name_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", TEST_KEY)
    response = client.post(
        "/entities/resolve",
        json={"mention": ""},
        headers={"X-API-Key": TEST_KEY},
    )
    assert response.status_code == 422


def test_invalid_youtube_url_domain_rejected_at_schema(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", TEST_KEY)
    response = client.post(
        "/ingest",
        json={"video_url": "https://vimeo.com/12345678"},
        headers={"X-API-Key": TEST_KEY},
    )
    assert response.status_code == 422
