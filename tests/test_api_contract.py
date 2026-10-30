from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
TEST_KEY = "test-api-key"


def test_api_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert {
        "/health",
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


def test_empty_entity_name_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", TEST_KEY)
    response = client.post(
        "/entities/resolve",
        json={"mention": ""},
        headers={"X-API-Key": TEST_KEY},
    )
    assert response.status_code == 422
