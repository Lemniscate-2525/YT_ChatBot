from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


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


def test_empty_entity_name_is_rejected():
    response = client.post("/entities/resolve", json={"mention": ""})
    assert response.status_code == 422
