from api.main import app


def test_api_routes_are_registered():
    paths = {route.path for route in app.routes}
    assert {"/health", "/ingest", "/ask", "/entities/resolve", "/graph/summary"}.issubset(paths)
