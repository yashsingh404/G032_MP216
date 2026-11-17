
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction():
    sample = {
        "date": "2026-09-01",
        "queue_wait_sec": 300,
        "agent_exp_months": 24,
        "previous_calls": 2,
        "issue_count": 3,
        "hour": 14,
        "caller_age": 35,
        "language": "english",
        "team": "Team Alpha"
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=sample)

    assert response.status_code == 200

    result = response.json()
    assert "predicted_handle_min" in result
    assert result["predicted_handle_min"] >= 0
