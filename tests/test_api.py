

from unittest.mock import patch

from fastapi.testclient import TestClient
from app.main import app


class FakeModel:
    """A simple fake model used only for automated tests."""

    def predict(self, data):
        # Return a predictable value without loading MLflow.
        return [10.86]


def test_health():
    with patch("app.main.mlflow.sklearn.load_model", return_value=FakeModel()):
        with TestClient(app) as client:
            response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction():
    sample = {
        "queue_wait_sec": 300,
        "agent_exp_months": 24,
        "previous_calls": 2,
        "issue_count": 3,
        "hour": 14,
        "caller_age": 35,
        "language": "english",
        "team": "Team Alpha"
    }

    with patch("app.main.mlflow.sklearn.load_model", return_value=FakeModel()):
        with TestClient(app) as client:
            response = client.post("/predict", json=sample)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_handle_min" in result
    assert result["predicted_handle_min"] == 10.86


def test_previous_calls_above_20_rejected():
    payload = {
        "queue_wait_sec": 300,
        "agent_exp_months": 24,
        "previous_calls": 21,
        "issue_count": 3,
        "hour": 14,
        "caller_age": 35,
        "language": "english",
        "team": "Team Alpha",
    }

    with patch("app.main.mlflow.sklearn.load_model", return_value=FakeModel()):
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_negative_agent_experience_rejected():
    payload = {
        "queue_wait_sec": 300,
        "agent_exp_months": -1,
        "previous_calls": 2,
        "issue_count": 3,
        "hour": 14,
        "caller_age": 35,
        "language": "english",
        "team": "Team Alpha",
    }

    with patch("app.main.mlflow.sklearn.load_model", return_value=FakeModel()):
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_date_field_rejected():
    payload = {
        "date": "2026-09-01",
        "queue_wait_sec": 300,
        "agent_exp_months": 24,
        "previous_calls": 2,
        "issue_count": 3,
        "hour": 14,
        "caller_age": 35,
        "language": "english",
        "team": "Team Alpha",
    }

    with patch("app.main.mlflow.sklearn.load_model", return_value=FakeModel()):
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)

    assert response.status_code == 422
