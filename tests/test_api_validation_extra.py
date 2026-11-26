
from unittest.mock import patch

from fastapi.testclient import TestClient
from app.main import app


class FakeModel:
    def predict(self, features):
        return [10.5]


def valid_payload():
    return {
        "queue_wait_sec": 300,
        "agent_exp_months": 24,
        "previous_calls": 20,
        "issue_count": 3,
        "hour": 14,
        "caller_age": 35,
        "language": "english",
        "team": "Team Alpha",
    }


def send(payload):
    # Mock model loaders to avoid requiring real model files.
    with patch(
        "app.main.mlflow.sklearn.load_model",
        return_value=FakeModel()
    ):
        with patch(
            "app.main.joblib.load",
            return_value=FakeModel()
        ):
            with TestClient(app) as client:
                return client.post("/predict", json=payload)


def test_previous_calls_exactly_20_allowed():
    response = send(valid_payload())

    assert response.status_code == 200
    assert response.json()["predicted_handle_min"] == 10.5


def test_previous_calls_21_rejected():
    data = valid_payload()
    data["previous_calls"] = 21

    assert send(data).status_code == 422


def test_negative_agent_experience_rejected():
    data = valid_payload()
    data["agent_exp_months"] = -0.5

    assert send(data).status_code == 422


def test_unexpected_date_field_rejected():
    data = valid_payload()
    data["date"] = "2026-09-01"

    assert send(data).status_code == 422
