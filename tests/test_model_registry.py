
import mlflow.sklearn
from mlflow import MlflowClient


MODEL_NAME = "MP216_Helpline_Model"


def test_champion_alias():
    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )

    assert champion.version == 3


def test_candidate_registration():
    client = MlflowClient()

    candidate = client.get_model_version(
        MODEL_NAME, "2"
    )

    assert candidate.status == "READY"
    assert candidate.tags.get(
        "evaluation_status"
    ) == "development_validated"


def test_candidate_model_loads():
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/2"
    )

    assert hasattr(model, "predict")
