
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from src.mp216.data_cleaning import clean_data


MODEL_URI = "models:/MP216_Helpline_Model@champion"

FEATURES = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age",
    "language",
    "team",
]


def main():
    holdout = pd.read_csv("data/raw/holdout_features.csv")

    cleaned = clean_data(
        holdout,
        fix_alpha_units=False,
    )

    model = mlflow.sklearn.load_model(MODEL_URI)
    predictions = model.predict(cleaned[FEATURES])

    assert len(predictions) == len(holdout)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    assert holdout["record_id"].notna().all()

    output = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions,
    })

    Path("predictions").mkdir(
        parents=True, exist_ok=True
    )

    path = "predictions/holdout_predictions.csv"
    output.to_csv(path, index=False)

    print("\n===== FINAL HOLDOUT PREDICTIONS =====")
    print("Model:", MODEL_URI)
    print("Records:", len(output))
    print("Columns:", output.columns.tolist())
    print("Missing predictions:", output["prediction"].isna().sum())
    print("\nFirst five rows:")
    print(output.head().to_string(index=False))
    print("\nSaved to:", path)
    print("===== COMPLETED =====")


if __name__ == "__main__":
    main()
