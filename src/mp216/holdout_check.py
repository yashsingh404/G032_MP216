
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.mp216.data_cleaning import clean_data


RUN_ID = "31412962d4d646b9b3b002a49cdc8f50"
MODEL_URI = f"runs:/{RUN_ID}/model"

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
    print("\n===== SEPTEMBER HOLDOUT CHECK =====")

    df = pd.read_csv("data/raw/holdout_features.csv")

    # Later-period data is already measured in seconds.
    df = clean_data(df, fix_alpha_units=False)

    model = mlflow.sklearn.load_model(MODEL_URI)

    predictions = model.predict(df[FEATURES])

    print("Records:", len(df))
    print("Predictions:", len(predictions))
    print("All predictions finite:", np.isfinite(predictions).all())
    print("All predictions nonnegative:", (predictions >= 0).all())

    print("\nPrediction summary:")
    print(pd.Series(predictions).describe().round(3))

    print("\n===== HOLDOUT CHECK COMPLETED =====")


if __name__ == "__main__":
    main()
