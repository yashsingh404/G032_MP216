
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error

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
    print("\n===== MODEL ERROR ANALYSIS =====")

    model = mlflow.sklearn.load_model(MODEL_URI)

    df = pd.read_csv("data/raw/later.csv")
    df = clean_data(df, fix_alpha_units=False)

    df["prediction"] = model.predict(df[FEATURES])
    df["absolute_error"] = (
        df["handle_min"] - df["prediction"]
    ).abs()

    print("\n===== MAE BY TEAM =====")
    print(
        df.groupby("team")["absolute_error"]
        .agg(["count", "mean", "median", "max"])
        .round(3)
    )

    print("\n===== MAE BY PREVIOUS CALLS =====")

    df["calls_group"] = pd.cut(
        df["previous_calls"],
        bins=[-1, 5, 10, float("inf")],
        labels=["0-5", "6-10", "11+"],
    )

    print(
        df.groupby("calls_group", observed=True)["absolute_error"]
        .agg(["count", "mean", "median"])
        .round(3)
    )

    # Signed error: positive means underprediction.
    df["signed_error"] = (
        df["handle_min"] - df["prediction"]
    )

    print("\n===== BIAS BY PREVIOUS CALLS =====")
    print(
        df.groupby("calls_group", observed=True)["signed_error"]
        .agg(["count", "mean", "median"])
        .round(3)
    )

    print("\n===== BIAS BY TEAM =====")
    print(
        df.groupby("team")["signed_error"]
        .agg(["count", "mean", "median"])
        .round(3)
    )

    print("\n===== MAE BY TEAM AND CALL GROUP =====")
    print(
        df.groupby(
            ["team", "calls_group"],
            observed=True
        )["absolute_error"]
        .agg(["count", "mean"])
        .round(3)
    )

    print("\n===== PREDICTION BIAS =====")
    print(
        "Actual mean:",
        round(df["handle_min"].mean(), 3)
    )
    print(
        "Predicted mean:",
        round(df["prediction"].mean(), 3)
    )

    print("\n===== OVERALL MAE =====")
    print(
        round(
            mean_absolute_error(
                df["handle_min"],
                df["prediction"]
            ),
            4,
        )
    )

    print("\n===== ANALYSIS COMPLETED =====")


if __name__ == "__main__":
    main()