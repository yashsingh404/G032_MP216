
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.mp216.data_cleaning import clean_data


CHAMPION_URI = "models:/MP216_Helpline_Model@champion"

CANDIDATE_RUN_ID = "31412962d4d646b9b3b002a49cdc8f50"
CANDIDATE_URI = f"runs:/{CANDIDATE_RUN_ID}/model"

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
    print("\n===== CANDIDATE MODEL EVALUATION =====")

    df = pd.read_csv("data/raw/later.csv")

    df = clean_data(
        df,
        fix_alpha_units=False
    )

    df["date"] = pd.to_datetime(df["date"])
    august = df[df["date"].dt.month == 8].copy()

    champion = mlflow.sklearn.load_model(CHAMPION_URI)
    candidate = mlflow.sklearn.load_model(CANDIDATE_URI)

    X = august[FEATURES]
    y = august["handle_min"]

    august["champion_prediction"] = champion.predict(X)
    august["candidate_prediction"] = candidate.predict(X)

    august["champion_error"] = (
        y - august["champion_prediction"]
    ).abs()

    august["candidate_error"] = (
        y - august["candidate_prediction"]
    ).abs()

    print("\n===== OVERALL AUGUST MAE =====")
    print(
        "Champion:",
        round(
            mean_absolute_error(
                y, august["champion_prediction"]
            ), 4
        )
    )
    print(
        "Candidate:",
        round(
            mean_absolute_error(
                y, august["candidate_prediction"]
            ), 4
        )
    )

    print("\n===== MAE BY TEAM =====")
    print(
        august.groupby("team")[
            ["champion_error", "candidate_error"]
        ].mean().round(3)
    )

    august["calls_group"] = pd.cut(
        august["previous_calls"],
        bins=[-1, 5, 10, float("inf")],
        labels=["0-5", "6-10", "11+"]
    )

    print("\n===== MAE BY PREVIOUS CALLS =====")
    print(
        august.groupby(
            "calls_group",
            observed=True
        )[["champion_error", "candidate_error"]]
        .mean().round(3)
    )

    print("\n===== MAE BY TEAM AND CALL GROUP =====")
    print(
        august.groupby(
            ["team", "calls_group"],
            observed=True
        )[["champion_error", "candidate_error"]]
        .agg(["count", "mean"])
        .round(3)
    )

    print("\n===== PREDICTION BIAS =====")
    for name in ["champion", "candidate"]:
        bias = (
            y - august[f"{name}_prediction"]
        ).mean()

        print(
            f"{name.capitalize()} mean signed error:",
            round(bias, 3)
        )

    print("\n===== EVALUATION COMPLETED =====")


if __name__ == "__main__":
    main()
