
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.mp216.data_cleaning import clean_data


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
    print("\n===== ASSIGNMENT V2 CHECK =====")

    original = clean_data(
        pd.read_csv("data/raw/train.csv"),
        fix_alpha_units=True,
    )

    later = clean_data(
        pd.read_csv("data/raw/later.csv"),
        fix_alpha_units=False,
    )

    # Stable chronological ordering
    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values(
        "date", kind="stable"
    ).reset_index(drop=True)

    split = int(len(later) * 0.7)

    later_train = later.iloc[:split]
    later_check = later.iloc[split:]

    training = pd.concat(
        [original, later_train],
        ignore_index=True,
    )

    print("Training records:", len(training))
    print("Check records:", len(later_check))
    print("Train-later latest date:", later_train["date"].max())
    print("Check earliest date:", later_check["date"].min())

    numeric = [
        "queue_wait_sec",
        "agent_exp_months",
        "previous_calls",
        "issue_count",
        "hour",
        "caller_age",
    ]

    categorical = ["language", "team"]

    preprocessor = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ])

    candidate = Pipeline([
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    candidate.fit(
        training[FEATURES],
        training["handle_min"],
    )

    champion = mlflow.sklearn.load_model(
        "models:/MP216_Helpline_Model@champion"
    )

    X_check = later_check[FEATURES]
    y_check = later_check["handle_min"]

    mae_v1 = mean_absolute_error(
        y_check, champion.predict(X_check)
    )
    mae_v2 = mean_absolute_error(
        y_check, candidate.predict(X_check)
    )

    improvement = (mae_v1 - mae_v2) / mae_v1 * 100
    decision = "promote" if improvement >= 12 else "keep"

    print(f"\nv1 check MAE: {mae_v1:.6f}")
    print(f"v2 check MAE: {mae_v2:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Decision: {decision}")

    mlflow.set_experiment("MP216_Assignment_V2")

    with mlflow.start_run():
        mlflow.log_param("later_train_fraction", 0.70)
        mlflow.log_param("n_estimators", 300)
        mlflow.log_metric("mae_v1_on_check", mae_v1)
        mlflow.log_metric("mae_v2_on_check", mae_v2)
        mlflow.log_metric("improvement_pct", improvement)
        mlflow.log_param("decision", decision)

        mlflow.sklearn.log_model(
            candidate,
            name="model",
            serialization_format="cloudpickle",
        )

    print("\n===== ASSIGNMENT CHECK COMPLETED =====")
    print("Champion alias has not been changed.")


if __name__ == "__main__":
    main()
