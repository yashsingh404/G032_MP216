
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

NUMERIC = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age",
]

CATEGORICAL = ["language", "team"]

CHAMPION_URI = "models:/MP216_Helpline_Model@champion"


def main():
    print("\n===== STAGE 7 CANDIDATE TRAINING =====")

    original = clean_data(
        pd.read_csv("data/raw/train.csv"),
        fix_alpha_units=True,
    )

    later = clean_data(
        pd.read_csv("data/raw/later.csv"),
        fix_alpha_units=False,
    )

    later["date"] = pd.to_datetime(later["date"])

    july = later[later["date"].dt.month == 7].copy()
    august = later[later["date"].dt.month == 8].copy()

    training = pd.concat(
        [original, july],
        ignore_index=True,
    )

    X_train = training[FEATURES]
    y_train = training["handle_min"]

    X_val = august[FEATURES]
    y_val = august["handle_min"]

    print("Training rows:", len(training))
    print("Validation rows:", len(august))

    preprocessor = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            NUMERIC,
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]),
            CATEGORICAL,
        ),
    ])

    champion = mlflow.sklearn.load_model(CHAMPION_URI)

    champion_predictions = champion.predict(X_val)

    champion_mae = mean_absolute_error(
        y_val, champion_predictions
    )

    print(f"\nChampion August MAE: {champion_mae:.4f}")

    mlflow.set_experiment("MP216_Candidate_Improvement")

    for n_estimators in [100, 200, 300]:
        with mlflow.start_run():
            candidate = Pipeline([
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ])

            candidate.fit(X_train, y_train)

            predictions = candidate.predict(X_val)

            candidate_mae = mean_absolute_error(
                y_val, predictions
            )

            improvement = (
                (champion_mae - candidate_mae)
                / champion_mae
            ) * 100

            mlflow.log_param(
                "n_estimators", n_estimators
            )
            mlflow.log_param(
                "training_rows", len(training)
            )
            mlflow.log_param(
                "validation_rows", len(august)
            )
            mlflow.log_metric(
                "champion_august_mae", champion_mae
            )
            mlflow.log_metric(
                "candidate_august_mae", candidate_mae
            )
            mlflow.log_metric(
                "improvement_pct", improvement
            )

            mlflow.sklearn.log_model(
                candidate,
                name="model",
                serialization_format="cloudpickle",
            )

            print(
                f"\nRandom Forest {n_estimators} trees"
            )
            print(
                f"Candidate MAE: {candidate_mae:.4f}"
            )
            print(
                f"Improvement: {improvement:.2f}%"
            )

    print("\n===== CANDIDATE TRAINING COMPLETED =====")
    print("Champion model remains unchanged.")


if __name__ == "__main__":
    main()
