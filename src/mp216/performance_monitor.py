
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


def evaluate_dataset(model, path, name, fix_alpha_units):
    df = pd.read_csv(path)
    df = clean_data(df, fix_alpha_units=fix_alpha_units)

    X = df[FEATURES]
    y = df["handle_min"]

    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)

    print(f"\n{name}:")
    print(f"Records: {len(df)}")
    print(f"MAE: {mae:.4f} minutes")

    return mae


def main():
    print("\n===== MODEL PERFORMANCE MONITORING =====")

    model = mlflow.sklearn.load_model(MODEL_URI)

    train_mae = evaluate_dataset(
        model,
        "data/raw/train.csv",
        "Training dataset",
        fix_alpha_units=True,
    )

    later_mae = evaluate_dataset(
        model,
        "data/raw/later.csv",
        "Later dataset",
        fix_alpha_units=False,
    )

    print("\n===== PERFORMANCE COMPARISON =====")
    print(f"Training MAE: {train_mae:.4f}")
    print(f"Later MAE: {later_mae:.4f}")
    print(f"MAE difference: {later_mae - train_mae:.4f}")

    if later_mae > train_mae * 1.25:
        print("Status: PERFORMANCE ALERT")
    else:
        print("Status: OK")

    print("\n===== MONITORING COMPLETED =====")


if __name__ == "__main__":
    main()
