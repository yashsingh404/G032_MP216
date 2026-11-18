
import pandas as pd

from src.mp216.data_cleaning import clean_data


def calculate_drift_status(train_mean, later_mean, threshold=25):
    if pd.isna(train_mean) or pd.isna(later_mean):
        return "INSUFFICIENT DATA"

    if train_mean == 0:
        return "INSUFFICIENT DATA"

    change_pct = abs(
        (later_mean - train_mean) / train_mean
    ) * 100

    return "DRIFT ALERT" if change_pct > threshold else "OK"

def monitor_data():
    # Load the existing datasets.
    train = pd.read_csv("data/raw/train.csv")
    later = pd.read_csv("data/raw/later.csv")

    # Apply the same cleaning rules to both datasets.
    train = clean_data(train, fix_alpha_units=True)
    later = clean_data(later, fix_alpha_units=False)

    print("\n========== DATA MONITORING ==========")
    print("Training rows:", len(train))
    print("Later rows:", len(later))

    # Compare numerical features.
    features = [
        "queue_wait_sec",
        "agent_exp_months",
        "previous_calls",
        "issue_count",
        "hour",
        "caller_age",
    ]

    print("\n========== NUMERICAL DRIFT ==========")

    for feature in features:
        train_mean = train[feature].mean()
        later_mean = later[feature].mean()

        difference = later_mean - train_mean

        # Calculate relative mean change.
        if pd.notna(train_mean) and pd.notna(later_mean) and train_mean != 0:
            change_pct = abs(difference / train_mean) * 100
        else:
            change_pct = float("nan")

        status = calculate_drift_status(
    train_mean,
    later_mean,
    threshold=25
)

    
        print(f"\nFeature: {feature}")
        print(f"Training mean: {train_mean:.2f}")
        print(f"Later mean: {later_mean:.2f}")
        print(f"Difference: {difference:.2f}")
        print(f"Relative change: {change_pct:.2f}%")
        print(f"Status: {status}")

    # Compare categorical features.
    print("\n========== CATEGORICAL DRIFT ==========")

    for feature in ["language", "team"]:
        print(f"\nFeature: {feature}")

        comparison = pd.concat(
            [
                train[feature].value_counts(normalize=True).rename("train"),
                later[feature].value_counts(normalize=True).rename("later"),
            ],
            axis=1,
        ).fillna(0)

        print((comparison * 100).round(2))

    print("\n========== MONITORING COMPLETED ==========")


if __name__ == "__main__":
    monitor_data()
