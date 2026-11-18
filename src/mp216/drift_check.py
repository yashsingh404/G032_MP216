
import pandas as pd

from src.mp216.data_cleaning import clean_data


def check_drift():
    train = clean_data(
        pd.read_csv("data/raw/train.csv"),
        fix_alpha_units=True
    )

    later = clean_data(
        pd.read_csv("data/raw/later.csv"),
        fix_alpha_units=False
    )
    print("\n===== QUEUE WAIT INVESTIGATION =====")

    for name, df in [("Training", train), ("Later", later)]:
        print(f"\n{name} dataset:")

        print(
            df["queue_wait_sec"].describe().round(2)
        )

        print("\nQueue waiting time by team:")
        print(
            df.groupby("team")["queue_wait_sec"]
            .agg(["count", "mean", "median", "max"])
            .round(2)
        )

        print("\nRecords exceeding 3600 seconds:")
        print((df["queue_wait_sec"] > 3600).sum())

    print("\n===== INVESTIGATION COMPLETED =====")


if __name__ == "__main__":
    check_drift()
