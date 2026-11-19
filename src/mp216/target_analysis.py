
import pandas as pd

from src.mp216.data_cleaning import clean_data


def analyze_targets():
    train = clean_data(
        pd.read_csv("data/raw/train.csv"),
        fix_alpha_units=True
    )

    later = clean_data(
        pd.read_csv("data/raw/later.csv"),
        fix_alpha_units=False
    )

    print("\n===== TARGET DISTRIBUTION =====")

    for name, df in [
        ("Training", train),
        ("Later", later)
    ]:
        print(f"\n{name} dataset:")
        print(df["handle_min"].describe().round(3))

        print("\nHandling time by team:")
        print(
            df.groupby("team")["handle_min"]
            .agg(["count", "mean", "median", "std"])
            .round(3)
        )

    print("\n===== TARGET BY PREVIOUS CALLS =====")

    for name, df in [
        ("Training", train),
        ("Later", later)
    ]:
        df = df.copy()

        df["calls_group"] = pd.cut(
            df["previous_calls"],
            bins=[-1, 5, 10, float("inf")],
            labels=["0-5", "6-10", "11+"]
        )

        print(f"\n{name}:")
        print(
            df.groupby(
                "calls_group",
                observed=True
            )["handle_min"]
            .agg(["count", "mean", "median"])
            .round(3)
        )

    print("\n===== TARGET ANALYSIS COMPLETED =====")


if __name__ == "__main__":
    analyze_targets()
