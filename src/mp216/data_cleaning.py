
import pandas as pd


def clean_data(df, fix_alpha_units=True):
    # Work on a copy to preserve the original data
    df = df.copy()

    # 1. Fix Team Alpha's waiting-time units
    if fix_alpha_units:
        alpha = df["team"] == "Team Alpha"
        df.loc[alpha, "queue_wait_sec"] *= 60

    # 2. Replace invalid ages with missing values
    df["caller_age"] = df["caller_age"].replace(-999, float("nan"))

    # 3. Standardize language categories
    df["language"] = (
        df["language"]
        .str.strip()
        .str.lower()
        .replace({
            "eng.": "english",
            "hin.": "hindi",
            "reg.": "regional"
        })
    )

    # 4. Mark stuck-hour readings as missing
    dates = pd.to_datetime(df["date"])

    stuck = (
        dates.between("2026-02-02", "2026-02-21")
        & (df["hour"] == 10)
    )

    df.loc[stuck, "hour"] = float("nan")

    return df


if __name__ == "__main__":
    train = pd.read_csv("data/raw/train.csv")
    cleaned = clean_data(train)

    print("Original shape:", train.shape)
    print("Cleaned shape:", cleaned.shape)

    print("\nMissing values after cleaning:")
    print(cleaned.isnull().sum())

    print("\nLanguages after cleaning:")
    print(cleaned["language"].value_counts())

    print("\nTeam waiting times after cleaning:")
    print(cleaned.groupby("team")["queue_wait_sec"].median())

    print("\nOriginal data remains unchanged.")
