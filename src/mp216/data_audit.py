
import pandas as pd

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("\n========== DATASET INFORMATION ==========")

print("\nTraining Dataset:")
print(train.shape)
print(train.head())

print("\nLater Dataset:")
print(later.shape)
print(later.head())

print("\nHoldout Dataset:")
print(holdout.shape)
print(holdout.head())

print("\n========== MISSING VALUES ==========")
print(train.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(train.duplicated().sum())

print("\n========== INVALID CALLER AGE ==========")
print(train[train["caller_age"] == -999].shape[0])

print("\n========== LANGUAGE VALUES ==========")
print(train["language"].value_counts())

print("\n========== TEAM WAITING TIME ==========")
print(train.groupby("team")["queue_wait_sec"].describe())

print("\n========== HOUR VALUES ==========")
print(train["hour"].value_counts().sort_index())

print("\n========== TARGET INFORMATION ==========")
print(train["handle_min"].describe())

print("\n========== DATA AUDIT COMPLETED ==========")


print("\n========== STUCK SENSOR DETECTION ==========")

hour_check = train.groupby("date")["hour"].agg(
    ["count", "nunique", "min", "max"]
)

stuck_dates = hour_check[
    (hour_check["nunique"] == 1) &
    (hour_check["min"] == 10)
]

print(stuck_dates)

print("\nFirst suspicious date:")
print(stuck_dates.index.min())

print("\nLast suspicious date:")
print(stuck_dates.index.max())

print("\nTotal suspicious dates:")
print(len(stuck_dates))