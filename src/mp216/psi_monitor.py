
import numpy as np
import pandas as pd

from src.mp216.data_cleaning import clean_data


FEATURES = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age",
]


def calculate_psi(reference, current, bins=10):
    reference = pd.to_numeric(
        reference, errors="coerce"
    ).dropna().to_numpy()

    current = pd.to_numeric(
        current, errors="coerce"
    ).dropna().to_numpy()

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("No valid data for PSI")

    # Ten equal-width bins based on training values.
    low = min(reference.min(), current.min())
    high = max(reference.max(), current.max())

    if high == low:
        return 0.0

    edges = np.linspace(low, high, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Avoid log(0) using small smoothing.
    ref_pct = np.maximum(
        ref_counts / ref_counts.sum(), 0.0001
    )
    cur_pct = np.maximum(
        cur_counts / cur_counts.sum(), 0.0001
    )

    psi = np.sum(
        (cur_pct - ref_pct)
        * np.log(cur_pct / ref_pct)
    )

    return float(psi)


def main():
    train = clean_data(
        pd.read_csv("data/raw/train.csv"),
        fix_alpha_units=True
    )

    later = clean_data(
        pd.read_csv("data/raw/later.csv"),
        fix_alpha_units=False
    )

    print("\n===== PSI MONITORING =====")

    results = {}

    for feature in FEATURES:
        psi = calculate_psi(
            train[feature],
            later[feature]
        )

        results[feature] = psi
        print(f"{feature}: {psi:.6f}")

    most_drifted = max(results, key=results.get)

    print("\nMost drifted feature:", most_drifted)
    print("Highest PSI:", round(results[most_drifted], 6))

    print("\n===== PSI COMPLETED =====")


if __name__ == "__main__":
    main()
