# Data Cleaning and Quality Checks

## Why cleaning was needed

The helpline dataset contains some values that could mislead the regression model if they were used without checking.

The original CSV files are kept unchanged. Cleaning is performed in Python before the data is used for downstream processing and modeling.

## Datasets

The project uses three datasets:

- `train.csv`: 1,000 labeled training records.
- `later.csv`: 300 labeled, more recent records.
- `holdout_features.csv`: 250 unlabeled records used for final predictions.

## Issues identified

### 1. Waiting-time unit mix-up

Historical Team Alpha training records contain `queue_wait_sec` values that require unit correction.

For these historical training records, the `queue_wait_sec` value is multiplied by 60.

The later and holdout datasets already use seconds, so this correction is not applied to them.

### 2. Sentinel value for caller age

The value `caller_age = -999` represents missing information.

It is replaced with a missing value so that downstream imputation can handle it correctly.

### 3. Inconsistent language labels

Different abbreviations are standardized:

- `eng.` → `english`
- `hin.` → `hindi`
- `reg.` → `regional`

This makes the categorical values consistent.

### 4. Suspicious hour values

Suspicious `hour` values occurring between 2026-02-02 and 2026-02-21 are treated as missing during cleaning.

## Where the cleaning happens

The cleaning logic is implemented in:

`src/mp216/data_cleaning.py`

Training data uses:

`clean_data(..., fix_alpha_units=True)`

Later and holdout datasets use:

`fix_alpha_units=False`

## Why the distinction matters

The historical Team Alpha unit correction must not be applied to the newer datasets.

Applying the correction to later or holdout records would create artificially large waiting times and could negatively affect predictions.

## Quality checks

Before using the cleaned data, the cleaning rules should be checked against the implementation in `src/mp216/data_cleaning.py` and the recorded findings in `findings.yaml`.

The original CSV files remain unchanged.

## Conclusion

The cleaning process handles unit inconsistencies, missing-value sentinels, inconsistent categorical labels, and suspicious hour values.

