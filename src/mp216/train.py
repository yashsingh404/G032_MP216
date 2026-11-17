
import pandas as pd
import mlflow
import mlflow.sklearn


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.mp216.data_cleaning import clean_data


# 1. Load and clean the original training data
df = pd.read_csv("data/raw/train.csv")
df = clean_data(df)

# 2. Select features and target
features = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age",
    "language",
    "team"
]

X = df[features]
y = df["handle_min"]

# 3. Split data into training and validation sets

# Sort records by date before splitting
df = df.sort_values("date").reset_index(drop=True)

# Use the first 80% for training
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
val_df = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df["handle_min"]

X_val = val_df[features]
y_val = val_df["handle_min"]

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))

print("Training date range:",
      train_df["date"].min(),
      train_df["date"].max())

print("Validation date range:",
      val_df["date"].min(),
      val_df["date"].max())


# 4. Prepare numerical and categorical features
numeric_features = [
    "queue_wait_sec",
    "agent_exp_months",
    "previous_calls",
    "issue_count",
    "hour",
    "caller_age"
]

categorical_features = ["language", "team"]

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

# 5. Configure MLflow
mlflow.set_experiment("MP216_Helpline_Regression")

# 6. Train three model configurations
for n_estimators in [50, 100, 200]:

    with mlflow.start_run():

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42
            ))
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        mae = mean_absolute_error(y_val, predictions)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        print(
            f"Random Forest {n_estimators} trees - MAE: {mae:.4f}"
        )

print("\nTraining and MLflow tracking completed!")
