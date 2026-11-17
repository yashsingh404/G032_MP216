
import mlflow
from mlflow import MlflowClient

# 1. Configuration
RUN_ID = "2bbef882a1fb44b7860ab2de434bc070"
MODEL_NAME = "MP216_Helpline_Model"

# 2. Connect to the local MLflow tracking server
client = MlflowClient()

# 3. Find the logged model in the selected run
logged_models = client.search_logged_models(
    experiment_ids=["1"],
    filter_string=f"source_run_id = '{RUN_ID}'"
)

if not logged_models:
    raise RuntimeError(
        "No logged model found for this Run ID."
    )

if len(logged_models) != 1:
    raise RuntimeError(
        "Multiple logged models found. "
        "Verify the correct model before registering."
    )

model = logged_models[0]

print("Found model:", model.model_id)

# 4. Register the logged model
registered = mlflow.register_model(
    model_uri=f"models:/{model.model_id}",
    name=MODEL_NAME
)

# 5. Assign champion alias
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=registered.version
)

# 6. Verify the alias
champion = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias="champion"
)

print("\nMODEL REGISTRATION SUCCESSFUL!")
print("Model name:", MODEL_NAME)
print("Version:", registered.version)
print("Champion version:", champion.version)
print("Champion run ID:", champion.run_id)
