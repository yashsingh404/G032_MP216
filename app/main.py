import json
import logging
import sys
import joblib
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import os
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }

        if hasattr(record, "prediction_min"):
            log_data["prediction_min"] = record.prediction_min

        return json.dumps(log_data)


logger = logging.getLogger("mp216_api")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)




MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/MP216_Helpline_Model@champion"
)

model = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if MODEL_URI.endswith(".joblib"):
        model = joblib.load(MODEL_URI)
        logger.info("joblib_model_loaded")
    else:
        model = mlflow.sklearn.load_model(MODEL_URI)
        logger.info("mlflow_model_loaded")

    yield



app = FastAPI(
    title="MP216 Helpline Prediction API",
    version="1.0",
    lifespan=lifespan,
)

class CallInput(BaseModel):
    queue_wait_sec: float = Field(ge=0)
    agent_exp_months: float = Field(ge=0)
    previous_calls: int = Field(ge=0, le=20)
    issue_count: int = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    caller_age: int
    language: str
    team: str

    model_config = {"extra": "forbid"}



@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_URI}


@app.post("/predict")
def predict(call: CallInput):
    
    row = pd.DataFrame([call.model_dump()])

    # API inputs already provide queue_wait_sec in seconds.
    # Do not apply the historical Team Alpha unit correction.

    row["caller_age"] = row["caller_age"].replace(
       -999, float("nan")
    )

    row["language"] = (
        row["language"]
        .str.strip()
        .str.lower()
        .replace({
            "eng.": "english",
           "hin.": "hindi",
            "reg.": "regional"
        })
    )


    features = [
        "queue_wait_sec",
        "agent_exp_months",
        "previous_calls",
        "issue_count",
        "hour",
        "caller_age",
        "language",
        "team",
    ]

    
    prediction = model.predict(row[features])[0]
    prediction_min = round(float(prediction), 2)

    logger.info(
        "prediction_completed",
        extra={"prediction_min": prediction_min},
    )

    return {
        "predicted_handle_min": prediction_min
    }

