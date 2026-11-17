
from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field




MODEL_URI = "models:/MP216_Helpline_Model@champion"
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = mlflow.sklearn.load_model(MODEL_URI)
    print("Champion model loaded successfully!")
    yield


app = FastAPI(
    title="MP216 Helpline Prediction API",
    version="1.0",
    lifespan=lifespan,
)


class CallInput(BaseModel):
    date: str
    queue_wait_sec: float = Field(ge=0)
    agent_exp_months: float = Field(ge=0)
    previous_calls: int = Field(ge=0)
    issue_count: int = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    caller_age: int
    language: str
    team: str


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

    return {
        "predicted_handle_min": round(float(prediction), 2)
    }
