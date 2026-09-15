"""
Employee Attrition Prediction API
A lightweight, self-contained FastAPI service wrapping a trained scikit-learn
LogisticRegression model (MLflow Model Registry Version 2, F1 = 0.5653).

No MLflow tracking server or SQLite backend needed at runtime — the model
was exported once to model.pkl, so this service has zero dependency on the
machine or paths used during training.
"""
import json
import os
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model", "feature_columns.json")

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predicts whether an employee is likely to leave (1) or stay (0).",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)
with open(COLUMNS_PATH) as f:
    FEATURE_COLUMNS: List[str] = json.load(f)


class PredictionRequest(BaseModel):
    # One record as a dict of {feature_name: value}, matching FEATURE_COLUMNS.
    records: List[dict] = Field(
        ...,
        description="List of employee feature records to score.",
        examples=[[{col: 0.0 for col in FEATURE_COLUMNS}]],
    )


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]


@app.get("/")
def root():
    return {
        "service": "Employee Attrition Prediction API",
        "status": "ok",
        "model": "EmployeeAttrition_BestModel v2 (LogisticRegression)",
        "n_features_expected": len(FEATURE_COLUMNS),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/features")
def features():
    """Returns the exact feature names and order the model expects."""
    return {"feature_columns": FEATURE_COLUMNS}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided.")
    try:
        df = pd.DataFrame(request.records)
        missing = set(FEATURE_COLUMNS) - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature columns: {sorted(missing)}",
            )
        df = df[FEATURE_COLUMNS]  # enforce correct column order
        preds = model.predict(df).tolist()
        probs = model.predict_proba(df)[:, 1].tolist()
        return {"predictions": preds, "probabilities": probs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
