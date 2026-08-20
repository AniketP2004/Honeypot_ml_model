from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from schemas import LogEntry
from sqlalchemy import desc
import pandas as pd
import os

from features import build_features_row, get_preprocessor
from database import get_db, engine, Base
from models import ClassificationLog
from schemas import PredictRequest, PredictResponse, LogEntry
from ml_registry import REG
import ml_registry as ml_models


app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return{"Status": "Ok"}

@app.on_event("startup")
def startup():
    ml_models.load_all()

VALID_MODELS = ["random_forest", "xgboost", "gradient_boosting"]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    if request.model_name not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"model_name must be one of the {VALID_MODELS}")

    df = build_features_row(request.dict())
    preprocessor = REG["preprocessor"]
    model = REG[request.model_name]
    le_multi = REG["le_multi"]

    features = preprocessor.transform(df)

    pred_encoded = model.predict(features)[0]
    prediction = le_multi.inverse_transform([pred_encoded])[0]
    confidence = max(model.predict_proba(features)[0])

    payload_val = df['payload_size'].iloc[0]

    log_entry = ClassificationLog(
        predicted_class = str(prediction),
        confidence = float(confidence),
        ip_request_rate = float(df['ip_request_rate'].iloc[0]),
        is_exploit_path = int(df['is_exploit_path'].iloc[0]),
        is_known_scanner = int(df['is_known_scanner'].iloc[0]),
        payload_size = float(payload_val) if pd.notna(payload_val) else None

    )

    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    return PredictResponse(model_used=request.model_name, predicted_class=str(prediction), confidence=float(confidence))

@app.get("/models")
def list_models():
    return {"Available_models": VALID_MODELS}

@app.get("/logs", response_model=List[LogEntry])
def get_logs(
    limit: int= 50,
    predicted_class: str | None = None,
    db: Session = Depends(get_db)
):
    query = db.query(ClassificationLog).order_by(desc(ClassificationLog.timestamp))
    if predicted_class:
        query= query.filter(ClassificationLog.predicted_class== predicted_class)
    return query.limit(limit).all()

