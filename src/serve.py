"""
serve.py
========
FastAPI-сервис инференса модели оттока.

Запуск локально:
    uvicorn src.serve:app --reload --port 8000

Эндпоинты:
    GET  /health          -- проверка живости
    POST /predict         -- предсказание вероятности оттока для клиента
"""


from __future__ import annotations
import os
from typing import List
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from churn_lib import FEATURE_COLS


app = FastAPI(title="Churn Prediction API", version="1.0.0")
MODEL = None
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn_model")


def load_model():
    global MODEL
    try:
        import mlflow.pyfunc
        mlflow.set_tracking_uri(MLFLOW_URI)
        MODEL = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
        print("Model loaded from MLflow registry.")
    except Exception as e:  # noqa: BLE001
        print(f"MLflow model not available ({e}); using heuristic fallback.")
        MODEL = None


@app.on_event("startup")
def _startup():
    load_model()


class CustomerFeatures(BaseModel):
    frequency: float
    monetary: float
    avg_order_value: float
    total_items: float
    avg_freight: float
    avg_review: float
    max_installments: float
    tenure: float
    avg_days_between_orders: float
    freight_ratio: float
    items_per_order: float
    is_repeat: int


def _fallback_score(f: CustomerFeatures) -> float:
    score = 0.5
    score += 0.2 if f.frequency <= 1 else -0.2
    score += 0.1 if f.avg_review < 3 else -0.1
    score += 0.1 if f.is_repeat == 0 else -0.1
    return float(np.clip(score, 0.0, 1.0))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict")
def predict(items: List[CustomerFeatures]):
    out = []
    for f in items:
        if MODEL is not None:
            X = np.array([[getattr(f, c) for c in FEATURE_COLS]])
            proba = float(MODEL.predict(X)[0])
        else:
            proba = _fallback_score(f)
        out.append({
            "churn_proba": round(proba, 4),
            "recommend_offer": proba >= 0.5,
        })
    return {"predictions": out}
