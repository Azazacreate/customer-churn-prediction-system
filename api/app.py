"""FastAPI-сервис онлайн-скоринга вероятности оттока.
Эндпоинты:
  GET  /health           — проверка живости
  GET  /model/info       — метаданные загруженной модели
  POST /predict          — скоринг одной записи
  POST /predict/batch    — скоринг списка записей
  POST /ab-test          — расчёт A/B-теста удержания
Модель загружается один раз при старте (lifespan) и переиспользуется.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from churn.ab_testing.ab_test import proportion_test
from churn.config import load_config
from churn.models.predict import predict_single
from churn.models.registry import load_meta, load_model
from churn.utils.logger import get_logger
log = get_logger("churn.api")
STATE: Dict[str, Any] = {}
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модель при старте приложения."""
    cfg = load_config()
    try:
        STATE["model"] = load_model(cfg)
        STATE["meta"] = load_meta(cfg)
        log.info("Модель загружена: %s", STATE["meta"].get("model"))
    except FileNotFoundError as exc:
        STATE["model"] = None
        log.warning("Модель не найдена: %s", exc)
    STATE["config"] = cfg
    yield
    STATE.clear()
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Онлайн-скоринг вероятности оттока клиентов (Classical ML & A/B)",
    version="1.0.0",
    lifespan=lifespan,
)
class Customer(BaseModel):
    """Признаки клиента для скоринга (совпадают с витриной)."""
    customer_id: Optional[str] = None
    tenure_months: int = Field(..., ge=0, le=120)
    contract_type: str = "Month-to-month"
    payment_method: str = "Electronic check"
    internet_service: str = "Fiber optic"
    region: str = "North"
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    avg_monthly_gb: float = Field(50.0, ge=0)
    num_support_tickets: int = Field(0, ge=0)
    days_since_last_login: float = Field(0, ge=0)
    num_logins_30d: int = Field(0, ge=0)
    satisfaction_score: float = Field(3.5, ge=1, le=5)
    age: int = Field(35, ge=18, le=100)
class AbRequest(BaseModel):
    control_engaged: int
    control_total: int
    treat_engaged: int
    treat_total: int
    alpha: float = 0.05
    metric: str = "retention_rate"
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": STATE.get("model") is not None}
@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    if not STATE.get("meta"):
        raise HTTPException(status_code=404, detail="Модель не загружена")
    return STATE["meta"]
@app.post("/predict")
def predict(customer: Customer) -> Dict[str, Any]:
    if STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return predict_single(customer.model_dump(), model=STATE["model"], cfg=STATE["config"])
@app.post("/predict/batch")
def predict_batch(customers: List[Customer]) -> List[Dict[str, Any]]:
    if STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    import pandas as pd
    from churn.models.predict import predict_proba
    df = pd.DataFrame([c.model_dump() for c in customers])
    scored = predict_proba(df, model=STATE["model"], cfg=STATE["config"])
    return scored.to_dict(orient="records")
@app.post("/ab-test")
def ab_test(req: AbRequest) -> Dict[str, Any]:
    result = proportion_test(
        control_engaged=req.control_engaged,
        control_total=req.control_total,
        treat_engaged=req.treat_engaged,
        treat_total=req.treat_total,
        alpha=req.alpha,
        metric=req.metric,
    )
    return result.to_dict()
