"""Инференс: батч-скоринг и онлайн-предсказание.
Модель — это обученный sklearn Pipeline(preprocessor -> estimator). Для новых
данных достаточно применить тот же feature engineering, что и при обучении,
и передать кадр в пайплайн: он сам выберет нужные колонки.
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from churn.config import Config, load_config
from churn.features.build_features import add_engineered_features
from churn.models.registry import load_model
from churn.utils.logger import get_logger
log = get_logger("churn.models.predict")
RISK_SEGMENTS = [(0.66, "high"), (0.33, "medium"), (0.0, "low")]
def _expected_columns(model: Any) -> List[str] | None:
    """Колонки, которые ожидает обученный препроцессор (если доступны)."""
    try:
        prep = model.named_steps["prep"]
        return list(prep.feature_names_in_)
    except Exception:
        return None
def prepare_inference_frame(df: pd.DataFrame, model: Any = None) -> pd.DataFrame:
    """Добавить engineered-признаки и оставить только нужные колонки."""
    df = add_engineered_features(df)
    if model is not None:
        cols = _expected_columns(model)
        if cols:
            missing = [c for c in cols if c not in df.columns]
            for c in missing:
                df[c] = np.nan
            df = df[cols]
    return df
def risk_segment(prob: float) -> str:
    for threshold, label in RISK_SEGMENTS:
        if prob >= threshold:
            return label
    return "low"
def predict_proba(df: pd.DataFrame, model: Any = None, cfg: Config | None = None) -> pd.DataFrame:
    """Посчитать вероятность оттока для каждого клиента.
    Returns
    -------
    pandas.DataFrame
        Колонки: ``customer_id`` (если есть), ``churn_probability``, ``risk_segment``.
    """
    cfg = cfg or load_config()
    model = model or load_model(cfg)
    frame = prepare_inference_frame(df, model)
    probs = model.predict_proba(frame)[:, 1]
    out = pd.DataFrame({"churn_probability": np.round(probs, 4)})
    if "customer_id" in df.columns:
        out.insert(0, "customer_id", df["customer_id"].values)
    out["risk_segment"] = out["churn_probability"].apply(risk_segment)
    out["predicted_churn"] = (out["churn_probability"] >= cfg.training.decision_threshold).astype(int)
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)
def score_file(
    input_path: str, output_path: str | None = None, cfg: Config | None = None
) -> pd.DataFrame:
    """Батч-скоринг файла (parquet/csv) с сохранением результата."""
    cfg = cfg or load_config()
    if str(input_path).endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)
    scored = predict_proba(df, cfg=cfg)
    if output_path:
        if output_path.endswith(".csv"):
            scored.to_csv(output_path, index=False)
        else:
            scored.to_parquet(output_path, index=False)
        log.info("Скоринг сохранён: %s (%d строк)", output_path, len(scored))
    return scored
def predict_single(record: Dict[str, Any], model: Any = None, cfg: Config | None = None) -> Dict[str, Any]:
    """Онлайн-предсказание для одной записи (используется FastAPI)."""
    df = pd.DataFrame([record])
    result = predict_proba(df, model=model, cfg=cfg).iloc[0].to_dict()
    result["churn_probability"] = float(result["churn_probability"])
    return result
