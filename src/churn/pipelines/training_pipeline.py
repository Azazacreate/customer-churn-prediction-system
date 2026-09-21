"""Сквозной пайплайн: сбор данных -> EDA -> обучение -> оценка -> отчёт.
Каждый шаг — отдельная функция без побочных эффектов (кроме записи файлов),
поэтому их удобно вызывать как из CLI, так и из Airflow DAG / TaskFlow.
"""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from churn.config import Config, load_config
from churn.data.make_dataset import build_raw_dataset
from churn.features.eda import build_eda_report, plot_churn_distribution
from churn.models.predict import predict_proba
from churn.models.registry import load_meta, load_model
from churn.models.train import train_all
from churn.utils.io import ensure_dir, save_json
from churn.utils.logger import get_logger
log = get_logger("churn.pipelines")
def run_data_step(cfg: Config | None = None, force_synthetic: bool = True) -> str:
    """Шаг 1: собрать сырой датасет."""
    cfg = cfg or load_config()
    build_raw_dataset(cfg, force_synthetic=force_synthetic)
    return str(cfg.paths.resolve("raw_data"))
def run_eda_step(cfg: Config | None = None) -> Dict[str, Any]:
    """Шаг 2: EDA-отчёт."""
    cfg = cfg or load_config()
    df = pd.read_parquet(cfg.paths.resolve("raw_data"))
    report = build_eda_report(df, cfg)
    plot_churn_distribution(df, cfg)
    return report
def run_train_step(cfg: Config | None = None) -> Dict[str, Any]:
    """Шаг 3: обучение моделей."""
    cfg = cfg or load_config()
    return train_all(cfg)
def run_inference_step(cfg: Config | None = None, sample: int = 1000) -> Dict[str, Any]:
    """Шаг 4: скоринг последних клиентов (демонстрация инференса)."""
    cfg = cfg or load_config()
    df = pd.read_parquet(cfg.paths.resolve("raw_data")).head(sample)
    model = load_model(cfg)
    scored = predict_proba(df, model=model, cfg=cfg)
    report_dir = ensure_dir(cfg.paths.resolve("report_dir"))
    out = report_dir / "scored_customers.parquet"
    scored.to_parquet(out, index=False)
    high_risk = int((scored["risk_segment"] == "high").sum())
    log.info("Скоринг готов: %s | high-risk клиентов: %d", out, high_risk)
    return {"scored_path": str(out), "high_risk": high_risk, "n": len(scored)}
def run_full_pipeline(cfg: Config | None = None) -> Dict[str, Any]:
    """Полный сквозной прогон: data -> eda -> train -> inference -> summary."""
    cfg = cfg or load_config()
    log.info(">>> СТАРТ полного пайплайна")
    raw_path = run_data_step(cfg)
    eda = run_eda_step(cfg)
    train = run_train_step(cfg)
    inference = run_inference_step(cfg)
    summary = {
        "raw_data": raw_path,
        "churn_rate": eda.get("churn", {}).get("churn_rate"),
        "best_model": train.get("best_model"),
        "best_score": train.get("best_score"),
        "model_meta": load_meta(cfg),
        "inference": inference,
    }
    save_json(summary, cfg.paths.resolve("report_dir") / "pipeline_summary.json")
    log.info(">>> ПАЙПЛАЙН ЗАВЕРШЁН: best=%s score=%.4f",
             train.get("best_model"), train.get("best_score", 0.0))
    return summary
if __name__ == "__main__":  # pragma: no cover
    run_full_pipeline()
