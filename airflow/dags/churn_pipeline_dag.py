"""Airflow DAG: регулярное переобучение и инференс модели оттока.
Задачи:
  1. collect_data    — сбор данных (REST API / БД / синтетика)
  2. run_eda         — EDA-отчёт
  3. train_models    — обучение + подбор гиперпараметров (MLflow)
  4. validate_model  — проверка, что метрика не ниже порога (quality gate)
  5. batch_inference — скоринг клиентов
  6. notify          — уведомление о результате
DAG построен на TaskFlow API и работает с пакетом `churn`.
"""
from __future__ import annotations
import datetime as dt
from airflow.decorators import dag, task
DEFAULT_ARGS = {
    "owner": "data-science",
    "retries": 2,
    "retry_delay": dt.timedelta(minutes=5),
    "email_on_failure": False,
}
def _min_roc_auc() -> float:
    """Порог качества модели (читается из Airflow Variable при выполнении)."""
    try:
        from airflow.models import Variable
        return float(Variable.get("churn_min_roc_auc", default_var=0.70))
    except Exception:  # pragma: no cover
        return 0.70
@dag(
    dag_id="churn_prediction_pipeline",
    description="Переобучение и инференс модели прогнозирования оттока",
    schedule="0 3 * * *",  # ежедневно в 03:00
    start_date=dt.datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["ml", "churn", "retention"],
)
def churn_pipeline():
    @task
    def collect_data() -> str:
        from churn.config import load_config
        from churn.data.make_dataset import build_raw_dataset
        cfg = load_config()
        build_raw_dataset(cfg, force_synthetic=True)
        return str(cfg.paths.resolve("raw_data"))
    @task
    def run_eda(raw_path: str) -> dict:
        import pandas as pd
        from churn.config import load_config
        from churn.features.eda import build_eda_report
        cfg = load_config()
        df = pd.read_parquet(raw_path)
        return build_eda_report(df, cfg).get("churn", {})
    @task
    def train_models(eda_info: dict) -> dict:
        from churn.config import load_config
        from churn.models.train import train_all
        cfg = load_config()
        result = train_all(cfg)
        return {
            "best_model": result["best_model"],
            "best_score": result["best_score"],
            "results": {k: v["metrics"] for k, v in result["results"].items()},
        }
    @task
    def validate_model(train_result: dict) -> dict:
        best = train_result["results"][train_result["best_model"]]
        roc_auc = best["roc_auc"]
        threshold = _min_roc_auc()
        if roc_auc < threshold:
            raise ValueError(
                f"Quality gate провален: ROC-AUC={roc_auc:.4f} < {threshold}"
            )
        return {"roc_auc": roc_auc, "passed": True}
    @task
    def batch_inference(validation: dict) -> dict:
        from churn.config import load_config
        from churn.pipelines.training_pipeline import run_inference_step
        cfg = load_config()
        return run_inference_step(cfg)
    @task
    def notify(inference: dict) -> None:
        import logging
        logging.getLogger("airflow.task").info(
            "Пайплайн оттока завершён. High-risk клиентов: %s",
            inference.get("high_risk"),
        )
    raw = collect_data()
    eda = run_eda(raw)
    trained = train_models(eda)
    validated = validate_model(trained)
    inferred = batch_inference(validated)
    notify(inferred)
churn_pipeline()
