"""
churn_pipeline_dag.py
Apache Airflow DAG для системы прогнозирования оттока.
  * churn_retraining  -- еженедельное переобучение модели (weekly)
  * churn_inference   -- ежедневный инференс и выгрузка скоров в CRM (daily)

Запуск: положить файл в $AIRFLOW_HOME/dags/
"""
from __future__ import annotations
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


PROJECT_DIR = "/opt/airflow/project"
PY = "python"


default_args = {
    "owner": "ml-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}
with DAG(
    dag_id="churn_retraining",
    description="Weekly retraining of the churn model (CatBoost/LightGBM/XGBoost)",
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "ml", "training"],
) as retrain_dag:
    extract = BashOperator(
        task_id="extract_datamart",
        bash_command=f"cd {PROJECT_DIR} && {PY} src/pipeline.py --stage extract",
    )
    validate = BashOperator(
        task_id="validate_data",
        bash_command=f"cd {PROJECT_DIR} && {PY} src/pipeline.py --stage validate",
    )
    train = BashOperator(
        task_id="train_models",
        bash_command=f"cd {PROJECT_DIR} && {PY} src/train.py --mlflow-uri http://mlflow:5000",
    )
    evaluate = BashOperator(
        task_id="evaluate_and_promote",
        bash_command=f"cd {PROJECT_DIR} && {PY} src/pipeline.py --stage evaluate --promote-if-better",
    )
    register = BashOperator(
        task_id="register_model",
        bash_command=f"cd {PROJECT_DIR} && {PY} src/pipeline.py --stage register",
    )
    extract >> validate >> train >> evaluate >> register


def push_scores_to_crm(**context):
    import pandas as pd
    print("Pushing churn scores to CRM ...")


with DAG(
    dag_id="churn_inference",
    description="Daily churn scoring and CRM push",
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["churn", "ml", "inference"],
) as inference_dag:
    load_features = BashOperator(
        task_id="load_features",
        bash_command=f"cd {PROJECT_DIR} && {PY} src/pipeline.py --stage load_features",
    )
    predict = BashOperator(
        task_id="predict",
        bash_command=f"cd {PROJECT_DIR} && {PY} src/inference.py --output artifacts/scores.parquet",
    )
    push = PythonOperator(
        task_id="push_to_crm",
        python_callable=push_scores_to_crm,
    )
    monitor = BashOperator(
        task_id="monitor_drift",
        bash_command=f"cd {PROJECT_DIR} && {PY} src/pipeline.py --stage monitor_drift",
    )
    load_features >> predict >> push >> monitor
