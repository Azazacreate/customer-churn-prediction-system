"""Быстрый сквозной демо-прогон (уменьшенные настройки, без MLflow).
Запуск:
    PYTHONPATH=src python scripts/run_demo.py
Показывает полный цикл: генерация данных -> EDA -> обучение -> инференс -> A/B.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn.ab_testing.ab_test import proportion_test, summary_table  # noqa: E402
from churn.ab_testing.power_analysis import sample_size_proportions  # noqa: E402
from churn.config import load_config  # noqa: E402
from churn.data.make_dataset import build_raw_dataset  # noqa: E402
from churn.features.eda import build_eda_report  # noqa: E402
from churn.models.train import train_all  # noqa: E402
from churn.pipelines.training_pipeline import run_inference_step  # noqa: E402
def main() -> None:
    cfg = load_config()
    cfg.mlflow.enabled = False          # демо без MLflow
    cfg.synthetic.n_samples = 8000
    cfg.training.models = ["logistic_regression", "xgboost", "catboost"]
    cfg.training.search.enabled = False
    cfg.training.cv_folds = 3
    print("\n[1/5] Сбор данных...")
    build_raw_dataset(cfg, force_synthetic=True)
    print("\n[2/5] EDA...")
    import pandas as pd
    df = pd.read_parquet(cfg.paths.resolve("raw_data"))
    eda = build_eda_report(df, cfg)
    print(f"   churn_rate = {eda['churn']['churn_rate']:.3f}, "
          f"строк = {eda['stats']['n_rows']}")
    print("\n[3/5] Обучение моделей...")
    result = train_all(cfg)
    for name, info in result["results"].items():
        m = info["metrics"]
        print(f"   {name:20s} ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")
    print(f"   >>> лучшая модель: {result['best_model']}")
    print("\n[4/5] Инференс...")
    inference = run_inference_step(cfg, sample=1000)
    print(f"   high-risk клиентов: {inference['high_risk']} из {inference['n']}")
    print("\n[5/5] A/B-тест удержания...")
    ab = proportion_test(420, 5000, 520, 5000)
    print(summary_table(ab))
    print("\n   Планирование теста (MDE=25%):")
    print("  ", sample_size_proportions(0.08, mde_relative=0.25)["sample_size_per_group"],
          "клиентов на группу")
    print("\nГотово ✅")
if __name__ == "__main__":
    main()
