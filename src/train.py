"""
train.py
python src/train.py --mlflow-uri http://localhost:5000
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from churn_lib import (  # noqa: E402
    load_raw, build_rfm_features, FEATURE_COLS,
    split_data, evaluate, find_optimal_threshold, DATA_DIR,
)
RANDOM_STATE = 42
COST_OFFER, CLV = 5.0, 120.0


def build_dataset() -> pd.DataFrame:
    raw = load_raw(DATA_DIR)
    df = build_rfm_features(raw, churn_days=180)
    df["avg_days_between_orders"] = df["avg_days_between_orders"].fillna(
        df["avg_days_between_orders"].median())
    df["avg_review"] = df["avg_review"].fillna(df["avg_review"].median())
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def get_models(scale_pos_weight: float):
    models = {}
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, random_seed=RANDOM_STATE,
            verbose=False, allow_writing_files=False)
    except Exception:
        pass
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, verbose=-1)
    except Exception:
        pass
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0)
    except Exception:
        pass
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    ap.add_argument("--experiment", default="churn_prediction")
    args = ap.parse_args()
    df = build_dataset()
    X_tr, X_te, y_tr, y_te = split_data(df, test_size=0.25, seed=RANDOM_STATE)
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    best = {"pr_auc": -1.0, "name": None, "model": None, "metrics": None}
    for name, model in get_models(spw).items():
        with mlflow.start_run(run_name=name):
            model.fit(X_tr, y_tr)
            metrics = evaluate(model, X_te, y_te, cost_offer=COST_OFFER, clv=CLV)
            best_t, best_profit = find_optimal_threshold(model, X_te, y_te, COST_OFFER, CLV)
            mlflow.log_params({"model": name, "scale_pos_weight": round(spw, 3),
                               "n_features": len(FEATURE_COLS)})
            mlflow.log_metrics({
                "pr_auc": metrics["pr_auc"], "roc_auc": metrics["roc_auc"],
                "f1": metrics["f1"], "net_profit": best_profit,
                "optimal_threshold": best_t,
            })
            mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"[{name}] PR-AUC={metrics['pr_auc']:.4f} ROC-AUC={metrics['roc_auc']:.4f} "
                  f"profit={best_profit:,.0f} thr={best_t:.2f}")
            if metrics["pr_auc"] > best["pr_auc"]:
                best.update({"pr_auc": metrics["pr_auc"], "name": name,
                             "model": model, "metrics": metrics})
    print(f"\n🏆 Best model: {best['name']} (PR-AUC={best['pr_auc']:.4f})")
    return best


if __name__ == "__main__":
    main()
