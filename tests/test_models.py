"""Тесты обучения, оценки и инференса (быстрые, на маленьком датасете)."""
import numpy as np
import pandas as pd
from churn.config import Config
from churn.data.synthetic import generate_churn_data
from churn.features.build_features import build_features, split_dataset
from churn.models.evaluate import evaluate_model, lift_at_decile
from churn.models.factory import make_model
from churn.models.predict import predict_proba, risk_segment
def test_factory_builds_models():
    for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]:
        model = make_model(name)
        assert hasattr(model, "fit")
def test_evaluate_model_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
    metrics = evaluate_model(y_true, y_prob)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert "lift@1" in metrics
def test_lift_at_decile_positive():
    rng = np.random.default_rng(3)
    y_true = rng.binomial(1, 0.2, 1000)
    y_prob = y_true * 0.6 + rng.uniform(0, 0.4, 1000)
    assert lift_at_decile(y_true, y_prob, decile=1) > 1.0
def test_end_to_end_train_and_predict():
    cfg = Config()
    cfg.training.models = ["logistic_regression"]
    cfg.training.search.enabled = False
    cfg.training.cv_folds = 3
    df = generate_churn_data(n_samples=800, seed=7)
    X, y, _, _, _ = build_features(df, cfg)
    X_tr, X_te, y_tr, y_te = split_dataset(X, y, cfg)
    from churn.models.train import _fit_one
    pipe, metrics, _ = _fit_one("logistic_regression", cfg, X_tr, y_tr, X_te, y_te)
    assert metrics["roc_auc"] > 0.5
    scored = predict_proba(df, model=pipe, cfg=cfg)
    assert "churn_probability" in scored.columns
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_segment"]).issubset({"low", "medium", "high"})
def test_risk_segment_bounds():
    assert risk_segment(0.9) == "high"
    assert risk_segment(0.5) == "medium"
    assert risk_segment(0.1) == "low"
