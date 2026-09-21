"""Тесты feature engineering и препроцессинга."""
import numpy as np
import pandas as pd

from churn.config import Config
from churn.data.synthetic import generate_churn_data
from churn.features.build_features import (
    add_engineered_features,
    build_features,
    build_preprocessor,
    get_feature_columns,
    split_dataset,
)


def _sample_df(n=300):
    return generate_churn_data(n_samples=n, seed=1)


def test_generate_has_expected_columns():
    df = _sample_df()
    for col in ["customer_id", "tenure_months", "monthly_charges", "churn"]:
        assert col in df.columns
    assert df["churn"].isin([0, 1]).all()


def test_engineered_features_created():
    df = add_engineered_features(_sample_df())
    for col in [
        "avg_charge_per_gb",
        "tickets_per_month",
        "charge_to_tenure_ratio",
        "is_new_customer",
        "engagement_score",
    ]:
        assert col in df.columns
    assert np.isfinite(df["avg_charge_per_gb"].dropna()).all()


def test_build_features_shapes():
    cfg = Config()
    X, y, preprocessor, numeric, categorical = build_features(_sample_df(), cfg)
    assert len(X) == len(y)
    assert "churn" not in X.columns
    assert len(numeric) > 0 and len(categorical) > 0
    # препроцессор обучается и выдаёт числовую матрицу
    Xt = preprocessor.fit_transform(X, y)
    assert Xt.shape[0] == len(X)


def test_preprocessor_handles_unknown_categories():
    cfg = Config()
    df = _sample_df()
    X, y, preprocessor, numeric, categorical = build_features(df, cfg)
    preprocessor.fit(X, y)
    X_new = X.copy()
    X_new.loc[X_new.index[0], "contract_type"] = "UNSEEN_VALUE"
    transformed = preprocessor.transform(X_new)
    assert transformed.shape[0] == len(X_new)


def test_split_is_stratified():
    cfg = Config()
    X, y, _, _, _ = build_features(_sample_df(1000), cfg)
    X_tr, X_te, y_tr, y_te = split_dataset(X, y, cfg)
    assert abs(y_tr.mean() - y_te.mean()) < 0.05
