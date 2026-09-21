"""Feature engineering + препроцессинг.
Модуль строит sklearn-совместимый препроцессор (`ColumnTransformer`) и
создаёт новые признаки:
  * avg_charge_per_gb       — удельная стоимость трафика;
  * tickets_per_month       — интенсивность обращений в поддержку;
  * charge_to_tenure_ratio  — цена к «стажу» клиента;
  * is_new_customer         — новый клиент (tenure <= 3);
  * engagement_score        — сводный индекс вовлечённости.
Категориальные признаки кодируются One-Hot, числовые — импутируются
медианой и масштабируются (для линейных моделей).
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from churn.config import Config, load_config
from churn.utils.logger import get_logger
log = get_logger("churn.features.build")
TARGET = "churn"
_IMPUTE_ALIASES = {"mode": "most_frequent", "avg": "mean"}
def _impute_strategy(value: str) -> str:
    return _IMPUTE_ALIASES.get(value, value)
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создать производные признаки (без утечки целевой переменной)."""
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["avg_charge_per_gb"] = (
            df["monthly_charges"] / df["avg_monthly_gb"].replace(0, np.nan)
        ).round(4)
        df["charge_to_tenure_ratio"] = (
            df["monthly_charges"] / df["tenure_months"].replace(0, np.nan)
        ).round(4)
    df["tickets_per_month"] = (
        df["num_support_tickets"] / df["tenure_months"].replace(0, np.nan)
    ).round(4)
    df["is_new_customer"] = (df["tenure_months"] <= 3).astype(int)
    engagement = (
        np.log1p(df["num_logins_30d"])
        - np.log1p(df["days_since_last_login"])
        + df["satisfaction_score"].fillna(df["satisfaction_score"].median())
    )
    df["engagement_score"] = engagement.round(4)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    log.info(
        "Добавлены engineered-признаки: %s",
        ["avg_charge_per_gb", "tickets_per_month", "charge_to_tenure_ratio",
         "is_new_customer", "engagement_score"],
    )
    return df
def get_feature_columns(df: pd.DataFrame, cfg: Config) -> Tuple[List[str], List[str]]:
    """Разделить колонки на числовые и категориальные (исключая drop/target)."""
    drop = set(cfg.features.drop_columns) | {TARGET}
    features = [c for c in df.columns if c not in drop]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical
def build_preprocessor(
    numeric: List[str], categorical: List[str], cfg: Config | None = None
) -> ColumnTransformer:
    """Собрать ColumnTransformer: impute + scale (числа) и impute + OHE (категории)."""
    cfg = cfg or load_config()
    num_strategy = _impute_strategy(cfg.features.numeric_impute)
    cat_strategy = _impute_strategy(cfg.features.categorical_impute)
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=num_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cat_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
def build_features(
    df: pd.DataFrame, cfg: Config | None = None
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, List[str], List[str]]:
    """Полный feature-инжиниринг: вернуть X (необработанный), y, препроцессор, списки колонок.
    Препроцессор возвращается необученным — он обучается внутри CV в train.py,
    чтобы избежать утечки данных (data leakage).
    """
    cfg = cfg or load_config()
    df = add_engineered_features(df)
    numeric, categorical = get_feature_columns(df, cfg)
    X = df[numeric + categorical]
    y = df[TARGET].astype(int)
    preprocessor = build_preprocessor(numeric, categorical, cfg)
    log.info("Признаки: %d числовых, %d категориальных, всего %d",
             len(numeric), len(categorical), len(numeric) + len(categorical))
    return X, y, preprocessor, numeric, categorical
def split_dataset(
    X: pd.DataFrame, y: pd.Series, cfg: Config | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Стратифицированное разбиение train/test."""
    from sklearn.model_selection import train_test_split
    cfg = cfg or load_config()
    return train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.project.seed,
        stratify=y,
    )
if __name__ == "__main__":  # pragma: no cover
    _cfg = load_config()
    _df = pd.read_parquet(_cfg.paths.resolve("raw_data"))
    X_, y_, _, n_, c_ = build_features(_df, _cfg)
    print("X shape:", X_.shape, "| pos rate:", round(y_.mean(), 4))
