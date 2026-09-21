"""Фабрика моделей: градиентный бустинг + классические алгоритмы.
Библиотеки XGBoost / LightGBM / CatBoost импортируются лениво: если пакет
не установлен, соответствующий вариант недоступен, а пайплайн продолжает
работать на Scikit-learn моделях (HistGradientBoosting как резерв).
"""
from __future__ import annotations
from typing import Any, Dict
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from churn.utils.logger import get_logger
log = get_logger("churn.models.train")
SEED = 42
def _try_import(module: str):
    try:
        return __import__(module, fromlist=["*"])
    except Exception:  # pragma: no cover
        return None
def available_models() -> Dict[str, bool]:
    """Какие модели реально доступны в окружении."""
    return {
        "logistic_regression": True,
        "random_forest": True,
        "hist_gradient_boosting": True,
        "xgboost": _try_import("xgboost") is not None,
        "lightgbm": _try_import("lightgbm") is not None,
        "catboost": _try_import("catboost") is not None,
    }
def make_model(name: str, params: Dict[str, Any] | None = None, seed: int = SEED):
    """Создать необученную модель по имени с учётом дисбаланса классов."""
    params = dict(params or {})
    name = name.lower()
    if name == "logistic_regression":
        params.setdefault("class_weight", "balanced")
        params.setdefault("max_iter", 2000)
        return LogisticRegression(**params)
    if name == "random_forest":
        params.setdefault("n_estimators", 400)
        params.setdefault("max_depth", None)
        params.setdefault("class_weight", "balanced")
        params.setdefault("n_jobs", -1)
        params.setdefault("random_state", seed)
        return RandomForestClassifier(**params)
    if name == "xgboost":
        xgb = _try_import("xgboost")
        if xgb is None:
            log.warning("xgboost не установлен — использую HistGradientBoosting")
            return HistGradientBoostingClassifier(random_state=seed, **params)
        params.setdefault("n_estimators", 400)
        params.setdefault("eval_metric", "auc")
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("scale_pos_weight", 5.0)
        return xgb.XGBClassifier(**params)
    if name == "lightgbm":
        lgb = _try_import("lightgbm")
        if lgb is None:
            log.warning("lightgbm не установлен — использую HistGradientBoosting")
            return HistGradientBoostingClassifier(random_state=seed, **params)
        params.setdefault("n_estimators", 400)
        params.setdefault("class_weight", "balanced")
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", -1)
        return lgb.LGBMClassifier(**params)
    if name == "catboost":
        cb = _try_import("catboost")
        if cb is None:
            log.warning("catboost не установлен — использую HistGradientBoosting")
            return HistGradientBoostingClassifier(random_state=seed, **params)
        params.setdefault("iterations", 400)
        params.setdefault("auto_class_weights", "Balanced")
        params.setdefault("random_seed", seed)
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        return cb.CatBoostClassifier(**params)
    if name in ("hist_gradient_boosting", "hgb"):
        params.setdefault("random_state", seed)
        return HistGradientBoostingClassifier(**params)
    raise ValueError(f"Неизвестная модель: {name}")
