"""Отбор признаков: важность модели, L1-регуляризация и RFE.

Работает поверх уже обученного препроцессора: получаем матрицу признаков
с осмысленными именами и ранжируем их.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression

from churn.utils.logger import get_logger

log = get_logger("churn.features.selection")


def _get_names(preprocessor, numeric: List[str], categorical: List[str]) -> np.ndarray:
    """Получить человекочитаемые имена признаков после препроцессинга."""
    try:
        return preprocessor.get_feature_names_out()
    except Exception:  # pragma: no cover
        return np.array(numeric + categorical)


def select_by_importance(
    model, X, y, preprocessor, numeric, categorical, top_k: int = 30
) -> Tuple[List[str], List[float]]:
    """Отбор по feature_importances_ / коэффициентам модели."""
    preprocessor.fit(X, y)
    Xt = preprocessor.transform(X)
    names = _get_names(preprocessor, numeric, categorical)

    model.fit(Xt, y)
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_).ravel()
    else:  # pragma: no cover
        scores = np.ones(Xt.shape[1])

    order = np.argsort(scores)[::-1][:top_k]
    selected = [str(names[i]) for i in order]
    log.info("Отобрано %d признаков методом importance", len(selected))
    return selected, [float(scores[i]) for i in order]


def select_by_l1(
    X, y, preprocessor, numeric, categorical, C: float = 0.5
) -> List[str]:
    """Отбор через L1-регуляризацию (SelectFromModel)."""
    preprocessor.fit(X, y)
    Xt = preprocessor.transform(X)
    names = _get_names(preprocessor, numeric, categorical)

    selector = SelectFromModel(
        LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000),
        threshold="median",
    ).fit(Xt, y)
    mask = selector.get_support()
    selected = [str(n) for n, keep in zip(names, mask) if keep]
    log.info("Отобрано %d признаков методом L1", len(selected))
    return selected


def select_by_rfe(
    X, y, preprocessor, numeric, categorical, n_features: int = 20
) -> List[str]:
    """Рекурсивное исключение признаков (RFE) на логистической регрессии."""
    preprocessor.fit(X, y)
    Xt = preprocessor.transform(X)
    names = _get_names(preprocessor, numeric, categorical)

    n_features = min(n_features, Xt.shape[1])
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=n_features).fit(Xt, y)
    selected = [str(n) for n, keep in zip(names, rfe.support_) if keep]
    log.info("Отобрано %d признаков методом RFE", len(selected))
    return selected


def select_features(
    method: str, model, X, y, preprocessor, numeric, categorical, **kwargs
) -> List[str]:
    """Диспетчер методов отбора признаков."""
    if method == "l1":
        return select_by_l1(X, y, preprocessor, numeric, categorical, **kwargs)
    if method == "rfe":
        return select_by_rfe(X, y, preprocessor, numeric, categorical, **kwargs)
    selected, _ = select_by_importance(
        model, X, y, preprocessor, numeric, categorical,
        top_k=kwargs.get("top_k", 30),
    )
    return selected
