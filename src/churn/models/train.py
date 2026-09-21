"""Обучение и подбор гиперпараметров моделей оттока.

Каждая модель обучается как sklearn Pipeline(preprocessor -> estimator) внутри
стратифицированной кросс-валидации, что исключает утечку данных. Опционально
запускается RandomizedSearchCV. Результаты логируются в MLflow.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from churn.config import Config, load_config
from churn.features.build_features import (
    TARGET,
    build_features,
    build_preprocessor,
    get_feature_columns,
    split_dataset,
)
from churn.models.evaluate import evaluate_model, find_best_threshold
from churn.models.factory import make_model
from churn.models.registry import save_model
from churn.utils.logger import get_logger

log = get_logger("churn.models.train")

# имена гиперпараметров, для которых лучше логарифмическая шкала
_LOG_SCALE = ("learning_rate", "lr", "alpha", "reg_lambda", "reg_alpha")


def _scoring(cfg: Config) -> str:
    return "average_precision" if cfg.training.scoring == "pr_auc" else "roc_auc"


def _maybe_mlflow(cfg: Config):
    """Настроить MLflow, если он включён и установлен."""
    if not cfg.mlflow.enabled:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        return mlflow
    except Exception as exc:  # pragma: no cover
        log.warning("MLflow недоступен: %s", exc)
        return None


class _NullCtx:
    """Заглушка контекстного менеджера, когда MLflow выключен."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def to_search_space(params: Dict[str, list]) -> Dict[str, Any]:
    """Преобразовать YAML-описание сетки в пространство для RandomizedSearchCV.

    * список из 2 чисел ``[lo, hi]`` -> непрерывное/дискретное распределение
      (для ``learning_rate`` и подобных — логарифмическое);
    * список из >2 значений -> дискретный выбор ``choices``.
    """
    space: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(
            isinstance(v, (int, float)) for v in value
        ):
            lo, hi = value
            if any(tag in key for tag in _LOG_SCALE):
                space[key] = stats.loguniform(lo, hi)
            elif all(isinstance(v, int) for v in value):
                space[key] = stats.randint(lo, hi + 1)
            else:
                space[key] = stats.uniform(lo, hi - lo)
        else:
            space[key] = value
    return space


def _fit_one(
    name: str, cfg: Config, X_tr: pd.DataFrame, y_tr: pd.Series,
    X_te: pd.DataFrame, y_te: pd.Series,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """Обучить одну модель, вернуть (pipeline, метрики, best_params)."""
    # колонки определяем строго по train-части (без утечки из test)
    tmp = X_tr.copy()
    tmp[TARGET] = y_tr.values
    numeric, categorical = get_feature_columns(tmp, cfg)
    preprocessor = build_preprocessor(numeric, categorical, cfg)

    base_params = cfg.training.search.params.get(name, {})
    estimator = make_model(name, seed=cfg.project.seed)
    pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
    best_params: Dict[str, Any] = {}

    if cfg.training.search.enabled and base_params:
        space = {f"clf__{k}": v for k, v in to_search_space(base_params).items()}
        search = RandomizedSearchCV(
            pipe,
            param_distributions=space,
            n_iter=cfg.training.search.n_iter,
            scoring=_scoring(cfg),
            cv=StratifiedKFold(cfg.training.cv_folds, shuffle=True, random_state=cfg.project.seed),
            random_state=cfg.project.seed,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        pipe = search.best_estimator_
        best_params = {k: str(v) for k, v in search.best_params_.items()}
        cv_score = float(search.best_score_)
    else:
        cv = StratifiedKFold(cfg.training.cv_folds, shuffle=True, random_state=cfg.project.seed)
        scores = cross_val_score(pipe, X_tr, y_tr, scoring=_scoring(cfg), cv=cv, n_jobs=-1)
        cv_score = float(np.mean(scores))
        pipe.fit(X_tr, y_tr)

    y_prob = pipe.predict_proba(X_te)[:, 1]
    thr_info = find_best_threshold(y_te.values, y_prob, metric="f1")
    metrics = evaluate_model(y_te.values, y_prob, threshold=thr_info["threshold"])
    metrics["cv_score"] = round(cv_score, 4)
    return pipe, metrics, best_params


def train_all(cfg: Config | None = None) -> Dict[str, Any]:
    """Обучить все модели из конфига и сохранить лучшую."""
    cfg = cfg or load_config()
    raw = pd.read_parquet(cfg.paths.resolve("raw_data"))
    X, y, _, _, _ = build_features(raw, cfg)
    X_tr, X_te, y_tr, y_te = split_dataset(X, y, cfg)

    mlflow = _maybe_mlflow(cfg)
    results: Dict[str, Any] = {}
    best_name, best_score, best_pipe = None, -1.0, None

    for name in cfg.training.models:
        log.info("=== Обучение модели: %s ===", name)
        run_ctx = mlflow.start_run(run_name=name) if mlflow else _NullCtx()
        with run_ctx:
            try:
                pipe, metrics, best_params = _fit_one(name, cfg, X_tr, y_tr, X_te, y_te)
            except Exception as exc:  # pragma: no cover
                log.error("Модель %s упала: %s", name, exc)
                continue

            results[name] = {"metrics": metrics, "params": best_params}
            if mlflow:
                mlflow.log_params({"model": name, **best_params})
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                )

            score = metrics["pr_auc"] if cfg.training.scoring == "pr_auc" else metrics["roc_auc"]
            if score > best_score:
                best_name, best_score, best_pipe = name, score, pipe
            log.info(
                "%s: roc_auc=%.4f pr_auc=%.4f cv=%.4f",
                name, metrics["roc_auc"], metrics["pr_auc"], metrics["cv_score"],
            )

    if best_pipe is not None:
        meta = {
            "model": best_name,
            "score": best_score,
            "metrics": results[best_name]["metrics"],
        }
        save_model(best_pipe, cfg, meta)
        if mlflow:
            mlflow.log_metric("best_score", best_score)

    log.info("Лучшая модель: %s (score=%.4f)", best_name, best_score)
    return {"best_model": best_name, "best_score": best_score, "results": results}


if __name__ == "__main__":  # pragma: no cover
    train_all()
