"""Оценка моделей: ROC-AUC, PR-AUC, F1, Lift@Decile, Precision/Recall@K."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def lift_at_decile(y_true: np.ndarray, y_prob: np.ndarray, decile: int = 1) -> float:
    """Lift в верхнем дециле риска относительно среднего уровня оттока."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    n = max(int(len(df) * decile / 10), 1)
    top = df.iloc[:n]
    overall = df["y"].mean()
    return float(top["y"].mean() / overall) if overall > 0 else float("nan")


def precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.1) -> Dict[str, float]:
    """Precision/Recall в топ-k% клиентов по риску (k — доля 0..1)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = max(int(len(y_true) * k), 1)
    order = np.argsort(y_prob)[::-1][:n]
    top_true = y_true[order]
    tp = top_true.sum()
    precision = tp / n
    recall = tp / max(y_true.sum(), 1)
    return {"precision_at_k": float(precision), "recall_at_k": float(recall)}


def evaluate_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Полный набор метрик для бинарного классификатора оттока."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "lift@1": lift_at_decile(y_true, y_prob, decile=1),
        "lift@2": lift_at_decile(y_true, y_prob, decile=2),
    }
    metrics.update(precision_recall_at_k(y_true, y_prob, k=0.1))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}


def find_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> Dict[str, float]:
    """Подобрать порог решения под бизнес-метрику (f1 | recall@precision)."""
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if metric == "f1":
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        idx = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    else:  # максимизируем recall при precision >= 0.5
        mask = prec[:-1] >= 0.5
        idx = int(np.argmax(np.where(mask, rec[:-1], -1)))
        if not mask.any():
            idx = int(np.argmax(rec[:-1]))
    return {
        "threshold": float(thr[idx]) if idx < len(thr) else 0.5,
        "precision": float(prec[idx]),
        "recall": float(rec[idx]),
    }


def decile_table(y_true: np.ndarray, y_prob: np.ndarray) -> List[Dict[str, float]]:
    """Таблица по децилям риска (для бизнес-отчёта)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, 10, labels=False) + 1
    overall = df["y"].mean()
    rows = []
    for d, grp in df.groupby("decile"):
        rows.append(
            {
                "decile": int(d),
                "n": int(len(grp)),
                "churn_rate": round(float(grp["y"].mean()), 4),
                "lift": round(float(grp["y"].mean() / overall), 3) if overall else None,
                "avg_prob": round(float(grp["p"].mean()), 4),
            }
        )
    return rows
