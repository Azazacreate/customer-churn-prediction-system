"""EDA: профилирование датасета и построение отчёта.

Функции не зависят от matplotlib на этапе импорта — графики строятся только
если библиотека доступна, что позволяет запускать EDA в headless-окружении.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from churn.config import Config, load_config
from churn.utils.io import save_json
from churn.utils.logger import get_logger

log = get_logger("churn.features.eda")


def basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Ключевые характеристики датасета."""
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(exclude=[np.number])
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().loc[lambda s: s > 0].to_dict(),
        "numeric_cols": list(numeric.columns),
        "categorical_cols": list(categorical.columns),
        "n_duplicates": int(df.duplicated().sum()),
    }


def churn_overview(df: pd.DataFrame, target: str = "churn") -> Dict[str, Any]:
    """Обзор целевой переменной и её связей с категориальными признаками."""
    if target not in df.columns:
        return {}
    rate = float(df[target].mean())
    overview: Dict[str, Any] = {
        "churn_rate": round(rate, 4),
        "n_churn": int(df[target].sum()),
        "n_active": int((df[target] == 0).sum()),
    }
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if col == target or df[col].nunique() > 30:
            continue
        grouped = df.groupby(col)[target].agg(["mean", "count"]).round(4)
        overview[f"churn_by_{col}"] = grouped.to_dict(orient="index")
    return overview


def numeric_correlations(df: pd.DataFrame, target: str = "churn", top: int = 15) -> Dict[str, float]:
    """Корреляции числовых признаков с таргетом (по модулю, убывание)."""
    if target not in df.columns:
        return {}
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()[target].drop(target, errors="ignore")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    return {k: round(float(v), 4) for k, v in corr.head(top).items()}


def build_eda_report(df: pd.DataFrame, cfg: Config | None = None) -> Dict[str, Any]:
    """Собрать полный отчёт EDA и сохранить в reports/eda_report.json."""
    cfg = cfg or load_config()
    report = {
        "stats": basic_stats(df),
        "churn": churn_overview(df),
        "top_correlations": numeric_correlations(df),
    }
    out = cfg.paths.resolve("report_dir") / "eda_report.json"
    save_json(report, out)
    log.info("EDA-отчёт сохранён: %s", out)
    return report


def plot_churn_distribution(df: pd.DataFrame, cfg: Config | None = None, target: str = "churn") -> List[Path]:
    """Построить базовые графики (если доступен matplotlib)."""
    cfg = cfg or load_config()
    report_dir = cfg.paths.resolve("report_dir")
    saved: List[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        log.warning("matplotlib недоступен, графики пропущены: %s", exc)
        return saved

    # распределение классов
    fig, ax = plt.subplots(figsize=(5, 4))
    df[target].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4C9F70", "#D9534F"])
    ax.set_title("Распределение оттока")
    ax.set_xlabel("churn")
    ax.set_ylabel("count")
    p = report_dir / "churn_distribution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    saved.append(p)

    # числовые признаки
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    if not numeric.empty:
        axes = numeric.hist(figsize=(12, 8), bins=30)
        fig = axes[0][0].figure
        p2 = report_dir / "numeric_hist.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        saved.append(p2)

    return saved


if __name__ == "__main__":  # pragma: no cover
    from churn.config import load_config as _lc

    _cfg = _lc()
    _df = pd.read_parquet(_cfg.paths.resolve("raw_data"))
    build_eda_report(_df, _cfg)
    plot_churn_distribution(_df, _cfg)
