"""Power analysis: расчёт MDE и необходимого размера выборки.

Позволяет спланировать A/B-тест: сколько клиентов нужно в каждой группе,
чтобы с заданной мощностью (power) поймать эффект заданной величины.
"""
from __future__ import annotations

import math
from typing import Any, Dict

from scipy import stats

from churn.utils.logger import get_logger

log = get_logger("churn.ab_testing.power")


def sample_size_proportions(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, Any]:
    """Размер выборки на группу для теста пропорций.

    Parameters
    ----------
    baseline_rate : базовая конверсия (например, 0.08 отток).
    mde_relative : минимально детектируемый эффект (относительный, 0.25 = 25%).
    """
    p1 = baseline_rate
    p2 = min(baseline_rate * (1 + mde_relative), 0.999)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_avg = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = math.ceil(numerator / (p2 - p1) ** 2)
    log.info("Нужно по %d клиентов на группу (MDE=%.0f%%)", n, mde_relative * 100)
    return {
        "baseline_rate": p1,
        "treatment_rate": round(p2, 6),
        "mde_relative": mde_relative,
        "mde_absolute": round(p2 - p1, 6),
        "alpha": alpha,
        "power": power,
        "sample_size_per_group": n,
        "total_sample_size": n * 2,
    }


def mde_proportions(
    baseline_rate: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, Any]:
    """Минимально детектируемый эффект при заданном размере выборки."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    se = math.sqrt(2 * baseline_rate * (1 - baseline_rate) / sample_size_per_group)
    mde_abs = (z_alpha + z_beta) * se
    log.info("MDE при n=%d: %.4f абсолютных (%.2f%% относительных)",
             sample_size_per_group, mde_abs, mde_abs / baseline_rate * 100)
    return {
        "baseline_rate": baseline_rate,
        "sample_size_per_group": sample_size_per_group,
        "alpha": alpha,
        "power": power,
        "mde_absolute": round(mde_abs, 6),
        "mde_relative": round(mde_abs / baseline_rate, 6),
    }


def achieved_power(
    baseline_rate: float,
    effect_relative: float,
    sample_size_per_group: int,
    alpha: float = 0.05,
) -> float:
    """Достигнутая мощность теста для данного эффекта и размера выборки."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + effect_relative)
    se = math.sqrt(p1 * (1 - p1) / sample_size_per_group + p2 * (1 - p2) / sample_size_per_group)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z = abs(p2 - p1) / se - z_alpha
    return round(float(stats.norm.cdf(z)), 4)
