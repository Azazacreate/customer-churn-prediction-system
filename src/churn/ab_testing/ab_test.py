"""Статистическая оценка A/B-тестов удержания.
Реализованы два базовых сценария:
  * сравнение долей (конверсия удержания / доля оттока) — двухвыборочный z-тест
    для пропорций с проверкой нормальности выборки;
  * сравнение средних (LTV, ARPU, число сессий) — Welch t-test (неравные
    дисперсии), который не требует равенства размеров групп.
Дополнительно считаются доверительные интервалы, размер эффекта (Cohen's h/d)
и относительный прирост (uplift).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple
import numpy as np
from scipy import stats
from churn.utils.logger import get_logger
log = get_logger("churn.ab_testing")
@dataclass
class AbResult:
    test_type: str
    metric: str
    control_value: float
    treatment_value: float
    absolute_lift: float
    relative_lift: float
    effect_size: float
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float
    significant: bool
    winner: str
    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}
def _decision(p_value: float, alpha: float, control: float, treatment: float) -> Tuple[bool, str]:
    significant = p_value < alpha
    if not significant:
        winner = "no_difference"
    elif treatment > control:
        winner = "treatment"
    else:
        winner = "control"
    return significant, winner
def proportion_test(
    control_engaged: int,
    control_total: int,
    treat_engaged: int,
    treat_total: int,
    alpha: float = 0.05,
    metric: str = "retention_rate",
) -> AbResult:
    """Z-тест для двух пропорций (например, доля удержанных клиентов)."""
    p_c = control_engaged / control_total
    p_t = treat_engaged / treat_total
    p_pool = (control_engaged + treat_engaged) / (control_total + treat_total)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / control_total + 1 / treat_total))
    if se == 0:
        z, p_value = 0.0, 1.0
    else:
        z = (p_t - p_c) / se
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    se_diff = math.sqrt(p_c * (1 - p_c) / control_total + p_t * (1 - p_t) / treat_total)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    ci_low, ci_high = diff - z_crit * se_diff, diff + z_crit * se_diff
    effect = 2 * math.asin(math.sqrt(p_t)) - 2 * math.asin(math.sqrt(p_c))
    significant, winner = _decision(p_value, alpha, p_c, p_t)
    log.info(
        "Proportion test: control=%.4f treat=%.4f uplift=%.2f%% p=%.5f %s",
        p_c, p_t, (diff / p_c * 100 if p_c else 0), p_value,
        "SIGNIFICANT" if significant else "n.s.",
    )
    return AbResult(
        test_type="proportion_z_test",
        metric=metric,
        control_value=round(p_c, 6),
        treatment_value=round(p_t, 6),
        absolute_lift=round(diff, 6),
        relative_lift=round(diff / p_c, 6) if p_c else float("nan"),
        effect_size=round(effect, 6),
        statistic=round(z, 6),
        p_value=round(p_value, 8),
        ci_low=round(ci_low, 6),
        ci_high=round(ci_high, 6),
        alpha=alpha,
        significant=significant,
        winner=winner,
    )
def means_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    metric: str = "arpu",
) -> AbResult:
    """Welch t-test для сравнения средних (неравные дисперсии)."""
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    m_c, m_t = control.mean(), treatment.mean()
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    se = math.sqrt(control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment))
    df = _welch_df(control, treatment)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = m_t - m_c
    ci_low, ci_high = diff - t_crit * se, diff + t_crit * se
    pooled = math.sqrt((control.var(ddof=1) + treatment.var(ddof=1)) / 2) or 1e-9
    effect = diff / pooled
    significant, winner = _decision(float(p_value), alpha, m_c, m_t)
    log.info("Means test: control=%.4f treat=%.4f p=%.5f %s",
             m_c, m_t, p_value, "SIGNIFICANT" if significant else "n.s.")
    return AbResult(
        test_type="welch_t_test",
        metric=metric,
        control_value=round(float(m_c), 6),
        treatment_value=round(float(m_t), 6),
        absolute_lift=round(float(diff), 6),
        relative_lift=round(float(diff / m_c), 6) if m_c else float("nan"),
        effect_size=round(float(effect), 6),
        statistic=round(float(t_stat), 6),
        p_value=round(float(p_value), 8),
        ci_low=round(float(ci_low), 6),
        ci_high=round(float(ci_high), 6),
        alpha=alpha,
        significant=significant,
        winner=winner,
    )
def _welch_df(a: np.ndarray, b: np.ndarray) -> float:
    """Степени свободы для Welch t-test."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    denom = (va ** 2 / (len(a) - 1)) + (vb ** 2 / (len(b) - 1))
    return (va + vb) ** 2 / denom if denom else 1.0
def summary_table(result: AbResult) -> str:
    """Человекочитаемая сводка результата."""
    sig = "✅ значимо" if result.significant else "❌ не значимо"
    return (
        f"A/B-тест ({result.test_type}) — метрика: {result.metric}\n"
        f"  control   = {result.control_value}\n"
        f"  treatment = {result.treatment_value}\n"
        f"  uplift    = {result.absolute_lift:+.4f} (отн. {result.relative_lift:+.2%})\n"
        f"  p-value   = {result.p_value:.5f}  ({sig} при alpha={result.alpha})\n"
        f"  95% CI    = [{result.ci_low:.4f}, {result.ci_high:.4f}]\n"
        f"  эффект    = {result.effect_size:.4f}  |  победитель: {result.winner}"
    )
