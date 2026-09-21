"""Генератор синтетического датасета оттока для демонстрации и тестов.

Данные правдоподобно имитируют B2C-подписку: чем короче контракт, выше цена,
чаще тикеты в поддержку и ниже вовлечённость — тем выше вероятность оттока.
Целевая переменная `churn` порождается логистической моделью от латентного
риска с добавлением нелинейных взаимодействий, поэтому градиентный бустинг
имеет преимущество над линейными моделями.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn.utils.logger import get_logger

log = get_logger("churn.data.synthetic")

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "None"]
REGIONS = ["North", "South", "East", "West"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_churn_data(
    n_samples: int = 20000,
    churn_rate: float = 0.14,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Сгенерировать демо-датасет оттока.

    Returns
    -------
    pandas.DataFrame
        Колонки соответствуют витрине, собираемой из БД/API, плюс таргет `churn`.
    """
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 72, size=n_samples)
    contract_type = rng.choice(CONTRACT_TYPES, size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(PAYMENT_METHODS, size=n_samples)
    internet_service = rng.choice(INTERNET_SERVICES, size=n_samples, p=[0.35, 0.5, 0.15])
    region = rng.choice(REGIONS, size=n_samples)

    # цена зависит от интернета и контракта
    base_price = 30.0
    base_price += np.where(internet_service == "Fiber optic", 25.0, 0.0)
    base_price += np.where(internet_service == "DSL", 10.0, 0.0)
    monthly_charges = base_price + rng.normal(0, 8, size=n_samples)
    monthly_charges = np.clip(monthly_charges, 15, 120).round(2)
    total_charges = (monthly_charges * tenure_months * rng.uniform(0.85, 1.0, n_samples)).round(2)

    avg_monthly_gb = np.clip(rng.gamma(2.0, 25.0, size=n_samples), 1, 500).round(1)
    num_support_tickets = rng.poisson(1.2, size=n_samples)
    num_support_tickets += np.where(contract_type == "Month-to-month", 1, 0)
    days_since_last_login = np.clip(rng.exponential(12, size=n_samples), 0, 180).round(0)
    num_logins_30d = np.clip(rng.poisson(15, size=n_samples), 0, 120)
    satisfaction_score = np.clip(rng.normal(3.6, 1.1, size=n_samples), 1, 5).round(1)
    age = rng.integers(18, 75, size=n_samples)

    # --- латентный риск оттока -------------------------------------------- #
    # линейная часть
    logit = (
        -0.8
        - 0.022 * tenure_months
        + 0.010 * monthly_charges
        + 0.28 * num_support_tickets
        + 0.02 * days_since_last_login
        - 0.03 * num_logins_30d
        - 0.45 * (satisfaction_score - 3.6)
        + np.where(contract_type == "Month-to-month", 1.1, 0.0)
        + np.where(contract_type == "Two year", -0.9, 0.0)
        + np.where(payment_method == "Electronic check", 0.35, 0.0)
        + np.where(internet_service == "Fiber optic", 0.25, 0.0)
    )

    # нелинейные взаимодействия (то, что линейные модели не ловят)
    logit += 1.4 * ((num_support_tickets >= 3) & (satisfaction_score < 3.0))
    logit += 1.0 * np.tanh((days_since_last_login - 20) / 10.0)
    logit -= 0.9 * (num_logins_30d > 25)
    logit += 0.9 * ((avg_monthly_gb < 20) & (monthly_charges > 60))
    logit -= 0.9 * ((tenure_months > 24) & (contract_type != "Month-to-month"))
    logit += 0.7 * (age < 25) * (payment_method == "Electronic check")
    logit += 0.9 * ((num_logins_30d < 5) & (days_since_last_login > 30))

    logit += np.log(churn_rate / (1 - churn_rate))  # калибровка базовой ставки
    prob = _sigmoid(logit)
    churn = rng.binomial(1, prob)

    # даты подписки (векторизованно, без устаревших numpy-timedelta)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    span = max((end - start).days - 30, 1)
    offsets = rng.integers(0, span, size=n_samples).astype("int64")
    signup_dates = (start + pd.to_timedelta(offsets, unit="D")).date

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n_samples)],
            "signup_date": signup_dates,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "region": region,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "avg_monthly_gb": avg_monthly_gb,
            "num_support_tickets": num_support_tickets,
            "days_since_last_login": days_since_last_login,
            "num_logins_30d": num_logins_30d,
            "satisfaction_score": satisfaction_score,
            "age": age,
            "churn": churn.astype(int),
        }
    )
    log.info(
        "Сгенерировано %d записей, фактический churn rate = %.3f",
        len(df), df["churn"].mean(),
    )
    return df
