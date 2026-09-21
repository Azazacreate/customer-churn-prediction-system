"""Сборка сырого датасета из всех источников.

Порядок работы:
1. Пробуем REST API и БД (PostgreSQL / ClickHouse) — реальные источники.
2. Если ничего не доступно (нет сети/драйверов) — генерируем синтетику.
3. Применяем join внешних признаков (скрапинг тарифов) при наличии.

Результат сохраняется в ``paths.raw_data``.
"""
from __future__ import annotations

import pandas as pd

from churn.config import Config, load_config
from churn.data.api_client import fetch_customers
from churn.data.db_connectors import get_connector
from churn.data.scrapers import PlanParser
from churn.data.synthetic import generate_churn_data
from churn.utils.logger import get_logger

log = get_logger("churn.data.make_dataset")

TARGET = "churn"


def _from_api(cfg: Config) -> pd.DataFrame:
    api = cfg.sources.api
    if not api.endpoints:
        return pd.DataFrame()
    frames = fetch_customers(api.base_url, api.endpoints, token=api.token or None)
    base = frames.get("customers", pd.DataFrame())
    if base.empty:
        return pd.DataFrame()
    for name, frame in frames.items():
        if name != "customers" and not frame.empty and "customer_id" in frame.columns:
            base = base.merge(frame, on="customer_id", how="left")
    return base


def _from_db(cfg: Config) -> pd.DataFrame:
    from churn.data.sql_queries import CUSTOMER_BASE_SQL

    pg = get_connector(cfg.sources.databases.postgres)
    if pg is not None:
        try:
            return pg.query(CUSTOMER_BASE_SQL)
        except Exception as exc:  # pragma: no cover - требует БД
            log.warning("PostgreSQL недоступен: %s", exc)
    return pd.DataFrame()


def _from_scraping(cfg: Config) -> pd.DataFrame:
    if not cfg.sources.scraping.enabled:
        return pd.DataFrame()
    parser = PlanParser(user_agent=cfg.sources.scraping.user_agent)
    parser.fetch(cfg.sources.scraping.target_url)
    return parser.to_frame()


def _synthetic(cfg: Config) -> pd.DataFrame:
    return generate_churn_data(
        n_samples=cfg.synthetic.n_samples,
        churn_rate=cfg.synthetic.churn_rate,
        start_date=cfg.synthetic.start_date,
        end_date=cfg.synthetic.end_date,
        seed=cfg.project.seed,
    )


def build_raw_dataset(cfg: Config | None = None, force_synthetic: bool = False) -> pd.DataFrame:
    """Собрать сырой датасет и сохранить в parquet."""
    cfg = cfg or load_config()

    df = pd.DataFrame()
    if not force_synthetic:
        df = _from_api(cfg)
        if df.empty:
            df = _from_db(cfg)

    # реальные источники могут не содержать целевую переменную (label store)
    if df.empty or TARGET not in df.columns:
        if not df.empty:
            log.warning("Источник без таргета '%s' — перехожу на синтетику", TARGET)
        else:
            log.info("Внешние источники недоступны — генерирую синтетический датасет")
        df = _synthetic(cfg)

    plans = _from_scraping(cfg)
    if not plans.empty:
        log.info("Добавлены внешние признаки из скрапинга: %d строк", len(plans))
        df["market_plan"] = plans.iloc[0].get("plan")

    out_path = cfg.paths.resolve("raw_data")
    df.to_parquet(out_path, index=False)
    log.info("Сырой датасет сохранён: %s (%d x %d)", out_path, *df.shape)
    return df


if __name__ == "__main__":  # pragma: no cover
    build_raw_dataset()
