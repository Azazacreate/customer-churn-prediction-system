"""Коннекторы к PostgreSQL и ClickHouse.

Все соединения — контекстные менеджеры; при отсутствии драйвера или БД
методы поднимают понятное исключение, а вызывающий код (make_dataset)
переходит на альтернативный источник данных.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import pandas as pd

from churn.utils.logger import get_logger

log = get_logger("churn.data.db")


class PostgresConnector:
    """Обёртка над psycopg2 для аналитических запросов."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    @contextmanager
    def connect(self) -> Iterator[Any]:
        import psycopg2  # локальный импорт

        conn = psycopg2.connect(self.dsn)
        try:
            yield conn
        finally:
            conn.close()

    def query(self, sql: str, params: tuple | None = None) -> pd.DataFrame:
        with self.connect() as conn:
            log.info("PostgreSQL: выполняется запрос (%d симв.)", len(sql))
            return pd.read_sql_query(sql, conn, params=params)


class ClickHouseConnector:
    """Обёртка над clickhouse-driver."""

    def __init__(self, host: str, port: int = 9000, database: str = "analytics",
                 user: str = "default", password: str = "") -> None:
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    @contextmanager
    def connect(self) -> Iterator[Any]:
        from clickhouse_driver import Client  # локальный импорт

        client = Client(
            host=self.host, port=self.port, database=self.database,
            user=self.user, password=self.password,
        )
        try:
            yield client
        finally:
            client.disconnect()

    def query(self, sql: str) -> pd.DataFrame:
        with self.connect() as client:
            log.info("ClickHouse: выполняется запрос (%d симв.)", len(sql))
            rows, columns = client.execute(sql, with_column_types=True)
            cols = [c[0] for c in columns]
            return pd.DataFrame(rows, columns=cols)


def get_connector(cfg: dict[str, Any]):
    """Фабрика коннектора по конфигу секции ``sources.databases``."""
    if not cfg.get("enabled"):
        return None
    if cfg.get("dsn"):
        return PostgresConnector(cfg["dsn"])
    if cfg.get("host"):
        return ClickHouseConnector(
            host=cfg["host"], port=cfg.get("port", 9000),
            database=cfg.get("database", "analytics"),
        )
    return None
