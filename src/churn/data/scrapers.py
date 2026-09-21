"""Веб-скрапинг внешних источников: Scrapy-паук + BeautifulSoup-парсер.

Используется для сбора публичных тарифных планов/конкурентов, которые затем
join-ятся к клиентским данным как внешние признаки (например, цена рынка).
Модуль устойчив к отсутствию сети: при ошибке возвращает пустой DataFrame,
а пайплайн продолжает работать на внутренних данных.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from churn.utils.logger import get_logger

log = get_logger("churn.data.scrapers")


# --------------------------------------------------------------------------- #
#  BeautifulSoup parser
# --------------------------------------------------------------------------- #
@dataclass
class PlanParser:
    """Парсит HTML-таблицу тарифных планов в структурированный список."""

    user_agent: str = "churn-research-bot/1.0"
    timeout: int = 30
    _rows: List[Dict[str, Any]] = field(default_factory=list)

    def parse(self, html: str) -> List[Dict[str, Any]]:
        """Извлечь строки `<table>` с колонками: plan, price, data_gb."""
        from bs4 import BeautifulSoup  # локальный импорт: тяжёлая зависимость

        soup = BeautifulSoup(html, "lxml")
        rows: List[Dict[str, Any]] = []
        for table in soup.find_all("table"):
            headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if not cells:
                    continue
                record = dict(zip(headers or ["plan", "price", "data_gb"], cells))
                rows.append(record)
        self._rows = rows
        return rows

    def fetch(self, url: str) -> List[Dict[str, Any]]:
        """Скачать страницу и распарсить (мягкая деградация при ошибке)."""
        import requests

        try:
            resp = requests.get(
                url, headers={"User-Agent": self.user_agent}, timeout=self.timeout
            )
            resp.raise_for_status()
            return self.parse(resp.text)
        except Exception as exc:  # pragma: no cover - сеть
            log.warning("Не удалось скачать %s: %s", url, exc)
            return []

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


# --------------------------------------------------------------------------- #
#  Scrapy spider
# --------------------------------------------------------------------------- #
def build_plan_spider(urls: List[str] | None = None):
    """Фабрика Scrapy-паука. Возвращает класс `PlanSpider`.

    Scrapy импортируется лениво, чтобы модуль не падал, если библиотека
    не установлена в лёгком окружении.
    """
    import scrapy

    class PlanSpider(scrapy.Spider):  # type: ignore[misc]
        name = "plan_spider"
        start_urls = urls or ["https://example.com/plans"]
        custom_settings = {"LOG_LEVEL": "ERROR", "DOWNLOAD_TIMEOUT": 20}

        def parse(self, response):  # noqa: D401
            """Собирает тарифные планы со страницы."""
            for table in response.css("table"):
                headers = [
                    th.css("::text").get(default="").strip().lower()
                    for th in table.css("th")
                ]
                for row in table.css("tr"):
                    cells = [c.css("::text").get(default="").strip() for c in row.css("td")]
                    if not cells:
                        continue
                    yield dict(zip(headers or ["plan", "price", "data_gb"], cells))

    return PlanSpider


def crawl_plans(urls: List[str] | None = None) -> pd.DataFrame:
    """Запустить паука синхронно и вернуть собранные данные.

    Реальный запуск требует сети; при любой ошибке вернём пустой DataFrame,
    чтобы не ломать пайплайн.
    """
    try:  # pragma: no cover - требует сети
        from scrapy.crawler import CrawlerProcess
        from scrapy.utils.project import get_project_settings

        items: List[Dict[str, Any]] = []
        spider_cls = build_plan_spider(urls)

        class _Collector(spider_cls):  # type: ignore[misc, valid-type]
            def parse(self, response):
                for item in super().parse(response):
                    items.append(item)
                    yield item

        process = CrawlerProcess(get_project_settings())
        process.crawl(_Collector)
        process.start()
        return pd.DataFrame(items)
    except Exception as exc:
        log.warning("Scrapy crawl недоступен: %s", exc)
        return pd.DataFrame()
