"""Клиент внутреннего REST API (клиенты, подписки, тикеты поддержки).
Оборачивает `requests` с ретраями и пагинацией, возвращает pandas DataFrame.
Предназначен для интеграции с внутренними сервисами компании.
"""
from __future__ import annotations
import time
from typing import Any, Dict, Iterator, List
import pandas as pd
from churn.utils.logger import get_logger
log = get_logger("churn.data.api_client")
class ApiClient:
    """Простой REST-клиент с ретраями и пагинацией."""
    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        token: str | None = None,
        max_retries: int = 3,
        backoff: float = 1.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token
        self.max_retries = max_retries
        self.backoff = backoff
    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    def get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        """GET с экспоненциальным backoff-ретраем."""
        import requests
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    url, params=params, headers=self._headers(), timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # pragma: no cover - сеть
                last_exc = exc
                wait = self.backoff ** attempt
                log.warning("GET %s попытка %d/%d: %s (retry in %.1fs)",
                            url, attempt, self.max_retries, exc, wait)
                time.sleep(wait)
        log.error("REST запрос не удался: %s", url)
        raise RuntimeError(f"API request failed: {url}") from last_exc
    def paginate(
        self, path: str, page_size: int = 500, max_pages: int | None = None
    ) -> Iterator[Dict[str, Any]]:
        """Итерироваться по страницам эндпоинта (limit/offset)."""
        page = 0
        while True:
            data = self.get(path, params={"limit": page_size, "offset": page * page_size})
            items = data.get("items", data) if isinstance(data, dict) else data
            if not items:
                break
            for item in items:
                yield item
            page += 1
            if max_pages is not None and page >= max_pages:
                break
    def fetch_dataframe(self, path: str, **kwargs: Any) -> pd.DataFrame:
        """Собрать все страницы в DataFrame."""
        records: List[Dict[str, Any]] = list(self.paginate(path, **kwargs))
        log.info("REST %s -> %d записей", path, len(records))
        return pd.DataFrame(records)
def fetch_customers(base_url: str, endpoints: Dict[str, str], token: str | None = None) -> Dict[str, pd.DataFrame]:
    """Забрать клиентов, подписки и тикеты одним вызовом."""
    client = ApiClient(base_url, token=token)
    result: Dict[str, pd.DataFrame] = {}
    for name, path in endpoints.items():
        try:
            result[name] = client.fetch_dataframe(path)
        except Exception as exc:  # pragma: no cover - сеть
            log.error("Не удалось получить %s: %s", name, exc)
            result[name] = pd.DataFrame()
    return result
