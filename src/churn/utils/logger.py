"""Единый логгер для всего пайплайна."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "churn", level: int = logging.INFO) -> logging.Logger:
    """Вернуть логгер с единым форматом (конфигурируется один раз)."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        root = logging.getLogger("churn")
        root.setLevel(level)
        root.handlers = [handler]
        root.propagate = False
        _CONFIGURED = True
    return logger
