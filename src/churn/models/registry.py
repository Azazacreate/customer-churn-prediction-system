"""Реестр моделей: сохранение/загрузка пайплайна и метаданных."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict

from churn.config import Config, load_config
from churn.utils.io import load_artifact, load_json, save_artifact, save_json
from churn.utils.logger import get_logger

log = get_logger("churn.models.registry")

MODEL_FILENAME = "churn_model.joblib"
META_FILENAME = "churn_model_meta.json"


def save_model(model: Any, cfg: Config | None = None, meta: Dict[str, Any] | None = None) -> Path:
    """Сохранить модель и метаданные в ``paths.model_dir``."""
    cfg = cfg or load_config()
    model_dir = cfg.paths.resolve("model_dir")
    model_path = save_artifact(model, model_dir / MODEL_FILENAME)

    meta = dict(meta or {})
    meta.setdefault("saved_at", dt.datetime.now().isoformat(timespec="seconds"))
    meta.setdefault("project", cfg.project.name)
    save_json(meta, model_dir / META_FILENAME)
    log.info("Модель сохранена: %s", model_path)
    return model_path


def load_model(cfg: Config | None = None) -> Any:
    """Загрузить обученный пайплайн."""
    cfg = cfg or load_config()
    model_path = cfg.paths.resolve("model_dir") / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}. Сначала запустите `churn train`.")
    return load_artifact(model_path)


def load_meta(cfg: Config | None = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    meta_path = cfg.paths.resolve("model_dir") / META_FILENAME
    return load_json(meta_path) if meta_path.exists() else {}
