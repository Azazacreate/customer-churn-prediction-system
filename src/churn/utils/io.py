"""Сериализация артефактов (joblib / json / parquet)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
def save_artifact(obj: Any, path: str | Path) -> Path:
    """Сохранить python-объект через joblib."""
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    return path
def load_artifact(path: str | Path) -> Any:
    return joblib.load(path)
def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
    return path
def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
