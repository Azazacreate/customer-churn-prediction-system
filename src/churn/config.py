"""Загрузка и валидация конфигурации пайплайна.
Конфиг хранится в YAML (`config/config.yaml`) и превращается в дерево
типизированных dataclass-объектов. Любой модуль получает конфиг через
`load_config()` и обращается к нему как к атрибутам, например
``cfg.training.test_size``.
Значения из `.env` (см. `.env.example`) переопределяют часть полей —
удобно для локальных паролей и DSN, которые не хочется хранить в репозитории.
"""
from __future__ import annotations
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, get_type_hints
import yaml
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
@dataclass
class ProjectCfg:
    name: str = "churn"
    seed: int = 42
@dataclass
class PathsCfg:
    raw_data: str = "data/raw/churn_raw.parquet"
    processed_data: str = "data/processed/churn_features.parquet"
    train_data: str = "data/processed/train.parquet"
    test_data: str = "data/processed/test.parquet"
    model_dir: str = "artifacts/models"
    report_dir: str = "reports"
    def resolve(self, key: str) -> Path:
        """Абсолютный путь к ресурсу (создаёт родительскую директорию)."""
        path = PROJECT_ROOT / getattr(self, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
@dataclass
class ApiCfg:
    base_url: str = "https://api.internal.example.com/v1"
    timeout: int = 30
    endpoints: Dict[str, str] = field(default_factory=dict)
    token: str = ""
@dataclass
class ScrapingCfg:
    enabled: bool = False
    target_url: str = ""
    user_agent: str = "churn-research-bot/1.0"
@dataclass
class DatabasesCfg:
    postgres: Dict[str, Any] = field(default_factory=dict)
    clickhouse: Dict[str, Any] = field(default_factory=dict)
@dataclass
class SourcesCfg:
    api: ApiCfg = field(default_factory=ApiCfg)
    scraping: ScrapingCfg = field(default_factory=ScrapingCfg)
    databases: DatabasesCfg = field(default_factory=DatabasesCfg)
@dataclass
class SyntheticCfg:
    n_samples: int = 20000
    churn_rate: float = 0.14
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
@dataclass
class SelectionCfg:
    enabled: bool = True
    method: str = "importance"  # importance | l1 | rfe
    top_k: int = 30
@dataclass
class FeaturesCfg:
    numeric_impute: str = "median"
    categorical_impute: str = "mode"
    drop_columns: List[str] = field(default_factory=list)
    engineered: List[str] = field(default_factory=list)
    selection: SelectionCfg = field(default_factory=SelectionCfg)
@dataclass
class SearchCfg:
    enabled: bool = True
    n_iter: int = 25
    params: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
@dataclass
class TrainingCfg:
    test_size: float = 0.2
    cv_folds: int = 5
    scoring: str = "roc_auc"
    class_weight: str = "balanced"
    decision_threshold: float = 0.5
    search: SearchCfg = field(default_factory=SearchCfg)
    models: List[str] = field(default_factory=list)
@dataclass
class MlflowCfg:
    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "churn-prediction"
@dataclass
class AbTestingCfg:
    alpha: float = 0.05
    power: float = 0.80
    two_sided: bool = True
@dataclass
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    sources: SourcesCfg = field(default_factory=SourcesCfg)
    synthetic: SyntheticCfg = field(default_factory=SyntheticCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    mlflow: MlflowCfg = field(default_factory=MlflowCfg)
    ab_testing: AbTestingCfg = field(default_factory=AbTestingCfg)
    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
def _build(cls, data: Any):
    """Рекурсивно строит dataclass из вложенного словаря.
    Типы полей разрешаются через ``get_type_hints`` (важно при
    ``from __future__ import annotations``, где аннотации — строки).
    """
    if data is None:
        return cls()
    if not isinstance(data, dict):
        return data
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build(ftype, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
def _load_dotenv() -> None:
    """Подхватить .env (если установлен python-dotenv и файл существует)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except Exception:  # pragma: no cover - dotenv опционален
        pass
def _env(key: str, default: Any = None) -> Any:
    return os.environ.get(key, default)
def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
def _apply_env(cfg: Config) -> Config:
    """Переопределить часть полей значениями из окружения (.env)."""
    if _env("PROJECT_SEED") is not None:
        cfg.project.seed = int(_env("PROJECT_SEED"))
    if _env("API_BASE_URL"):
        cfg.sources.api.base_url = _env("API_BASE_URL")
    if _env("API_TIMEOUT"):
        cfg.sources.api.timeout = int(_env("API_TIMEOUT"))
    if _env("API_TOKEN"):
        cfg.sources.api.token = _env("API_TOKEN")
    pg = cfg.sources.databases.postgres
    if _env("POSTGRES_DSN"):
        pg["dsn"] = _env("POSTGRES_DSN")
    pg["enabled"] = _env_bool("POSTGRES_ENABLED", pg.get("enabled", False))
    ch = cfg.sources.databases.clickhouse
    if _env("CLICKHOUSE_HOST"):
        ch["host"] = _env("CLICKHOUSE_HOST")
    if _env("CLICKHOUSE_PORT"):
        ch["port"] = int(_env("CLICKHOUSE_PORT"))
    if _env("CLICKHOUSE_DB"):
        ch["database"] = _env("CLICKHOUSE_DB")
    ch["enabled"] = _env_bool("CLICKHOUSE_ENABLED", ch.get("enabled", False))
    cfg.mlflow.enabled = _env_bool("MLFLOW_ENABLED", cfg.mlflow.enabled)
    if _env("MLFLOW_TRACKING_URI"):
        cfg.mlflow.tracking_uri = _env("MLFLOW_TRACKING_URI")
    if _env("MLFLOW_EXPERIMENT_NAME"):
        cfg.mlflow.experiment_name = _env("MLFLOW_EXPERIMENT_NAME")
    if _env("SYNTHETIC_N_SAMPLES"):
        cfg.synthetic.n_samples = int(_env("SYNTHETIC_N_SAMPLES"))
    if _env("SYNTHETIC_CHURN_RATE"):
        cfg.synthetic.churn_rate = float(_env("SYNTHETIC_CHURN_RATE"))
    return cfg
def load_config(path: str | os.PathLike | None = None) -> Config:
    """Прочитать YAML (+ .env) и вернуть типизированный ``Config``."""
    _load_dotenv()
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return _apply_env(Config())
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _apply_env(_build(Config, raw))
