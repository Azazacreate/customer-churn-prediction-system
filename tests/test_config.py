"""Тесты загрузки конфигурации и утилит."""
from churn.config import Config, load_config, _build
from churn.utils.io import load_json, save_json


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.project.seed == 42
    assert cfg.training.test_size > 0
    # вложенные dataclass-объекты должны быть типизированы
    assert hasattr(cfg.ab_testing, "alpha")
    assert hasattr(cfg.synthetic, "n_samples")
    assert isinstance(cfg.training.models, list)


def test_build_from_dict():
    cfg = _build(Config, {"project": {"name": "x", "seed": 7}, "training": {"cv_folds": 3}})
    assert cfg.project.name == "x"
    assert cfg.project.seed == 7
    assert cfg.training.cv_folds == 3


def test_config_as_dict_roundtrip():
    cfg = Config()
    d = cfg.as_dict()
    assert d["project"]["name"] == "churn"


def test_json_roundtrip(tmp_path):
    path = tmp_path / "x" / "data.json"
    save_json({"a": 1, "b": [1, 2, 3]}, path)
    assert load_json(path)["b"] == [1, 2, 3]
