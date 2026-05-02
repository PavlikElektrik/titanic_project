"""Хелперы для загрузки и валидации конфигураций WCG-экспериментов (YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Загрузить и проверить YAML-конфигурацию для WCG-пайплайна.

    Бросает `FileNotFoundError`, если файл не найден, и `ValueError`, если содержимое не словарь.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dictionary-like YAML object")

    return cfg
