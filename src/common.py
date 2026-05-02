from __future__ import annotations

"""Общие вспомогательные утилиты для запуска и отчётности пайплайна.

Модуль содержит небольшие надёжные функции, используемые в нескольких
скриптах запуска, чтобы поведение было согласованным и удобным для тестирования.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def make_artifact_dirs(artifact_dir: Path) -> Dict[str, Path]:
    """Создать и вернуть стандартные подкаталоги для артефактов.

    Возвращает словарь с ключами: reports, submissions, predictions,
    metrics, figures, models.
    """
    report_dir = artifact_dir / "reports"
    sub_dir = artifact_dir / "submissions"
    pred_dir = artifact_dir / "predictions"
    metrics_dir = artifact_dir / "metrics"
    figures_dir = artifact_dir / "figures"
    models_dir = artifact_dir / "models"

    for p in (report_dir, sub_dir, pred_dir, metrics_dir, figures_dir, models_dir):
        p.mkdir(parents=True, exist_ok=True)

    return {
        "reports": report_dir,
        "submissions": sub_dir,
        "predictions": pred_dir,
        "metrics": metrics_dir,
        "figures": figures_dir,
        "models": models_dir,
    }


def load_train_test(data_dir: Path, train_name: str = "train.csv", test_name: str = "test.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузить CSV-файлы train/test из `data_dir` и вернуть DataFrame.

    Функция небольшая и оставлена простой, чтобы вызывать её удобно из скриптов.
    """
    train_df = pd.read_csv(data_dir / train_name)
    test_df = pd.read_csv(data_dir / test_name)
    return train_df, test_df


def dump_json(path: Path, obj: Dict) -> None:
    """Записать словарь в файл `path` в формате JSON (utf-8, отступы)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
