from __future__ import annotations

"""Shared helpers for launching and reporting pipeline runs.

This module extracts small, well-tested utilities used by multiple entry
scripts so shared behavior remains consistent and easier to test.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def make_artifact_dirs(artifact_dir: Path) -> Dict[str, Path]:
    """Create and return common artifact subdirectories.

    Returns a dict with keys: reports, submissions, predictions, metrics, figures, models
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
    """Load train and test CSVs from `data_dir` and return DataFrames.

    The function is intentionally small so callers may adjust names if needed.
    """
    train_df = pd.read_csv(data_dir / train_name)
    test_df = pd.read_csv(data_dir / test_name)
    return train_df, test_df


def dump_json(path: Path, obj: Dict) -> None:
    """Write a dictionary to `path` as JSON with utf-8 and indenting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
