from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_cv(X: pd.DataFrame, y: pd.Series, models: dict, n_splits: int, seed: int) -> pd.DataFrame:
    rows = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": float(scores.mean()),
                "cv_accuracy_std": float(scores.std()),
            }
        )

    return pd.DataFrame(rows).sort_values(by="cv_accuracy_mean", ascending=False)


def get_oof_predictions(X: pd.DataFrame, y: pd.Series, models: dict, n_splits: int, seed: int) -> tuple[np.ndarray, list[str]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model_names = list(models.keys())
    oof = np.zeros((len(X), len(model_names)), dtype=float)

    for tr_idx, va_idx in cv.split(X, y):
        x_tr, x_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y.iloc[tr_idx]

        for m_idx, m_name in enumerate(model_names):
            model = clone(models[m_name])
            model.fit(x_tr, y_tr)
            oof[va_idx, m_idx] = model.predict_proba(x_va)[:, 1]

    return oof, model_names


def tune_blend_weights(oof_pred: np.ndarray, y: pd.Series, n_trials: int) -> list[float]:
    y_arr = y.values

    def objective(trial: optuna.Trial) -> float:
        raw = [trial.suggest_float(f"w{i}", 0.01, 1.0) for i in range(oof_pred.shape[1])]
        weights = np.array(raw, dtype=float)
        weights /= weights.sum()

        blend = oof_pred @ weights
        pred = (blend >= 0.5).astype(int)
        return float(accuracy_score(y_arr, pred))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw_best = np.array([study.best_params[f"w{i}"] for i in range(oof_pred.shape[1])], dtype=float)
    raw_best /= raw_best.sum()
    return raw_best.tolist()


def weighted_proba(proba_map: dict[str, np.ndarray], weights_map: dict[str, float]) -> np.ndarray:
    arr = None
    for model_name, weight in weights_map.items():
        part = proba_map[model_name] * float(weight)
        arr = part if arr is None else arr + part
    return arr


def save_submission(proba: np.ndarray, passenger_id: pd.Series, out_dir: Path, suffix: str) -> Path:
    pred = (proba >= 0.5).astype(int)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = out_dir / f"submission_{suffix}_{ts}.csv"
    pd.DataFrame({"PassengerId": passenger_id, "Survived": pred}).to_csv(sub_path, index=False)
    return sub_path
