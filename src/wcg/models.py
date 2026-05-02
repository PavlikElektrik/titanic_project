"""Определения моделей и поиск гиперпараметров для WCG-пайплайна Titanic.

Содержит конструктор набора моделей и утилиту для тюнинга деревьев через Optuna.
"""

from __future__ import annotations

import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.torch_models import TorchBinaryClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def make_models(params: dict, seed: int) -> dict:
    """Создать набор кандидатных моделей, используемых в WCG-экспериментах."""
    rf = RandomForestClassifier(random_state=seed, **params["rf"])
    et = ExtraTreesClassifier(random_state=seed, **params["et"])
    cb = CatBoostClassifier(random_seed=seed, verbose=False, **params["cb"])
    xgb = None
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            **params["xgb"],
        )
    lr = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    torch_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                TorchBinaryClassifier(
                    hidden_layers=tuple(params["torch"]["hidden_layers"]),
                    dropout=float(params["torch"]["dropout"]),
                    learning_rate=float(params["torch"]["learning_rate"]),
                    weight_decay=float(params["torch"]["weight_decay"]),
                    batch_size=int(params["torch"]["batch_size"]),
                    max_epochs=int(params["torch"]["max_epochs"]),
                    patience=int(params["torch"]["patience"]),
                    val_fraction=float(params["torch"]["val_fraction"]),
                    random_state=seed,
                ),
            ),
        ]
    )

    models = {
        "rf_wcg": rf,
        "et_wcg": et,
        "cb_wcg": cb,
        "torch_wcg": torch_model,
        "logreg_wcg": lr,
    }

    if xgb is not None:
        models["xgb_wcg"] = xgb

    return models


def tune_tree_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_trials: int,
    n_splits: int,
    seed: int,
) -> dict:
    """Настроить параметрическое пространство для древовидных моделей через Optuna.

    Возвращает лучшие параметры в виде словаря.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        # Keep the search space compact so the result stays explainable and reproducible.
        if model_name == "rf":
            model = RandomForestClassifier(
                random_state=seed,
                n_estimators=trial.suggest_int("n_estimators", 400, 2200, step=200),
                max_depth=trial.suggest_int("max_depth", 4, 12),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 14),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            )
        elif model_name == "et":
            model = ExtraTreesClassifier(
                random_state=seed,
                n_estimators=trial.suggest_int("n_estimators", 400, 2400, step=200),
                max_depth=trial.suggest_int("max_depth", 4, 14),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 14),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            )
        elif model_name == "cb":
            model = CatBoostClassifier(
                random_seed=seed,
                verbose=False,
                loss_function="Logloss",
                iterations=trial.suggest_int("iterations", 500, 2200, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                depth=trial.suggest_int("depth", 4, 9),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            )
        elif model_name == "xgb":
            if XGBClassifier is None:
                raise ValueError("xgboost is not available")
            model = XGBClassifier(
                random_state=seed,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                n_estimators=trial.suggest_int("n_estimators", 300, 1800, step=150),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            )
        else:
            raise ValueError(f"Unknown model name for tuning: {model_name}")

        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
