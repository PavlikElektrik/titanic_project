"""Titanic feature engineering, preprocessing, and baseline model utilities."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.torch_models import TorchBinaryClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

SEED = 42

CAT_COLS = ["Sex", "Embarked", "Deck", "Title", "Pclass"]
NUM_COLS = [
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "NameLength",
    "TicketGroupSize",
    "FarePerPerson",
    "HasCabin",
    "SibSp",
    "Parch",
    "SurnameSurvivalPrior",
    "TicketSurvivalPrior",
    "GroupSurvivalPrior",
]
FEATURE_COLS = CAT_COLS + NUM_COLS


def extract_title(name: str) -> str:
    """Extract a normalized passenger title from the name field."""
    title = pd.Series(name).str.extract(r" ([A-Za-z]+)\\.", expand=False).iloc[0]
    if pd.isna(title):
        return "Unknown"

    rare = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    if title in rare:
        return "Rare"
    if title in {"Mlle", "Ms"}:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    return title


def extract_surname(name: str) -> str:
    """Extract a lower-cased surname to support group-based features."""
    surname = pd.Series(name).str.extract(r"^([^,]+),", expand=False).iloc[0]
    return surname.strip().lower() if isinstance(surname, str) else "unknown"


def get_ticket_group_sizes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """Count how common each ticket is across train and test."""
    all_tickets = pd.concat([train_df["Ticket"], test_df["Ticket"]], axis=0)
    return all_tickets.value_counts(dropna=False)


def _fit_imputation_stats(train_part: pd.DataFrame) -> Dict[str, object]:
    """Collect robust training-only statistics used to fill missing values."""
    tmp = train_part.copy()
    tmp["Title"] = tmp["Name"].apply(extract_title)

    age_map = tmp.groupby(["Title", "Pclass"])["Age"].median().to_dict()
    age_global = tmp["Age"].median()

    fare_pclass = tmp.groupby("Pclass")["Fare"].median().to_dict()
    fare_global = tmp["Fare"].median()

    embarked_mode = tmp["Embarked"].mode(dropna=True)
    embarked_value = embarked_mode.iloc[0] if len(embarked_mode) else "S"

    return {
        "age_map": age_map,
        "age_global": float(age_global),
        "fare_pclass": fare_pclass,
        "fare_global": float(fare_global),
        "embarked_value": embarked_value,
    }


def _apply_base_features(
    df: pd.DataFrame,
    ticket_group_sizes: pd.Series,
    stats: Dict[str, object],
) -> pd.DataFrame:
    """Build the core Titanic features from raw columns and train-only statistics."""
    out = df.copy()

    out["Title"] = out["Name"].apply(extract_title)
    out["Surname"] = out["Name"].apply(extract_surname)

    out["Deck"] = out["Cabin"].fillna("U").astype(str).str[0]
    out["HasCabin"] = out["Cabin"].notna().astype(int)

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["NameLength"] = out["Name"].astype(str).str.len()

    out["TicketGroupSize"] = out["Ticket"].map(ticket_group_sizes).fillna(1).astype(int)

    def fill_age(row: pd.Series) -> float:
        key = (row["Title"], row["Pclass"])
        if pd.notna(row["Age"]):
            return float(row["Age"])
        if key in stats["age_map"] and pd.notna(stats["age_map"][key]):
            return float(stats["age_map"][key])
        return float(stats["age_global"])

    out["Age"] = out.apply(fill_age, axis=1)

    def fill_fare(row: pd.Series) -> float:
        if pd.notna(row["Fare"]):
            return float(row["Fare"])
        if row["Pclass"] in stats["fare_pclass"] and pd.notna(stats["fare_pclass"][row["Pclass"]]):
            return float(stats["fare_pclass"][row["Pclass"]])
        return float(stats["fare_global"])

    out["Fare"] = out.apply(fill_fare, axis=1)
    out["Embarked"] = out["Embarked"].fillna(stats["embarked_value"])

    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, 1)
    out["Pclass"] = out["Pclass"].astype(str)

    return out


def fit_group_priors(train_part: pd.DataFrame, y_train_part: pd.Series, alpha: float = 3.0) -> Dict[str, object]:
    """Estimate smoothed survival priors for surname and ticket groups."""
    tmp = train_part[["Surname", "Ticket"]].copy()
    tmp["Survived"] = y_train_part.values
    global_rate = float(y_train_part.mean())

    surname_stats = tmp.groupby("Surname")["Survived"].agg(["sum", "count"])
    surname_prior = (surname_stats["sum"] + alpha * global_rate) / (surname_stats["count"] + alpha)

    ticket_stats = tmp.groupby("Ticket")["Survived"].agg(["sum", "count"])
    ticket_prior = (ticket_stats["sum"] + alpha * global_rate) / (ticket_stats["count"] + alpha)

    return {
        "global_rate": global_rate,
        "surname_prior": surname_prior.to_dict(),
        "ticket_prior": ticket_prior.to_dict(),
    }


def apply_group_priors(df: pd.DataFrame, priors: Dict[str, object]) -> pd.DataFrame:
    """Attach the group survival priors to a feature frame."""
    out = df.copy()
    global_rate = float(priors["global_rate"])

    out["SurnameSurvivalPrior"] = out["Surname"].map(priors["surname_prior"])
    out["TicketSurvivalPrior"] = out["Ticket"].map(priors["ticket_prior"])

    out["GroupSurvivalPrior"] = out[["SurnameSurvivalPrior", "TicketSurvivalPrior"]].mean(axis=1)

    out["SurnameSurvivalPrior"] = out["SurnameSurvivalPrior"].fillna(global_rate)
    out["TicketSurvivalPrior"] = out["TicketSurvivalPrior"].fillna(global_rate)
    out["GroupSurvivalPrior"] = out["GroupSurvivalPrior"].fillna(global_rate)

    return out


def make_features(
    fit_df: pd.DataFrame,
    transform_df: pd.DataFrame,
    y_fit: pd.Series,
    ticket_group_sizes: pd.Series,
) -> pd.DataFrame:
    """Create model-ready Titanic features using train-fit statistics only."""
    stats = _fit_imputation_stats(fit_df)
    fit_base = _apply_base_features(fit_df, ticket_group_sizes, stats)
    transform_base = _apply_base_features(transform_df, ticket_group_sizes, stats)

    priors = fit_group_priors(fit_base, y_fit)
    transform_full = apply_group_priors(transform_base, priors)

    return transform_full[FEATURE_COLS]


def build_sklearn_pipeline(model_name: str) -> Pipeline:
    """Construct the preprocessing + estimator pipeline for a named model."""
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, CAT_COLS),
            ("num", numeric_pipe, NUM_COLS),
        ]
    )

    if model_name == "logreg":
        model = LogisticRegression(max_iter=2000, random_state=SEED)
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=8,
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=SEED,
        )
    elif model_name == "et":
        model = ExtraTreesClassifier(
            n_estimators=1200,
            max_depth=8,
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=SEED,
        )
    elif model_name == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            min_samples_leaf=20,
            random_state=SEED,
        )
    elif model_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=SEED,
        )
    elif model_name == "xgb":
        if XGBClassifier is None:
            raise ValueError("xgboost is not available")
        model = XGBClassifier(
            random_state=SEED,
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=700,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            n_jobs=-1,
        )
    elif model_name == "torch":
        model = TorchBinaryClassifier(
            hidden_layers=(64, 32),
            dropout=0.15,
            learning_rate=0.001,
            weight_decay=1e-4,
            batch_size=32,
            max_epochs=80,
            patience=10,
            val_fraction=0.2,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_models(train_df: pd.DataFrame, y: pd.Series, ticket_group_sizes: pd.Series, n_splits: int) -> pd.DataFrame:
    """Evaluate all baseline models with stratified CV and report accuracy/AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    scores = {
        "catboost": {"acc": [], "auc": []},
        "rf": {"acc": [], "auc": []},
        "et": {"acc": [], "auc": []},
        "hgb": {"acc": [], "auc": []},
        "mlp": {"acc": [], "auc": []},
        "xgb": {"acc": [], "auc": []},
        "torch": {"acc": [], "auc": []},
        "logreg": {"acc": [], "auc": []},
    }

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_df, y), start=1):
        x_train_raw = train_df.iloc[train_idx].copy()
        x_valid_raw = train_df.iloc[valid_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        x_train = make_features(x_train_raw, x_train_raw, y_train, ticket_group_sizes)
        x_valid = make_features(x_train_raw, x_valid_raw, y_train, ticket_group_sizes)

        cat_model = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.02,
            depth=6,
            l2_leaf_reg=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED + fold,
            verbose=False,
        )

        cat_model.fit(
            x_train,
            y_train,
            cat_features=CAT_COLS,
            eval_set=(x_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=150,
            verbose=False,
        )

        cat_proba = cat_model.predict_proba(x_valid)[:, 1]
        cat_pred = (cat_proba >= 0.5).astype(int)
        scores["catboost"]["acc"].append(accuracy_score(y_valid, cat_pred))
        scores["catboost"]["auc"].append(roc_auc_score(y_valid, cat_proba))

        for model_name in ["rf", "et", "hgb", "mlp", "xgb", "torch", "logreg"]:
            model = build_sklearn_pipeline(model_name)
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_valid)[:, 1]
            pred = (proba >= 0.5).astype(int)
            scores[model_name]["acc"].append(accuracy_score(y_valid, pred))
            scores[model_name]["auc"].append(roc_auc_score(y_valid, proba))

    rows = []
    for model_name, metric_values in scores.items():
        rows.append(
            {
                "model": model_name,
                "cv_accuracy_mean": float(np.mean(metric_values["acc"])),
                "cv_accuracy_std": float(np.std(metric_values["acc"])),
                "cv_auc_mean": float(np.mean(metric_values["auc"])),
                "cv_auc_std": float(np.std(metric_values["auc"])),
            }
        )

    score_df = pd.DataFrame(rows).sort_values(
        by=["cv_accuracy_mean", "cv_auc_mean"], ascending=False
    )
    return score_df


def fit_final_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y: pd.Series,
    ticket_group_sizes: pd.Series,
):
    x_train = make_features(train_df, train_df, y, ticket_group_sizes)
    x_test = make_features(train_df, test_df, y, ticket_group_sizes)

    if model_name == "catboost":
        model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED,
            verbose=False,
        )
        model.fit(x_train, y, cat_features=CAT_COLS, verbose=False)
        test_proba = model.predict_proba(x_test)[:, 1]
    else:
        model = build_sklearn_pipeline(model_name)
        model.fit(x_train, y)
        test_proba = model.predict_proba(x_test)[:, 1]

    test_pred = (test_proba >= 0.5).astype(int)
    return model, test_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Titanic models and build Kaggle submission.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--force-model", type=str, default="", choices=["", "catboost", "rf", "et", "hgb", "mlp", "xgb", "torch", "logreg"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)

    from src.common import make_artifact_dirs, load_train_test

    dirs = make_artifact_dirs(artifact_dir)
    report_dir = dirs["reports"]
    submission_dir = dirs["submissions"]

    train_df, test_df = load_train_test(data_dir)

    y = train_df["Survived"].astype(int)

    ticket_group_sizes = get_ticket_group_sizes(train_df, test_df)

    score_df = evaluate_models(train_df, y, ticket_group_sizes, n_splits=args.n_splits)
    score_df.to_csv(report_dir / "cv_scores.csv", index=False)

    if args.force_model:
        best_model_name = args.force_model
    else:
        best_model_name = score_df.iloc[0]["model"]

    final_model, test_pred = fit_final_model(best_model_name, train_df, test_df, y, ticket_group_sizes)

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": test_pred,
        }
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = submission_dir / f"submission_{best_model_name}_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    model_report = {
        "selected_model": best_model_name,
        "cv_table": score_df.to_dict(orient="records"),
        "submission_path": str(submission_path),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "feature_count": int(len(FEATURE_COLS)),
    }

    with open(report_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_report, f, ensure_ascii=False, indent=2)

    print("CV results:")
    print(score_df.to_string(index=False))
    print(f"\nSelected model: {best_model_name}")
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
