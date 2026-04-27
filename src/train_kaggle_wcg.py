from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Cabin",
    "FamilySize",
    "IsAlone",
    "NameLength",
    "TicketGroupSize",
    "FarePerPerson",
    "Title",
    "GroupSurvival",
]


def build_group_survival_feature(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    all_data = pd.concat([train_df.copy(), test_df.copy()], sort=False).reset_index(drop=True)
    all_data["Surname"] = all_data["Name"].str.extract(r"^([^,]+),", expand=False).fillna("Unknown")
    all_data["GroupSurvival"] = 0.5

    for _, grp_df in all_data.groupby(["Surname", "Fare"], dropna=False):
        if len(grp_df) <= 1:
            continue
        idx = grp_df.index
        surv = grp_df["Survived"]
        if surv.notna().any():
            for i in idx:
                others = surv.drop(i, errors="ignore")
                if others.notna().any():
                    if (others == 1).any():
                        all_data.loc[i, "GroupSurvival"] = 1.0
                    elif (others == 0).any():
                        all_data.loc[i, "GroupSurvival"] = 0.0

    for _, grp_df in all_data.groupby("Ticket", dropna=False):
        if len(grp_df) <= 1:
            continue
        idx = grp_df.index
        surv = grp_df["Survived"]
        if surv.notna().any():
            for i in idx:
                if all_data.loc[i, "GroupSurvival"] != 0.5:
                    continue
                others = surv.drop(i, errors="ignore")
                if others.notna().any():
                    if (others == 1).any():
                        all_data.loc[i, "GroupSurvival"] = 1.0
                    elif (others == 0).any():
                        all_data.loc[i, "GroupSurvival"] = 0.0

    train_group = all_data.loc[: len(train_df) - 1, "GroupSurvival"].reset_index(drop=True)
    test_group = all_data.loc[len(train_df) :, "GroupSurvival"].reset_index(drop=True)
    return train_group, test_group


def preprocess_with_wcg(df: pd.DataFrame, group_survival: pd.Series, fit_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    out["Title"] = out["Name"].str.extract(r" ([A-Za-z]+)\\.", expand=False)
    title_map = {
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
    }
    out["Title"] = out["Title"].map(lambda x: title_map.get(x, x))

    out["Sex"] = out["Sex"].map({"male": 0, "female": 1}).astype(float)
    out["Cabin"] = out["Cabin"].fillna("U").astype(str).str[0]
    out["Embarked"] = out["Embarked"].fillna("S")

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["NameLength"] = out["Name"].astype(str).str.len()
    out["TicketGroupSize"] = out.groupby("Ticket")["Ticket"].transform("count")
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, 1)
    out["GroupSurvival"] = group_survival.values

    if fit_stats is None:
        stats = {
            "age_map": out.groupby(["Title", "Pclass"])["Age"].median().to_dict(),
            "age_global": float(out["Age"].median()),
            "fare_pclass": out.groupby("Pclass")["Fare"].median().to_dict(),
            "fare_global": float(out["Fare"].median()),
            "cabin_levels": sorted(out["Cabin"].unique().tolist()),
            "title_levels": sorted(out["Title"].dropna().unique().tolist()),
            "embarked_levels": sorted(out["Embarked"].dropna().unique().tolist()),
        }
    else:
        stats = fit_stats

    def fill_age(row: pd.Series) -> float:
        if pd.notna(row["Age"]):
            return float(row["Age"])
        key = (row["Title"], row["Pclass"])
        if key in stats["age_map"] and pd.notna(stats["age_map"][key]):
            return float(stats["age_map"][key])
        return float(stats["age_global"])

    def fill_fare(row: pd.Series) -> float:
        if pd.notna(row["Fare"]):
            return float(row["Fare"])
        if row["Pclass"] in stats["fare_pclass"] and pd.notna(stats["fare_pclass"][row["Pclass"]]):
            return float(stats["fare_pclass"][row["Pclass"]])
        return float(stats["fare_global"])

    out["Age"] = out.apply(fill_age, axis=1)
    out["Fare"] = out.apply(fill_fare, axis=1)
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, 1)

    for col in ["Cabin", "Title", "Embarked"]:
        levels = stats[f"{col.lower()}_levels"]
        out[col] = pd.Categorical(out[col], categories=levels)
        out[col] = out[col].cat.codes.replace(-1, 0)

    return out[FEATURES].copy(), stats


def make_models(params: dict | None = None) -> dict:
    params = params or {}
    rf_params = params.get(
        "rf",
        {
            "n_estimators": 1200,
            "max_depth": 7,
            "min_samples_leaf": 2,
            "min_samples_split": 4,
            "criterion": "entropy",
        },
    )
    et_params = params.get(
        "et",
        {
            "n_estimators": 1600,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "min_samples_split": 4,
        },
    )
    cb_params = params.get(
        "cb",
        {
            "iterations": 1500,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 5,
            "loss_function": "Logloss",
        },
    )

    rf = RandomForestClassifier(random_state=SEED, **rf_params)
    et = ExtraTreesClassifier(random_state=SEED, **et_params)
    cb = CatBoostClassifier(random_seed=SEED, verbose=False, **cb_params)
    lr = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]
    )

    return {
        "rf_wcg": rf,
        "et_wcg": et,
        "cb_wcg": cb,
        "logreg_wcg": lr,
    }


def evaluate_cv(X: pd.DataFrame, y: pd.Series, models: dict, n_splits: int) -> pd.DataFrame:
    rows = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

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


def tune_tree_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_trials: int,
    n_splits: int,
) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        if model_name == "rf":
            model = RandomForestClassifier(
                random_state=SEED,
                n_estimators=trial.suggest_int("n_estimators", 400, 2200, step=200),
                max_depth=trial.suggest_int("max_depth", 4, 12),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 14),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            )
        elif model_name == "et":
            model = ExtraTreesClassifier(
                random_state=SEED,
                n_estimators=trial.suggest_int("n_estimators", 400, 2400, step=200),
                max_depth=trial.suggest_int("max_depth", 4, 14),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 14),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            )
        elif model_name == "cb":
            model = CatBoostClassifier(
                random_seed=SEED,
                verbose=False,
                loss_function="Logloss",
                iterations=trial.suggest_int("iterations", 500, 2200, step=100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                depth=trial.suggest_int("depth", 4, 9),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            )
        else:
            raise ValueError(f"Unknown model name for tuning: {model_name}")

        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def get_oof_predictions(X: pd.DataFrame, y: pd.Series, models: dict, n_splits: int) -> tuple[np.ndarray, list[str]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    model_names = list(models.keys())
    oof = np.zeros((len(X), len(model_names)), dtype=float)

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
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


def save_submission(proba: np.ndarray, passenger_id: pd.Series, out_dir: Path, suffix: str) -> Path:
    pred = (proba >= 0.5).astype(int)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = out_dir / f"submission_{suffix}_{ts}.csv"
    pd.DataFrame({"PassengerId": passenger_id, "Survived": pred}).to_csv(sub_path, index=False)
    return sub_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Titanic Kaggle-oriented WCG training and submission.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    report_dir = artifact_dir / "reports"
    sub_dir = artifact_dir / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    y_train = train_df["Survived"].astype(int)

    train_group, test_group = build_group_survival_feature(train_df, test_df)
    X_train, fit_stats = preprocess_with_wcg(train_df, train_group)
    X_test, _ = preprocess_with_wcg(test_df, test_group, fit_stats=fit_stats)

    tuned_params = {}
    if args.tune:
        per_model_trials = max(6, args.n_trials // 3)
        tuned_params["rf"] = tune_tree_model(X_train, y_train, "rf", per_model_trials, args.cv_splits)
        tuned_params["et"] = tune_tree_model(X_train, y_train, "et", per_model_trials, args.cv_splits)
        tuned_params["cb"] = tune_tree_model(X_train, y_train, "cb", per_model_trials, args.cv_splits)

    models = make_models(tuned_params)
    cv_df = evaluate_cv(X_train, y_train, models, n_splits=args.cv_splits)
    cv_df.to_csv(report_dir / "cv_scores_wcg.csv", index=False)

    oof_models = {k: v for k, v in models.items() if k in ["rf_wcg", "et_wcg", "cb_wcg"]}
    oof_pred, oof_names = get_oof_predictions(X_train, y_train, oof_models, n_splits=args.cv_splits)
    blend_weights = tune_blend_weights(oof_pred, y_train, n_trials=max(10, args.n_trials))

    fitted = {}
    test_proba = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        test_proba[name] = model.predict_proba(X_test)[:, 1]

    submission_paths = {}
    submission_paths["rf"] = str(save_submission(test_proba["rf_wcg"], test_df["PassengerId"], sub_dir, "wcg_rf"))
    submission_paths["et"] = str(save_submission(test_proba["et_wcg"], test_df["PassengerId"], sub_dir, "wcg_et"))
    submission_paths["cb"] = str(save_submission(test_proba["cb_wcg"], test_df["PassengerId"], sub_dir, "wcg_cb"))

    equal_blend = (test_proba["rf_wcg"] + test_proba["et_wcg"] + test_proba["cb_wcg"]) / 3.0
    submission_paths["blend_equal"] = str(save_submission(equal_blend, test_df["PassengerId"], sub_dir, "wcg_blend_equal"))

    tuned_blend = (
        blend_weights[0] * test_proba["rf_wcg"]
        + blend_weights[1] * test_proba["et_wcg"]
        + blend_weights[2] * test_proba["cb_wcg"]
    )
    submission_paths["blend_optuna"] = str(
        save_submission(tuned_blend, test_df["PassengerId"], sub_dir, "wcg_blend_optuna")
    )

    train_sanity = accuracy_score(y_train, fitted["rf_wcg"].predict(X_train))

    summary = {
        "tuning_enabled": args.tune,
        "n_trials": args.n_trials,
        "cv_splits": args.cv_splits,
        "tuned_params": tuned_params,
        "blend_oof_model_order": oof_names,
        "blend_optuna_weights": blend_weights,
        "cv_table": cv_df.to_dict(orient="records"),
        "train_accuracy_sanity": float(train_sanity),
        "submissions": submission_paths,
    }

    with open(report_dir / "wcg_tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("WCG CV results:")
    print(cv_df.to_string(index=False))
    print(f"\nTrain accuracy sanity-check: {train_sanity:.4f}")
    print("Generated submissions:")
    for key, value in submission_paths.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
