from __future__ import annotations

"""Titanic training entry point that enables the WCG baseline and blending."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

from wcg.config import load_config
from wcg.features import build_group_survival_feature, preprocess_with_wcg
from wcg.models import make_models, tune_tree_model
from wcg.training import (
    evaluate_cv,
    get_oof_predictions,
    save_submission,
    tune_blend_weights,
    weighted_proba,
)


def _resolve_cfg_value(cli_value, cfg_value):
    """Prefer CLI overrides when present, otherwise fall back to the YAML config."""
    return cli_value if cli_value is not None else cfg_value


def main() -> None:
    """Run the WCG experiment end to end and save reports plus submissions."""
    parser = argparse.ArgumentParser(description="Titanic WCG training with external config files.")
    parser.add_argument("--config", type=str, default="configs/wcg/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--cv-splits", type=int, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_dir = Path(_resolve_cfg_value(args.data_dir, cfg["paths"]["data_dir"]))
    artifact_dir = Path(_resolve_cfg_value(args.artifact_dir, cfg["paths"]["artifact_dir"]))

    report_dir = artifact_dir / "reports"
    sub_dir = artifact_dir / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["experiment"]["seed"])
    cv_splits = int(_resolve_cfg_value(args.cv_splits, cfg["experiment"]["cv_splits"]))

    tune_models = bool(cfg["training"]["tune_models"]) or args.tune
    tune_model_trials = int(_resolve_cfg_value(args.n_trials, cfg["training"]["tune_model_trials"]))
    tune_blend = bool(cfg["training"]["tune_blend_weights"])
    tune_blend_trials = int(cfg["training"]["tune_blend_trials"])

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    y_train = train_df["Survived"].astype(int)

    train_group, test_group = build_group_survival_feature(train_df, test_df)
    X_train, fit_stats = preprocess_with_wcg(train_df, train_group)
    X_test, _ = preprocess_with_wcg(test_df, test_group, fit_stats=fit_stats)

    model_params = cfg["models"].copy()
    tuned_params = {}

    if tune_models:
        tuned_params["rf"] = tune_tree_model(X_train, y_train, "rf", tune_model_trials, cv_splits, seed)
        tuned_params["et"] = tune_tree_model(X_train, y_train, "et", tune_model_trials, cv_splits, seed)
        tuned_params["cb"] = tune_tree_model(X_train, y_train, "cb", tune_model_trials, cv_splits, seed)
        model_params.update(tuned_params)

    models = make_models(model_params, seed=seed)

    cv_df = evaluate_cv(X_train, y_train, models, n_splits=cv_splits, seed=seed)
    cv_df.to_csv(report_dir / "cv_scores_wcg.csv", index=False)

    oof_models = {k: v for k, v in models.items() if k in ["rf_wcg", "et_wcg", "cb_wcg"]}
    oof_pred, oof_names = get_oof_predictions(X_train, y_train, oof_models, n_splits=cv_splits, seed=seed)

    if tune_blend:
        blend_weights_list = tune_blend_weights(oof_pred, y_train, n_trials=tune_blend_trials)
        blend_weights_optuna = {
            "rf_wcg": blend_weights_list[0],
            "et_wcg": blend_weights_list[1],
            "cb_wcg": blend_weights_list[2],
        }
    else:
        blend_weights_optuna = None

    fitted = {}
    test_proba = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        test_proba[name] = model.predict_proba(X_test)[:, 1]

    generate = cfg["outputs"]["generate_submissions"]
    submission_paths = {}

    if generate.get("rf", False):
        submission_paths["rf"] = str(save_submission(test_proba["rf_wcg"], test_df["PassengerId"], sub_dir, "wcg_rf"))
    if generate.get("et", False):
        submission_paths["et"] = str(save_submission(test_proba["et_wcg"], test_df["PassengerId"], sub_dir, "wcg_et"))
    if generate.get("cb", False):
        submission_paths["cb"] = str(save_submission(test_proba["cb_wcg"], test_df["PassengerId"], sub_dir, "wcg_cb"))
    if generate.get("xgb", False) and "xgb_wcg" in test_proba:
        submission_paths["xgb"] = str(save_submission(test_proba["xgb_wcg"], test_df["PassengerId"], sub_dir, "wcg_xgb"))
    if generate.get("torch", False) and "torch_wcg" in test_proba:
        submission_paths["torch"] = str(save_submission(test_proba["torch_wcg"], test_df["PassengerId"], sub_dir, "wcg_torch"))

    if generate.get("blend_equal", False):
        equal_weights = {"rf_wcg": 1.0 / 3.0, "et_wcg": 1.0 / 3.0, "cb_wcg": 1.0 / 3.0}
        equal_blend = weighted_proba(test_proba, equal_weights)
        submission_paths["blend_equal"] = str(
            save_submission(equal_blend, test_df["PassengerId"], sub_dir, "wcg_blend_equal")
        )

    if generate.get("blend_fixed", False):
        fixed_weights = cfg["blend"]["fixed_weights"]
        fixed_blend = weighted_proba(test_proba, fixed_weights)
        submission_paths["blend_fixed"] = str(
            save_submission(fixed_blend, test_df["PassengerId"], sub_dir, "wcg_blend_fixed")
        )

    if generate.get("blend_optuna", False) and blend_weights_optuna is not None:
        optuna_blend = weighted_proba(test_proba, blend_weights_optuna)
        submission_paths["blend_optuna"] = str(
            save_submission(optuna_blend, test_df["PassengerId"], sub_dir, "wcg_blend_optuna")
        )

    train_sanity = accuracy_score(y_train, fitted["rf_wcg"].predict(X_train))

    summary = {
        "config_path": str(args.config),
        "seed": seed,
        "cv_splits": cv_splits,
        "tune_models": tune_models,
        "tune_model_trials": tune_model_trials,
        "tune_blend": tune_blend,
        "tune_blend_trials": tune_blend_trials,
        "effective_model_params": model_params,
        "tuned_params": tuned_params,
        "blend_oof_model_order": oof_names,
        "blend_optuna_weights": blend_weights_optuna,
        "blend_fixed_weights": cfg["blend"]["fixed_weights"],
        "cv_table": cv_df.to_dict(orient="records"),
        "train_accuracy_sanity": float(train_sanity),
        "submissions": submission_paths,
    }

    with open(report_dir / "wcg_tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("WCG CV results:")
    print(cv_df.to_string(index=False))
    print(f"\nTrain accuracy sanity-check: {train_sanity:.4f}")
    print(f"Config used: {args.config}")
    print("Generated submissions:")
    for key, value in submission_paths.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
