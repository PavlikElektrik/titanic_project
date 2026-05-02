from __future__ import annotations

"""Единая CLI-точка входа для Titanic-пайплайна.

По умолчанию запускает WCG-режим с YAML-конфигом, но также умеет запускать
базовый пайплайн и полный прогон через один интерфейс.
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    """Запустить команду из корня проекта и упасть при ошибке."""
    print("Запускаем:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Titanic entrypoint.")
    parser.add_argument("--config", type=str, default="configs/wcg/default.yaml")
    parser.add_argument("--mode", choices=["wcg", "baseline", "all"], default="wcg")
    parser.add_argument("--run-all", action="store_true", help="Alias for --mode all")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--force-model",
        type=str,
        default="",
        choices=["", "catboost", "rf", "et", "hgb", "mlp", "xgb", "torch", "logreg"],
    )
    parser.add_argument("--cv-splits", type=int, default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    mode = "all" if args.run_all else args.mode

    if mode == "all":
        _run([sys.executable, "-m", "src.run_eda", "--data-dir", args.data_dir, "--artifact-dir", args.artifact_dir])
        _run(
            [
                sys.executable,
                "-m",
                "src.train_and_submit",
                "--data-dir",
                args.data_dir,
                "--artifact-dir",
                args.artifact_dir,
                "--n-splits",
                str(args.n_splits),
            ]
            + (["--force-model", args.force_model] if args.force_model else [])
        )
        _run(
            [
                sys.executable,
                "-m",
                "src.train_kaggle_wcg",
                "--config",
                args.config,
                "--data-dir",
                args.data_dir,
                "--artifact-dir",
                args.artifact_dir,
            ]
            + (["--cv-splits", str(args.cv_splits)] if args.cv_splits is not None else [])
            + (["--tune"] if args.tune else [])
            + (["--n-trials", str(args.n_trials)] if args.n_trials is not None else [])
        )
        return

    if mode == "baseline":
        _run(
            [
                sys.executable,
                "-m",
                "src.train_and_submit",
                "--data-dir",
                args.data_dir,
                "--artifact-dir",
                args.artifact_dir,
                "--n-splits",
                str(args.n_splits),
            ]
            + (["--force-model", args.force_model] if args.force_model else [])
        )
        return

    _run(
        [
            sys.executable,
            "-m",
            "src.train_kaggle_wcg",
            "--config",
            args.config,
            "--data-dir",
            args.data_dir,
            "--artifact-dir",
            args.artifact_dir,
        ]
        + (["--cv-splits", str(args.cv_splits)] if args.cv_splits is not None else [])
        + (["--tune"] if args.tune else [])
        + (["--n-trials", str(args.n_trials)] if args.n_trials is not None else [])
    )


if __name__ == "__main__":
    main()
