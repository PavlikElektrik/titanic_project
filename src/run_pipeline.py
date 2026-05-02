from __future__ import annotations

"""Утилитарный лаунчер для поочерёдного запуска EDA и обучения моделей.

Вызывает последовательность скриптов, чтобы быстро прогнать полный пайплайн.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


COMMANDS = [
    [PYTHON, "src/run_eda.py", "--data-dir", "data", "--artifact-dir", "artifacts"],
    [PYTHON, "src/train_and_submit.py", "--data-dir", "data", "--artifact-dir", "artifacts", "--n-splits", "5"],
    [PYTHON, "src/train_kaggle_wcg.py", "--data-dir", "data", "--artifact-dir", "artifacts"],
]


def main() -> None:
    """Запустить полный Titanic-пайплайн от EDA до обучения моделей."""
    for cmd in COMMANDS:
        print("Запускаем:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

    print("\nПайплайн завершён. Смотрите reports/ и submissions/ в artifacts/.")


if __name__ == "__main__":
    main()
