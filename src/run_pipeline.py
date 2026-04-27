from __future__ import annotations

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
    for cmd in COMMANDS:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

    print("\nPipeline finished. Check artifacts/ for reports and submissions.")


if __name__ == "__main__":
    main()
