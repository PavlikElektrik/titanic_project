from __future__ import annotations

"""Простой модуль EDA: строит графики и собирает текстовый отчёт по данным Titanic.

Функции сохраняют набор png-файлов и markdown-отчёт для быстрого обзора данных.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def extract_title(name: str) -> str:
    """Извлечь нормализованный титул пассажира из поля `Name`.

    Возвращает общие варианты (`Miss`, `Mrs`, `Rare`, `Unknown`) для удобства анализа.
    """
    title = pd.Series(name).str.extract(r" ([A-Za-z]+)\.", expand=False).iloc[0]
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


def save_missing_plot(df: pd.DataFrame, out_path: Path) -> None:
    """Сохранить гистограмму доли пропусков по столбцам в `out_path`.

    Если пропусков нет — записать картинку с текстом "No missing values".
    """
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    plt.figure(figsize=(10, 5))
    if len(missing):
        sns.barplot(x=missing.index, y=missing.values, color="#2a9d8f")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Missing ratio")
        plt.title("Missing Values in Train")
    else:
        plt.text(0.5, 0.5, "No missing values", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_survival_plots(train_df: pd.DataFrame, plot_dir: Path) -> None:
    """Сгенерировать и сохранить несколько графиков выживаемости.

    Создаются графики по `Sex`, `Pclass` и `Title`.
    """
    train_plot = train_df.copy()
    train_plot["Title"] = train_plot["Name"].apply(extract_title)

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=train_plot,
        x="Sex",
        y="Survived",
        hue="Sex",
        estimator="mean",
        errorbar=None,
        palette="Set2",
        legend=False,
    )
    plt.title("Survival Rate by Sex")
    plt.tight_layout()
    plt.savefig(plot_dir / "survival_by_sex.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=train_plot,
        x="Pclass",
        y="Survived",
        hue="Pclass",
        estimator="mean",
        errorbar=None,
        palette="Set1",
        legend=False,
    )
    plt.title("Survival Rate by Pclass")
    plt.tight_layout()
    plt.savefig(plot_dir / "survival_by_pclass.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    title_order = (
        train_plot.groupby("Title")["Survived"].mean().sort_values(ascending=False).index.tolist()
    )
    sns.barplot(data=train_plot, x="Title", y="Survived", estimator="mean", errorbar=None, order=title_order, color="#e76f51")
    plt.xticks(rotation=35, ha="right")
    plt.title("Survival Rate by Title")
    plt.tight_layout()
    plt.savefig(plot_dir / "survival_by_title.png", dpi=150)
    plt.close()


def save_distribution_plots(train_df: pd.DataFrame, plot_dir: Path) -> None:
    """Сохранить распределения `Age` и `Fare` в виде png.
    """
    plt.figure(figsize=(7, 4))
    sns.histplot(train_df["Age"], bins=30, kde=True, color="#457b9d")
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "age_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(train_df["Fare"], bins=40, kde=True, color="#f4a261")
    plt.title("Fare Distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "fare_distribution.png", dpi=150)
    plt.close()


def save_corr_plot(train_df: pd.DataFrame, plot_dir: Path) -> None:
    """Построить и сохранить тепловую карту корреляций для числовых столбцов.
    """
    numeric_cols = [
        "Survived",
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
    ]
    corr = train_df[numeric_cols].corr(numeric_only=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Numeric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plot_dir / "numeric_corr.png", dpi=150)
    plt.close()


def build_report(train_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path) -> None:
    """Собрать простой markdown-отчёт по данным и сохранить в `out_path`.
    """
    missing_df = train_df.isna().sum().to_frame("missing_count")
    missing_df["missing_ratio"] = (missing_df["missing_count"] / len(train_df)).round(4)

    surv_rate = train_df["Survived"].mean()
    sex_rates = train_df.groupby("Sex")["Survived"].mean().round(4).to_dict()
    pclass_rates = train_df.groupby("Pclass")["Survived"].mean().round(4).to_dict()

    text = f"""# Titanic EDA Report

## 1) Dataset snapshot

- Train shape: {train_df.shape}
- Test shape: {test_df.shape}
- Target mean (Survived): {surv_rate:.4f}

## 2) Missing values in train

| Column | Missing Count | Missing Ratio |
|---|---:|---:|
"""

    for col, row in missing_df.sort_values("missing_ratio", ascending=False).iterrows():
        text += f"| {col} | {int(row['missing_count'])} | {row['missing_ratio']:.4f} |\n"

    text += "\n## 3) Survival rates\n\n"
    text += "By Sex:\n"
    for key, value in sex_rates.items():
        text += f"- {key}: {value:.4f}\n"

    text += "\nBy Pclass:\n"
    for key, value in pclass_rates.items():
        text += f"- class {key}: {value:.4f}\n"

    text += "\n## 4) Plot files\n\n"
    text += "- missing_values.png\n"
    text += "- survival_by_sex.png\n"
    text += "- survival_by_pclass.png\n"
    text += "- survival_by_title.png\n"
    text += "- age_distribution.png\n"
    text += "- fare_distribution.png\n"
    text += "- numeric_corr.png\n"

    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить EDA и сохранить отчёт/графики.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)

    report_dir = artifact_dir / "reports"
    plot_dir = artifact_dir / "eda"

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    save_missing_plot(train_df, plot_dir / "missing_values.png")
    save_survival_plots(train_df, plot_dir)
    save_distribution_plots(train_df, plot_dir)
    save_corr_plot(train_df, plot_dir)
    build_report(train_df, test_df, report_dir / "eda_report.md")

    print(f"EDA plots saved to: {plot_dir}")
    print(f"EDA report saved to: {report_dir / 'eda_report.md'}")


if __name__ == "__main__":
    main()
