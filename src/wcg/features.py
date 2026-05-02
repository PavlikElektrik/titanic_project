"""Извлечение признаков для Titanic, включая WCG-групповый сигнал.

Модуль собирает признаки, специфичные для соревнования (группы по фамилиям,
размер билета и т.д.) и возвращает готовый набор признаков и служебные статистики.
"""

from __future__ import annotations

import pandas as pd

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
    """Построить эвристическую фичу групповой выживаемости по фамилии и билету.

    Для каждой группы (фамилия+fare, затем ticket) проверяются известные метки
    и проставляется 0.0, 1.0 или 0.5 (неизвестно).
    """
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
    """Преобразовать сырые столбцы в числовые/категориальные признаки, применяя WCG-сигнал.

    Если `fit_stats` не переданы, функция вычисляет статистики по текущему набору
    (median для Age/Fare, уровни категорий) и возвращает их для повторного использования.
    """
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
    # Держим Kaggle-эвристику явно видимой, чтобы её было легко отследить или отключить.
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
