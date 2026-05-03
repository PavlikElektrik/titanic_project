"""Feature engineering, препроцессинг и базовые модели для соревнования Titanic.

Модуль собирает все шаги обучения в одном месте:
- извлечение признаков из сырых столбцов (Title, Surname, FamilySize и т.д.)
- сглаженные групповые приоры выживаемости по фамилии и билету
- сборка sklearn-пайплайнов для всех кандидатных моделей
- честная стратифицированная CV с пересчётом признаков на каждом фолде
- обучение финальной модели на всём train и генерация Kaggle-сабмишена

Главное архитектурное решение: статистики для импутации и групповые приоры
вычисляются ИСКЛЮЧИТЕЛЬНО на train-части каждого фолда. Это гарантирует, что
CV-оценка не завышена утечкой таргета из валидации.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

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

# XGBoost оборачиваем в try/except — модель опциональная, чтобы пайплайн
# запускался и в окружении без неё (тогда xgb просто не попадёт в реестр).
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

SEED = 42

# Категориальные и числовые колонки, которые попадают в модель после
# feature engineering. Порядок важен — он сохраняется в FEATURE_COLS,
# который потом фиксирует схему DataFrame, передаваемого в модели.
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
    """Извлечь нормализованный титул пассажира из поля Name.

    Имена в Titanic имеют формат "Фамилия, Титул. Имя", например
    "Braund, Mr. Owen Harris". Регулярка ловит слово между пробелом и точкой —
    это и есть титул (Mr, Miss, Master, Dr и т.д.).

    Редкие титулы (Lady, Capt, Col, Don, ...) сворачиваются в "Rare", чтобы
    у модели не было категорий с одним-двумя примерами. Локальные варианты
    (Mlle → Miss, Mme → Mrs, Ms → Miss) приводятся к каноническим формам,
    потому что это одно и то же по смыслу, просто на разных языках.
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


def extract_surname(name: str) -> str:
    """Извлечь фамилию (в нижнем регистре) для группировки по семьям.

    Формат имени "Фамилия, Титул. Имя" — берём всё до первой запятой.
    Нижний регистр нужен, чтобы "Smith" и "smith" попали в одну группу.
    """
    surname = pd.Series(name).str.extract(r"^([^,]+),", expand=False).iloc[0]
    return surname.strip().lower() if isinstance(surname, str) else "unknown"


def get_ticket_group_sizes(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """Посчитать, сколько раз каждый билет встречается в объединённом train+test.

    Размер группы по билету — слабая, но безопасная фича: это просто факт о
    данных (сколько людей купили билет вместе), не зависящий от таргета,
    поэтому считать его на полном датасете не утечка.
    """
    all_tickets = pd.concat([train_df["Ticket"], test_df["Ticket"]], axis=0)
    return all_tickets.value_counts(dropna=False)


def _fit_imputation_stats(train_part: pd.DataFrame) -> dict:
    """Собрать статистики для заполнения пропусков ТОЛЬКО по train-части.

    Возвращает словарь со статистиками, которые потом одинаково применяются
    и к train, и к valid/test. Идея в том, чтобы валидация не "подсматривала"
    собственные значения через медиану.

    Что собираем:
    - age_map: медиана Age по парам (Title, Pclass) — Age сильно зависит от
      сочетания титула и класса (например, Master в 3 классе ≈ ребёнок),
      поэтому такая импутация точнее, чем глобальная медиана.
    - age_global: запасной вариант, если пары (Title, Pclass) нет в train.
    - fare_pclass: медиана Fare по Pclass — цена билета сильно коррелирует
      с классом каюты.
    - fare_global / embarked_value: глобальные fallback'и.
    """
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
    stats: dict,
) -> pd.DataFrame:
    """Построить базовые признаки Titanic из сырых столбцов.

    Принимает на вход уже посчитанные `stats` (из `_fit_imputation_stats`),
    чтобы train и valid/test обрабатывались одинаковыми правилами и не было
    утечки. Сама функция чистая — никаких глобальных статистик не вычисляет.

    Что добавляется:
    - Title, Surname — извлечены из Name
    - Deck — первая буква Cabin (часто это палуба, "U" для пропусков)
    - HasCabin — есть ли вообще запись о каюте (часто скоррелировано с классом)
    - FamilySize, IsAlone — размер семьи на борту и флаг одиночки
    - NameLength — длина имени (на удивление неплохой прокси для социального
      статуса: у богатых пассажиров имена обычно длиннее)
    - TicketGroupSize — сколько людей разделили один билет (часто это группы
      друзей или больших семей, не пойманные через SibSp/Parch)
    - FarePerPerson — Fare поделить на размер группы
    """
    out = df.copy()

    out["Title"] = out["Name"].apply(extract_title)
    out["Surname"] = out["Name"].apply(extract_surname)

    out["Deck"] = out["Cabin"].fillna("U").astype(str).str[0]
    out["HasCabin"] = out["Cabin"].notna().astype(int)

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["NameLength"] = out["Name"].astype(str).str.len()

    out["TicketGroupSize"] = out["Ticket"].map(ticket_group_sizes).fillna(1).astype(int)

    # Распаковываем stats в локальные переменные. Это и читается чище, и
    # снимает претензии type-checker'а к индексации внутри вложенных функций.
    age_map = stats["age_map"]
    age_global = stats["age_global"]
    fare_pclass = stats["fare_pclass"]
    fare_global = stats["fare_global"]
    embarked_value = stats["embarked_value"]

    def fill_age(row: pd.Series) -> float:
        # Стратегия: медиана по (Title, Pclass) → глобальная медиана.
        # Делаем построчно, потому что ключ зависит сразу от двух колонок.
        key = (row["Title"], row["Pclass"])
        if pd.notna(row["Age"]):
            return float(row["Age"])
        if key in age_map and pd.notna(age_map[key]):
            return float(age_map[key])
        return float(age_global)

    out["Age"] = out.apply(fill_age, axis=1)

    def fill_fare(row: pd.Series) -> float:
        # Та же логика: медиана по Pclass → глобальная медиана.
        if pd.notna(row["Fare"]):
            return float(row["Fare"])
        if row["Pclass"] in fare_pclass and pd.notna(fare_pclass[row["Pclass"]]):
            return float(fare_pclass[row["Pclass"]])
        return float(fare_global)

    out["Fare"] = out.apply(fill_fare, axis=1)
    out["Embarked"] = out["Embarked"].fillna(embarked_value)

    # FarePerPerson пересчитываем после импутации Fare и FamilySize,
    # чтобы он отражал заполненные значения.
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, 1)

    # Pclass переводим в строку, чтобы он шёл через категориальный пайплайн
    # (one-hot), а не через числовой scaler. С точки зрения смысла это
    # категория из 3 уровней, а не упорядоченное число.
    out["Pclass"] = out["Pclass"].astype(str)

    return out


def fit_group_priors(train_part: pd.DataFrame, y_train_part: pd.Series, alpha: float = 3.0) -> dict:
    """Посчитать сглаженные приоры выживаемости по фамилии и билету.

    Это сердцевина group-features: для каждой фамилии и каждого билета мы
    знаем, какая доля её представителей в train выжила. Сырая доля очень
    шумная для маленьких групп (если в группе 1 человек и он выжил, prior
    будет 1.0 — почти гарантированная утечка через переобучение).

    Поэтому используем сглаживание по формуле байесовского shrinkage:

        prior = (sum + alpha * global_rate) / (count + alpha)

    Это эквивалентно тому, что мы добавляем к каждой группе alpha "виртуальных"
    наблюдений с глобальной долей выживших. Эффект:
    - для большой группы (count >> alpha) prior ≈ настоящая доля выживших
    - для маленькой (count ≤ alpha) prior смещается к глобальному среднему
    - alpha = 3 — практичный компромисс: одиночные группы почти не вносят
      сигнала, а группы из 5+ человек уже доверяем

    ВАЖНО: считаем ТОЛЬКО на train-фолде. Если посчитать на полном train+valid,
    получим target leakage — модель будет видеть метки валидации через приор.
    """
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


def apply_group_priors(df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """Прикрепить заранее посчитанные приоры к датафрейму с признаками.

    Если фамилии или билета не было в train (например, новый пассажир в test),
    map вернёт NaN — заполняем глобальной долей выживших, чтобы модель не
    спотыкалась о пропуски и не получала "необычно низкую" оценку для
    пассажиров из неизвестных групп.

    GroupSurvivalPrior — простое усреднение двух приоров. Это эвристика:
    билет и фамилия часто пересекаются (одна семья на одном билете), но
    не всегда, поэтому средний сигнал стабильнее каждого по отдельности.
    """
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
    """Собрать готовые к модели признаки, используя статистики только из fit_df.

    Это публичный API модуля для feature engineering. Разделение fit_df vs
    transform_df — ключевое: на train-фолде статистики и приоры считаются
    по fit_df (это train), а потом применяются к transform_df (это либо тот
    же train, либо valid, либо test).

    Типичные сценарии:
    - На фолде CV: fit_df = train-фолд, transform_df = valid-фолд → статистики
      по train, применение к valid. Никакой утечки.
    - На финальном обучении: fit_df = весь train, transform_df = test →
      статистики по всему train, применение к test.
    - Чтобы получить признаки самого train-фолда: fit_df = transform_df = train.
    """
    stats = _fit_imputation_stats(fit_df)
    fit_base = _apply_base_features(fit_df, ticket_group_sizes, stats)
    transform_base = _apply_base_features(transform_df, ticket_group_sizes, stats)

    # Приоры считаются на fit-части после извлечения базовых признаков
    # (потому что нам нужны Surname и Ticket в нормализованном виде).
    priors = fit_group_priors(fit_base, y_fit)
    transform_full = apply_group_priors(transform_base, priors)

    return transform_full[FEATURE_COLS]


def build_sklearn_pipeline(model_name: str) -> Pipeline:
    """Собрать sklearn-пайплайн (препроцессор + модель) по имени модели.

    Все sklearn-совместимые модели обёрнуты в одинаковый ColumnTransformer,
    чтобы препроцессинг был идентичен и сравнение моделей было честным:
    - категориальные: most_frequent imputer + OneHotEncoder с handle_unknown="ignore"
      (ignore нужен, чтобы новая категория в test не валила пайплайн)
    - числовые: median imputer + StandardScaler (scaler нужен в первую очередь
      для линейных моделей и MLP, деревьям он не мешает).

    Поддерживаются: logreg, rf, et, hgb, mlp, xgb, torch.
    CatBoost обрабатывается отдельно в evaluate_models, потому что он умеет
    сам переваривать категориальные колонки и не нуждается в OHE.
    """
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
    """Оценить все кандидатные модели через стратифицированную CV.

    Ключевое отличие от наивной CV: на КАЖДОМ фолде мы заново вызываем
    `make_features` отдельно для train- и valid-частей. Это гарантирует, что
    статистики импутации и групповые приоры считаются только по train-фолду,
    не подсматривая в valid. Без этого CV-метрика была бы завышена (и сильно).

    CatBoost обучается отдельным блоком, потому что:
    - умеет работать с категориальными колонками напрямую (через cat_features),
      без OneHotEncoder, что обычно даёт лучше скор и быстрее
    - использует eval_set с early stopping, поэтому ему нужен явный valid

    Для остальных моделей единая логика: построить pipeline, fit, predict_proba,
    посчитать accuracy и AUC.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # Заранее регистрируем все модели в словаре, чтобы сразу было видно,
    # какие кандидаты сравниваются. Для каждой копим acc и auc по фолдам.
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

        # Признаки для train: fit_df = transform_df = x_train_raw.
        # Признаки для valid: fit_df = x_train_raw, transform_df = x_valid_raw.
        # Разные fit_df на разных фолдах — поэтому это не утечка.
        x_train = make_features(x_train_raw, x_train_raw, y_train, ticket_group_sizes)
        x_valid = make_features(x_train_raw, x_valid_raw, y_train, ticket_group_sizes)

        # CatBoost получает другой seed на каждом фолде — это не критично
        # для метрик, но даёт более независимые модели для возможного бленда.
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

        # Остальные модели — единым циклом, потому что у них общий API
        # после оборачивания в build_sklearn_pipeline.
        for model_name in ["rf", "et", "hgb", "mlp", "xgb", "torch", "logreg"]:
            model = build_sklearn_pipeline(model_name)
            model.fit(x_train, y_train)
            proba = model.predict_proba(x_valid)[:, 1]
            pred = (proba >= 0.5).astype(int)
            scores[model_name]["acc"].append(accuracy_score(y_valid, pred))
            scores[model_name]["auc"].append(roc_auc_score(y_valid, proba))

    # Сворачиваем списки в среднее и std по фолдам — это итоговая таблица CV.
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
    """Обучить выбранную модель на всём train и получить предсказания для test.

    На этом этапе делить данные уже не нужно — все 891 пример Titanic-train
    идут в обучение, а статистики и приоры считаются на полном train.
    Возвращает кортеж (модель, бинарные предсказания на test).

    Для CatBoost берём более скромные iterations и learning_rate, чем в CV:
    в CV была early stopping и eval_set, которые помогали выбирать оптимальное
    число итераций. На финальном обучении eval_set'а нет, поэтому хардкодим
    разумные значения, которые показали себя стабильно.
    """
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

    # Порог 0.5 — стандарт для бинарной классификации. Можно тюнить под
    # F1/precision-recall, но для accuracy на сбалансированном датасете
    # это разумное умолчание.
    test_pred = (test_proba >= 0.5).astype(int)
    return model, test_pred


def main() -> None:
    """Главная точка входа: CV → выбор лучшей модели → финальное обучение → submission."""
    parser = argparse.ArgumentParser(description="Train Titanic models and build Kaggle submission.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--force-model",
        type=str,
        default="",
        choices=["", "catboost", "rf", "et", "hgb", "mlp", "xgb", "torch", "logreg"],
        help="Принудительно использовать модель для финального сабмита, минуя авто-выбор по CV.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)

    # Импорт внутри main, чтобы при импорте модуля как библиотеки
    # не выполнялась логика создания директорий.
    from src.common import make_artifact_dirs, load_train_test

    dirs = make_artifact_dirs(artifact_dir)
    report_dir = dirs["reports"]
    submission_dir = dirs["submissions"]

    train_df, test_df = load_train_test(data_dir)

    y = train_df["Survived"].astype(int)

    # Размер группы по билету считаем на полном train+test — это безопасно,
    # потому что фича не зависит от таргета.
    ticket_group_sizes = get_ticket_group_sizes(train_df, test_df)

    # CV-оценка всех моделей. Результат сохраняем в CSV, чтобы было видно
    # историю экспериментов и можно было сравнивать запуски.
    score_df = evaluate_models(train_df, y, ticket_group_sizes, n_splits=args.n_splits)
    score_df.to_csv(report_dir / "cv_scores.csv", index=False)

    # Выбор финальной модели: либо по флагу --force-model, либо лучшая по CV
    # (таблица отсортирована по cv_accuracy_mean → cv_auc_mean).
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

    # Сабмишен с таймстемпом, чтобы старые запуски не затирались —
    # при ревью полезно иметь несколько версий для сравнения.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = submission_dir / f"submission_{best_model_name}_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)

    # Сводный отчёт: фиксируем выбранную модель, всю CV-таблицу и
    # путь к сабмишену. Этот JSON — главный артефакт для разбора запуска.
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