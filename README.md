# Titanic Kaggle Pipeline

Этот проект делает полноценный локальный цикл для соревнования Titanic:
- EDA с графиками и markdown-отчетом
- feature engineering
- честная кросс-валидация без утечек для групповых survival-фич
- авто-выбор лучшей модели
- генерация submission CSV

## Структура

- data/ - исходные файлы train.csv и test.csv
- src/run_eda.py - EDA-скрипт
- src/train_and_submit.py - честное обучение, CV (без target leakage), сабмит
- src/train_kaggle_wcg.py - Kaggle-ориентированный WCG-режим (для буста public LB)
- src/run_pipeline.py - единый запуск всего пайплайна
- artifacts/ - результаты (создается автоматически)

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

### 1) EDA

```bash
python src/run_eda.py --data-dir data --artifact-dir artifacts
```

### 2) Обучение и submission

```bash
python src/train_and_submit.py --data-dir data --artifact-dir artifacts --n-splits 5
```

### 2.1) Kaggle WCG режим (агрессивнее под LB)

```bash
python src/train_kaggle_wcg.py --data-dir data --artifact-dir artifacts
```

### 2.2) WCG + Optuna + набор кандидатных сабмитов

```bash
python src/train_kaggle_wcg.py --data-dir data --artifact-dir artifacts --tune --n-trials 30 --cv-splits 5
```

Что делает команда:
- подбирает гиперпараметры для RF, ExtraTrees и CatBoost через Optuna
- строит несколько сабмитов: rf, et, cb
- строит два бленда: равновесный и Optuna-weighted по OOF-предиктам

### 3) Все сразу

```bash
python src/run_pipeline.py
```

## Что внутри фичей

Используются:
- базовые признаки: Sex, Pclass, Age, Fare, Embarked, SibSp, Parch
- engineered признаки: Title, Deck, FamilySize, IsAlone, NameLength, TicketGroupSize, FarePerPerson
- групповой признак WCG: GroupSurvival (эвристика по фамилии+тарифу и по билету)

Важно: групповые survival-фичи считаются OOF-способом в CV, поэтому метрика ближе к реальному Kaggle-сценарию и не завышается утечками.

## Результаты

После запуска смотри:
- artifacts/reports/eda_report.md
- artifacts/reports/cv_scores.csv
- artifacts/reports/cv_scores_wcg.csv
- artifacts/reports/train_summary.json
- artifacts/reports/wcg_tuning_summary.json
- artifacts/submissions/submission_*.csv

Готовый submission загружается в Kaggle как есть.
