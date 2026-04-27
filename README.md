# Titanic Kaggle Pipeline

Этот проект делает полноценный локальный цикл для соревнования Titanic:
- EDA с графиками и markdown-отчетом
- feature engineering
- честная кросс-валидация без утечек для групповых survival-фич
- авто-выбор лучшей модели
- генерация submission CSV

## Структура

- data/ - исходные файлы train.csv и test.csv
- configs/wcg/default.yaml - базовый конфиг WCG
- configs/wcg/high_score.yaml - конфиг для агрессивного тюнинга
- src/run_eda.py - EDA-скрипт
- src/train_and_submit.py - честное обучение, CV (без target leakage), сабмит
- src/train_kaggle_wcg.py - конфигурируемый WCG entrypoint
- src/wcg/config.py - загрузка YAML-конфигов
- src/wcg/features.py - построение WCG-фичей
- src/wcg/models.py - реестр и тюнинг моделей
- src/wcg/training.py - CV, OOF-бленды, генерация submission
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

### 2.1) Kaggle WCG режим через конфиг

```bash
python src/train_kaggle_wcg.py --config configs/wcg/default.yaml
```

### 2.2) WCG + Optuna + набор кандидатных сабмитов

```bash
python src/train_kaggle_wcg.py --config configs/wcg/high_score.yaml
```

Что делает команда:
- подбирает гиперпараметры для RF, ExtraTrees и CatBoost через Optuna
- строит несколько сабмитов: rf, et, cb
- строит три бленда: равный, fixed weights из конфига, Optuna-weighted

Можно переопределить поля из конфига через CLI:

```bash
python src/train_kaggle_wcg.py --config configs/wcg/default.yaml --cv-splits 6 --tune --n-trials 40
```

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

## Как менять параметры без правки кода

Редактируй только YAML-файлы в папке configs/wcg:
- models.rf, models.et, models.cb - гиперпараметры моделей
- training.tune_models и training.tune_model_trials - включение и интенсивность Optuna-тюнинга
- blend.fixed_weights - веса фиксированного бленда
- outputs.generate_submissions - какие типы сабмитов генерировать

После изменений запускай:

```bash
python src/train_kaggle_wcg.py --config configs/wcg/default.yaml
```
