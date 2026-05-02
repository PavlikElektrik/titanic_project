# Titanic Kaggle Pipeline

Этот проект делает полноценный локальный цикл для соревнования Titanic:
- EDA с графиками и markdown-отчетом
- feature engineering
- честная кросс-валидация без утечек для групповых survival-фич
- авто-выбор лучшей модели
- генерация submission CSV

## Что этот проект должен показать

Здесь важно не просто получить хороший скор, а показать полный и понятный Kaggle-процесс:

- сначала читаем постановку задачи и метрику
- затем выбираем простой, объяснимый бейзлайн
- отдельно объясняем, зачем нужен WCG-режим и где он опасен
- сохраняем артефакты, чтобы ревьюер видел ход экспериментов
- пишем код так, чтобы его можно было переиспользовать в следующем соревновании

## Структура

- data/ - исходные файлы train.csv и test.csv
- configs/wcg/default.yaml - базовый конфиг WCG
- configs/wcg/high_score.yaml - конфиг для агрессивного тюнинга
- src/run_eda.py - EDA-скрипт
- src/train_and_submit.py - честное обучение, CV (без target leakage), сабмит
- src/train_kaggle_wcg.py - конфигурируемый WCG entrypoint
- src/main.py - короткий алиас для запуска основного пайплайна
- src/wcg/config.py - загрузка YAML-конфигов
- src/wcg/features.py - построение WCG-фичей
- src/wcg/models.py - реестр и тюнинг моделей
- src/wcg/training.py - CV, OOF-бленды, генерация submission
- src/run_pipeline.py - единый запуск всего пайплайна
- notebooks/eda.ipynb - отдельный notebook с EDA и визуализациями
- artifacts/ - результаты (создается автоматически)

## Как читать проект

Сначала открой [docs/kaggle_playbook.md](docs/kaggle_playbook.md) — это общий шаблон для решения Kaggle-соревнований.

Потом прочитай [docs/titanic_rationale.md](docs/titanic_rationale.md) — там объяснено, почему для Titanic выбран именно такой бейзлайн и почему WCG-режим вынесен отдельно.

После этого смотри на код в таком порядке:

- `src/run_pipeline.py` - единая точка входа
- `src/train_and_submit.py` - основной честный пайплайн
- `src/train_kaggle_wcg.py` - режим с групповой эвристикой и тюнингом
- `src/wcg/features.py` - как строятся признаки
- `src/wcg/models.py` - почему в модели входят именно эти алгоритмы
- `src/wcg/training.py` - как устроены CV, OOF и блендинг

## 3. Titanic: постановка задачи и бейзлайн

| Вопрос | Ответ |
|---|---|
| Что предсказывается? | `Survived` (0/1) |
| Тип задачи | Бинарная классификация |
| Метрика | Accuracy (доля правильных ответов) |
| Бейзлайн | Logistic Regression, Random Forest, ExtraTrees, CatBoost |
| Доп. кандидаты | XGBoost, HistGradientBoosting, Torch MLP |
| Расширенный режим | WCG (`GroupSurvival`) как отдельная Kaggle-эвристика |

## 4. Titanic: метрики

### Baseline CV

| Модель | CV Accuracy, mean | CV Accuracy, std | CV AUC, mean | CV AUC, std |
|---|---:|---:|---:|---:|
| ExtraTrees | 0.783422 | 0.018221 | 0.837633 | 0.019326 |
| LogisticRegression | 0.756487 | 0.019166 | 0.797665 | 0.030442 |
| RandomForest | 0.745245 | 0.021298 | 0.827180 | 0.013118 |
| CatBoost | 0.717199 | 0.019854 | 0.835016 | 0.019397 |

### WCG CV

| Модель | CV Accuracy, mean | CV Accuracy, std |
|---|---:|---:|
| cb_wcg | 0.856318 | 0.017058 |
| et_wcg | 0.848490 | 0.016610 |
| rf_wcg | 0.848465 | 0.020842 |
| logreg_wcg | 0.833902 | 0.016036 |

| Дополнительно | Значение |
|---|---:|
| Train sanity accuracy (WCG) | 0.883277 |

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

## Что такое WCG и почему это важно

- **WCG (Within-Competition Grouping)**: режим/набор фич в этом проекте, который строит групповые признаки типа `GroupSurvival` — эвристические оценки шансов выжить на основе других участников той же группы (фамилия+fare, тот же билет и т.п.).
- **Почему это работает на Kaggle:** на задачах типа Titanic сильны внутригрупповые корреляции (члены семьи или пассажиры с одним билетом часто имеют одинаковую метку). Если групповые фичи посчитать, когда известны метки соседей (или использовать их нечаянно в CV), метрики CV и private LB могут существенно улучшиться.
- **Почему это риск для продакшна:** в реальном продакшне при онлайн‑инференсе вы обычно не имеете размеченных соседей в группе — значит `GroupSurvival` либо будет иметь нейтральное значение, либо даст неверный сигнал. Кроме того, если групповые фичи вычислять неправильно (на полном наборе данных, включая валидацию или тест), это приводит к утечке таргета и завышенным ожиданиям качества.
- **Как минимизировать риск:**
	- считать групповые фичи в OOF‑режиме при CV (статистики берём только с train‑части для каждого фолда), а при финальном трейне — на всём train;
	- применять сглаживание/регуляризацию для редких групп (бейзлайн — глобальная средняя);
	- добавлять фича‑флаг, чтобы отключать `GroupSurvival` в проде или для A/B тестов;
	- проверять модель на отложенном holdout, не использовавшемся при тюнинге.

В репозитории WCG реализован как опциональный режим (см. `configs/wcg/*` и `src/wcg/*`); рекомендую держать его для Kaggle‑экспериментов, но аккуратно тестировать и при необходимости отключать для продакшн‑запуска.

### 2.2) WCG + Optuna + набор кандидатных сабмитов

```bash
python src/train_kaggle_wcg.py --config configs/wcg/high_score.yaml
```

Что делает команда:
- подбирает гиперпараметры для RF, ExtraTrees и CatBoost через Optuna
- строит несколько сабмитов: rf, et, cb, xgb, torch
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

Если ты готовишь проект для ревью, начни именно с `artifacts/reports/train_summary.json` и `artifacts/reports/wcg_tuning_summary.json`: в них должен быть виден смысл решений, а не только числа.

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
