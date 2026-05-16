# ML-хакатон: Система ранжирования кандидатов на смены

Полное руководство по проекту ML-хакатона от Авито Подработка.

## 📋 Содержание

- [Описание проекта](#описание-проекта)
- [Архитектура решения](#архитектура-решения)
- [Структура проекта](#структура-проекта)
- [Установка и настройка](#установка-и-настройка)
- [Обучение модели](#обучение-модели)
- [Запуск сервиса](#запуск-сервиса)
- [Тестирование](#тестирование)
- [API документация](#api-документация)
- [Метрики и оценка](#метрики-и-оценка)
- [Troubleshooting](#troubleshooting)

---

## Описание проекта

### Бизнес-задача

Предсказать, какие пользователи с наибольшей вероятностью выйдут на конкретную смену. Система должна ранжировать кандидатов по вероятности успешного выхода на работу.

### Данные

**Train данные (январь 2026):**
- 5,154 пользователя
- 39,999 смен
- 383,883 событий (VIEW, APPLY, FINISHED, USER_CANCEL, SYSTEM_CANCEL)

**Validation данные:**
- Предоставляются организаторами отдельно
- Используются для финальной оценки решения

### Целевая метрика

```
ROC-AUC с ограничением FPR
max_fpr = min(1.0, capacity / 10)
pool_size = 10 (топ-10 кандидатов на смену)
```

Агрегация: по группам емкости внутри дня, затем среднее по дням.

### Текущие результаты

| Метрика | Baseline (LogisticRegression) | Текущее решение (CatBoost) |
|---------|-------------------------------|------------------------------|
| Target metric | 0.78 | 0.987 |
| Признаков | 18 | 31 |
| Модель | LogisticRegression | CatBoostClassifier |

---

## Архитектура решения

### Компоненты системы

```
┌─────────────────────────────────────────────────────────────┐
│                     RPC-сервис (Zero)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Repository │  │PrepareManager│  │  Predictor   │       │
│  │   (SQLite)   │  │              │  │  (CatBoost)  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ TCP (port 8000)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Клиенты / Eval                           │
│  • user/shift/event загрузка                                │
│  • prepare/ready проверка                                   │
│  • predict ранжирование                                     │
└─────────────────────────────────────────────────────────────┘
```

### Поток данных

1. **Загрузка данных:** `user()`, `shift()`, `event()` → SQLite
2. **Подготовка модели:** `prepare()` → загрузка модели и данных
3. **Проверка готовности:** `ready()` → статус модели
4. **Предсказание:** `predict()` → ранжирование кандидатов

---

## Структура проекта

```
ml-hackathon/
├── hackaton/                    # Основной пакет
│   ├── service/                 # RPC-сервис
│   │   ├── app.py               # Основная логика API
│   │   ├── predictor.py         # ML-модель и feature engineering
│   │   ├── repositories.py      # Работа с БД
│   │   ├── prepare_manager.py   # Управление состоянием
│   │   ├── dto.py               # Data Transfer Objects
│   │   ├── config.py            # Конфигурация
│   │   ├── db.py                # Инициализация БД
│   │   └── main.py              # Точка входа сервиса
│   ├── train/                   # Обучение модели
│   │   ├── training.py          # Пайплайн обучения
│   │   └── cli.py               # CLI для обучения
│   └── eval/                    # Оценка качества
│       ├── evaluator.py         # Симуляция дневного цикла
│       ├── metric.py            # Расчет целевой метрики
│       └── cli.py               # CLI для eval
├── tests/                       # Тесты
│   ├── unit/                    # Unit-тесты
│   └── e2e/                     # E2E-тесты
├── data/                        # Данные
│   ├── train/                   # Train CSV
│   └── validation/              # Validation CSV (от организаторов)
├── notebooks/                   # Jupyter ноутбуки
│   └── avito_analysis_with_edits.ipynb
├── scripts/                     # Вспомогательные скрипты
├── pyproject.toml               # Зависимости (Poetry)
├── README.md                    # Подробное руководство (этот файл)
├── HOW-TO.md                    # Практический гайд
├── DATA.md                      # Форматы данных
├── TRAIN.md                     # Детали обучения
├── REGLAMENT.md                 # Регламент оценки
└── CHECKLIST.md                 # Чеклист перед коммитом
```

---

## Установка и настройка

### Требования

- Python 3.12+
- Poetry 1.8+
- Git

### Установка зависимостей

```bash
# Клонировать репозиторий
git clone <repository-url>
cd ml-hackathon

# Установить зависимости через Poetry
poetry install

# Активировать виртуальное окружение
poetry shell
```

### Основные зависимости

```toml
[tool.poetry.dependencies]
python = "^3.12"
pydantic = "^2.11.7"          # Валидация данных
aiosqlite = "^0.21.0"         # Async SQLite
prometheus-client = "^0.23.1" # Метрики
zeroapi = "v1.0.1"            # RPC-фреймворк
pandas = "^3.0.2"             # Обработка данных
numpy = "^2.4.4"              # Численные вычисления
scikit-learn = "^1.8.0"       # ML-утилиты
catboost = "^1.2.10"          # ML-модель
lightgbm = "^4.6.0"           # ML-модель (legacy)
shap = "^0.51.0"              # Интерпретация модели
matplotlib = "^3.10.8"        # Визуализация
```

### Инициализация базы данных

```bash
# База создается автоматически при первом запуске сервиса
# Или вручную:
poetry run python -c "
from hackaton.service.db import init_db_for
import asyncio
asyncio.run(init_db_for('./data/hackaton.db'))
"
```

---

## Обучение модели

### Быстрый старт

```bash
poetry run python -m hackaton.train.cli train \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/train \
  --skip-shap
```

### Параметры обучения

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--user-path` | Путь к user.csv | Обязательный |
| `--shift-path` | Путь к shift.csv | Обязательный |
| `--event-path` | Путь к event.csv | Обязательный |
| `--output-dir` | Директория для артефактов | Обязательный |
| `--random-state` | Random seed | 42 |
| `--test-ratio` | Доля test выборки | 0.2 |
| `--skip-shap` | Пропустить SHAP анализ | False |
| `--shap-sample-size` | Размер выборки для SHAP | 1000 |

### Этапы обучения

**Stage 1/8:** Загрузка и валидация CSV
- Проверка обязательных колонок
- Приведение типов данных
- Очистка от пропусков

**Stage 2/8:** Построение обучающего фрейма
- Создание пар user-shift
- Извлечение целевой переменной (APPLY/FINISHED)
- Feature engineering (31 признак)

**Stage 3/8:** Временной сплит (80/20)
- Разделение по датам (без утечки будущего)
- Train: более ранние даты
- Test: более поздние даты

**Stage 4/8:** Список признаков и превью
- Вывод всех 31 признака
- Примеры значений

**Stage 5/8:** Обучение CatBoost
- 2000 итераций максимум
- Early stopping (50 итераций без улучшения)
- Scale pos weight = 7 (компенсация дисбаланса 1:7)

**Stage 6/8:** Инференс и расчет метрики
- Предсказание на test выборке
- Расчет target metric по регламенту

**Stage 7/8:** Сохранение артефактов
- `model.pkl` — сериализованная модель
- `metrics.json` — метрики качества
- `feature_schema.json` — схема признаков
- `train_config.json` — конфигурация обучения
- `train_report.md` — отчет

**Stage 8/8:** Завершение

### Артефакты обучения

После обучения в `artifacts/train/` создаются:

```
artifacts/train/
├── model.pkl                 # 5.3 MB - CatBoost модель
├── metrics.json              # Метрики качества
├── feature_schema.json       # Список признаков + примеры
├── train_config.json         # Параметры обучения
├── train_report.md           # Человекочитаемый отчет
└── data_contract_check.json  # Результаты валидации данных
```

### Признаки модели (31 шт)

**Характеристики смены (4):**
- `hours` — длительность смены
- `reward` — вознаграждение
- `reward_per_hour` — вознаграждение в час
- `capacity` — емкость смены

**Временные признаки (3):**
- `hour` — час начала смены
- `weekday` — день недели
- `is_holiday` — новогодние праздники

**Активность пользователя (8):**
- `view_cnt` — просмотры текущей смены
- `user_hist_views` — история просмотров
- `user_hist_applies` — история откликов
- `user_hist_finished` — завершенные смены
- `user_hist_cancels` — отмены
- `user_apply_rate` — конверсия просмотр → отклик
- `user_finish_rate` — конверсия отклик → завершение
- `user_cancel_rate` — доля отмен

**Опыт пользователя (3):**
- `cum_shifts_viewed` — накопленное число просмотров
- `recency_days` — давность последней активности
- `worked_long_recently` — работал 8+ч за последние 2 дня

**Совпадения (6):**
- `same_location` — совпадение локации
- `mk_ok` — соответствие медкнижки
- `is_strict_location` — строгая привязка к локации
- `worked_employer_before` — работал у работодателя
- `worked_workplace_before` — работал на точке
- `task_match` — совпадение типа задачи

**Популярность (3):**
- `emp_ctr` — CTR работодателя
- `fill_rate` — заполненность смены
- `days_to_shift` — дней до смены с первого просмотра

**Требования (4):**
- `need_mk` — требуется медкнижка
- `id_differential` — требуется удостоверение
- `has_mk` — есть медкнижка
- `task_type` — тип задачи (категориальный)

---

## Запуск сервиса

### Быстрый старт

```bash
# Запустить сервис
poetry run python -m hackaton.service.main

# Или через make
make run
```

### Параметры запуска

Конфигурация через переменные окружения или `.env`:

```bash
# .env файл
HOST=0.0.0.0
PORT=8000
DB_PATH=./data/hackaton.db
MODEL_PATH=artifacts/train/model.pkl
PREPARE_SLEEP_SECONDS=10
```

### Логи запуска

```
Model loaded successfully from artifacts/train/model.pkl
Starting service: host=0.0.0.0 port=8000
Database initialized successfully
Starting server at tcp://0.0.0.0:8000
Starting worker 1...16
```

### Проверка работы

```bash
# Простая проверка
poetry run python check_service.py

# Вывод:
# Health: {'status': 'ok', 'status_code': 200}
# Users count: {'count': 0}
# Shifts count: {'count': 0}
# Events count: {'count': 0}
```

### Остановка сервиса

```bash
# Ctrl+C в терминале

# Или найти и убить процесс
netstat -ano | findstr :8000
taskkill /PID <номер_процесса> /F
```

---

## Тестирование

### Запуск всех тестов

```bash
# Все тесты + coverage
poetry run pytest tests/ -v

# Или через make
make test
```

### Структура тестов

**Unit-тесты (`tests/unit/`):**
- `test_service_smoke.py` — базовая функциональность сервиса
- `test_eval_metric.py` — расчет целевой метрики
- `test_eval_config_and_cli.py` — конфигурация eval
- `test_train_smoke.py` — smoke-тест обучения

**E2E-тесты (`tests/e2e/`):**
- `test_rpc_api_contract_e2e.py` — полный цикл API

### Coverage gate

Проект требует минимум **80% покрытия кода**:

```toml
[tool.pytest.ini_options]
addopts = [
  "--cov=hackaton.eval.cli",
  "--cov=hackaton.eval.metric",
  "--cov=hackaton.service.app",
  "--cov=hackaton.service.prepare_manager",
  "--cov-report=term-missing",
  "--cov-fail-under=80",
]
```

Текущее покрытие: **92.56%**

### Pre-commit hooks

```bash
# Установить pre-commit hooks
poetry run pre-commit install

# Запустить вручную
poetry run pre-commit run --all-files

# Или через make
make precommit
```

Hooks включают:
- `ruff` — линтер и форматтер
- `pytest` — запуск тестов
- Проверка YAML/JSON
- Проверка trailing whitespace

---

## API документация

### Эндпоинты

#### 1. `health` — Проверка здоровья сервиса

**Запрос:**
```python
client.call("health", None)
```

**Ответ:**
```json
{
  "status": "ok",
  "status_code": 200
}
```

---

#### 2. `user` — Загрузка пользователей

**Запрос:**
```python
payload = {
  "items": [
    {
      "id": "user-123",
      "location_id": "loc-1",
      "is_strict_location": true,
      "has_mk": true
    }
  ]
}
client.call("user", payload)
```

**Ответ:**
```json
{
  "accepted": 1
}
```

---

#### 3. `shift` — Загрузка смен

**Запрос:**
```python
payload = {
  "items": [
    {
      "id": "shift-456",
      "start_at": "2026-05-16T10:00:00Z",
      "location_id": "loc-1",
      "task_type": "loader",
      "employer_id": "emp-1",
      "workplace_id": "wp-1",
      "need_mk": true,
      "id_differential": false,
      "hours": 8,
      "reward": 1200.0,
      "capacity": 2
    }
  ]
}
client.call("shift", payload)
```

**Ответ:**
```json
{
  "accepted": 1
}
```

---

#### 4. `event` — Загрузка событий

**Запрос:**
```python
payload = {
  "items": [
    {
      "id": "event-789",
      "shift_id": "shift-456",
      "user_id": "user-123",
      "interaction": "VIEW",
      "ts": "2026-05-16T09:00:00Z"
    }
  ]
}
client.call("event", payload)
```

**Ответ:**
```json
{
  "accepted": 1
}
```

**Типы interaction:**
- `VIEW` — просмотр смены
- `APPLY` — отклик на смену
- `FINISHED` — завершение смены
- `USER_CANCEL` — отмена пользователем
- `SYSTEM_CANCEL` — отмена системой

---

#### 5. `prepare` — Подготовка модели

**Запрос:**
```python
client.call("prepare", None)
```

**Ответ (успех):**
```json
{
  "status": "started",
  "status_code": 200
}
```

**Ответ (уже запущен):**
```json
{
  "status": "already_running",
  "status_code": 409
}
```

---

#### 6. `ready` — Проверка готовности

**Запрос:**
```python
client.call("ready", None)
```

**Ответ (готов):**
```json
{
  "ready": true,
  "status_code": 200
}
```

**Ответ (не готов):**
```json
{
  "ready": false,
  "status_code": 425
}
```

---

#### 7. `predict` — Ранжирование кандидатов

**Запрос:**
```python
payload = {
  "shift": {
    "id": "shift-456",
    "start_at": "2026-05-16T10:00:00Z",
    "location_id": "loc-1",
    "task_type": "loader",
    "employer_id": "emp-1",
    "workplace_id": "wp-1",
    "need_mk": true,
    "id_differential": false,
    "hours": 8,
    "reward": 1200.0,
    "capacity": 2
  },
  "limit": 10
}
client.call("predict", payload)
```

**Ответ (успех):**
```json
{
  "user_ids": ["user-123", "user-456", "user-789"],
  "status_code": 200
}
```

**Ответ (модель не готова):**
```json
{
  "user_ids": [],
  "status_code": 503,
  "detail": "model is in prepare state"
}
```

**Ответ (нет кандидатов):**
```json
{
  "user_ids": [],
  "status_code": 400,
  "detail": "no users loaded"
}
```

---

#### 8. `user_stat`, `shift_stat`, `event_stat` — Статистика

**Запрос:**
```python
client.call("user_stat", None)
```

**Ответ:**
```json
{
  "count": 5154
}
```

---

#### 9. `metrics` — Prometheus метрики

**Запрос:**
```python
client.call("metrics", None)
```

**Ответ:**
```json
{
  "content_type": "text/plain; version=0.0.4",
  "payload": "# HELP api_requests_total Total API requests\n...",
  "status_code": 200
}
```

---

## Метрики и оценка

### Запуск eval

```bash
poetry run python -m hackaton.eval.cli run \
  --host 127.0.0.1 \
  --port 8000 \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_run \
  --predict-max-concurrency 4 \
  --predict-max-rpm 200
```

### Что делает eval

1. Загружает train данные в сервис
2. Вызывает `prepare()` и ждет `ready()`
3. Для каждого дня validation:
   - Загружает новые смены и события
   - Для каждой смены вызывает `predict()`
   - Сравнивает топ-10 с фактом из apply.csv
   - Считает ROC-AUC с ограничением FPR
4. Агрегирует метрики по дням
5. Создает `eval_report.md`

### Целевая метрика

**Формула:**
```python
def calculate_target_metric(frame):
    # Для каждой смены:
    # 1. Берем топ-10 кандидатов по score
    # 2. Из них берем топ-capacity
    # 3. Считаем ROC-AUC с max_fpr = min(1.0, capacity/10)
    # 4. Агрегируем по группам (день, capacity)
    # 5. Среднее по дням
```

**Пример:**
- Смена с capacity=5
- Топ-10 кандидатов: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
- Берем топ-5: [0.9, 0.8, 0.7, 0.6, 0.5]
- max_fpr = min(1.0, 5/10) = 0.5
- ROC-AUC с max_fpr=0.5

### Интерпретация метрики

| Target metric | Интерпретация |
|---------------|---------------|
| 0.5 | Случайное угадывание |
| 0.7 | Приемлемое качество |
| 0.8 | Хорошее качество |
| 0.9+ | Отличное качество |
| 0.987 | Текущий результат |

---

## Troubleshooting

### Проблема: Address in use (port 8000)

**Причина:** Сервис уже запущен на порту 8000.

**Решение:**
```bash
# Найти процесс
netstat -ano | findstr :8000

# Убить процесс
taskkill /PID <номер> /F

# Или использовать другой порт
poetry run python -m hackaton.service.main --port 8001
```

---

### Проблема: Model not found

**Причина:** Модель не обучена или путь неверный.

**Решение:**
```bash
# Проверить наличие модели
ls artifacts/train/model.pkl

# Обучить модель
poetry run python -m hackaton.train.cli train \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/train \
  --skip-shap
```

---

### Проблема: Feature mismatch

**Причина:** Модель обучена на другом наборе признаков.

**Решение:**
```bash
# Проверить количество признаков
python -c "
import pickle
model = pickle.load(open('artifacts/train/model.pkl', 'rb'))
print('Model expects:', model.n_features_, 'features')
"

# Переобучить модель с правильными признаками
poetry run python -m hackaton.train.cli train ...
```

---

### Проблема: Tests failing

**Причина:** Изменения в коде нарушили тесты.

**Решение:**
```bash
# Запустить тесты с подробным выводом
poetry run pytest tests/ -v -s

# Проверить coverage
poetry run pytest tests/ --cov-report=html
# Открыть htmlcov/index.html

# Обновить моки если изменился API
```

---

### Проблема: Low target metric

**Причина:** Модель недостаточно хорошо обучена.

**Решение:**
1. Проверить качество данных (пропуски, выбросы)
2. Добавить новые признаки
3. Настроить гиперпараметры CatBoost
4. Проверить утечку данных (temporal leakage)
5. Увеличить размер train выборки

---

### Проблема: Slow predictions

**Причина:** Feature engineering слишком медленный.

**Решение:**
1. Оптимизировать SQL-запросы (индексы)
2. Кэшировать статические признаки (emp_ctr)
3. Уменьшить pool кандидатов (100 → 50)
4. Использовать batch predictions

---

## Дополнительные ресурсы

### Документация

- `README.md` — краткое описание
- `HOW-TO.md` — практический гайд для участников
- `DATA.md` — форматы данных с примерами
- `TRAIN.md` — детали train-пайплайна
- `REGLAMENT.md` — официальный регламент оценки
- `CHECKLIST.md` — чеклист перед коммитом
- `CLAUDE.md` — инструкции для разработки

### Полезные команды

```bash
# Установка
make install

# Миграции БД
make migrate

# Запуск сервиса
make run

# Тесты
make test

# Pre-commit
make precommit

# Нагрузочное тестирование
make load-test

# Docker
make compose-up
make compose-down
```

### CI/CD

**GitHub Actions workflows:**
- `.github/workflows/ci.yml` — lint, test, coverage, smoke-test
- `.github/workflows/load-test.yml` — нагрузочное тестирование

**Требования для merge:**
- ✅ Все тесты проходят
- ✅ Coverage >= 80%
- ✅ Ruff проверки проходят
- ✅ Smoke-test проходит