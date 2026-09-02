# Проект: Улучшение baseline-модели для прогнозирования оттока клиентов

## Описание проекта

Цель проекта — улучшить качество прогнозирования оттока клиентов (Churn) с использованием MLflow, генерации признаков, отбора признаков и подбора гиперпараметров.

**Технологии:** Python, Pandas, Scikit-learn, CatBoost, Optuna, MLflow, PostgreSQL, S3 (Yandex Cloud).

---

## Структура репозитория

- `mlflow_server/` — скрипты для запуска MLflow и регистрации модели
- `model_improvement/` — Jupyter Notebook с этапами проекта
- `assets/` — графики и артефакты
- `requirements.txt` — зависимости
- `.gitignore` — игнорируемые файлы

---

## Инструкция по запуску

```bash
# 1. Клонировать репозиторий
git clone https://github.com/01ga-Shkrebta/mle-project-sprint-2.git
cd mle-project-sprint-2

# 2. Создать и активировать виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить MLflow сервер
./mlflow_server/run_mlflow_server.sh
