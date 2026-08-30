import os
import optuna
import numpy as np
import pandas as pd
import psycopg2
import mlflow
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, log_loss
from collections import defaultdict

# ====== ЗАГРУЗКА ДАННЫХ ======
connection = {
    "sslmode": "require",
    "target_session_attrs": "read-write",
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}

with psycopg2.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users_churn")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

# ====== ПРИЗНАКИ ======
cat_features = [
    'type', 'paperless_billing', 'payment_method', 'internet_service',
    'online_security', 'online_backup', 'device_protection', 'tech_support',
    'streaming_tv', 'streaming_movies', 'gender', 'partner', 'dependents', 'multiple_lines'
]

for col in cat_features:
    df[col] = df[col].fillna("missing")

X = df.drop(columns=['target', 'customer_id', 'begin_date', 'end_date'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== MLflow ======
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

EXPERIMENT_NAME = "churn_nikolaistepanov"
RUN_NAME = "model_bayesian_search"
STUDY_DB_NAME = "sqlite:///local.study.db"
STUDY_NAME = "churn_model"

# ====== РОДИТЕЛЬСКИЙ RUN ======
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as parent_run:
    parent_run_id = parent_run.info.run_id
    print(f"🔗 Parent Run ID: {parent_run_id}")

    def objective(trial: optuna.Trial) -> float:
        # Каждый trial — дочерний run
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as child_run:
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)

            param = {
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "depth": trial.suggest_int("depth", 1, 12),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 5),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5),
                "loss_function": "Logloss",
                "task_type": "CPU",
                "random_seed": 0,
                "iterations": 300,
                "verbose": False
            }
            model = CatBoostClassifier(**param)

            skf = StratifiedKFold(n_splits=2)
            metrics = defaultdict(list)

            for train_index, val_index in skf.split(X_train, y_train):
                train_x = X_train.iloc[train_index]
                val_x = X_train.iloc[val_index]
                train_y = y_train.iloc[train_index]
                val_y = y_train.iloc[val_index]

                model.fit(train_x, train_y, cat_features=cat_features, eval_set=(val_x, val_y), verbose=False)

                prediction = model.predict(val_x)
                probas = model.predict_proba(val_x)[:, 1]

                _, err1, _, err2 = confusion_matrix(val_y, prediction, normalize='all').ravel()
                auc = roc_auc_score(val_y, probas)
                precision = precision_score(val_y, prediction)
                recall = recall_score(val_y, prediction)
                f1 = f1_score(val_y, prediction)
                logloss = log_loss(val_y, prediction)

                metrics["err1"].append(err1)
                metrics["err2"].append(err2)
                metrics["auc"].append(auc)
                metrics["precision"].append(precision)
                metrics["recall"].append(recall)
                metrics["f1"].append(f1)
                metrics["logloss"].append(logloss)

            err1 = sum(metrics["err1"]) / len(metrics["err1"])
            err2 = sum(metrics["err2"]) / len(metrics["err2"])
            auc = sum(metrics["auc"]) / len(metrics["auc"])
            precision = sum(metrics["precision"]) / len(metrics["precision"])
            recall = sum(metrics["recall"]) / len(metrics["recall"])
            f1 = sum(metrics["f1"]) / len(metrics["f1"])
            logloss = sum(metrics["logloss"]) / len(metrics["logloss"])

            mlflow.log_metrics({
                "err1": err1,
                "err2": err2,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "logloss": logloss
            })
            mlflow.log_params(param)

            return auc

    # ====== OPTUNA ======
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STUDY_DB_NAME,
        sampler=optuna.samplers.TPESampler(),
        direction="maximize",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=10)

    # ====== ЛОГИРОВАНИЕ CV ======
    # ====== ЛОГИРОВАНИЕ CV ======
    # ====== ЛОГИРОВАНИЕ CV ======
    import pickle
    with open("model.cb", "wb") as f:
        pickle.dump(StratifiedKFold(n_splits=2), f)
    mlflow.log_artifact("model.cb", artifact_path="cv")     

    # ====== ЛУЧШАЯ МОДЕЛЬ ======
    best_params = study.best_params
    best_model = CatBoostClassifier(**best_params, loss_function="Logloss", task_type="CPU", random_seed=0, iterations=300, verbose=False)
    best_model.fit(X_train, y_train, cat_features=cat_features)

    mlflow.catboost.log_model(
        best_model,
        artifact_path="models",
        registered_model_name="churn_model_nikolaistepanov"
    )

    print(f"✅ Байесовская оптимизация завершена")
    print(f"🔗 Parent Run ID: {parent_run_id}")
    print(f"📌 Лучшие параметры: {best_params}")
    print(f"📊 Лучший AUC: {study.best_value}")
