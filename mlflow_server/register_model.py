import os
import pandas as pd
import psycopg2
import mlflow
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, log_loss

# ====== ПОДКЛЮЧЕНИЕ К БД ======
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

# ====== ПРИЗНАКИ И ЦЕЛЕВАЯ ======
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

# ====== ОБУЧЕНИЕ BASELINE-МОДЕЛИ ======
model = CatBoostClassifier(
    loss_function="Logloss",
    task_type="CPU",
    random_seed=0,
    iterations=300,
    verbose=False
)

model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), verbose=False)

# ====== ПРЕДСКАЗАНИЯ ======
prediction = model.predict(X_test)
probas = model.predict_proba(X_test)[:, 1]

# ====== МЕТРИКИ ======
metrics = {
    "auc": roc_auc_score(y_test, probas),
    "f1": f1_score(y_test, prediction),
    "precision": precision_score(y_test, prediction),
    "recall": recall_score(y_test, prediction),
    "logloss": log_loss(y_test, probas)
}

# ====== MLflow ======
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

EXPERIMENT_NAME = "churn_project_experiment"
RUN_NAME = "baseline_model"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    mlflow.log_params({"iterations": 300, "loss_function": "Logloss", "task_type": "CPU", "random_seed": 0})
    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X_test, prediction)
    input_example = X_test[:5]

    model_info = mlflow.catboost.log_model(
        cb_model=model,
        artifact_path="models",
        registered_model_name="churn_baseline_model",
        signature=signature,
        input_example=input_example
    )

    print(f"✅ Базовая модель зарегистрирована")
    print(f"   Run ID: {run_id}")
    print(f"   Experiment ID: {experiment_id}")
    print(f"   Experiment Name: {EXPERIMENT_NAME}")
