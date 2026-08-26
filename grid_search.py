import os
import pandas as pd
import psycopg2
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, log_loss

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

# ====== ОСТАЛЬНОЙ КОД ======
TABLE_NAME = "users_churn"

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = "churn_nikolaistepanov"
RUN_NAME = "model_grid_search"
REGISTRY_MODEL_NAME = "churn_model_nikolaistepanov"

features = ["monthly_charges", "total_charges", "senior_citizen"]
target = "target"

split_column = "begin_date"
stratify_column = target
test_size = 0.2

df = df.sort_values(by=[split_column])

X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df[target],
    test_size=test_size,
    shuffle=False
)

print(f"Размер выборки для обучения: {X_train.shape}")
print(f"Размер выборки для теста: {X_test.shape}")

loss_function = "Logloss"
task_type = 'CPU'
random_seed = 0
iterations = 300
verbose = False

params = {
    'depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.3],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5]
}

model = CatBoostClassifier(
    loss_function=loss_function,
    task_type=task_type,
    random_seed=random_seed,
    iterations=iterations,
    verbose=verbose
)

cv = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=2,
    scoring='roc_auc',
    n_jobs=-1
)

clf = cv.fit(X_train, y_train)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

cv_results = pd.DataFrame(clf.cv_results_)

best_params = clf.best_params_

model_best = CatBoostClassifier(
    loss_function=loss_function,
    task_type=task_type,
    random_seed=random_seed,
    verbose=verbose,
    **best_params
)

model_best.fit(X_train, y_train)

prediction = model_best.predict(X_test)
probas = model_best.predict_proba(X_test)[:, 1]

metrics = {}

_, err1, _, err2 = confusion_matrix(y_test, prediction, normalize='all').ravel()
auc = roc_auc_score(y_test, probas)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
logloss = log_loss(y_test, prediction)

metrics["err1"] = err1
metrics["err2"] = err2
metrics["auc"] = auc
metrics["precision"] = precision
metrics["recall"] = recall
metrics["f1"] = f1
metrics["logloss"] = logloss

metrics["mean_fit_time"] = cv_results['mean_fit_time'].mean()
metrics["std_fit_time"] = cv_results['std_fit_time'].mean()
metrics["mean_test_score"] = cv_results['mean_test_score'].mean()
metrics["std_test_score"] = cv_results['std_test_score'].mean()
metrics["best_score"] = clf.best_score_

pip_requirements = "requirements.txt"
signature = mlflow.models.infer_signature(X_test, prediction)
input_example = X_test[:10]

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)
    cv_info = mlflow.sklearn.log_model(cv, artifact_path='cv')
    model_info = mlflow.catboost.log_model(
        cb_model=model_best,
        artifact_path="models",
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements
    )
    print(f"✅ Grid Search завершён")
    print(f"   Run ID: {run_id}")
