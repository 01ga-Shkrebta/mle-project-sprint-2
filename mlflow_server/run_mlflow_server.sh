#!/bin/bash

export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

mlflow server \
  --backend-store-uri postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME?sslmode=require \
  --default-artifact-root s3://$S3_BUCKET_NAME \
  --no-serve-artifacts \
  --host 0.0.0.0 \
  --port 5000
