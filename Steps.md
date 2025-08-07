# Training Process
- Start MLflow under **src/model training** folder under pipenv shell
```bash
    mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file:./artifacts \
  --serve-artifacts \
  --host 127.0.0.1 \
  --port 5001
```

or

```bash
mlflow server --config conf/mlflow_local.yaml --serve-artifacts
```