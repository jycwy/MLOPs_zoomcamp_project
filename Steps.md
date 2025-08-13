# Training and Promote Process
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

- Start prefect service under pipenv shell
```bash
prefect server start
```

  - create a work pool (once) 
  ```bash
  prefect work-pool create -t process default
  ```

  - start a worker pool
  ```bash
  prefect worker start -p default
  ```

  - deploy the flow under the **root folder**
  ```bash
  prefect deploy src/train_promote_flow.py:train_and_promote_flow -n test-training-promote-flow -p default
  ```

  - start a run of the deployed flow
    - from ui
    - command line
    ```bash
    prefect deployment run train_and_promote_flow/test-training-promote-flow
    ```