import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.run_info import RunInfo
from mlflow.entities import Run
from datetime import datetime

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'test-experiment')
ALIAS = os.getenv("MODEL_ALIAS")

def init_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not exp:
        raise ValueError(f"Experiment {MLFLOW_EXPERIMENT_NAME} not found")
    
    runs = client.search_runs([exp.experiment_id])
    if not runs:
        raise ValueError(f"No runs found in experiment {MLFLOW_EXPERIMENT_NAME}")

    return client, exp, runs

def pick_best_model(
        runs: list[Run],
        metric_name: str
    ) -> Run:
    
    # Filter runs to only include those that have the specified metric
    runs_with_metric = [run for run in runs if metric_name in run.data.metrics]
    if not runs_with_metric:
        raise ValueError(f"No runs found with metric '{metric_name}'")

    print(f"Found {len(runs_with_metric)} runs with metric '{metric_name}'")
    
    best_run = (
        max(runs_with_metric, key=lambda x: x.data.metrics[metric_name])
    )
    best_val = best_run.data.metrics[metric_name]
    print(f"Best {metric_name} value: {best_val}")
    
    return best_run

def register_model(
        client: MlflowClient,
        run: Run,
        model_name: str | None = None
    ):
    best_run_id = run.info.run_id

    if not model_name:
        model_type = run.data.tags.get("mlflow.model-type")
        model_name = f"{model_type}-{best_run_id}"

    mv = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=model_name
    )
    
    alias = ALIAS if ALIAS else "best_" + datetime.now().strftime("%Y%m")
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=mv.version
    )

    print(f"Registered model: {mv.name}")
    return mv.version, best_run_id, model_name

def promote_best_model():
    client, exp, runs = init_mlflow()
    print(f"Found {len(runs)} runs in experiment {MLFLOW_EXPERIMENT_NAME}")
    best_run = pick_best_model(runs, "roc_auc_score")
    print(f"Best run: {best_run.info.run_id}")
    version, run_id, model_name = register_model(client, best_run)
    print(f"Registered model: {model_name} with version {version} and run_id {run_id}")

if __name__ == "__main__":
    promote_best_model()
