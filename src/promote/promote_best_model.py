import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.run_info import RunInfo
from mlflow.entities import Run
from datetime import datetime
import tempfile
import boto3
from botocore.exceptions import NoCredentialsError
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=project_root / ".env", override=True)

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


def upload_model_to_s3(model_name: str, version: int):
    # Create a temporary directory to download the model artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the model URI
        model_uri = f"models:/{model_name}/{version}"
        
        # Download the artifacts
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=temp_dir
        )
        print(f"Model artifacts downloaded to: {local_path}")

        # Find the model.pkl file within the downloaded artifacts
        model_pkl_path = None
        for root, _, files in os.walk(local_path):
            if "model.pkl" in files:
                model_pkl_path = os.path.join(root, "model.pkl")
                break
        
        if model_pkl_path:
            print(f"Successfully found and downloaded model.pkl at: {model_pkl_path}")
        else:
            print("Error: model.pkl not found in the downloaded artifacts.")
            return
    
        # link to s3 bucket
        s3_bucket = os.getenv("S3_BUCKET")
        if not s3_bucket:
            print("S3_BUCKET environment variable not set. Skipping S3 upload.")
            return

        s3 = boto3.client('s3')

        # Define the S3 key for the model.pkl file
        s3_key = os.path.join(model_name, str(version), "model.pkl")

        print(f"Uploading {model_pkl_path} to s3://{s3_bucket}/{s3_key}")
        try:
            s3.upload_file(model_pkl_path, s3_bucket, s3_key)
        except NoCredentialsError:
            print("Credentials not available for S3 upload.")
            return
        except Exception as e:
            print(f"Error uploading file: {e}")
            return
        
        print(f"Model {model_name} version {version} uploaded to s3://{s3_bucket}/{model_name}/{version}")

def promote_best_model():
    client, exp, runs = init_mlflow()
    print(f"Found {len(runs)} runs in experiment {MLFLOW_EXPERIMENT_NAME}")
    best_run = pick_best_model(runs, "roc_auc_score")
    print(f"Best run: {best_run.info.run_id}")
    version, run_id, model_name = register_model(client, best_run)
    print(f"Registered model: {model_name} with version {version} and run_id {run_id}")

    upload_model_to_s3(model_name, version)

if __name__ == "__main__":
    promote_best_model()
