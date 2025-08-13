import os
import mlflow
import json
import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from mlflow.models import infer_signature
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from data_utils import (
    load_data,
    vectorize,
    split_data_randomly,
    load_parquet,
    prepare_xgb_data,
    load_parquet_task,
    prepare_xgb_data_task,
)
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from tqdm import tqdm
from prefect import flow, get_run_logger, task

# Load environment variables from .env file
# Robustly find the repository root by locating common markers
# This ensures paths are correct when code is executed from Prefect temp dirs

def find_project_root(start_path: Path) -> Path:
    for candidate in [start_path] + list(start_path.parents):
        if (candidate / "Pipfile").exists() or (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    return start_path

project_root = find_project_root(Path(__file__).resolve())
load_dotenv(dotenv_path=project_root / ".env", override=True)

# Disable MLflow "Logged Models" feature to maintain compatibility with older servers
os.environ.setdefault("MLFLOW_ENABLE_LOGGED_MODELS", "false")

# Get environment variables
TRAINING_DATA_PATH = project_root / os.getenv('TRAINING_DATA_PATH', './data/Churn')

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'test-experiment')

# Previously discovered best hyperparameters (obtained from an earlier tuning run)
cached_best_params = {
    "objective": "binary:logistic",
    "n_estimators": 350,
    "max_depth": 19,
    "min_child_weight": 3.6054254230376572,
    "learning_rate": 0.0064027841872601595,
    "gamma": 0.46063546324565136,
    "colsample_bytree": 0.54,
    "colsample_bynode": 0.81,
    "colsample_bylevel": 0.75,
    "subsample": 0.8,
    "reg_alpha": 4.399584091428253,
    "reg_lambda": 2.0034060827004825,
}

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def get_xgb_params():
    """
    Get the search space for the XGBoost model
    """
    XGB_MAX_DEPTH = 25
    xgb_space = {
        "objective": hp.choice("objective", ["binary:logistic"]),
        "n_estimators": hp.choice("n_estimators", np.arange(50, 500, 50, dtype=int)),
        "max_depth": hp.choice("max_depth", np.arange(1, XGB_MAX_DEPTH, dtype=int)),
        "min_child_weight": hp.uniform("min_child_weight", 0, 5),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.2)),
        "gamma": hp.uniform("gamma", 0, 5),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.1, 1, 0.01),
        "colsample_bynode": hp.quniform("colsample_bynode", 0.1, 1, 0.01),
        "colsample_bylevel": hp.quniform("colsample_bylevel", 0.1, 1, 0.01),
        "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
        "reg_lambda": hp.uniform("reg_lambda", 0, 5),
    }
    return xgb_space

def tune_hyperparameters(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        search_space: dict
    ):
    def xgb_objective(params: Dict) -> float:
        with mlflow.start_run(nested=True):
            # cast discrete params to int
            params = params.copy()
            params["max_depth"] = int(params["max_depth"])
            params["n_estimators"] = int(params["n_estimators"])
            mlflow.log_params(params)

            # Use only training data for cross-validation
            d_train = xgb.DMatrix(X_train.astype("category"), label=y_train,
                             enable_categorical=True)

            # Set up parameters for xgb.cv
            cv_params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "random_state": 42,
                "max_depth": params["max_depth"],
                "min_child_weight": params["min_child_weight"],
                "learning_rate": params["learning_rate"],
                "gamma": params["gamma"],
                "colsample_bytree": params["colsample_bytree"],
                "colsample_bynode": params["colsample_bynode"],
                "colsample_bylevel": params["colsample_bylevel"],
                "subsample": params["subsample"],
                "reg_alpha": params["reg_alpha"],
                "reg_lambda": params["reg_lambda"],
            }

            # Use cross-validation
            cv_results = xgb.cv(
                cv_params,
                d_train,
                num_boost_round=params["n_estimators"],
                nfold=5,
                early_stopping_rounds=20,
                verbose_eval=False,
                seed=42
            )

            # Get the best AUC score from cross-validation
            auc = cv_results['test-auc-mean'].max()
            mlflow.log_metric("auc", auc)

            # Update progress bar if it exists in global scope
            if hasattr(tune_hyperparameters, '_pbar'):
                tune_hyperparameters._pbar.update(1)
                # Display current best score
                completed_trials = [trial for trial in tune_hyperparameters._trials.trials if trial['result']['status'] == STATUS_OK]
                if completed_trials:
                    current_best = -min([trial['result']['loss'] for trial in completed_trials])
                    tune_hyperparameters._pbar.set_postfix({'Best AUC': f'{current_best:.4f}'})

            # Hyperopt minimizes loss → return negative AUC
            return {"loss": -auc, "status": STATUS_OK}

    # Set up progress bar
    max_evals = 10 # 60
    tune_hyperparameters._pbar = tqdm(total=max_evals, desc="Hyperparameter Optimization")
    
    trials = Trials()
    tune_hyperparameters._trials = trials
    
    try:
        best_indices = fmin(
            fn=xgb_objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(42),
            show_progressbar=False,  # Disable hyperopt's built-in progress bar
        )
    finally:
        tune_hyperparameters._pbar.close()
        # Clean up
        if hasattr(tune_hyperparameters, '_pbar'):
            delattr(tune_hyperparameters, '_pbar')
        if hasattr(tune_hyperparameters, '_trials'):
            delattr(tune_hyperparameters, '_trials')

    best_params = space_eval(search_space, best_indices)
    print("\nBest hyper-parameters:")
    print(best_params)

    return best_params

 
@task(name="train_xgb_classifier")
def train_xgb_classifier_task(best_params: Dict, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    xgb_model = XGBClassifier(**best_params, enable_categorical=True)
    xgb_model.fit(X_train, y_train)
    return xgb_model


@task(name="predict")
def predict_task(model: XGBClassifier, X_val: pd.DataFrame) -> np.ndarray:
    return model.predict(X_val)


@flow(name="train_xgb_model_flow")
def train_xgb_model(data_path: str, model_name: str, random_state: int, test_size: float):
    logger = get_run_logger()
    logger.info(f"Starting training flow: data_path={data_path}, model_name={model_name}")

    logger.info("Loading training data from parquet")
    churn_data_future = load_parquet_task.submit(data_path)
    churn_data = churn_data_future.result()
    logger.info(f"Loaded data shape: {churn_data.shape}")
    
    # Use vectorize function to properly handle categorical variables
    logger.info("Preparing train/validation splits for XGBoost")
    prepared_future = prepare_xgb_data_task.submit(churn_data)
    x_train, x_val, y_train, y_val = prepared_future.result()
    logger.info(f"Prepared splits: X_train={getattr(x_train, 'shape', None)}, X_val={getattr(x_val, 'shape', None)}")

    mlflow.xgboost.autolog(log_models=False)

    with mlflow.start_run():
        mlflow.set_tag("mlflow.model-type", "xgboost")
        # When the environment variable `SKIP_HYPERPARAM_TUNING` is truthy, reuse the cached
        # parameters instead of running the (time-consuming) tuning procedure again.
        if os.getenv("SKIP_HYPERPARAM_TUNING", "false").strip().lower() in {"1", "true", "yes"}:
            print("Skipping hyperparameter tuning")
            logger.info("Skipping hyperparameter tuning; using cached best parameters")
            best_params = cached_best_params
            mlflow.log_params(best_params)
            mlflow.set_tag("hpo_status", "skipped")
        else:
            print("Tuning hyperparameters")
            logger.info("Running hyperparameter tuning")
            mlflow.set_tag("hpo_status", "executed")
            search_space = get_xgb_params()
            best_params = tune_hyperparameters(x_train, y_train, search_space)
            mlflow.log_params(best_params)
        try:
            logger.info(f"Best hyperparameters: {json.dumps(best_params)}")
        except Exception:
            logger.info("Best hyperparameters selected")

        mlflow.set_tag("hpo_status", "best_params_used")

        # Enable categorical support so XGBoost accepts pandas "category" dtypes
        logger.info("Training XGBClassifier")
        xgb_model = train_xgb_classifier_task(best_params, x_train, y_train)
        logger.info("Model training complete")

        logger.info("Predicting on validation set")
        y_pred = predict_task(xgb_model, x_val)

        # Explicitly log the model, since autologging can be unreliable.
        signature = infer_signature(x_val, y_pred)
        mlflow.sklearn.log_model(                      # <-- sklearn flavor
            sk_model=xgb_model,
            artifact_path="model",
            signature=signature,
            input_example=x_val.iloc[:5],
            # serialization_format="pickle",  # optional; default is cloudpickle (still .pkl)
        )
        logger.info("Logged model artifact to MLflow")

        accuracy_metric = accuracy_score(y_val, y_pred)
        precision_metric = precision_score(y_val, y_pred)
        recall_metric = recall_score(y_val, y_pred)
        f1_score_metric = f1_score(y_val, y_pred)
        roc_auc_score_metric = roc_auc_score(y_val, xgb_model.predict_proba(x_val)[:, 1])

        mlflow.log_metric("accuracy", accuracy_metric)
        mlflow.log_metric("precision", precision_metric)
        mlflow.log_metric("recall", recall_metric)
        mlflow.log_metric("f1_score", f1_score_metric)
        mlflow.log_metric("roc_auc_score", roc_auc_score_metric)
        logger.info(
            "Validation metrics - accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, roc_auc=%.4f",
            accuracy_metric,
            precision_metric,
            recall_metric,
            f1_score_metric,
            roc_auc_score_metric,
        )
         
 
if __name__ == "__main__":
    # Check if file exists
    churn_train_path = Path(TRAINING_DATA_PATH) / "Churn_train.parquet"
    if not churn_train_path.exists():
        raise FileNotFoundError(f"Training data not found at: {churn_train_path}")
     
    train_xgb_model(str(churn_train_path), "xgboost_model", 42, 0.2)
