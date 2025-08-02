import os
import mlflow
import xgboost
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from data_utils import load_data, vectorize, split_data_randomly, load_parquet
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

# Load environment variables from .env file
project_root = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=project_root / ".env")

# Get environment variables
TRAINING_DATA_PATH = os.getenv('TRAINING_DATA_PATH', './data/Churn')
# MODEL_OUTPUT_PATH = os.getenv('MODEL_OUTPUT_PATH', 'models/')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'test-experiment')
# MODEL_NAME = os.getenv('MODEL_NAME', 'xgboost_model')
# RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))
# TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))

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

def turn_hyperparameters(data: pd.DataFrame, search_space: dict):
    def xgb_objective(params: Dict) -> float:
        pass 

    best = fmin(
        fn=xgb_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        rstate=np.random.default_rng(42),
    )

    return best

    

def train_xgb_model(data_path: str, model_name: str, random_state: int, test_size: float):
    churn_data = load_parquet(data_path)
    X_train, X_val, y_train, y_val = split_data_randomly(churn_data, "Churn", test_size=test_size, seed=random_state)

    # check if the data is load and vectorize correctly
    assert X_train is not None and X_val is not None, "Training and validation features are not loaded correctly."
    assert y_train is not None and y_val is not None, "Training and validation labels are not loaded correctly."
    #assert X_train.shape[0] == y_train.shape[0], "Mismatch in number of training samples and labels."
    #assert X_val.shape[0] == y_val.shape[0], "Mismatch in number of validation samples and labels."
    print("Data loaded and vectorized correctly.")

    print(X_train.shape[0], X_val.shape[0])

    # search_space = get_xgb_params()
    # best_params = turn_hyperparameters(churn_data, search_space)

if __name__ == "__main__":
    # Check if file exists
    train_data_path = Path(TRAINING_DATA_PATH) / "Churn_train.parquet"
    print(Path(TRAINING_DATA_PATH))
    print(train_data_path)
    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found at: {train_data_path}")
    
    # train_xgb_model(str(train_data_path), "xgboost_model", 42, 0.2)
