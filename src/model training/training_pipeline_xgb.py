import os
import mlflow
import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from data_utils import load_data, vectorize, split_data_randomly, load_parquet, prepare_xgb_data
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from tqdm import tqdm

# Load environment variables from .env file
project_root = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=project_root / ".env")

# Get environment variables
TRAINING_DATA_PATH = project_root / os.getenv('TRAINING_DATA_PATH', './data/Churn')

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
        # cast discrete params to int
        params = params.copy()
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])

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
    max_evals = 60
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

    

def train_xgb_model(data_path: str, model_name: str, random_state: int, test_size: float):
    churn_data = load_parquet(data_path)
    
    # Use vectorize function to properly handle categorical variables
    x_train, x_val, y_train, y_val  = prepare_xgb_data(churn_data)
    
    # best params:
    # {'colsample_bylevel': 0.75, 'colsample_bynode': 0.81, 'colsample_bytree': 0.54, 'gamma': 0.46063546324565136, 'learning_rate': 0.0064027841872601595, 'max_depth': np.int64(19), 'min_child_weight': 3.6054254230376572, 'n_estimators': np.int64(350), 'objective': 'binary:logistic', 'reg_alpha': 4.399584091428253, 'reg_lambda': 2.0034060827004825, 'subsample': 0.8}
    search_space = get_xgb_params()
    best_params = tune_hyperparameters(x_train, y_train, search_space)

if __name__ == "__main__":
    # Check if file exists
    churn_train_path = Path(TRAINING_DATA_PATH) / "Churn_train.parquet"
    if not churn_train_path.exists():
        raise FileNotFoundError(f"Training data not found at: {churn_train_path}")
    
    train_xgb_model(str(churn_train_path), "xgboost_model", 42, 0.2)
