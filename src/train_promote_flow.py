import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task

# Ensure imports work despite the space in `training` directory
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
MODEL_TRAINING_DIR = CURRENT_DIR / "training"

# Put `src/` and `src/training/` on sys.path so we can import modules easily
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(MODEL_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_TRAINING_DIR))

# Lazy imports after sys.path adjustments
from training.training_pipeline_xgb import train_xgb_model  # noqa: E402
from promote.promote_best_model import promote_best_model  # noqa: E402


@task(name="promote_best_model_task")
def promote_best_model_task() -> None:
    promote_best_model()


@flow(name="train_and_promote_flow")
def train_and_promote_flow(
    model_name: str = "xgboost_model",
    random_state: int = 42,
    test_size: float = 0.2
) -> None:
    """Run training subflow then promote the best model.

    Parameters
    ----------
    model_name: Name to register/log the model under
    random_state: Random seed for data splits/model
    test_size: Validation size used during training
    skip_hpo: If True, sets SKIP_HYPERPARAM_TUNING so training uses cached params
    """
    logger = get_run_logger()

    # Load environment variables from project-level .env
    load_dotenv(dotenv_path=CURRENT_DIR / ".env", override=True)

    # Resolve training data path (same convention as training flow's __main__)
    training_dir = os.getenv("TRAINING_DATA_PATH", "./data/Churn")
    churn_train_path = (CURRENT_DIR / training_dir / "Churn_train.parquet").resolve()

    if not churn_train_path.exists():
        raise FileNotFoundError(f"Training data not found at: {churn_train_path}")

    logger.info("Starting training subflow")
    train_xgb_model(str(churn_train_path), model_name, random_state, test_size)


    logger.info("Training complete. Promoting best model...")
    # promote_best_model_task()
    # logger.info("Promotion complete.")


if __name__ == "__main__":
    # Defaults match the training script's main
    train_and_promote_flow()
