import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

def load_data(path: str):
    """Load data from CSV file"""
    df = pd.read_csv(path)
    return df

def load_parquet(path: str):
    """Load data from Parquet file"""
    df = pd.read_parquet(path)
    return df

def split_data_randomly(
        df, 
        target_col: str = "Churn", 
        test_size: float = 0.2, 
        seed: int = 42
    ):  
    """
    Split data randomly into training and validation sets
    """

    y_values = df[target_col].values
    x_values = df.drop(columns=[target_col], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x_values, y_values, test_size=test_size, random_state=seed
    )
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def vectorize(df, target_col: str = "Churn"):
    """
    Returns X_train, X_val, y_train, y_val, DictVectorizer
    """
    y = df[target_col].values
    dicts = df.drop(columns=[target_col]).to_dict(orient="records")

    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(dicts)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_val, y_train, y_val, dv

def prepare_xgb_data(df, target_col: str = "Churn"):
    """
    Prepare data for XGBoost
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.to_list()
    categorical_cols = df.select_dtypes(include=["object"]).columns.to_list()

    # Convert categorical columns to category dtype
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    split = split_data_randomly(df)

    x_train, x_val, y_train, y_val = split["x_train"], split["x_test"], split["y_train"], split["y_test"]

    return x_train, x_val, y_train, y_val
    