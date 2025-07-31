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
