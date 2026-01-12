import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.core.config import PROCESSED_DATA_DIR


def scale_numeric(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_numeric: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """
    Scale numeric columns with RobustScaler.

    RobustScaler uses the median and interquartile range (IQR), making it resistant to outliers.
    """

    # Initialize the RobustScaler
    scaler = RobustScaler()

    # Fit on training data and transform all splits
    X_train[:, :n_numeric] = scaler.fit_transform(X_train[:, :n_numeric])
    X_val[:, :n_numeric] = scaler.transform(X_val[:, :n_numeric])
    X_test[:, :n_numeric] = scaler.transform(X_test[:, :n_numeric])

    return X_train, X_val, X_test, scaler


def encode_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OneHotEncoder | None]:
    """
    One-hot encode categorical features.

    The encoder is fitted only on the training set to avoid data leakage.
    """

    # Handle edge case: no categorical columns
    if len(cat_cols) == 0:
        empty_train = np.empty((len(X_train), 0))
        empty_val = np.empty((len(X_val), 0))
        empty_test = np.empty((len(X_test), 0))
        return empty_train, empty_val, empty_test, None

    # Create copies for safety
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Initialize the one-hot encoder
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    # Fit on training data and transform all splits
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_val_cat = encoder.transform(X_val[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    return X_train_cat, X_val_cat, X_test_cat, encoder


def prepare_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare final model-ready feature matrices.

    This function:
    - Separates numeric and categorical features
    - Encodes categoricals
    - Combines everything into NumPy arrays
    """

    # Create copies to avoid mutating original data
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        exclude=["int64", "float64"]
    ).columns.tolist()

    # Extract numeric features as NumPy arrays
    X_train_num = X_train[numeric_cols].to_numpy()
    X_val_num = X_val[numeric_cols].to_numpy()
    X_test_num = X_test[numeric_cols].to_numpy()

    # Encode categorical features
    X_train_cat, X_val_cat, X_test_cat, _ = encode_categoricals(
        X_train,
        X_val,
        X_test,
        categorical_cols,
    )

    # Combine numeric and categorical features
    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_val_final = np.hstack([X_val_num, X_val_cat])
    X_test_final = np.hstack([X_test_num, X_test_cat])

    return X_train_final, X_val_final, X_test_final


def save_features(X: pd.DataFrame, filename: str) -> None:
    """Save processed features to disk."""
    path = PROCESSED_DATA_DIR / filename
    X.to_parquet(path)
