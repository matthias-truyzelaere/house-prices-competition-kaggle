import numpy as np
from sklearn.metrics import root_mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return root_mean_squared_error(y_true, y_pred)
