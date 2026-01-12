import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


def make_predictions(model: RegressorMixin, X: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions using the trained model.

    Note: The pipeline handles inverse log-transformation automatically.
    """

    # Generate predictions
    return model.predict(X)
