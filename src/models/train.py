import pandas as pd
from sklearn.base import RegressorMixin


def train_model(model: RegressorMixin, X: pd.DataFrame, y: pd.Series) -> RegressorMixin:
    """Fits the provided model or pipeline to the training data."""

    # Fit the model/pipeline
    model.fit(X, y)

    return model
