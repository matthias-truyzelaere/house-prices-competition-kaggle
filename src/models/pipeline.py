import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def get_ensemble_pipeline() -> Pipeline:
    """Returns an Ensemble pipeline combining XGBoost and Lasso."""

    # Numeric preprocessing pipeline
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # Categorical preprocessing pipeline
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # ColumnTransformer placeholder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, []),
            ("cat", categorical_pipeline, []),
        ]
    )

    # XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.4,
        min_child_weight=1,
        random_state=42,
        objective="reg:squarederror",
    )

    # Lasso Regressor (Linear)
    # LassoCV automatically tunes the 'alpha' (regularization strength)
    lasso_reg = LassoCV(
        alphas=[0.0001, 0.0005, 0.001, 0.01],
        cv=5,
        max_iter=10000,
        random_state=42,
    )

    # Voting Regressor (The Ensemble)
    # We give XGBoost slightly more weight as it is usually more powerful
    ensemble = VotingRegressor(
        estimators=[
            ("xgb", xgb_reg),
            ("lasso", lasso_reg),
        ],
        weights=[0.4, 0.6],
    )

    # Wrap the entire ensemble in a Log-Target transformation
    model = TransformedTargetRegressor(
        regressor=ensemble,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def set_pipeline_columns(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Dynamically set numeric and categorical columns in the pipeline.

    Drops numeric features with weak correlation (corr < 0.1).
    """

    # Select all numeric columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Select all categorical columns
    categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # Drop numeric features with weak correlation (corr < 0.1)
    strong_numeric_cols = [col for col in numeric_cols if abs(X[col].corr(y)) > 0.1]

    # Update ColumnTransformer
    pipeline.named_steps["preprocessor"].transformers[0] = (
        "num",
        pipeline.named_steps["preprocessor"].transformers[0][1],
        strong_numeric_cols,
    )
    pipeline.named_steps["preprocessor"].transformers[1] = (
        "cat",
        pipeline.named_steps["preprocessor"].transformers[1][1],
        categorical_cols,
    )

    return pipeline
