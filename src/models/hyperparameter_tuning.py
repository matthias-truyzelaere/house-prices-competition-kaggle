from sklearn.model_selection import KFold, RandomizedSearchCV

from src.core.logging import get_logger
from src.data.load import load_train_data
from src.data.preprocess import clean_data
from src.models.pipeline import get_ensemble_pipeline, set_pipeline_columns

# Initialize the logger
logger = get_logger()


def main() -> None:
    """Perform hyperparameter tuning for the XGBoost pipeline."""

    logger.info("🔎 Loading and cleaning data for tuning...")

    # Load and clean training data
    train_df = clean_data(load_train_data(), is_training=True)
    X = train_df.drop(columns=["SalePrice", "Id"])
    y = train_df["SalePrice"]

    # Initialize the base pipeline
    pipeline = get_ensemble_pipeline()
    pipeline = set_pipeline_columns(pipeline, X, y)

    # Define the search space
    param_distributions = {
        # XGBoost specific parameters
        "model__regressor__xgb__n_estimators": [1000, 2000, 3000],
        "model__regressor__xgb__max_depth": [3, 4, 5],
        "model__regressor__xgb__learning_rate": [0.01, 0.05],
        "model__regressor__xgb__subsample": [0.6, 0.7, 0.8],
        "model__regressor__xgb__colsample_bytree": [0.4, 0.6, 0.8],
        # Ensemble Weights (Tuning the blend)
        "model__regressor__weights": [
            [0.5, 0.5],  # Equal split
            [0.6, 0.4],  # Original
            [0.7, 0.3],  # Heavy XGB
            [0.8, 0.2],  # Very Heavy XGB
            [0.4, 0.6],  # Heavy Lasso
        ],
    }

    # Initialize RandomizedSearchCV
    # Searches across 20 different random combinations using 3-fold cross-validation
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("🏃 Running Randomized Search (this may take a few minutes)...")

    # Fit on the RandomizedSearchCV
    search.fit(X, y)

    # Output results
    logger.info(f"Best RMSE: {-search.best_score_:.3f}")
    logger.info(f"Best Parameters: {search.best_params_}")


if __name__ == "__main__":
    main()
