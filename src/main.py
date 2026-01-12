import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from src.core.config import PROCESSED_DATA_DIR, SUBMISSION_DIR, TARGET
from src.core.logging import get_logger
from src.data.load import load_test_data, load_train_data
from src.data.preprocess import clean_data, save_processed_data
from src.models.pipeline import get_ensemble_pipeline, set_pipeline_columns
from src.models.predict import make_predictions
from src.models.train import train_model

# Initialize the logger
logger = get_logger()


def main() -> None:
    """Run the full ML pipeline using the reusable Ridge pipeline."""

    logger.info("🔎 Loading and cleaning data...")

    # Clean train set with outlier removal enabled
    train_df = clean_data(load_train_data(), is_training=True)

    # Clean test set with outlier removal disabled
    test_df = clean_data(load_test_data(), is_training=False)

    # Save processed data
    save_processed_data(train_df, is_train=True)
    save_processed_data(test_df, is_train=False)

    logger.info(f"💾 Cleaned data saved to {PROCESSED_DATA_DIR}")
    logger.info("🖖 Splitting features and target...")

    # Split features and target and drop 'Id' from features because it's just a sequence number
    X = train_df.drop(columns=[TARGET, "Id"])
    y = train_df[TARGET]
    X_test = test_df.drop(columns=["Id"], errors="ignore")

    logger.info("🧱 Building pipeline...")

    # Initialize pipeline
    pipeline = get_ensemble_pipeline()
    pipeline = set_pipeline_columns(pipeline, X, y)

    logger.info("🏃 Running Cross-Validation...")

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Calculate RMSE (Root Mean Squared Error)
    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=kf,
    )

    # Calculate RMSLE (Root Mean Squared Log Error)
    cv_rmsle_scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="neg_mean_squared_log_error",
        cv=kf,
    )

    # Calculate the average RMSE (Root Mean Squared Error)
    avg_rmse = -cv_scores.mean()

    # Calculate the average RMSLE (Root Mean Squared Log Error)
    avg_rmsle = np.sqrt(-cv_rmsle_scores.mean())

    logger.info(f"📊 Average CV RMSE: ${avg_rmse:.3f} (+/- ${cv_scores.std():.3f})")
    logger.info(f"🏆 Average CV RMSLE: {avg_rmsle:.5f}")
    logger.info("💪 Training final model on full dataset...")

    # Fit pipeline using our training module
    pipeline = train_model(pipeline, X, y)

    logger.info("🔮 Predicting test data...")

    # Predict test data
    test_preds = make_predictions(pipeline, X_test)

    # Ensure submission directory exists
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # Create submission DataFrame
    submission = pd.DataFrame({"Id": test_df["Id"], TARGET: test_preds})

    # Save to CSV
    output_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)

    logger.info(f"✅ Submission saved to {output_path}")


# Run the main function
if __name__ == "__main__":
    main()
