import pandas as pd

from src.core.config import (
    PROCESSED_TEST_FILE,
    PROCESSED_TRAIN_FILE,
    TEST_FILE,
    TRAIN_FILE,
)


def load_train_data() -> pd.DataFrame:
    """Load the train set."""

    # Throw an error (FileNotFound) when the file does not exist
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"{TRAIN_FILE} not found")

    return pd.read_csv(TRAIN_FILE, encoding="utf-8")


def load_test_data() -> pd.DataFrame:
    """Load the test set."""

    # Throw an error (FileNotFound) when the file does not exist
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"{TEST_FILE} not found")

    return pd.read_csv(TEST_FILE, encoding="utf-8")


def load_processed_data(is_train: bool = True) -> pd.DataFrame:
    """
    Load the cleaned/processed data used for notebooks or separate analysis scripts.
    """

    # Define the path
    path = PROCESSED_TRAIN_FILE if is_train else PROCESSED_TEST_FILE

    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the main pipeline first!")

    return pd.read_parquet(path, engine="pyarrow")
