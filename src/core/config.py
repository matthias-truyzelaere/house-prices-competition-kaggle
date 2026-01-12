from pathlib import Path

# Define the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define the directories where data is being stored
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SUBMISSION_DIR = DATA_DIR / "submissions"

# Define the raw train and test path
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"

# Define the processed paths
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train_cleaned.parquet"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test_cleaned.parquet"

# Define the target - what you want to predict
TARGET = "SalePrice"
