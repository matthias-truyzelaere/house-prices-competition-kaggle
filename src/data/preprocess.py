import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.core.config import PROCESSED_TEST_FILE, PROCESSED_TRAIN_FILE


def clean_data(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """Clean raw data by handling categorical 'NA' values, mapping ordinals and generating aggregate features."""

    # Create a copy of the provided DataFrame
    df = df.copy()

    # Outlier Removal (Training only)
    # Houses > 4000 sqft with low prices are outliers that confuse the model
    if is_training and "SalePrice" in df.columns:
        df = df.drop(df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index)

    # List of columns where NA means 'None'
    cols_with_na_as_category = [
        "Alley",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
    ]

    for col in cols_with_na_as_category:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Dictionary for Quality/Condition columns
    qual_dict = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    # Dictionary for Basement Exposure
    exp_dict = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}

    # Dictionary for Garage Finish
    fin_dict = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}

    # List of columns where the standard quality ordinal is being used
    qual_cols = [
        "ExterQual",
        "ExterCond",
        "BsmtQual",
        "BsmtCond",
        "HeatingQC",
        "KitchenQual",
        "FireplaceQu",
        "GarageQual",
        "GarageCond",
    ]

    for col in qual_cols:
        if col in df.columns:
            df[col] = df[col].map(qual_dict).fillna(0)

    # Apply specific mappings for exposure and finish
    if "BsmtExposure" in df.columns:
        df["BsmtExposure"] = df["BsmtExposure"].map(exp_dict).fillna(0)
    if "GarageFinish" in df.columns:
        df["GarageFinish"] = df["GarageFinish"].map(fin_dict).fillna(0)

    # Log-transform highly skewed features
    # Reduces the impact of very large lots on the model
    if "LotArea" in df.columns:
        df["LotArea"] = np.log1p(df["LotArea"])

    # Fill numeric NAs with 0 before engineering (specifically for area features)
    area_cols = [
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "GrLivArea",
    ]

    for col in area_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Total Square Footage
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # Total Bathrooms
    df["TotalBath"] = (
        df["FullBath"]
        + (0.5 * df["HalfBath"])
        + df["BsmtFullBath"]
        + (0.5 * df["BsmtHalfBath"])
    )

    # Age of House
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # Feature Interaction: Quality * Size
    # This captures the premium paid for high-quality large spaces
    df["QualSF"] = df["OverallQual"] * df["TotalSF"]

    return df


def split_features_target(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separates the target from the features."""

    # Drop the target from the features
    X = df.drop(columns=[target])

    # Create a DataFrame with only the target
    y = df[target]

    return X, y


def preprocess_numeric(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Impute missing values for numeric features."""

    # Create a copy for safety
    X_train = X_train.copy()
    X_val = X_val.copy()

    # Initialize the imputer with median as strategy
    imputer = SimpleImputer(strategy="median")

    # Fit on training data and transform validation split
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])

    return X_train, X_val, imputer


def save_processed_data(df: pd.DataFrame, is_train: bool = True) -> None:
    """
    Save the cleaned/processed DataFrame to the processed data directory.
    """

    # Define the path
    path = PROCESSED_TRAIN_FILE if is_train else PROCESSED_TEST_FILE

    # Ensure directory exists and save
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet for efficiency
    df.to_parquet(path, engine="pyarrow", index=False)
