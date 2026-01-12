# 🏡 House Prices: Advanced Regression Techniques

A modular, professional machine learning pipeline designed for the Kaggle House Prices competition. This project achieves an elite-tier score by blending gradient boosting and linear regression.

## 🚀 Highlights

- **Ensemble Modeling:** Hybrid `VotingRegressor` combining XGBoost (40%) and Lasso (60%).
- **Advanced Preprocessing:** Custom logic for handling ordinal categories and "NA" as a feature.
- **Robust Validation:** 5-fold Cross-Validation with stable performance metrics.
- **Modern Tooling:** Built using `uv` for lightning-fast dependency management.

## 📁 Project Structure

```text
.
├── data/               # Raw, processed, and submission files
├── src/
│   ├── core/           # Configuration and logging
│   ├── data/           # Loading and preprocessing logic
│   ├── models/         # Training, tuning, and prediction modules
│   └── main.py         # The orchestrator of the pipeline
```

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv). To set up the environment:

```bash
uv sync
```

## 🏋️ Usage

### 1. Hyperparameter Tuning

To find the optimal settings for the XGBoost model and ensemble weights:

```bash
uv run python -m src.models.hyperparameter_tuning
```

### 2. Main Pipeline

To clean the data, run cross-validation, and generate a submission file:

```bash
uv run python -m src.main
```

## 📊 Final Results

| Metric             | Local CV Value |
| :----------------- | :------------- |
| **RMSE**           | $20,500        |
| **RMSLE (Kaggle)** | 0.11276        |

**Status:** Currently ranked #2089 on the Public Leaderboard (~Top 15%).

## 🖖 Methodology

- **Outlier Removal:** Houses > 4000 sqft with anomalous pricing were removed.
- **Feature Engineering:** Log-transformations of skewed features (`LotArea`) and creation of interaction terms (`Quality * SF`).
- **Ensemble Logic:** Combined the non-linear flexibility of XGBoost with the conservative stability of Lasso Regression.
