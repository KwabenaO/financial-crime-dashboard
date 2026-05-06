# Financial Crime Detection Dashboard

An interactive transaction monitoring system that detects suspicious financial activity using **rule-based flags** and **XGBoost classification**, with SHAP explainability for every flagged transaction.

## Background and Overview

Transaction monitoring is a core function in financial institutions. Compliance teams rely on a combination of business rules and machine learning to flag suspicious activity and reduce false positives. This project demonstrates how both approaches work together in a real detection pipeline.

The dashboard uses the IEEE-CIS Fraud Detection dataset from Kaggle as its default data source. Users can also upload their own CSV files with automatic column mapping to test the pipeline on different data.

## Features

- **Dual detection approach**: configurable rule-based business logic flags plus XGBoost ML predictions working together
- **SHAP explainability**: every flagged transaction includes a waterfall chart showing which features drove the prediction
- **Flexible data input**: ships with the IEEE-CIS Fraud Detection dataset and supports user-uploaded CSVs with automatic column mapping
- **Configurable thresholds**: adjust rule parameters and ML probability cutoff via the dashboard sidebar
- **Five dashboard views**: Overview, Model Performance, Visual Analytics, Single Transaction investigation, and Batch Scoring
- **Batch scoring**: upload a full CSV, score all transactions, and download results with risk scores and flags attached

## Architecture

```
app.py (Streamlit UI — 5 tabs + sidebar)
  ├── src/data_loader.py    → Load and validate datasets, column mapping
  ├── src/feature_engine.py → Derive features (velocity, time, amount stats)
  ├── src/rule_engine.py    → Apply configurable rule-based flags
  ├── src/model.py          → Train/evaluate XGBoost with class imbalance handling
  ├── src/explainer.py      → Generate SHAP explanations per transaction
  └── src/utils.py          → Shared config and helpers
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/financial-crime-dashboard.git
cd financial-crime-dashboard

# Install dependencies
pip install -r requirements.txt

# Download the IEEE-CIS dataset from Kaggle and place two files in data/raw/
# https://www.kaggle.com/competitions/ieee-fraud-detection/data
# Only need: train_transaction.csv and train_identity.csv

# Run the dashboard
streamlit run app.py
```

## Dataset

**Default:** IEEE-CIS Fraud Detection dataset (~590K transactions, ~3.5% fraud rate). Requires two files in `data/raw/`:

| File | Description |
|---|---|
| train_transaction.csv | Transactions with fraud labels, amounts, card info, email domains |
| train_identity.csv | Device type, device info, and identity features (joins on TransactionID) |

The pipeline joins both files on TransactionID using a left join, since not every transaction has identity info. You can ignore the test files and sample_submission.csv from the Kaggle download.

**Custom:** Upload any CSV with these minimum columns:
| Column | Type | Description |
|---|---|---|
| transaction_id | string/int | Unique identifier |
| amount | float | Transaction amount |
| timestamp | datetime | Transaction time |
| label | int (0/1) | Fraud label |

Optional columns (merchant, category, customer_id) unlock additional rule-based checks.

## Design Decisions

- **XGBoost over Random Forest or deep learning**: gradient boosting handles tabular fraud data well and is simpler to explain in a business context. Random Forest is also a strong choice but XGBoost tends to perform better on structured data with class imbalance.
- **SMOTE for class imbalance**: preserves all real fraud cases rather than undersampling the majority class. This matters because fraud examples are scarce and losing any reduces the model's ability to learn fraud patterns.
- **Rule engine plus ML**: mirrors how real AML systems work in banks. Business rules catch known patterns (structuring, velocity, time anomalies) while the ML model catches novel patterns the rules miss.
- **SHAP over LIME**: TreeExplainer is exact for tree-based models and faster than perturbation-based methods. It also provides consistent explanations across runs.
- **Realistic metrics over perfect scores**: a fraud detection model that scores 1.000 on all metrics is almost certainly overfitting or leaking data. Our dashboard shows honest numbers and explains why imperfect scores are expected.
- **Streamlit**: fast prototyping for dashboard interfaces and deploys free on Streamlit Community Cloud.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn · SHAP · Streamlit · Plotly
