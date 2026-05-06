# Financial Crime Detection Dashboard

## Project Overview
A transaction monitoring prototype that detects suspicious financial activity using rule-based and ML-based approaches, presented in an interactive Streamlit dashboard. Built as a portfolio project targeting data analytics and AI roles in financial services.

## Architecture

```
financial-crime-dashboard/
├── CLAUDE.md                  # This file - project context for Claude Code
├── README.md                  # Public-facing documentation
├── requirements.txt           # Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml            # Streamlit theme and settings
├── data/
│   ├── raw/                   # Original IEEE-CIS dataset (not committed to git)
│   └── processed/             # Cleaned and feature-engineered data
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading, validation, column mapping
│   ├── feature_engine.py      # Feature engineering and derived variables
│   ├── rule_engine.py         # Rule-based detection flags
│   ├── model.py               # XGBoost training, evaluation, prediction
│   ├── explainer.py           # SHAP explanations for flagged transactions
│   └── utils.py               # Shared helpers (logging, config, formatting)
├── tests/
│   ├── test_data_loader.py
│   ├── test_rule_engine.py
│   └── test_model.py
├── notebooks/
│   └── eda.ipynb              # Exploratory data analysis
├── docs/
│   └── architecture.md        # Architecture decisions and interview prep notes
└── app.py                     # Streamlit dashboard entry point
```

## Tech Stack
- **Language:** Python 3.10+
- **Data:** Pandas, NumPy
- **ML:** Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Dashboard:** Streamlit
- **Visualization:** Plotly
- **Testing:** pytest

## Coding Standards
- Type hints on all function signatures
- Docstrings on every function (Google style)
- No hardcoded values — use constants at top of module or a config dict
- Each module should be independently testable
- Keep Streamlit UI code in app.py only — all logic lives in src/
- Use st.cache_data for expensive computations
- Handle errors gracefully with user-friendly messages in the dashboard

## Data Flow
1. User selects default dataset (IEEE-CIS) or uploads their own CSV
2. data_loader.py validates schema, maps columns if needed, returns clean DataFrame
3. feature_engine.py creates derived features (rolling averages, velocity, time features)
4. rule_engine.py applies rule-based flags (amount threshold, velocity, time anomaly)
5. model.py runs XGBoost predictions with probability scores
6. explainer.py generates SHAP values for flagged transactions
7. app.py renders dashboard with alert queue, drilldown, and summary metrics

## Default Dataset
- **Source:** IEEE-CIS Fraud Detection from Kaggle
- **URL:** https://www.kaggle.com/competitions/ieee-fraud-detection/data
- **Files needed (place in data/raw/):**
  - `train_transaction.csv` — 590K transactions with fraud labels and features
  - `train_identity.csv` — device and identity info, joins to transactions on TransactionID
  - Ignore test_transaction.csv, test_identity.csv, and sample_submission.csv (no labels)
- **Key columns:** TransactionAmt, TransactionDT, isFraud, ProductCD, card1-card6, addr1/addr2, dist1/dist2, email domain, DeviceType, DeviceInfo
- **Class imbalance:** ~3.5% fraud rate — must handle with SMOTE or class_weight
- **Data loading step:** Join train_transaction and train_identity on TransactionID (left join, not all transactions have identity info)

## User Upload Requirements
- CSV format
- Minimum required columns: transaction_id, amount, timestamp, label (0/1)
- Optional columns: merchant, category, customer_id, location
- Column mapping UI in sidebar if names don't match expected schema
- Validate: no nulls in required fields, label is binary, amount is numeric

## Rule Engine Flags
- **Amount threshold:** Transactions above 95th percentile of dataset
- **Velocity check:** More than N transactions within a rolling time window per customer
- **Time anomaly:** Transactions outside normal business hours (configurable)
- **Repeat pattern:** Same amount to same merchant in short succession
- Each flag is independent and additive — a transaction can trigger multiple flags

## Model Requirements
- XGBoost classifier with class_weight or SMOTE for imbalance
- Train/test split: 80/20 stratified by label
- Evaluation metrics: Precision, Recall, F1, AUC-ROC (not accuracy)
- Save trained model with joblib for reuse
- Retrain button in dashboard when new data is loaded

## Dashboard Layout

### Sidebar (Configuration)
- Data source toggle: default Kaggle dataset or upload CSV
- Column mapping UI if uploaded columns don't match schema
- Fraud alert threshold slider (ML probability cutoff, 0.10 to 0.90)
- Rule engine threshold controls: amount percentile, velocity max/window, night hours
- Train / Retrain Model button

### Tab 1: Overview
- Quick summary metrics row: total transactions, flagged (rules), flagged (ML), fraud recall, ROC AUC
- How to use this dashboard (numbered steps)
- How detection works: side-by-side explanation of rule engine vs ML model

### Tab 2: Model Performance
- Metrics row: precision, recall, F1, ROC AUC
- Charts: confusion matrix, ROC curve, precision-recall curve, global SHAP feature importance

### Tab 3: Visual Analytics
- Class distribution (fraud vs legitimate)
- Transaction amount distribution split by label
- Fraud rate by hour of day
- Rule engine flag breakdown (which rules flagged what)
- Feature correlation heatmap

### Tab 4: Single Transaction
- Transaction ID selector or search
- Summary card: amount, timestamp, customer, risk score
- SHAP waterfall chart for the selected transaction
- Rule flags summary showing which rules fired and why

### Tab 5: Batch Scoring
- Upload CSV for bulk scoring
- Run all transactions through rule engine and ML model
- Display summary stats: total scored, flagged count, flag rate
- Preview table of scored results
- Download button for scored CSV with risk scores and flags attached

## Build Order
Build and test each module in this sequence:
1. data_loader.py + test_data_loader.py
2. feature_engine.py
3. rule_engine.py + test_rule_engine.py
4. model.py + test_model.py
5. explainer.py
6. app.py (wire everything together)
7. README.md and docs/architecture.md

## Git Practices
- Meaningful commit messages: "feat: add rule engine with velocity check"
- One module per commit where possible
- Do not commit raw data files — add data/raw/ to .gitignore
