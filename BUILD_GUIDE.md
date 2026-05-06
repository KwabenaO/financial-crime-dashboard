# Build Guide

Step-by-step instructions for building this project using Claude Code.
Run `claude` in the project root folder. It will read CLAUDE.md automatically.

## Module 1: Data Loader

Prompt Claude Code with:
```
Implement data_loader.py. Load the two IEEE-CIS CSVs (train_transaction.csv and train_identity.csv) from data/raw/, join them on TransactionID with a left join, and build the validate_schema, load_user_dataset, and get_column_mapping_suggestions functions.
```

After it finishes:
- Ask: "Explain data_loader.py to me as if I'm presenting it in an interview."
- Read the code and make sure you understand the join and validation logic.
- Open docs/architecture.md and fill in the data_loader.py section in your own words.

## Module 2: Feature Engineering

Prompt Claude Code with:
```
Implement feature_engine.py. Build the time features, velocity features, and amount features. Make sure engineer_features calls all three functions and returns the full DataFrame.
```

After it finishes:
- Ask: "Explain what each derived feature means and why it matters for fraud detection."
- Try adding one new feature yourself (for example, a flag for round-number amounts).
- Fill in the feature_engine.py section in docs/architecture.md.

## Module 3: Rule Engine

Prompt Claude Code with:
```
Implement rule_engine.py. Build all four rule functions (amount threshold, velocity, time anomaly, repeat pattern) and the apply_all_rules function that combines them into a rule_score column.
```

After it finishes:
- Ask: "Walk me through how apply_all_rules works and how the rule_score is calculated."
- Try changing a threshold (for example, set amount percentile to 0.90) and see how the flag count changes.
- Fill in the rule_engine.py section in docs/architecture.md.

## Module 4: ML Model

Prompt Claude Code with:
```
Implement model.py. Train an XGBoost classifier with SMOTE for class imbalance. Use stratified train/test split. Evaluate with precision, recall, F1, and AUC-ROC. Include save and load functions using joblib.
```

After it finishes:
- Ask: "Why did we use XGBoost instead of Random Forest? Why SMOTE instead of undersampling? Why not just use accuracy?"
- Look at the evaluation metrics. If anything is 1.000 across the board, something is wrong.
- Fill in the model.py section in docs/architecture.md.

## Module 5: SHAP Explainer

Prompt Claude Code with:
```
Implement explainer.py. Create a TreeExplainer for the trained XGBoost model. Build the explain_transaction function for individual SHAP waterfall plots and get_global_importance for the summary bar chart.
```

After it finishes:
- Ask: "Explain how SHAP values work and what a waterfall chart tells an analyst."
- Fill in the explainer.py section in docs/architecture.md.

## Module 6: Wire Up the Dashboard

Prompt Claude Code with:
```
Wire everything together in app.py. Connect all five tabs to the src modules. The sidebar config values should pass through to the rule engine and model. Use st.cache_data for expensive operations. Make sure all charts render with Plotly.
```

After it finishes:
- Run `streamlit run app.py` and click through all five tabs.
- Test the sidebar controls: change the threshold, switch data sources, retrain the model.
- Take a screenshot for your portfolio.

## Module 7: Tests

Prompt Claude Code with:
```
Implement the test files in tests/. Write real test cases for data validation, rule engine flags, and model output. Use small sample DataFrames, not the full dataset.
```

After it finishes:
- Run `pytest` and make sure all tests pass.
- Try breaking something on purpose (change a column name) and confirm the tests catch it.

## Module 8: Final Polish

Prompt Claude Code with:
```
Review the full project. Clean up any unused imports, add any missing docstrings, and make sure the README is accurate. Generate a clean commit history.
```

Then:
- Fill in the interview Q&A section in docs/architecture.md.
- Push to GitHub.
- Deploy to Streamlit Community Cloud.
- Add the project card to your portfolio site.
