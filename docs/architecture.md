# Architecture & Design Decisions

> Use this document as your interview prep cheat sheet. After each module is built,
> write one paragraph here in your own words explaining what it does and why.

## System Overview

An end-to-end transaction monitoring system that combines configurable rule-based flags with an XGBoost classifier to detect suspicious financial activity across ~590K IEEE-CIS transactions (3.5% fraud rate). The pipeline runs from raw CSV ingestion through feature engineering, rule evaluation, ML scoring, and SHAP explainability, all exposed in a five-tab Streamlit dashboard where analysts can investigate individual transactions, tune detection thresholds, and export batch-scored results.

## Module Breakdown

### data_loader.py
This module takes the raw data and returns a clean, uniformly-named DataFrame. The left join keeps all transactions while attaching identity columns where they exist. The validation step enforces four rules: mandatory columns exist, amount is numeric, label is strictly binary (0/1), and timestamp is parseable as datetime. If a user uploads a dataset with different column names, the module suggests mappings based on name similarity and data types.

### feature_engine.py
Takes a Validated Dataframe from the data_loader abd adds columns that give the ML model and rule engine extensive signals than just the raw transaction field alone.

Features and the fraud reasoning behind each

| Feature | How it's computed | Why it matters |
|---|---|---|
| hour_of_day, day_of_week | dt.hour, dt.dayofweek | Raw temporal inputs; model learns non-linear patterns (e.g. 3am is different from 3pm) |
| is_weekend | day_of_week >= 5 | Fraud operations teams are smaller on weekends; slower response time |
| is_night | hour >= 22 or < 6 | Victims are asleep, less likely to spot a transaction alert |
| txn_count_1h/6h/24h | Rolling count per customer | Velocity spike is one of the strongest fraud signals; a stolen card is monetised fast |
| log_amount | log1p(amount) | Transaction amounts are right-skewed; log compresses scale so a $10,000 transaction doesn't dominate by magnitude alone |
| amount_percentile | rank(pct=True) | Where this transaction sits globally; feeds the rule engine's high-amount threshold |
| amount_zscore | Per-customer deviation / std | Captures behavioural deviation; a $5,000 transaction is normal for a business, suspicious for a student |
| is_round_amount | amount % 1 == 0 | Legitimate purchases have irregular cents ($47.83); round amounts ($500.00) are disproportionately common in fraud |
| is_small_probe | amount < $10 | Marks the card-test transaction; a small charge to confirm the card is live |
| preceded_by_probe | Prior small transaction by same customer in last 24h | Marks the follow-up cash-out; combined with high amount, this is the card-testing pattern |

The velocity algorithm - for each transaction, count how many times that customer transacted in the last N hours.

Because pandas groupby().rolling() returns results grouped by customer rather than in global row order, using .values directly would silently assign counts to the wrong transactions. The fix: sort the entire DataFrame by timestamp first, compute per-customer rolling counts using np.searchsorted (find the leftmost row in each customer's group that falls within the time window, then count = current position − that boundary + 1), and store results into a pre-allocated array using the group's positional indices. To restore the original row order at the end, argsort(argsort(sort_order)) produces the inverse permutation — a standard numpy trick. The whole operation is O(n log n) and fully vectorised with no Python inner loop.

### rule_engine.py

Four rules and one conditional flag were implemented.

1. Flag high amount: Flags transactions above a configurable percentile threshold (default 95th percentile).

2. Flag velocity: Flags any transaction where the customer exceeded the maximum number of transactions (5) within the rolling window (default 1 hour).

3. Flag time anomaly: Flags transactions between 10pm and 6am.

4. Flag repeat pattern: Flags any transaction where the same amount appeared at the same merchant within 30 minutes, in either direction. Both the original and the repeat are flagged, not just the second occurrence.

5. Flag probe (conditional): Not a standalone function. apply_all_rules checks whether preceded_by_probe exists in the DataFrame (created by feature_engine) and promotes it directly to a flag column. This reuses the card-testing detection without duplicating logic.

Each rule runs independently and contributes a boolean flag. A combined rule_score (sum of all flags triggered) determines whether a transaction goes to an analyst queue, triggers an automatic decline, or passes through.

### model.py
Why XGBoost?
XGBoost handles missing values natively, is robust to the mixed feature types in transaction data, and supports asymmetric cost weighting for class imbalance. It is also the only common model family where SHAP produces exact rather than approximate explanations, which matters because fraud decisions need to be auditable.

How was class imbalance handled?
SMOTE generates synthetic fraud samples by interpolating between real ones rather than duplicating them. For each fraud transaction it finds k nearest neighbours and creates new points along the lines connecting them, giving the model a richer view of the fraud region. It runs on the training set only. Applying it to the test set would inflate recall artificially.

When fewer than 6 fraud samples exist in training, the code falls back to XGBoost's scale_pos_weight, which tells the model each fraud case counts roughly 28x more than a legitimate one. Together they ensure fraud is treated as a first-class signal rather than statistical noise.

Why not accuracy?
The dataset is 96.5% legitimate. A model that predicts "not fraud" for everything scores 96.5% accuracy but catches zero fraud. Evaluation uses precision, recall, F1, and AUC-ROC instead. Recall is prioritized because a missed fraud case (false negative) has a higher cost than a false alarm (false positive).

I added that last section because interviewers almost always ask why you didn't use accuracy. Having it ready saves you from stumbling on a question that sounds simple but catches a lot of candidates off guard.

### explainer.py
Why SHAP over LIME?
SHAP computes exact Shapley values for tree models. Run it twice and you get the same answer. LIME resamples randomly and can produce different explanations each time. SHAP satisfies consistency, and the same values power both per-transaction waterfall charts and global feature importance. LIME is purely local with no way to aggregate across transactions.
Performance also matters. TreeExplainer runs in O(TLD) time, making batch explanations feasible for a dashboard. LIME refits a new linear model per transaction, which is too slow when you need to explain hundreds of flagged events.
In short: SHAP is exact, consistent, fast, and works at both local and global level. LIME is none of those.

What does a waterfall chart show an analyst?
It decomposes a single fraud score into every feature's contribution. Red bars push the score toward fraud, blue bars push toward legitimate, and all contributions sum exactly to the final score.
The starting point is the base value, which is the model's prior fraud probability before seeing this specific transaction. From there, each feature nudges the score up or down based on what the model learned.
The key insight is the dominant driver. A velocity-led score points to a card-testing burst. An amount-led score points to a single large cash-out. These are different fraud typologies requiring different investigations. It also satisfies the regulatory expectation that automated decisions be explainable, giving analysts something auditable to put in a case file.

### app.py
app.py is the only file that contains Streamlit UI code. All business logic stays in src/. It is structured around a persistent sidebar and five tabs.

**Sidebar**
The sidebar holds all configuration controls. The user chooses between the default IEEE-CIS dataset and uploading their own CSV. If a CSV is uploaded, the sidebar runs get_column_mapping_suggestions() and renders one selectbox per required column so the user can confirm or correct the fuzzy-matched mapping before loading. Below that are a fraud alert threshold slider (the ML probability cutoff), rule engine threshold controls (amount percentile, velocity window, night hours), and a Train / Retrain Model button. Every sidebar value is returned in a config dict that gets passed into the tab renderers, so the sidebar is the single source of truth for all runtime parameters.

**Data flow on every rerun**
Streamlit reruns the entire script on every user interaction. Three @st.cache_data functions absorb the expensive parts: _cached_load_default (disk I/O + join), _cached_engineer (feature engineering across 590K rows), and _cached_apply_rules (rule flags, keyed by a JSON-serialised thresholds string so changing a sidebar slider invalidates only the rules cache, not the load or engineering cache). Trained artifacts — model, explainer, metrics, risk scores — live in st.session_state so they survive reruns without being re-trained.

**Tab 1 — Overview**
Shows five summary metrics: total transactions, transactions flagged by rules, transactions flagged by ML (using the sidebar threshold), fraud recall, and ROC AUC. These update live as the threshold slider moves. Below the metrics is a usage guide and a side-by-side explanation of the rule engine versus the ML model, written for a non-technical stakeholder audience.

**Tab 2 — Model Performance**
Displays four evaluation charts using Plotly: a confusion matrix heatmap (px.imshow), a ROC curve, a precision-recall curve, and a horizontal bar chart of global SHAP feature importance (mean |SHAP| across a 500-row background sample). All four are gated behind a "train model first" message until the user clicks the train button.

**Tab 3 — Visual Analytics**
Five data exploration charts: class distribution (fraud vs legitimate counts), transaction amount histogram split by label with a log y-axis, fraud rate by hour of day, rule engine flag breakdown (how many transactions each individual rule flagged), and a feature correlation heatmap of the top 12 engineered features. These populate as soon as data is loaded, with no model required.

**Tab 4 — Single Transaction**
The analyst investigation view. A selectbox is pre-populated with the top 200 highest-risk flagged transactions (sorted by risk score) so it renders instantly regardless of how many transactions were flagged in total. Selecting a transaction shows a summary card on the left (amount, timestamp, risk score, rule score) and a SHAP waterfall chart on the right. The waterfall is built as a custom Plotly horizontal bar chart — red bars push the score toward fraud, blue push toward legitimate, sorted by absolute SHAP value with the strongest driver at the top. SHAP computations are cached in st.session_state keyed by (transaction_id, explainer_id) so switching between previously viewed transactions is instant.

**Tab 5 — Batch Scoring**
Accepts a CSV upload separate from the main data source. After column mapping confirmation, clicking Run batch scoring runs the full pipeline: engineer_features → apply_all_rules → predict_risk_scores. Any feature columns the model expects that are absent from the uploaded file are filled with NaN (XGBoost handles missing values natively) so the tab works with both full IEEE-CIS format files and minimal 4-column CSVs. Results show summary stats (total scored, flagged by rules, flagged by ML, flag rate) and a preview table, with a download button for the full scored CSV.

## Interview Q&A Prep

**Q: Walk me through your project.**
A: _TODO: Write your 60-second pitch here._

**Q: Why did you combine rules and ML?**
A: _TODO_

**Q: How did you handle the class imbalance?**
A: _TODO_

**Q: What would you do differently with more time?**
A: _TODO_

**Q: How would this work in production?**
A: _TODO_
