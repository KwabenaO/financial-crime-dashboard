"""
Financial Crime Detection Dashboard — Streamlit Entry Point.

This is the only file that contains Streamlit UI code.
All business logic lives in src/ modules.

Run with: streamlit run app.py
"""

import json
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data_loader import (
    REQUIRED_COLUMNS,
    get_column_mapping_suggestions,
    load_default_dataset,
    load_user_dataset,
    validate_schema,
)
from src.explainer import create_explainer, explain_transaction, get_global_importance
from src.feature_engine import engineer_features
from src.model import evaluate_model, predict_risk_scores, train_model
from src.rule_engine import apply_all_rules
from src.utils import format_currency, format_percentage, setup_logging

setup_logging()
logger = logging.getLogger("fraud_dashboard")

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Financial Crime Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached pipeline helpers
# ---------------------------------------------------------------------------

@st.cache_data
def _cached_load_default(path: str) -> pd.DataFrame:
    """Load the default IEEE-CIS dataset, cached across reruns."""
    return load_default_dataset(path)


@st.cache_data
def _cached_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering, cached so reruns skip the 590K-row computation."""
    return engineer_features(df)


@st.cache_data
def _cached_apply_rules(df: pd.DataFrame, thresholds_json: str) -> pd.DataFrame:
    """Apply rule flags with a JSON-serialised thresholds key for cache invalidation.

    Args:
        df: Feature-engineered DataFrame.
        thresholds_json: JSON string of the thresholds dict. Changing any
            threshold value produces a different key, invalidating only this
            cache layer without clearing the load or engineer caches.

    Returns:
        DataFrame with flag columns and rule_score attached.
    """
    thresholds = json.loads(thresholds_json)
    return apply_all_rules(df, thresholds)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_feature_cols(df: pd.DataFrame) -> list:
    """Return numeric columns suitable for model training."""
    exclude = {"label", "transaction_id", "rule_score"}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        c for c in numeric
        if c not in exclude and not c.startswith("flag_")
    ]


def _build_thresholds(config: dict) -> dict:
    """Map sidebar config values to the rule engine's DEFAULT_THRESHOLDS key names.

    Args:
        config: Dict returned by render_sidebar().

    Returns:
        Dict suitable for passing to apply_all_rules() as the thresholds argument.
    """
    return {
        "amount_percentile": config["amount_percentile"],
        "velocity_max_txns": int(config["velocity_max"]),
        "velocity_window_hours": int(config["velocity_window"]),
        "night_start_hour": int(config["night_start"]),
        "night_end_hour": int(config["night_end"]),
    }


def _run_training(df: pd.DataFrame, config: dict) -> None:
    """Train the XGBoost model and store all artifacts in st.session_state.

    Runs train_model → evaluate_model → create_explainer → predict_risk_scores
    inside a spinner. All outputs are written to session_state so they survive
    subsequent Streamlit reruns without retraining.

    Args:
        df: Fully processed DataFrame (features + rule flags).
        config: Sidebar config dict (currently unused but kept for forward
            compatibility if per-run hyperparameter overrides are added).
    """
    feature_cols = _get_feature_cols(df)
    with st.spinner("Training XGBoost model…"):
        model, X_test, y_test, feature_cols = train_model(
            df, feature_cols, use_smote=True
        )
        metrics = evaluate_model(model, X_test, y_test)
        y_scores = model.predict_proba(X_test)[:, 1]

        background = df[feature_cols].sample(
            min(500, len(df)), random_state=42
        )
        explainer = create_explainer(model, background)

        risk_scores = predict_risk_scores(model, df, feature_cols)

    st.session_state["model"] = model
    st.session_state["metrics"] = metrics
    st.session_state["feature_cols"] = feature_cols
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["y_scores"] = y_scores
    st.session_state["explainer"] = explainer
    st.session_state["risk_scores"] = risk_scores
    st.success("Model trained successfully.")


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Render a 2×2 confusion matrix as an annotated Plotly heatmap.

    Args:
        cm: 2×2 numpy array [[TN, FP], [FN, TP]] from evaluate_model().

    Returns:
        Plotly Figure.
    """
    labels = ["Legitimate", "Fraud"]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual"},
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
    return fig


def _plot_roc_curve(y_test, y_scores) -> go.Figure:
    """Plot a ROC curve with a random-classifier diagonal for reference.

    Args:
        y_test: True binary labels.
        y_scores: Predicted fraud probabilities.

    Returns:
        Plotly Figure.
    """
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="grey"), name="Random"
        )
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        showlegend=False,
    )
    return fig


def _plot_pr_curve(y_test, y_scores) -> go.Figure:
    """Plot a precision-recall curve.

    Preferred over the ROC curve for imbalanced datasets because it focuses
    on the minority (fraud) class rather than the majority class.

    Args:
        y_test: True binary labels.
        y_scores: Predicted fraud probabilities.

    Returns:
        Plotly Figure.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
    )
    return fig


def _plot_shap_importance(explainer, df: pd.DataFrame, feature_cols: list) -> go.Figure:
    """Plot global SHAP feature importance as a horizontal bar chart.

    Computes mean |SHAP| over a 500-row sample of the dataset. Sorted from
    highest to lowest importance with the most important feature at the top.

    Args:
        explainer: SHAP TreeExplainer instance.
        df: Full processed DataFrame (a random sample is taken internally).
        feature_cols: Feature columns to include.

    Returns:
        Plotly Figure.
    """
    sample = df[feature_cols].sample(min(500, len(df)), random_state=42)
    importance_df = get_global_importance(explainer, sample, max_features=15)
    fig = px.bar(
        importance_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Mean |SHAP|", "feature": ""},
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
    return fig


def _plot_shap_waterfall(shap_exp) -> go.Figure:
    """Custom Plotly waterfall for a single SHAP Explanation row."""
    values = shap_exp.values
    feature_names = shap_exp.feature_names if shap_exp.feature_names is not None else [
        f"f{i}" for i in range(len(values))
    ]
    base_value = float(shap_exp.base_values)

    # Sort by absolute SHAP value, top 15
    order = np.argsort(np.abs(values))[::-1][:15]
    sorted_values = values[order]
    sorted_names = [feature_names[i] for i in order]
    sorted_display = [shap_exp.data[i] for i in order]

    colors = ["#d73027" if v > 0 else "#4575b4" for v in sorted_values]
    labels = [
        f"{name}={val:.3g}" for name, val in zip(sorted_names, sorted_display)
    ]

    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in sorted_values],
        textposition="auto",
    ))
    fig.update_layout(
        xaxis_title="SHAP value (impact on fraud score)",
        xaxis=dict(title_standoff=12),
        yaxis=dict(automargin=True),
        annotations=[
            dict(
                x=0, y=1.05, xref="paper", yref="paper",
                text=f"Base value: {base_value:.3f}",
                showarrow=False, font=dict(size=11),
            )
        ],
        margin=dict(t=40, b=50, l=10, r=20),
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render the sidebar with configuration controls."""

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.caption(
            "Select your data source, adjust detection thresholds, "
            "and optionally upload your own dataset."
        )

        # Data source selection
        st.subheader("Choose data source")
        data_source = st.radio(
            "Data source:",
            options=["Use default Kaggle dataset", "Upload my own CSV"],
            label_visibility="collapsed",
        )

        uploaded_file = None
        column_mapping: dict = {}

        if data_source == "Upload my own CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                try:
                    raw_preview = pd.read_csv(uploaded_file)
                    uploaded_file.seek(0)
                    suggestions = get_column_mapping_suggestions(raw_preview)
                    st.markdown("**Column mapping**")
                    st.caption("Confirm or adjust how your columns map to the required schema.")
                    for req_col in REQUIRED_COLUMNS:
                        default_idx = 0
                        all_cols = list(raw_preview.columns)
                        if req_col in suggestions and suggestions[req_col] in all_cols:
                            default_idx = all_cols.index(suggestions[req_col])
                        chosen = st.selectbox(
                            f"`{req_col}` ←",
                            options=all_cols,
                            index=default_idx,
                            key=f"map_{req_col}",
                        )
                        column_mapping[chosen] = req_col
                except Exception as exc:
                    st.error(f"Could not read file: {exc}")

        st.divider()

        # Fraud alert threshold
        st.subheader("Fraud alert threshold")
        fraud_threshold = st.slider(
            "Fraud alert threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            label_visibility="collapsed",
            help="Probability cutoff for flagging a transaction as suspicious.",
        )
        st.caption(f"Current threshold: **{fraud_threshold:.2f}**")

        st.divider()

        # Rule engine thresholds
        st.subheader("Rule engine thresholds")
        amount_percentile = st.slider(
            "Amount percentile flag",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Transactions above this percentile are flagged as high-amount.",
        )
        velocity_max = st.number_input(
            "Max transactions per window",
            min_value=2,
            max_value=20,
            value=5,
        )
        velocity_window = st.number_input(
            "Velocity window (hours)",
            min_value=1,
            max_value=24,
            value=1,
        )
        night_start = st.number_input(
            "Night hours start (24h)",
            min_value=18,
            max_value=23,
            value=22,
        )
        night_end = st.number_input(
            "Night hours end (24h)",
            min_value=1,
            max_value=8,
            value=6,
        )

        st.divider()

        train_button = st.button(
            "Train / Retrain Model", type="primary", use_container_width=True
        )

    return {
        "data_source": data_source,
        "uploaded_file": uploaded_file,
        "column_mapping": column_mapping,
        "fraud_threshold": fraud_threshold,
        "amount_percentile": amount_percentile,
        "velocity_max": velocity_max,
        "velocity_window": velocity_window,
        "night_start": night_start,
        "night_end": night_end,
        "train_button": train_button,
    }


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def render_overview(df: pd.DataFrame | None, config: dict) -> None:
    """Tab 1: Quick summary metrics and usage guide."""
    st.subheader("📋 Quick summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = st.session_state.get("metrics")
    risk_scores = st.session_state.get("risk_scores")
    threshold = config["fraud_threshold"]

    with col1:
        st.metric("Total transactions", f"{len(df):,}" if df is not None else "—")
    with col2:
        if df is not None and "rule_score" in df.columns:
            flagged_rules = int((df["rule_score"] > 0).sum())
            st.metric("Flagged (rules)", f"{flagged_rules:,}")
        else:
            st.metric("Flagged (rules)", "—")
    with col3:
        if risk_scores is not None:
            flagged_ml = int((risk_scores >= threshold).sum())
            st.metric("Flagged (ML)", f"{flagged_ml:,}")
        else:
            st.metric("Flagged (ML)", "—")
    with col4:
        if metrics:
            st.metric("Fraud recall", format_percentage(metrics["recall"]))
        else:
            st.metric("Fraud recall", "—")
    with col5:
        if metrics:
            st.metric("ROC AUC", f"{metrics['auc_roc']:.3f}")
        else:
            st.metric("ROC AUC", "—")

    st.divider()

    st.subheader("How to use this dashboard")
    st.markdown(
        """
        1. Choose your data source and adjust thresholds in the left sidebar.
        2. Click **Train / Retrain Model** to fit the XGBoost classifier on the loaded data.
        3. Review overall performance under the **Model Performance** tab.
        4. Explore fraud patterns and feature distributions under **Visual Analytics**.
        5. Investigate individual transactions with SHAP explanations under **Single Transaction**.
        6. Score an entire uploaded dataset and download results under **Batch Scoring**.
        """
    )

    st.divider()

    st.subheader("How detection works")
    col_rules, col_ml = st.columns(2)
    with col_rules:
        st.markdown("**Rule engine**")
        st.markdown(
            "Configurable business rules that flag transactions based on "
            "amount thresholds, transaction velocity, time-of-day anomalies, "
            "and repeat patterns. These mirror the rule-based checks used by "
            "compliance teams in financial institutions."
        )
    with col_ml:
        st.markdown("**ML model (XGBoost)**")
        st.markdown(
            "A gradient boosted classifier trained on labeled transaction data. "
            "Handles class imbalance with SMOTE resampling. Every prediction "
            "includes a SHAP explanation showing which features drove the risk score."
        )


def render_model_performance() -> None:
    """Tab 2: Model evaluation metrics, confusion matrix, ROC/PR curves, SHAP importance."""
    st.subheader("📊 Model evaluation")

    metrics = st.session_state.get("metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Precision", f"{metrics['precision']:.3f}" if metrics else "—")
    with col2:
        st.metric("Recall", f"{metrics['recall']:.3f}" if metrics else "—")
    with col3:
        st.metric("F1 Score", f"{metrics['f1']:.3f}" if metrics else "—")
    with col4:
        st.metric("ROC AUC", f"{metrics['auc_roc']:.3f}" if metrics else "—")

    st.divider()

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown("**Confusion matrix**")
        if metrics:
            st.plotly_chart(
                _plot_confusion_matrix(metrics["confusion_matrix"]),
                use_container_width=True,
            )
        else:
            st.info("Train a model to view the confusion matrix.")

    with col_roc:
        st.markdown("**ROC curve**")
        y_test = st.session_state.get("y_test")
        y_scores = st.session_state.get("y_scores")
        if y_test is not None and y_scores is not None:
            st.plotly_chart(
                _plot_roc_curve(y_test, y_scores), use_container_width=True
            )
        else:
            st.info("Train a model to view the ROC curve.")

    st.divider()

    col_pr, col_fi = st.columns(2)
    with col_pr:
        st.markdown("**Precision-recall curve**")
        if y_test is not None and y_scores is not None:
            st.plotly_chart(
                _plot_pr_curve(y_test, y_scores), use_container_width=True
            )
        else:
            st.info("Train a model to view the precision-recall curve.")

    with col_fi:
        st.markdown("**Feature importance (SHAP)**")
        explainer = st.session_state.get("explainer")
        feature_cols = st.session_state.get("feature_cols")
        df = st.session_state.get("df")
        if explainer is not None and feature_cols is not None and df is not None:
            st.plotly_chart(
                _plot_shap_importance(explainer, df, feature_cols),
                use_container_width=True,
            )
        else:
            st.info("Train a model to view feature importance.")


def render_visual_analytics(df: pd.DataFrame | None) -> None:
    """Tab 3: Data exploration, distributions, fraud patterns."""
    st.subheader("📈 Visual Analytics")

    if df is None:
        st.info("Load a dataset to view analytics.")
        return

    col_balance, col_amount = st.columns(2)
    with col_balance:
        st.markdown("**Class distribution**")
        counts = df["label"].value_counts().reset_index()
        counts.columns = ["label", "count"]
        counts["type"] = counts["label"].map({0: "Legitimate", 1: "Fraud"})
        counts["pct"] = counts["count"] / counts["count"].sum() * 100
        fig = px.bar(
            counts, x="type", y="count",
            text=counts["pct"].map(lambda p: f"{p:.1f}%"),
            color="type",
            color_discrete_map={"Legitimate": "#4575b4", "Fraud": "#d73027"},
            labels={"type": "", "count": "Transactions"},
        )
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_amount:
        st.markdown("**Transaction amount distribution**")
        plot_df = df[["amount", "label"]].copy()
        plot_df["type"] = plot_df["label"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.histogram(
            plot_df, x="amount", color="type", barmode="overlay",
            log_y=True, nbins=80,
            color_discrete_map={"Legitimate": "#4575b4", "Fraud": "#d73027"},
            labels={"amount": "Transaction Amount ($)", "count": "Count (log)"},
        )
        fig.update_layout(margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col_time, col_rules = st.columns(2)
    with col_time:
        st.markdown("**Fraud rate by hour of day**")
        if "hour_of_day" in df.columns:
            hourly = (
                df.groupby("hour_of_day")["label"]
                .mean()
                .reset_index()
                .rename(columns={"label": "fraud_rate"})
            )
            fig = px.bar(
                hourly, x="hour_of_day", y="fraud_rate",
                labels={"hour_of_day": "Hour (24h)", "fraud_rate": "Fraud Rate"},
            )
            fig.update_layout(margin=dict(t=10, b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run feature engineering to view hourly fraud rates.")

    with col_rules:
        st.markdown("**Rule engine flag breakdown**")
        flag_cols = [c for c in df.columns if c.startswith("flag_")]
        if flag_cols:
            flag_counts = df[flag_cols].sum().reset_index()
            flag_counts.columns = ["rule", "count"]
            flag_counts["rule"] = flag_counts["rule"].str.replace("flag_", "", regex=False)
            fig = px.bar(
                flag_counts.sort_values("count", ascending=False),
                x="rule", y="count",
                labels={"rule": "Rule", "count": "Transactions Flagged"},
            )
            fig.update_layout(margin=dict(t=10, b=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Apply rules to view flag breakdown.")

    st.divider()

    st.markdown("**Feature correlation heatmap**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_corr = {"label", "transaction_id", "rule_score"}
    heatmap_cols = [
        c for c in numeric_cols
        if c not in exclude_corr and not c.startswith("flag_")
    ][:12]
    if len(heatmap_cols) >= 2:
        corr = df[heatmap_cols].corr()
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
        )
        fig.update_layout(margin=dict(t=10, b=10), height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric features for correlation heatmap.")


def render_single_transaction(df: pd.DataFrame | None, config: dict) -> None:
    """Tab 4: Investigate one transaction with SHAP waterfall."""
    st.subheader("🔎 Single transaction investigation")
    st.markdown(
        "Select a flagged transaction to see its full breakdown: "
        "risk score, rule flags, and SHAP waterfall chart."
    )

    explainer = st.session_state.get("explainer")
    feature_cols = st.session_state.get("feature_cols")
    risk_scores = st.session_state.get("risk_scores")

    if df is None:
        st.info("Load a dataset to investigate transactions.")
        return

    threshold = config["fraud_threshold"]

    # Build list of flagged transaction IDs — capped at top 200 by risk score
    # so the selectbox renders instantly regardless of dataset size.
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    rule_flagged = (df["rule_score"] > 0) if "rule_score" in df.columns else pd.Series(False, index=df.index)
    ml_flagged = (risk_scores >= threshold) if risk_scores is not None else pd.Series(False, index=df.index)
    flagged_mask = rule_flagged | ml_flagged

    flagged_df = df.loc[flagged_mask, ["transaction_id"]].copy()
    if risk_scores is not None:
        flagged_df["_score"] = risk_scores.loc[flagged_mask].values
        flagged_df = flagged_df.sort_values("_score", ascending=False)
    flagged_ids = flagged_df["transaction_id"].head(200).tolist()

    col_select, col_detail = st.columns([1, 2])
    with col_select:
        if flagged_ids:
            selected_id = st.selectbox(
                "Select flagged transaction (top 200 by risk score)",
                options=flagged_ids,
                format_func=lambda x: str(x),
            )
            row = df[df["transaction_id"] == selected_id].iloc[0]

            st.markdown("**Transaction summary**")
            st.markdown(f"- **Amount:** {format_currency(row['amount'])}")
            st.markdown(f"- **Timestamp:** {row['timestamp']}")
            if "customer_id" in df.columns:
                st.markdown(f"- **Customer:** {row.get('customer_id', '—')}")
            if risk_scores is not None:
                score = risk_scores.loc[row.name]
                st.markdown(f"- **Risk score:** {score:.3f}")
            if "rule_score" in df.columns:
                st.markdown(f"- **Rule score:** {int(row['rule_score'])}")
        else:
            st.info("No flagged transactions. Train the model or lower the threshold.")
            selected_id = None

    with col_detail:
        st.markdown("**SHAP explanation**")
        if selected_id is not None and explainer is not None and feature_cols is not None:
            row_df = df[df["transaction_id"] == selected_id]
            # Cache SHAP explanations in session state so switching between
            # previously viewed transactions is instant.
            shap_cache = st.session_state.setdefault("shap_cache", {})
            cache_key = (selected_id, id(explainer))
            try:
                if cache_key not in shap_cache:
                    with st.spinner("Computing SHAP explanation…"):
                        shap_cache[cache_key] = explain_transaction(explainer, row_df, feature_cols)
                shap_exp = shap_cache[cache_key]
                st.plotly_chart(
                    _plot_shap_waterfall(shap_exp), use_container_width=True
                )
            except Exception as exc:
                st.warning(f"Could not generate SHAP explanation: {exc}")
        elif explainer is None:
            st.info("Train the model to view SHAP explanations.")
        else:
            st.info("Select a transaction to view its explanation.")

        st.markdown("**Rule flags**")
        if selected_id is not None and flag_cols:
            row = df[df["transaction_id"] == selected_id].iloc[0]
            fired = [c.replace("flag_", "") for c in flag_cols if row[c]]
            if fired:
                for rule in fired:
                    st.markdown(f"- ✅ `{rule}`")
            else:
                st.markdown("No rule flags triggered.")
        elif not flag_cols:
            st.info("Apply rules to view flag details.")


def render_batch_scoring(df_main: pd.DataFrame | None, config: dict) -> None:
    """Tab 5: Upload a dataset, score all transactions, download results."""
    st.subheader("📦 Batch scoring")
    st.markdown(
        "Upload a CSV of transactions to score them all against the trained model "
        "and rule engine. Download the results with risk scores and flag columns attached."
    )

    model = st.session_state.get("model")
    feature_cols = st.session_state.get("feature_cols")

    batch_file = st.file_uploader(
        "Upload CSV for batch scoring",
        type=["csv"],
        key="batch_upload",
    )

    if batch_file is None:
        st.info("Upload a CSV file to score transactions in batch.")
        return

    if model is None:
        st.warning("Train a model first before running batch scoring.")
        return

    try:
        raw_batch = pd.read_csv(batch_file)
        batch_file.seek(0)
        suggestions = get_column_mapping_suggestions(raw_batch)
        batch_mapping: dict = {}
        st.markdown("**Column mapping for batch file**")
        for req_col in REQUIRED_COLUMNS:
            all_cols = list(raw_batch.columns)
            default_idx = 0
            if req_col in suggestions and suggestions[req_col] in all_cols:
                default_idx = all_cols.index(suggestions[req_col])
            chosen = st.selectbox(
                f"`{req_col}` ←",
                options=all_cols,
                index=default_idx,
                key=f"batch_map_{req_col}",
            )
            batch_mapping[chosen] = req_col

        if st.button("Run batch scoring", type="primary"):
            with st.spinner("Scoring transactions…"):
                batch_file.seek(0)
                df_batch = load_user_dataset(batch_file, column_mapping=batch_mapping)
                df_eng = engineer_features(df_batch)
                thresholds = _build_thresholds(config)
                thresholds_json = json.dumps(thresholds, sort_keys=True)
                df_scored = _cached_apply_rules(df_eng, thresholds_json)
                # Add any columns the model expects that are absent in this
                # upload as NaN — XGBoost handles missing values natively.
                missing_cols = [c for c in feature_cols if c not in df_scored.columns]
                if missing_cols:
                    padding = pd.DataFrame(
                        np.nan, index=df_scored.index, columns=missing_cols
                    )
                    df_scored = pd.concat([df_scored, padding], axis=1)
                scores = predict_risk_scores(model, df_scored, feature_cols)
                df_scored["risk_score"] = scores

            threshold = config["fraud_threshold"]
            total = len(df_scored)
            rule_flagged = int((df_scored["rule_score"] > 0).sum()) if "rule_score" in df_scored.columns else 0
            ml_flagged = int((scores >= threshold).sum())

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total scored", f"{total:,}")
            col2.metric("Flagged by rules", f"{rule_flagged:,}")
            col3.metric("Flagged by ML", f"{ml_flagged:,}")
            col4.metric(
                "ML flag rate",
                format_percentage(ml_flagged / total) if total else "—"
            )

            st.markdown("**Preview (top 50 rows)**")
            st.dataframe(df_scored.head(50), use_container_width=True)

            csv_bytes = df_scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download scored CSV",
                data=csv_bytes,
                file_name="scored_transactions.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Batch scoring failed: {exc}")
        logger.exception("Batch scoring error")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Main dashboard application."""

    st.title("🔍 Financial Crime Detection Dashboard")
    st.caption(
        "Detect suspicious transactions using configurable business rules "
        "and an XGBoost classifier with SHAP explainability."
    )

    config = render_sidebar()

    # ---- Load and process data ----
    df: pd.DataFrame | None = None
    try:
        if config["data_source"] == "Upload my own CSV" and config["uploaded_file"] is not None:
            df_raw = load_user_dataset(
                config["uploaded_file"],
                column_mapping=config["column_mapping"] or None,
            )
            df_eng = _cached_engineer(df_raw)
        else:
            df_raw = _cached_load_default("data/raw/")
            df_eng = _cached_engineer(df_raw)

        thresholds = _build_thresholds(config)
        thresholds_json = json.dumps(thresholds, sort_keys=True)
        df = _cached_apply_rules(df_eng, thresholds_json)

        # Attach risk scores if model exists
        model = st.session_state.get("model")
        feature_cols = st.session_state.get("feature_cols")
        if model is not None and feature_cols is not None:
            risk_scores = predict_risk_scores(model, df, feature_cols)
            st.session_state["risk_scores"] = risk_scores

        st.session_state["df"] = df

    except FileNotFoundError:
        st.info(
            "No dataset found. The dashboard looks for the IEEE-CIS files in "
            "`data/raw/` and falls back to the bundled sample in `data/sample/`. "
            "Either add `train_transaction.csv` and `train_identity.csv` to "
            "`data/raw/`, or upload your own CSV using the sidebar."
        )
    except ValueError as exc:
        st.error(f"Data validation error: {exc}")
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        logger.exception("Data load error")

    # ---- Train on button click ----
    if config["train_button"]:
        if df is not None:
            _run_training(df, config)
        else:
            st.warning("Load a dataset before training.")

    # ---- Render tabs ----
    tab_overview, tab_performance, tab_analytics, tab_single, tab_batch = st.tabs(
        ["📋 Overview", "📊 Model Performance", "📈 Visual Analytics",
         "🔎 Single Transaction", "📦 Batch Scoring"]
    )

    with tab_overview:
        render_overview(df, config)

    with tab_performance:
        render_model_performance()

    with tab_analytics:
        render_visual_analytics(df)

    with tab_single:
        render_single_transaction(df, config)

    with tab_batch:
        render_batch_scoring(df, config)


if __name__ == "__main__":
    main()
