"""Model page - Decision Engine Dashboard."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.theme import get_theme, PROCESSED_DIR
from app.styles import (
    page_header, section_header, muted_callout,
)
from app.components import render_kpi_row
from app.charts import plot_cv_folds, plot_feature_importance
from models.insight_layer import FEATURE_LABELS


def render():
    t = get_theme()

    report_path = PROCESSED_DIR / "model_report.json"
    val_path = PROCESSED_DIR / "validation_report.json"

    if not report_path.exists():
        st.markdown(muted_callout("model_report.json not found. Run the pipeline first."), unsafe_allow_html=True)
        return

    with open(report_path) as f:
        model_report = json.load(f)

    st.markdown(page_header(
        "Decision Engine",
        "How the system analyzes concepts and generates recommendations",
        tag="Model"
    ), unsafe_allow_html=True)

    st.markdown(section_header("Pipeline Overview"), unsafe_allow_html=True)
    clustering = model_report.get("clustering", {})
    classifier = model_report.get("classifier", {})
    render_kpi_row([
        {"label": "Data Generation", "value": f"{model_report.get('concepts_scored', 'N/A')} concepts", "sub": "Synthetic signals", "accent": "blue"},
        {"label": "Feature Engineering", "value": "22 features", "sub": "Cleaned + normalized", "accent": "blue"},
        {"label": "Clustering", "value": f"K={clustering.get('n_clusters', 'N/A')}", "sub": "K-Means grouping", "accent": "purple"},
        {"label": "Classification", "value": f"{classifier.get('n_estimators', 'N/A')} trees", "sub": "Random Forest", "accent": "green"},
    ])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(section_header("Model Validation"), unsafe_allow_html=True)
    cv = model_report.get("cross_validation", {})
    if cv:
        render_kpi_row([
            {"label": "Method", "value": cv.get("method", "N/A"), "accent": "blue"},
            {"label": "Mean Accuracy", "value": f"{cv.get('accuracy_mean', 0):.0%}", "accent": "green"},
            {"label": "Std Dev", "value": f"+/-{cv.get('accuracy_std', 0):.0%}", "accent": "orange"},
        ])
        folds = cv.get("fold_scores", [])
        if folds:
            st.pyplot(plot_cv_folds(folds))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(section_header("Model Configuration"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Baseline Weights**", unsafe_allow_html=True)
        weights = model_report.get("baseline_model", {}).get("weights", {})
        weights_df = pd.DataFrame([
            {"Feature": k.replace("_", " ").title(), "Weight": v}
            for k, v in weights.items()
        ]).sort_values("Weight", ascending=False)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown(f"**Random Forest Configuration**", unsafe_allow_html=True)
        classifier = model_report.get("classifier", {})
        st.markdown(
            f"- Trees: {classifier.get('n_estimators', 'N/A')}<br>"
            f"- Training labels: {classifier.get('training_labels', 'N/A')}<br>"
            f"- Classes: {len(classifier.get('outcome_distribution', {}))}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Outcome distribution:**", unsafe_allow_html=True)
        for outcome, count in sorted(classifier.get("outcome_distribution", {}).items(), key=lambda x: -x[1]):
            st.markdown(f"- {outcome}: **{count}** concepts", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(section_header("Feature Importance"), unsafe_allow_html=True)
    c3, c4 = st.columns([2, 1])
    with c3:
        top_feats = model_report.get("top_features", [])
        if top_feats:
            feats_df = pd.DataFrame(top_feats)
            feats_df["label"] = feats_df["feature"].map(FEATURE_LABELS).fillna(feats_df["feature"])
            st.dataframe(
                feats_df[["label", "importance"]].rename(columns={"label": "Feature", "importance": "Importance"}),
                use_container_width=True, hide_index=True,
            )
    with c4:
        st.pyplot(plot_feature_importance(feats_df if top_feats else pd.DataFrame({"feature": [], "importance": []}), FEATURE_LABELS))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    cluster_path = PROCESSED_DIR / "cluster_summary.csv"
    if cluster_path.exists():
        st.markdown(section_header("Cluster Profiles"), unsafe_allow_html=True)
        clusters = pd.read_csv(cluster_path)
        st.dataframe(clusters, use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if val_path.exists():
        st.markdown(section_header("Data Quality"), unsafe_allow_html=True)
        with open(val_path) as f:
            val = json.load(f)
        status = "Passed" if val.get("validation_passed") else "Failed"
        st.markdown(
            f"- Schema validation: **{status}**<br>"
            f"- Concepts: **{val.get('concept_count', 'N/A')}**<br>"
            f"- Interactions: **{val.get('interaction_count', 'N/A')}**",
            unsafe_allow_html=True,
        )
        feat_summary = val.get("feature_summary", {})
        if feat_summary:
            feat_stats = pd.DataFrame(feat_summary).T
            feat_stats.columns = ["Mean", "Min", "Max"]
            st.dataframe(feat_stats.round(4), use_container_width=True)
