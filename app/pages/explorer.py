"""Explorer page - Executive Decision Report.

Business question: "Why did this concept receive this recommendation?"

Pipeline: RF Prediction → SHAP → Evidence Dict → LLM Summary → Dashboard
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.styles import page_header, section_header, muted_callout, OUTCOME_META
from app.components import (
    render_concept_header, render_concept_metrics,
    render_narrative, render_evidence_bullets, render_decision_summary,
    render_shap_evidence_table,
)
from app.charts import plot_shap_waterfall
from models.insight_layer import FEATURE_LABELS


def render(artifacts: dict, insights: pd.DataFrame):
    ranked = insights.sort_values("portfolio_rank")
    feature_names = artifacts["feature_columns"]

    # Concept Selector
    concept_names = ranked["concept_name"].tolist()
    selected = st.selectbox(
        "Select concept",
        concept_names,
        key="explorer_select",
        label_visibility="collapsed",
        placeholder="Search concepts...",
    )
    row = ranked[ranked["concept_name"] == selected].iloc[0]

    # Page Header
    st.markdown(page_header(
        f"Decision Report: {row['concept_name']}",
        f"{row['industry']} &middot; {row.get('problem_area', '')} &middot; Rank #{int(row['portfolio_rank'])}",
        tag="Explorer"
    ), unsafe_allow_html=True)

    # Metrics
    render_concept_metrics(row)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Decision Section
    st.markdown(section_header("Decision Summary"), unsafe_allow_html=True)
    render_decision_summary(row)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # LLM Summary (generated from SHAP evidence)
    st.markdown(section_header("AI Analysis"), unsafe_allow_html=True)
    render_narrative(row.get("ai_narrative", ""))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Structured SHAP Evidence
    st.markdown(section_header("SHAP Evidence Breakdown"), unsafe_allow_html=True)
    evidence = row.get("shap_evidence", None)
    if isinstance(evidence, str):
        import json
        try:
            evidence = json.loads(evidence)
        except (json.JSONDecodeError, TypeError):
            evidence = None
    render_shap_evidence_table(evidence)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # SHAP Waterfall
    st.markdown(section_header("Feature Contributions (SHAP)"), unsafe_allow_html=True)
    fig = plot_shap_waterfall(row, feature_names, FEATURE_LABELS)
    if fig:
        st.pyplot(fig)
    else:
        st.markdown(muted_callout("No SHAP data available for this concept."), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Raw Features
    with st.expander("Raw Feature Values", expanded=False):
        feat_data = {
            FEATURE_LABELS.get(f, f): round(float(row[f]), 4)
            for f in feature_names if f in row.index
        }
        st.json(feat_data)
