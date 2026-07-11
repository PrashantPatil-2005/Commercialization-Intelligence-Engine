"""Analytics page - Pattern Analysis Dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.styles import page_header, section_header, muted_callout
from app.charts import (
    plot_readiness_bar, plot_outcome_distribution,
    plot_cluster_scatter, plot_feature_importance, plot_correlation_matrix,
)
from models.insight_layer import FEATURE_LABELS


def render(artifacts: dict, insights: pd.DataFrame):
    ranked = insights.sort_values("portfolio_rank")

    st.markdown(page_header(
        "Portfolio Analytics",
        "Patterns, clusters, and feature relationships across concepts",
        tag="Analytics"
    ), unsafe_allow_html=True)

    st.markdown(section_header("Portfolio Performance"), unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(plot_readiness_bar(ranked))
    with c2:
        st.pyplot(plot_outcome_distribution(ranked))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(section_header("Customer Behavior Patterns"), unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(plot_cluster_scatter(artifacts["df"]))
    with c4:
        st.pyplot(plot_feature_importance(artifacts["importance"], FEATURE_LABELS))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(section_header("Feature Relationships"), unsafe_allow_html=True)
    numeric_features = [
        "demand_intensity", "engagement_depth_norm", "feasibility_risk",
        "repeatability", "segment_similarity", "revenue_potential",
        "strategic_fit", "confidence",
    ]
    available = [f for f in numeric_features if f in artifacts["df"].columns]
    if available:
        st.pyplot(plot_correlation_matrix(artifacts["df"], available))
    else:
        st.markdown(muted_callout("Not enough numeric features for correlation analysis."), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(section_header("Industry Breakdown"), unsafe_allow_html=True)
    industry_stats = (
        ranked.groupby("industry")
        .agg(
            count=("concept_name", "count"),
            avg_readiness=("readiness_score", "mean"),
            avg_confidence=("confidence_score", "mean"),
        )
        .sort_values("avg_readiness", ascending=False)
        .round(2)
    )
    st.dataframe(
        industry_stats.rename(columns={
            "count": "Concepts",
            "avg_readiness": "Avg Readiness",
            "avg_confidence": "Avg Confidence",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg Readiness": st.column_config.NumberColumn(format="%.1f"),
            "Avg Confidence": st.column_config.NumberColumn(format=".0%"),
        },
    )
