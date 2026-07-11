"""Overview page - Portfolio Health Dashboard.

Business question: "How healthy is the commercialization portfolio?"
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.styles import page_header, section_header, OUTCOME_META
from app.components import render_kpi_row, render_portfolio_table
from app.charts import plot_readiness_bar, plot_outcome_distribution


def render(report: dict, artifacts: dict, insights: pd.DataFrame):
    ranked = insights.sort_values("portfolio_rank")

    # Page Header
    st.markdown(page_header(
        "Portfolio Health",
        "At-a-glance status of the commercialization pipeline"
    ), unsafe_allow_html=True)

    # KPI Row
    outcomes = report["classifier"]["outcome_distribution"]
    n_forward = outcomes.get("MVP Build", 0) + outcomes.get("Customer Pilot", 0) + outcomes.get("Reusable Asset", 0)
    n_incubate = outcomes.get("Incubate", 0)
    n_archive = outcomes.get("Archive", 0)
    cv = report.get("cross_validation", {})
    avg_conf = ranked["confidence_score"].mean()
    avg_readiness = ranked["readiness_score"].mean()

    render_kpi_row([
        {"label": "Concepts Analyzed", "value": str(report["concepts_scored"]), "sub": "total in portfolio", "accent": "blue"},
        {"label": "Avg Readiness", "value": f"{avg_readiness:.1f}", "sub": "portfolio mean", "accent": "blue"},
        {"label": "Investment Candidates", "value": str(n_forward), "sub": "Pilot + Asset + MVP", "accent": "green"},
        {"label": "Needs Evidence", "value": str(n_incubate), "sub": "Incubate", "accent": "orange"},
        {"label": "Low Priority", "value": str(n_archive), "sub": "Archive candidates", "accent": "red"},
        {"label": "Model Accuracy", "value": f"{cv.get('accuracy_mean', 0):.0%}", "sub": cv.get("method", ""), "accent": "purple"},
    ])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Charts Row: Readiness + Distribution
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(section_header("Readiness Ranking"), unsafe_allow_html=True)
        st.pyplot(plot_readiness_bar(ranked))
    with c2:
        st.markdown(section_header("Recommendation Split"), unsafe_allow_html=True)
        st.pyplot(plot_outcome_distribution(ranked))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Portfolio Table
    render_portfolio_table(ranked, key="overview_table")
