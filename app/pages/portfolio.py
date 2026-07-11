"""Portfolio page - Investment Decision Dashboard.

Business question: "Which concepts deserve investment?"
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.styles import page_header, section_header, OUTCOME_META
from app.components import render_portfolio_table, render_kpi_row


def render(report: dict, artifacts: dict, insights: pd.DataFrame):
    ranked = insights.sort_values("portfolio_rank")

    # Page Header
    st.markdown(page_header(
        "Investment Decisions",
        "Filter, compare, and select concepts for advancement"
    ), unsafe_allow_html=True)

    # Quick summary KPIs
    outcomes = report["classifier"]["outcome_distribution"]
    n_forward = outcomes.get("MVP Build", 0) + outcomes.get("Customer Pilot", 0) + outcomes.get("Reusable Asset", 0)
    n_archive = outcomes.get("Archive", 0)
    avg_readiness = ranked["readiness_score"].mean()

    render_kpi_row([
        {"label": "Investment Candidates", "value": str(n_forward), "sub": "ready to advance", "accent": "green"},
        {"label": "Avg Readiness", "value": f"{avg_readiness:.1f}", "sub": "portfolio mean", "accent": "blue"},
        {"label": "Low Priority", "value": str(n_archive), "sub": "deprioritize", "accent": "red"},
    ])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Full portfolio table
    render_portfolio_table(ranked, key="portfolio_full")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Outcome breakdown
    st.markdown(section_header("Outcome Breakdown"), unsafe_allow_html=True)

    cols = st.columns(len(outcomes))
    for col, (outcome, count) in zip(cols, sorted(outcomes.items(), key=lambda x: -x[1])):
        with col:
            meta = OUTCOME_META.get(outcome, {})
            color = meta.get("color", "#8b949e")
            st.markdown(
                f'<div style="text-align:center;padding:10px;border:1px solid {color}30;'
                f"border-radius:6px;background:{color}08\">"
                f'<div style="font-size:18px;font-weight:700;color:{color}">{count}</div>'
                f'<div style="font-size:10px;color:#656d76;margin-top:2px">{outcome}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
