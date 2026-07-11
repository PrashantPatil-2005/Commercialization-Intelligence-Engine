"""Portfolio page - Investment Decision Dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.theme import get_theme, OUTCOME_META, FONT_SM, FONT_2XL, WEIGHT_BOLD, RADIUS_MD
from app.styles import page_header, section_header
from app.components import render_portfolio_table, render_kpi_row


def render(report: dict, artifacts: dict, insights: pd.DataFrame):
    t = get_theme()
    ranked = insights.sort_values("portfolio_rank")

    st.markdown(page_header(
        "Investment Decisions",
        "Filter, compare, and select concepts for advancement",
        tag="Portfolio"
    ), unsafe_allow_html=True)

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

    render_portfolio_table(ranked, key="portfolio_full")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(section_header("Outcome Breakdown"), unsafe_allow_html=True)

    n_cols = min(len(outcomes), 3)
    cols = st.columns(n_cols)
    sorted_outcomes = sorted(outcomes.items(), key=lambda x: -x[1])
    for idx, (outcome, count) in enumerate(sorted_outcomes):
        col = cols[idx % n_cols]
        with col:
            meta = OUTCOME_META.get(outcome, {})
            color = meta.get("color", t["text_muted"])
            st.markdown(
                f'<div style="text-align:center;padding:10px;border:1px solid {color}30;'
                f'border-radius:{RADIUS_MD};background:{color}08">'
                f'<div style="font-size:{FONT_2XL};font-weight:{WEIGHT_BOLD};color:{color}">{count}</div>'
                f'<div style="font-size:{FONT_SM};color:{t["text_secondary"]};margin-top:2px">{outcome}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
