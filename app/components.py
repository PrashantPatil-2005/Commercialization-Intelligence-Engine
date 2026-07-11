"""Reusable UI components for the enterprise dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.styles import (
    OUTCOME_META, CLUSTER_COLORS,
    kpi_card, section_header, progress_bar, outcome_badge,
    risk_badge, confidence_badge, decision_card,
    info_callout, muted_callout, page_header,
)


def render_kpi_row(metrics: list[dict]):
    """Render a row of compact KPI cards.

    Each metric dict: {"label": str, "value": str, "sub": str, "accent": str}
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.markdown(
                kpi_card(m["label"], m["value"], m.get("sub", ""), m.get("accent", "")),
                unsafe_allow_html=True,
            )


def render_portfolio_table(
    df: pd.DataFrame,
    key: str = "portfolio_table",
):
    """Render the main portfolio table with sorting and filtering."""
    st.markdown(section_header("Portfolio Ranking"), unsafe_allow_html=True)

    # Filters row
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        industry_filter = st.multiselect(
            "Industry",
            options=sorted(df["industry"].unique()),
            default=[],
            placeholder="All industries",
            key=f"{key}_ind",
            label_visibility="collapsed",
        )
    with f2:
        outcome_filter = st.multiselect(
            "Outcome",
            options=sorted(df["recommended_outcome"].unique()),
            default=[],
            placeholder="All outcomes",
            key=f"{key}_out",
            label_visibility="collapsed",
        )
    with f3:
        search = st.text_input(
            "Search",
            placeholder="Search concepts...",
            key=f"{key}_search",
            label_visibility="collapsed",
        )

    filtered = df.copy()
    if industry_filter:
        filtered = filtered[filtered["industry"].isin(industry_filter)]
    if outcome_filter:
        filtered = filtered[filtered["recommended_outcome"].isin(outcome_filter)]
    if search:
        mask = filtered["concept_name"].str.contains(search, case=False, na=False)
        filtered = filtered[mask]

    # Build display dataframe
    display = filtered[[
        "portfolio_rank", "concept_name", "industry", "problem_area",
        "readiness_score", "confidence_score", "recommended_outcome",
    ]].copy()
    display = display.rename(columns={
        "portfolio_rank": "#",
        "concept_name": "Concept",
        "industry": "Industry",
        "problem_area": "Problem Area",
        "readiness_score": "Readiness",
        "confidence_score": "Confidence",
        "recommended_outcome": "Outcome",
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        key=key,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Readiness": st.column_config.ProgressColumn(
                min_value=0, max_value=100, width="medium",
                help="Commercialization readiness score (1-100)",
            ),
            "Confidence": st.column_config.NumberColumn(
                format="%.0f%%", width="small",
                help="Model confidence based on data volume and certainty",
            ),
        },
    )

    st.markdown(
        f'<div style="font-size:10px;color:#8b949e;margin-top:2px">'
        f'Showing {len(filtered)} of {len(df)} concepts</div>',
        unsafe_allow_html=True,
    )


def render_concept_header(row: pd.Series):
    """Render concept detail header section."""
    outcome = row["recommended_outcome"]
    meta = OUTCOME_META.get(outcome, OUTCOME_META["Archive"])

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
        f'<h3 style="margin:0;font-size:16px;font-weight:600">{row["concept_name"]}</h3>'
        f'{outcome_badge(outcome)}'
        f'</div>'
        f'<div style="font-size:11px;color:#656d76;margin-bottom:12px">'
        f'{row["industry"]} &middot; {row.get("problem_area", "")} &middot; Rank #{int(row["portfolio_rank"])}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_concept_metrics(row: pd.Series):
    """Render concept detail metrics row."""
    readiness = row["readiness_score"]
    confidence = row["confidence_score"]
    outcome = row["recommended_outcome"]

    render_kpi_row([
        {"label": "Readiness Score", "value": f"{readiness:.1f}", "sub": "out of 100", "accent": "blue"},
        {"label": "Confidence", "value": f"{confidence:.0%}", "sub": "model certainty", "accent": "green"},
        {"label": "Recommendation", "value": OUTCOME_META.get(outcome, {}).get("label", outcome), "sub": "ML outcome", "accent": "purple"},
    ])

    # Readiness progress bar
    st.markdown(
        f'<div style="margin-top:2px;margin-bottom:12px">'
        f'{progress_bar(readiness, 4)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_decision_summary(row: pd.Series):
    """Render executive decision summary: Decision, Evidence, Risks, Confidence, Next Step."""
    outcome = row["recommended_outcome"]
    evidence = row.get("key_evidence", "")
    confidence = row["confidence_score"]

    # Next step based on outcome
    next_steps = {
        "MVP Build": "Allocate engineering resources. Build focused prototype for validation.",
        "Customer Pilot": "Identify 1-2 pilot customers. Define success metrics. Begin contracting.",
        "Reusable Asset": "Evaluate platform packaging. Test cross-sell potential across industries.",
        "Incubate": "Run additional demos next quarter. Sharpen positioning. Test with new segments.",
        "Archive": "Document learnings. Deprioritize. Reallocate team to higher-ranked concepts.",
    }
    next_step = next_steps.get(outcome, "Gather more evidence before deciding.")

    # Risk assessment
    fr = row.get("feasibility_risk", 0.5)
    if fr > 0.55:
        risk_text = "High delivery complexity"
        risk_level = "High"
    elif fr > 0.40:
        risk_text = "Moderate delivery complexity"
        risk_level = "Moderate"
    else:
        risk_text = "Low delivery complexity"
        risk_level = "Low"

    st.markdown(decision_card("Decision", f'{outcome_badge(outcome)} &nbsp; {risk_badge(risk_level)}'), unsafe_allow_html=True)
    st.markdown(decision_card("Evidence", evidence if evidence else "No evidence available."), unsafe_allow_html=True)
    st.markdown(decision_card("Risk", risk_text), unsafe_allow_html=True)
    st.markdown(decision_card("Confidence", f'{confidence_badge(confidence)} &nbsp; {confidence:.0%}'), unsafe_allow_html=True)
    st.markdown(decision_card("Next Step", next_step), unsafe_allow_html=True)


def render_evidence_bullets(evidence_str: str):
    """Render evidence as compact bullet points."""
    if not evidence_str or evidence_str == "---":
        st.markdown(muted_callout("No evidence available."), unsafe_allow_html=True)
        return

    bullets = [e.strip() for e in evidence_str.split(";") if e.strip()]
    html = '<ul class="evidence-list">'
    for b in bullets:
        html += f"<li>{b}</li>"
    html += "</ul>"
    st.markdown(html, unsafe_allow_html=True)


def render_narrative(narrative: str):
    """Render AI narrative in a compact box."""
    if not narrative:
        st.markdown(muted_callout("No recommendation available."), unsafe_allow_html=True)
        return
    st.markdown(
        f'<div style="background:#f6f8fa;border:1px solid #d1d9e0;border-left:3px solid #0969da;'
        f'border-radius:0 6px 6px 0;padding:10px 12px;font-size:12px;color:#1f2328;line-height:1.6">'
        f'{narrative}</div>',
        unsafe_allow_html=True,
    )
