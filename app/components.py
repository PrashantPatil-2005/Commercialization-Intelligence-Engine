"""
Reusable UI Components — Theme-Aware

All colors read from the active theme via get_theme().
"""

from __future__ import annotations

import html as html_mod
import pandas as pd
import streamlit as st

from app.theme import (
    get_theme, OUTCOME_META,
    FONT_SM, FONT_MD, FONT_LG, FONT_XL,
    WEIGHT_SEMIBOLD, WEIGHT_BOLD,
    LEADING_NORMAL, LEADING_RELAXED,
)
from app.styles import (
    kpi_card, section_header, progress_bar, outcome_badge,
    risk_badge, confidence_badge, decision_card,
    info_callout, muted_callout, page_header,
)


def render_kpi_row(metrics: list[dict]):
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
    t = get_theme()
    st.markdown(section_header("Portfolio Ranking"), unsafe_allow_html=True)

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
                format=".0%", width="small",
                help="Model confidence based on data volume and certainty",
            ),
        },
    )

    st.markdown(
        f'<div style="font-size:{FONT_SM};color:{t["text_muted"]};margin-top:2px">'
        f'Showing {len(filtered)} of {len(df)} concepts</div>',
        unsafe_allow_html=True,
    )


def render_concept_header(row: pd.Series):
    t = get_theme()
    outcome = row["recommended_outcome"]

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
        f'<h3 style="margin:0;font-size:{FONT_XL};font-weight:{WEIGHT_SEMIBOLD};color:{t["text_primary"]}">{row["concept_name"]}</h3>'
        f'{outcome_badge(outcome)}'
        f'</div>'
        f'<div style="font-size:{FONT_MD};color:{t["text_secondary"]};margin-bottom:12px">'
        f'{row["industry"]} &middot; {row.get("problem_area", "")} &middot; Rank #{int(row["portfolio_rank"])}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_concept_metrics(row: pd.Series):
    readiness = row["readiness_score"]
    confidence = row["confidence_score"]
    outcome = row["recommended_outcome"]

    render_kpi_row([
        {"label": "Readiness Score", "value": f"{readiness:.1f}", "sub": "out of 100", "accent": "blue"},
        {"label": "Confidence", "value": f"{confidence:.0%}", "sub": "model certainty", "accent": "green"},
        {"label": "Recommendation", "value": OUTCOME_META.get(outcome, {}).get("label", outcome), "sub": "ML outcome", "accent": "purple"},
    ])

    st.markdown(
        f'<div style="margin-top:2px;margin-bottom:12px">'
        f'{progress_bar(readiness, 4)}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_decision_summary(row: pd.Series):
    outcome = row["recommended_outcome"]
    evidence = row.get("key_evidence", "")
    confidence = row["confidence_score"]

    next_steps = {
        "MVP Build": "Allocate engineering resources. Build focused prototype for validation.",
        "Customer Pilot": "Identify 1-2 pilot customers. Define success metrics. Begin contracting.",
        "Reusable Asset": "Evaluate platform packaging. Test cross-sell potential across industries.",
        "Incubate": "Run additional demos next quarter. Sharpen positioning. Test with new segments.",
        "Archive": "Document learnings. Deprioritize. Reallocate team to higher-ranked concepts.",
    }
    next_step = next_steps.get(outcome, "Gather more evidence before deciding.")

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
    t = get_theme()
    if not narrative:
        st.markdown(muted_callout("No recommendation available."), unsafe_allow_html=True)
        return
    st.markdown(
        f'<div style="background:{t["info_subtle"]};border:1px solid {t["info"]}40;border-left:3px solid {t["info"]};'
        f'border-radius:0 6px 6px 0;padding:10px 12px;font-size:{FONT_MD};color:{t["text_primary"]};line-height:{LEADING_RELAXED}">'
        f'{narrative}</div>',
        unsafe_allow_html=True,
    )


def render_shap_evidence_table(evidence: dict):
    t = get_theme()
    if not evidence:
        st.markdown(muted_callout("No SHAP evidence available."), unsafe_allow_html=True)
        return

    supporting = evidence.get("supporting", [])
    counter = evidence.get("counter", [])

    if supporting:
        st.markdown(
            f'<div style="font-size:{FONT_SM};font-weight:{WEIGHT_SEMIBOLD};color:{t["success"]};text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:6px">Supporting Evidence</div>',
            unsafe_allow_html=True,
        )
        rows = []
        for s in supporting:
            rows.append({
                "Feature": s["label"],
                "Value": s["description"],
                "SHAP": f'{s["shap"]:+.4f}',
                "Impact": s["magnitude"],
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            key="shap_supporting",
        )

    if counter:
        st.markdown(
            f'<div style="font-size:{FONT_SM};font-weight:{WEIGHT_SEMIBOLD};color:{t["danger"]};text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:6px;margin-top:10px">Counter Evidence</div>',
            unsafe_allow_html=True,
        )
        rows = []
        for c in counter:
            rows.append({
                "Feature": c["label"],
                "Value": c["description"],
                "SHAP": f'{c["shap"]:+.4f}',
                "Impact": c["magnitude"],
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            key="shap_counter",
        )

    if not supporting and not counter:
        st.markdown(muted_callout("No SHAP contributions found."), unsafe_allow_html=True)
