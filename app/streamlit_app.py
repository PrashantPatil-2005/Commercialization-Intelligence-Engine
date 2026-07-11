"""
Commercialization Intelligence Engine — Enterprise Dashboard

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.theme import (
    get_theme, set_theme, toggle_theme, is_dark,
    PROCESSED_DIR, PAGE_CONFIG,
    FONT_SM, FONT_MD, FONT_LG,
    WEIGHT_SEMIBOLD,
)
from app.styles import inject_css
from app.components import muted_callout
from models.decision_engine import run_decision_engine
from models.insight_layer import build_insights

FEATURES_PATH = PROCESSED_DIR / "concept_features.csv"


st.set_page_config(**PAGE_CONFIG)

# Initialize theme
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"


@st.cache_data(ttl=300, show_spinner="Running pipeline...")
def load_pipeline():
    if not FEATURES_PATH.exists():
        return None
    report, artifacts = run_decision_engine(FEATURES_PATH, PROCESSED_DIR)
    insights = build_insights(artifacts)
    return report, artifacts, insights


def render_sidebar():
    t = get_theme()

    with st.sidebar:
        # --- Theme Toggle (single button) ---
        icon = "☀️" if is_dark() else "🌙"
        if st.button(icon, key="theme_toggle", help="Toggle theme"):
            toggle_theme()
            st.rerun()

        st.markdown(f'<hr style="border-color:{t["border"]};margin:8px 0">', unsafe_allow_html=True)

        # --- Branding ---
        st.markdown(
            f'<div style="padding:4px 0">'
            f'<div style="font-size:{FONT_MD};font-weight:{WEIGHT_SEMIBOLD};color:{t["text_primary"]}">CIE</div>'
            f'<div style="font-size:{FONT_SM};color:{t["text_secondary"]}">Commercialization Intelligence Engine</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<hr style="border-color:{t["border"]};margin:8px 0">', unsafe_allow_html=True)

        # --- Navigation ---
        page = st.radio(
            "Navigation",
            ["Overview", "Portfolio", "Explorer", "Analytics", "Model"],
            label_visibility="collapsed",
        )

        st.markdown(f'<hr style="border-color:{t["border"]};margin:8px 0">', unsafe_allow_html=True)

        # --- Actions ---
        st.markdown(
            f'<div style="font-size:{FONT_SM};color:{t["text_secondary"]};margin-bottom:4px">Actions</div>',
            unsafe_allow_html=True,
        )
        if st.button("Regenerate Data", key="sidebar_regen", use_container_width=True):
            with st.spinner("Regenerating..."):
                subprocess.run(
                    [sys.executable, str(ROOT / "data" / "generate_mock_data.py")],
                    cwd=str(ROOT),
                )
                subprocess.run(
                    [sys.executable, str(ROOT / "data" / "prepare_features.py")],
                    cwd=str(ROOT),
                )
            st.cache_data.clear()
            st.rerun()

        # --- Footer ---
        st.markdown(
            f'<div style="padding-top:20px;font-size:{FONT_SM};color:{t["text_muted"]};text-align:center">'
            f'v2.0 &middot; Python, scikit-learn, SHAP'
            f'</div>',
            unsafe_allow_html=True,
        )

    return page


def main():
    inject_css()

    data = load_pipeline()
    if data is None:
        st.markdown(
            muted_callout("Data not found. Click 'Regenerate Data' in the sidebar or run the pipeline manually."),
            unsafe_allow_html=True,
        )
        st.stop()

    report, artifacts, insights = data
    page = render_sidebar()

    if page == "Overview":
        from app.pages.overview import render
        render(report, artifacts, insights)
    elif page == "Portfolio":
        from app.pages.portfolio import render
        render(report, artifacts, insights)
    elif page == "Explorer":
        from app.pages.explorer import render
        render(artifacts, insights)
    elif page == "Analytics":
        from app.pages.analytics import render
        render(artifacts, insights)
    elif page == "Model":
        from app.pages.model import render
        render()


if __name__ == "__main__":
    main()
