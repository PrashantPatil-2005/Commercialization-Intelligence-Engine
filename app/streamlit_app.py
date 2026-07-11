"""
Commercialization Intelligence Engine - Enterprise Dashboard

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.styles import inject_css, PAGE_CONFIG, PROCESSED_DIR
from app.components import muted_callout
from models.decision_engine import run_decision_engine
from models.insight_layer import build_insights

FEATURES_PATH = PROCESSED_DIR / "concept_features.csv"


st.set_page_config(**PAGE_CONFIG)


@st.cache_data(ttl=300, show_spinner="Running pipeline...")
def load_pipeline():
    if not FEATURES_PATH.exists():
        return None
    report, artifacts = run_decision_engine(FEATURES_PATH, PROCESSED_DIR)
    insights = build_insights(artifacts)
    return report, artifacts, insights


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="padding:4px 0">'
            '<div style="font-size:13px;font-weight:600;color:#e6edf3">CIE</div>'
            '<div style="font-size:10px;color:#8b949e">Commercialization Intelligence Engine</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="border-color:#30363d;margin:6px 0">', unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Overview", "Portfolio", "Explorer", "Analytics", "Model"],
            label_visibility="collapsed",
        )

        st.markdown('<hr style="border-color:#30363d;margin:6px 0">', unsafe_allow_html=True)

        if st.button("Regenerate Data", key="sidebar_regen"):
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

        st.markdown(
            '<div style="position:fixed;bottom:12px;left:12px;font-size:9px;color:#484f58">'
            "v2.0 &middot; Python, scikit-learn, SHAP"
            "</div>",
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
