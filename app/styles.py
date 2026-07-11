"""
Dynamic Theme CSS — Generated from active theme.

Call inject_css() on every page render. CSS is generated from the
active theme dict, so light/dark switching is instant.
"""

import html as html_mod

from app.theme import (
    get_theme, OUTCOME_META,
    FONT_FAMILY,
    FONT_XS, FONT_SM, FONT_BASE, FONT_MD, FONT_LG, FONT_XL, FONT_2XL, FONT_3XL,
    WEIGHT_REGULAR, WEIGHT_MEDIUM, WEIGHT_SEMIBOLD, WEIGHT_BOLD,
    LEADING_TIGHT, LEADING_NORMAL, LEADING_RELAXED,
    RADIUS_MD,
)


def _css(t: dict) -> str:
    """Generate the full CSS string from a theme dict."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ============================================================
   ROOT VARIABLES
   ============================================================ */
:root {{
    --bg: {t["bg_primary"]};
    --bg-s: {t["bg_secondary"]};
    --bg-t: {t["bg_tertiary"]};
    --bg-sidebar: {t["sidebar_bg"]};
    --bg-input: {t["bg_input"]};
    --border: {t["border"]};
    --border-l: {t["border_light"]};
    --border-f: {t["border_focus"]};
    --tx: {t["text_primary"]};
    --tx2: {t["text_secondary"]};
    --txm: {t["text_muted"]};
    --txl: {t["text_link"]};
    --ok: {t["success"]};
    --ok-bg: {t["success_subtle"]};
    --warn: {t["warning"]};
    --warn-bg: {t["warning_subtle"]};
    --err: {t["danger"]};
    --err-bg: {t["danger_subtle"]};
    --info: {t["info"]};
    --info-bg: {t["info_subtle"]};
    --font: {FONT_FAMILY};
    --r: {RADIUS_MD};
}}

/* ============================================================
   BASE
   ============================================================ */
.stApp {{ background: var(--bg) !important; }}
.stApp > header {{ background: transparent !important; }}
html, body, [class*="css"] {{
    font-family: var(--font);
    color: var(--tx);
    font-size: {FONT_LG};
    line-height: {LEADING_NORMAL};
}}
.block-container {{ background: var(--bg) !important; }}

/* ============================================================
   STREAMLIT OVERRIDES
   ============================================================ */
.stDeployButton {{ display: none; }}
#MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
header[data-testid="stHeader"] * {{ visibility: visible !important; }}

/* ============================================================
   SIDEBAR
   ============================================================ */
section[data-testid="stSidebar"] {{
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}}
section[data-testid="stSidebar"] p {{
    color: var(--tx2) !important;
}}
section[data-testid="stSidebar"] label {{
    color: var(--tx2) !important;
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: var(--tx2) !important;
    font-size: {FONT_BASE};
}}
section[data-testid="stSidebar"] .stButton > button {{
    background: var(--bg-t) !important;
    color: var(--tx) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r);
    font-size: {FONT_BASE};
    font-weight: {WEIGHT_MEDIUM};
    padding: 6px 12px;
    width: 100%;
    transition: background 0.15s, border-color 0.15s;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: var(--border) !important;
    border-color: var(--txm) !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: var(--border) !important;
    margin: 8px 0;
}}
section[data-testid="stSidebar"] .stRadio > div {{ gap: 2px; }}
section[data-testid="stSidebar"] .stRadio > div > label {{
    padding: 6px 10px;
    border-radius: var(--r);
    font-size: {FONT_MD};
    transition: background 0.15s;
}}
section[data-testid="stSidebar"] .stRadio > div > label:hover {{
    background: var(--bg-t) !important;
}}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {{
    background: {t["sidebar_active_bg"]} !important;
    color: {t["sidebar_active_text"]} !important;
}}

/* ============================================================
   SIDEBAR HEADER
   ============================================================ */

/* ============================================================
   SIDEBAR FOOTER
   ============================================================ */
.sidebar-footer {{
    padding: 12px 0 4px 0;
    border-top: 1px solid var(--border);
    margin-top: auto;
}}
.sidebar-footer-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 3px 0;
}}
.sidebar-footer-label {{
    font-size: {FONT_XS};
    color: var(--txm);
}}
.sidebar-footer-value {{
    font-size: {FONT_XS};
    color: var(--tx2);
    font-weight: {WEIGHT_MEDIUM};
}}
.sidebar-status-dot {{
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {t["success"]};
    margin-right: 4px;
    vertical-align: middle;
}}

/* ============================================================
   PAGE HEADER
   ============================================================ */
.page-header {{
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 10px;
}}
.page-header h2 {{
    font-size: {FONT_2XL};
    font-weight: {WEIGHT_SEMIBOLD};
    color: var(--tx);
    margin: 0;
}}
.page-header .page-tag {{
    font-size: {FONT_XS};
    font-weight: {WEIGHT_MEDIUM};
    color: var(--txm);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border: 1px solid var(--border);
    border-radius: var(--r);
    white-space: nowrap;
}}
.page-header p {{
    font-size: {FONT_MD};
    color: var(--tx2);
    margin: 0;
    flex-basis: 100%;
}}

/* ============================================================
   KPI CARDS
   ============================================================ */
.kpi-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }}
.kpi-card {{ flex: 1 1 150px; min-width: 150px; }}
.kpi-card {{
    flex: 1;
    background: var(--bg-s);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 12px 14px;
    min-width: 0;
    transition: border-color 0.15s;
}}
.kpi-card:hover {{ border-color: var(--txm); }}
.kpi-card .kpi-label {{
    font-size: {FONT_SM};
    font-weight: {WEIGHT_SEMIBOLD};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--tx2);
    margin-bottom: 4px;
}}
.kpi-card .kpi-value {{
    font-size: {FONT_3XL};
    font-weight: {WEIGHT_BOLD};
    color: var(--tx);
    line-height: {LEADING_TIGHT};
}}
.kpi-card .kpi-sub {{
    font-size: {FONT_SM};
    color: var(--txm);
    margin-top: 2px;
}}
.kpi-card.accent-blue {{ border-left: 3px solid {t["info"]}; }}
.kpi-card.accent-green {{ border-left: 3px solid {t["success"]}; }}
.kpi-card.accent-purple {{ border-left: 3px solid {OUTCOME_META["Reusable Asset"]["color"]}; }}
.kpi-card.accent-orange {{ border-left: 3px solid {t["warning"]}; }}
.kpi-card.accent-red {{ border-left: 3px solid {t["danger"]}; }}

/* ============================================================
   SECTION HEADERS
   ============================================================ */
.section-header {{
    font-size: {FONT_SM};
    font-weight: {WEIGHT_SEMIBOLD};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--tx2);
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
    margin: 20px 0 10px 0;
}}

/* ============================================================
   BADGES
   ============================================================ */
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: {FONT_SM};
    font-weight: {WEIGHT_SEMIBOLD};
    line-height: {LEADING_NORMAL};
    white-space: nowrap;
}}

/* ============================================================
   CALLOUTS
   ============================================================ */
.callout {{
    padding: 8px 12px;
    border-radius: var(--r);
    font-size: {FONT_MD};
    line-height: {LEADING_NORMAL};
    margin-bottom: 8px;
    border: 1px solid;
}}
.callout-info {{
    background: {t["callout_info_bg"]};
    border-color: {t["callout_info_border"]}40;
    color: {t["info"]};
}}
.callout-warn {{
    background: {t["callout_warn_bg"]};
    border-color: {t["callout_warn_border"]}40;
    color: {t["warning"]};
}}
.callout-muted {{
    background: {t["callout_muted_bg"]};
    border-color: {t["callout_muted_border"]};
    color: var(--tx2);
}}

/* ============================================================
   DECISION CARD
   ============================================================ */
.decision-card {{
    background: var(--bg-s);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}}
.decision-card:hover {{ border-color: var(--txm); }}
.decision-card .dc-label {{
    font-size: {FONT_SM};
    font-weight: {WEIGHT_SEMIBOLD};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--tx2);
    margin-bottom: 4px;
}}
.decision-card .dc-value {{
    font-size: {FONT_LG};
    color: var(--tx);
    line-height: {LEADING_RELAXED};
}}

/* ============================================================
   EVIDENCE LIST
   ============================================================ */
.evidence-list {{ list-style: none; margin: 0; padding: 0; }}
.evidence-list li {{
    font-size: {FONT_MD};
    color: var(--tx2);
    padding: 4px 0;
    border-bottom: 1px solid var(--border-l);
    line-height: {LEADING_NORMAL};
}}
.evidence-list li:last-child {{ border-bottom: none; }}

/* ============================================================
   PROGRESS BAR
   ============================================================ */
.progress-track {{
    background: var(--bg-t);
    border-radius: 3px;
    height: 5px;
    overflow: hidden;
}}
.progress-fill {{ height: 100%; border-radius: 3px; }}

/* ============================================================
   SECTION CONTAINER
   ============================================================ */
.section-container {{
    background: var(--bg-s);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 14px 16px;
    margin-bottom: 12px;
}}

/* ============================================================
   DIVIDER
   ============================================================ */
hr.section-divider {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 10px 0;
}}

/* ============================================================
   TABS
   ============================================================ */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: {FONT_MD};
    font-weight: {WEIGHT_MEDIUM};
    padding: 6px 14px;
    border: none;
    border-bottom: 2px solid transparent;
    background: transparent !important;
    color: var(--tx2) !important;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: var(--tx) !important;
    border-bottom-color: var(--info) !important;
    background: transparent !important;
    box-shadow: none !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ============================================================
   EXPANDER
   ============================================================ */
.stExpander {{
    border: 1px solid var(--border);
    border-radius: var(--r);
    background: var(--bg-s);
}}
.stExpander summary {{ color: var(--tx) !important; }}

/* ============================================================
   SELECTBOX / INPUTS
   ============================================================ */
.stSelectbox > div > div {{
    border-radius: var(--r);
    border-color: var(--border);
    font-size: {FONT_MD};
    background: var(--bg-input) !important;
    color: var(--tx) !important;
}}
.stMultiSelect > div > div {{
    border-radius: var(--r);
    border-color: var(--border);
    background: var(--bg-input) !important;
}}

/* ============================================================
   DATAFRAME / TABLE
   ============================================================ */
.stDataFrame {{
    border: 1px solid var(--border);
    border-radius: var(--r);
    overflow: hidden;
}}
[data-testid="stDataFrame"] table {{
    background: var(--bg) !important;
}}
[data-testid="stDataFrame"] th {{
    background: {t["table_header_bg"]} !important;
    color: var(--tx2) !important;
    border-bottom: 1px solid var(--border) !important;
    font-size: {FONT_SM} !important;
    font-weight: {WEIGHT_SEMIBOLD} !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
[data-testid="stDataFrame"] td {{
    background: var(--bg) !important;
    color: var(--tx) !important;
    border-bottom: 1px solid var(--border-l) !important;
    font-size: {FONT_MD} !important;
}}
[data-testid="stDataFrame"] tr:hover td {{
    background: {t["table_row_hover"]} !important;
}}

/* ============================================================
   CHARTS
   ============================================================ */
.stPlotlyChart, .stPyplotFigure {{
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 4px;
    background: var(--bg-s);
}}

/* ============================================================
   SCROLLBAR
   ============================================================ */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {t["scrollbar"]}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {t["scrollbar_hover"]}; }}

/* ============================================================
    FOCUS VISIBLE
    ============================================================ */
.stButton > button:focus-visible,
.stSelectbox:focus-visible,
.stRadio label:focus-visible,
.stMultiSelect:focus-visible {{
    outline: 2px solid var(--border-f);
    outline-offset: 2px;
}}

/* ============================================================
    TOOLTIP
   ============================================================ */
[data-testid="stTooltip"] {{
    background: var(--bg-t) !important;
    color: var(--tx) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-size: {FONT_BASE} !important;
}}

/* ============================================================
   JSON / CODE
   ============================================================ */
.stJson, .stCode {{
    background: var(--bg-s) !important;
    border: 1px solid var(--border);
    border-radius: var(--r);
}}
</style>
"""


def inject_css():
    """Inject theme-aware CSS into the Streamlit page."""
    import streamlit as st
    t = get_theme()
    st.markdown(_css(t), unsafe_allow_html=True)


# ============================================================
# HTML HELPER FUNCTIONS
# ============================================================

def page_header(title: str, subtitle: str = "", tag: str = "") -> str:
    t = get_theme()
    safe_title = html_mod.escape(title)
    safe_sub = html_mod.escape(subtitle) if subtitle else ""
    safe_tag = html_mod.escape(tag) if tag else ""
    tag_html = f'<span class="page-tag">{safe_tag}</span>' if safe_tag else ""
    sub_html = f'<p>{safe_sub}</p>' if safe_sub else ""
    return f'<div class="page-header"><h2>{safe_title}</h2>{tag_html}{sub_html}</div>'


def section_header(text: str) -> str:
    return f'<div class="section-header">{html_mod.escape(text)}</div>'


def kpi_card(label: str, value: str, sub: str = "", accent: str = "") -> str:
    cls = f"kpi-card accent-{accent}" if accent else "kpi-card"
    sub_html = f'<div class="kpi-sub">{html_mod.escape(sub)}</div>' if sub else ""
    return f'<div class="{cls}" role="region" aria-label="{html_mod.escape(label)}"><div class="kpi-label">{html_mod.escape(label)}</div><div class="kpi-value">{html_mod.escape(value)}</div>{sub_html}</div>'


def progress_bar(score: float, height: int = 5) -> str:
    t = get_theme()
    score = max(0.0, min(100.0, score))
    if score >= 55:
        color = t["progress_high"]
    elif score >= 45:
        color = t["progress_moderate"]
    else:
        color = t["progress_low"]
    return (
        f'<div class="progress-track" style="height:{height}px" '
        f'role="progressbar" aria-valuenow="{score:.0f}" aria-valuemin="0" aria-valuemax="100">'
        f'<div class="progress-fill" style="width:{score}%;background:{color}"></div></div>'
    )


def outcome_badge(outcome: str) -> str:
    meta = OUTCOME_META.get(outcome, OUTCOME_META["Archive"])
    color = meta["color"]
    return f'<span class="badge" style="background:{color}18;color:{color};border:1px solid {color}30">{meta["label"]}</span>'


def risk_badge(level: str) -> str:
    t = get_theme()
    colors = {
        "High":     (t["danger"],  t["danger_subtle"]),
        "Moderate": (t["warning"], t["warning_subtle"]),
        "Low":      (t["success"], t["success_subtle"]),
    }
    color, bg = colors.get(level, (t["text_secondary"], t["bg_secondary"]))
    return f'<span class="badge" style="background:{bg};color:{color};border:1px solid {color}30">{level} Risk</span>'


def confidence_badge(confidence: float) -> str:
    t = get_theme()
    if confidence >= 0.80:
        return f'<span class="badge" style="background:{t["success_subtle"]};color:{t["success"]};border:1px solid {t["success"]}30">High Confidence</span>'
    elif confidence >= 0.65:
        return f'<span class="badge" style="background:{t["warning_subtle"]};color:{t["warning"]};border:1px solid {t["warning"]}30">Moderate Confidence</span>'
    else:
        return f'<span class="badge" style="background:{t["danger_subtle"]};color:{t["danger"]};border:1px solid {t["danger"]}30">Low Confidence</span>'


def decision_card(label: str, value: str) -> str:
    return f'<div class="decision-card"><div class="dc-label">{html_mod.escape(label)}</div><div class="dc-value">{value}</div></div>'


def info_callout(text: str) -> str:
    return f'<div class="callout callout-info">{text}</div>'


def muted_callout(text: str) -> str:
    return f'<div class="callout callout-muted">{text}</div>'
