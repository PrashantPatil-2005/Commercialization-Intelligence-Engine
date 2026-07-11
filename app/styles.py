"""Enterprise-grade CSS for the Commercialization Intelligence Engine."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

PAGE_CONFIG = {
    "page_title": "CIE - Commercialization Intelligence Engine",
    "page_icon": None,
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Outcome metadata: color, label
OUTCOME_META = {
    "MVP Build":       {"color": "#0e9f6e", "label": "MVP Build"},
    "Customer Pilot":  {"color": "#1f6feb", "label": "Customer Pilot"},
    "Reusable Asset":  {"color": "#8957e5", "label": "Reusable Asset"},
    "Incubate":        {"color": "#bf8700", "label": "Incubate"},
    "Archive":         {"color": "#cf222e", "label": "Archive"},
}

CLUSTER_COLORS = ["#1f6feb", "#0e9f6e", "#bf8700", "#cf222e", "#8957e5", "#e16f24"]

EnterpriseCSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #f6f8fa;
    --surface: #ffffff;
    --surface-alt: #f6f8fa;
    --border: #d1d9e0;
    --border-light: #e8ecf0;
    --text: #1f2328;
    --text-secondary: #656d76;
    --text-tertiary: #8b949e;
    --blue: #0969da;
    --blue-subtle: #ddf4ff;
    --green: #0e9f6e;
    --green-subtle: #dafbe1;
    --red: #cf222e;
    --red-subtle: #ffebe9;
    --orange: #bf8700;
    --orange-subtle: #fff8c5;
    --purple: #8957e5;
    --purple-subtle: #fbefff;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --mono: 'SFMono-Regular', 'Consolas', 'Liberation Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font);
    color: var(--text);
    font-size: 13px;
    line-height: 1.5;
}

/* Streamlit Overrides */
.stDeployButton { display: none; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1b1f24;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #8b949e !important;
    font-size: 11px;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #21262d;
    color: #c9d1d9 !important;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 500;
    padding: 4px 10px;
    width: 100%;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #30363d;
    border-color: #484f58;
}
section[data-testid="stSidebar"] hr {
    border-color: #30363d !important;
    margin: 6px 0;
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 2px;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 12px;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: #21262d;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: #1f6feb22;
    color: #58a6ff !important;
}

/* Page Header */
.page-header {
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.page-header h2 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 4px 0;
}
.page-header p {
    font-size: 12px;
    color: var(--text-secondary);
    margin: 0;
}

/* KPI Cards */
.kpi-row {
    display: flex;
    gap: 10px;
    margin-bottom: 16px;
}
.kpi-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    min-width: 0;
}
.kpi-card .kpi-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-tertiary);
    margin-bottom: 2px;
}
.kpi-card .kpi-value {
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
}
.kpi-card .kpi-sub {
    font-size: 10px;
    color: var(--text-tertiary);
    margin-top: 1px;
}
.kpi-card.accent-blue { border-left: 3px solid var(--blue); }
.kpi-card.accent-green { border-left: 3px solid var(--green); }
.kpi-card.accent-purple { border-left: 3px solid var(--purple); }
.kpi-card.accent-orange { border-left: 3px solid var(--orange); }
.kpi-card.accent-red { border-left: 3px solid var(--red); }

/* Section Headers */
.section-header {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
    margin: 16px 0 8px 0;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    line-height: 1.6;
    white-space: nowrap;
}

/* Callouts */
.callout {
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
    line-height: 1.5;
    margin-bottom: 8px;
    border: 1px solid;
}
.callout-info {
    background: var(--blue-subtle);
    border-color: #54aeff;
    color: #0969da;
}
.callout-warn {
    background: var(--orange-subtle);
    border-color: #d4a72c;
    color: #9a6700;
}
.callout-muted {
    background: var(--surface-alt);
    border-color: var(--border);
    color: var(--text-secondary);
}

/* Decision Card */
.decision-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.decision-card .dc-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-tertiary);
    margin-bottom: 4px;
}
.decision-card .dc-value {
    font-size: 13px;
    color: var(--text);
    line-height: 1.6;
}

/* Evidence List */
.evidence-list {
    list-style: none;
    margin: 0;
    padding: 0;
}
.evidence-list li {
    font-size: 12px;
    color: var(--text-secondary);
    padding: 3px 0;
    border-bottom: 1px solid var(--border-light);
    line-height: 1.5;
}
.evidence-list li:last-child {
    border-bottom: none;
}

/* Progress Bar */
.progress-track {
    background: #e8ecf0;
    border-radius: 3px;
    height: 5px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 3px;
}

/* Container */
.section-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    margin-bottom: 12px;
}

/* Divider */
hr.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 8px 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid var(--border);
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 12px;
    font-weight: 500;
    padding: 6px 14px;
    border: none;
    border-bottom: 2px solid transparent;
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: #0969da !important;
    background: transparent !important;
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Expander */
.stExpander {
    border: 1px solid var(--border);
    border-radius: 6px;
}

/* Selectbox */
.stSelectbox > div > div {
    border-radius: 6px;
    border-color: var(--border);
    font-size: 12px;
}

/* Charts */
.stPlotlyChart, .stPyplotFigure {
    border: 1px solid var(--border-light);
    border-radius: 6px;
    padding: 2px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d1d9e0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #afb8c1; }

/* Remove Streamlit branding */
#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
.stFooter { display: none !important; }
</style>
"""


def inject_css():
    """Inject enterprise CSS into the Streamlit page."""
    import streamlit as st
    st.markdown(EnterpriseCSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> str:
    """Return HTML for a page header."""
    sub_html = f'<p>{subtitle}</p>' if subtitle else ""
    return f'<div class="page-header"><h2>{title}</h2>{sub_html}</div>'


def section_header(text: str) -> str:
    """Return HTML for a section header."""
    return f'<div class="section-header">{text}</div>'


def kpi_card(label: str, value: str, sub: str = "", accent: str = "") -> str:
    """Return HTML for a compact KPI card."""
    cls = f"kpi-card accent-{accent}" if accent else "kpi-card"
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f'<div class="{cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{sub_html}</div>'


def progress_bar(score: float, height: int = 5) -> str:
    """Return HTML for a minimal progress bar."""
    if score >= 60:
        color = "#0e9f6e"
    elif score >= 45:
        color = "#bf8700"
    else:
        color = "#cf222e"
    return f'<div class="progress-track" style="height:{height}px"><div class="progress-fill" style="width:{score}%;background:{color}"></div></div>'


def outcome_badge(outcome: str) -> str:
    """Return HTML badge for an outcome."""
    meta = OUTCOME_META.get(outcome, OUTCOME_META["Archive"])
    color = meta["color"]
    return f'<span class="badge" style="background:{color}15;color:{color};border:1px solid {color}30">{meta["label"]}</span>'


def risk_badge(level: str) -> str:
    """Return HTML badge for risk level."""
    colors = {
        "High": ("#cf222e", "#ffebe9"),
        "Moderate": ("#bf8700", "#fff8c5"),
        "Low": ("#0e9f6e", "#dafbe1"),
    }
    color, bg = colors.get(level, ("#8b949e", "#f6f8fa"))
    return f'<span class="badge" style="background:{bg};color:{color};border:1px solid {color}30">{level} Risk</span>'


def confidence_badge(confidence: float) -> str:
    """Return HTML badge for confidence level."""
    if confidence >= 0.80:
        return risk_badge("Low").replace("Risk", "Confidence")
    elif confidence >= 0.65:
        return f'<span class="badge" style="background:#fff8c5;color:#9a6700;border:1px solid #9a670030">Moderate Confidence</span>'
    else:
        return f'<span class="badge" style="background:#ffebe9;color:#cf222e;border:1px solid #cf222e30">Low Confidence</span>'


def decision_card(label: str, value: str) -> str:
    """Return HTML for a decision summary card."""
    return f'<div class="decision-card"><div class="dc-label">{label}</div><div class="dc-value">{value}</div></div>'


def info_callout(text: str) -> str:
    return f'<div class="callout callout-info">{text}</div>'


def muted_callout(text: str) -> str:
    return f'<div class="callout callout-muted">{text}</div>'
