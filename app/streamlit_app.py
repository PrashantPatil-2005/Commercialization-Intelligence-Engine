"""
Commercialization Intelligence Engine — Streamlit Dashboard

A polished, human-centered dashboard that guides stakeholders through
AI-powered commercialization decisions for early-stage product concepts.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.decision_engine import run_decision_engine
from models.insight_layer import FEATURE_LABELS, build_insights, generate_executive_summary

PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "concept_features.csv"

OUTCOME_META = {
    "MVP Build": {"color": "#10b981", "icon": "🚀", "desc": "Strong signal — build a focused prototype for validation"},
    "Customer Pilot": {"color": "#3b82f6", "icon": "🤝", "desc": "Clear customer pull — ready to test with real customers"},
    "Reusable Asset": {"color": "#8b5cf6", "icon": "♻️", "desc": "Repeatable demand — potential to scale across segments"},
    "Incubate": {"color": "#f59e0b", "icon": "🧪", "desc": "Promising but needs more evidence before committing"},
    "Archive": {"color": "#ef4444", "icon": "📦", "desc": "Weak signal — deprioritize and reallocate resources"},
}

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Commercialization Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --text: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --accent: #6366f1;
    --accent-light: #eef2ff;
    --green: #10b981;
    --blue: #3b82f6;
    --purple: #8b5cf6;
    --amber: #f59e0b;
    --red: #ef4444;
    --radius: 12px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.05), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 15px rgba(0,0,0,0.06), 0 4px 6px rgba(0,0,0,0.04);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stMarkdown a {
    color: #818cf8 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.01em;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99,102,241,0.4);
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: var(--radius);
    padding: 2rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-lg);
}
.hero-banner h1 {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.hero-banner p {
    font-size: 0.92rem;
    opacity: 0.9;
    margin: 0;
    line-height: 1.5;
    max-width: 700px;
}
.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.8rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    transition: all 0.2s ease;
    height: 100%;
}
.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
}
.card-hint {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
    line-height: 1.4;
}
.card-accent {
    border-left: 4px solid var(--accent);
}
.card-accent.green { border-left-color: var(--green); }
.card-accent.blue { border-left-color: var(--blue); }
.card-accent.purple { border-left-color: var(--purple); }
.card-accent.amber { border-left-color: var(--amber); }
.card-accent.red { border-left-color: var(--red); }

/* ── Outcome Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    color: white;
    letter-spacing: 0.01em;
    white-space: nowrap;
}

/* ── Info Callouts ── */
.callout {
    background: var(--accent-light);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #4338ca;
    line-height: 1.5;
    margin-bottom: 1rem;
}
.callout-warn {
    background: #fffbeb;
    border-left-color: var(--amber);
    color: #92400e;
}
.callout-success {
    background: #ecfdf5;
    border-left-color: var(--green);
    color: #065f46;
}

/* ── Section Headers ── */
.section-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 14px;
    background: var(--accent);
    border-radius: 2px;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Narrative & Evidence ── */
.narrative-box {
    background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
    border-left: 3px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    font-size: 0.88rem;
    color: #312e81;
    line-height: 1.65;
}
.evidence-box {
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.83rem;
    color: #475569;
    line-height: 1.55;
}

/* ── Progress Bar ── */
.progress-track {
    background: #e2e8f0;
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.progress-meta {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-top: 0.4rem;
}
.progress-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
}
.progress-score {
    font-size: 1.1rem;
    font-weight: 700;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #f1f5f9;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
    font-size: 0.85rem;
    border: none;
    transition: all 0.15s ease;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Table ── */
.stDataFrame {
    border-radius: var(--radius);
    overflow: hidden;
}

/* ── Divider ── */
hr.custom-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── Tooltip ── */
.tooltip-trigger {
    cursor: help;
    border-bottom: 1px dashed var(--text-muted);
}
.info-dot {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #e2e8f0;
    color: var(--text-muted);
    font-size: 0.6rem;
    font-weight: 700;
    cursor: help;
    margin-left: 4px;
    vertical-align: middle;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--text-muted);
    font-size: 0.75rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}
.footer a {
    color: var(--accent);
    text-decoration: none;
}
.footer a:hover {
    text-decoration: underline;
}

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeIn 0.4s ease-out;
}
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipeline():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Data not found. To generate it, run:\n\n"
            "```bash\n"
            "python data/generate_mock_data.py\n"
            "python data/prepare_features.py\n"
            "```"
        )
    report, artifacts = run_decision_engine(FEATURES_PATH, PROCESSED_DIR)
    insights = build_insights(artifacts)
    return report, artifacts, insights


def badge_html(outcome: str) -> str:
    meta = OUTCOME_META.get(outcome, OUTCOME_META["Archive"])
    return f'<span class="badge" style="background:{meta["color"]}">{meta["icon"]} {outcome}</span>'


def card_html(label: str, value: str, hint: str = "", accent: str = "") -> str:
    cls = f"card card-accent {accent}" if accent else "card card-accent"
    hint_html = f'<div class="card-hint">{hint}</div>' if hint else ""
    return f"""
    <div class="{cls}">
        <div class="card-label">{label}</div>
        <div class="card-value">{value}</div>
        {hint_html}
    </div>
    """


def progress_html(score: float) -> str:
    if score >= 60:
        color = "#10b981"
        label = "Strong signal"
    elif score >= 45:
        color = "#f59e0b"
        label = "Moderate signal"
    else:
        color = "#ef4444"
        label = "Weak signal"
    return f"""
    <div class="progress-track">
        <div class="progress-fill" style="width:{score}%;background:{color}"></div>
    </div>
    <div class="progress-meta">
        <span class="progress-label">{label}</span>
        <span class="progress-score" style="color:{color}">{score:.1f} / 100</span>
    </div>
    """


def info_callout(text: str) -> str:
    return f'<div class="callout">💡 {text}</div>'


def success_callout(text: str) -> str:
    return f'<div class="callout callout-success">✅ {text}</div>'


# ---------------------------------------------------------------------------
# Chart builders (polished)
# ---------------------------------------------------------------------------

def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)


def plot_readiness_bar(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#fafbfc")
    ax.set_facecolor("#fafbfc")
    ordered = df.sort_values("readiness_score", ascending=True)
    colors = [OUTCOME_META.get(o, OUTCOME_META["Archive"])["color"] for o in ordered["recommended_outcome"]]
    bars = ax.barh(ordered["concept_name"], ordered["readiness_score"], color=colors, height=0.62, edgecolor="white", linewidth=0.5)
    for bar, score in zip(bars, ordered["readiness_score"]):
        ax.text(bar.get_width() + 0.7, bar.get_y() + bar.get_height() / 2, f"{score:.1f}", va="center", fontsize=8, fontweight="600", color="#475569")
    ax.set_xlabel("Readiness Score (1-100)", fontsize=10, color="#475569", labelpad=8)
    ax.set_title("Which concepts deserve investment?", fontsize=13, fontweight="bold", color="#1e293b", pad=14)
    ax.set_xlim(0, 108)
    _clean_ax(ax)
    plt.tight_layout()
    return fig


def plot_outcome_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    fig.patch.set_facecolor("#fafbfc")
    counts = df["recommended_outcome"].value_counts()
    colors = [OUTCOME_META.get(o, OUTCOME_META["Archive"])["color"] for o in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, autopct=lambda p: f"{p:.0f}%" if p > 5 else "",
        colors=colors, startangle=140, pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2.5),
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
        t.set_color("white")
    legend_labels = [f"{OUTCOME_META.get(o,{}).get('icon','')} {o} ({c})" for o, c in zip(counts.index, counts.values)]
    ax.legend(legend_labels, loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=7.5, frameon=False, labelspacing=0.7)
    ax.set_title("How is the portfolio split?", fontsize=13, fontweight="bold", color="#1e293b", pad=14)
    plt.tight_layout()
    return fig


def plot_cluster_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor("#fafbfc")
    ax.set_facecolor("#fafbfc")
    palette = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]
    for cid, group in df.groupby("cluster_id"):
        color = palette[cid % len(palette)]
        ax.scatter(
            group["demand_intensity"], group["feasibility_risk"],
            s=group["readiness_score"] * 5.5, alpha=0.65,
            label=group["cluster_profile"].iloc[0],
            color=color, edgecolors="white", linewidth=1,
        )
        for _, row in group.iterrows():
            name = row["concept_name"].split()[0]
            ax.annotate(name, (row["demand_intensity"], row["feasibility_risk"]),
                        fontsize=7.5, alpha=0.9, color="#334155", fontweight="500")
    ax.set_xlabel("Demand Intensity (higher = more customer pull)", fontsize=9.5, color="#475569", labelpad=8)
    ax.set_ylabel("Feasibility Risk (higher = harder to deliver)", fontsize=9.5, color="#475569", labelpad=8)
    ax.set_title("Concepts grouped by demand vs delivery effort", fontsize=13, fontweight="bold", color="#1e293b", pad=14)
    ax.legend(loc="best", fontsize=7.5, frameon=True, fancybox=True, framealpha=0.95, edgecolor="#e2e8f0")
    _clean_ax(ax)
    ax.grid(True, alpha=0.25, color="#cbd5e1", linestyle="--")
    plt.tight_layout()
    return fig


def plot_feature_importance(importance: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafbfc")
    ax.set_facecolor("#fafbfc")
    top = importance.head(8).copy()
    top["label"] = top["feature"].map(FEATURE_LABELS).fillna(top["feature"])
    palette = sns.color_palette("viridis_r", n_colors=len(top))
    bars = ax.barh(top["label"], top["importance"], color=palette, height=0.58, edgecolor="white", linewidth=0.5)
    for bar, imp in zip(bars, top["importance"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{imp:.1%}", va="center", fontsize=8, fontweight="500", color="#475569")
    ax.set_xlabel("Relative Importance", fontsize=10, color="#475569", labelpad=8)
    ax.set_title("What drives the decisions?", fontsize=13, fontweight="bold", color="#1e293b", pad=14)
    ax.invert_yaxis()
    _clean_ax(ax)
    plt.tight_layout()
    return fig


def plot_shap_waterfall(row: pd.Series, feature_names: list[str]):
    shap_cols = [f"shap_{f}" for f in feature_names if f"shap_{f}" in row.index]
    if not shap_cols:
        return None
    labels = [FEATURE_LABELS.get(c.replace("shap_", ""), c) for c in shap_cols]
    values = [row[c] for c in shap_cols]
    pairs = sorted(zip(labels, values), key=lambda x: abs(x[1]), reverse=True)[:8]
    labels, values = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#fafbfc")
    ax.set_facecolor("#fafbfc")
    colors = ["#10b981" if v > 0 else "#ef4444" for v in values]
    bars = ax.barh(labels, values, color=colors, height=0.52, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, values):
        offset = 0.002 if val >= 0 else -0.002
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", fontsize=7.5, fontweight="500",
                color="#10b981" if val > 0 else "#ef4444",
                ha="left" if val >= 0 else "right")
    ax.set_xlabel("SHAP contribution (green = supports, red = weakens)", fontsize=9.5, color="#475569", labelpad=8)
    ax.set_title(f"Why this concept got: {row['recommended_outcome']}", fontsize=12, fontweight="bold", color="#1e293b", pad=14)
    ax.invert_yaxis()
    _clean_ax(ax)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(report, artifacts):
    with st.sidebar:
        st.markdown("# 🧠 CIE")
        st.markdown("**Commercialization Intelligence Engine**")
        st.markdown("*Turning customer signals into investment decisions*")
        st.markdown("---")

        st.markdown("#### 📊 Portfolio Snapshot")
        ranked = artifacts["ranked"]
        top = ranked.iloc[0]
        st.markdown(f"**{report['concepts_scored']}** concepts across **7** industries")
        st.markdown(f"**Top pick:** {top['concept_name']}")
        n_forward = sum(1 for o in ranked["recommended_outcome"] if o in ("MVP Build", "Customer Pilot", "Reusable Asset"))
        st.markdown(f"**{n_forward}** worth advancing")

        st.markdown("---")
        st.markdown("#### ⚡ Quick Actions")
        if st.button("🔄 Regenerate Fresh Data", use_container_width=True):
            with st.spinner("Generating new dataset..."):
                subprocess.run([sys.executable, str(ROOT / "data" / "generate_mock_data.py")], cwd=str(ROOT))
                subprocess.run([sys.executable, str(ROOT / "data" / "prepare_features.py")], cwd=str(ROOT))
            st.cache_data.clear()
            st.success("Done! Refresh the page to see new results.")
            st.rerun()

        st.markdown("---")
        st.markdown("#### ℹ️ About This Tool")
        st.markdown(
            "This engine uses machine learning to analyze customer behavior signals "
            "and recommend commercialization actions for early-stage AI product concepts."
        )
        st.markdown("")
        st.markdown(
            "**How to use:**\n"
            "1. Start with **Overview** for the big picture\n"
            "2. Check **Analytics** to see patterns\n"
            "3. Dive into **Concept Explorer** for individual decisions\n"
            "4. Review **Model Report** for technical details"
        )
        st.markdown("---")
        st.markdown(
            f"**v2.0** · Built with Python, scikit-learn, SHAP\n"
            f"© 2025 PrashantPatil-2005"
        )


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def tab_overview(report, artifacts, insights):
    ranked = insights.sort_values("portfolio_rank")

    # --- Hero ---
    st.markdown("""
    <div class="hero-banner fade-in">
        <h1>Commercialization Intelligence Engine</h1>
        <p>We analyzed customer signals across 12 AI product concepts — demo feedback, sandbox usage, commercial intent, and text comments — to recommend which ones deserve your team's investment.</p>
        <div class="hero-tag">AI-Powered Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    # --- How to read this section ---
    st.markdown(info_callout(
        "Below you'll find a quick snapshot of the portfolio. "
        "Each concept is scored from 1-100 based on demand strength, engagement depth, "
        "revenue potential, feasibility, and customer sentiment."
    ))

    # --- KPI Cards ---
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(card_html(
            "Concepts Analyzed",
            str(report["concepts_scored"]),
            "Total AI product ideas in portfolio",
            "blue"
        ), unsafe_allow_html=True)
    with c2:
        top_score = ranked.iloc[0]["readiness_score"]
        st.markdown(card_html(
            "Top Readiness Score",
            f"{top_score:.1f}",
            f"{ranked.iloc[0]['concept_name']}",
            "green"
        ), unsafe_allow_html=True)
    with c3:
        outcomes = report["classifier"]["outcome_distribution"]
        n_forward = outcomes.get("MVP Build", 0) + outcomes.get("Customer Pilot", 0) + outcomes.get("Reusable Asset", 0)
        st.markdown(card_html(
            "Ready to Advance",
            str(n_forward),
            "MVP Build + Pilot + Reusable Asset",
            "purple"
        ), unsafe_allow_html=True)
    with c4:
        n_archive = len(report.get("archive_candidates", []))
        st.markdown(card_html(
            "Archive Candidates",
            str(n_archive),
            "Weak signal — consider deprioritizing",
            "red"
        ), unsafe_allow_html=True)
    with c5:
        cv = report.get("cross_validation", {})
        acc = cv.get("accuracy_mean", 0)
        st.markdown(card_html(
            "Model Accuracy",
            f"{acc:.0%}",
            f"{cv.get('method', 'N/A')} cross-validation",
            "amber"
        ), unsafe_allow_html=True)

    st.markdown("")

    # --- Executive Summary ---
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown(info_callout(
        "This is a plain-English summary of what the AI recommends. "
        "Read this first if you're short on time."
    ))

    summary = generate_executive_summary(artifacts["ranked"])
    # Convert to styled display
    for line in summary.split("\n"):
        if line.startswith("---"):
            st.markdown("---")
        elif line.startswith("Executive Summary"):
            st.markdown(f"**{line}**")
        elif line.startswith("The portfolio"):
            st.markdown(line)
        elif line.startswith("  #"):
            st.markdown(f"**{line.strip()}**")
        elif line.startswith("  Action:"):
            st.markdown(f"*{line.strip()}*")
        elif line.strip():
            st.markdown(line)

    st.markdown("")

    # --- Ranked Portfolio ---
    st.markdown('<div class="section-title">Ranked Portfolio</div>', unsafe_allow_html=True)
    st.markdown(info_callout(
        "Every concept is ranked by readiness score. Green progress bars mean strong signals. "
        "Outcome badges show what the ML model recommends doing next."
    ))

    display_df = ranked[[
        "portfolio_rank", "concept_name", "industry", "readiness_score",
        "confidence_score", "recommended_outcome",
    ]].copy()
    display_df["recommended_outcome"] = display_df["recommended_outcome"].apply(
        lambda o: f"{OUTCOME_META.get(o,{}).get('icon','')} {o}"
    )

    st.dataframe(
        display_df.rename(columns={
            "portfolio_rank": "#",
            "concept_name": "Concept",
            "industry": "Industry",
            "readiness_score": "Readiness",
            "confidence_score": "Confidence",
            "recommended_outcome": "Recommended Action",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Readiness": st.column_config.ProgressColumn(min_value=0, max_value=100, width="medium"),
            "Confidence": st.column_config.NumberColumn(format="%.0%%", width="small"),
        },
    )


# ---------------------------------------------------------------------------
# Tab: Analytics
# ---------------------------------------------------------------------------

def tab_analytics(artifacts, insights):
    ranked = insights.sort_values("portfolio_rank")

    st.markdown(info_callout(
        "These charts help you see patterns across the portfolio. "
        "Bigger dots in the cluster chart mean higher readiness scores."
    ))

    st.markdown('<div class="section-title">Portfolio Analytics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Readiness Ranking** — *Which concepts are closest to ready?*")
        st.pyplot(plot_readiness_bar(ranked))
    with c2:
        st.markdown("**Outcome Split** — *How is the portfolio distributed?*")
        st.pyplot(plot_outcome_distribution(ranked))

    st.markdown("")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Demand vs Effort** — *Natural groupings of concepts*")
        st.pyplot(plot_cluster_scatter(artifacts["df"]))
    with c4:
        st.markdown("**Feature Importance** — *What matters most for decisions?*")
        st.pyplot(plot_feature_importance(artifacts["importance"]))


# ---------------------------------------------------------------------------
# Tab: Concept Explorer
# ---------------------------------------------------------------------------

def tab_explorer(artifacts, insights):
    ranked = insights.sort_values("portfolio_rank")
    feature_names = artifacts["feature_columns"]

    st.markdown(info_callout(
        "Select any concept to see exactly why the AI made its recommendation. "
        "The SHAP waterfall chart shows which factors pushed the decision."
    ))

    concept_names = ranked["concept_name"].tolist()
    selected = st.selectbox(
        "Choose a concept to explore",
        concept_names,
        label_visibility="collapsed",
        placeholder="Search concepts...",
    )
    row = ranked[ranked["concept_name"] == selected].iloc[0]

    # --- Concept Header ---
    meta = OUTCOME_META.get(row["recommended_outcome"], OUTCOME_META["Archive"])
    st.markdown(f"### {row['concept_name']} {badge_html(row['recommended_outcome'])}", unsafe_allow_html=True)
    st.markdown(f"*{row['industry']} · {row['cluster_profile']}*")

    st.markdown("")

    # --- Top Metrics ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(card_html("Readiness Score", f"{row['readiness_score']:.1f}", f"Rank #{int(row['portfolio_rank'])} of {len(ranked)}", "blue"), unsafe_allow_html=True)
    with m2:
        st.markdown(card_html("Confidence", f"{row['confidence_score']:.0%}", "How certain the model is", "green"), unsafe_allow_html=True)
    with m3:
        st.markdown(card_html("Recommended Action", f"{meta['icon']} {row['recommended_outcome']}", meta["desc"], "purple"), unsafe_allow_html=True)

    st.markdown("")

    # --- Readiness Bar ---
    st.markdown("#### Readiness Level")
    st.markdown(progress_html(row["readiness_score"]), unsafe_allow_html=True)

    st.markdown("")

    # --- Narrative + Evidence Side by Side ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### What the AI Says")
        st.markdown(info_callout("This is a plain-English explanation of the recommendation, generated from SHAP feature contributions."))
        st.markdown(f'<div class="narrative-box">{row["ai_narrative"]}</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown("#### Key Evidence")
        st.markdown(info_callout("These are the strongest signals — both positive and negative — that influenced this decision."))
        st.markdown(f'<div class="evidence-box">{row.get("key_evidence", "---")}</div>', unsafe_allow_html=True)

    st.markdown("")

    # --- SHAP Waterfall ---
    st.markdown("#### Feature Contributions (SHAP)")
    st.markdown(info_callout(
        "Each bar shows how much a feature pushed the prediction. "
        "Green bars support the recommended outcome; red bars weaken it. "
        "Longer bars = stronger influence."
    ))
    fig = plot_shap_waterfall(row, feature_names)
    if fig:
        st.pyplot(fig)

    st.markdown("")

    # --- Raw Features ---
    with st.expander("📋 View All Raw Feature Values", expanded=False):
        st.markdown("These are the engineered feature values fed into the ML models. All values are normalized to 0-1 unless noted.")
        feat_data = {f: round(float(row[f]), 4) for f in feature_names if f in row.index}
        st.json(feat_data)


# ---------------------------------------------------------------------------
# Tab: Model Report
# ---------------------------------------------------------------------------

def tab_model_report():
    import json

    st.markdown(info_callout(
        "This tab shows the technical details of the ML pipeline — "
        "how the models were configured, how well they performed, and what the data looks like."
    ))

    report_path = PROCESSED_DIR / "model_report.json"
    val_path = PROCESSED_DIR / "validation_report.json"

    if not report_path.exists():
        st.warning("model_report.json not found. Run the pipeline first.")
        return

    with open(report_path) as f:
        model_report = json.load(f)

    # --- Cross-Validation ---
    st.markdown('<div class="section-title">Cross-Validation Results</div>', unsafe_allow_html=True)
    cv = model_report.get("cross_validation", {})
    if cv:
        st.markdown("Cross-validation tests how well the model generalizes. With only 12 concepts, these numbers are illustrative — not statistically rigorous.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Method", cv.get("method", "N/A"))
        c2.metric("Mean Accuracy", f"{cv.get('accuracy_mean', 0):.2%}")
        c3.metric("Std Dev", f"\u00b1{cv.get('accuracy_std', 0):.2%}")

        folds = cv.get("fold_scores", [])
        if folds:
            st.markdown(f"**Individual fold scores:** {', '.join(f'{s:.0%}' for s in folds)}")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.bar(range(1, len(folds) + 1), folds, color="#6366f1", edgecolor="white", width=0.5)
            ax.set_xlabel("Fold", fontsize=9)
            ax.set_ylabel("Accuracy", fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_xticks(range(1, len(folds) + 1))
            _clean_ax(ax)
            st.pyplot(fig)

    st.markdown("")

    # --- Model Configuration ---
    st.markdown('<div class="section-title">Model Configuration</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Baseline Weighted Score**")
        st.markdown("A deterministic scoring function with domain-informed weights. Higher weight = more influence on the final score.")
        weights = model_report.get("baseline_model", {}).get("weights", {})
        weights_df = pd.DataFrame([{"Feature": k.replace("_", " ").title(), "Weight": v} for k, v in weights.items()])
        weights_df = weights_df.sort_values("Weight", ascending=False)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Random Forest Classifier**")
        st.markdown("An ensemble of 200 decision trees that predicts the outcome class. Trained on synthetic pseudo-labels.")
        classifier = model_report.get("classifier", {})
        st.markdown(f"- **Trees:** {classifier.get('n_estimators', 'N/A')}")
        st.markdown(f"- **Training labels:** {classifier.get('training_labels', 'N/A')}")
        st.markdown("**Outcome distribution:**")
        for outcome, count in classifier.get("outcome_distribution", {}).items():
            meta = OUTCOME_META.get(outcome, {})
            st.markdown(f"  {meta.get('icon', '')} {outcome}: **{count}** concepts")

    st.markdown("")

    # --- Feature Importance ---
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown("Which features matter most to the Random Forest? Higher importance = more influence on predictions.")
    top_feats = model_report.get("top_features", [])
    if top_feats:
        feats_df = pd.DataFrame(top_feats)
        feats_df["label"] = feats_df["feature"].map(FEATURE_LABELS).fillna(feats_df["feature"])
        st.dataframe(
            feats_df[["label", "importance"]].rename(columns={"label": "Feature", "importance": "Importance"}),
            use_container_width=True, hide_index=True
        )

    st.markdown("")

    # --- Cluster Profiles ---
    cluster_path = PROCESSED_DIR / "cluster_summary.csv"
    if cluster_path.exists():
        st.markdown('<div class="section-title">Cluster Profiles</div>', unsafe_allow_html=True)
        st.markdown("K-Means groups concepts into natural clusters based on demand and effort patterns. No outcome labels used.")
        clusters = pd.read_csv(cluster_path)
        st.dataframe(clusters, use_container_width=True, hide_index=True)

    st.markdown("")

    # --- Data Quality ---
    if val_path.exists():
        st.markdown('<div class="section-title">Data Quality Report</div>', unsafe_allow_html=True)
        with open(val_path) as f:
            val = json.load(f)
        st.markdown(f"Schema validation: {'✅ Passed' if val.get('validation_passed') else '❌ Failed'}")
        st.markdown(f"Concepts: **{val.get('concept_count')}** | Interactions: **{val.get('interaction_count')}**")
        feat_summary = val.get("feature_summary", {})
        if feat_summary:
            feat_stats = pd.DataFrame(feat_summary).T
            feat_stats.columns = ["Mean", "Min", "Max"]
            st.dataframe(feat_stats, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        report, artifacts, insights = load_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(report, artifacts)

    t_overview, t_analytics, t_explorer, t_report = st.tabs([
        "📊 Overview", "📈 Analytics", "🔍 Concept Explorer", "📋 Model Report"
    ])

    with t_overview:
        tab_overview(report, artifacts, insights)

    with t_analytics:
        tab_analytics(artifacts, insights)

    with t_explorer:
        tab_explorer(artifacts, insights)

    with t_report:
        tab_model_report()

    # --- Footer ---
    st.markdown("""
    <div class="footer">
        <strong>Commercialization Intelligence Engine</strong> v2.0 · Built with Python, scikit-learn, SHAP, Streamlit<br>
        © 2025 PrashantPatil-2005 · <a href="https://github.com/PrashantPatil-2005" target="_blank">GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
