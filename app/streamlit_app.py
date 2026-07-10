"""
Phase 5: Streamlit Presentation Layer

Commercialization Intelligence Engine dashboard.
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
RAW_DIR = ROOT / "data" / "raw"

OUTCOME_COLORS = {
    "MVP Build": "#2ecc71",
    "Customer Pilot": "#3498db",
    "Reusable Asset": "#9b59b6",
    "Incubate": "#f39c12",
    "Archive": "#e74c3c",
}

OUTCOME_ICONS = {
    "MVP Build": "🚀",
    "Customer Pilot": "🤝",
    "Reusable Asset": "♻️",
    "Incubate": "🧪",
    "Archive": "📦",
}

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Commercialization Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* --- Global --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e2e8f0;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

/* --- KPI Cards --- */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    border-left: 4px solid #667eea;
    margin-bottom: 0.5rem;
    transition: transform 0.15s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.kpi-card.green  { border-left-color: #2ecc71; }
.kpi-card.blue   { border-left-color: #3498db; }
.kpi-card.purple { border-left-color: #9b59b6; }
.kpi-card.orange { border-left-color: #f39c12; }
.kpi-card.red    { border-left-color: #e74c3c; }
.kpi-label {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.2;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* --- Outcome badges --- */
.outcome-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
    letter-spacing: 0.02em;
}
.badge-mvp      { background: #2ecc71; }
.badge-pilot    { background: #3498db; }
.badge-asset    { background: #9b59b6; }
.badge-incubate { background: #f39c12; }
.badge-archive  { background: #e74c3c; }

/* --- Section headers --- */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 1rem;
}

/* --- Narrative box --- */
.narrative-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-left: 4px solid #3498db;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-size: 0.9rem;
    color: #334155;
    line-height: 1.6;
}

/* --- Evidence box --- */
.evidence-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.5;
}

/* --- Progress bar wrapper --- */
.progress-outer {
    background: #e2e8f0;
    border-radius: 10px;
    height: 12px;
    overflow: hidden;
}
.progress-inner {
    height: 100%;
    border-radius: 10px;
    transition: width 0.5s ease;
}
.progress-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    margin-top: 0.3rem;
}

/* --- Tab styling --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
}

/* --- Table container --- */
.table-wrapper {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
</style>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipeline():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "concept_features.csv not found. Run:\n"
            "  python data/generate_mock_data.py\n"
            "  python data/prepare_features.py"
        )
    report, artifacts = run_decision_engine(FEATURES_PATH, PROCESSED_DIR)
    insights = build_insights(artifacts)
    return report, artifacts, insights


def outcome_badge_html(outcome: str) -> str:
    badge_cls = {
        "MVP Build": "badge-mvp",
        "Customer Pilot": "badge-pilot",
        "Reusable Asset": "badge-asset",
        "Incubate": "badge-incubate",
        "Archive": "badge-archive",
    }.get(outcome, "badge-archive")
    return f'<span class="outcome-badge {badge_cls}">{outcome}</span>'


def kpi_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    cls = f"kpi-card {color}" if color else "kpi-card"
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """


def readiness_bar_html(score: float) -> str:
    if score >= 60:
        color = "#2ecc71"
    elif score >= 40:
        color = "#f39c12"
    else:
        color = "#e74c3c"
    return f"""
    <div class="progress-outer">
        <div class="progress-inner" style="width: {score}%; background: {color};"></div>
    </div>
    <div class="progress-label">{score:.1f} / 100</div>
    """


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def plot_readiness_bar(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#fafbfc")
    ax.set_facecolor("#fafbfc")
    ordered = df.sort_values("readiness_score", ascending=True)
    colors = [OUTCOME_COLORS.get(o, "#95a5a6") for o in ordered["recommended_outcome"]]
    bars = ax.barh(ordered["concept_name"], ordered["readiness_score"], color=colors, height=0.65, edgecolor="white", linewidth=0.5)
    for bar, score in zip(bars, ordered["readiness_score"]):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"{score:.1f}", va="center", fontsize=8, fontweight="500", color="#475569")
    ax.set_xlabel("Readiness Score (1-100)", fontsize=10, color="#475569")
    ax.set_title("Concept Readiness Ranking", fontsize=13, fontweight="bold", color="#1e293b", pad=12)
    ax.set_xlim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)
    plt.tight_layout()
    return fig


def plot_outcome_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#fafbfc")
    counts = df["recommended_outcome"].value_counts()
    colors = [OUTCOME_COLORS.get(o, "#95a5a6") for o in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, autopct="%1.0f%%", colors=colors,
        startangle=140, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
        t.set_color("white")
    ax.legend(
        counts.index, loc="center left", bbox_to_anchor=(0.95, 0.5),
        fontsize=8, frameon=False,
    )
    ax.set_title("Outcome Distribution", fontsize=13, fontweight="bold", color="#1e293b", pad=12)
    plt.tight_layout()
    return fig


def plot_cluster_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor("#fafbfc")
    ax.set_facecolor("#fafbfc")
    palette = ["#667eea", "#2ecc71", "#f39c12", "#e74c3c"]
    for cid, group in df.groupby("cluster_id"):
        color = palette[cid % len(palette)]
        ax.scatter(
            group["demand_intensity"], group["feasibility_risk"],
            s=group["readiness_score"] * 5, alpha=0.7,
            label=group["cluster_profile"].iloc[0],
            color=color, edgecolors="white", linewidth=0.8,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["concept_name"].split()[0],
                (row["demand_intensity"], row["feasibility_risk"]),
                fontsize=7, alpha=0.85, color="#334155",
                fontweight="500",
            )
    ax.set_xlabel("Demand Intensity", fontsize=10, color="#475569")
    ax.set_ylabel("Feasibility Risk", fontsize=10, color="#475569")
    ax.set_title("K-Means Clusters: Demand vs Effort", fontsize=13, fontweight="bold", color="#1e293b", pad=12)
    ax.legend(loc="best", fontsize=8, frameon=True, fancybox=True, shadow=False, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.grid(True, alpha=0.3, color="#cbd5e1")
    plt.tight_layout()
    return fig


def plot_feature_importance(importance: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fafbfc")
    ax.set_facecolor("#fafbfc")
    top = importance.head(8).copy()
    top["feature_label"] = top["feature"].map(FEATURE_LABELS).fillna(top["feature"])
    palette = sns.color_palette("viridis_r", n_colors=len(top))
    bars = ax.barh(top["feature_label"], top["importance"], color=palette, height=0.6, edgecolor="white", linewidth=0.5)
    for bar, imp in zip(bars, top["importance"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{imp:.3f}", va="center", fontsize=8, fontweight="500", color="#475569")
    ax.set_xlabel("Importance", fontsize=10, color="#475569")
    ax.set_title("Feature Importance (Random Forest)", fontsize=13, fontweight="bold", color="#1e293b", pad=12)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)
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
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
    ax.barh(labels, values, color=colors, height=0.55, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP contribution toward predicted outcome", fontsize=10, color="#475569")
    ax.set_title(f"Why: {row['concept_name']} -> {row['recommended_outcome']}", fontsize=13, fontweight="bold", color="#1e293b", pad=12)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.tick_params(colors="#64748b", labelsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(report, artifacts):
    with st.sidebar:
        st.markdown("## 🧠 CIE")
        st.markdown("**Commercialization Intelligence Engine**")
        st.markdown("---")

        st.markdown("### Pipeline Status")
        data_exists = FEATURES_PATH.exists()
        if data_exists:
            st.success("Data & models ready")
        else:
            st.warning("Data missing")

        st.markdown("---")
        st.markdown("### Quick Stats")
        st.markdown(f"**{report['concepts_scored']}** concepts scored")
        top_name = artifacts["ranked"].iloc[0]["concept_name"]
        st.markdown(f"**Top:** {top_name}")
        n_archive = len(report.get("archive_candidates", []))
        st.markdown(f"**Archive candidates:** {n_archive}")

        st.markdown("---")
        st.markdown("### Actions")
        if st.button("🔄 Regenerate Data", use_container_width=True):
            with st.spinner("Running pipeline..."):
                subprocess.run([sys.executable, str(ROOT / "data" / "generate_mock_data.py")], cwd=str(ROOT))
                subprocess.run([sys.executable, str(ROOT / "data" / "prepare_features.py")], cwd=str(ROOT))
            st.cache_data.clear()
            st.success("Done! Refresh the page.")
            st.rerun()

        st.markdown("---")
        st.markdown(
            "**Author:** PrashantPatil-2005\n"
            "**Version:** 2.0\n"
            "**Tech:** Python, scikit-learn, SHAP, Streamlit"
        )


# ---------------------------------------------------------------------------
# Tab: Overview
# ---------------------------------------------------------------------------

def tab_overview(report, artifacts, insights):
    ranked = insights.sort_values("portfolio_rank")

    # --- Executive summary text ---
    st.markdown(generate_executive_summary(artifacts["ranked"]))

    st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

    # --- KPI cards row ---
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi_card("Concepts", str(report["concepts_scored"]), "Total analyzed", "blue"), unsafe_allow_html=True)
    with c2:
        top_score = ranked.iloc[0]["readiness_score"]
        st.markdown(kpi_card("Top Readiness", f"{top_score:.1f}", ranked.iloc[0]["concept_name"], "green"), unsafe_allow_html=True)
    with c3:
        outcomes = report["classifier"]["outcome_distribution"]
        pilots = outcomes.get("Customer Pilot", 0) + outcomes.get("MVP Build", 0)
        st.markdown(kpi_card("Ready to Move", str(pilots), "Pilot + MVP Build", "purple"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Archive", str(n_archive := len(report.get("archive_candidates", []))), "Deprioritize", "red"), unsafe_allow_html=True)
    with c5:
        cv = report.get("cross_validation", {})
        acc = cv.get("accuracy_mean", 0)
        st.markdown(kpi_card("CV Accuracy", f"{acc:.0%}", cv.get("method", ""), "orange"), unsafe_allow_html=True)

    st.markdown("")

    # --- Portfolio table with outcome badges ---
    st.markdown('<div class="section-header">Ranked Portfolio</div>', unsafe_allow_html=True)
    display_df = ranked[[
        "portfolio_rank", "concept_name", "industry", "readiness_score",
        "confidence_score", "recommended_outcome", "cluster_profile",
    ]].copy()
    display_df["outcome_badge"] = display_df["recommended_outcome"].apply(outcome_badge_html)

    # Render as styled dataframe
    st.dataframe(
        display_df.rename(columns={
            "portfolio_rank": "#",
            "concept_name": "Concept",
            "industry": "Industry",
            "readiness_score": "Readiness",
            "confidence_score": "Confidence",
            "outcome_badge": "Outcome",
            "cluster_profile": "Cluster",
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

    st.markdown('<div class="section-header">Signal Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_readiness_bar(ranked))
    with c2:
        st.pyplot(plot_outcome_distribution(ranked))

    st.markdown("")
    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(plot_cluster_scatter(artifacts["df"]))
    with c4:
        st.pyplot(plot_feature_importance(artifacts["importance"]))


# ---------------------------------------------------------------------------
# Tab: Concept Explorer
# ---------------------------------------------------------------------------

def tab_explorer(artifacts, insights):
    ranked = insights.sort_values("portfolio_rank")
    feature_names = artifacts["feature_columns"]

    st.markdown('<div class="section-header">Concept Drill-Down</div>', unsafe_allow_html=True)

    concept_names = ranked["concept_name"].tolist()
    selected = st.selectbox("Select a concept to explore", concept_names, label_visibility="collapsed")
    row = ranked[ranked["concept_name"] == selected].iloc[0]

    # --- Concept header with badge ---
    badge = outcome_badge_html(row["recommended_outcome"])
    st.markdown(f"### {row['concept_name']} {badge}", unsafe_allow_html=True)
    st.markdown(f"*{row['industry']} | {row['cluster_profile']}*")

    st.markdown("")

    # --- Metrics row ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(kpi_card("Readiness Score", f"{row['readiness_score']:.1f}", f"Rank #{int(row['portfolio_rank'])}", "blue"), unsafe_allow_html=True)
    with m2:
        st.markdown(kpi_card("Confidence", f"{row['confidence_score']:.0%}", "Model certainty", "green"), unsafe_allow_html=True)
    with m3:
        icon = OUTCOME_ICONS.get(row["recommended_outcome"], "")
        st.markdown(kpi_card("Recommended Action", f"{icon} {row['recommended_outcome']}", "ML decision", "purple"), unsafe_allow_html=True)

    st.markdown("")

    # --- Readiness progress bar ---
    st.markdown("#### Readiness")
    st.markdown(readiness_bar_html(row["readiness_score"]), unsafe_allow_html=True)

    st.markdown("")

    # --- Narrative and evidence ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### AI Narrative")
        st.markdown(f'<div class="narrative-box">{row["ai_narrative"]}</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown("#### Evidence Summary")
        st.markdown(f'<div class="evidence-box">{row.get("key_evidence", "---")}</div>', unsafe_allow_html=True)

    st.markdown("")

    # --- SHAP waterfall ---
    st.markdown("#### SHAP Explainability")
    fig = plot_shap_waterfall(row, feature_names)
    if fig:
        st.pyplot(fig)

    # --- Raw features expander ---
    with st.expander("Raw Feature Values", expanded=False):
        feat_data = {f: round(float(row[f]), 4) for f in feature_names if f in row.index}
        st.json(feat_data)


# ---------------------------------------------------------------------------
# Tab: Model Report
# ---------------------------------------------------------------------------

def tab_model_report():
    import json

    st.markdown('<div class="section-header">Model Performance Report</div>', unsafe_allow_html=True)

    report_path = PROCESSED_DIR / "model_report.json"
    val_path = PROCESSED_DIR / "validation_report.json"

    if not report_path.exists():
        st.warning("model_report.json not found. Run the pipeline first.")
        return

    with open(report_path) as f:
        model_report = json.load(f)

    # --- Cross-validation ---
    cv = model_report.get("cross_validation", {})
    if cv:
        st.markdown("#### Cross-Validation")
        c1, c2, c3 = st.columns(3)
        c1.metric("Method", cv.get("method", "N/A"))
        c2.metric("Mean Accuracy", f"{cv.get('accuracy_mean', 0):.2%}")
        c3.metric("Std Dev", f"±{cv.get('accuracy_std', 0):.2%}")
        st.markdown(f"Fold scores: {cv.get('fold_scores', [])}")

    st.markdown("")

    # --- Baseline weights ---
    st.markdown("#### Baseline Weights")
    weights = model_report.get("baseline_model", {}).get("weights", {})
    weights_df = pd.DataFrame([
        {"Feature": k, "Weight": v} for k, v in weights.items()
    ]).sort_values("Weight", ascending=False)
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    st.markdown("")

    # --- Feature importance ---
    st.markdown("#### Top Features (Random Forest)")
    top_feats = model_report.get("top_features", [])
    feats_df = pd.DataFrame(top_feats)
    if not feats_df.empty:
        feats_df["label"] = feats_df["feature"].map(FEATURE_LABELS).fillna(feats_df["feature"])
        st.dataframe(feats_df[["label", "importance"]].rename(columns={"label": "Feature", "importance": "Importance"}),
                     use_container_width=True, hide_index=True)

    st.markdown("")

    # --- Cluster summary ---
    cluster_path = PROCESSED_DIR / "cluster_summary.csv"
    if cluster_path.exists():
        st.markdown("#### Cluster Profiles")
        clusters = pd.read_csv(cluster_path)
        st.dataframe(clusters, use_container_width=True, hide_index=True)

    st.markdown("")

    # --- Validation report ---
    if val_path.exists():
        with open(val_path) as f:
            val = json.load(f)
        st.markdown("#### Data Validation")
        st.markdown(f"**Passed:** {'Yes' if val.get('validation_passed') else 'No'}")
        st.markdown(f"**Concepts:** {val.get('concept_count')} | **Interactions:** {val.get('interaction_count')}")
        feat_summary = val.get("feature_summary", {})
        if feat_summary:
            feat_stats = pd.DataFrame(feat_summary).T
            st.dataframe(feat_stats, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("🧠 Commercialization Intelligence Engine")
    st.markdown("*Customer signals → ML analysis → AI explanation → Commercial decision*")

    try:
        report, artifacts, insights = load_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(report, artifacts)

    tab_overview, tab_analytics, tab_explorer, tab_report = st.tabs([
        "📊 Overview", "📈 Analytics", "🔍 Concept Explorer", "📋 Model Report"
    ])

    with tab_overview:
        tab_overview(report, artifacts, insights)

    with tab_analytics:
        tab_analytics(artifacts, insights)

    with tab_explorer:
        tab_explorer(artifacts, insights)

    with tab_report:
        tab_model_report()


if __name__ == "__main__":
    main()
