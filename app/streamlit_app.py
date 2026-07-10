"""
Phase 5: Streamlit Presentation Layer

Commercialization Intelligence Engine dashboard.
Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

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

OUTCOME_COLORS = {
    "MVP Build": "#2ecc71",
    "Customer Pilot": "#3498db",
    "Reusable Asset": "#9b59b6",
    "Incubate": "#f39c12",
    "Archive": "#e74c3c",
}

st.set_page_config(
    page_title="Commercialization Intelligence Engine",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner="Running ML pipeline...")
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


def plot_readiness_bar(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = df.sort_values("readiness_score", ascending=True)
    colors = [OUTCOME_COLORS.get(o, "#95a5a6") for o in ordered["recommended_outcome"]]
    ax.barh(ordered["concept_name"], ordered["readiness_score"], color=colors)
    ax.set_xlabel("Readiness Score (1–100)")
    ax.set_title("Concept Readiness Ranking")
    ax.set_xlim(0, 100)
    plt.tight_layout()
    return fig


def plot_outcome_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["recommended_outcome"].value_counts()
    colors = [OUTCOME_COLORS.get(o, "#95a5a6") for o in counts.index]
    ax.pie(counts, labels=counts.index, autopct="%1.0f%%", colors=colors, startangle=140)
    ax.set_title("Recommended Outcome Distribution")
    plt.tight_layout()
    return fig


def plot_cluster_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("Set2", df["cluster_id"].nunique())
    for cid, group in df.groupby("cluster_id"):
        ax.scatter(
            group["demand_intensity"],
            group["feasibility_risk"],
            s=group["readiness_score"] * 4,
            alpha=0.75,
            label=group["cluster_profile"].iloc[0],
            color=palette[cid % len(palette)],
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["concept_name"].split()[0],
                (row["demand_intensity"], row["feasibility_risk"]),
                fontsize=7,
                alpha=0.8,
            )
    ax.set_xlabel("Demand Intensity")
    ax.set_ylabel("Feasibility Risk")
    ax.set_title("K-Means Clusters: Demand vs Effort")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return fig


def plot_feature_importance(importance: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    top = importance.head(8)
    sns.barplot(data=top, x="importance", y="feature", palette="Blues_r", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
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

    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution toward predicted outcome")
    ax.set_title(f"Why: {row['concept_name']} → {row['recommended_outcome']}")
    plt.tight_layout()
    return fig


def main():
    st.title("Commercialization Intelligence Engine")
    st.caption("Customer signals → ML analysis → AI explanation → Commercial decision")

    try:
        report, artifacts, insights = load_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    ranked = insights.sort_values("portfolio_rank")

    # --- Executive summary ---
    st.header("Executive Summary")
    st.text(generate_executive_summary(artifacts["ranked"]))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Concepts Analyzed", report["concepts_scored"])
    col2.metric("Top Readiness", f"{ranked.iloc[0]['readiness_score']}/100")
    col3.metric("Top Concept", ranked.iloc[0]["concept_name"])
    col4.metric("Archive Candidates", len(report.get("archive_candidates", [])))

    st.divider()

    # --- Ranked portfolio ---
    st.header("Ranked Portfolio")
    display_cols = [
        "portfolio_rank",
        "concept_name",
        "industry",
        "readiness_score",
        "confidence_score",
        "recommended_outcome",
        "cluster_profile",
        "ai_narrative",
    ]
    st.dataframe(
        ranked[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --- Visual analytics ---
    st.header("Signal Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_readiness_bar(ranked))
    with c2:
        st.pyplot(plot_outcome_distribution(ranked))

    c3, c4 = st.columns(2)
    with c3:
        st.pyplot(plot_cluster_scatter(artifacts["df"]))
    with c4:
        st.pyplot(plot_feature_importance(artifacts["importance"]))

    st.divider()

    # --- Concept drill-down ---
    st.header("Concept Explainability (SHAP)")
    concept_names = ranked["concept_name"].tolist()
    selected = st.selectbox("Select a concept", concept_names)
    row = ranked[ranked["concept_name"] == selected].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Readiness", f"{row['readiness_score']}/100")
    m2.metric("Confidence", f"{row['confidence_score']:.0%}")
    m3.metric("Outcome", row["recommended_outcome"])

    st.subheader("AI Narrative")
    st.info(row["ai_narrative"])

    st.subheader("Evidence Summary")
    st.write(row.get("key_evidence", "—"))

    fig = plot_shap_waterfall(row, artifacts["feature_columns"])
    if fig:
        st.pyplot(fig)

    with st.expander("Raw feature values"):
        feat_cols = artifacts["feature_columns"]
        st.json({f: round(float(row[f]), 3) for f in feat_cols if f in row.index})


if __name__ == "__main__":
    main()
