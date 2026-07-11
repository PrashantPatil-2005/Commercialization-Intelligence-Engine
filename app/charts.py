"""Professional chart builders using matplotlib."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.styles import OUTCOME_META, CLUSTER_COLORS


# Consistent style - tight layout, minimal whitespace
plt.rcParams.update({
    "font.family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.labelsize": 9,
    "axes.labelcolor": "#656d76",
    "xtick.color": "#656d76",
    "ytick.color": "#656d76",
    "axes.edgecolor": "#d1d9e0",
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "grid.color": "#e8ecf0",
    "grid.linewidth": 0.5,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "MVP Build": "#0e9f6e",
    "Customer Pilot": "#1f6feb",
    "Reusable Asset": "#8957e5",
    "Incubate": "#bf8700",
    "Archive": "#cf222e",
}


def _clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d9e0")
    ax.spines["bottom"].set_color("#d1d9e0")
    ax.tick_params(colors="#656d76", labelsize=8)


def plot_readiness_bar(df: pd.DataFrame, figsize=(10, 4)):
    """Horizontal bar chart of readiness scores, colored by outcome."""
    fig, ax = plt.subplots(figsize=figsize)
    ordered = df.sort_values("readiness_score", ascending=True)
    colors = [COLORS.get(o, "#8b949e") for o in ordered["recommended_outcome"]]
    bars = ax.barh(
        ordered["concept_name"], ordered["readiness_score"],
        color=colors, height=0.55, edgecolor="white", linewidth=0.5,
    )
    for bar, score in zip(bars, ordered["readiness_score"]):
        ax.text(
            bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}", va="center", fontsize=7.5, fontweight="500", color="#656d76",
        )
    ax.set_xlabel("Readiness Score", fontsize=9)
    ax.set_xlim(0, max(ordered["readiness_score"]) * 1.12)
    ax.set_title("Commercialization Readiness by Concept", fontsize=11, fontweight="600", pad=8)
    _clean_ax(ax)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(pad=0.5)
    return fig


def plot_outcome_distribution(df: pd.DataFrame, figsize=(5, 3)):
    """Horizontal stacked bar showing outcome counts."""
    fig, ax = plt.subplots(figsize=figsize)
    counts = df["recommended_outcome"].value_counts()
    colors = [COLORS.get(o, "#8b949e") for o in counts.index]
    bars = ax.barh(["Portfolio"], [counts.values[0]], color=colors[0], height=0.45)
    left = counts.values[0]
    for o, c, color in zip(counts.index[1:], counts.values[1:], colors[1:]):
        ax.barh(["Portfolio"], [c], left=left, color=color, height=0.45, label=f"{o} ({c})")
        left += c
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS.get(o, "#8b949e"), label=f"{o} ({c})")
        for o, c in zip(counts.index, counts.values)
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=min(len(counts), 3), fontsize=7.5, frameon=False)
    ax.set_title("Recommendation Distribution", fontsize=11, fontweight="600", pad=8)
    ax.set_xlim(0, counts.sum() * 1.05)
    _clean_ax(ax)
    ax.set_yticks([])
    fig.tight_layout(pad=0.5)
    return fig


def plot_cluster_scatter(df: pd.DataFrame, figsize=(8, 4.5)):
    """Scatter plot of demand vs feasibility, colored by cluster."""
    fig, ax = plt.subplots(figsize=figsize)
    for cid, group in df.groupby("cluster_id"):
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax.scatter(
            group["demand_intensity"], group["feasibility_risk"],
            s=group["readiness_score"] * 3.5, alpha=0.7,
            label=group["cluster_profile"].iloc[0],
            color=color, edgecolors="white", linewidth=0.8,
        )
        for _, row in group.iterrows():
            short = row["concept_name"].split()[0][:10]
            ax.annotate(
                short, (row["demand_intensity"], row["feasibility_risk"]),
                fontsize=6.5, alpha=0.8, color="#484f58",
            )
    ax.set_xlabel("Demand Intensity", fontsize=9)
    ax.set_ylabel("Feasibility Risk", fontsize=9)
    ax.set_title("Cluster Analysis: Demand vs Delivery Effort", fontsize=11, fontweight="600", pad=8)
    ax.legend(fontsize=7.5, frameon=True, edgecolor="#d1d9e0", loc="best")
    _clean_ax(ax)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout(pad=0.5)
    return fig


def plot_feature_importance(importance: pd.DataFrame, feature_labels: dict, figsize=(8, 3.5)):
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=figsize)
    top = importance.head(8).copy()
    top["label"] = top["feature"].map(feature_labels).fillna(top["feature"])
    top = top.sort_values("importance", ascending=True)
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.8, len(top)))
    bars = ax.barh(top["label"], top["importance"], color=colors, height=0.5, edgecolor="white")
    for bar, imp in zip(bars, top["importance"]):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{imp:.1%}", va="center", fontsize=7.5, color="#656d76",
        )
    ax.set_xlabel("Relative Importance", fontsize=9)
    ax.set_title("Feature Importance (Random Forest)", fontsize=11, fontweight="600", pad=8)
    _clean_ax(ax)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(pad=0.5)
    return fig


def plot_shap_waterfall(row: pd.Series, feature_names: list, feature_labels: dict, figsize=(9, 3.5)):
    """SHAP waterfall chart for a single concept."""
    shap_cols = [f"shap_{f}" for f in feature_names if f"shap_{f}" in row.index]
    if not shap_cols:
        return None
    labels = [feature_labels.get(c.replace("shap_", ""), c) for c in shap_cols]
    values = [row[c] for c in shap_cols]
    pairs = sorted(zip(labels, values), key=lambda x: abs(x[1]), reverse=True)[:8]
    labels, values = zip(*pairs)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#0e9f6e" if v > 0 else "#cf222e" for v in values]
    bars = ax.barh(labels, values, color=colors, height=0.45, edgecolor="white")
    ax.axvline(0, color="#8b949e", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, values):
        offset = 0.0008 if val >= 0 else -0.0008
        ax.text(
            bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center", fontsize=7,
            color="#0e9f6e" if val > 0 else "#cf222e",
            ha="left" if val >= 0 else "right",
        )
    ax.set_xlabel("SHAP contribution", fontsize=9)
    ax.set_title("Feature Contributions to Recommendation", fontsize=11, fontweight="600", pad=8)
    ax.invert_yaxis()
    _clean_ax(ax)
    fig.tight_layout(pad=0.5)
    return fig


def plot_cv_folds(folds: list, figsize=(5, 2.2)):
    """Bar chart of cross-validation fold scores."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#1f6feb" if s >= 0.7 else "#bf8700" if s >= 0.5 else "#cf222e" for s in folds]
    bars = ax.bar(range(1, len(folds) + 1), folds, color=colors, edgecolor="white", width=0.45)
    for bar, score in zip(bars, folds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{score:.0%}", ha="center", fontsize=7.5, fontweight="500", color="#656d76",
        )
    ax.set_xlabel("Fold", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(1, len(folds) + 1))
    ax.set_title("Cross-Validation Results", fontsize=11, fontweight="600", pad=8)
    _clean_ax(ax)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(pad=0.5)
    return fig


def plot_correlation_matrix(df: pd.DataFrame, features: list, figsize=(7, 5.5)):
    """Correlation heatmap of numeric features."""
    fig, ax = plt.subplots(figsize=figsize)
    corr = df[features].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    short_labels = [f.replace("_norm", "").replace("_text", "")[:12] for f in features]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(short_labels, fontsize=7.5)
    ax.set_title("Feature Correlation Matrix", fontsize=11, fontweight="600", pad=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.5)
    return fig
