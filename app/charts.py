"""
Chart Builders — Theme-Aware

Every chart reads the active theme via get_theme().
Colors adapt automatically when the user switches Light/Dark.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.theme import (
    get_theme, OUTCOME_META, CLUSTER_COLORS,
    FONT_FAMILY, FONT_XS_F, FONT_SM_F, FONT_BASE_F,
    WEIGHT_SEMIBOLD, WEIGHT_MEDIUM,
)


# ============================================================
# Helpers
# ============================================================

def _apply_rc(t: dict):
    """Apply matplotlib rcParams from the active theme."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SM_F,
        "axes.titlesize": FONT_BASE_F,
        "axes.titleweight": WEIGHT_SEMIBOLD,
        "axes.labelsize": FONT_SM_F,
        "axes.labelcolor": t["chart_text"],
        "xtick.color": t["chart_text"],
        "ytick.color": t["chart_text"],
        "axes.edgecolor": t["chart_edge"],
        "axes.facecolor": t["chart_bg"],
        "figure.facecolor": t["chart_bg"],
        "grid.color": t["chart_grid"],
        "grid.linewidth": 0.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": t["chart_bg"],
        "text.color": t["text_primary"],
    })


def _clean_ax(ax, t: dict):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(t["chart_edge"])
    ax.spines["bottom"].set_color(t["chart_edge"])
    ax.tick_params(colors=t["chart_text"], labelsize=FONT_SM_F)
    ax.set_facecolor(t["chart_bg"])


def _title_style(ax, title: str, t: dict):
    ax.set_title(title, fontsize=FONT_BASE_F, fontweight=WEIGHT_SEMIBOLD, pad=10, color=t["chart_title"])


COLORS = {k: v["color"] for k, v in OUTCOME_META.items()}


# ============================================================
# Charts
# ============================================================

def plot_readiness_bar(df: pd.DataFrame, figsize=(10, 4)):
    t = get_theme()
    _apply_rc(t)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["chart_bg"])
    ax.set_facecolor(t["chart_bg"])

    ordered = df.sort_values("readiness_score", ascending=True)
    colors = [COLORS.get(o, t["text_muted"]) for o in ordered["recommended_outcome"]]
    bars = ax.barh(
        ordered["concept_name"], ordered["readiness_score"],
        color=colors, height=0.55, edgecolor=t["chart_bar_edge"], linewidth=0.5,
    )
    for bar, score in zip(bars, ordered["readiness_score"]):
        ax.text(
            bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}", va="center", fontsize=FONT_XS_F - 1, fontweight=WEIGHT_MEDIUM, color=t["chart_text"],
        )
    ax.set_xlabel("Readiness Score", fontsize=FONT_XS_F, color=t["chart_text"])
    ax.set_xlim(0, max(ordered["readiness_score"]) * 1.12)
    _title_style(ax, "Commercialization Readiness by Concept", t)
    _clean_ax(ax, t)
    ax.grid(axis="x", alpha=0.2, color=t["chart_grid"])
    fig.tight_layout(pad=0.5)
    plt.close(fig)
    return fig


def plot_outcome_distribution(df: pd.DataFrame, figsize=(5, 3)):
    t = get_theme()
    _apply_rc(t)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["chart_bg"])
    ax.set_facecolor(t["chart_bg"])

    counts = df["recommended_outcome"].value_counts()
    colors = [COLORS.get(o, t["text_muted"]) for o in counts.index]
    bars = ax.barh(["Portfolio"], [counts.values[0]], color=colors[0], height=0.45, edgecolor=t["chart_bar_edge"])
    left = counts.values[0]
    for o, c, color in zip(counts.index[1:], counts.values[1:], colors[1:]):
        ax.barh(["Portfolio"], [c], left=left, color=color, height=0.45, edgecolor=t["chart_bar_edge"])
        left += c

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS.get(o, t["text_muted"]), label=f"{o} ({c})")
        for o, c in zip(counts.index, counts.values)
    ]
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(counts), 3), fontsize=FONT_XS_F - 1, frameon=False,
        labelcolor=t["chart_text"],
    )
    _title_style(ax, "Recommendation Distribution", t)
    ax.set_xlim(0, counts.sum() * 1.05)
    _clean_ax(ax, t)
    ax.set_yticks([])
    fig.tight_layout(pad=0.5)
    plt.close(fig)
    return fig


def plot_cluster_scatter(df: pd.DataFrame, figsize=(8, 4.5)):
    t = get_theme()
    _apply_rc(t)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["chart_bg"])
    ax.set_facecolor(t["chart_bg"])

    for cid, group in df.groupby("cluster_id"):
        color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
        ax.scatter(
            group["demand_intensity"], group["feasibility_risk"],
            s=group["readiness_score"] * 3.5, alpha=0.7,
            label=group["cluster_profile"].iloc[0],
            color=color, edgecolors=t["chart_bar_edge"], linewidth=0.8,
        )
        for _, row in group.iterrows():
            short = row["concept_name"].split()[0][:10]
            ax.annotate(
                short, (row["demand_intensity"], row["feasibility_risk"]),
                fontsize=FONT_SM_F, alpha=0.8, color=t["chart_text"],
            )
    ax.set_xlabel("Demand Intensity", fontsize=FONT_XS_F, color=t["chart_text"])
    ax.set_ylabel("Feasibility Risk", fontsize=FONT_XS_F, color=t["chart_text"])
    _title_style(ax, "Cluster Analysis: Demand vs Delivery Effort", t)
    ax.legend(fontsize=FONT_SM_F, frameon=True, edgecolor=t["chart_edge"], loc="best",
              facecolor=t["chart_bg"], labelcolor=t["chart_text"])
    _clean_ax(ax, t)
    ax.grid(True, alpha=0.15, linestyle="--", color=t["chart_grid"])
    fig.tight_layout(pad=0.5)
    plt.close(fig)
    return fig


def plot_feature_importance(importance: pd.DataFrame, feature_labels: dict, figsize=(8, 3.5)):
    t = get_theme()
    _apply_rc(t)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["chart_bg"])
    ax.set_facecolor(t["chart_bg"])

    top = importance.head(8).copy()
    top["label"] = top["feature"].map(feature_labels).fillna(top["feature"])
    top = top.sort_values("importance", ascending=True)

    colors = [t["info"]] * len(top)
    bars = ax.barh(top["label"], top["importance"], color=colors, height=0.5, edgecolor=t["chart_bar_edge"])
    for bar, imp in zip(bars, top["importance"]):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{imp:.1%}", va="center", fontsize=FONT_XS_F - 1, color=t["chart_text"],
        )
    ax.set_xlabel("Relative Importance", fontsize=FONT_XS_F, color=t["chart_text"])
    _title_style(ax, "Feature Importance (Random Forest)", t)
    _clean_ax(ax, t)
    ax.grid(axis="x", alpha=0.2, color=t["chart_grid"])
    fig.tight_layout(pad=0.5)
    plt.close(fig)
    return fig


def plot_shap_waterfall(row: pd.Series, feature_names: list, feature_labels: dict, figsize=(9, 3.5)):
    t = get_theme()
    _apply_rc(t)

    shap_cols = [f"shap_{f}" for f in feature_names if f"shap_{f}" in row.index]
    if not shap_cols:
        return None

    labels = [feature_labels.get(c.replace("shap_", ""), c) for c in shap_cols]
    values = [row[c] for c in shap_cols]
    pairs = sorted(zip(labels, values), key=lambda x: abs(x[1]), reverse=True)[:8]
    labels, values = zip(*pairs)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["chart_bg"])
    ax.set_facecolor(t["chart_bg"])

    colors = [t["chart_positive"] if v > 0 else t["chart_negative"] for v in values]
    bars = ax.barh(labels, values, color=colors, height=0.45, edgecolor=t["chart_bar_edge"])
    ax.axvline(0, color=t["text_muted"], linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, values):
        offset = 0.0008 if val >= 0 else -0.0008
        ax.text(
            bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center", fontsize=FONT_XS_F - 2,
            color=t["chart_positive"] if val > 0 else t["chart_negative"],
            ha="left" if val >= 0 else "right",
        )
    ax.set_xlabel("SHAP contribution", fontsize=FONT_XS_F, color=t["chart_text"])
    _title_style(ax, "Feature Contributions to Recommendation", t)
    ax.invert_yaxis()
    _clean_ax(ax, t)
    fig.tight_layout(pad=0.5)
    plt.close(fig)
    return fig


def plot_cv_folds(folds: list, figsize=(5, 2.2)):
    t = get_theme()
    _apply_rc(t)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["chart_bg"])
    ax.set_facecolor(t["chart_bg"])

    colors = [t["info"] if s >= 0.7 else t["warning"] if s >= 0.5 else t["danger"] for s in folds]
    bars = ax.bar(range(1, len(folds) + 1), folds, color=colors, edgecolor=t["chart_bar_edge"], width=0.45)
    for bar, score in zip(bars, folds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{score:.0%}", ha="center", fontsize=FONT_XS_F - 1, fontweight=WEIGHT_MEDIUM, color=t["chart_text"],
        )
    ax.set_xlabel("Fold", fontsize=FONT_XS_F, color=t["chart_text"])
    ax.set_ylabel("Accuracy", fontsize=FONT_XS_F, color=t["chart_text"])
    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(1, len(folds) + 1))
    _title_style(ax, "Cross-Validation Results", t)
    _clean_ax(ax, t)
    ax.grid(axis="y", alpha=0.2, color=t["chart_grid"])
    fig.tight_layout(pad=0.5)
    plt.close(fig)
    return fig


def plot_correlation_matrix(df: pd.DataFrame, features: list, figsize=(7, 5.5)):
    t = get_theme()
    _apply_rc(t)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["chart_bg"])
    ax.set_facecolor(t["chart_bg"])

    corr = df[features].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    short_labels = [f.replace("_norm", "").replace("_text", "")[:12] for f in features]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=FONT_XS_F - 1, color=t["chart_text"])
    ax.set_yticklabels(short_labels, fontsize=FONT_XS_F - 1, color=t["chart_text"])
    _title_style(ax, "Feature Correlation Matrix", t)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=t["chart_text"], labelsize=FONT_XS_F - 1)
    fig.tight_layout(pad=0.5)
    plt.close(fig)
    return fig
