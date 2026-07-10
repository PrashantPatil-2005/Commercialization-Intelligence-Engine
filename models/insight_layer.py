"""
Phase 4: AI Insight Layer

Uses SHAP to explain Random Forest outcome predictions and generates
stakeholder-readable narrative summaries per concept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Human-readable feature labels for narratives
# ---------------------------------------------------------------------------

FEATURE_LABELS: dict[str, str] = {
    "demand_intensity": "demand intensity",
    "engagement_depth_norm": "sandbox engagement depth",
    "feasibility_risk": "implementation risk",
    "repeatability": "usage repeatability",
    "segment_similarity": "cross-segment demand consistency",
    "revenue_potential": "revenue potential",
    "strategic_fit": "strategic fit",
    "confidence": "evidence confidence",
    "follow_up_rate": "customer follow-up rate",
    "avg_pilot_interest": "pilot interest",
}

# Features where higher raw value is generally positive for commercialization
POSITIVE_FEATURES = {
    "demand_intensity",
    "engagement_depth_norm",
    "repeatability",
    "segment_similarity",
    "revenue_potential",
    "strategic_fit",
    "confidence",
    "follow_up_rate",
    "avg_pilot_interest",
}

RISK_FEATURES = {"feasibility_risk"}


def _display_score(feature: str, value: float) -> float:
    """Map feature value to a 0–10 readability scale."""
    return round(float(value) * 10, 1)


def _strength_label(feature: str, value: float) -> str:
    score = _display_score(feature, value) if feature != "feasibility_risk" else _display_score(feature, value)
    if feature in RISK_FEATURES:
        if value >= 0.55:
            return "high"
        if value >= 0.35:
            return "moderate"
        return "low"
    if value >= 0.55:
        return "strong"
    if value >= 0.35:
        return "moderate"
    return "weak"


def compute_shap_values(clf, X: pd.DataFrame) -> tuple[shap.TreeExplainer, np.ndarray | list]:
    """Compute SHAP values for the trained Random Forest classifier."""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def _shap_for_prediction(
    shap_values: np.ndarray | list,
    clf,
    row_index: int,
    predicted_class: str,
) -> np.ndarray:
    """Extract SHAP vector for one row and its predicted outcome class."""
    class_index = list(clf.classes_).index(predicted_class)
    if isinstance(shap_values, list):
        return np.array(shap_values[class_index][row_index])
    # shap >= 0.44 may return 3D array (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        return np.array(shap_values[row_index, :, class_index])
    return np.array(shap_values[row_index])


def _format_contribution(feature: str, row: pd.Series, shap_val: float) -> str:
    label = FEATURE_LABELS.get(feature, feature.replace("_", " "))
    value = float(row[feature])
    strength = _strength_label(feature, value)
    score = _display_score(feature, value)

    if feature in RISK_FEATURES:
        if shap_val > 0:
            return f"elevated {label} (score: {score})"
        return f"manageable {label} (score: {score})"

    if shap_val > 0:
        return f"{strength} {label} (score: {score})"
    return f"limited {label} (score: {score})"


def generate_narrative(
    row: pd.Series,
    shap_vector: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> str:
    """
    Build a clean narrative from top SHAP contributors.

    Example:
    "Recommended Outcome: Customer Pilot. Evidence: Strong demand intensity
    (score: 8.5) and high pilot interest, despite moderate implementation risk."
    """
    outcome = row["recommended_outcome"]
    concept = row["concept_name"]

    contributions = [
        (feature_names[i], float(shap_vector[i]), float(row[feature_names[i]]))
        for i in range(len(feature_names))
    ]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    supporting = [c for c in contributions if c[1] > 0][:top_n]
    counter = [c for c in contributions if c[1] < 0][:1]

    evidence_parts = [
        _format_contribution(feat, row, shap_val)
        for feat, shap_val, _ in supporting
    ]

    narrative = (
        f"Recommended Outcome: {outcome}. "
        f"Evidence: {', '.join(evidence_parts)}"
    )

    if counter:
        feat, shap_val, _ = counter[0]
        caveat = _format_contribution(feat, row, shap_val)
        narrative += f", despite {caveat}"

    narrative += "."
    return narrative


def generate_executive_summary(recommendations: pd.DataFrame, top_n: int = 3) -> str:
    """Portfolio-level summary for stakeholders."""
    top = recommendations.nsmallest(top_n, "portfolio_rank")
    archive = recommendations[recommendations["recommended_outcome"] == "Archive"]

    lines = [
        "Executive Summary",
        f"The portfolio analysis ranks {len(recommendations)} concepts by commercialization readiness.",
        "",
        "Move forward:",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"  • {row['concept_name']} — {row['recommended_outcome']} "
            f"(readiness {row['readiness_score']}/100, confidence {row['confidence_score']:.0%})"
        )

    if not archive.empty:
        lines.append("")
        lines.append("Deprioritize / archive:")
        for _, row in archive.iterrows():
            lines.append(f"  • {row['concept_name']} — weak demand and limited engagement")

    return "\n".join(lines)


def build_insights(
    artifacts: dict,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Attach SHAP-based narratives to each concept in the decision output.

    Returns a dataframe with shap columns and ai_narrative per concept.
    """
    df = artifacts["df"].copy()
    X = artifacts["X"]
    clf = artifacts["clf"]
    feature_names = artifacts["feature_columns"]

    _, shap_values = compute_shap_values(clf, X)

    narratives = []
    shap_records = []

    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        outcome = row["recommended_outcome"]
        shap_vec = _shap_for_prediction(shap_values, clf, idx, outcome)
        narrative = generate_narrative(row, shap_vec, feature_names, top_n=top_n)

        shap_dict = {f"shap_{f}": round(float(shap_vec[j]), 4) for j, f in enumerate(feature_names)}
        shap_records.append(shap_dict)
        narratives.append(narrative)

    shap_df = pd.DataFrame(shap_records)
    result = pd.concat([df.reset_index(drop=True), shap_df, pd.Series(narratives, name="ai_narrative")], axis=1)
    return result


def run_insight_layer(artifacts: dict, output_dir: Path, top_n: int = 3) -> pd.DataFrame:
    """Generate and persist insight narratives."""
    insights = build_insights(artifacts, top_n=top_n)
    output_dir.mkdir(parents=True, exist_ok=True)

    insight_cols = [
        "portfolio_rank",
        "concept_id",
        "concept_name",
        "recommended_outcome",
        "readiness_score",
        "confidence_score",
        "ai_narrative",
        "key_evidence",
    ] + [c for c in insights.columns if c.startswith("shap_")]

    available = [c for c in insight_cols if c in insights.columns]
    ranked = insights.sort_values("portfolio_rank")
    ranked[available].to_csv(output_dir / "concept_insights.csv", index=False)

    summary = generate_executive_summary(
        artifacts["ranked"] if "ranked" in artifacts else ranked
    )
    (output_dir / "executive_summary.txt").write_text(summary, encoding="utf-8")

    return insights


def main() -> None:
    import argparse
    import sys

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    from models.decision_engine import run_decision_engine

    parser = argparse.ArgumentParser(description="Generate SHAP narratives for concept decisions")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "concept_features.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed",
    )
    args = parser.parse_args()

    _report, artifacts = run_decision_engine(args.features, args.output)
    insights = run_insight_layer(artifacts, args.output)

    print("--- Phase 4: AI Insight Layer ---")
    print(f"Narratives generated: {len(insights)}")
    print("\nSample narrative:")
    sample = insights.sort_values("portfolio_rank").iloc[0]
    print(f"  {sample['concept_name']}: {sample['ai_narrative']}")
    print(f"\nOutput written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
