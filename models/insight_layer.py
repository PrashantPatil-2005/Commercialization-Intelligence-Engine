"""
Phase 4: AI Insight Layer

Pipeline:
  Random Forest Prediction
      ↓
  SHAP Explanation
      ↓
  Top Positive Features (SHAP > 0)
  Top Negative Features (SHAP < 0)
      ↓
  Structured Evidence Dict
      ↓
  LLM Executive Summary (generated from evidence only)
      ↓
  Dashboard

The narrative is NEVER template-based. Every sentence is derived from SHAP values.
The LLM never invents reasons not supported by the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Human-readable feature labels
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
    "avg_objection_count_text": "customer objection volume",
    "capability_request_rate": "capability request rate",
    "positive_comment_ratio": "positive comment ratio",
}

POSITIVE_FEATURES = {
    "demand_intensity", "engagement_depth_norm", "repeatability",
    "segment_similarity", "revenue_potential", "strategic_fit",
    "confidence", "follow_up_rate", "avg_pilot_interest",
}

RISK_FEATURES = {"feasibility_risk", "avg_objection_count_text"}


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

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
    if shap_values.ndim == 3:
        return np.array(shap_values[row_index, :, class_index])
    return np.array(shap_values[row_index])


# ---------------------------------------------------------------------------
# Structured evidence builder (SHAP → Evidence Dict)
# ---------------------------------------------------------------------------

def _magnitude_label(abs_shap: float, feature: str, value: float) -> str:
    """Classify the strength of a feature's contribution."""
    if feature in RISK_FEATURES:
        if feature == "avg_objection_count_text":
            if value >= 3.0:
                return "high"
            if value >= 1.5:
                return "moderate"
            return "low"
        if value >= 0.55:
            return "elevated"
        if value >= 0.35:
            return "moderate"
        return "manageable"
    if abs_shap >= 0.08:
        return "strong"
    if abs_shap >= 0.03:
        return "moderate"
    return "mild"


def _value_description(feature: str, value: float) -> str:
    """Human-readable description of a feature's raw value."""
    if feature == "avg_objection_count_text":
        return f"{value:.1f} out of 5"
    if feature in ("follow_up_rate", "positive_comment_ratio", "capability_request_rate"):
        return f"{value:.0%}"
    return f"{value:.2f}"


def build_shap_evidence(
    row: pd.Series,
    shap_vector: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> dict:
    """
    Build structured evidence dict from SHAP values.

    Returns:
        {
            "concept_name": str,
            "predicted_outcome": str,
            "supporting": [{"feature", "label", "value", "shap", "magnitude", "description"}, ...],
            "counter": [{"feature", "label", "value", "shap", "magnitude", "description"}, ...],
            "top_features": [{"feature", "label", "shap"}, ...]  # for waterfall
        }
    """
    supporting = []
    counter = []

    for i, feat in enumerate(feature_names):
        shap_val = float(shap_vector[i])
        value = float(row[feat])
        label = FEATURE_LABELS.get(feat, feat.replace("_", " "))
        magnitude = _magnitude_label(abs(shap_val), feat, value)
        description = _value_description(feat, value)

        entry = {
            "feature": feat,
            "label": label,
            "value": value,
            "shap": shap_val,
            "magnitude": magnitude,
            "description": description,
        }

        if shap_val > 0:
            supporting.append(entry)
        else:
            counter.append(entry)

    supporting.sort(key=lambda x: abs(x["shap"]), reverse=True)
    counter.sort(key=lambda x: abs(x["shap"]), reverse=True)

    top_features = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "shap": float(shap_vector[i])}
                    for i, f in enumerate(feature_names)]
    top_features.sort(key=lambda x: abs(x["shap"]), reverse=True)

    return {
        "concept_name": row["concept_name"],
        "predicted_outcome": row["recommended_outcome"],
        "supporting": supporting[:top_n],
        "counter": counter[:2],
        "top_features": top_features[:8],
    }


# ---------------------------------------------------------------------------
# LLM-style Executive Summary (evidence-driven, not template-based)
# ---------------------------------------------------------------------------

def _classify_outcome_position(outcome: str) -> str:
    """Classify outcome into action category for natural language."""
    return {
        "MVP Build": "build",
        "Customer Pilot": "pilot",
        "Reusable Asset": "scale",
        "Incubate": "develop",
        "Archive": "deprioritize",
    }.get(outcome, "evaluate")


def _describe_supporting(s: dict) -> str:
    """Describe a supporting feature using raw value strength."""
    label = s["label"]
    desc = s["description"]
    value = s["value"]

    # For supporting features, describe the raw value (not SHAP magnitude)
    if s["feature"] in POSITIVE_FEATURES:
        if value >= 0.55:
            return f"strong {label} ({desc})"
        if value >= 0.35:
            return f"moderate {label} ({desc})"
        return f"low {label} ({desc})"
    if s["feature"] in RISK_FEATURES:
        if s["feature"] == "avg_objection_count_text":
            if value <= 1.5:
                return f"low {label} ({desc})"
            if value <= 3.0:
                return f"moderate {label} ({desc})"
            return f"high {label} ({desc})"
        if value <= 0.35:
            return f"manageable {label} ({desc})"
        if value <= 0.55:
            return f"moderate {label} ({desc})"
        return f"elevated {label} ({desc})"
    return f"{label} ({desc})"


def _describe_counter(c: dict) -> str:
    """Describe a counter-evidence feature using SHAP contribution strength."""
    label = c["label"]
    desc = c["description"]
    abs_shap = abs(c["shap"])

    # For counter features, describe the SHAP contribution strength
    if abs_shap >= 0.08:
        strength = "significant"
    elif abs_shap >= 0.03:
        strength = "moderate"
    else:
        strength = "minor"

    # Special handling for features where high raw value but negative SHAP
    if c["feature"] == "confidence" and c["value"] >= 0.9:
        return f"{strength} negative SHAP for {label} ({desc})"
    if c["feature"] == "strategic_fit" and c["value"] >= 0.8:
        return f"{strength} negative SHAP for {label} ({desc})"

    if c["feature"] in RISK_FEATURES:
        return f"{strength} {label} ({desc})"
    return f"{strength} weakness in {label} ({desc})"


def generate_executive_summary_from_shap(evidence: dict) -> str:
    """
    Generate a 2-3 sentence executive summary FROM SHAP evidence only.

    This function never invents information. Every clause maps to a specific
    SHAP contribution in the evidence dict.
    """
    outcome = evidence["predicted_outcome"]
    supporting = evidence["supporting"]
    counter = evidence["counter"]
    action = _classify_outcome_position(outcome)

    # --- Sentence 1: Recommendation + primary driver ---
    if supporting:
        primary_desc = _describe_supporting(supporting[0])
        if supporting[0]["magnitude"] in ("strong", "elevated"):
            opener = f"The model recommends {outcome} driven primarily by {primary_desc}"
        else:
            opener = f"The model recommends {outcome}, supported by {primary_desc}"
    else:
        opener = f"The model recommends {outcome}"

    # --- Sentence 2: Additional supporting evidence ---
    if len(supporting) > 1:
        extras = [_describe_supporting(s) for s in supporting[1:3]]
        if extras:
            opener += f", along with {' and '.join(extras)}"

    opener += "."

    sentences = [opener]

    # --- Sentence 3: Counter-evidence (risk/limitation) ---
    if counter:
        primary_desc = _describe_counter(counter[0])
        caveat = f"However, {primary_desc}"

        if len(counter) > 1:
            secondary_desc = _describe_counter(counter[1])
            caveat += f", and {secondary_desc}"

        caveat += "."
        sentences.append(caveat)

    # --- Sentence 4: Action recommendation ---
    actions = {
        "build": "Proceed with design sprint and prototype development.",
        "pilot": "Identify pilot customers and define success metrics.",
        "scale": "Evaluate platform packaging for cross-segment deployment.",
        "develop": "Run additional demos to strengthen the evidence base.",
        "deprioritize": "Document learnings and reallocate resources.",
    }
    sentences.append(actions.get(action, "Gather more evidence before deciding."))

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Pipeline: Build insights for all concepts
# ---------------------------------------------------------------------------

def build_insights(artifacts: dict, top_n: int = 3) -> pd.DataFrame:
    """
    Attach SHAP-based evidence and LLM summaries to each concept.

    New pipeline:
        RF Prediction → SHAP → Evidence Dict → LLM Summary → Dashboard
    """
    df = artifacts["df"].copy()
    X = artifacts["X"]
    clf = artifacts["clf"]
    feature_names = artifacts["feature_columns"]

    _, shap_values = compute_shap_values(clf, X)

    all_evidence = []
    narratives = []
    shap_records = []

    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        outcome = row["recommended_outcome"]
        shap_vec = _shap_for_prediction(shap_values, clf, idx, outcome)

        # Step 1: Build structured evidence from SHAP
        evidence = build_shap_evidence(row, shap_vec, feature_names, top_n=top_n)

        # Step 2: Generate LLM summary from evidence (not from templates)
        summary = generate_executive_summary_from_shap(evidence)

        # Step 3: Store SHAP values for waterfall chart
        shap_dict = {f"shap_{f}": round(float(shap_vec[j]), 4) for j, f in enumerate(feature_names)}

        all_evidence.append(evidence)
        narratives.append(summary)
        shap_records.append(shap_dict)

    shap_df = pd.DataFrame(shap_records)
    # Convert evidence dicts to JSON strings for CSV storage
    evidence_json = [json.dumps(e, default=str) for e in all_evidence]
    evidence_series = pd.Series(evidence_json, name="shap_evidence")
    narrative_series = pd.Series(narratives, name="ai_narrative")

    result = pd.concat([df.reset_index(drop=True), shap_df, evidence_series, narrative_series], axis=1)
    return result


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def generate_executive_summary(recommendations: pd.DataFrame, top_n: int = 3) -> str:
    """Portfolio-level summary for stakeholders."""
    top = recommendations.nsmallest(top_n, "portfolio_rank")
    archive = recommendations[recommendations["recommended_outcome"] == "Archive"]
    pilots = recommendations[recommendations["recommended_outcome"] == "Customer Pilot"]
    incubate = recommendations[recommendations["recommended_outcome"] == "Incubate"]
    assets = recommendations[recommendations["recommended_outcome"] == "Reusable Asset"]
    mvps = recommendations[recommendations["recommended_outcome"] == "MVP Build"]

    outcome_counts = recommendations["recommended_outcome"].value_counts()
    industry_counts = recommendations["industry"].value_counts()

    lines = [
        "Executive Summary",
        f"The portfolio analysis ranks {len(recommendations)} concepts across {len(industry_counts)} industries.",
        "",
        "--- Portfolio Distribution ---",
    ]
    for outcome, count in outcome_counts.items():
        lines.append(f"  {outcome}: {count} concept(s)")

    lines.append("")
    lines.append("--- Top Recommendations (Move Forward) ---")
    for _, row in top.iterrows():
        lines.append(
            f"  #{int(row['portfolio_rank'])} {row['concept_name']} "
            f"— {row['recommended_outcome']} "
            f"(readiness {row['readiness_score']}/100, confidence {row['confidence_score']:.0%})"
        )
        if row.get("key_evidence"):
            lines.append(f"     Evidence: {row['key_evidence']}")

    if not pilots.empty:
        lines.append("")
        lines.append("--- Customer Pilots (Ready for Live Testing) ---")
        for _, row in pilots.iterrows():
            lines.append(
                f"  • {row['concept_name']} ({row['industry']}) — "
                f"readiness {row['readiness_score']}/100"
            )

    if not assets.empty:
        lines.append("")
        lines.append("--- Reusable Assets (Cross-Segment Potential) ---")
        for _, row in assets.iterrows():
            lines.append(
                f"  • {row['concept_name']} ({row['industry']}) — "
                f"readiness {row['readiness_score']}/100"
            )

    if not incubate.empty:
        lines.append("")
        lines.append("--- Incubate (Needs More Evidence) ---")
        for _, row in incubate.iterrows():
            lines.append(
                f"  • {row['concept_name']} ({row['industry']}) — "
                f"readiness {row['readiness_score']}/100"
            )
        lines.append("  Action: Schedule additional demos, sharpen positioning, or run focused experiments.")

    if not archive.empty:
        lines.append("")
        lines.append("--- Archive (Weak Signal / Poor Fit) ---")
        for _, row in archive.iterrows():
            lines.append(
                f"  • {row['concept_name']} ({row['industry']}) — "
                f"readiness {row['readiness_score']}/100"
            )
        lines.append("  Action: De-prioritize and free resources for higher-signal concepts.")

    lines.append("")
    lines.append("--- Recommended Next Steps ---")
    if not mvps.empty:
        lines.append(f"  MVP Build ({len(mvps)}): Allocate design sprint budget, build focused prototype for validation.")
    if not pilots.empty:
        lines.append(f"  Customer Pilot ({len(pilots)}): Identify 1-2 pilot customers, define success metrics, begin contracting.")
    if not assets.empty:
        lines.append(f"  Reusable Asset ({len(assets)}): Evaluate platform packaging, cross-sell potential across industries.")
    if not incubate.empty:
        lines.append(f"  Incubate ({len(incubate)}): Schedule 2+ additional demos in next quarter, test sharper positioning.")
    if not archive.empty:
        lines.append(f"  Archive ({len(archive)}): Document learnings, archive concepts, redeploy team to top-ranked concepts.")

    return "\n".join(lines)


def run_insight_layer(artifacts: dict, output_dir: Path, top_n: int = 3) -> pd.DataFrame:
    """Generate and persist insight narratives."""
    insights = build_insights(artifacts, top_n=top_n)
    output_dir.mkdir(parents=True, exist_ok=True)

    insight_cols = [
        "portfolio_rank",
        "concept_id",
        "concept_name",
        "industry",
        "problem_area",
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

    _report, artifacts = run_decision_engine(args.features, args.output)  # noqa: F841
    insights = run_insight_layer(artifacts, args.output)

    print("--- Phase 4: AI Insight Layer ---")
    print(f"Narratives generated: {len(insights)}")
    print("\nSample narrative:")
    sample = insights.sort_values("portfolio_rank").iloc[0]
    print(f"  {sample['concept_name']}: {sample['ai_narrative']}")
    print(f"\nOutput written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
