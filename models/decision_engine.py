"""
Phase 3: ML Decision Engine

Combines three decision layers:
  1. Baseline weighted scoring (readiness 1–100)
  2. K-Means clustering on behavioral patterns
  3. Random Forest classifier for commercial outcomes

Outcomes: MVP Build, Customer Pilot, Reusable Asset, Incubate, Archive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTCOMES = [
    "MVP Build",
    "Customer Pilot",
    "Reusable Asset",
    "Incubate",
    "Archive",
]

FEATURE_COLUMNS = [
    "demand_intensity",
    "engagement_depth_norm",
    "feasibility_risk",
    "repeatability",
    "segment_similarity",
    "revenue_potential",
    "strategic_fit",
    "confidence",
    "follow_up_rate",
    "avg_pilot_interest",
    "avg_objection_count_text",
    "capability_request_rate",
    "positive_comment_ratio",
]

# Baseline weights (positive signals + inverted risk)
BASELINE_WEIGHTS = {
    "demand_intensity": 0.18,
    "engagement_depth_norm": 0.13,
    "repeatability": 0.13,
    "segment_similarity": 0.09,
    "revenue_potential": 0.18,
    "strategic_fit": 0.09,
    "feasibility_ease": 0.09,  # derived as 1 - feasibility_risk
    "positive_comment_ratio": 0.05,
    "capability_request_rate": 0.03,
    "objection_ease": 0.03,  # derived as 1 - normalized objection count
}

N_CLUSTERS = 4
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------

def compute_baseline_readiness(df: pd.DataFrame) -> pd.Series:
    """
    Weighted scoring function mapping engineered features to readiness 1–100.
    Higher = stronger commercialization signal.
    """
    ease = 1.0 - df["feasibility_risk"]
    obj_ease = 1.0 - (df["avg_objection_count_text"] / 5.0).clip(0, 1)
    raw = (
        df["demand_intensity"] * BASELINE_WEIGHTS["demand_intensity"]
        + df["engagement_depth_norm"] * BASELINE_WEIGHTS["engagement_depth_norm"]
        + df["repeatability"] * BASELINE_WEIGHTS["repeatability"]
        + df["segment_similarity"] * BASELINE_WEIGHTS["segment_similarity"]
        + df["revenue_potential"] * BASELINE_WEIGHTS["revenue_potential"]
        + df["strategic_fit"] * BASELINE_WEIGHTS["strategic_fit"]
        + ease * BASELINE_WEIGHTS["feasibility_ease"]
        + df["positive_comment_ratio"] * BASELINE_WEIGHTS["positive_comment_ratio"]
        + df["capability_request_rate"] * BASELINE_WEIGHTS["capability_request_rate"]
        + obj_ease * BASELINE_WEIGHTS["objection_ease"]
    )
    return (raw * 100).clip(1, 100).round(1)


# ---------------------------------------------------------------------------
# Synthetic labels for Random Forest training
# ---------------------------------------------------------------------------

def assign_synthetic_outcome(row: pd.Series) -> str:
    """
    Rule-based pseudo-labels derived from commercial judgment logic.
    Used to train the Random Forest when real outcome labels are unavailable.
    """
    demand = row["demand_intensity"]
    engagement = row["engagement_depth_norm"]
    repeat = row["repeatability"]
    segments = row["segment_similarity"]
    revenue = row["revenue_potential"]
    risk = row["feasibility_risk"]
    conf = row["confidence"]
    fit = row["strategic_fit"]
    follow_up = row["follow_up_rate"]
    pilot = row["avg_pilot_interest"]

    # Archive — weak signal, poor fit, or high effort vs value
    if demand < 0.22 and revenue < 0.30:
        return "Archive"
    if risk > 0.58 and demand < 0.40:
        return "Archive"
    if conf < 0.62 and demand < 0.25:
        return "Archive"

    # Reusable Asset — repeatable cross-segment demand
    if demand >= 0.45 and repeat >= 0.30 and segments >= 0.90 and revenue >= 0.40:
        return "Reusable Asset"

    # Customer Pilot — clear customer pull, ready to test
    if demand >= 0.42 and follow_up >= 0.35 and pilot >= 0.35 and risk < 0.55:
        return "Customer Pilot"

    # MVP Build — strong focused signal for validation build
    if demand >= 0.48 and engagement >= 0.45 and revenue >= 0.42 and fit >= 0.55:
        return "MVP Build"

    # Incubate — promising but insufficient evidence
    return "Incubate"


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------

def run_kmeans_clustering(
    X_scaled: np.ndarray,
    df: pd.DataFrame,
    cluster_features: list[str],
) -> tuple[np.ndarray, dict[int, str], KMeans]:
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    cluster_names: dict[int, str] = {}
    demand_median = df["demand_intensity"].median()
    risk_median = df["feasibility_risk"].median()

    for cluster_id in range(N_CLUSTERS):
        mask = labels == cluster_id
        cluster_df = df.loc[mask, cluster_features]
        demand = float(cluster_df["demand_intensity"].mean())
        risk = float(cluster_df["feasibility_risk"].mean())
        demand_level = "High Demand" if demand >= demand_median else "Low Demand"
        effort_level = "Low Effort" if risk < risk_median else "High Effort"
        cluster_names[cluster_id] = f"{demand_level} / {effort_level}"

    return labels, cluster_names, kmeans


# ---------------------------------------------------------------------------
# Random Forest classifier
# ---------------------------------------------------------------------------

def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    clf.fit(X, y)
    return clf


def extract_feature_importance(
    clf: RandomForestClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": clf.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Evidence builder
# ---------------------------------------------------------------------------

def build_evidence(row: pd.Series) -> str:
    """Key evidence bullets for stakeholder-facing output."""
    points = []

    if row["demand_intensity"] >= 0.45:
        points.append(f"strong demand intensity ({row['demand_intensity']:.2f})")
    elif row["demand_intensity"] < 0.25:
        points.append(f"weak demand intensity ({row['demand_intensity']:.2f})")

    if row["engagement_depth_norm"] >= 0.50:
        points.append(f"deep sandbox engagement (norm={row['engagement_depth_norm']:.2f})")
    elif row["engagement_depth_norm"] < 0.10:
        points.append("minimal sandbox engagement")

    if row["repeatability"] >= 0.30:
        points.append(f"repeatable usage patterns ({row['repeatability']:.2f})")

    if row["segment_similarity"] >= 0.90:
        points.append("demand consistent across segments")

    if row["revenue_potential"] >= 0.40:
        points.append(f"solid revenue potential ({row['revenue_potential']:.2f})")
    elif row["revenue_potential"] < 0.28:
        points.append(f"low revenue potential ({row['revenue_potential']:.2f})")

    if row["feasibility_risk"] >= 0.55:
        points.append(f"elevated feasibility risk ({row['feasibility_risk']:.2f})")
    elif row["feasibility_risk"] <= 0.30:
        points.append(f"favorable feasibility profile ({row['feasibility_risk']:.2f})")

    if row["confidence"] < 0.70:
        points.append(f"limited evidence confidence ({row['confidence']:.2f})")

    if row["follow_up_rate"] >= 0.40:
        points.append(f"high follow-up rate ({row['follow_up_rate']:.0%})")

    if "positive_comment_ratio" in row.index and row["positive_comment_ratio"] >= 0.50:
        points.append(f"positive customer sentiment ({row['positive_comment_ratio']:.0%})")
    elif "positive_comment_ratio" in row.index and row["positive_comment_ratio"] < 0.25:
        points.append(f"negative customer sentiment ({row['positive_comment_ratio']:.0%})")

    if "capability_request_rate" in row.index and row["capability_request_rate"] >= 0.60:
        points.append(f"high capability request rate ({row['capability_request_rate']:.0%})")

    return "; ".join(points[:5]) if points else "mixed signals across dimensions"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_decision_engine(features_path: Path, output_dir: Path) -> tuple[dict, dict]:
    df = pd.read_csv(features_path)
    missing_feats = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_feats:
        raise ValueError(f"concept_features missing columns: {sorted(missing_feats)}")

    X = df[FEATURE_COLUMNS].copy()

    # --- Layer 1: Baseline ---
    df["baseline_readiness_score"] = compute_baseline_readiness(df)
    df["baseline_rank"] = df["baseline_readiness_score"].rank(ascending=False, method="min").astype(int)

    # --- Layer 2: K-Means ---
    cluster_features = [
        "demand_intensity",
        "engagement_depth_norm",
        "feasibility_risk",
        "repeatability",
        "revenue_potential",
    ]
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])
    cluster_labels, cluster_names, kmeans = run_kmeans_clustering(
        X_cluster, df, cluster_features
    )

    df["cluster_id"] = cluster_labels
    df["cluster_profile"] = df["cluster_id"].map(cluster_names)

    # --- Layer 3: Random Forest ---
    df["synthetic_outcome"] = df.apply(assign_synthetic_outcome, axis=1)
    clf = train_random_forest(X, df["synthetic_outcome"])

    # Cross-validation (3-fold; small dataset, for robustness check only)
    n_classes_in_data = df["synthetic_outcome"].nunique()
    n_splits = min(3, n_classes_in_data)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(clf, X, df["synthetic_outcome"], cv=cv, scoring="accuracy")
    cv_accuracy_mean = float(cv_scores.mean())
    cv_accuracy_std = float(cv_scores.std())

    df["recommended_outcome"] = clf.predict(X)
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    df["ml_confidence_score"] = proba.max(axis=1).round(3)

    for i, cls in enumerate(classes):
        df[f"prob_{cls.replace(' ', '_')}"] = proba[:, i].round(3)

    # Blended readiness: baseline + top-class probability
    df["readiness_score"] = (
        df["baseline_readiness_score"] * 0.70 + df["ml_confidence_score"] * 30
    ).round(1)
    df["portfolio_rank"] = df["readiness_score"].rank(ascending=False, method="min").astype(int)

    # Final confidence blends data volume confidence with model certainty
    df["confidence_score"] = (
        df["confidence"] * 0.55 + df["ml_confidence_score"] * 0.45
    ).round(3)

    df["key_evidence"] = df.apply(build_evidence, axis=1)

    importance = extract_feature_importance(clf, FEATURE_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)

    result_cols = [
        "portfolio_rank",
        "concept_id",
        "concept_name",
        "industry",
        "readiness_score",
        "confidence_score",
        "recommended_outcome",
        "cluster_profile",
        "baseline_readiness_score",
        "ml_confidence_score",
        "demand_intensity",
        "engagement_depth_norm",
        "feasibility_risk",
        "repeatability",
        "segment_similarity",
        "revenue_potential",
        "strategic_fit",
        "positive_comment_ratio",
        "capability_request_rate",
        "avg_objection_count_text",
        "key_evidence",
    ]
    ranked = df.sort_values("portfolio_rank")[result_cols]
    ranked.to_csv(output_dir / "concept_recommendations.csv", index=False)
    df.to_csv(output_dir / "concept_decisions_full.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    cluster_summary = (
        df.groupby(["cluster_id", "cluster_profile"], as_index=False)
        .agg(
            concept_count=("concept_id", "count"),
            avg_demand=("demand_intensity", "mean"),
            avg_risk=("feasibility_risk", "mean"),
            avg_readiness=("readiness_score", "mean"),
            concepts=("concept_name", lambda s: ", ".join(s)),
        )
        .round(3)
    )
    cluster_summary.to_csv(output_dir / "cluster_summary.csv", index=False)

    outcome_dist = df["recommended_outcome"].value_counts().to_dict()
    report = {
        "concepts_scored": len(df),
        "baseline_model": {
            "type": "weighted_linear_score",
            "weights": BASELINE_WEIGHTS,
            "score_range": [1, 100],
        },
        "clustering": {
            "method": "KMeans",
            "n_clusters": N_CLUSTERS,
            "cluster_profiles": cluster_names,
        },
        "classifier": {
            "method": "RandomForestClassifier",
            "n_estimators": 200,
            "training_labels": "synthetic_rule_based",
            "outcome_distribution": outcome_dist,
        },
        "cross_validation": {
            "method": f"{n_splits}-fold stratified",
            "accuracy_mean": round(cv_accuracy_mean, 4),
            "accuracy_std": round(cv_accuracy_std, 4),
            "fold_scores": [round(float(s), 4) for s in cv_scores],
        },
        "top_features": importance.head(5).to_dict(orient="records"),
        "top_3_concepts": ranked.head(3)[
            ["portfolio_rank", "concept_name", "readiness_score", "recommended_outcome"]
        ].to_dict(orient="records"),
        "archive_candidates": ranked[ranked["recommended_outcome"] == "Archive"][
            "concept_name"
        ].tolist(),
    }

    with open(output_dir / "model_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    artifacts = {
        "df": df,
        "X": X,
        "clf": clf,
        "feature_columns": FEATURE_COLUMNS,
        "importance": importance,
        "ranked": ranked,
        "cluster_summary": cluster_summary,
        "cluster_names": cluster_names,
    }

    return report, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML decision engine on concept features")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "concept_features.csv",
        help="Path to concept_features.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed",
        help="Output directory",
    )
    args = parser.parse_args()

    report, _artifacts = run_decision_engine(args.features, args.output)

    print("--- Phase 3: ML Decision Engine ---")
    print(f"Concepts scored: {report['concepts_scored']}")
    print("\nOutcome distribution:")
    for outcome, count in report["classifier"]["outcome_distribution"].items():
        print(f"  {outcome}: {count}")

    print("\nCluster profiles:")
    for cid, name in report["clustering"]["cluster_profiles"].items():
        print(f"  Cluster {cid}: {name}")

    print("\nTop feature importances:")
    for feat in report["top_features"]:
        print(f"  {feat['feature']}: {feat['importance']:.3f}")

    print("\nTop 3 concepts:")
    for item in report["top_3_concepts"]:
        print(
            f"  #{item['portfolio_rank']} {item['concept_name']} "
            f"— readiness {item['readiness_score']}, outcome: {item['recommended_outcome']}"
        )

    print(f"\nOutput written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
