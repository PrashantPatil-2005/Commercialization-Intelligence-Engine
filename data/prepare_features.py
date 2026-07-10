"""
Phase 2: Data Cleaning, Validation, and Feature Engineering

Loads raw CSVs from Phase 1, cleans noisy/missing data, validates schema,
and engineers concept-level business signal features for the ML pipeline.

Core engineered signals (per assignment brief):
  - demand_intensity
  - engagement_depth
  - feasibility_risk
  - repeatability
  - segment_similarity
  - revenue_potential
  - strategic_fit
  - confidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema expectations
# ---------------------------------------------------------------------------

SCHEMA = {
    "product_concepts": [
        "concept_id",
        "concept_name",
        "industry",
        "problem_area",
        "target_user",
        "delivery_complexity",
        "strategic_fit",
    ],
    "customer_demo_signals": [
        "customer_id",
        "concept_id",
        "segment",
        "demo_date",
        "feedback_score",
        "follow_up_requested",
        "decision_maker_present",
        "objections_count",
    ],
    "sandbox_usage": [
        "customer_id",
        "concept_id",
        "trial_sessions",
        "feature_clicks",
        "repeat_usage_days",
        "active_users",
        "time_spent",
        "abandoned_features",
    ],
    "commercial_signals": [
        "customer_id",
        "concept_id",
        "pilot_interest",
        "urgency_score",
        "budget_signal",
        "willingness_to_pay",
        "expected_value",
        "implementation_risk",
    ],
    "text_feedback": [
        "customer_id",
        "concept_id",
        "customer_comments",
        "pain_point_statements",
        "objection_themes",
        "requested_capabilities",
    ],
}

NUMERIC_RANGES = {
    "feedback_score": (1.0, 5.0),
    "delivery_complexity": (1, 5),
    "strategic_fit": (0.0, 1.0),
    "pilot_interest": (0.0, 1.0),
    "urgency_score": (0.0, 1.0),
    "budget_signal": (0.0, 1.0),
    "willingness_to_pay": (0.0, 1.0),
    "expected_value": (0.0, 1.0),
    "implementation_risk": (0.0, 1.0),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "product_concepts": raw_dir / "product_concepts.csv",
        "customer_demo_signals": raw_dir / "customer_demo_signals.csv",
        "sandbox_usage": raw_dir / "sandbox_usage.csv",
        "commercial_signals": raw_dir / "commercial_signals.csv",
        "text_feedback": raw_dir / "text_feedback.csv",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}. Run generate_mock_data.py first.")

    return {name: pd.read_csv(path) for name, path in paths.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_schema(dfs: dict[str, pd.DataFrame]) -> list[str]:
    issues = []
    for name, expected_cols in SCHEMA.items():
        df = dfs[name]
        missing_cols = set(expected_cols) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_cols)
        if missing_cols:
            issues.append(f"{name}: missing columns {sorted(missing_cols)}")
        if extra_cols:
            issues.append(f"{name}: unexpected columns {sorted(extra_cols)}")
        if df.empty:
            issues.append(f"{name}: dataframe is empty")
    return issues


def validate_ranges(dfs: dict[str, pd.DataFrame]) -> list[str]:
    issues = []
    for col, (low, high) in NUMERIC_RANGES.items():
        for name, df in dfs.items():
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            out_of_range = series[(series < low) | (series > high)]
            if len(out_of_range) > 0:
                issues.append(
                    f"{name}.{col}: {len(out_of_range)} values outside [{low}, {high}]"
                )
    return issues


def validate_referential_integrity(dfs: dict[str, pd.DataFrame]) -> list[str]:
    issues = []
    concept_ids = set(dfs["product_concepts"]["concept_id"])

    for name in ["customer_demo_signals", "sandbox_usage", "commercial_signals", "text_feedback"]:
        orphans = set(dfs[name]["concept_id"]) - concept_ids
        if orphans:
            issues.append(f"{name}: unknown concept_ids {sorted(orphans)[:5]}")

    return issues


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _cap_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Winsorize outliers using IQR bounds."""
    clean = series.dropna()
    if len(clean) < 4:
        return series
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return series.clip(lower=lower, upper=upper)


def clean_demo_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["demo_date"] = pd.to_datetime(out["demo_date"], errors="coerce")
    out["follow_up_requested"] = out["follow_up_requested"].astype(bool)
    out["decision_maker_present"] = out["decision_maker_present"].astype(bool)

    # Impute missing feedback with concept-level median, fallback global median
    global_median = out["feedback_score"].median()
    out["feedback_score"] = out.groupby("concept_id")["feedback_score"].transform(
        lambda s: s.fillna(s.median())
    )
    out["feedback_score"] = out["feedback_score"].fillna(global_median)
    out["feedback_score"] = out["feedback_score"].clip(1.0, 5.0)

    out["objections_count"] = out["objections_count"].fillna(0).astype(int)
    return out


def clean_sandbox_usage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["trial_sessions", "feature_clicks", "active_users", "abandoned_features"]:
        out[col] = out[col].fillna(0).astype(int)

    # Impute time_spent with concept median; cap extreme outliers
    out["time_spent"] = out.groupby("concept_id")["time_spent"].transform(
        lambda s: s.fillna(s.median())
    )
    out["time_spent"] = out["time_spent"].fillna(out["time_spent"].median())
    out["time_spent"] = _cap_outliers_iqr(out["time_spent"])

    out["repeat_usage_days"] = out.groupby("concept_id")["repeat_usage_days"].transform(
        lambda s: s.fillna(s.median())
    )
    out["repeat_usage_days"] = out["repeat_usage_days"].fillna(0)
    out["repeat_usage_days"] = out["repeat_usage_days"].clip(lower=0)

    return out


def clean_commercial_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "pilot_interest",
        "urgency_score",
        "budget_signal",
        "willingness_to_pay",
        "expected_value",
        "implementation_risk",
    ]
    for col in numeric_cols:
        out[col] = out.groupby("concept_id")[col].transform(lambda s: s.fillna(s.median()))
        out[col] = out[col].fillna(out[col].median())
        out[col] = out[col].clip(0.0, 1.0)
    return out


def clean_text_feedback(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_cols = [
        "customer_comments",
        "pain_point_statements",
        "objection_themes",
        "requested_capabilities",
    ]
    for col in text_cols:
        out[col] = out[col].fillna("").astype(str).str.strip()
    return out


def clean_product_concepts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delivery_complexity"] = out["delivery_complexity"].clip(1, 5).astype(int)
    out["strategic_fit"] = out["strategic_fit"].clip(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Interaction-level feature engineering
# ---------------------------------------------------------------------------

def build_interaction_features(
    demos: pd.DataFrame,
    usage: pd.DataFrame,
    commercial: pd.DataFrame,
    concepts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Customer-concept grain. Computes row-level business signals before
    concept-level aggregation.
    """
    base = (
        demos.merge(commercial, on=["customer_id", "concept_id"], how="left")
        .merge(usage, on=["customer_id", "concept_id"], how="left")
        .merge(
            concepts[["concept_id", "delivery_complexity", "strategic_fit"]],
            on="concept_id",
            how="left",
        )
    )

    # Fill usage gaps for non-trial customers with zeros
    usage_cols = [
        "trial_sessions",
        "feature_clicks",
        "repeat_usage_days",
        "active_users",
        "time_spent",
        "abandoned_features",
    ]
    for col in usage_cols:
        base[col] = base[col].fillna(0)

    commercial_cols = [
        "pilot_interest",
        "urgency_score",
        "budget_signal",
        "willingness_to_pay",
        "expected_value",
        "implementation_risk",
    ]
    for col in commercial_cols:
        base[col] = base.groupby("concept_id")[col].transform(lambda s: s.fillna(s.median()))
        base[col] = base[col].fillna(base[col].median())

    # --- Demand Intensity ---
    # Combines feedback_score, urgency_score, and follow_up_requested
    base["follow_up_flag"] = base["follow_up_requested"].astype(int)
    base["demand_intensity"] = (
        (base["feedback_score"] / 5.0) * 0.45
        + base["urgency_score"] * 0.35
        + base["follow_up_flag"] * 0.20
    )

    # --- Engagement Depth ---
    # trial_sessions * time_spent / abandoned_features (safe denominator)
    denominator = base["abandoned_features"].clip(lower=1)
    base["engagement_depth"] = (base["trial_sessions"] * base["time_spent"]) / denominator

    # --- Feasibility Risk ---
    # Invert delivery_complexity (1=easy, 5=hard) and combine with implementation_risk
    inverted_complexity = (6 - base["delivery_complexity"]) / 5.0
    base["feasibility_risk"] = (
        (1.0 - inverted_complexity) * 0.55 + base["implementation_risk"] * 0.45
    )

    # --- Revenue Potential (interaction-level) ---
    base["revenue_potential"] = (
        base["willingness_to_pay"] * 0.40
        + base["expected_value"] * 0.30
        + base["budget_signal"] * 0.20
        + base["pilot_interest"] * 0.10
    )

    return base


# ---------------------------------------------------------------------------
# Concept-level aggregation
# ---------------------------------------------------------------------------

def _segment_similarity(segment_series: pd.Series) -> float:
    """
    High score = demand is consistent across segments (not concentrated in one).
    Uses normalized entropy of segment counts.
    """
    counts = segment_series.value_counts(normalize=True)
    if len(counts) <= 1:
        return 0.0
    entropy = -float((counts * np.log(counts + 1e-9)).sum())
    max_entropy = np.log(len(counts))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _confidence_score(n_demos: int, n_trials: int, missing_rate: float) -> float:
    """
    Higher when more observations and less missingness.
    """
    volume = min(1.0, (n_demos / 25.0) * 0.6 + (n_trials / 10.0) * 0.4)
    completeness = 1.0 - missing_rate
    return float(np.clip(volume * 0.65 + completeness * 0.35, 0.1, 1.0))


def build_concept_features(
    interactions: pd.DataFrame,
    concepts: pd.DataFrame,
    demo_missing_rate: pd.Series | None = None,
) -> pd.DataFrame:
    """Aggregate interaction features to one row per concept."""
    agg = interactions.groupby("concept_id").agg(
        n_demos=("customer_id", "count"),
        n_trials=("trial_sessions", lambda s: int((s > 0).sum())),
        avg_feedback_score=("feedback_score", "mean"),
        follow_up_rate=("follow_up_flag", "mean"),
        decision_maker_rate=("decision_maker_present", "mean"),
        avg_objections=("objections_count", "mean"),
        demand_intensity=("demand_intensity", "mean"),
        engagement_depth=("engagement_depth", "mean"),
        feasibility_risk=("feasibility_risk", "mean"),
        revenue_potential=("revenue_potential", "mean"),
        avg_trial_sessions=("trial_sessions", "mean"),
        avg_time_spent=("time_spent", "mean"),
        avg_repeat_usage_days=("repeat_usage_days", "mean"),
        avg_feature_clicks=("feature_clicks", "mean"),
        avg_active_users=("active_users", "mean"),
        avg_abandoned_features=("abandoned_features", "mean"),
        avg_willingness_to_pay=("willingness_to_pay", "mean"),
        avg_pilot_interest=("pilot_interest", "mean"),
        avg_implementation_risk=("implementation_risk", "mean"),
        segments_reached=("segment", "nunique"),
    )

    # Repeatability: repeat usage + follow-up + returning trial behavior
    repeatability = interactions.groupby("concept_id").apply(
        lambda g: float(
            np.clip(
                (g["repeat_usage_days"].mean() / 15.0) * 0.35
                + g["follow_up_flag"].mean() * 0.35
                + (g["trial_sessions"].mean() / 20.0) * 0.30,
                0,
                1,
            )
        ),
        include_groups=False,
    )
    agg["repeatability"] = repeatability

    segment_sim = interactions.groupby("concept_id")["segment"].apply(_segment_similarity)
    agg["segment_similarity"] = segment_sim

    # delivery_complexity already present on concepts; strategic_fit carried via final merge

    # Confidence based on observation volume and pre-clean missingness
    if demo_missing_rate is None:
        demo_missing_rate = pd.Series(0.0, index=agg.index)
    agg["confidence"] = [
        _confidence_score(
            int(agg.loc[cid, "n_demos"]),
            int(agg.loc[cid, "n_trials"]),
            float(demo_missing_rate.get(cid, 0.0)),
        )
        for cid in agg.index
    ]

    # Normalize skewed features to 0-1 for ML readiness
    skewed = ["engagement_depth", "avg_feature_clicks", "avg_trial_sessions"]
    for col in skewed:
        max_val = agg[col].max()
        if max_val > 0:
            agg[f"{col}_norm"] = (agg[col] / max_val).clip(0, 1)
        else:
            agg[f"{col}_norm"] = 0.0

    agg["engagement_depth_norm"] = (
        agg["engagement_depth"] / agg["engagement_depth"].max()
        if agg["engagement_depth"].max() > 0
        else 0.0
    )

    result = concepts.merge(agg.reset_index(), on="concept_id", how="left", suffixes=("", "_agg"))
    dup_cols = [c for c in result.columns if c.endswith("_agg")]
    result = result.drop(columns=dup_cols)
    result["strategic_fit"] = result["strategic_fit"].clip(0, 1)
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(raw_dir: Path, output_dir: Path) -> dict:
    dfs = load_raw_data(raw_dir)

    validation_issues = (
        validate_schema(dfs)
        + validate_ranges(dfs)
        + validate_referential_integrity(dfs)
    )

    raw_demos = dfs["customer_demo_signals"]
    demo_missing_rate = raw_demos.groupby("concept_id")["feedback_score"].apply(
        lambda s: float(s.isna().mean())
    )

    concepts = clean_product_concepts(dfs["product_concepts"])
    demos = clean_demo_signals(dfs["customer_demo_signals"])
    usage = clean_sandbox_usage(dfs["sandbox_usage"])
    commercial = clean_commercial_signals(dfs["commercial_signals"])
    text = clean_text_feedback(dfs["text_feedback"])

    interactions = build_interaction_features(demos, usage, commercial, concepts)
    concept_features = build_concept_features(interactions, concepts, demo_missing_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    concepts.to_csv(output_dir / "product_concepts_clean.csv", index=False)
    demos.to_csv(output_dir / "customer_demo_signals_clean.csv", index=False)
    usage.to_csv(output_dir / "sandbox_usage_clean.csv", index=False)
    commercial.to_csv(output_dir / "commercial_signals_clean.csv", index=False)
    text.to_csv(output_dir / "text_feedback_clean.csv", index=False)
    interactions.to_csv(output_dir / "interaction_features.csv", index=False)

    feature_cols = [
        "concept_id",
        "concept_name",
        "industry",
        "problem_area",
        "target_user",
        "delivery_complexity",
        "strategic_fit",
        "n_demos",
        "n_trials",
        "demand_intensity",
        "engagement_depth",
        "engagement_depth_norm",
        "feasibility_risk",
        "repeatability",
        "segment_similarity",
        "revenue_potential",
        "confidence",
        "avg_feedback_score",
        "follow_up_rate",
        "decision_maker_rate",
        "avg_willingness_to_pay",
        "avg_pilot_interest",
        "segments_reached",
    ]
    concept_features[feature_cols].to_csv(output_dir / "concept_features.csv", index=False)
    concept_features.to_csv(output_dir / "concept_features_full.csv", index=False)

    report = {
        "validation_issues": validation_issues,
        "validation_passed": len(validation_issues) == 0,
        "concept_count": len(concept_features),
        "interaction_count": len(interactions),
        "feature_summary": {
            col: {
                "mean": round(float(concept_features[col].mean()), 4),
                "min": round(float(concept_features[col].min()), 4),
                "max": round(float(concept_features[col].max()), 4),
            }
            for col in [
                "demand_intensity",
                "engagement_depth",
                "feasibility_risk",
                "repeatability",
                "segment_similarity",
                "revenue_potential",
                "strategic_fit",
                "confidence",
            ]
        },
    }

    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw data and engineer concept features")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path(__file__).parent / "raw",
        help="Directory containing raw CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "processed",
        help="Directory for cleaned data and features",
    )
    args = parser.parse_args()

    report = run_pipeline(args.raw, args.output)

    print("--- Phase 2: Cleaning & Feature Engineering ---")
    print(f"Validation passed: {report['validation_passed']}")
    if report["validation_issues"]:
        for issue in report["validation_issues"]:
            print(f"  ! {issue}")

    print(f"\nConcepts processed: {report['concept_count']}")
    print(f"Interactions processed: {report['interaction_count']}")
    print("\nFeature ranges (concept-level):")
    for feat, stats in report["feature_summary"].items():
        print(f"  {feat}: mean={stats['mean']}, min={stats['min']}, max={stats['max']}")

    print(f"\nOutput written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
