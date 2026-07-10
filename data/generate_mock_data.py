"""
Phase 1: Mock Data Generator
Generates realistic, noisy customer interaction data for the
AI/ML Commercialization Decision Engine prototype.

Five data areas:
  1. Product concepts
  2. Customer demo signals
  3. Sandbox usage behavior
  4. Commercial signals
  5. Text feedback

Each concept has a hidden latent_commercial_potential that drives
correlated signals across demos, usage, and commercial intent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
NUM_CONCEPTS = 12
NUM_CUSTOMERS = 80
DEMO_START = datetime(2024, 6, 1)
DEMO_END = datetime(2025, 6, 30)

INDUSTRIES = [
    "Financial Services",
    "Healthcare",
    "Manufacturing",
    "Retail",
    "Telecommunications",
    "Energy & Utilities",
    "Public Sector",
]

PROBLEM_AREAS = [
    "Document Intelligence",
    "Predictive Maintenance",
    "Customer Service Automation",
    "Fraud Detection",
    "Supply Chain Optimization",
    "Workforce Productivity",
    "Compliance Monitoring",
    "Revenue Forecasting",
]

TARGET_USERS = [
    "Operations Manager",
    "Data Science Lead",
    "CIO / CTO",
    "Business Analyst",
    "Customer Success Director",
    "Risk & Compliance Officer",
]

SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Public Sector", "Startup"]

PAIN_POINTS = [
    "manual data entry consumes 30% of analyst time",
    "legacy systems lack real-time visibility",
    "compliance audits require weeks of preparation",
    "customer churn signals arrive too late to act",
    "forecasting accuracy below 60% quarter over quarter",
    "incident response depends on tribal knowledge",
    "cross-team handoffs create duplicate work",
    "scaling pilots to production takes 9+ months",
]

OBJECTION_THEMES = [
    "data privacy and residency concerns",
    "integration complexity with existing ERP",
    "unclear ROI timeline",
    "lack of internal ML expertise",
    "vendor lock-in risk",
    "change management burden",
    "accuracy not proven on our data",
    "budget already allocated elsewhere",
]

REQUESTED_CAPABILITIES = [
    "SSO and role-based access control",
    "API integration with Salesforce",
    "custom model retraining pipeline",
    "audit trail and explainability dashboard",
    "multi-language document support",
    "on-premise deployment option",
    "real-time alerting via Slack/Teams",
    "batch and streaming inference modes",
]

POSITIVE_COMMENTS = [
    "This directly addresses our top-3 pain point from last quarter's review.",
    "The demo workflow felt intuitive — our team picked it up in under an hour.",
    "We would pilot this with our APAC division if pricing is reasonable.",
    "Decision-makers were impressed by the accuracy on sample documents.",
    "Strong alignment with our 2025 digital transformation roadmap.",
    "Sandbox trial showed measurable time savings on a real use case.",
]

NEUTRAL_COMMENTS = [
    "Interesting concept but we need to see it on our own data first.",
    "Promising direction — scheduling a follow-up with the architecture team.",
    "Some features overlap with tools we already own.",
    "Would need executive sponsorship before moving forward.",
    "Good demo, but our procurement cycle is 6+ months.",
]

NEGATIVE_COMMENTS = [
    "Accuracy on edge cases was disappointing during the live demo.",
    "Implementation timeline seems unrealistic for our infrastructure.",
    "We do not have budget for new AI initiatives this fiscal year.",
    "The value proposition was unclear for our specific segment.",
    "Competing internal project may cover similar ground.",
]

CONCEPT_NAME_TEMPLATES = [
    "{adj} {noun} Assistant",
    "Auto-{noun} for {industry_short}",
    "{noun} Intelligence Platform",
    "Smart {noun} Copilot",
]

CONCEPT_NOUNS = [
    "Document",
    "Contract",
    "Invoice",
    "Ticket",
    "Forecast",
    "Anomaly",
    "Compliance",
    "Workflow",
    "Insight",
]

CONCEPT_ADJECTIVES = [
    "Intelligent",
    "Adaptive",
    "Predictive",
    "Conversational",
    "Automated",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(series: pd.Series, low: float, high: float) -> pd.Series:
    return series.clip(low, high)


def _add_noise(values: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    return values + rng.normal(0, scale, size=values.shape)


def _maybe_missing(
    series: pd.Series,
    missing_rate: float,
    rng: np.random.Generator,
) -> pd.Series:
    mask = rng.random(len(series)) < missing_rate
    out = series.copy()
    out.loc[mask] = np.nan
    return out


def _random_date(rng: np.random.Generator, start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    offset = int(rng.integers(0, max(delta, 1)))
    return start + timedelta(days=offset)


def _concept_name(rng: np.random.Generator, industry: str) -> str:
    template = rng.choice(CONCEPT_NAME_TEMPLATES)
    industry_short = industry.split()[0]
    return template.format(
        adj=rng.choice(CONCEPT_ADJECTIVES),
        noun=rng.choice(CONCEPT_NOUNS),
        industry_short=industry_short,
    )


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_product_concepts(rng: np.random.Generator) -> pd.DataFrame:
    """One row per concept. latent_commercial_potential is hidden driver."""
    rows = []
    for i in range(NUM_CONCEPTS):
        concept_id = f"CONCEPT-{i + 1:03d}"
        industry = rng.choice(INDUSTRIES)
        latent = float(rng.beta(2, 2))  # spread across portfolio quality

        # Strategic fit correlates with latent potential but not perfectly
        strategic_fit = _clip(
            pd.Series([latent * 0.6 + rng.uniform(0.15, 0.35)]),
            0.1,
            1.0,
        ).iloc[0]

        # High-value concepts tend to be harder to deliver (realistic tension)
        delivery_complexity = int(
            np.round(
                np.clip(
                    rng.normal(loc=3.5 - latent * 1.2, scale=0.9),
                    1,
                    5,
                )
            )
        )

        rows.append(
            {
                "concept_id": concept_id,
                "concept_name": _concept_name(rng, industry),
                "industry": industry,
                "problem_area": rng.choice(PROBLEM_AREAS),
                "target_user": rng.choice(TARGET_USERS),
                "delivery_complexity": delivery_complexity,
                "strategic_fit": round(strategic_fit, 3),
                "_latent_commercial_potential": round(latent, 4),  # dev only
            }
        )

    return pd.DataFrame(rows)


def generate_customer_demo_signals(
    concepts: pd.DataFrame,
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Multiple demo sessions per concept across customer segments."""
    rows = []
    latent_map = concepts.set_index("concept_id")["_latent_commercial_potential"].to_dict()

    for _, concept in concepts.iterrows():
        concept_id = concept["concept_id"]
        latent = latent_map[concept_id]
        n_demos = int(rng.integers(12, 35))

        demo_customers = customers.sample(n=n_demos, replace=True, random_state=int(rng.integers(0, 1_000_000)))

        for _, customer in demo_customers.iterrows():
            segment = customer["segment"]
            segment_boost = {"Enterprise": 0.08, "Mid-Market": 0.04, "SMB": -0.05, "Public Sector": 0.02, "Startup": -0.02}
            base_feedback = latent * 4.5 + segment_boost.get(segment, 0) + rng.normal(0, 0.55)
            feedback_score = float(np.clip(np.round(base_feedback, 1), 1.0, 5.0))

            dm_prob = np.clip(0.25 + latent * 0.45 + (feedback_score - 3) * 0.08, 0.05, 0.95)
            decision_maker_present = bool(rng.random() < dm_prob)

            follow_up_prob = np.clip(0.15 + latent * 0.5 + (feedback_score - 3) * 0.12, 0.03, 0.92)
            follow_up_requested = bool(rng.random() < follow_up_prob)

            objections = int(
                np.clip(
                    rng.poisson(lam=max(0.3, 3.5 - latent * 2.5 - (feedback_score - 3) * 0.4)),
                    0,
                    8,
                )
            )

            rows.append(
                {
                    "customer_id": customer["customer_id"],
                    "concept_id": concept_id,
                    "segment": segment,
                    "demo_date": _random_date(rng, DEMO_START, DEMO_END).strftime("%Y-%m-%d"),
                    "feedback_score": feedback_score,
                    "follow_up_requested": follow_up_requested,
                    "decision_maker_present": decision_maker_present,
                    "objections_count": objections,
                }
            )

    df = pd.DataFrame(rows)

    # ~4% missing feedback scores (customer did not complete survey)
    df["feedback_score"] = _maybe_missing(df["feedback_score"], 0.04, rng)

    return df


def generate_sandbox_usage(
    demos: pd.DataFrame,
    concepts: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sandbox trials for customers who requested follow-up or had strong demos.
    Usage metrics correlate with concept latent potential and demo feedback.
    """
    latent_map = concepts.set_index("concept_id")["_latent_commercial_potential"].to_dict()
    rows = []

    # Eligible: follow-up requested OR high feedback
    eligible = demos[
        (demos["follow_up_requested"] == True)  # noqa: E712
        | (demos["feedback_score"].fillna(0) >= 4.0)
    ].drop_duplicates(subset=["customer_id", "concept_id"])

    # Not everyone who is eligible actually trials (~75% conversion)
    trial_mask = rng.random(len(eligible)) < 0.75
    trial_candidates = eligible.iloc[trial_mask]

    for _, row in trial_candidates.iterrows():
        concept_id = row["concept_id"]
        latent = latent_map[concept_id]
        feedback = row["feedback_score"] if pd.notna(row["feedback_score"]) else 3.0

        engagement_base = latent * 0.55 + (feedback - 3) * 0.08

        trial_sessions = int(np.clip(rng.poisson(lam=2 + engagement_base * 12), 1, 40))
        feature_clicks = int(np.clip(rng.poisson(lam=15 + engagement_base * 80), 3, 500))
        repeat_usage_days = int(np.clip(rng.poisson(lam=1 + engagement_base * 10), 0, 30))
        active_users = int(np.clip(rng.poisson(lam=1 + engagement_base * 6), 1, 25))

        # time_spent in minutes; inject outliers (~3%)
        time_spent = float(np.clip(rng.lognormal(mean=3.2 + engagement_base * 1.5, sigma=0.6), 5, 600))
        if rng.random() < 0.03:
            time_spent = float(rng.choice([1200, 1500, 0.5, 2.0]))  # outliers

        abandoned_features = int(np.clip(rng.poisson(lam=max(0.5, 4 - engagement_base * 3)), 0, 10))

        rows.append(
            {
                "customer_id": row["customer_id"],
                "concept_id": concept_id,
                "trial_sessions": trial_sessions,
                "feature_clicks": feature_clicks,
                "repeat_usage_days": repeat_usage_days,
                "active_users": active_users,
                "time_spent": round(time_spent, 1),
                "abandoned_features": abandoned_features,
            }
        )

    df = pd.DataFrame(rows)

    # ~6% entirely missing usage rows handled at join time; within-row gaps:
    df["repeat_usage_days"] = _maybe_missing(df["repeat_usage_days"], 0.05, rng)
    df["time_spent"] = _maybe_missing(df["time_spent"], 0.03, rng)

    return df


def generate_commercial_signals(
    demos: pd.DataFrame,
    usage: pd.DataFrame,
    concepts: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Commercial intent per customer-concept interaction.
    willingness_to_pay correlates with feedback_score + trial_sessions.
    """
    latent_map = concepts.set_index("concept_id")["_latent_commercial_potential"].to_dict()
    complexity_map = concepts.set_index("concept_id")["delivery_complexity"].to_dict()

    usage_agg = (
        usage.groupby(["customer_id", "concept_id"])
        .agg(
            trial_sessions=("trial_sessions", "mean"),
            repeat_usage_days=("repeat_usage_days", "mean"),
        )
        .reset_index()
    )

    merged = demos.merge(usage_agg, on=["customer_id", "concept_id"], how="left")
    rows = []

    for _, row in merged.drop_duplicates(subset=["customer_id", "concept_id"]).iterrows():
        concept_id = row["concept_id"]
        latent = latent_map[concept_id]
        complexity = complexity_map[concept_id]

        feedback = row["feedback_score"] if pd.notna(row["feedback_score"]) else 3.0
        sessions = row["trial_sessions"] if pd.notna(row.get("trial_sessions")) else 0.0
        repeat_days = row["repeat_usage_days"] if pd.notna(row.get("repeat_usage_days")) else 0.0

        # Normalized engagement contribution
        session_norm = min(sessions / 20.0, 1.0)
        repeat_norm = min(repeat_days / 15.0, 1.0)
        feedback_norm = (feedback - 1) / 4.0

        demand_signal = (
            latent * 0.35
            + feedback_norm * 0.30
            + session_norm * 0.20
            + repeat_norm * 0.15
            + rng.normal(0, 0.06)
        )
        demand_signal = float(np.clip(demand_signal, 0, 1))

        pilot_interest = round(float(np.clip(demand_signal * 0.9 + rng.normal(0, 0.08), 0, 1)), 3)
        urgency_score = round(float(np.clip(demand_signal * 0.7 + rng.normal(0, 0.1), 0, 1)), 3)
        budget_signal = round(float(np.clip(demand_signal * 0.65 + rng.normal(0, 0.12), 0, 1)), 3)

        # Key correlation: willingness_to_pay driven by feedback + sessions
        wtp_base = (
            0.20
            + feedback_norm * 0.30
            + session_norm * 0.25
            + latent * 0.20
            + rng.normal(0, 0.08)
        )
        willingness_to_pay = round(float(np.clip(wtp_base, 0, 1)), 3)

        expected_value = round(
            float(np.clip(willingness_to_pay * 0.7 + urgency_score * 0.2 + rng.normal(0, 0.07), 0, 1)),
            3,
        )

        # Implementation risk rises with delivery complexity, falls slightly with engagement
        impl_risk = (
            (complexity - 1) / 4.0 * 0.55
            + (1 - demand_signal) * 0.25
            + rng.normal(0, 0.08)
        )
        implementation_risk = round(float(np.clip(impl_risk, 0, 1)), 3)

        rows.append(
            {
                "customer_id": row["customer_id"],
                "concept_id": concept_id,
                "pilot_interest": pilot_interest,
                "urgency_score": urgency_score,
                "budget_signal": budget_signal,
                "willingness_to_pay": willingness_to_pay,
                "expected_value": expected_value,
                "implementation_risk": implementation_risk,
            }
        )

    df = pd.DataFrame(rows)

    # ~7% missing budget signals (unknown procurement status)
    df["budget_signal"] = _maybe_missing(df["budget_signal"], 0.07, rng)
    # ~5% missing willingness_to_pay (customer declined pricing discussion)
    df["willingness_to_pay"] = _maybe_missing(df["willingness_to_pay"], 0.05, rng)

    return df


def generate_text_feedback(
    demos: pd.DataFrame,
    commercial: pd.DataFrame,
    concepts: pd.DataFrame,
    faker: Faker,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Qualitative feedback correlated with commercial signal strength."""
    latent_map = concepts.set_index("concept_id")["_latent_commercial_potential"].to_dict()
    commercial_map = commercial.set_index(["customer_id", "concept_id"])["willingness_to_pay"].to_dict()

    rows = []
    for _, row in demos.drop_duplicates(subset=["customer_id", "concept_id"]).iterrows():
        key = (row["customer_id"], row["concept_id"])
        latent = latent_map[row["concept_id"]]
        wtp = commercial_map.get(key, np.nan)
        feedback = row["feedback_score"] if pd.notna(row["feedback_score"]) else 3.0

        sentiment = latent * 0.4 + (feedback - 3) / 4.0 * 0.35
        if pd.notna(wtp):
            sentiment += (wtp - 0.5) * 0.25
        sentiment += rng.normal(0, 0.12)

        if sentiment > 0.55:
            comment_pool = POSITIVE_COMMENTS
        elif sentiment < 0.25:
            comment_pool = NEGATIVE_COMMENTS
        else:
            comment_pool = NEUTRAL_COMMENTS

        n_objections = int(row["objections_count"])
        objection_sample = rng.choice(OBJECTION_THEMES, size=min(n_objections, 3), replace=False)
        objection_text = "; ".join(objection_sample) if len(objection_sample) > 0 else np.nan

        rows.append(
            {
                "customer_id": row["customer_id"],
                "concept_id": row["concept_id"],
                "customer_comments": rng.choice(comment_pool),
                "pain_point_statements": rng.choice(PAIN_POINTS),
                "objection_themes": objection_text,
                "requested_capabilities": "; ".join(
                    rng.choice(REQUESTED_CAPABILITIES, size=int(rng.integers(1, 4)), replace=False)
                ),
            }
        )

    df = pd.DataFrame(rows)

    # Realistic text gaps
    df["customer_comments"] = _maybe_missing(df["customer_comments"], 0.12, rng)
    df["pain_point_statements"] = _maybe_missing(df["pain_point_statements"], 0.08, rng)
    df["objection_themes"] = _maybe_missing(df["objection_themes"], 0.15, rng)
    df["requested_capabilities"] = _maybe_missing(df["requested_capabilities"], 0.10, rng)

    # Fill a subset of missing comments with generic Faker text (partial imputation example)
    missing_comments = df["customer_comments"].isna()
    if missing_comments.any():
        df.loc[missing_comments, "customer_comments"] = [
            faker.sentence(nb_words=12) for _ in range(missing_comments.sum())
        ]

    return df


def generate_customers(faker: Faker, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i in range(NUM_CUSTOMERS):
        rows.append(
            {
                "customer_id": f"CUST-{i + 1:04d}",
                "customer_name": faker.company(),
                "segment": rng.choice(SEGMENTS, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(
    concepts: pd.DataFrame,
    demos: pd.DataFrame,
    usage: pd.DataFrame,
    commercial: pd.DataFrame,
    text: pd.DataFrame,
) -> dict:
    """Basic schema and quality checks."""
    report = {
        "concept_count": len(concepts),
        "demo_records": len(demos),
        "usage_records": len(usage),
        "commercial_records": len(commercial),
        "text_records": len(text),
        "missing_rates": {},
        "correlation_wtp_feedback": None,
    }

    for name, df in [
        ("demos", demos),
        ("usage", usage),
        ("commercial", commercial),
        ("text", text),
    ]:
        report["missing_rates"][name] = {
            col: round(float(df[col].isna().mean()), 4)
            for col in df.columns
            if df[col].isna().any()
        }

    # Verify intentional correlation
    merged = demos.merge(commercial, on=["customer_id", "concept_id"]).merge(
        usage, on=["customer_id", "concept_id"], how="left"
    )
    merged = merged.dropna(subset=["feedback_score", "willingness_to_pay"])
    if len(merged) > 10:
        corr = merged[["feedback_score", "willingness_to_pay", "trial_sessions"]].corr()
        report["correlation_wtp_feedback"] = round(float(corr.loc["feedback_score", "willingness_to_pay"]), 3)
        report["correlation_wtp_sessions"] = round(float(corr.loc["trial_sessions", "willingness_to_pay"]), 3)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: Path, seed: int = RANDOM_SEED) -> None:
    rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating product concepts...")
    concepts = generate_product_concepts(rng)

    print("Generating customers...")
    customers = generate_customers(faker, rng)

    print("Generating customer demo signals...")
    demos = generate_customer_demo_signals(concepts, customers, rng)

    print("Generating sandbox usage behavior...")
    usage = generate_sandbox_usage(demos, concepts, rng)

    print("Generating commercial signals...")
    commercial = generate_commercial_signals(demos, usage, concepts, rng)

    print("Generating text feedback...")
    text = generate_text_feedback(demos, commercial, concepts, faker, rng)

    # Export concepts without hidden latent column (keep dev copy separate)
    concepts_public = concepts.drop(columns=["_latent_commercial_potential"])
    concepts.to_csv(output_dir / "_concepts_with_latent_dev_only.csv", index=False)

    concepts_public.to_csv(output_dir / "product_concepts.csv", index=False)
    customers.to_csv(output_dir / "customers.csv", index=False)
    demos.to_csv(output_dir / "customer_demo_signals.csv", index=False)
    usage.to_csv(output_dir / "sandbox_usage.csv", index=False)
    commercial.to_csv(output_dir / "commercial_signals.csv", index=False)
    text.to_csv(output_dir / "text_feedback.csv", index=False)

    report = validate_dataset(concepts_public, demos, usage, commercial, text)

    print("\n--- Dataset Summary ---")
    for key, value in report.items():
        print(f"  {key}: {value}")

    print(f"\nFiles written to: {output_dir.resolve()}")
    print("Note: _concepts_with_latent_dev_only.csv is for development/debugging only.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock commercialization dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "raw",
        help="Output directory for CSV files (default: data/raw)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()
    main(args.output, args.seed)
