# 🧠 Commercialization Intelligence Engine

> **An AI/ML-powered decision engine that transforms raw customer signals into evidence-based commercialization recommendations for early-stage product concepts.**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Business Context](#business-context)
3. [Problem Statement](#problem-statement)
4. [Solution Overview](#solution-overview)
5. [How It Works](#how-it-works)
6. [Key Results](#key-results)
7. [Architecture](#architecture)
8. [Technical Deep Dive](#technical-deep-dive)
9. [Data Design](#data-design)
10. [Feature Engineering](#feature-engineering)
11. [Machine Learning Pipeline](#machine-learning-pipeline)
12. [AI Explainability Layer](#ai-explainability-layer)
13. [Dashboard & Presentation](#dashboard--presentation)
14. [Project Structure](#project-structure)
15. [Getting Started](#getting-started)
16. [Usage Guide](#usage-guide)
17. [Limitations & Future Work](#limitations--future-work)
18. [Author](#author)

---

## Executive Summary

An innovation team maintains a portfolio of early-stage AI product concepts. Each concept has been shown to different customer groups through demos, discovery sessions, proof-of-concept workshops, and sandbox trials. The team needs to decide which concepts deserve investment and which should be paused, reshaped, or retired.

This project builds a **working Python prototype** that:

- Analyzes mock customer interaction and usage data across 12 product concepts
- Identifies commercialization signals from 5 data areas (demo feedback, sandbox usage, commercial intent, text feedback, concept metadata)
- Ranks product concepts using a **three-layer ML system** (weighted scoring + K-Means clustering + Random Forest classification)
- Recommends one of five commercial outcomes per concept with **explainable AI reasoning**
- Presents everything in an **interactive Streamlit dashboard** with SHAP waterfall charts and human-readable narratives

### Bottom Line

| Metric | Value |
|--------|-------|
| Concepts Analyzed | 12 |
| Industries Covered | 7 |
| Customer Segments | 5 |
| Data Records Processed | 873 |
| Engineered Features | 26 |
| ML Models Used | 3 (Baseline + K-Means + Random Forest) |
| Outcomes Recommended | 5 (MVP Build, Customer Pilot, Reusable Asset, Incubate, Archive) |

---

## Business Context

### The Challenge

An innovation team has several early-stage AI product concepts at various stages of customer validation. Every concept begins as a commercial possibility. As customer observations accumulate, the team needs to narrow the portfolio into clear outcomes.

**The core question:** Which concepts should we build, pilot, reuse, incubate, or archive?

### Why It Matters

- **Resource constraints** — Teams cannot pursue all 12 concepts simultaneously
- **Signal noise** — Customer feedback is inconsistent, contradictory, and spread across multiple touchpoints
- **Decision quality** — Opinion-based decisions lead to misallocated investment
- **Speed** — The team needs a systematic, repeatable way to evaluate concepts as new data arrives

### The Approach

Instead of relying on gut feeling, this engine converts observable customer behavior into evidence-based recommendations. It answers:

1. **How strong is the demand signal?** (Are customers actually interested?)
2. **Is it repeatable?** (Does demand appear across segments and over time?)
3. **Is it feasible?** (Can we deliver this without excessive effort?)
4. **Is it valuable?** (Will customers pay for it?)
5. **Does it fit our strategy?** (Does it align with our portfolio priorities?)

---

## Problem Statement

### Assignment Objective

Build a working Python-based prototype that:
- Analyzes mock customer interaction and usage data
- Identifies commercialization signals
- Ranks product concepts
- Recommends the next commercial action with explainable reasoning

### Required Commercial Outcomes

| Outcome | Meaning |
|---------|---------|
| **MVP Build** | Strong signal to build a focused version for validation |
| **Customer Pilot** | Clear customer pull and enough readiness to test with one or more customers |
| **Reusable Asset** | Repeatable demand across segments and strong potential for scaling beyond one customer |
| **Incubate** | Promising but insufficient evidence; needs more demos, sharper positioning, or experiments |
| **Archive** | Weak signal, low repeatability, poor fit, or high delivery effort relative to value |

### Evaluation Criteria

| Area | What Good Looks Like |
|------|---------------------|
| Problem Framing | Connects commercialization decisions to observable customer behavior |
| Data Thinking | Creates realistic mock data, handles noisy/missing signals, defines meaningful features |
| ML Application | Uses suitable analytical method, explains choice, accounts for uncertainty |
| AI Insight Layer | Turns model output into clear recommendations, evidence summaries, explainable narratives |
| Commercial Judgment | Balances demand, repeatability, value, feasibility, strategic fit, and delivery risk |
| Execution Quality | Produces clean code, usable visuals, and outputs stakeholders can understand quickly |

---

## Solution Overview

### The Pipeline Flow

```
Customer Signals  →  Feature Engineering  →  ML Decision Engine  →  AI Explainability  →  Dashboard
     (Phase 1)            (Phase 2)              (Phase 3)              (Phase 4)          (Phase 5)
```

### What Each Phase Does

| Phase | What It Does | Input | Output |
|-------|-------------|-------|--------|
| **Phase 1** | Generate realistic synthetic data | Random seed | 6 raw CSV files |
| **Phase 2** | Clean, validate, and engineer features | Raw CSVs | 12 concept-level features + 3 text features |
| **Phase 3** | Run 3-layer ML decision engine | Concept features | Readiness scores, outcomes, clusters |
| **Phase 4** | Generate SHAP explanations and narratives | Model artifacts | Per-concept AI narratives + executive summary |
| **Phase 5** | Present results in interactive dashboard | All outputs | Streamlit app with charts and drill-down |

### The Three-Layer ML System

**Why three layers?** No single method captures all aspects of commercialization readiness. Each layer compensates for the others' weaknesses.

| Layer | Method | Purpose | Strength |
|-------|--------|---------|----------|
| 1. Baseline | Weighted linear score | Deterministic ranking (1-100) | Interpretable, consistent, auditable |
| 2. Clustering | K-Means (k=4) | Group concepts by demand/effort profiles | Discovers natural patterns without labels |
| 3. Classification | Random Forest (200 trees) | Predict outcome class | Handles non-linear relationships, provides feature importance |

**Final score blending:** 70% baseline + 30% ML confidence

---

## How It Works

### Step-by-Step Walkthrough

#### 1. Data Generation

The system starts by generating realistic synthetic data. Each of the 12 product concepts has a hidden `_latent_commercial_potential` variable (Beta(2,2) distributed) that drives all downstream signals. High-potential concepts generate stronger feedback, more follow-up requests, higher willingness to pay, and deeper sandbox engagement.

**What makes the data realistic:**
- Missing values (~4%) — customers don't always complete surveys
- Outliers (~3%) — some trial sessions are unusually long or short
- Noise — random variation prevents perfect correlation
- Segment variation — Enterprise customers respond differently than SMBs

#### 2. Feature Engineering

Raw signals are transformed into 26 concept-level features across 8 dimensions:

- **Demand Intensity** — How strongly customers want this (feedback + urgency + follow-up)
- **Engagement Depth** — How deeply customers interact (sessions x time / abandoned features)
- **Feasibility Risk** — How hard it is to deliver (complexity + implementation risk)
- **Revenue Potential** — Willingness to pay + expected value + budget signals
- **Repeatability** — Whether usage patterns return over time
- **Segment Similarity** — Whether demand is consistent across customer segments
- **Strategic Fit** — Alignment with portfolio priorities
- **Confidence** — How much data we have (observation volume + completeness)

Plus 3 text-derived features:
- **Objection Count** — Number of distinct objection themes
- **Capability Request Rate** — Fraction of interactions with feature requests
- **Positive Comment Ratio** — Fraction of positive customer comments

#### 3. ML Decision Engine

Three models run sequentially:

**Layer 1 — Baseline Score:**
A weighted linear combination of normalized features. Weights are domain-informed (e.g., demand intensity and revenue potential each get 18% weight). Produces a deterministic score from 1-100.

**Layer 2 — K-Means Clustering:**
Groups concepts into 4 clusters based on demand, engagement, feasibility, repeatability, and revenue. Reveals natural groupings like "High Demand / Low Effort" vs "Low Demand / High Effort."

**Layer 3 — Random Forest Classifier:**
Trained on synthetic pseudo-labels (rule-based outcome assignments) to predict one of 5 outcome classes. Provides feature importance and class probabilities for confidence scoring.

#### 4. AI Explainability

SHAP (SHapley Additive exPlanations) TreeExplainer computes per-prediction feature contributions. For each concept, we know exactly which features pushed the prediction toward or away from each outcome.

A narrative generator converts SHAP values into human-readable text:
> "Recommended Outcome: Reusable Asset. Evidence: strong demand intensity (score: 5.8), low customer objection volume (score: 2.7), despite limited capability request rate (score: 8.3)."

#### 5. Interactive Dashboard

A Streamlit app with 4 tabs:
- **Overview** — Executive summary, KPI cards, ranked portfolio table
- **Analytics** — Readiness bar chart, outcome pie chart, cluster scatter, feature importance
- **Concept Explorer** — Per-concept drill-down with readiness progress bar, SHAP waterfall, AI narrative
- **Model Report** — Cross-validation results, baseline weights, cluster profiles, validation report

---

## Key Results

### Portfolio Distribution

| Outcome | Count | Percentage |
|---------|-------|-----------|
| Incubate | 4 | 33% |
| Customer Pilot | 3 | 25% |
| Archive | 3 | 25% |
| Reusable Asset | 2 | 17% |
| MVP Build | 0 | 0% |

### Top 3 Recommendations

| Rank | Concept | Industry | Readiness | Confidence | Outcome |
|------|---------|----------|-----------|------------|---------|
| 1 | Auto-Compliance for Healthcare | Healthcare | 65.7/100 | 89% | Reusable Asset |
| 2 | Smart Forecast Copilot | Energy & Utilities | 63.9/100 | 88% | Customer Pilot |
| 3 | Predictive Contract Assistant | Retail | 61.5/100 | 87% | Reusable Asset |

### Archive Candidates (Deprioritize)

| Concept | Industry | Readiness | Why |
|---------|----------|-----------|-----|
| Auto-Document for Energy | Energy & Utilities | 49.7/100 | Weak demand, minimal engagement, low revenue |
| Smart Workflow Copilot | Financial Services | 47.3/100 | Weak demand, minimal engagement, low revenue |
| Predictive Forecast Assistant | Financial Services | 46.4/100 | Weak demand, high feasibility risk, low revenue |

### Top Features by Importance

| Feature | Importance | What It Measures |
|---------|-----------|-----------------|
| Demand Intensity | 12.2% | How strongly customers want this concept |
| Pilot Interest | 11.9% | Fraction of customers requesting pilot |
| Customer Objection Volume | 10.8% | Number of distinct objection themes |
| Repeatability | 9.4% | Whether usage patterns return over time |
| Engagement Depth | 8.7% | How deeply customers interact in sandbox |

### Cross-Validation

| Method | Mean Accuracy | Std Dev |
|--------|--------------|---------|
| 3-fold stratified | 41.67% | +/- 11.79% |

> Note: With only 12 concepts and 4-5 outcome classes, this is illustrative of the pipeline rather than a statistically rigorous evaluation.

---

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT DASHBOARD                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │ Overview │ │Analytics │ │   Explorer   │ │  Model Report    │  │
│  └──────────┘ └──────────┘ └──────────────┘ └──────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                      AI EXPLAINABILITY LAYER                        │
│  ┌───────────────────────────┐  ┌──────────────────────────────┐   │
│  │   SHAP TreeExplainer      │  │   Narrative Generator         │   │
│  │   (per-prediction         │  │   (human-readable summaries   │   │
│  │    feature attribution)   │  │    per concept)               │   │
│  └───────────────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                       ML DECISION ENGINE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐    │
│  │   Baseline   │  │   K-Means    │  │   Random Forest       │    │
│  │   Weighted   │  │   Clustering │  │   Classifier          │    │
│  │   Score      │  │   (k=4)      │  │   (200 trees)         │    │
│  └──────────────┘  └──────────────┘  └───────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                     FEATURE ENGINEERING                              │
│  demand_intensity | engagement_depth | feasibility_risk |            │
│  revenue_potential | repeatability | segment_similarity |            │
│  strategic_fit | confidence | + 3 text features                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                         DATA LAYER                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Concepts │ │ Demos    │ │ Usage    │ │Commercial│ │ Feedback │ │
│  │ (12)     │ │ (257)    │ │ (76)     │ │ (230)    │ │ (230)    │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
generate_mock_data.py  →  data/raw/*.csv  (6 files)
         ↓
prepare_features.py    →  data/processed/*.csv  (14 files)
         ↓
decision_engine.py     →  concept_recommendations.csv, model_report.json
         ↓
insight_layer.py       →  concept_insights.csv, executive_summary.txt
         ↓
streamlit_app.py       →  Interactive Dashboard (localhost:8501)
```

---

## Technical Deep Dive

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.12+ | Core implementation |
| Data Processing | pandas | >=2.0.0 | Data manipulation and aggregation |
| Numerical Computing | NumPy | >=1.24.0 | Array operations, statistics |
| Machine Learning | scikit-learn | >=1.3.0 | Random Forest, K-Means, preprocessing |
| Explainability | SHAP | >=0.44.0 | TreeExplainer for feature attribution |
| Visualization | matplotlib | >=3.7.0 | Chart generation |
| Statistical Viz | seaborn | >=0.13.0 | Enhanced statistical plots |
| Dashboard | Streamlit | >=1.28.0 | Interactive web application |
| Data Generation | Faker | >=22.0.0 | Realistic synthetic customer names |
| Notebook | Jupyter | >=1.0.0 | Results notebook |

### ML Model Details

#### Baseline Weighted Score

```
readiness = (
    demand_intensity    × 0.18
  + engagement_depth    × 0.13
  + repeatability       × 0.13
  + segment_similarity  × 0.09
  + revenue_potential   × 0.18
  + strategic_fit       × 0.09
  + feasibility_ease    × 0.09    (1 - feasibility_risk)
  + positive_comment    × 0.05
  + capability_request  × 0.03
  + objection_ease      × 0.03    (1 - normalized_objection_count)
) × 100
```

**Why these weights?** Demand intensity and revenue potential are the strongest predictors of commercial success. Engagement depth and repeatability indicate sustained interest. Feasibility ease ensures we don't over-invest in overly complex concepts.

#### K-Means Clustering

- **Algorithm:** K-Means with k=4, 10 initializations
- **Features used:** demand_intensity, engagement_depth_norm, feasibility_risk, repeatability, revenue_potential
- **Cluster naming:** High/Low Demand × High/Low Effort (based on median splits)
- **Purpose:** Unsupervised grouping reveals natural concept groupings without using outcome labels

#### Random Forest Classifier

- **Architecture:** 200 decision trees, max_depth=4, class_weight="balanced"
- **Training labels:** Synthetic rule-based pseudo-labels (not real-world outcomes)
- **Features:** 13 features (10 original + 3 text-derived)
- **Output:** 5-class probability distribution per concept
- **Confidence:** Max class probability

#### SHAP Explainer

- **Method:** TreeExplainer (exact, tree-based Shapley values)
- **Output:** Per-feature contribution to each prediction
- **Narrative generation:** Top 3 positive contributors + top 1 negative contributor

### Cross-Validation

| Fold | Accuracy |
|------|----------|
| Fold 1 | 25.0% |
| Fold 2 | 50.0% |
| Fold 3 | 50.0% |
| **Mean** | **41.67%** |
| **Std Dev** | **+/- 11.79%** |

> With 12 samples and 4 classes, this is expected to be noisy. The value is in demonstrating the pipeline works, not in achieving high accuracy on synthetic data.

---

## Data Design

### Data Areas

| Area | File | Records | Key Fields | Purpose |
|------|------|---------|------------|---------|
| Product Concepts | `product_concepts.csv` | 12 | concept_name, industry, problem_area, delivery_complexity, strategic_fit | What we're evaluating |
| Customer Profiles | `customers.csv` | 80 | customer_name, segment | Who we're showing to |
| Demo Signals | `customer_demo_signals.csv` | 257 | feedback_score, follow_up_requested, decision_maker_present, objections_count | What happened in demos |
| Sandbox Usage | `sandbox_usage.csv` | 76 | trial_sessions, feature_clicks, repeat_usage_days, time_spent, abandoned_features | How customers used the product |
| Commercial Signals | `commercial_signals.csv` | 230 | pilot_interest, urgency_score, budget_signal, willingness_to_pay, implementation_risk | Intent to buy |
| Text Feedback | `text_feedback.csv` | 230 | customer_comments, pain_point_statements, objection_themes, requested_capabilities | Qualitative input |

### Data Quality

| Metric | Value |
|--------|-------|
| Total Records | 873 |
| Missing Values | ~4% (intentionally injected) |
| Outliers | ~3% (intentionally injected) |
| Schema Validation | Passed |
| Referential Integrity | Passed |
| Range Validation | Passed |

### Synthetic Data Design

The data generator uses a hidden `_latent_commercial_potential` variable (Beta(2,2) distributed) that drives all downstream signals:

- **High latent potential** → stronger feedback scores, more follow-up requests, higher willingness to pay, deeper sandbox engagement, fewer objections
- **Low latent potential** → weaker signals across all areas
- **Segment variation** — Enterprise customers give slightly higher scores than SMBs
- **Complexity tension** — Higher-potential concepts tend to be harder to deliver (realistic tradeoff)

The latent variable is **never exposed to the models** — it's only used to generate correlated data that mimics real-world patterns.

---

## Feature Engineering

### Interaction-Level Features (computed per customer-concept pair)

| Feature | Formula | Range | Meaning |
|---------|---------|-------|---------|
| `demand_intensity` | feedback/5 × 0.45 + urgency × 0.35 + follow_up × 0.20 | 0-1 | How strongly this customer wants this concept |
| `engagement_depth` | (trial_sessions × time_spent) / max(abandoned_features, 1) | 0+ | How deeply they engaged in sandbox |
| `feasibility_risk` | (1 - inverted_complexity) × 0.55 + impl_risk × 0.45 | 0-1 | How hard it would be to deliver |
| `revenue_potential` | wtp × 0.40 + expected_value × 0.30 + budget × 0.20 + pilot × 0.10 | 0-1 | Commercial value signal |

### Concept-Level Features (aggregated to one row per concept)

| Feature | Aggregation | Meaning |
|---------|-------------|---------|
| `demand_intensity` | Mean across interactions | Average demand signal |
| `engagement_depth_norm` | Normalized mean | Relative engagement depth |
| `repeatability` | Weighted composite | Whether usage returns over time |
| `segment_similarity` | Entropy-based score | Cross-segment consistency |
| `confidence` | Volume + completeness | How much data backs this concept |
| `avg_objection_count_text` | Mean from text | Average objection themes |
| `capability_request_rate` | Fraction with requests | Feature demand signal |
| `positive_comment_ratio` | Fraction positive | Customer sentiment |

---

## AI Explainability Layer

### How SHAP Works

SHAP (SHapley Additive exPlanations) is a game theory approach to explaining ML predictions. For each concept, SHAP answers: "How much did each feature contribute to pushing the prediction toward the assigned outcome?"

### Example Output

**Concept:** Auto-Compliance for Healthcare  
**Predicted Outcome:** Reusable Asset  
**Readiness Score:** 65.7/100

| Feature | SHAP Value | Direction | Interpretation |
|---------|-----------|-----------|----------------|
| demand_intensity | +0.098 | Toward | Strong demand pushes toward Reusable Asset |
| engagement_depth | +0.052 | Toward | Deep sandbox engagement supports outcome |
| repeatability | +0.049 | Toward | Usage patterns return over time |
| avg_objection_count_text | +0.070 | Toward | Low objection volume is positive |
| revenue_potential | +0.050 | Toward | Solid revenue signal |
| capability_request_rate | -0.008 | Against | Fewer capability requests slightly weakens signal |

**AI Narrative:**
> "Recommended Outcome: Reusable Asset. Evidence: strong demand intensity (score: 5.8), low customer objection volume (score: 2.7), weak positive comment ratio (score: 1.1), despite limited capability request rate (score: 8.3)."

### Narrative Generation Rules

1. Take the top 3 features with positive SHAP values (supporting the outcome)
2. Take the top 1 feature with negative SHAP value (countering the outcome)
3. Convert to human-readable labels with strength descriptors (strong/moderate/weak)
4. Format as: "Recommended Outcome: X. Evidence: [supporting], despite [countering]."

---

## Dashboard & Presentation

### Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Overview** | Executive summary, 5 KPI cards, ranked portfolio table with outcome badges |
| **Analytics** | Readiness bar chart, outcome donut chart, cluster scatter plot, feature importance chart |
| **Concept Explorer** | Concept selector, readiness progress bar, SHAP waterfall, AI narrative, evidence summary |
| **Model Report** | Cross-validation results, baseline weights, feature importance table, cluster profiles, validation report |

### Dashboard Features

- **Dark sidebar** with project info, pipeline status, and regenerate button
- **KPI cards** with colored left borders and hover effects
- **Outcome badges** — color-coded pills (green=Build, blue=Pilot, purple=Asset, yellow=Incubate, red=Archive)
- **Readiness progress bar** — green (>60), yellow (40-60), red (<40)
- **SHAP waterfall charts** — green bars for supporting features, red for countering
- **Cached pipeline** — Streamlit caches the ML pipeline so the dashboard loads fast on refresh

---

## Project Structure

```
Commercialization Intelligence Engine/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── app/
│   └── streamlit_app.py              # Interactive dashboard (Phase 5)
│
├── data/
│   ├── generate_mock_data.py         # Synthetic data generator (Phase 1)
│   ├── prepare_features.py           # Data cleaning & feature engineering (Phase 2)
│   │
│   ├── raw/                          # 6 raw CSV files
│   │   ├── product_concepts.csv
│   │   ├── customers.csv
│   │   ├── customer_demo_signals.csv
│   │   ├── sandbox_usage.csv
│   │   ├── commercial_signals.csv
│   │   └── text_feedback.csv
│   │
│   └── processed/                    # 14 output files
│       ├── concept_features.csv          # Primary ML input
│       ├── concept_recommendations.csv   # Final ranked portfolio
│       ├── concept_insights.csv          # SHAP values + narratives
│       ├── concept_decisions_full.csv    # All model outputs
│       ├── feature_importance.csv        # RF feature importance
│       ├── cluster_summary.csv           # Cluster profiles
│       ├── interaction_features.csv      # Customer-concept interactions
│       ├── model_report.json             # Model metadata
│       ├── validation_report.json        # Data quality report
│       ├── executive_summary.txt         # Stakeholder summary
│       └── *_clean.csv                   # Cleaned raw data
│
├── models/
│   ├── __init__.py
│   ├── decision_engine.py            # ML pipeline (Phase 3)
│   └── insight_layer.py              # SHAP + narratives (Phase 4)
│
└── notebooks/
    └── commercialization_results.ipynb  # Results notebook (10 sections)
```

---

## Getting Started

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/PrashantPatil-2005/Commercialization-Intelligence-Engine.git
cd Commercialization-Intelligence-Engine

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Option 1: Run the dashboard (recommended)
streamlit run app/streamlit_app.py

# Option 2: Run the notebook
jupyter notebook notebooks/commercialization_results.ipynb

# Option 3: Regenerate data and run pipeline
python data/generate_mock_data.py
python data/prepare_features.py
python models/decision_engine.py
python models/insight_layer.py
```

> Pre-computed outputs are available in `data/processed/`, so the dashboard works without re-running the full pipeline.

---

## Usage Guide

### Running the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at **http://localhost:8501** with 4 tabs:

1. **Overview** — See the executive summary and ranked portfolio
2. **Analytics** — Explore visual analytics (readiness, outcomes, clusters, features)
3. **Concept Explorer** — Drill into any concept for SHAP explanations and AI narratives
4. **Model Report** — Review model performance, cross-validation, and data quality

### Regenerating Data

If you want fresh synthetic data:

```bash
python data/generate_mock_data.py
python data/prepare_features.py
```

Then refresh the dashboard (it caches the pipeline, so you may need to clear the cache).

### Running Individual Phases

```bash
# Phase 1: Generate data
python data/generate_mock_data.py

# Phase 2: Clean and engineer features
python data/prepare_features.py

# Phase 3: Run ML decision engine
python models/decision_engine.py

# Phase 4: Generate insights
python models/insight_layer.py
```

---

## Limitations & Future Work

### Current Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| Synthetic data only | Results are illustrative, not validated business decisions | Pipeline architecture transfers to real data |
| Rule-based pseudo-labels | Random Forest trained on synthetic labels | Demonstrates the ML pipeline; real labels would improve accuracy |
| Small dataset (12 concepts) | Cross-validation is noisy | Appropriate for prototype; production would need 50+ concepts |
| Single random seed | Results are deterministic | Different seeds produce different but structurally similar outcomes |
| Keyword-based sentiment | Text feedback uses simple keyword matching | Transformer embeddings would improve accuracy |

### Future Work

| Area | Enhancement |
|------|-------------|
| **Real Data** | Connect to actual CRM/demo data instead of synthetic |
| **NLP** | Use sentence transformers for text feedback analysis |
| **Time Series** | Track concept signals over time for trend detection |
| **A/B Testing** | Compare different outcome assignment strategies |
| **Multi-label** | Allow concepts to have multiple outcomes (e.g., Pilot + Incubate) |
| **API** | REST API for integration with existing tools |
| **Alerts** | Notifications when concept scores cross thresholds |
| **What-If** | Scenario modeling ("what if we add 5 more demos?") |

---

## License

This project is for educational purposes as part of the NTT Case Study Intern Assignment.

---

## Author

**PrashantPatil-2005**  
Email: prashanthpatil02@gmail.com  
GitHub: [PrashantPatil-2005](https://github.com/PrashantPatil-2005)

---

## Acknowledgments

- NTT for the internship assignment brief
- SHAP library for explainability
- Streamlit for the dashboard framework
- scikit-learn for ML tools
