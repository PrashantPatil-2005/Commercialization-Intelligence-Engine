# Commercialization Intelligence Engine

An AI/ML-powered decision engine that transforms raw customer signals—demo feedback, sandbox usage, commercial intent, and text comments—into actionable commercial recommendations for a portfolio of product concepts.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

> Pre-computed outputs are available in `data/processed/`, so the dashboard works without re-running the full pipeline.

## Architecture

```
Customer Signals ──> Feature Engineering ──> ML Decision Engine ──> AI Explainability ──> Dashboard
   (Phase 1)              (Phase 2)              (Phase 3)             (Phase 4)          (Phase 5)
```

The pipeline evaluates 12 product concepts across 7 industries and recommends one of five outcomes: **MVP Build**, **Customer Pilot**, **Reusable Asset**, **Incubate**, or **Archive**.

## Key Features

- **Three-layer decision system** — Baseline weighted scoring + K-Means clustering + Random Forest classification
- **SHAP explainability** — Per-concept feature contributions with human-readable AI narratives
- **Interactive Streamlit dashboard** — Ranked portfolio, visual analytics, and concept drill-down
- **Synthetic data generator** — Realistic noisy data with hidden latent commercial potential for ground truth

## Project Structure

```
Commercialization Intelligence Engine/
├── README.md
├── requirements.txt
├── app/
│   └── streamlit_app.py              # Interactive dashboard (Phase 5)
├── data/
│   ├── generate_mock_data.py         # Synthetic data generator (Phase 1)
│   ├── prepare_features.py           # Data cleaning & feature engineering (Phase 2)
│   ├── raw/                          # 6 raw CSV files (customers, signals, feedback)
│   └── processed/                    # 14 output files (features, decisions, insights)
├── models/
│   ├── decision_engine.py            # ML pipeline: scoring, clustering, classification (Phase 3)
│   └── insight_layer.py              # SHAP values & narrative generation (Phase 4)
└── notebooks/
    └── commercialization_results.ipynb  # Results notebook with all visualizations
```

## Prerequisites

- Python 3.12+

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit Dashboard (Recommended)

```bash
streamlit run app/streamlit_app.py
```

Opens an interactive dashboard with:
- Executive summary and key metrics
- Ranked portfolio table
- Readiness bar chart, outcome pie chart, cluster scatter plot, feature importance chart
- Per-concept SHAP waterfall charts and AI narratives

### Jupyter Notebook

```bash
jupyter notebook notebooks/commercialization_results.ipynb
```

Runs the full pipeline and displays all results inline across 9 sections.

### Regenerate Data

```bash
python data/generate_mock_data.py
python data/prepare_features.py
```

## Pipeline Phases

### Phase 1: Data Generation (`data/generate_mock_data.py`)
Generates synthetic customer interaction data for 12 product concepts across 80 customers. Includes a hidden `_latent_commercial_potential` variable (Beta(2,2) distributed) that drives all downstream signals. Intentionally injects missing values (~4%), outliers (~3%), and noise to test the cleaning pipeline.

### Phase 2: Feature Engineering (`data/prepare_features.py`)
Loads raw CSVs, validates schemas, imputes missing values (concept-level medians), caps outliers (IQR method), and engineers four interaction-level features:
- **demand_intensity** — Weighted blend of feedback score, urgency, and follow-up signals
- **engagement_depth** — Trial sessions × time spent / abandoned features
- **feasibility_risk** — Inverted delivery complexity + implementation risk
- **revenue_potential** — Willingness to pay + expected value + budget signal + pilot interest

Aggregates to concept-level features (12 concepts × 23 features) with min-max normalization.

### Phase 3: Decision Engine (`models/decision_engine.py`)
Three-layer system:
1. **Baseline weighted score** (readiness 1–100) with domain-informed weights
2. **K-Means clustering** (k=4) on demand, engagement, feasibility, repeatability, and revenue
3. **Random Forest classifier** (200 estimators, max_depth=4, balanced weights) predicting 5 outcome classes

Final blended score: 70% baseline + 30% ML confidence.

### Phase 4: Insight Layer (`models/insight_layer.py`)
Uses SHAP TreeExplainer on the trained Random Forest to compute per-prediction feature contributions. Generates human-readable narratives summarizing top drivers (e.g., "strong demand intensity, moderate pilot interest").

### Phase 5: Presentation (`app/streamlit_app.py`)
Interactive Streamlit dashboard with cached pipeline execution, 4 visual analytics charts, and per-concept drill-down with SHAP waterfall charts.

## Data

### Raw Data (`data/raw/`)

| File | Records | Description |
|------|---------|-------------|
| `product_concepts.csv` | 12 | Concept metadata (name, industry, problem area, complexity, strategic fit) |
| `customers.csv` | 80 | Customer profiles across 5 segments |
| `customer_demo_signals.csv` | 257 | Demo session records (feedback score, follow-up, objections) |
| `sandbox_usage.csv` | 76 | Trial usage metrics (sessions, clicks, time spent) |
| `commercial_signals.csv` | 230 | Commercial intent (pilot interest, urgency, budget, willingness to pay) |
| `text_feedback.csv` | 230 | Qualitative feedback (comments, pain points, capabilities) |

### Processed Data (`data/processed/`)

| File | Description |
|------|-------------|
| `concept_features.csv` | Primary ML input: 12 concepts × 23 engineered features |
| `concept_recommendations.csv` | Final ranked portfolio with readiness, outcome, evidence |
| `concept_insights.csv` | SHAP values per concept + AI narrative |
| `concept_decisions_full.csv` | All intermediate model outputs |
| `feature_importance.csv` | Top 10 features by Random Forest importance |
| `cluster_summary.csv` | 4 cluster profiles with concept lists |
| `interaction_features.csv` | 257 customer-concept interaction rows |
| `model_report.json` | Full model metadata, weights, distributions |
| `validation_report.json` | Schema validation results, feature statistics |
| `executive_summary.txt` | Stakeholder-facing summary |

## Models

| Model | Type | Purpose |
|-------|------|---------|
| Baseline Score | Weighted linear combination | Deterministic readiness ranking (1–100) |
| K-Means | Unsupervised clustering (k=4) | Segment concepts by demand/effort profiles |
| Random Forest | Supervised classifier (200 trees) | Predict outcome class (MVP/Pilot/Asset/Incubate/Archive) |
| SHAP Explainer | Tree-based Shapley values | Per-prediction feature attribution |

**Top features by importance:** avg_pilot_interest (18.8%), demand_intensity (16.7%), strategic_fit (11.7%), follow_up_rate (11.1%), revenue_potential (10.2%)

## Limitations

- All data is synthetic; the Random Forest is trained on rule-based pseudo-labels, not real-world outcomes
- Cluster assignments and outcome predictions are illustrative of the pipeline, not validated business decisions
- The hidden `_latent_commercial_potential` variable is used only for data generation and is never exposed to the models

## Author

**PrashantPatil-2005** — prashanthpatil02@gmail.com
