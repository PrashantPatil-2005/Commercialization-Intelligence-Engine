# Commercialization Intelligence Engine

An ML-powered prototype that helps an innovation team decide which early-stage AI product concepts to build, pilot, incubate, or archive — based on customer behavior signals rather than gut feeling.

## Problem

An innovation team has 12 AI product concepts shown to customers through demos, sandbox trials, and discovery sessions. They need to decide which deserve investment. This project builds a Python prototype that analyzes customer signals and recommends one of five outcomes per concept:

| Outcome | Meaning |
|---------|---------|
| **MVP Build** | Strong signal — build a focused prototype |
| **Customer Pilot** | Ready to test with real customers |
| **Reusable Asset** | Repeatable demand across segments |
| **Incubate** | Promising but needs more evidence |
| **Archive** | Weak signal — deprioritize |

## How It Works

```
Raw customer data → Clean & engineer features → ML models → Explain decisions → Dashboard
```

The pipeline has 5 phases:

1. **Data generation** — Synthesize realistic customer interaction data (12 concepts, 80 customers, 5 data areas)
2. **Feature engineering** — Transform raw signals into 13 concept-level features (demand intensity, engagement depth, revenue potential, etc.)
3. **ML decision engine** — Three-layer system: weighted baseline score + K-Means clustering + Random Forest classifier
4. **Explainability** — SHAP values show which features drove each recommendation, plus human-readable narratives
5. **Dashboard** — Interactive Streamlit app with portfolio ranking, charts, and per-concept drill-down

## Key Results

| Metric | Value |
|--------|-------|
| Concepts analyzed | 12 |
| Industries | 7 |
| Top concept | Auto-Compliance for Healthcare (readiness 65.7/100) |
| Recommended for advancement | 5 concepts (MVP Build + Pilot + Asset) |
| Archive candidates | 3 concepts |

**Top 3 by readiness:**
1. Auto-Compliance for Healthcare — Reusable Asset (65.7/100, 89% confidence)
2. Smart Forecast Copilot — Customer Pilot (63.9/100, 88% confidence)
3. Predictive Contract Assistant — Reusable Asset (61.5/100, 87% confidence)

## Project Structure

```
Commercialization Intelligence Engine/
├── app/
│   └── streamlit_app.py           # Dashboard (Phase 5)
├── data/
│   ├── generate_mock_data.py      # Synthetic data generator (Phase 1)
│   ├── prepare_features.py        # Cleaning & feature engineering (Phase 2)
│   ├── raw/                       # 6 CSV files (concepts, customers, signals, feedback)
│   └── processed/                 # Features, decisions, insights, reports
├── models/
│   ├── decision_engine.py         # ML pipeline (Phase 3)
│   └── insight_layer.py           # SHAP + narratives (Phase 4)
├── notebooks/
│   └── commercialization_results.ipynb  # Results notebook
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

# Option 1: Dashboard (recommended)
streamlit run app/streamlit_app.py

# Option 2: Notebook
jupyter notebook notebooks/commercialization_results.ipynb

# Option 3: Regenerate data
python data/generate_mock_data.py
python data/prepare_features.py
```

> Pre-computed outputs exist in `data/processed/`, so the dashboard works without re-running the pipeline.

## ML Pipeline Details

### Why These Methods?

| Method | Why |
|--------|-----|
| **Weighted scoring** | Interpretable baseline — stakeholders can audit and adjust weights |
| **K-Means (k=4)** | Discovers natural concept groupings without outcome labels |
| **Random Forest** | Handles small datasets, provides feature importance, works with SHAP |
| **SHAP** | Gold-standard for ML explainability — shows exactly why each decision was made |

### Baseline Weights

| Feature | Weight | Rationale |
|---------|--------|-----------|
| Demand intensity | 0.18 | Strongest predictor of customer interest |
| Revenue potential | 0.18 | Commercial viability is critical |
| Engagement depth | 0.13 | Deep sandbox use signals real interest |
| Repeatability | 0.13 | Return usage indicates sustained demand |
| Strategic fit | 0.09 | Portfolio alignment matters |
| Feasibility ease | 0.09 | Easier to deliver = faster validation |
| Segment similarity | 0.09 | Cross-segment demand = scalable |
| Positive comments | 0.05 | Sentiment signal |
| Capability requests | 0.03 | Feature demand signal |
| Objection ease | 0.03 | Fewer objections = smoother path |

### Feature Engineering

13 features across 8 dimensions:

| Feature | Source | What It Measures |
|---------|--------|-----------------|
| demand_intensity | feedback + urgency + follow-up | How much customers want this |
| engagement_depth | sessions × time / abandoned | How deeply they used the sandbox |
| feasibility_risk | complexity + implementation risk | How hard it is to deliver |
| revenue_potential | willingness to pay + expected value + budget | Commercial value |
| repeatability | return usage + follow-up + sessions | Whether interest is sustained |
| segment_similarity | entropy across segments | Cross-segment consistency |
| confidence | observation volume + completeness | How much data backs this concept |
| strategic_fit | concept metadata | Portfolio alignment |
| avg_objection_count_text | text feedback | Customer pushback level |
| capability_request_rate | text feedback | Feature demand |
| positive_comment_ratio | text feedback | Customer sentiment |
| follow_up_rate | demo signals | Interest in continuing |
| avg_pilot_interest | commercial signals | Intent to pilot |

## Data

### Raw Data (6 files)

| File | Records | Description |
|------|---------|-------------|
| product_concepts.csv | 12 | Concept metadata |
| customers.csv | 80 | Customer profiles across 5 segments |
| customer_demo_signals.csv | 257 | Demo session records |
| sandbox_usage.csv | 76 | Trial usage metrics |
| commercial_signals.csv | 230 | Commercial intent signals |
| text_feedback.csv | 230 | Qualitative feedback |

### Synthetic Data Design

Each concept has a hidden `_latent_commercial_potential` (Beta(2,2) distributed) that drives all downstream signals. High-potential concepts generate stronger feedback, more follow-ups, higher willingness to pay, and deeper engagement. The latent variable is never exposed to the models — it only generates correlated data that mimics real-world patterns.

Missing values (~4%), outliers (~3%), and noise are intentionally injected to test the cleaning pipeline.

## Limitations

- **Synthetic data** — Results are illustrative, not validated business decisions. The architecture transfers to real data.
- **Rule-based labels** — Random Forest is trained on synthetic pseudo-labels, not real outcomes. With real data, accuracy would be meaningful.
- **Small dataset** — 12 concepts is too few for robust ML. Cross-validation accuracy (~42%) is noisy. Production use needs 50+ concepts.
- **Single seed** — Results are deterministic. Different seeds produce structurally similar but different rankings.
- **Keyword sentiment** — Text analysis uses simple keyword matching, not transformer embeddings.

## Future Improvements

- Connect to real CRM/demo data
- Use sentence transformers for text feedback analysis
- Add time-series tracking for concept signal trends
- Build what-if scenario modeling ("what if we run 5 more demos?")
- Add REST API for integration with existing tools

## Author

**PrashantPatil-2005** — prashanthpatil02@gmail.com
