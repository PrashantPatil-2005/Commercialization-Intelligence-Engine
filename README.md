# Commercialization Intelligence Engine

> NTT DATA Internship Case Study — AI/ML Decision Engine for Product Portfolio Optimization

An ML prototype that helps innovation teams decide which early-stage AI product concepts to build, pilot, incubate, or archive based on customer behavior signals and explainable machine learning.

---

## Problem

Innovation teams manage 10-20 early-stage AI concepts with limited engineering resources. Investment decisions are made in meetings based on subjective judgment — no data-driven, auditable process exists.

CIE solves this by converting customer signals into evidence-based recommendations with five commercial outcomes:

| Outcome | Meaning | Next Step |
|---------|---------|-----------|
| **MVP Build** | Strong signal across all dimensions | Build a focused prototype |
| **Customer Pilot** | Clear customer pull, ready to test | Find pilot customers, define metrics |
| **Reusable Asset** | Repeatable demand across segments | Evaluate platform packaging |
| **Incubate** | Promising but needs more evidence | Run more demos, sharpen positioning |
| **Archive** | Weak signal, poor fit, or high effort | Document learnings, reallocate |

---

## Pipeline

```
Raw data → Clean & engineer features → ML models → Explain decisions → Dashboard
```

| Phase | Script | What It Does |
|-------|--------|-------------|
| 1. Data Generation | `data/generate_mock_data.py` | 12 concepts, 80 customers, 6 CSVs with latent potential driving correlated signals |
| 2. Feature Engineering | `data/prepare_features.py` | Clean, validate, engineer 14 concept-level features |
| 3. ML Decision Engine | `models/decision_engine.py` | Weighted baseline + K-Means clustering + Random Forest classifier |
| 4. AI Insight Layer | `models/insight_layer.py` | SHAP values → structured evidence → human-readable narratives |
| 5. Dashboard | `app/streamlit_app.py` | 5-page Streamlit app with dark/light theme |

Each phase reads CSVs from disk, writes CSVs to disk, and stops. **Any stage can be inspected, replayed, or replaced independently.**

---

## ML Methods

| Method | Purpose | Why This Choice |
|--------|---------|-----------------|
| Weighted Scoring | Interpretable baseline (readiness 1-100) | Stakeholders can audit the weights |
| K-Means (k=4) | Discover natural groupings | Unsupervised — no outcome labels needed |
| Random Forest (200 trees) | Predict commercial outcome | Handles small data, SHAP-compatible, robust |
| SHAP | Per-feature explanation | Gold-standard for ML explainability |

**Cross-validation:** 3-fold stratified, ~75% accuracy (against synthetic labels).

**Readiness score:** `baseline × 0.70 + ml_confidence × 0.30` (1-100 scale).

---

## Results

| Metric | Value |
|--------|-------|
| Concepts analyzed | 12 |
| Engineered features | 14 |
| CV accuracy (3-fold) | ~75% |
| Outcome categories | 5 |
| Dashboard pages | 5 |

**Outcome distribution:** 6 Archive, 4 Customer Pilot, 1 Incubate, 1 Reusable Asset.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app/streamlit_app.py

# Or run notebook
jupyter notebook notebooks/commercialization_results.ipynb

# Or regenerate data from scratch
python data/generate_mock_data.py
python data/prepare_features.py
python models/decision_engine.py
python models/insight_layer.py
```

Pre-computed outputs exist in `data/processed/`, so the dashboard works without re-running the pipeline.

---

## Project Structure

```
├── app/                            # Streamlit dashboard
│   ├── streamlit_app.py            # Entry point, sidebar, routing
│   ├── theme.py                    # Light/Dark theme (single source of truth)
│   ├── styles.py                   # CSS generation, HTML helpers
│   ├── components.py               # Reusable UI components
│   ├── charts.py                   # Matplotlib chart builders
│   └── pages/                      # 5 page modules
│       ├── overview.py             # Portfolio health + KPIs
│       ├── portfolio.py            # Investment decisions table
│       ├── explorer.py             # Per-concept executive report
│       ├── analytics.py            # Clusters + correlations
│       └── model.py                # Pipeline transparency
│
├── data/
│   ├── generate_mock_data.py       # Phase 1: synthetic data generation
│   ├── prepare_features.py         # Phase 2: cleaning + feature engineering
│   ├── raw/                        # 7 CSV files (generated)
│   └── processed/                  # 16 files (features, decisions, insights)
│
├── models/
│   ├── decision_engine.py          # Phase 3: baseline + K-Means + Random Forest
│   └── insight_layer.py            # Phase 4: SHAP + evidence + narratives
│
├── notebooks/
│   └── commercialization_results.ipynb
│
├── docs/
│   ├── techinical_documentation.md       # tech documentataion
│ 
│
├── TECHNICAL_DOCUMENTATION.md      # Full technical documentation
├── README.md                       # This file
└── requirements.txt
```

---

## Data Design

Each concept has a hidden `_latent_commercial_potential` (Beta(2,2)) that drives all downstream signals. High-potential concepts generate stronger feedback, more follow-ups, higher willingness to pay, and deeper engagement.

**Dependency chain:** Industry → Problem Area → Target User → Concept Name (ensures every concept is a plausible business proposition).

**Data quality:** Missing values (~4%) and outliers (~3%) intentionally injected to test robustness.

---



---

## Author

**Prashant Patil** — prashanthpatil02@gmail.com

NTT DATA Internship Case Study
