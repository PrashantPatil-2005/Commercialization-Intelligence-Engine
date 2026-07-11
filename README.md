# Commercialization Intelligence Engine

ML prototype that helps an innovation team decide which early-stage AI product concepts to build, pilot, incubate, or archive based on customer behavior signals.

## Problem

An innovation team has 12 AI product concepts shown to customers through demos, sandbox trials, and discovery sessions. They need to decide which deserve investment. The system recommends one of five outcomes per concept:

| Outcome | Meaning |
|---------|---------|
| MVP Build | Strong signal — build a focused prototype |
| Customer Pilot | Ready to test with real customers |
| Reusable Asset | Repeatable demand across segments |
| Incubate | Promising but needs more evidence |
| Archive | Weak signal — deprioritize |

## Pipeline

```
Raw data → Clean & engineer features → ML models → Explain decisions → Dashboard
```

1. **Data generation** — 12 concepts, 80 customers, 5 data areas with latent potential driving correlated signals
2. **Feature engineering** — 13 concept-level features (demand intensity, engagement depth, revenue potential, etc.)
3. **ML decision engine** — Weighted baseline score + K-Means clustering + Random Forest classifier
4. **Explainability** — SHAP values show which features drove each recommendation, with human-readable narratives
5. **Dashboard** — Streamlit app with portfolio ranking, charts, and per-concept drill-down

## Results

| Metric | Value |
|--------|-------|
| Concepts analyzed | 12 |
| Recommended for advancement | Customer Pilot + Reusable Asset |
| Archive candidates | 2-5 (varies by run) |
| CV accuracy | ~75% (3-fold, synthetic labels) |

## Run

```bash
pip install -r requirements.txt

# Dashboard
streamlit run app/streamlit_app.py

# Or notebook
jupyter notebook notebooks/commercialization_results.ipynb

# Or regenerate data
python data/generate_mock_data.py
python data/prepare_features.py
```

Pre-computed outputs exist in `data/processed/`, so the dashboard works without re-running the pipeline.

## Structure

```
├── app/                    # Streamlit dashboard
│   ├── streamlit_app.py
│   ├── styles.py, components.py, charts.py
│   └── pages/              # 5 page modules
├── data/
│   ├── generate_mock_data.py   # Phase 1: synthetic data
│   ├── prepare_features.py     # Phase 2: cleaning + features
│   ├── raw/                    # 6 CSV files
│   └── processed/              # Features, decisions, insights
├── models/
│   ├── decision_engine.py      # Phase 3: baseline + K-Means + RF
│   └── insight_layer.py        # Phase 4: SHAP + narratives
├── notebooks/
│   └── commercialization_results.ipynb
└── requirements.txt
```

## ML Methods

| Method | Why |
|--------|-----|
| Weighted scoring | Interpretable baseline stakeholders can audit |
| K-Means (k=4) | Discovers natural groupings without outcome labels |
| Random Forest | Handles small data, provides feature importance, SHAP-compatible |
| SHAP | Gold-standard for ML explainability |

## Data

Each concept has a hidden `_latent_commercial_potential` (Beta(2,2)) that drives all downstream signals. High-potential concepts generate stronger feedback, more follow-ups, higher willingness to pay, and deeper engagement. Missing values (~4%) and outliers (~3%) are intentionally injected.

## Limitations

- Synthetic data — architecture transfers to real data
- Rule-based labels — RF trained on pseudo-labels; real outcomes needed for meaningful accuracy
- 12 concepts — too few for robust ML; production needs 50+
- Keyword sentiment — not transformer embeddings

## Author

PrashantPatil-2005 — prashanthpatil02@gmail.com
