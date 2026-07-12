# Commercialization Intelligence Engine — Technical Documentation

> NTT DATA Internship Case Study: AI/ML Decision Engine for Product Portfolio Optimization

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Phase 1: Synthetic Data Generation](#4-phase-1-synthetic-data-generation)
5. [Phase 2: Feature Engineering](#5-phase-2-feature-engineering)
6. [Phase 3: ML Decision Engine](#6-phase-3-ml-decision-engine)
7. [Phase 4: AI Insight Layer](#7-phase-4-ai-insight-layer)
8. [Dashboard](#8-dashboard)
9. [Data Flow & Persistence](#9-data-flow--persistence)
10. [Results & Evaluation](#10-results--evaluation)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [Appendix: File Reference](#12-appendix-file-reference)

---

## 1. Executive Summary

The Commercialization Intelligence Engine (CIE) is a prototype AI/ML system that converts customer behavior signals into evidence-based commercialization recommendations for early-stage AI product concepts.

**Core capability:** For each of 12 product concepts, the system recommends one of five outcomes — MVP Build, Customer Pilot, Reusable Asset, Incubate, or Archive — supported by explainable ML and human-readable narratives.

**Key metrics:**
| Metric | Value |
|--------|-------|
| Concepts analyzed | 12 |
| Engineered features | 14 |
| ML algorithms | 3 (Weighted Scoring, K-Means, Random Forest) |
| CV accuracy (3-fold) | ~75% |
| Commercial outcomes | 5 |
| Dashboard pages | 5 |
| Explainability method | SHAP + AI narrative |

---

## 2. Problem Statement

Innovation teams manage portfolios of 10-20 early-stage AI concepts. Investment decisions are currently made in meetings based on subjective judgment — there is no data-driven, auditable process.

**Business requirements (from assignment brief):**
- Mock data generation simulating customer signals
- Feature engineering capturing commercialization dimensions
- ML pipeline combining multiple techniques
- Explainable outputs (not black-box predictions)
- AI insight layer translating ML outputs into business language
- Visual dashboard for stakeholder consumption

**Design constraints:**
- Prototype must run on synthetic data (real commercialization data is proprietary)
- 12 concepts — too few for deep learning; traditional ML is appropriate
- Explainability is mandatory, not optional
- Must demonstrate end-to-end pipeline from raw data to dashboard

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMMERCIALIZATION INTELLIGENCE ENGINE         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  PHASE 1     │    │  PHASE 2     │    │  PHASE 3     │      │
│  │  Data Gen    │───▶│  Features    │───▶│  ML Engine   │      │
│  │              │    │              │    │              │      │
│  │  6 CSVs      │    │  14 features │    │  3 layers    │      │
│  │  80 customers│    │  Concept-level│    │  5 outcomes  │      │
│  │  12 concepts │    │  Aggregated  │    │  Readiness   │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│                                          ┌──────────────┐      │
│                                          │  PHASE 4     │      │
│                                          │  Insight     │      │
│                                          │              │      │
│                                          │  SHAP values │      │
│                                          │  Evidence    │      │
│                                          │  Narratives  │      │
│                                          └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│                                          ┌──────────────┐      │
│                                          │  DASHBOARD   │      │
│                                          │  5 pages     │      │
│                                          │  Streamlit   │      │
│                                          └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Independence principle:** Each phase reads from disk, writes to disk, and stops. Intermediate CSVs are the contracts between phases. Any stage can be inspected, replayed, or replaced without touching others.

---

## 4. Phase 1: Synthetic Data Generation

**Script:** `data/generate_mock_data.py`

### 4.1 Dependency Chain Model

The data generation follows a top-down dependency chain to ensure every concept is a plausible business proposition:

```
Industry → Problem Area → Target User → Concept Name
```

**Example chain:**
```
Healthcare → Revenue Cycle Optimization → Hospital CFO → AI Claims Assistant
```

**Implementation:**
- `INDUSTRY_PROBLEM_AREAS`: 7 industries × 2-3 problem areas each
- `PROBLEM_TARGET_USERS`: maps problem area → target user
- `PROBLEM_CONCEPT_NAMES`: maps problem area → 3-4 concept names
- `PROBLEM_COMPLEXITY`: maps problem area → delivery complexity (1-5)
- `PROBLEM_STRATEGIC_FIT`: maps problem area → base strategic fit range

### 4.2 Latent Variable

Each concept has a hidden `_latent_commercial_potential` drawn from Beta(2,2):

```python
_latent = np.random.beta(2, 2)  # symmetric, mean=0.5, bounded [0,1]
```

This variable drives ALL downstream signals but is never exposed to the ML model. It exists only during generation to create realistic correlations.

**Propagation:** High-latent concepts generate:
- Higher feedback scores (+β × latent)
- More follow-up requests
- Higher willingness to pay
- Deeper sandbox engagement
- More positive text feedback

### 4.3 Generated Tables

| Table | Rows | Columns | Description |
|-------|------|---------|-------------|
| `product_concepts.csv` | 12 | 7 | Concept metadata (name, industry, problem area, target user, complexity, strategic fit) |
| `customers.csv` | 80 | 4 | Customer profiles (industry, segment, company size) |
| `customer_demo_signals.csv` | ~240 | 8 | Demo session signals per customer-concept pair |
| `sandbox_usage.csv` | ~240 | 7 | Sandbox trial usage metrics |
| `commercial_signals.csv` | ~240 | 6 | Commercial intent (willingness_to_pay, urgency, expected_value, budget_aligned) |
| `text_feedback.csv` | ~240 | 5 | Qualitative feedback (comments, objections, capabilities) |

### 4.4 Data Quality Injection

| Issue | Rate | Method |
|-------|------|--------|
| Missing values | ~4% | Random NaN injection |
| Outliers | ~3% | Clipped at 1st/99th percentile |

### 4.5 Industry-Specific Pain Points

Each of 7 industries has 3 unique pain points, ensuring realistic signal variation:

```python
INDUSTRY_PAIN_POINTS = {
    "Healthcare": ["claim_denials", "patient_wait_times", "staff_burnout"],
    "Finance": ["manual_reconciliation", "fraud_detection_gaps", "compliance_burden"],
    # ... 5 more industries
}
```

### 4.6 Feedback-Dependent Capabilities

Requested capabilities correlate with feedback score:
- High feedback (≥0.7): advanced features (AI automation, predictive analytics)
- Mid feedback (0.4-0.7): moderate features (reporting, dashboards)
- Low feedback (<0.4): basic features (data entry, CSV export)

---

## 5. Phase 2: Feature Engineering

**Script:** `data/prepare_features.py`

### 5.1 Cleaning Pipeline

1. **Load** 5 raw CSVs
2. **Impute** missing numeric values with column median
3. **Clip** outliers at 1st/99th percentile
4. **Validate** schema (required columns present)
5. **Export** cleaned CSVs to `data/processed/`

### 5.2 Feature Engineering

From the cleaned data, 14 concept-level features are engineered:

| # | Feature | Formula | Business Meaning |
|---|---------|---------|------------------|
| 1 | `demand_intensity` | feedback×0.45 + urgency×0.35 + follow_up×0.20 | Overall market pull |
| 2 | `engagement_depth_norm` | (sessions × time_spent × (1 - abandoned_pct)) normalized | How deeply customers use sandbox |
| 3 | `feasibility_risk` | delivery_complexity / 5.0 | Implementation difficulty |
| 4 | `repeatability` | std(concept_scores) / mean(concept_scores) | Consistency across customers |
| 5 | `segment_similarity` | 1 - coefficient of variation across segments | Cross-segment demand consistency |
| 6 | `revenue_potential` | wtp×0.40 + expected_value×0.35 + budget_aligned×0.25 | Financial viability |
| 7 | `strategic_fit` | From metadata (0.30-0.95 range) | Organizational alignment |
| 8 | `confidence` | data_volume×0.65 + completeness×0.35 | Evidence quality (not model certainty) |
| 9 | `follow_up_rate` | follow_up_requested / total_demos | Customer intent signal |
| 10 | `avg_pilot_interest` | mean(pilot_interest_score) | Direct pilot readiness |
| 11 | `avg_objection_count_text` | mean(objections_count) | Sales friction |
| 12 | `capability_request_rate` | capability_mentions / total_feedback | Feature demand signal |
| 13 | `positive_comment_ratio` | positive_comments / total_comments | Sentiment direction |
| 14 | `feasibility_ease` | 1 - feasibility_risk | Inverted risk for positive scoring |

### 5.3 Output Files

| File | Description |
|------|-------------|
| `concept_features.csv` | **Primary ML input** — 12 rows × 14 features |
| `concept_features_full.csv` | All columns including metadata |
| `interaction_features.csv` | Customer-concept grain (before aggregation) |
| `*_clean.csv` (5 files) | Cleaned versions of raw tables |
| `validation_report.json` | Schema validation results |

---

## 6. Phase 3: ML Decision Engine

**Script:** `models/decision_engine.py`

### 6.1 Three-Layer Architecture

```
Layer 1: Weighted Baseline Scoring
    ↓ readiness score (1-100)
Layer 2: K-Means Clustering (k=4)
    ↓ cluster labels
Layer 3: Random Forest Classifier (200 trees)
    ↓ predicted outcome + probabilities
    ↓
Blend: readiness = baseline × 0.70 + ml_confidence × 30
```

### 6.2 Layer 1: Weighted Baseline

**Purpose:** Interpretable score that stakeholders can audit.

```python
BASELINE_WEIGHTS = {
    "demand_intensity": 0.18,      # highest — strongest signal
    "revenue_potential": 0.18,     # highest — financial viability
    "engagement_depth_norm": 0.13, # medium — usage signal
    "repeatability": 0.13,         # medium — consistency
    "segment_similarity": 0.09,    # lower — cross-segment
    "strategic_fit": 0.09,         # lower — organizational
    "feasibility_ease": 0.09,      # lower — inverted risk
    "positive_comment_ratio": 0.05,# minimal — sentiment
    "capability_request_rate": 0.03,# minimal — feature demand
    "objection_ease": 0.03,        # minimal — inverted objections
}
```

**Formula:**
```python
raw = Σ(weight_i × feature_i)
readiness = (raw × 100).clip(1, 100)
```

### 6.3 Layer 2: K-Means Clustering

**Purpose:** Discover natural groupings without outcome labels.

**Configuration:**
- `n_clusters = 4`
- `StandardScaler` applied before clustering
- Input: 5 behavioral features (demand, engagement, repeatability, revenue, risk)

**Cluster labeling:** Based on median split of mean demand and risk:
| Demand | Risk | Label |
|--------|------|-------|
| High | Low | "High Demand / Low Effort" |
| High | High | "High Demand / High Effort" |
| Low | Low | "Low Demand / Low Effort" |
| Low | High | "Low Demand / High Effort" |

### 6.4 Layer 3: Random Forest Classifier

**Purpose:** Predict the final commercial outcome.

**Configuration:**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=4,
    class_weight="balanced",
    random_state=42
)
```

**Why Random Forest:**
- Handles small datasets (12 samples)
- Provides exact SHAP values via TreeSHAP
- Robust to hyperparameter choices
- Built-in feature importance

### 6.5 Training Labels

Rule-based pseudo-labels (no human annotations available):

| Rule | Outcome |
|------|---------|
| demand < 0.22 AND revenue < 0.30 | Archive |
| risk > 0.58 AND demand < 0.40 | Archive |
| conf < 0.62 AND demand < 0.25 | Archive |
| demand ≥ 0.45 AND repeat ≥ 0.30 AND segments ≥ 0.90 AND revenue ≥ 0.40 | Reusable Asset |
| demand ≥ 0.42 AND follow_up ≥ 0.35 AND pilot ≥ 0.35 AND risk < 0.55 | Customer Pilot |
| demand ≥ 0.45 AND engagement ≥ 0.35 AND revenue ≥ 0.38 AND fit ≥ 0.50 | MVP Build |
| default | Incubate |

### 6.6 Confidence Score

```python
confidence = data_confidence × 0.55 + ml_certainty × 0.45
```

Where:
- `data_confidence = data_volume × 0.65 + completeness × 0.35`
- `ml_certainty = max(RF class probabilities)`

### 6.7 Cross-Validation

```python
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
```

3-fold chosen over 5-fold because with 12 samples, 5-fold would test on only 2-3 samples per fold — too few for meaningful accuracy.

### 6.8 Output Files

| File | Description |
|------|-------------|
| `concept_recommendations.csv` | Ranked portfolio recommendations |
| `concept_decisions_full.csv` | Full decision DataFrame (all computed columns) |
| `feature_importance.csv` | Random Forest feature importances |
| `cluster_summary.csv` | K-Means cluster statistics |
| `model_report.json` | CV scores, outcome distribution, config |

---

## 7. Phase 4: AI Insight Layer

**Script:** `models/insight_layer.py`

### 7.1 Pipeline

```
Random Forest Prediction
    ↓
SHAP TreeExplainer
    ↓
Top Positive Features (SHAP > 0)
Top Negative Features (SHAP < 0)
    ↓
Structured Evidence Dict
    ↓
AI Executive Summary (generated from evidence only)
    ↓
Dashboard
```

**Critical design principle:** The narrative is NEVER template-based. Every sentence is derived from SHAP values. The LLM never invents reasons not supported by the model.

### 7.2 SHAP Computation

```python
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)
```

For each concept, extract the SHAP vector for its predicted class:
```python
class_index = list(clf.classes_).index(predicted_class)
shap_vector = shap_values[row_index, :, class_index]  # multiclass case
```

### 7.3 Structured Evidence Builder

`build_shap_evidence()` converts raw SHAP into human-readable format:

```python
{
    "concept_name": "AI Claims Assistant",
    "predicted_outcome": "Customer Pilot",
    "supporting": [
        {
            "feature": "demand_intensity",
            "label": "demand intensity",
            "value": 0.42,
            "shap": 0.085,
            "magnitude": "strong",
            "description": "0.42"
        },
        # ...
    ],
    "counter": [
        {
            "feature": "feasibility_risk",
            "label": "implementation risk",
            "value": 0.52,
            "shap": -0.063,
            "magnitude": "elevated",
            "description": "0.52"
        },
        # ...
    ],
    "top_features": [...]  # for waterfall chart
}
```

### 7.4 Magnitude Classification

| Feature Type | Threshold | Label |
|-------------|-----------|-------|
| Positive (demand, engagement, etc.) | SHAP ≥ 0.08 | strong |
| Positive | SHAP ≥ 0.03 | moderate |
| Positive | SHAP < 0.03 | mild |
| Risk (feasibility_risk) | value ≥ 0.55 | elevated |
| Risk | value ≥ 0.35 | moderate |
| Risk | value < 0.35 | manageable |
| Objections | value ≥ 3.0 | high |
| Objections | value ≥ 1.5 | moderate |
| Objections | value < 1.5 | low |

### 7.5 AI Narrative Generation

`generate_executive_summary_from_shap()` produces per-concept narratives:

**Input:** Structured evidence dict
**Output:** 3-4 sentence executive summary

**Example output:**
> "The model recommends Customer Pilot driven primarily by moderate pilot interest (0.36), along with moderate follow-up rate (47%). However, elevated implementation risk (0.52). Identify pilot customers and define success metrics."

**Structure:**
1. Recommendation + top supporting feature
2. Additional supporting evidence
3. Key risk/concern (if any)
4. Next step action

### 7.6 Feature Labels

Human-readable labels for all 13 features:

```python
FEATURE_LABELS = {
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
```

### 7.7 Output Files

| File | Description |
|------|-------------|
| `concept_insights.csv` | SHAP evidence, AI narratives, all shap_* columns |
| `executive_summary.txt` | Portfolio-level executive summary |

---

## 8. Dashboard

**Entry point:** `app/streamlit_app.py`

### 8.1 Theme System

**Single source of truth:** `app/theme.py` defines LIGHT and DARK dictionaries (55 keys each).

| Component | File |
|-----------|------|
| Theme constants | `app/theme.py` |
| CSS generation | `app/styles.py` |
| Components | `app/components.py` |
| Charts | `app/charts.py` |

**Toggle:** Single icon button (☀️/🌙) in sidebar, persists via `st.session_state["theme_mode"]`.

**Color palette:**
- Dark mode: neutral grays (#111111/#1a1a1a)
- Light mode: warm whites (#f5f5f5)
- Outcome colors: muted professional (not vibrant)

### 8.2 Pages

| Page | Module | Business Question |
|------|--------|-------------------|
| Overview | `app/pages/overview.py` | "How healthy is my portfolio?" |
| Portfolio | `app/pages/portfolio.py` | "Which concepts should I invest in?" |
| Explorer | `app/pages/explorer.py` | "Why this recommendation for this concept?" |
| Analytics | `app/pages/analytics.py` | "What patterns exist across the portfolio?" |
| Model | `app/pages/model.py` | "How does the system work?" |

### 8.3 Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `kpi_card()` | `app/styles.py` | Metric display boxes |
| `decision_card()` | `app/styles.py` | Outcome recommendation cards |
| `progress_bar()` | `app/styles.py` | ARIA-accessible progress bars |
| `plot_shap_waterfall()` | `app/charts.py` | SHAP waterfall visualization |
| `plot_feature_importance()` | `app/charts.py` | Global feature importance |
| `plot_cluster_scatter()` | `app/charts.py` | Cluster visualization |
| `render_shap_evidence_table()` | `app/components.py` | SHAP evidence display |

### 8.4 Accessibility

- WCAG AA contrast ratios for all text
- `:focus-visible` outlines for keyboard navigation
- ARIA labels on progress bars
- Font scale: 15px base (readable at distance)

---

## 9. Data Flow & Persistence

### 9.1 Phase Handoff Map

```
Phase 1: generate_mock_data.py
    NO INPUT
        ↓
    data/raw/*.csv (7 files)
        ↓
Phase 2: prepare_features.py
    READS: 5 CSVs from data/raw/
        ↓
    data/processed/concept_features.csv  ← PRIMARY HANDOFF
        ↓
Phase 3: decision_engine.py
    READS: data/processed/concept_features.csv
        ↓
    data/processed/concept_recommendations.csv
    data/processed/model_report.json
        ↓
Phase 4: insight_layer.py
    READS: data/processed/concept_features.csv (via Phase 3 in-memory)
        ↓
    data/processed/concept_insights.csv  ← FINAL OUTPUT
    data/processed/executive_summary.txt
```

### 9.2 Critical Handoff File

**`data/processed/concept_features.csv`** is the contract between all phases:
- Output of Phase 2
- Input to Phase 3
- Input to Phase 4 (via Phase 3)

This file can be replaced with real data without touching any pipeline code.

### 9.3 Complete File Inventory

| Directory | Source Files | Data Files |
|-----------|-------------|------------|
| `app/` | 6 Python modules | — |
| `app/pages/` | 5 page modules | — |
| `data/` | 2 scripts | 7 raw + 16 processed |
| `models/` | 2 Python modules | — |
| `notebooks/` | 1 notebook | — |
| `docs/` | 2 docs + 1 PPT | — |
| Root | 3 files | — |

---

## 10. Results & Evaluation

### 10.1 Outcome Distribution

| Outcome | Count | Action |
|---------|-------|--------|
| Archive | 6 | Document learnings, reallocate resources |
| Customer Pilot | 4 | Find pilot customers, define success metrics |
| Incubate | 1 | Run more demos, sharpen positioning |
| Reusable Asset | 1 | Evaluate platform packaging |
| MVP Build | 0 | (None in this run — rule thresholds not met) |

### 10.2 Readiness Scores

- Range: 45 – 65 (on 0-100 scale)
- Mean: ~55
- Baseline contributes 70%, ML confidence contributes 30%

### 10.3 Confidence Scores

- Range: 0.69 – 0.93
- Higher confidence = more data available + higher ML certainty

### 10.4 Cross-Validation

- 3-fold stratified: ~75% mean accuracy
- Measured against synthetic labels (not real outcomes)
- Demonstrates pipeline runs end-to-end

### 10.5 Feature Importance (Random Forest)

| Feature | Importance |
|---------|-----------|
| feasibility_risk | ~13.3% |
| demand_intensity | ~12.1% |
| revenue_potential | ~11.8% |
| engagement_depth_norm | ~10.5% |
| strategic_fit | ~9.7% |

---

## 11. Limitations & Future Work

### 11.1 Honest Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Synthetic data | Results don't validate business value | Architecture transfers to real data |
| Circular labels | RF trained on rules from same features | Needs human annotations |
| 12 concepts | Too few for robust ML | Production needs 50+ |
| Keyword sentiment | Not transformer embeddings | Sufficient for prototype |
| No time dimension | Snapshot, not trend | Add time-series in production |
| No real CRM integration | Manual data entry | Priority for production |

### 11.2 Production Roadmap

1. **Real CRM integration** — Connect to Salesforce/HubSpot for actual customer data
2. **Human-labeled training data** — Replace rule-based labels with expert annotations
3. **Time-series features** — Add trend signals (improving/declining demand)
4. **Transformer embeddings** — Replace keyword sentiment with BERT/GPT embeddings
5. **A/B testing framework** — Measure if ML recommendations outperform human decisions
6. **Multi-tenant deployment** — Support multiple innovation teams

---

## 12. Appendix: File Reference

### 12.1 Core Pipeline

| File | Purpose | Key Functions |
|------|---------|---------------|
| `data/generate_mock_data.py` | Phase 1: Synthetic data | `generate_concepts()`, `generate_customer_signals()` |
| `data/prepare_features.py` | Phase 2: Feature engineering | `load_raw_data()`, `engineer_concept_features()` |
| `models/decision_engine.py` | Phase 3: ML pipeline | `compute_baseline_readiness()`, `run_kmeans_clustering()`, `run_decision_engine()` |
| `models/insight_layer.py` | Phase 4: Explainability | `compute_shap_values()`, `build_shap_evidence()`, `generate_executive_summary_from_shap()` |

### 12.2 Dashboard

| File | Purpose |
|------|---------|
| `app/streamlit_app.py` | Entry point, sidebar, page routing |
| `app/theme.py` | LIGHT/DARK theme constants, typography |
| `app/styles.py` | CSS generation, HTML helper functions |
| `app/components.py` | Reusable UI components |
| `app/charts.py` | Matplotlib chart builders |
| `app/pages/overview.py` | Portfolio health dashboard |
| `app/pages/portfolio.py` | Investment decisions table |
| `app/pages/explorer.py` | Per-concept executive report |
| `app/pages/analytics.py` | Pattern analysis |
| `app/pages/model.py` | Pipeline transparency |

### 12.3 Documentation

| File | Purpose |
|------|---------|
| `README.md` | Project overview and quickstart |
| `TECHNICAL_DOCUMENTATION.md` | This document |
| `docs/PPT_SPEAKING_GUIDE.md` | Presentation scripts |
| `docs/CIE_Presentation.pptx` | Interview presentation |

### 12.4 Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
shap>=0.44.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.13.0
Faker>=22.0.0
jupyter>=1.0.0
ipykernel>=6.25.0
```

---

*Prashant Patil — prashanthpatil02@gmail.com*
