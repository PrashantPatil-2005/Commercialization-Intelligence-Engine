# Presentation Speaking Guide

> Slide-by-slide script for presenting CIE in a Google Meet interview. What to say, how long, what to emphasize.

---

## Total Time: 10-12 minutes

Concise, focused presentation. The interviewer wants to see your work clearly, then dig into code and ask questions.

| Phase | Time | What Happens |
|-------|------|--------------|
| Your presentation | 10-12 min | Walk through 7 slides |
| Their questions | 20-30 min | Code review, deep dives, challenges |
| Your questions | 5 min | Ask them questions |

---

## Pre-Meeting Checklist

- [ ] Presentation open in PowerPoint (not PDF)
- [ ] Screen share tested
- [ ] Microphone tested
- [ ] Git repo open in VS Code (they'll ask to see code)
- [ ] Terminal ready to run `streamlit run app/streamlit_app.py` (live demo)
- [ ] Close all unrelated tabs

---

## Slide-by-Slide Script

---

### Slide 1: Title (15 seconds)

**Say this:**

> "Hi, I'm Prashant. I built a Commercialization Intelligence Engine — it helps innovation teams decide which AI product concepts to invest in, based on customer behavior signals and explainable machine learning. Let me walk you through how it works."

---

### Slide 2: Architecture (2 minutes)

**Say this:**

> "The system has four phases.
>
> First, data generation — I generate synthetic customer signals with a dependency chain: Industry determines the Problem Area, which determines the Target User, which determines the Concept Name. Every concept is a plausible business proposition. There's a hidden latent potential — Beta(2,2) — that drives all downstream signals but is never exposed to the ML model.
>
> Second, feature engineering — I clean the raw data, handle missing values and outliers, then engineer 14 features from the signals: demand intensity, engagement depth, revenue potential, feasibility risk, and more.
>
> Third, the ML pipeline — three layers: weighted baseline scoring for interpretability, K-Means clustering for natural groupings, and Random Forest classification for the final outcome.
>
> Fourth, explainability — SHAP values decompose each recommendation into feature-level evidence, and an AI layer converts that into business-readable narratives."

**Key point to land:** The pipeline is modular. Each phase reads CSVs from disk and writes CSVs to disk — any stage can be inspected, replayed, or replaced independently.

**If they ask "why not just one model?":**
> "Each layer serves a different purpose. The baseline is interpretable. The clustering reveals groupings without outcome labels. The classification predicts the outcome. No single model does all three."

---

### Slide 3: Algorithms (2.5 minutes)

**Say this:**

> "Three algorithms, each chosen for a specific reason.
>
> Layer 1 is a weighted baseline — 10 features with manually calibrated weights. This gives an interpretable score stakeholders can audit. Demand and revenue potential get 0.18 each — the highest — because they're the strongest commercial signals. Engagement and repeatability get 0.13 each.
>
> Layer 2 is K-Means with k equals 4. This reveals natural groupings — concepts that behave similarly should be compared against each other. The clusters get labeled like 'High Demand, Low Effort' based on median splits. I use StandardScaler before clustering because the features have different scales.
>
> Layer 3 is Random Forest — 200 trees, max depth 4, balanced classes. It predicts the final outcome: MVP Build, Customer Pilot, Reusable Asset, Incubate, or Archive. I chose Random Forest because it handles small datasets, provides exact SHAP values via TreeSHAP, and is robust to hyperparameter choices.
>
> The final readiness score blends baseline at 70 percent and ML confidence at 30 percent. The 70-30 split is because the baseline is more interpretable and the ML is trained on synthetic labels."

**If they ask "why not deep learning?":**
> "12 samples is orders of magnitude too small. Neural networks need thousands. Random Forest is the right tool for this data size."

**If they ask "why not XGBoost?":**
> "Gradient boosting overfits on small data. Random Forest is more robust with fewer hyperparameters to tune."

**If they ask "why 3-fold CV?":**
> "With 12 samples, 5-fold would test on only 2-3 samples per fold. Too few for a meaningful estimate. 3-fold gives 4 test samples per fold."

---

### Slide 4: Decision Logic (1.5 minutes)

**Say this:**

> "How outcomes are decided.
>
> The readiness score combines the baseline at 70 percent weight and ML confidence at 30 percent. It ranges from 1 to 100.
>
> Confidence is different — it measures how much to trust the recommendation. It combines data confidence (based on data volume and completeness) at 55 percent with ML certainty at 45 percent.
>
> Training labels are rule-based pseudo-labels. For example: if demand is below 0.22 and revenue below 0.30, that's Archive — no market pull and no financial viability. If demand is above 0.45 and engagement above 0.35 and revenue above 0.38, that's MVP Build — strong across all dimensions.
>
> The five outcomes: MVP Build for concepts ready for engineering, Customer Pilot for those needing real-world validation, Reusable Asset for cross-segment potential, Incubate for needs-more-evidence, and Archive for deprioritize."

**Key point to land:** The rules are transparent. Anyone can audit why a concept got a specific outcome.

---

### Slide 5: Explainability (2 minutes)

**Say this:**

> "This is the part I'm most proud of. Explainability is not optional — it's what makes ML decisions trustworthy.
>
> Four layers of explainability. First, feature importance — globally, which features matter most across the portfolio. Second, SHAP values — per-concept, each feature's contribution to the prediction. Third, structured evidence — human-readable format with feature labels, values, and magnitude classifications. Fourth, AI narrative — natural language executive summary per concept.
>
> For example, here's a Customer Pilot recommendation: 'The model recommends Customer Pilot driven primarily by moderate pilot interest, along with moderate follow-up rate. However, elevated implementation risk. Identify pilot customers and define success metrics.' Every sentence maps to a specific SHAP contribution — it's not template-based.
>
> The key distinction: ML produces the evidence — scoring, clustering, classification, SHAP values. AI produces the explanation — translating SHAP into evidence, classifying magnitudes, generating narratives. They're distinct layers."

**If they ask "what's the difference between ML and AI here?":**
> "ML is quantitative — the modeling. AI is qualitative reasoning over those outputs — the translation layer that makes it readable for business stakeholders."

---

### Slide 6: Results (2 minutes)

**Say this:**

> "Key numbers: 12 concepts analyzed, 14 engineered features, 75 percent cross-validation accuracy with 3-fold stratified, 5 outcome categories.
>
> The outcome distribution: 6 Archive, 4 Customer Pilot, 1 Incubate, 1 Reusable Asset. No MVP Build in this run — the rule-based labels didn't generate enough strong signals for that outcome. Readiness scores range from 45 to 65. Confidence from 0.69 to 0.93.
>
> Everything is served through a 5-page Streamlit dashboard. Overview shows portfolio health. Portfolio is the investment decisions table. Explorer is the executive decision report — per-concept drill-down with SHAP evidence and waterfall charts. Analytics shows clusters and correlations. Model shows pipeline transparency.
>
> I want to be honest about limitations. The dataset is synthetic — architecture transfers to real data, but results don't validate business value. Training labels are circular — RF trained on rules from the same features it predicts. 12 concepts is too few — production needs 50 or more. For production, I'd prioritize real CRM integration first, then human-labeled training data, then time-series features."

**Key point to land:** Be honest about numbers. 75% accuracy is against synthetic labels, not real outcomes. Own the limitations.

**If they ask "is 75% good?":**
> "It's measured against synthetic labels, so it's not a meaningful performance metric. The number demonstrates the pipeline runs end-to-end. Real accuracy requires human annotations."

---

### Slide 7: Thank You (30 seconds)

**Say this:**

> "The biggest lesson from this project was that explainability isn't optional. It's what makes ML decisions trustworthy and auditable. The SHAP evidence and narrative layer are what turn a model output into a business recommendation.
>
> That's the presentation. I'm happy to walk through the code, answer questions, or do a live demo. Thank you."

**Then stop talking.** Let them ask questions.

---

## Timing Summary

| Slide | Time | Cumulative |
|-------|------|------------|
| 1. Title | 0:15 | 0:15 |
| 2. Architecture | 2:00 | 2:15 |
| 3. Algorithms | 2:30 | 4:45 |
| 4. Decision Logic | 1:30 | 6:15 |
| 5. Explainability | 2:00 | 8:15 |
| 6. Results | 2:00 | 10:15 |
| 7. Thank You | 0:30 | 10:45 |

**Target: 10-12 minutes.** Leaves room for questions.

---

## Post-Presentation: What Happens Next

1. **Ask to see the code** — Have VS Code open with `models/decision_engine.py` and `models/insight_layer.py`
2. **Ask about specific files** — `data/prepare_features.py` for features, `models/insight_layer.py` for SHAP
3. **Ask for live demo** — `streamlit run app/streamlit_app.py`
4. **Challenge your choices** — "Why not X?" questions. See `docs/INTERVIEW_PREP.md`
5. **Ask about weaknesses** — Be honest, be specific, have a fix for each

---

## Google Meet Tips

- **Screen share the presentation window**, not your entire screen
- **Keep eye contact with the camera** when speaking
- **Mute when they're talking**
- **Use "pause" instead of "um"** — 2 seconds of silence sounds confident
- **If you don't know something, say so** — "I haven't explored that, but here's how I'd approach it"
- **If they interrupt mid-slide, answer then ask "Should I continue?"**

---

## Emergency: Running Out of Time

If you're at 8 minutes and still on slide 5, skip to slide 6 (Results + Limitations) — most important after architecture. The dashboard can be shown live.
