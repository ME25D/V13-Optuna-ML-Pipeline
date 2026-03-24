# Zindi Financial Health Prediction Challenge

> **Platform:** [Zindi / data.org](https://zindi.africa/competitions/financial-health-prediction-challenge)  
> **Task:** Multi-class classification — SME Financial Health Index (Low / Medium / High)  
> **Metric:** Weighted F1 (WF1)  
> **Final OOF WF1:** `0.8633` (V13 Optuna Blend) | **High Recall:** `0.7149` (α-swept)

---

## Problem

Predict the financial health category of small and medium-sized enterprises (SMEs) from business and demographic features.

| Class | Count | Ratio |
|---|---|---|
| Low | 6,280 | 65.3% |
| Medium | 2,868 | 29.8% |
| High | 470 | **4.9%** ← imbalanced |

Train: 9,618 samples × 37 features — Test: 2,405 samples

The high class imbalance (High = 4.9%) makes recall on the "High" class a critical secondary metric alongside WF1.

---

## Solution Architecture — V13 Deep Optuna (4-Phase Decoupled)

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Train.csv / Test.csv                                │
│  ZindiFeatureEngineer  →  16 domain features                │
│  ZindiClusterEngineer  →  K-Means(k=8,12) + K-Fold TE       │
│                           65 total features per fold        │
└──────────────────────────┬──────────────────────────────────┘
                           │  5-fold OOF (frozen split)
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       LightGBM         XGBoost         CatBoost
    100 HP trials     100 HP trials    50 HP trials
    WF1=0.86472       WF1=0.86083      WF1=0.84849
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                  FAZ 2: Ensemble Blend
                  50 trial simplex search
                  CB=0.398 LGBM=0.532 XGB=0.071
                  OOF WF1=0.8633 | HRec=0.7043
                           │
                           ▼
                  FAZ 3: Alpha Sweep
                  α=1.08 → HRec=0.7149 ✓
                           │
                           ▼
                  FAZ 4: Seed Stability
                  WF1=0.8614 ± 0.0006 ✓ stable
```

---

## Feature Engineering

### ZindiFeatureEngineer
16 leakage-free domain features including:
- `net_profit_margin`, `expense_coverage`, `turnover_per_month` (log1p)
- `financial_access_score`, `insurance_score`, `cashflow_risk_score`
- `digital_maturity_score`, `vulnerability_index`, `formality_score`
- `risk_coverage_mismatch`, `age_stability_interact`

### ZindiClusterEngineer
K-Means clustering (k=8 and k=12) on financial features → K-Fold Target Encoding per cluster:
- Fit only on training folds (inner 3-fold CV + Laplace smoothing `α=10`)
- Produces `cluster_id`, `cluster_dist`, `cluster_te_low/mid/high` per k
- **10 cluster features** added leakage-free per fold

---

## Models & Hyperparameters

### LightGBM (FAZ 1A — 100 trials, 0 pruned)
```
Best trial   : #95 (band HRec max selected over #97 abs best WF1)
num_leaves   : 187      learning_rate : 0.01257
min_child    : 11       subsample     : 0.601
colsample    : 0.740    reg_lambda    : 0.571
class_weight : balanced (fixed)
OOF WF1      : 0.86472
```

### XGBoost (FAZ 1B — 100 trials, 0 pruned)
```
Best trial   : #62 (band HRec max selected over #97 abs best WF1)
max_depth    : 8        learning_rate : 0.02599
min_child_w  : 5        subsample     : 0.713
colsample    : 0.797    reg_lambda    : 3.464
sample_weight: balanced (fixed)
OOF WF1      : 0.86083
```

### CatBoost (FAZ 1C — 50 trials, 0 pruned)
```
Best trial   : #27 (band HRec max — HRec=0.817 vs abs best HRec=0.796)
depth        : 7        learning_rate : 0.02142
l2_leaf_reg  : 5.644    iterations    : 3000 (ES=150)
class_weights: {0:1, 1:2, 2:12} (fixed)
OOF WF1      : 0.84849
⚠ Fold 1 bestIteration=21 (instability persists)
```

---

## Results

### OOF Performance (5-fold frozen split)

| Model | OOF WF1 | OOF HRec | Fold Std |
|---|---|---|---|
| CatBoost | 0.84849 | — | 0.01258 |
| LightGBM | **0.86472** | — | 0.01268 |
| XGBoost | 0.86083 | — | 0.01100 |
| **Optuna Blend** | **0.86329** | **0.70426** | — |
| α=1.08 sweep | 0.8623+ | **0.71489** | — |

**Ensemble weights:** CB=0.398 · LGBM=0.532 · XGB=0.071

### Fold-Level Detail (FAZ 1.5 full train)

| Fold | CB WF1 | LGBM WF1 | XGB WF1 | CB bestIter |
|---|---|---|---|---|
| 1 | 0.82809 | 0.85288 | 0.85077 | **21** ⚠ |
| 2 | 0.86582 | 0.88559 | 0.87794 | 111 ✓ |
| 3 | 0.84579 | 0.85187 | 0.84726 | 507 ✓ |
| 4 | 0.84582 | 0.86113 | 0.86115 | 127 ✓ |
| 5 | 0.85623 | 0.87182 | 0.86702 | 248 ✓ |

### Seed Stability (FAZ 4)

| Seed | Blend WF1 | HRec | phc |
|---|---|---|---|
| 42 | 0.86221 | 0.69787 | 461 |
| 123 | 0.86105 | 0.69574 | 456 |
| 2025 | 0.86087 | 0.68723 | 456 |
| **Mean ± Std** | **0.8614 ± 0.0006** | **0.6936 ± 0.0046** | 458 ± 2 |

✅ `instability_flag = False` — model is seed-stable (std < 0.005 threshold)

---

## Key Engineering Decisions

**Leakage prevention:** FE and CE fit only on training folds. K-Fold TE uses inner 3-fold CV with Laplace smoothing — no validation data ever touches the fit.

**Fold precomputation [V13-PERF]:** `precompute_fold_data()` runs FE+CE once for all 5 folds before HP studies begin. Reduces overhead from ~1500 passes to 5 passes (~60× speedup).

**Dual trial selection:** For each HP study, both the absolute best WF1 trial and the band-best HRec trial are reported. Final selection prioritizes HRec within a WF1 ≥ best−0.001 band, with pred_high_count and weight entropy as tiebreakers.

**Guardrail system:** Every threshold/alpha decision validated against: Medium F1 drop (max 0.015), Medium Recall drop (max 0.020), pred_high_count ceiling (1.25× baseline). Violations automatically reject candidates.

**CatBoost Fold 1 instability:** `auto_class_weights="Balanced"` removed; explicit `class_weights={0:1, 1:2, 2:12}` used. Fold 1 bestIteration=21 persists — CB weight intentionally reduced to 0.398 by ensemble optimization.

**Proba-based gated rescue [V13 BUG-B]:** Gate uses `(cb[:,2] > t_cb) AND (blend[:,2] > t_blend)` instead of argmax — works even when CB proba is low. 121 threshold combinations searched (grid, [0.20–0.70]).

---

## Version History

| Version | OOF WF1 | HRec | Key Change |
|---|---|---|---|
| V9 | ~0.855 | — | K-Means + K-Fold TE introduced |
| V12 | 0.860 | — | WF1 primary objective, Faz-0 sweep |
| V12.1 | 0.8607 | 0.6809 | CB OD fix, dynamic threshold, recall_score fix |
| **V13** | **0.8633** | **0.7149** | Decoupled HP Optuna, proba gate, seed stability |

---

## Repository Structure

```
zindi-financial-health-challenge/
│
├── README.md
├── .gitignore
│
├── src/
│   ├── train_pipeline_v13.py                    # Main pipeline (V13)
│   └── train_pipeline_v12_1_hotfix_cb_nan_v4.py # V12.1 baseline reference
│
├── outputs/
│   ├── evidence_pack_v13.json                   # Full experiment evidence
│   ├── evidence_pack_v12_1.json                 # V12.1 reference evidence
│   └── fold_indices_v12.json                    # Frozen fold split
│
├── notebooks/
│   └── exploration.ipynb                        # EDA
│
└── docs/
    └── approach.md                              # Extended technical writeup
```

> **Note:** Raw data files (`Train.csv`, `Test.csv`) not included per Zindi's data policy.  
> Download from the [competition page](https://zindi.africa/competitions/financial-health-prediction-challenge).

---

## Installation & Usage

```bash
git clone https://github.com/<your-username>/zindi-financial-health-challenge.git
cd zindi-financial-health-challenge

pip install lightgbm xgboost catboost optuna scikit-learn pandas numpy

mkdir data
# Place Train.csv and Test.csv in data/

python src/train_pipeline_v13.py
```

**Expected outputs in `outputs/`:**
```
submission_v13_lgbm.csv
submission_v13_optuna_blend.csv
evidence_pack_v13.json
fold_indices_v12.json
```

**Expected runtime:** ~4–6 hours (RTX 4050 / Kaggle T4)

---

## Dependencies

```
Python       >= 3.9
lightgbm     >= 4.0
xgboost      >= 2.0
catboost     >= 1.2
optuna       >= 3.0
scikit-learn >= 1.3
pandas       >= 2.0
numpy        >= 1.24
```

---

## License

MIT — free to use, adapt, and build upon.

---

*Built with a multi-agent War Room validation approach: pipeline decisions cross-checked across Claude, ChatGPT, and Gemini.*
