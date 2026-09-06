# AirSense

**Air Quality Prediction Using Statistical Analysis and Machine Learning**

The 2014 held-out test was opened once, scored once, and is now **exhausted**. No model, feature, split, hyperparameter, prediction or metric may change.

> ### 2014 HELD-OUT TEST — final result
>
> | Model | MAE (µg/m³) | RMSE (µg/m³) | R² | Rank |
> |---|---:|---:|---:|---:|
> | M0 naive constant | 65.417388 | 96.741589 | −0.069943 | 4 |
> | M1 linear regression | 52.900194 | 73.103403 | 0.389045 | 3 |
> | M2 decision tree | 51.248814 | 75.522046 | 0.347949 | 2 |
> | **M3 random forest** | **48.535341** | **70.277633** | **0.435364** | **1** |
>
> **Winner by the pre-registered primary metric (MAE): M3**, improving on
> the naive baseline by **25.81%**. Fitted on 2010–2013, scored once on
> 8,661 unseen hours.

**Two limitations that qualify that headline:**

- **Severe pollution remains the central failure.** On the 512 hours above
  the development-derived P95 (282 µg/m³), M3's MAE rises to **160.9** —
  3.3× its overall figure — and it **under-predicts 96.3%** of them.
- **Errors remain temporally structured.** M3's residuals correlate at
  **0.903 at a one-hour lag**, so substantial temporal information is left
  unexplained by the concurrent-hour design.

**V1 is concurrent-hour estimation, not forecasting** — no lagged PM2.5 is
used as an input. No operational, regulatory or clinical readiness is
claimed, and no causal claim is made anywhere.

**[Read the full report →](docs/V1_FINAL_REPORT.md)** ·
**[Reproduce the evidence →](docs/V1_REPRODUCIBILITY.md)** ·
freeze receipt: `artifacts/v1_final_freeze_receipt.json`

## Motivation

Airborne particulate matter under 2.5 micrometres (PM2.5) penetrates deep
into the respiratory system and is among the most consequential air-quality
measurements for public health. Beijing's PM2.5 record from 2010 to 2014
covers a period of severe and widely studied pollution episodes.

Pollution concentration is not driven by emissions alone. Meteorology
governs how pollutants disperse or accumulate: wind clears the air, still
cold conditions trap it, and pressure and humidity shape the boundary layer.
That makes the relationship between weather and PM2.5 a genuine statistical
question rather than a bookkeeping one — and a well-suited subject for
classical regression methods.

AirSense asks how far the classical toolkit gets on that question, and is
equally interested in where it fails.

---

## V1 research question

> **How accurately can classical statistical and machine-learning methods
> available in 2019 predict PM2.5 concentration from meteorological and
> temporal information?**

The V1 task is **regression** on the target `pm2.5`, in µg/m³.

It is **concurrent-hour estimation**, not forecasting: PM2.5 for an hour is
predicted from the meteorological and calendar description of that same hour.
No past PM2.5 value is used as an input, so the model must explain pollution
from weather and time alone. That is the harder and more interesting
question, and it is fixed deliberately — see
[`docs/RESEARCH_QUESTION.md`](docs/RESEARCH_QUESTION.md).

---

## Dataset

**UCI Beijing PM2.5 Data Set** (repository ID 381, DOI `10.24432/C5JS49`),
donated 2017-01-18 by Song Xi Chen, Peking University.

| | |
|---|---|
| Period | 2010-01-01 to 2014-12-31, hourly |
| Rows | 43,824 |
| Columns | 13 (1 index + 11 features + 1 target) |
| Target | `pm2.5` (µg/m³) |
| Missing | 2,067 `pm2.5` values (4.7166%); all other columns complete |
| PM2.5 source | US Embassy, Beijing |
| Meteorology source | Beijing Capital International Airport |
| License | CC BY 4.0 as currently presented by UCI |

The original UCI-distributed CSV is used, not a pre-cleaned derivative. It is
stored read-only in `data/raw/` with SHA-256
`4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`.

Full provenance: [`data/README.md`](data/README.md).
Full audit: [`docs/DATASET_AUDIT.md`](docs/DATASET_AUDIT.md).

The dataset's 2017 donation date places it comfortably before the 2019
cutoff, so its use is period-consistent.

---

## Historical technology constraints

**Hard cutoff: 2019-04-26.**

Reference environment (what V1 source is written against):

| Component | Pin |
|---|---|
| OS | Ubuntu 18.04 LTS |
| Python | CPython 3.6.7 |
| numpy | 1.15.4 |
| pandas | 0.23.4 |
| scipy | 1.1.0 |
| scikit-learn | 0.20.0 |
| matplotlib | 3.0.2 |
| seaborn | 0.9.0 |
| jupyter | 1.0.0 |

Every pin was verified against PyPI as published before the cutoff, and every
one ships a CPython 3.6 wheel. Frozen in
[`requirements-v1-2019.txt`](requirements-v1-2019.txt).

**Excluded from V1:** TensorFlow, PyTorch, Keras, XGBoost, LightGBM,
CatBoost, transformers, Hugging Face, LLM APIs, foundation models,
generative AI, agent frameworks, SHAP, MLflow, Weights & Biases, Optuna,
AutoML, modern time-series foundation models, deep neural networks,
cloud-managed AI services, and any API introduced after 2019-04-26.

Reasoning and the full list: [`docs/HISTORICAL_COMPATIBILITY.md`](docs/HISTORICAL_COMPATIBILITY.md).

> **Environment status: resolved.** The host runs Debian 12 with CPython
> 3.11, on which the pinned 2019 stack cannot be installed. Rather than
> modernise the pins, a conforming **CPython 3.6.7** environment was built in
> user space and lives in-repo at `venv/` (untracked). All seven direct pins
> install at their frozen versions, and the transitive closure is locked to
> pre-cutoff releases by
> [`requirements-v1-2019-lock.txt`](requirements-v1-2019-lock.txt) — without
> it, an unlocked install resolves 57 of 69 distributions to post-2019
> releases. `scripts/preflight.py` reports **READY** (exit 0) under `venv/`,
> and still reports **HOST-COMPATIBILITY WARNING** under the system
> interpreter. Details in
> [`docs/PHASE_00A_RUNTIME_RECORD.md`](docs/PHASE_00A_RUNTIME_RECORD.md).

---

## Planned methodology

Four models, declared in advance:

| ID | Model |
|---|---|
| **M0** | Simple baseline predictor (constant) — **complete**, training-median |
| **M1** | Linear Regression — **complete** (2013 validation) |
| **M2** | Decision Tree Regressor — **complete** (2013 validation) |
| **M3** | Random Forest Regressor — **complete** (2013 validation) |

Primary metrics, declared in advance: **MAE**, **RMSE**, **R²**. The
comparison is decided on MAE.

**Evaluation uses a chronological split** — an earlier period trains, a later
period tests. A random split on an hourly time series lets adjacent hours
land on both sides, so the model is scored largely on interpolation between
near-copies and looks better than it is. A random split may appear later as a
clearly labelled *secondary* comparison, because such workflows were common
in 2019 introductory training and the gap between the two designs is
instructive — but it never becomes the headline result. Full rule:
[`docs/V1_RESEARCH_PROTOCOL.md`](docs/V1_RESEARCH_PROTOCOL.md) §3.

Twelve phases, from environment reconstruction to the final report, are fixed
in [`docs/V1_RESEARCH_PROTOCOL.md`](docs/V1_RESEARCH_PROTOCOL.md).

---

## Project structure

```
AirSense/
├── README.md
├── requirements-v1-2019.txt      frozen 2019 dependency set
├── data/
│   ├── README.md                 dataset provenance
│   ├── raw/                      original UCI CSV, read-only, never modified
│   └── processed/                supervised datasets, partitions, features/
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── analysis/                 EDA implementation
│   ├── data/                     data loading and auditing
│   ├── features/                 feature preparation (empty)
│   ├── models/                   model code (empty)
│   └── visualization/            plotting (empty)
├── requirements-v1-2019-lock.txt pre-cutoff transitive closure
├── scripts/
│   ├── preflight.py              read-only foundation validation
│   ├── audit_dataset.py          writes artifacts/data_audit.json
│   ├── runtime_snapshot.py       writes artifacts/runtime_snapshot.json
│   ├── run_eda.py                runs the Phase 3 analysis
│   ├── prepare_data.py           runs the Phase 4 cleaning and split freeze
│   ├── prepare_features.py       runs the Phase 5 feature preparation
│   ├── run_m0_baseline.py        runs the Phase 6 M0 baseline
│   ├── run_m1_linear_regression.py  runs the Phase 7 M1 linear regression
│   ├── run_m2_decision_tree.py   runs the Phase 8 M2 decision tree
│   ├── run_m3_random_forest.py   runs the Phase 9 M3 random forest
│   ├── create_pretest_freeze.py  writes the pre-test development freeze
│   ├── prepare_final_predictions.py  Phase 10A blind final refit
│   ├── open_final_test.py        Phase 10B held-out evaluation
│   ├── run_error_analysis.py     Phase 11 post-evaluation error analysis
│   ├── build_v1_summary.py       Phase 12 summary tables + claims ledger
│   ├── freeze_v1_evidence.py     Phase 12 evidence registry and freeze
│   └── verify_v1_final.py        verifies the frozen V1 package
├── artifacts/
│   ├── data_audit.json           machine-readable dataset audit
│   ├── runtime_snapshot.json     machine-readable runtime evidence
│   ├── eda_summary.json          machine-readable EDA summary
│   ├── split_manifest.json       frozen chronological split record
│   ├── target_missing_exclusions.csv  2,067-row exclusion audit trail
│   ├── feature_schema.json       frozen 43-feature schema
│   ├── feature_row_manifest.csv  41,757-row alignment manifest
│   ├── m0_baseline_manifest.json M0 baseline record
│   ├── m1_linear_regression_manifest.json  M1 record
│   ├── m2_decision_tree_manifest.json      M2 record
│   ├── m3_random_forest_manifest.json      M3 record
│   ├── pretest_development_freeze.json     sealed pre-test state
│   ├── final_refit_manifest.json           Phase 10A refit record
│   ├── final_blind_prediction_freeze.json  frozen blind predictions
│   ├── final_test_opening_receipt.json     single target-opening receipt
│   ├── final_evaluation_manifest.json      final held-out results
│   ├── error_analysis_manifest.json        Phase 11 error analysis
│   ├── v1_claims_ledger.csv                every report claim, traced
│   ├── v1_evidence_registry.csv            canonical file registry
│   ├── v1_final_evidence_manifest.json     final evidence manifest
│   └── v1_final_freeze_receipt.json        V1 freeze declaration
├── results/eda/                  15 EDA result tables
├── results/preprocessing/        partition and reconciliation audit
├── results/features/             feature decisions and matrix validation
├── results/models/m0/            M0 validation metrics and predictions
├── results/models/m1/            M1 coefficients, diagnostics, metrics, predictions
├── results/models/m2/            M2 grid, importances, metrics, predictions
├── results/models/m3/            M3 grid, importances, metrics, predictions
├── results/final_test/           2014 held-out evaluation + blind refit
├── results/error_analysis/       Phase 11 subgroup and residual tables
├── results/v1_summary/           final model comparison + key findings
├── figures/                      9 EDA figures
├── docs/
│   ├── HISTORICAL_COMPATIBILITY.md
│   ├── DATASET_AUDIT.md
│   ├── RESEARCH_QUESTION.md
│   ├── FEATURE_POLICY.md
│   ├── V1_RESEARCH_PROTOCOL.md
│   ├── PHASE_00_FOUNDATION_RECORD.md
│   ├── PHASE_00A_RUNTIME_RECORD.md
│   ├── EDA_ANALYSIS.md
│   ├── PHASE_01_EDA_RECORD.md
│   ├── CLEANING_AND_SPLIT_POLICY.md
│   ├── PHASE_02_CLEANING_RECORD.md
│   ├── FEATURE_PREPARATION.md
│   ├── PHASE_03_FEATURE_RECORD.md
│   ├── M0_BASELINE.md
│   ├── PHASE_04_M0_BASELINE_RECORD.md
│   ├── M1_LINEAR_REGRESSION.md
│   ├── PHASE_05_M1_LINEAR_REGRESSION_RECORD.md
│   ├── M2_DECISION_TREE.md
│   ├── PHASE_06_M2_DECISION_TREE_RECORD.md
│   ├── M3_RANDOM_FOREST.md
│   ├── PHASE_07_M3_RANDOM_FOREST_RECORD.md
│   ├── FINAL_HELD_OUT_EVALUATION.md
│   ├── PHASE_08_FINAL_TEST_RECORD.md
│   ├── ERROR_ANALYSIS.md
│   ├── PHASE_09_ERROR_ANALYSIS_RECORD.md
│   ├── V1_FINAL_REPORT.md
│   ├── V1_REPRODUCIBILITY.md
│   └── PHASE_10_FINAL_V1_RECORD.md
├── tests/                        (empty)
└── venv/                         CPython 3.6.7 runtime - UNTRACKED
```

`venv/` is the execution environment, not source: it is gitignored and
rebuilt from the two requirements files. Run V1 code with `venv/bin/python`.
Despite the name it is a conda-style prefix, not a `python -m venv`
environment, so invoke the interpreter directly rather than activating it.

---

## Current status

**Phase 0 — foundation. Complete.**

| Phase | Status |
|---|---|
| 1. Environment reconstruction | Complete — CPython 3.6.7 built in-repo at `venv/`, preflight READY |
| 2. Dataset provenance and audit | Complete |
| 3. Exploratory data analysis | **Complete** — see [`docs/EDA_ANALYSIS.md`](docs/EDA_ANALYSIS.md) |
| 4. Cleaning and split freeze | **Complete** — see [`docs/CLEANING_AND_SPLIT_POLICY.md`](docs/CLEANING_AND_SPLIT_POLICY.md) |
| 5. Feature preparation | **Complete** — see [`docs/FEATURE_PREPARATION.md`](docs/FEATURE_PREPARATION.md) |
| 6. Baseline regression (M0) | **Complete** — see [`docs/M0_BASELINE.md`](docs/M0_BASELINE.md) |
| 7. Linear regression (M1) | **Complete** — see [`docs/M1_LINEAR_REGRESSION.md`](docs/M1_LINEAR_REGRESSION.md) |
| 8. Decision tree (M2) | **Complete** — see [`docs/M2_DECISION_TREE.md`](docs/M2_DECISION_TREE.md) |
| 9. Random forest (M3) | **Complete** — see [`docs/M3_RANDOM_FOREST.md`](docs/M3_RANDOM_FOREST.md) |
| 10. Held-out evaluation | **Complete** — see [`docs/FINAL_HELD_OUT_EVALUATION.md`](docs/FINAL_HELD_OUT_EVALUATION.md) |
| 11. Error analysis | **Complete** — see [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) |
| 12. Final report and evidence freeze | **Complete** — see [`docs/V1_FINAL_REPORT.md`](docs/V1_FINAL_REPORT.md) |

### Validated findings from Phase 3

High-level only; the full analysis is in
[`docs/EDA_ANALYSIS.md`](docs/EDA_ANALYSIS.md).

- PM2.5 is strongly right-skewed — mean 98.6 against median 72.0 µg/m³,
  skewness 1.80, maximum 994.
- The hourly timeline is **complete**: 43,824 of 43,824 expected timestamps,
  no gap and no duplicate, already in chronological order.
- Missing values affect the **target only** (2,067; 4.72%), and they are
  **strongly year-dependent** — 8.31% in 2011 against 0.94% in 2013. This
  matters for the chronological split and is carried into Phase 4.
- Cumulative wind speed has the strongest single association with PM2.5
  measured here, and it is negative (Spearman −0.360).
- Wind direction separates the target more than any other variable examined:
  median 31 µg/m³ under `NW` against 98 µg/m³ under `cv`.
- The meteorological predictors are strongly intercorrelated
  (|r| ≈ 0.78–0.83 among `DEWP`, `TEMP`, `PRES`), which will limit how far
  individual linear coefficients can be interpreted.

All EDA artifacts are deterministic: two runs produce byte-identical CSVs,
JSON and PNGs. No model has been trained and no predictive metric exists.

### Phase 4 — cleaning and split freeze

- **Missing targets are excluded from supervised modelling, not imputed.**
  All 2,067 remain in the raw data, and every one is recorded in
  `artifacts/target_missing_exclusions.csv`.
- **41,757 supervised observations** in total.
- **2010–2012 development training** (24,418 rows) · **2013 validation**
  (8,678) · **2014 locked final test** (8,661).
- Split membership is assigned from **calendar time on the full raw
  timeline, before** any target-based exclusion.
- Zero-valued and high-concentration PM2.5 observations are **retained
  unchanged**.
- Target missingness is far higher in the training period than later —
  7.17% against 0.94% (2013) and 1.13% (2014), a label-availability
  asymmetry documented for reporting.
- **The raw dataset is unchanged**, byte-identical and read-only.

The 2014 target is **quarantined until Protocol Phase 10**. Boundaries were
frozen before any model was fitted; no predictive metric exists anywhere in
this repository.

### Phase 5 — feature preparation

- **One shared 43-feature matrix** for M1, M2 and M3, so the later
  comparison reflects model class rather than differing feature engineering.
- **Six raw meteorological features** (`DEWP`, `TEMP`, `PRES`, `Iws`, `Is`,
  `Ir`) used exactly as distributed.
- **`month`, `hour` and `cbwd` one-hot encoded** from the 2010–2012 training
  vocabulary only, with `month_1`, `hour_0` and `cbwd_NE` as declared
  reference levels: 11 + 23 + 3 columns.
- **Excluded:** `No` (identifier), `year` (disjoint future years),
  `day` (no declared signal), `season` (redundant with month), `weekend`
  (the Phase 0 admission condition was not met), cyclical sin/cos, and all
  lag, rolling, interaction and target-derived features.
- **No scaling and no target transformation.** PM2.5 is written unmodified.
- **The 2014 target remained quarantined** — the test file is read with an
  explicit column list omitting `pm2.5`, and no `y_test` artifact exists.

Matrices: `X_train` 24,418 × 43 · `X_validation` 8,678 × 43 ·
`X_test` 8,661 × 43. All feature decisions were fixed before any model
performance existed.

### Phase 6 — M0 naive baseline

**M0 is a training-median constant baseline**: the constant is the median
of the 2010–2012 development-training target, and every prediction equals
it. M0 uses none of the 43 predictors — that is what makes it a naive floor.

The median was chosen because MAE was pre-registered as the decision metric
and the median minimises absolute error for a constant predictor. Exactly
one baseline was implemented; no mean-versus-median comparison was run.

**Baseline constant: 73.0 µg/m³** (fitted on 2010–2012).

> #### 2013 DEVELOPMENT VALIDATION — not final test performance
>
> | Metric | Value |
> |---|---:|
> | MAE | **67.591150** µg/m³ |
> | RMSE | **102.181753** µg/m³ |
> | R² | **−0.085726** |
>
> These are development-validation figures from 2013. They are **not**
> held-out test results and **not** project performance. The negative R² is
> expected: R² is measured against the evaluation year's own mean, while M0
> is optimised for MAE via the median.

**The 2014 target was never read.** No 2014 prediction or metric exists;
the locked test partition is reserved for Protocol Phase 10, where M0–M3
are evaluated together, once.

### Phase 7 — M1 linear regression

**M1 is ordinary least squares** — `LinearRegression(fit_intercept=True)`,
scikit-learn 0.20.0 — fitted on the **frozen 43-feature matrix**, on
2010–2012, and evaluated on 2013. No scaling, no regularization, no
hyperparameter search, no feature ablation.

> #### 2013 DEVELOPMENT VALIDATION ONLY — not final test performance
>
> | Metric | M0 (constant) | **M1** | Improvement |
> |---|---:|---:|---:|
> | **MAE** *(primary)* | 67.591150 | **53.333361** | **−21.09%** |
> | RMSE | 102.181753 | **76.558423** | −25.08% |
> | R² | −0.085726 | **0.390520** | +0.476246 |
>
> **M1 improves on the naive floor by 21.09% of MAE**, so the 43 predictors
> carry real information a constant does not. These are development
> figures from one year; they are **not** held-out results.

Two limitations recorded rather than corrected: the training design is
ill-conditioned (condition number 2.1 × 10⁵, though full rank), and **7.58%
of validation predictions are negative** — physically impossible for a
concentration, and an unavoidable consequence of unconstrained OLS on a
non-negative target. Clipping them would improve the metrics and was not
done.

**The 2014 target was still never read.**

### Phase 8 — M2 decision tree

**M2 is `DecisionTreeRegressor`** on the **same frozen 43-feature matrix as
M1**, fitted on 2010–2012. **20 candidates** — `max_depth` ∈ {3, 5, 8, 12,
None} × `min_samples_leaf` ∈ {1, 10, 50, 100} — were declared in code
**before any tree was fitted**, with `criterion="mse"`, `splitter="best"`,
`random_state=42` fixed. Selection was by lowest 2013 validation MAE under a
tie-break policy frozen in advance.

**Selected: `max_depth=12`, `min_samples_leaf=10`** (candidate 14) — depth
12, 1,597 nodes, 799 leaves, root split on `DEWP`.

> #### 2013 DEVELOPMENT VALIDATION ONLY — not final test performance
>
> | Metric | M0 | M1 | **M2** | vs M0 | vs M1 |
> |---|---:|---:|---:|---:|---:|
> | **MAE** *(primary)* | 67.591150 | 53.333361 | **50.827853** | **−24.80%** | **−4.70%** |
> | RMSE | 102.181753 | 76.558423 | 77.984115 | −23.68% | **+1.86% worse** |
> | R² | −0.085726 | 0.390520 | 0.367609 | +0.453 | **−0.023 worse** |
>
> **M2 wins on the pre-registered decision metric (MAE) but loses to M1 on
> RMSE and R².** By the declared criterion M2 is better; the disagreement is
> reported rather than resolved by switching metrics. M2 improves the
> typical hour while making worse extreme errors.

Two structural notes: M2 produced **zero negative predictions** (M1 produced
658), since a tree outputs leaf means of observed targets; and the
unrestricted-depth candidate is a textbook overfit — training MAE 0.056,
validation MAE 64.41, the **worst** of all twenty.

**The 2014 target was still never read**, and no 2014 information influenced
hyperparameter selection.

### Phase 9 — M3 random forest

**M3 is `RandomForestRegressor`** on the **same frozen 43-feature matrix as
M1 and M2**, fitted on 2010–2012. **24 candidates** — `max_depth` ∈ {8, 12,
None} × `min_samples_leaf` ∈ {1, 10, 50, 100} × `max_features` ∈ {"auto",
0.5} — were declared **before any forest was fitted**, with
`n_estimators=100`, `criterion="mse"`, `bootstrap=True`, `random_state=42`,
`n_jobs=1`, `oob_score=False` fixed. Selection was by lowest 2013 validation
MAE under a tie-break policy frozen in advance.

**Selected: `max_depth=12`, `min_samples_leaf=1`, `max_features="auto"`**
(candidate 9) — 100 trees, all depth 12, mean 1,662 leaves each.

> #### 2013 DEVELOPMENT VALIDATION ONLY — not final test performance
>
> | Metric | M0 | M1 | M2 | **M3** |
> |---|---:|---:|---:|---:|
> | **MAE** *(primary)* | 67.591150 | 53.333361 | 50.827853 | **47.938823** |
> | RMSE | 102.181753 | 76.558423 | 77.984115 | **73.090585** |
> | R² | −0.085726 | 0.390520 | 0.367609 | **0.444484** |
>
> MAE improvement: **−29.08% vs M0**, **−10.11% vs M1**, **−5.68% vs M2**.
>
> **M3 is better than M0, M1 and M2 on all three metrics simultaneously** —
> the only model for which that holds — and it repairs the RMSE/R²
> regression M2 had against M1. These are development figures from one
> year; they are **not** held-out results, and M2 and M3 both used 2013 to
> select hyperparameters while M0 and M1 selected nothing.

With M3 complete the development programme finished, and
[`artifacts/pretest_development_freeze.json`](artifacts/pretest_development_freeze.json)
seals the whole pre-test state — raw, split and feature digests plus each
model's specification, selected hyperparameters, manifest digest and
Phase-10 refit rule. From here, the feature schema, model specifications,
split boundaries, primary metric and test policy are immutable; changing any
of them would require invalidating and resetting the planned final-test
evaluation, documented **before** the test is reopened.

---

## Final 2014 held-out test results

> **This is the first and only planned final evaluation.** The 2014 target
> was sealed until all four prediction vectors had already been generated
> and frozen. All four models were refitted on **2010–2013** (33,096 rows)
> and scored once against the same **8,661** unseen observations.
>
> These figures are **final held-out performance** — distinct from, and not
> to be confused with, the 2013 development-validation figures reported
> above.

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² | MAE rank |
|---|---:|---:|---:|---:|
| **M0** naive constant | 65.417388 | 96.741589 | −0.069943 | 4 |
| **M1** linear regression | 52.900194 | 73.103403 | 0.389045 | 3 |
| **M2** decision tree | 51.248814 | 75.522046 | 0.347949 | 2 |
| **M3 random forest** | **48.535341** | **70.277633** | **0.435364** | **1 — best** |

**Best final model by MAE: M3, the random forest** — 25.8% better than the
naive floor, 8.2% better than linear regression, 5.3% better than the single
tree. M3 is best on all three metrics.

Two results worth stating plainly rather than smoothing over:

- **The rankings disagree in the middle.** By RMSE and R² the order is
  M3 < M1 < M2 < M0 — M1 beats M2 on both, while M2 beats M1 on MAE. MAE was
  pre-registered as the decision metric and was not changed to make the
  ordering tidy.
- **The models that used 2013 to pick hyperparameters got slightly worse on
  2014, and the ones that didn't got slightly better.** M0 −2.17 and M1
  −0.43 MAE; M2 **+0.42** and M3 **+0.60**. That is the selection optimism
  Phases 8–9 warned about, showing up in the predicted direction at about
  1% of MAE — small enough that the ranking held, real enough to justify the
  locked test.

**The 2014 target is now exhausted.** It is no longer an unseen test set,
and no model changed from here on may claim a fresh evaluation on it.

### Phase 11 — where the error lives

A **descriptive** post-evaluation analysis of the frozen 2014 predictions.
No model was fitted, no prediction changed, and the final metrics and
ranking are untouched. Concentration bands and the severe-hour threshold
were frozen from the **2010–2013 development target** before any 2014
subgroup metric was computed.

- **M3's advantage is broad.** It has the lowest MAE in all four seasons, at
  every hour of the day, and in five of six concentration bands — though it
  wins on only **53.4% of individual hours**, so its edge comes from winning
  by more, not more often.
- **Severe pollution remains the central failure.** Of 8,661 test hours,
  **512 exceed the development P95 (282 µg/m³)**, falling in **41 episodes**
  of up to **76 consecutive hours**. There M3's MAE is **160.9** — 3.3× its
  overall figure — and it **under-predicts 96.3%** of them. M0 and M1
  under-predict **100%**. Every model regresses toward the middle: all
  over-predict low concentrations and under-predict high ones.
- **Residual autocorrelation remains high.** M3's errors correlate at
  **0.903 at one hour** and 0.344 at a full day, matched on exact
  timestamps. Substantial predictable structure remains that a
  concurrent-hour model cannot reach — a hypothesis for future time-series
  work, **not** a V1 change.
- **M1 still emits physically impossible output.** 466 predictions
  (**5.38%**) are negative, minimum −162.1 µg/m³, and its MAE on those rows
  (59.0) is worse than its overall. **Not clipped** — that would be
  post-test modification.

Full analysis: [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md).

Done: project structure; historical compatibility contract with verified
package release dates; frozen requirements plus a pre-cutoff transitive
lock; a conforming CPython 3.6.7 runtime; dataset acquired from the original
UCI source, audited and hashed; research question, feature policy and
research protocol pre-registered; read-only preflight utility; and the
Phase 3 exploratory data analysis.

**Not done, deliberately:** no imputation, no scaling, no target
transformation, and no post-test model change of any kind. All four models
were evaluated once on the locked 2014 test set; none was retrained, retuned
or dropped afterwards, and no fifth model was added. **V1 is closed.** Modern
methods belong to a separate V2 that must not reuse the 2014 partition as an
unseen test.

`artifacts/` contains measurements only — a dataset audit, a runtime
snapshot, and a descriptive EDA summary. There are no model results anywhere
in this repository, and no placeholder values that could be mistaken for
results.

Verify the foundation, and reproduce the analysis, at any time:

```sh
venv/bin/python scripts/preflight.py    # expect: Overall: READY, exit 0
venv/bin/python scripts/run_eda.py      # regenerates results/eda/ and figures/
venv/bin/python scripts/prepare_data.py # regenerates data/processed/ and manifests
venv/bin/python scripts/prepare_features.py  # regenerates the 43-feature matrices
venv/bin/python scripts/run_m0_baseline.py   # regenerates the M0 baseline results
venv/bin/python scripts/run_m1_linear_regression.py  # regenerates the M1 results
venv/bin/python scripts/run_m2_decision_tree.py      # regenerates the M2 results
venv/bin/python scripts/run_m3_random_forest.py      # regenerates the M3 results
venv/bin/python scripts/create_pretest_freeze.py     # rewrites the pre-test freeze
venv/bin/python scripts/prepare_final_predictions.py # Phase 10A: blind final refit
venv/bin/python scripts/open_final_test.py           # Phase 10B: evaluate (target already open)
venv/bin/python scripts/run_error_analysis.py        # Phase 11: error analysis
venv/bin/python scripts/build_v1_summary.py          # Phase 12: summary + claims ledger
venv/bin/python scripts/freeze_v1_evidence.py        # Phase 12: registry + freeze
venv/bin/python scripts/verify_v1_final.py           # verify the frozen package
```

---

## Reproducibility philosophy

1. **Raw data is immutable.** `data/raw/` is read-only and never modified.
   Derived data goes to `data/processed/`. The raw SHA-256 is recorded so
   any later change is detectable.
2. **Provenance is recorded, not remembered.** Source URL, retrieval date,
   donation date, license and digest are all written down, and the
   reconstruction date is never presented as a historical one.
3. **Decisions are pre-registered.** Models, metrics and evaluation design
   were fixed before any result was seen, so they cannot be quietly retrofitted
   to the outcome.
4. **The test set is used once.** It does not inform model choice, features,
   hyperparameters or the split boundary.
5. **Negative results are reported.** Every declared model is reported,
   including poor performers.
6. **No fabricated evidence.** No invented history, no illustrative metrics,
   no placeholder numbers in results files. A number appears only after it
   has actually been computed.
7. **Constraints are documented rather than dissolved.** Where the historical
   environment could not be reproduced, that is recorded as a blocker instead
   of resolved by modernising the pins.

---

## Citation

Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5JS49

Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H., &
Chen, S. X. (2015). Assessing Beijing's PM2.5 pollution: severity, weather
impact, APEC and winter heating. *Proceedings of the Royal Society A*,
471(2182).
