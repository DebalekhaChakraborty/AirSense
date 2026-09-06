# AirSense

**Air Quality Prediction Using Statistical Analysis and Machine Learning**

AirSense asks how accurately classical statistical and machine-learning
methods estimate hourly PM2.5 concentration in Beijing from meteorological
and calendar information alone. V1 is the complete classical study: four
pre-registered models, one frozen feature matrix, a strictly chronological
evaluation design, and a 2014 held-out test opened exactly once.

---

## V1 Status: COMPLETE AND FROZEN

AirSense V1 modelling and evaluation are complete. The 2014 held-out test has
been consumed and is no longer an unseen test set.

*Frozen* is a statement about the science, not about the documentation: no
model, feature, split, hyperparameter, prediction, metric or ranking changes,
and no model evaluated here may claim a fresh evaluation on 2014. Factual and
documentation corrections remain permitted and are made transparently, as
recorded in
[`artifacts/v1_final_freeze_receipt.json`](artifacts/v1_final_freeze_receipt.json).

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
> 8,661 unseen hours. Final MAE ranking: **M3 < M2 < M1 < M0**.

**Two limitations that qualify that headline:**

- **Severe pollution remains the central failure.** On the 512 hours above
  the development-derived P95 (282 µg/m³), M3's MAE rises to **160.9** —
  3.3× its overall figure — and it **under-predicts 96.3%** of them.
- **Errors remain temporally structured.** M3's residuals correlate at
  **0.903 at a one-hour lag**, so substantial temporal information is left
  unexplained by the concurrent-hour design.

**V1 is concurrent-hour PM2.5 estimation under chronological generalisation,
not future-horizon forecasting** — no lagged PM2.5 history is used as an
input, so nothing here predicts a future hour from the pollution record. No
operational, regulatory or clinical readiness is claimed, and no causal claim
is made anywhere.

**[Read the full report →](docs/V1_FINAL_REPORT.md)** ·
**[Reproduce the evidence →](docs/V1_REPRODUCIBILITY.md)** ·
[pre-registered protocol](docs/V1_RESEARCH_PROTOCOL.md) ·
[error analysis](docs/ERROR_ANALYSIS.md) ·
[freeze receipt](artifacts/v1_final_freeze_receipt.json)

---

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
question, and it was fixed deliberately — see
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

What the exploratory analysis established, in short — full version in
[`docs/EDA_ANALYSIS.md`](docs/EDA_ANALYSIS.md):

- PM2.5 is strongly right-skewed: mean 98.6 against median 72.0 µg/m³,
  skewness 1.80, maximum 994.
- The hourly timeline is **complete** — 43,824 of 43,824 expected timestamps,
  no gap and no duplicate — and missing values affect the **target only**
  (2,067; 4.72%), strongly year-dependent (8.31% in 2011 against 0.94% in
  2013).
- Cumulative wind speed has the strongest single association with PM2.5
  measured here, and it is negative (Spearman −0.360); wind direction
  separates the target more than any other variable examined (median 31
  µg/m³ under `NW` against 98 under `cv`). The meteorological predictors are
  strongly intercorrelated (|r| ≈ 0.78–0.83 among `DEWP`, `TEMP`, `PRES`),
  which limits how far individual linear coefficients can be interpreted.

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

## Methodology

Four models, declared in advance and all carried through to the final test:

| ID | Model | Selection |
|---|---|---|
| **M0** | Training-median constant baseline | none — no predictor is used |
| **M1** | Linear Regression (OLS) | none — no hyperparameter search |
| **M2** | Decision Tree Regressor | 20 pre-declared candidates, chosen on 2013 MAE |
| **M3** | Random Forest Regressor | 24 pre-declared candidates, chosen on 2013 MAE |

Primary metrics, declared in advance: **MAE**, **RMSE**, **R²**. The
comparison is decided on MAE.

M1, M2 and M3 share **one frozen 43-feature matrix** — six raw meteorological
features plus one-hot `month`, `hour` and `cbwd` built from the 2010–2012
training vocabulary — so the comparison reflects model class rather than
differing feature engineering. No scaling, no target transformation, no
imputation, and no lag, rolling, interaction or target-derived features. The
2,067 missing targets are excluded from supervised modelling and audited in
`artifacts/target_missing_exclusions.csv`; the raw dataset itself is
unchanged.

**Evaluation uses a chronological split** — an earlier period trains, a later
period tests. A random split on an hourly time series lets adjacent hours
land on both sides, so the model is scored largely on interpolation between
near-copies and looks better than it is. The protocol allowed a random split
to appear as a clearly labelled *secondary* comparison, because such
workflows were common in 2019 introductory training; it was never run, and
V1's only reported design is the chronological one. Full rule:
[`docs/V1_RESEARCH_PROTOCOL.md`](docs/V1_RESEARCH_PROTOCOL.md) §3.

**41,757 supervised observations**: **2010–2012 development training**
(24,418 rows) · **2013 validation** (8,678) · **2014 locked final test**
(8,661). Split membership was assigned from calendar time on the full raw
timeline, before any target-based exclusion, and the boundaries were frozen
before any model was fitted.

The twelve phases, from environment reconstruction to the final report, are
fixed in [`docs/V1_RESEARCH_PROTOCOL.md`](docs/V1_RESEARCH_PROTOCOL.md) and
each has a phase record in [`docs/`](docs/).

---

## Development results (2013 validation)

> These are **development-validation figures from 2013**. They are not
> held-out test results and not project performance. M2 and M3 used 2013 to
> select hyperparameters; M0 and M1 selected nothing.

| Metric | M0 | M1 | M2 | **M3** |
|---|---:|---:|---:|---:|
| **MAE** *(primary)* | 67.591150 | 53.333361 | 50.827853 | **47.938823** |
| RMSE | 102.181753 | 76.558423 | 77.984115 | **73.090585** |
| R² | −0.085726 | 0.390520 | 0.367609 | **0.444484** |

- **M0** predicts the constant **73.0 µg/m³**, the median of the 2010–2012
  training target — the median because MAE was pre-registered and minimises
  absolute error for a constant predictor.
- **M1** is unregularised OLS. Two limitations were recorded rather than
  corrected: the training design is ill-conditioned (condition number
  2.1 × 10⁵, though full rank), and **7.58% of validation predictions (658)
  are negative** — physically impossible for a concentration. Clipping them
  would improve the metrics and was not done.
- **M2** selected `max_depth=12`, `min_samples_leaf=10` — depth 12, 1,597
  nodes, 799 leaves, root split on `DEWP`. It **wins on MAE but loses to M1
  on RMSE and R²**; the disagreement is reported rather than resolved by
  switching metrics. The unrestricted-depth candidate is a textbook overfit:
  training MAE 0.056, validation MAE 64.41, the worst of all twenty.
- **M3** selected `max_depth=12`, `min_samples_leaf=1`, `max_features="auto"`
  — 100 trees, all depth 12, mean 1,662 leaves each. It is better than M0, M1
  and M2 on all three metrics simultaneously, the only model for which that
  holds.

Throughout development the **2014 target was never read**. With M3 complete,
[`artifacts/pretest_development_freeze.json`](artifacts/pretest_development_freeze.json)
sealed the whole pre-test state — raw, split and feature digests plus each
model's specification, selected hyperparameters, manifest digest and the
final refit rule — before the test was opened.

---

## How the final test was run

> **This was the first and only planned final evaluation.** The 2014 target
> was sealed until all four prediction vectors had already been generated and
> frozen. All four models were refitted on **2010–2013** (33,096 rows) and
> scored once against the same **8,661** unseen observations. The headline
> table is at the top of this README.

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
  the development phases warned about, showing up in the predicted direction
  at about 1% of MAE — small enough that the ranking held, real enough to
  justify the locked test.

**The 2014 target is now exhausted.** It is no longer an unseen test set, and
no model changed from here on may claim a fresh evaluation on it.

### Where the error lives

A **descriptive** post-evaluation analysis of the frozen 2014 predictions. No
model was fitted, no prediction changed, and the final metrics and ranking
are untouched. Concentration bands and the severe-hour threshold were frozen
from the **2010–2013 development target** before any 2014 subgroup metric was
computed.

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

**Not done, deliberately:** no imputation, no scaling, no target
transformation, and no post-test model change of any kind. All four models
were evaluated once on the locked 2014 test set; none was retrained, retuned
or dropped afterwards, and no fifth model was added. Modern methods belong to
a separate V2 that must not reuse the 2014 partition as an unseen test.

---

## Project structure

```
AirSense/
├── README.md
├── requirements-v1-2019.txt      frozen 2019 dependency set
├── requirements-v1-2019-lock.txt pre-cutoff transitive closure
├── data/
│   ├── README.md                 dataset provenance
│   ├── raw/                      original UCI CSV, read-only, never modified
│   └── processed/                supervised datasets, partitions, features/
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── analysis/                 EDA implementation
│   ├── data/                     data loading, auditing, cleaning and split
│   ├── features/                 frozen feature-preparation implementation
│   ├── models/                   M0–M3 model implementations
│   ├── evaluation/               final refit and post-test error-analysis logic
│   └── visualization/            plotting package, unused in V1
├── scripts/
│   ├── preflight.py              read-only foundation validation
│   ├── audit_dataset.py          writes artifacts/data_audit.json
│   ├── runtime_snapshot.py       writes artifacts/runtime_snapshot.json
│   ├── run_eda.py                exploratory data analysis
│   ├── prepare_data.py           cleaning and chronological split freeze
│   ├── prepare_features.py       frozen 43-feature matrices
│   ├── run_m0_baseline.py        M0 training-median baseline
│   ├── run_m1_linear_regression.py  M1 ordinary least squares
│   ├── run_m2_decision_tree.py   M2 decision tree over the declared grid
│   ├── run_m3_random_forest.py   M3 random forest over the declared grid
│   ├── create_pretest_freeze.py  seals the pre-test development state
│   ├── prepare_final_predictions.py  blind final refit on 2010–2013
│   ├── open_final_test.py        opens and scores the 2014 held-out test
│   ├── run_error_analysis.py     post-test descriptive error analysis
│   ├── build_v1_summary.py       summary tables and claims ledger
│   ├── freeze_v1_evidence.py     evidence registry and final freeze
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
│   ├── final_refit_manifest.json           2010–2013 refit record
│   ├── final_blind_prediction_freeze.json  frozen blind predictions
│   ├── final_test_opening_receipt.json     single target-opening receipt
│   ├── final_evaluation_manifest.json      final held-out results
│   ├── error_analysis_manifest.json        post-test error analysis
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
├── results/error_analysis/       subgroup and residual tables
├── results/v1_summary/           final model comparison + key findings
├── figures/                      EDA, model, final-test and error-analysis figures
├── docs/
│   ├── V1_RESEARCH_PROTOCOL.md   pre-registered twelve-phase protocol
│   ├── V1_FINAL_REPORT.md        authoritative consolidated report
│   ├── V1_REPRODUCIBILITY.md     end-to-end reproduction instructions
│   ├── ERROR_ANALYSIS.md         post-test error analysis
│   └── ...                       policy documents and per-phase records
├── tests/                        empty — V1 verification is script-based
└── venv/                         CPython 3.6.7 runtime - UNTRACKED
```

`venv/` is the execution environment, not source: it is gitignored and
rebuilt from the two requirements files. Run V1 code with `venv/bin/python`.
Despite the name it is a conda-style prefix, not a `python -m venv`
environment, so invoke the interpreter directly rather than activating it.

---

## Documentation

| Topic | Document |
|---|---|
| Pre-registered protocol | [`V1_RESEARCH_PROTOCOL.md`](docs/V1_RESEARCH_PROTOCOL.md) |
| Research question | [`RESEARCH_QUESTION.md`](docs/RESEARCH_QUESTION.md) |
| Historical compatibility contract | [`HISTORICAL_COMPATIBILITY.md`](docs/HISTORICAL_COMPATIBILITY.md) |
| Dataset audit | [`DATASET_AUDIT.md`](docs/DATASET_AUDIT.md) |
| Exploratory analysis | [`EDA_ANALYSIS.md`](docs/EDA_ANALYSIS.md) |
| Cleaning and split policy | [`CLEANING_AND_SPLIT_POLICY.md`](docs/CLEANING_AND_SPLIT_POLICY.md) |
| Features | [`FEATURE_POLICY.md`](docs/FEATURE_POLICY.md) · [`FEATURE_PREPARATION.md`](docs/FEATURE_PREPARATION.md) |
| Models | [`M0_BASELINE.md`](docs/M0_BASELINE.md) · [`M1_LINEAR_REGRESSION.md`](docs/M1_LINEAR_REGRESSION.md) · [`M2_DECISION_TREE.md`](docs/M2_DECISION_TREE.md) · [`M3_RANDOM_FOREST.md`](docs/M3_RANDOM_FOREST.md) |
| Final held-out evaluation | [`FINAL_HELD_OUT_EVALUATION.md`](docs/FINAL_HELD_OUT_EVALUATION.md) |
| Error analysis | [`ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md) |
| Final report | [`V1_FINAL_REPORT.md`](docs/V1_FINAL_REPORT.md) |
| Reproduction | [`V1_REPRODUCIBILITY.md`](docs/V1_REPRODUCIBILITY.md) |
| Phase records | `docs/PHASE_00_FOUNDATION_RECORD.md` through [`PHASE_10_FINAL_V1_RECORD.md`](docs/PHASE_10_FINAL_V1_RECORD.md) |

---

## Reproduce and verify

The frozen package can be checked, and the whole pipeline re-run, at any
time:

```sh
venv/bin/python scripts/verify_v1_final.py    # read-only check of the frozen package
venv/bin/python scripts/preflight.py    # expect: Overall: READY, exit 0
venv/bin/python scripts/run_eda.py      # regenerates results/eda/ and figures/
venv/bin/python scripts/prepare_data.py # regenerates data/processed/ and manifests
venv/bin/python scripts/prepare_features.py  # regenerates the 43-feature matrices
venv/bin/python scripts/run_m0_baseline.py   # regenerates the M0 baseline results
venv/bin/python scripts/run_m1_linear_regression.py  # regenerates the M1 results
venv/bin/python scripts/run_m2_decision_tree.py      # regenerates the M2 results
venv/bin/python scripts/run_m3_random_forest.py      # regenerates the M3 results
venv/bin/python scripts/create_pretest_freeze.py     # rewrites the pre-test freeze
venv/bin/python scripts/prepare_final_predictions.py # blind final refit
venv/bin/python scripts/open_final_test.py           # evaluate (target already open)
venv/bin/python scripts/run_error_analysis.py        # post-test error analysis
venv/bin/python scripts/build_v1_summary.py          # summary + claims ledger
venv/bin/python scripts/freeze_v1_evidence.py        # registry + freeze
```

Step-by-step instructions, including expected digests:
[`docs/V1_REPRODUCIBILITY.md`](docs/V1_REPRODUCIBILITY.md).

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
4. **The test set is used once.** It did not inform model choice, features,
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
