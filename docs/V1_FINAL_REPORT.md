# AirSense V1
## Air Quality Prediction Using Statistical Analysis and Machine Learning

**Authoritative consolidated technical report. Protocol Phase 12.**

> **Provenance.** AirSense V1 is a period-authentic reconstruction of a
> classical Data Science study, carried out recently and constrained to
> methods and software available on or before **2019-04-26**. It does not
> claim to have existed in 2019. Every date in this repository is labelled
> as either historical or reconstruction.
>
> Every number below is traceable to a frozen artifact via
> `artifacts/v1_claims_ledger.csv`. Where prose and machine-readable
> evidence could disagree, the evidence governs.

---

## Abstract

We ask how accurately classical statistical and machine-learning methods
available in 2019 can estimate hourly PM2.5 concentration from
meteorological and calendar covariates. Using the UCI Beijing PM2.5 data set
(43,824 hourly records, 2010-01-01 to 2014-12-31, of which 41,757 carry an
observed target), we adopt a strictly chronological evaluation design:
2010–2012 for development training, 2013 for validation and hyperparameter
selection, and **2014 as a locked held-out test opened exactly once**. Four
pre-registered models share one frozen 43-feature representation: a
training-median constant baseline (M0), ordinary least squares (M1), a
decision tree (M2) and a random forest (M3). All four were refitted on
2010–2013 and scored on the same 8,661 unseen observations.

**M3 achieved the best final held-out MAE at 48.535 µg/m³**, a 25.81%
improvement on the naive baseline, and was also best on RMSE (70.278) and R²
(0.435). Performance degraded severely in the upper tail: on the 512 hours
exceeding the development-derived 95th percentile (282 µg/m³), M3's MAE rose
to **160.9 µg/m³** and it under-predicted **96.3%** of them. Residuals
remained strongly temporally structured (M3 lag-1 autocorrelation **0.903**),
indicating substantial temporal information the concurrent-hour design does
not capture.

Classical 2019 methods therefore predict ordinary hourly PM2.5 meaningfully
better than a trivial baseline, but they are weakest precisely where
accuracy matters most. **No operational, regulatory or clinical readiness is
claimed.**

---

## 1. Study Motivation

Airborne particulate matter below 2.5 µm penetrates deep into the
respiratory system and is among the most consequential air-quality
measurements for public health. Beijing's 2010–2014 record covers a period
of severe, widely studied pollution.

Concentration is not driven by emissions alone: meteorology governs whether
pollutants disperse or accumulate. That makes the relationship between
weather and PM2.5 a genuine statistical question, and a natural subject for
classical regression.

AirSense asks how far the classical toolkit gets — and is equally interested
in where it fails.

## 2. Research Question

> **How accurately can classical statistical and machine-learning methods
> available in 2019 predict PM2.5 concentration from meteorological and
> temporal information?**

**The task is concurrent-hour estimation**, not forecasting. PM2.5 for an
hour is estimated from the meteorological and calendar description of *that
same hour*. **No past PM2.5 value is used as an input.**

This distinction is deliberate and is maintained throughout. **V1 must not be
described as next-hour or future-horizon forecasting.** A lagged-target
autoregressive formulation would be an easier and different question, and was
explicitly excluded (§8.7).

## 3. Historical Scope and 2019 Constraint

Hard cutoff **2019-04-26**. Reference environment: Ubuntu 18.04 LTS,
CPython 3.6.7, numpy 1.15.4, pandas 0.23.4, scipy 1.1.0, scikit-learn 0.20.0,
matplotlib 3.0.2, seaborn 0.9.0, jupyter 1.0.0 — every pin verified against
PyPI as published before the cutoff.

Excluded by the contract: deep-learning frameworks, gradient-boosting
libraries, transformers, LLM APIs, SHAP, MLflow, AutoML, and any API
introduced after the cutoff. The full list and its rationale are in
`HISTORICAL_COMPATIBILITY.md`.

An unlocked install of the seven direct pins resolved **57 of 69**
distributions to post-2019 releases, so `requirements-v1-2019-lock.txt` pins
the complete transitive closure to pre-cutoff versions. The executing runtime
contains **57 distributions, none postdating 2019-04-26**.

---

## 4. Dataset

### 4.1 Source

UCI Machine Learning Repository, **Beijing PM2.5** (ID 381, DOI
`10.24432/C5JS49`), donated 2017-01-18 by Song Xi Chen, Peking University.
PM2.5 measured at the US Embassy in Beijing; meteorology at Beijing Capital
International Airport, joined on the hourly timestamp. The original
UCI-distributed CSV was used — not a cleaned derivative.

### 4.2 Observation period

**2010-01-01 00:00 to 2014-12-31 23:00**, hourly. 1,826 days × 24 = **43,824
records**, and the file contains exactly that many: **the hourly timeline is
complete, with no absent timestamp and no duplicate.**

### 4.3 Schema

13 columns: `No` (index), `year`, `month`, `day`, `hour`, **`pm2.5`**
(target, µg/m³), `DEWP` (°C), `TEMP` (°C), `PRES` (hPa), `cbwd` (categorical
wind direction), `Iws` (cumulated wind speed, m/s), `Is` and `Ir` (cumulated
hours of snow and rain).

### 4.4 Raw-data integrity

SHA-256 **`4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`**,
2,010,494 bytes, stored read-only (mode `444`). **The digest has been
re-verified at the start and end of every subsequent phase and has never
changed.**

### 4.5 Missing-target structure

**2,067 of 43,824 targets are missing (4.7166%). No predictor has any
missing value.** After exclusion, **41,757 supervised observations** remain.

Missingness is **strongly non-uniform in time**: 7.64% (2010), 8.31% (2011),
5.57% (2012), 0.94% (2013), 1.13% (2014) — 91.2% of missing targets fall in
2010–2012. It is essentially independent of hour of day (4.11%–5.42%).

This matters for the chronological design: the training period is more
selectively observed than the evaluation periods, a **target-availability
shift** (not predictor covariate shift, since all predictors are complete).

---

## 5. Exploratory Findings

All from Phase 3, computed on the full dataset before any split was frozen.

### 5.1 PM2.5 distribution

n = 41,757 · mean **98.61** · median **72.0** · std 92.05 · min 0 · max
**994** · IQR 108 · P90 221 · P95 284 · P99 420 · **skewness 1.80** · excess
kurtosis 4.77. Strongly right-skewed with a long upper tail.

High values are described as **high-concentration observations, not
outliers**: nothing establishes them as measurement error, and discarding
them would remove exactly the events the project exists to model.

### 5.2 Temporal patterns

A clear diurnal cycle: mean lowest at hour 15 (85.5) and highest at hour 01
(113.7). Annual means lie in a 91–104 band across the five years with **no
monotonic trend**.

### 5.3 Seasonal behaviour

Winter mean 112.78 with median 70.0 and std 119.20; summer mean 90.44 with
median **77.0** and std 64.17. **Mean and median disagree**: winter is not
uniformly dirtier — it holds both the cleanest and by far the most extreme
hours.

### 5.4 Meteorological relationships

Strongest association is cumulated wind speed, negative: Pearson −0.248,
Spearman **−0.360**. Dew point is the strongest positive (Spearman +0.300).
**Temperature's Pearson (−0.091) and Spearman (+0.011) disagree in sign**, a
signature of a non-monotone relationship neither coefficient summarises.
`Ir`'s rank association is not distinguishable from zero.

### 5.5 Wind-direction patterns

The largest categorical separation observed: median PM2.5 **31 µg/m³ under
`NW`** against **98 under `cv`**, with `NE` and `SE` between.

### 5.6 Correlation and multicollinearity

Among the meteorological predictors: `DEWP`–`TEMP` **+0.825**,
`TEMP`–`PRES` **−0.827**, `DEWP`–`PRES` **−0.778**. All associations with
PM2.5 are small in absolute terms (largest |ρ| = 0.360) — meteorology alone
explains a limited share of hourly variation, unsurprising given that
emissions and traffic are unmeasured.

---

## 6. Cleaning Policy

### 6.1 Missing-target exclusion

**A row is eligible for supervised modelling if and only if `pm2.5` is
non-missing.** Ineligible rows are excluded from the *derived* datasets only.
`data/raw/` retains all 43,824 rows, and every excluded row is recorded in
`artifacts/target_missing_exclusions.csv`.

### 6.2 Why target imputation was rejected

For those 2,067 hours the predictors exist but the ground truth does not.
Imputing the target would manufacture the quantity the model must learn.
Interpolation is the worst option precisely *because* hourly PM2.5 is
strongly autocorrelated: the fabricated value looks plausible, adds no
information, and in an evaluation period would leak a neighbouring observed
value into a scored prediction. **Nothing was filled, interpolated or
synthesised.**

### 6.3 Retention of zero values

Two `pm2.5 == 0` observations were **retained**. Whether they are genuine
near-zero readings or a reporting floor cannot be determined from this file,
and "unusual" is not evidence of error.

### 6.4 Retention of high-concentration observations

**Retained unchanged**, including the 994 µg/m³ maximum. Nothing was clipped,
winsorized or capped.

---

## 7. Evaluation Design

### 7.1 Development train — 2010–2012

26,304 raw rows, **24,418 supervised**.

### 7.2 Validation — 2013

8,760 raw rows, **8,678 supervised**. Used for model and hyperparameter
selection.

### 7.3 Final held-out test — 2014

8,760 raw rows, **8,661 supervised**. Opened exactly once, at Phase 10.

Missing-target rate by partition: **7.170% / 0.936% / 1.130%** — the training
period is roughly seven times more selectively observed than validation.

### 7.4 Temporal leakage controls

Membership is assigned from **calendar time on the full raw timeline, before
any target-based exclusion** — enforced in code, so the year-dependent
missing rate cannot shift the boundaries. Every training observation
precedes every validation observation, which precedes every test
observation. No shuffling. Every encoding parameter was fitted on the
training period only. **The optional random-split comparison permitted by the
protocol as a secondary educational exercise was not performed**, and V1
closes without it.

### 7.5 Test quarantine and blind prediction freeze

The 2014 target was sealed from Phase 4 to Phase 10. Phase 10 ran in two
mandatory stages: **Stage A** refitted all four models on 2010–2013 and
generated 2014 predictions from **predictors only**, then froze them
(`final_blind_prediction_freeze.json`); **Stage B**, which imports no
estimator and contains no `fit` or `predict` call, opened the target and
scored the already-frozen vectors. The scored predictions were verified
**byte-identical** to the pre-opening ones.

---

## 8. Frozen Feature Representation

**One 43-column matrix shared by M1, M2 and M3**, so the comparison reflects
model class rather than differing feature engineering.

### 8.1 Six meteorological features

`DEWP`, `TEMP`, `PRES`, `Iws`, `Is`, `Ir`, used **exactly as distributed** —
no transform, no binning, no polynomial expansion.

### 8.2 Month representation

One-hot, reference `month_1` omitted → **11 columns**. Monthly means are
non-monotone (Feb 125.7, Aug 80.0), so an integer encoding would force a
straight line through a pattern that is not one.

### 8.3 Hour representation

One-hot, reference `hour_0` omitted → **23 columns**. An integer encoding
would assert `23 > 22 > … > 0`, which is both false and unable to represent
a cycle.

### 8.4 Wind-direction representation

One-hot over the training vocabulary `{NE, NW, SE, cv}`, reference `cbwd_NE`
omitted → **3 columns**. Vocabulary learned from 2010–2012 only; an unseen
level in validation or test would have raised rather than silently added a
column. None occurred.

### 8.5 Excluded fields

`No` (a distributor's running index, collinear with time), `year` (evaluation
years are disjoint from training years, so a numeric year could only be
extrapolated), `day` (no declared physical interpretation), `season`
(deterministic function of month, already carried more granularly), and a
weekend indicator — whose Phase-0 admission condition required a
weekday/weekend difference established by the pre-modelling EDA, which was
never tested. **The EDA was not reopened to make it eligible.** This records
adherence to the frozen policy, not evidence that weekends have no effect.

### 8.6 Why no scaling

Ordinary least squares is invariant to rescaling a predictor; trees split on
thresholds and are invariant to monotone rescaling. Scaling would change
nothing while adding learned preprocessing state that must be applied
correctly to a locked test set. The M1 design is ill-conditioned
(§9.2) — that is reported, not corrected.

### 8.7 Why no lagged PM2.5

A lagged target would change the question from concurrent-hour estimation to
forecasting, would need its own baseline (persistence, not a training mean),
and would not be comparable to the results here. It was excluded in Phase 0
and remains excluded. **§12.7 measures the cost of that choice.**

---

## 9. Models

### 9.1 M0 — Training-median baseline

Constant predictor. The median minimises absolute error for a constant, and
**MAE was pre-registered as the decision metric**, so the choice follows
deductively rather than from any observed result. No mean baseline was
computed as a competitor. Final constant, recomputed on 2010–2013:
**73.0 µg/m³**.

### 9.2 M1 — Ordinary Linear Regression

`LinearRegression(fit_intercept=True)`, no scaling, no regularization, no
tuning. Final intercept **2435.4089340577**. The final design is **full rank
(44/44)** but ill-conditioned — condition number **213,048** — so individual
coefficients are numerically fragile and, combined with the |r| ≈ 0.78–0.83
collinearity, are **not interpretable as effect sizes**. No coefficient is
given a causal reading anywhere in this report.

### 9.3 M2 — Decision Tree Regressor

`criterion="mse"`, `splitter="best"`, `max_depth=12`,
`min_samples_split=2`, `min_samples_leaf=10`, `max_features=None`,
`random_state=42`. Selected from a **20-candidate grid frozen before any tree
was fitted**. Final tree: depth 12, 1,787 nodes, 894 leaves.

### 9.4 M3 — Random Forest Regressor

`n_estimators=100`, `criterion="mse"`, `bootstrap=True`, `max_depth=12`,
`min_samples_split=2`, `min_samples_leaf=1`, `max_features="auto"`,
`random_state=42`, `n_jobs=1`, `oob_score=False`. Selected from a
**24-candidate grid frozen before any forest was fitted**. `n_estimators` was
fixed, not tuned. Out-of-bag scoring was deliberately not used for selection
— it ignores temporal ordering. Final forest: 100 trees, mean depth 12.00,
mean 1,784.5 leaves.

---

## 10. Development Validation

**2013 figures. Development only — superseded by §11.**

### 10.1 M0 results

MAE 67.591 · RMSE 102.182 · R² −0.086. The negative R² is expected: R² is
measured against variance about the evaluation year's mean, while M0 is
optimised for MAE via the median.

### 10.2 M1 results

MAE 53.333 · RMSE 76.558 · R² 0.391 — a 21.09% MAE improvement on M0.
Produced 658 negative predictions (7.58%).

### 10.3 M2 hyperparameter selection

20 candidates; selected `max_depth=12`, `min_samples_leaf=10` by lowest 2013
MAE (50.828). **M2 beat M1 on MAE but lost on RMSE and R²** — the first
appearance of a metric disagreement that persists to the final test. The
unrestricted-depth, `min_samples_leaf=1` candidate is a textbook overfit:
training MAE 0.056, validation MAE 64.41, the worst of the twenty.

### 10.4 M3 hyperparameter selection

24 candidates; selected `max_depth=12`, `min_samples_leaf=1`,
`max_features="auto"` by lowest 2013 MAE (47.939). Notably, **candidate 11
had better RMSE and R²** but higher MAE by 0.024 µg/m³ and was not selected:
the pre-registered metric decided. M3 repaired M2's RMSE/R² regression
against M1.

### 10.5 Development-selection caveat

**M2 and M3 used 2013 to choose hyperparameters. M0 and M1 did not** — M0 has
nothing to select and M1 has an analytic solution. Their 2013 figures are
therefore not comparable in kind: M2's and M3's carry selection optimism,
M0's and M1's do not.

The final 2014 test is the proper unbiased comparison under the declared
protocol, and the observed direction is consistent with the caveat:

| Model | Selected on 2013? | ΔMAE (test − validation) |
|---|---|---:|
| M0 | no | **−2.174** |
| M1 | no | **−0.433** |
| M2 | **yes** | **+0.421** |
| M3 | **yes** | **+0.597** |

Both selecting models got slightly worse; both non-selecting models got
slightly better. The magnitude is ~1% of MAE. **This is one study on one
year and is not offered as a universal statistical result** — but it is in
the predicted direction, and it is why the locked test existed.

---

## 11. Final Held-Out Evaluation

### 11.1 Final refit

All four models refitted on the full permitted **2010–2013 development
population: 33,096 × 43**. The Phase-5 matrices were concatenated unchanged —
no feature engineering was re-run, no reference category changed, no scaling
introduced. Hyperparameters remained exactly as frozen.

### 11.2 Blind prediction freeze

Stage A was executed **twice before the target was touched**; all six
artifacts were byte-identical. Independent checks confirmed M0's predictions
were a single value equal to the development median, M1's reconstructed from
`intercept + X·β` to 9.7 × 10⁻¹³, and M2's and M3's lay inside the
development target range. The blind artifact contained **no target column**.

### 11.3 Test opening

With the pre-test freeze (51 hashes) and blind freeze (7 hashes) verified,
Stage B performed the single authorised read of the 2014 target, loading only
identifier, calendar and target columns. **All 8,661 identifiers and
timestamps matched the frozen predictions in order.** The revealed target was
frozen to a snapshot and an opening receipt written; the Phase-4 source has
not been reopened since.

### 11.4 Final MAE/RMSE/R²

> **2014 LOCKED HELD-OUT TEST** — fitted on 2010–2013, scored once on the
> same 8,661 observations.

| Model | MAE (µg/m³) | RMSE (µg/m³) | R² | MAE rank |
|---|---:|---:|---:|---:|
| **M0** naive constant | 65.417388 | 96.741589 | −0.069943 | 4 |
| **M1** linear regression | 52.900194 | 73.103403 | 0.389045 | 3 |
| **M2** decision tree | 51.248814 | 75.522046 | 0.347949 | 2 |
| **M3 random forest** | **48.535341** | **70.277633** | **0.435364** | **1** |

**A metric-ranking disagreement exists and is not hidden: M2 beats M1 on MAE,
while M1 beats M2 on both RMSE and R².** The two metrics weight errors
differently — RMSE squares them and is dominated by the largest misses, and
M2's residual range (−391 to +516) is wider than M1's (−208 to +520). MAE was
pre-registered and was not changed.

### 11.5 Final ranking

**M3 < M2 < M1 < M0** by the pre-registered primary metric.

**M3 is the final best V1 model under the pre-registered MAE criterion.** It
improves on the naive baseline by **16.882 µg/m³ (25.81%)**, on M1 by 8.25%
and on M2 by 5.30%.

M3 also happened to be best on final RMSE and R², so the headline does not
depend on which metric is privileged. **This is not a claim that M3 is
universally superior under every possible criterion** — the M1/M2 ordering
demonstrably reverses between metrics, and §12 identifies regimes where the
picture is more mixed.

---

## 12. Error Analysis

Descriptive post-evaluation characterisation. **No model, prediction or
metric was changed.**

### 12.1 Overall error distribution

| Model | Mean | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|
| M0 | 65.417 | 49.000 | 148.000 | 235.000 | 598.000 |
| M1 | 52.900 | 40.180 | **108.113** | **150.416** | 520.049 |
| M2 | 51.249 | 35.374 | 118.017 | 163.750 | 515.544 |
| **M3** | **48.535** | **34.220** | 109.561 | 153.127 | **504.744** |

M3 is best on the mean, median, P99 and maximum. **M1 is marginally better at
P90 and P95** — its mid-upper error tail is slightly tighter despite a worse
typical error. Median absolute errors sit far below the means for every
model, so a minority of hours carries a disproportionate share of the error.

### 12.2 Seasonal error

M3 MAE: Winter **62.738**, Spring 39.624, Summer **38.014**, Autumn 54.241.
**M3 is best in all four seasons** — its advantage is broad, not localised.
Winter error is 65% above summer.

### 12.3 Diurnal error

M3 MAE ranges **39.634 (hour 15)** to **56.281 (hour 01)**, best at every
hour. The error retains the same diurnal shape Phase 3 found in the
concentration itself: the models inherited the cycle rather than removing it.

### 12.4 Concentration-dependent error

**The dominant structure in the study.** Bands are derived from the
2010–2013 development target and were frozen before any 2014 subgroup metric:
P25 29, P50 73, P75 138, P90 221, **P95 282** µg/m³. These are **descriptive
relative bands, not regulatory AQI categories.**

M3 MAE by band: 33.333 → 39.038 → 34.373 → 59.597 → 94.174 → **160.897**.
M3 mean residual: **−33.1 → +155.9**. Under-prediction rate: **4.07% →
96.29%**.

**Every model over-predicts low concentrations and under-predicts high ones,
monotonically.** This is regression toward the conditional mean — not a bug,
but the central practical limitation of V1.

### 12.5 Severe high-concentration hours

**512 of 8,661 test hours (5.9%) exceed 282 µg/m³.**

| Model | MAE | RMSE | Under-prediction | Max abs. error |
|---|---:|---:|---:|---:|
| M0 | 293.990 | 302.226 | **100.0%** | 598.0 |
| M1 | 194.676 | 206.806 | **100.0%** | 520.0 |
| M2 | 164.260 | 188.734 | 94.7% | 515.5 |
| **M3** | **160.897** | **181.215** | 96.3% | **504.7** |

**Every model fails badly here.** M3 is best of four, yet its tail MAE is
**3.3× its overall MAE** and it under-predicts 96.3% of severe hours by an
average of 156 µg/m³. **No event-detection or classification claim is made:
these models predict a concentration, not an event class.**

### 12.6 Severe episodes

**41 contiguous episodes**, durations min 1 / median 7 / mean 12.5 /
**max 76 consecutive hours**. A model whose errors persist across such a
stretch is wrong for days at a time, not for isolated hours.

### 12.7 Residual temporal autocorrelation

Computed by **exact timestamp matching** — never row shifting, since the
test set holds 8,661 of 8,760 hours.

| Lag | 1 h | 6 h | 12 h | 24 h | 48 h | 168 h |
|---|---:|---:|---:|---:|---:|---:|
| **M3** | **0.903** | 0.592 | 0.438 | 0.344 | 0.193 | 0.103 |

**Errors remain strongly temporally structured**, particularly at short lags.
This indicates **substantial temporal information remains unexplained by the
concurrent-hour V1 design**.

Stated carefully: it does **not** mean any model is invalid, it does not
identify a mechanism, and **it does not prove any specific future model will
solve the problem**. It is a measured property of V1's residuals and a
motivation for prospective time-series work (§18).

### 12.8 M3 versus M2

M3 has the lower absolute error on **4,623 of 8,661 rows (53.38%)**, M2 on
4,038, with no ties. **M3's advantage is real but not universal row-by-row**
— it wins by more, not much more often (mean per-row difference −2.71 µg/m³).

### 12.9 Negative M1 predictions

M1 produced **466 physically impossible negative predictions (5.38%)**,
minimum −162.14 µg/m³. On those rows the actual PM2.5 averaged 21.2 and M1's
MAE was 58.99 — worse than its overall figure, so they are not harmless.
**They were not clipped**; clipping after the test was opened would be a
post-test model modification. **M2 and M3 produced none**, structurally,
because their outputs are averages of observed non-negative training targets.

---

## 13. Principal Findings

1. **Classical non-linear models outperform the naive baseline under
   chronological testing.** M3 improves on M0 by 16.882 µg/m³ MAE (25.81%)
   on genuinely unseen data.
2. **M3 produced the best final held-out MAE** (48.535 µg/m³) and was also
   best on RMSE and R².
3. **The random forest's improvement over the single tree is real but not
   universal row-by-row** — 53.38% of hours.
4. **Linear regression substantially improves on M0** (19.0% MAE) **but
   generates physically impossible negative concentration estimates** — 466
   of 8,661.
5. **M2 and M3 avoid negative predictions** because their outputs derive
   from observed-target leaf averages.
6. **Performance is much worse during high-concentration episodes** — M3's
   tail MAE is 3.3× its overall MAE.
7. **High-concentration observations are overwhelmingly under-predicted** —
   96.3% for M3 above the development P95.
8. **Residual temporal autocorrelation remains strong** — 0.903 at one hour
   for M3.
9. **A meteorology-and-calendar-only concurrent-hour feature set leaves
   important variation unexplained** — R² 0.435 means roughly 56% of hourly
   variance is unaccounted for.
10. **Metric choice changes the middle of the ranking.** M2 beats M1 on MAE;
    M1 beats M2 on RMSE and R².

**All of these are predictive associations. None is a causal claim about the
atmosphere.**

---

## 14. Limitations

- **Single monitoring location** — one US Embassy site.
- **Beijing-specific setting**; nothing generalises to other cities.
- **Fixed 2010–2014 historical window**; five annual aggregates cannot
  establish a trend.
- **Only 41,757 observed supervised labels from 43,824 hourly records.**
- **Target missingness is temporally non-uniform** — ~7× higher in training
  than in evaluation.
- **No emissions data.** **No traffic data.** **No industrial-activity
  data.** **No satellite information.** **No multi-station spatial
  information.**
- **No PM2.5 lag or history**, and therefore **no true future-horizon
  forecasting setup**; covariates are concurrent meteorological measurements.
- **Severe-tail weakness** — the central practical limitation.
- **Residual autocorrelation** — substantial unexplained temporal structure.
- **M1 numerical conditioning** — condition number 213,048 with |r| ≈
  0.78–0.83 collinearity; coefficients are not interpretable as effect sizes.
- **M1 negative predictions** — 5.38% of test rows.
- **Tree and forest impurity importance is not causal** — it favours
  high-cardinality predictors and splits credit arbitrarily among correlated
  ones.
- **M2/M3 validation-selection optimism** — quantified at +0.42 and +0.60
  µg/m³.
- **The 2014 test is exhausted.** It is no longer unseen, and **these results
  cannot be treated as a fresh evaluation for any future variant.**
- **Subgroup sizes are uneven** (e.g. `p90_p95` holds 346 observations).
- **No significance test, confidence interval or bootstrap ranking inference
  was performed** — none was pre-declared.

## 15. What V1 Cannot Answer

V1 **cannot** establish:

- causal effects of meteorological variables on PM2.5;
- regulatory AQI risk classification;
- future-horizon forecast accuracy;
- generalisation to other cities or monitoring stations;
- operational alert performance;
- extreme-event detection accuracy;
- whether lagged target information improves forecasting;
- whether deep-learning or foundation models outperform V1;
- whether spatial modelling improves performance.

Each requires a study V1 did not perform.

## 16. Reproducibility and Evidence Integrity

Every phase is reproducible from frozen inputs, and every deterministic
artifact was verified byte-identical across two independent runs — including
figures, achieved by pinning PNG metadata and writing no execution timestamp
into any scientific artifact.

The evidence chain is hash-linked: the raw digest is carried into the split
manifest, the feature schema, each model manifest, the pre-test development
freeze, the blind-prediction freeze, the opening receipt, the final
evaluation manifest, and the error-analysis manifest. **95 upstream hashes
were re-verified at the start of this phase; all matched.**

Every numerical claim in this report is traced in
`artifacts/v1_claims_ledger.csv` (**43 claims, all verified** — 40 by literal
lookup in the named artifact, 3 by recomputing a stated derivation).

Full instructions: `docs/V1_REPRODUCIBILITY.md`.

## 17. Conclusions

Classical statistical and machine-learning methods available in 2019 can
estimate hourly Beijing PM2.5 from meteorology and calendar position
**meaningfully better than a naive baseline** under a strict chronological
evaluation: the random forest reduced mean absolute error by 25.81% relative
to a training-median constant on a year of genuinely unseen data.

They are, however, **weakest exactly where accuracy matters most**. Above the
development 95th percentile the best model's error more than triples and it
under-predicts almost every hour. A user asking "is the air ordinary today?"
is served reasonably; a user asking "will the air be dangerous?" is not.

Two structural facts bound what this design can achieve. Roughly 56% of
hourly variance is unexplained, consistent with emissions, traffic and
industrial activity being absent from the dataset. And the residuals remain
strongly autocorrelated at short lags, indicating temporal information the
concurrent-hour formulation cannot reach by construction.

**No operational, regulatory or clinical readiness is claimed.**

## 18. Future Research Directions

**Labelled as future V2 directions. None is implemented, evaluated or
depended upon in V1.**

- **Lag-aware forecasting**, with a persistence or autoregressive baseline —
  the natural response to §12.7, and a *different* question from V1's.
- Gradient-boosted trees.
- LSTM / GRU recurrent models; temporal convolutional models.
- Modern time-series transformers; foundation time-series models.
- Multi-station and spatial modelling; graph neural networks.
- Uncertainty estimation and calibrated extreme-event forecasting — the
  natural response to §12.5.
- Explainability methods beyond impurity importance.
- External emissions, traffic and satellite features.

**V2 must be conducted separately from frozen V1**, and must not reuse the
2014 partition as an unseen test.

---

*AirSense V1 is complete and frozen. Its evidence is immutable; see
`artifacts/v1_final_freeze_receipt.json`.*
