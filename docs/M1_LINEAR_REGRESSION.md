# AirSense V1 M1 Linear Regression

**Research-protocol mapping: Protocol Phase 7 — Linear Regression (M1).**

> **The metrics reported here are 2013 development-validation metrics. They
> are not final held-out test results.** The 2014 partition remains locked
> and was not read. A final M1 score does not exist until Protocol Phase 10.

---

## Purpose

Fit the first feature-based model and establish whether the 43 prepared
predictors carry information a constant does not — and if so, how much.

M1 also serves a second purpose: it is the point at which the *linear
functional form itself* is put on trial. Phase 3 found relationships that
are visibly non-linear and predictors that are strongly intercorrelated.
M1 is expected to expose those limits rather than hide them.

---

## Protocol position

M1 was pre-registered in [`RESEARCH_QUESTION.md`](RESEARCH_QUESTION.md) §3
as *"Linear Regression — simplest parametric model; tests how much of PM2.5
is linearly explainable by weather and time."* It follows the M0 naive
baseline and precedes M2 Decision Tree.

---

## Frozen feature representation

The **exact 43-column Phase 5 matrix**, consumed as produced. M1 does not
reconstruct, add, remove, transform, scale or reorder any feature, and it
verifies the column order against
[`artifacts/feature_schema.json`](../artifacts/feature_schema.json) before
fitting.

| Block | Columns | Count |
|---|---|---:|
| Raw meteorological | `DEWP`, `TEMP`, `PRES`, `Iws`, `Is`, `Ir` | 6 |
| Month one-hot | `month_2` … `month_12` (ref `month_1`) | 11 |
| Hour one-hot | `hour_1` … `hour_23` (ref `hour_0`) | 23 |
| Wind direction one-hot | `cbwd_NW`, `cbwd_SE`, `cbwd_cv` (ref `cbwd_NE`) | 3 |
| | **Total** | **43** |

Rationale in [`FEATURE_PREPARATION.md`](FEATURE_PREPARATION.md), unchanged.

## Training period

**2010-01-01 00:00 → 2012-12-31 23:00** — 24,418 observations. The 2013
target was not consulted during fitting.

## Validation period

**2013-01-01 00:00 → 2013-12-31 23:00** — 8,678 observations.

---

## Model specification

```
LinearRegression(fit_intercept=True)
```

`sklearn.linear_model.LinearRegression`, scikit-learn 0.20.0. Ordinary
least squares, no other scientifically consequential option set. One
predeclared fit — **no hyperparameter search, no `GridSearchCV`, no
`RandomizedSearchCV`, no `TimeSeriesSplit`**, because this specification has
nothing to tune.

**Intercept: 2491.5560610308.**

---

## Why no scaling was introduced

Phase 5 froze "no scaling", and that decision stands.

It is also the mathematically correct call for this estimator: ordinary
least squares is **invariant** to rescaling a predictor. Rescaling a column
rescales its coefficient by the reciprocal and leaves the fitted values,
the residuals and every metric unchanged. There is no penalty term whose
behaviour depends on feature magnitude.

Scaling would therefore change nothing about the result, while adding a
piece of learned preprocessing state that would have to be fitted on
training data and applied correctly to the locked test set later.

The diagnostics below do show a large condition number. **That is a finding
to report, not a licence to change the experiment.** Introducing scaling
after seeing it would be a post-hoc modification of a pre-registered
specification.

## Why no regularization was introduced

Ridge, Lasso and ElasticNet are **different models**, outside the frozen
M0–M3 set. Adding one after observing coefficient instability would be
exactly the result-driven model selection the pre-registration exists to
prevent.

The purpose of M1 is specifically to evaluate an **ordinary** linear model —
including wherever that turns out to be a poor choice. Its weaknesses are
part of the finding.

Equally, **no feature was removed because of multicollinearity**, and no
ablation variant (`M1 without PRES`, `meteorology-only`, and so on) was
fitted. Those would be post-result feature selection.

---

## Training-design diagnostics

Computed on the **training matrix only**, with `numpy.linalg.svd`. No
dependency was added — statsmodels was not installed for VIF, and rank plus
singular values are sufficient here.

**Condition number is defined throughout as**

```
largest singular value / smallest NON-ZERO singular value
```

where "non-zero" means above `max(m, n) · eps · largest`, the standard
numerical rank tolerance.

Three matrices are reported, because they answer different questions and
quoting only one would mislead:

| Matrix | Shape | Rank | Rank deficient | σ_max | σ_min (non-zero) | Condition number |
|---|---|---:|---|---:|---:|---:|
| **`[1 \| X]` design actually solved** | 24,418 × 44 | **44** | **no** | 158,931 | 0.748356 | **212,373** |
| Raw `X_train` | 24,418 × 43 | 43 | no | 158,931 | 6.20043 | 25,632.2 |
| Centred `X_train` (what sklearn decomposes) | 24,418 × 43 | 43 | no | 8,176.89 | 6.20024 | 1,318.8 |

scikit-learn's own estimator attributes agree with the third row:
`rank_ = 43`, 43 singular values, largest 8,176.892, smallest 6.200238.

**The design is full rank — it is not rank deficient.** The 43 features are
linearly independent, and the reference-category omissions in each one-hot
block did their job: had all levels been retained, the design would have
been exactly singular.

**The headline condition number is 2.1 × 10⁵**, which indicates genuine
ill-conditioning. The dominant cause is visible in the σ_max values: `PRES`
lives near 1016 hPa while the 37 one-hot columns are 0/1, so the raw design
spans five orders of magnitude in column norm before any correlation is
considered. Adding the intercept column raises the condition number by a
further factor of ~8 (25,632 → 212,373), which is the numerical signature of
predictors that are far from centred.

The practical consequence is stated plainly: **individual coefficients are
numerically fragile.** The fitted values themselves are far better
conditioned than the coefficients, which is why the predictions below are
usable while the coefficients are not individually interpretable.

---

## Multicollinearity caution

Phase 3 measured strong pairwise relationships among the meteorological
predictors:

| Pair | Pearson r |
|---|---:|
| `DEWP` – `TEMP` | +0.825 |
| `TEMP` – `PRES` | −0.827 |
| `DEWP` – `PRES` | −0.778 |

Combined with the condition number above, this means individual
coefficients **cannot be read as effect sizes**.

Concretely, the following statements are **not** supported and are not made
anywhere in this project:

- ✗ "a one-unit increase in `TEMP` causes PM2.5 to fall by 2.92 µg/m³"
- ✗ "`DEWP` is the most important predictor because its coefficient is
  largest among the meteorological block"
- ✗ any ranking of scientific importance by coefficient magnitude

Each fails for a specific reason: causality is not identified by a
regression on observational data; magnitudes are not comparable because the
units differ and no scaling was applied; correlated predictors split a
shared association arbitrarily between themselves; and every one-hot
coefficient is defined only **relative to its omitted reference level**
(`month_1`, `hour_0`, `cbwd_NE`).

Where coefficients are discussed at all, the correct phrasing is
**"conditional association within this fitted linear specification"**.

The full coefficient vector is stored in
`results/models/m1/m1_coefficients.csv` in the frozen feature order — **not
sorted by value**, deliberately, so the file cannot be mistaken for an
importance ranking.

---

## 2013 validation results

**2013 DEVELOPMENT VALIDATION — not final test performance.**

| Metric | Definition | M1 |
|---|---|---:|
| **MAE** | `mean(abs(y_true - y_pred))` | **53.333361** µg/m³ |
| **RMSE** | `sqrt(mean((y_true - y_pred)^2))` | **76.558423** µg/m³ |
| **R²** | `1 - SS_res / SS_tot` | **0.390520** |

scikit-learn 0.20.0; RMSE is an explicit square root of MSE because that
version's `mean_squared_error` has no `squared=` keyword. Only these three
metrics were computed.

All three were independently recomputed from the prediction file without
scikit-learn and agree to within 7.1 × 10⁻¹⁵.

---

## Comparison with M0

M0's figures are read from
[`artifacts/m0_baseline_manifest.json`](../artifacts/m0_baseline_manifest.json)
rather than transcribed, so the comparison cannot drift from the recorded
baseline.

| Metric | M0 (constant 73.0) | M1 | Absolute change | Relative |
|---|---:|---:|---:|---:|
| **MAE** *(primary)* | 67.591150 | **53.333361** | **−14.257789** | **−21.09%** |
| RMSE | 102.181753 | **76.558423** | −25.623330 | −25.08% |
| R² | −0.085726 | **0.390520** | **+0.476246** | — |

**M1 improves on the naive floor by 21.09% of MAE on 2013.** The 43
predictors carry substantial information that a constant does not: a linear
combination of weather and calendar position explains roughly 39% of the
variance in hourly PM2.5, against a constant's slightly-negative R².

The **primary comparison metric remains MAE**, as pre-registered. It was not
switched to RMSE or R² because those show a larger improvement — although
they do, and the reason is instructive: M0 was deliberately optimised for
MAE via the median, so it was always going to look worst on the
squared-error metrics. Comparing on MAE is the harder and fairer test, and
it is the one that was declared.

---

## Residual diagnostics

Descriptive only, on 2013 validation. **Residual is defined throughout as
`actual − predicted`.** Nothing here was used to modify M1.

| Statistic | Value (µg/m³) |
|---|---:|
| Mean residual | +6.560091 |
| Median residual | −7.353553 |
| Residual std. dev. | 76.281242 |
| Minimum residual | −175.470247 |
| Maximum residual | **+687.220866** |
| Pearson corr(predicted, residual) | +0.022515 |

Three things stand out, recorded as observations rather than conclusions.

**Mean and median residuals have opposite signs.** The mean is +6.56 while
the median is −7.35. M1 slightly *over*-predicts the typical hour and badly
*under*-predicts a minority of severe ones — the +687 µg/m³ maximum residual
is a single episode the model missed by a factor of several. This is the
expected behaviour of a squared-error-fitted linear model on a strongly
right-skewed target, and it means the mean residual understates the model's
difficulty with the events that matter most.

**Predicted values and residuals are essentially uncorrelated** (r = 0.023),
which is what a well-specified linear fit should show on held-out data and
gives no evidence of gross systematic bias across the prediction range.

**M1 produces physically impossible predictions.** 658 of 8,678 validation
predictions (**7.58%**) are **below zero µg/m³**, the most extreme being
−150.54. PM2.5 concentration cannot be negative. This is an unavoidable
consequence of an unconstrained linear model on a non-negative target and is
recorded as a genuine limitation. **It was not "fixed"** — clipping
predictions at zero would improve the metrics and would be an undeclared
post-hoc modification of the specification.

Prediction spread: 8,675 distinct values, standard deviation 59.94, range
−150.54 to +264.58 µg/m³. M1 is verifiably not behaving like a constant.

**Figures** (both labelled *2013 Development Validation*):
`figures/m1_validation_actual_vs_predicted.png` (with a y = x reference
line) and `figures/m1_validation_residuals_vs_predicted.png`.

High-concentration validation observations were **not removed and not
separately refitted around**. Full severe-episode analysis belongs to
Protocol Phase 11.

---

## Coefficient interpretation limits

Summarised from the sections above, because this is the most likely thing
for a reader to get wrong:

1. Coefficients are **not causal**.
2. Magnitudes are **not comparable across features** — units differ, and no
   scaling was applied.
3. Correlated predictors **split a shared association arbitrarily**.
4. One-hot coefficients are **relative to an omitted reference**.
5. The condition number of 2.1 × 10⁵ means the coefficients are
   **numerically fragile** even given the data.

The coefficient file exists for reproducibility and for the equation
reconstruction check, not as a findings table.

---

## Test-set quarantine

**The 2014 partition was not read, not predicted and not evaluated.**

A token-level audit of both new source files found **zero executable
references** to `X_test`, `y_test`, the test partition file, `m1_test` or
2014 evaluation — every textual occurrence is inside a comment, docstring or
explanatory string. No `m1_test_predictions.csv`, `test_predictions.csv` or
`y_test.csv` exists anywhere in the repository.

The manifest records `test_evaluation_status: not_evaluated` and
`final_test_status: not_evaluated`, with no placeholder 2014 number.

---

## Final-refit policy

**Documented; deliberately not executed.**

At Protocol Phase 10, after all model-development decisions are frozen:

1. Combine the development data from 2010–2013.
2. Apply the already-frozen Phase 5 feature policy.
3. Refit M1 **once** on the full 2010–2013 development set.
4. Generate 2014 predictions.
5. Open the locked 2014 target.
6. Calculate MAE, RMSE and R² **once**.

The coefficients reported here are fitted on 2010–2012 only and are **not**
the final M1 coefficients.

**No serialized model was written.** Since Phase 10 refits M1 on a different
period, a pickled development model would not be the final model, and the
coefficients, intercept, diagnostics, predictions and metrics already
describe this fit completely and deterministically.

---

## Limitations

- **Development validation, not generalization.** These are 2013 figures
  from one year. They say nothing about 2014 performance, and nothing about
  production readiness.
- **The linear form is the binding constraint.** Phase 3 showed
  `TEMP`'s Pearson and Spearman coefficients disagree in sign — a
  non-monotone relationship a straight line cannot represent. M1 cannot
  capture it, and cannot represent any interaction either.
- **Negative predictions** (7.58% of validation rows) are physically
  meaningless and are not corrected.
- **Severe episodes are badly under-predicted**, with a maximum residual of
  +687 µg/m³.
- **Ill-conditioned design** (condition number 2.1 × 10⁵) makes individual
  coefficients unreliable, though the fitted values are far better behaved.
- **Meteorology is an incomplete system.** Emissions, traffic and industrial
  activity are unmeasured, which bounds what any V1 model can achieve.
- **No claim of superiority.** M1 beats M0 on 2013. That is a
  development-validation observation about two of four planned models, not a
  conclusion about the project.

---

## What remains for M2

Protocol Phase 8 fits **M2, a Decision Tree Regressor**, on the same
2010–2012 partition using the same frozen 43-feature matrix, evaluated on
2013 with the same three metrics.

M2 is of interest precisely where M1 is weakest: it can represent
non-monotone relationships and interactions without manual transformation,
and it cannot produce a negative prediction, since its outputs are averages
of observed training targets.

Nothing about M2 has been implemented. No tree estimator was imported or
instantiated in this phase.
