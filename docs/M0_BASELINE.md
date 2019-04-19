# AirSense V1 M0 Baseline

**Research-protocol mapping: Protocol Phase 6 — Baseline Regression (M0).**

> **The metrics reported here are development-validation metrics from 2013.
> They are not final held-out test results.** The 2014 partition remains
> locked and was not evaluated. A final M0 score does not exist until
> Protocol Phase 10.

---

## Purpose

Establish the quantitative error floor that M1, M2 and M3 must beat.

Reporting a linear regression or a random forest without a naive reference
makes its error figure uninterpretable: 67 µg/m³ of mean absolute error
means nothing until you know what a model that ignores every input
achieves. M0 supplies that number.

---

## Position in the research protocol

M0 was pre-registered in
[`RESEARCH_QUESTION.md`](RESEARCH_QUESTION.md) §3 as *"a constant predictor
(mean or median of the training target)"* — one of the four models fixed in
[`V1_RESEARCH_PROTOCOL.md`](V1_RESEARCH_PROTOCOL.md) §1 before any result
existed. This phase resolves that declared option to the median, on grounds
given below, and computes its 2013 validation metrics.

It is the first phase in which any predictive metric is permitted.

---

## Training period

**2010-01-01 00:00 → 2012-12-31 23:00** — the development-training
partition frozen in
[`CLEANING_AND_SPLIT_POLICY.md`](CLEANING_AND_SPLIT_POLICY.md).

**24,418 observed target values.** The 2013 validation target was not
consulted when fitting.

## Validation period

**2013-01-01 00:00 → 2013-12-31 23:00** — the development-validation
partition. **8,678 observed target values.**

---

## Baseline definition

```
c = median(y_train)                       fitted on 2010-2012 only
prediction(row) = c                       for every validation row
```

**Baseline constant: `c = 73.0` µg/m³.**

M0 uses **none** of the 43 prepared predictors. `X_train`, `X_validation`
and `X_test` are not read by the baseline at all. A model that consulted
features would not be a naive floor.

M0 has **no hyperparameters**. The single learned quantity is the median
above.

---

## Why the median was chosen

The decision metric was frozen in Phase 0 as **MAE**. For a constant
predictor, the sample **median** is the minimiser of mean absolute error,
just as the mean is the minimiser of squared error.

Choosing the median therefore follows **deductively from a metric declared
long before any result existed**. It is not a selection made after
comparing candidates.

This matters procedurally as much as mathematically:

- **No mean baseline was computed as a competing model.** There is no M0a
  and no M0b, and no mean-versus-median contest whose winner was adopted.
- **The strategy was written into the code and this document before its
  validation metrics were observed**, and was not revised afterwards.
- Had both been computed and the better one kept, M0 would have been
  quietly tuned on the validation set — the precise failure the
  pre-registration exists to prevent.

The training mean (97.82 µg/m³) appears in this document only as
descriptive context for the discussion below. It was never used as a
predictor.

---

## Relationship to the primary MAE metric

Two consequences follow from optimising a constant for MAE, and both are
visible in the results.

**M0 is close to the best constant possible for MAE on 2013.** The 2013
target's own median is 71.5; a predictor using it — which M0 could not
legitimately use, since it would require reading the evaluation target —
would achieve MAE 67.581. M0, using the *training* median of 73.0, achieves
**67.591**. The gap is 0.010 µg/m³. The central tendency of PM2.5 barely
moved between the training and validation periods, so the training median
transferred almost perfectly.

**M0 is deliberately not optimised for RMSE or R².** Those are
squared-error metrics, minimised by the *mean*, and the 2013 mean is
101.71 — far from 73.0, because the distribution is strongly right-skewed
(Phase 3: mean 98.61 against median 72.00, skewness 1.80). A median-based
constant necessarily scores worse on squared-error metrics than a
mean-based one would.

That trade-off is a property of the declared metric, not a defect, and it
was accepted when MAE was fixed as the decision metric.

---

## Validation metrics

**2013 DEVELOPMENT VALIDATION — not final test performance.**

| Metric | Definition | Value |
|---|---|---:|
| **MAE** | `mean(abs(y_true - y_pred))` | **67.591150** µg/m³ |
| **RMSE** | `sqrt(mean((y_true - y_pred)^2))` | **102.181753** µg/m³ |
| **R²** | `1 - SS_res / SS_tot` | **−0.085726** |

Computed with scikit-learn 0.20.0. Because that version's
`mean_squared_error` has no `squared=` keyword — it arrived in 0.22 — RMSE
is the explicit square root of MSE. Only these three metrics were computed;
no MAPE, SMAPE, explained variance, median absolute error or adjusted R².

All three were independently recomputed from the prediction file without
scikit-learn and agree to within 1×10⁻¹².

*Files:* `results/models/m0/m0_validation_metrics.csv`,
`results/models/m0/m0_validation_predictions.csv`,
[`artifacts/m0_baseline_manifest.json`](../artifacts/m0_baseline_manifest.json)

---

## Interpretation

**On the error floor.** Any model that cannot beat **MAE 67.59 µg/m³** on
2013 has learned nothing from the 43 predictors that a single constant does
not already provide. That is the number M1–M3 must improve on, and it is
the only purpose M0 serves.

**On the negative R².** R² is measured against the variance about the
*evaluation period's own mean*. A predictor that returns the 2013 mean
would score exactly 0. M0 returns 73.0, which is 28.7 µg/m³ below that
mean, so it scores slightly below zero.

This is **expected and correct**, not a sign of failure. It says only that
a median-based constant explains less squared variance than a mean-based
constant would — which is true by construction, and is the price of
minimising the metric the protocol actually declared. Reading R² = −0.086
as "M0 is broken" would be a misreading; reading it as "a constant carries
no information about which hours are polluted" is the correct
interpretation, and is exactly what a naive floor should show.

**On what this does not say.** These numbers judge one constant on one
year. They are not a verdict on the project, on the dataset, or on what
classical methods can achieve here. **No comparison to M1, M2 or M3 is
possible or attempted** — none of those models exists yet.

---

## Test-set quarantine

**The 2014 partition was not read, not predicted and not evaluated.**

The M0 implementation contains no path to the test target or the test
feature matrix. A token-level audit of both new source files confirmed
**zero executable references** to `X_test`, `y_test`, the test partition
file or 2014 evaluation — every textual occurrence is inside a comment,
docstring or explanatory string. No `y_test.csv`, `test_predictions.csv` or
`m0_test_predictions.csv` exists anywhere in the repository.

The manifest records `test_evaluation_status: not_evaluated` and
`final_test_status: not_evaluated`, with no placeholder 2014 number.

**M0 having no hyperparameters is not a licence to score the test early.**
That argument is tempting and wrong: Protocol Phase 10 evaluates M0–M3
together, once, on the same locked partition, so that no model has seen the
test data more often or earlier than any other. Scoring M0 now would break
that symmetry for no scientific gain.

---

## Final-refit policy

**Documented here; deliberately not executed.**

At Protocol Phase 10:

1. Combine the permitted development targets — 2010–2012 training **plus**
   2013 validation.
2. Recompute the M0 median over the full 2010–2013 development target.
3. Generate 2014 predictions **once**.
4. Evaluate **once** on the locked 2014 target.

**The constant reported in this phase is not automatically the final M0
constant.** It is fitted on 2010–2012 only; the Phase 10 constant will be
fitted on 2010–2013 and will very likely differ.

No combined 2010–2013 target was created in this phase, and no refit was
performed.

---

## Limitations

- **A constant carries no information about individual hours.** M0 cannot
  distinguish a clear afternoon from a severe winter episode. That is the
  point of a naive floor, but it means M0's error is dominated by the
  spread of PM2.5 rather than by any modelling choice.
- **One evaluation year.** These metrics describe 2013 alone. The 2013
  target happens to be centred close to the training target; a year whose
  central tendency had shifted more would have produced a worse MAE
  through no change in the method.
- **The median is a period-specific quantity.** Its transferability across
  periods is an empirical fact about this data, not a guarantee.
- **R² is an awkward fit for a median-optimised constant**, as discussed
  above. It is reported because the protocol pre-registered it, and it is
  reported with that caveat rather than dropped.
- **Development validation, not held-out performance.** Nothing here is a
  final result.

---

## What remains for M1

Protocol Phase 7 fits **M1, Linear Regression**, on the same 2010–2012
training partition using the frozen 43-feature matrix from
[`FEATURE_PREPARATION.md`](FEATURE_PREPARATION.md), and evaluates it on
2013 with the same three metrics.

Nothing about M1 has been implemented. No estimator was imported or
instantiated in this phase, and the M0 result stands alone until M1 exists
to be compared against it — at which point the comparison will still be a
development-validation comparison, not a final one.
