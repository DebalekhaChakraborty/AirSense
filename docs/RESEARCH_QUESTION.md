# V1 Research Question — AirSense

Declared in Phase 0, **before any model was fitted and before any
performance figure was observed**.

---

## 1. Primary question

> **How accurately can classical statistical and machine-learning methods
> available in 2019 predict PM2.5 concentration from meteorological and
> temporal information?**

Underlying problem statement:

> Can historical meteorological and air-pollution measurements be used to
> predict PM2.5 concentration using statistical analysis and classical
> machine-learning techniques available by April 2019?

---

## 2. Task definition

**Task type: regression.**

| Element | Definition |
|---|---|
| Target | `pm2.5` — PM2.5 concentration in µg/m³ |
| Unit of observation | one hour at one location (Beijing) |
| Predictors | meteorological and calendar/temporal variables available for that same hour |
| Dataset | UCI Beijing PM2.5, 2010-01-01 to 2014-12-31, 43,824 hourly rows |

The V1 task is **concurrent-hour estimation**: predicting the PM2.5
concentration of an hour from the meteorological and calendar description of
*that same hour*. It is deliberately **not** a forecasting task. No past
PM2.5 value is used as an input, so the model must explain pollution from
weather and time alone.

This distinction matters and is stated up front because the two tasks are
routinely conflated, and because a lagged-PM2.5 model would produce far
better numbers while answering a different question. See
[`FEATURE_POLICY.md`](FEATURE_POLICY.md) §5.

---

## 3. Planned model comparison

Declared in advance. No results exist for any of these.

| ID | Model | Role |
|---|---|---|
| **M0** | Simple baseline predictor | Reference floor. A constant predictor (mean or median of the training target). Establishes the error any real model must beat. |
| **M1** | Linear Regression | Simplest parametric model; tests how much of PM2.5 is linearly explainable by weather and time. |
| **M2** | Decision Tree Regressor | Single tree; captures non-linearity and interactions, at a known risk of overfitting. |
| **M3** | Random Forest Regressor | Ensemble of trees; expected to be the strongest of the four, and the test of whether ensembling repays its interpretability cost. |

All four are available in scikit-learn 0.20.0 and are squarely within the
2019 classical toolkit. No model outside this list is part of the V1
comparison. Prohibited families are listed in
[`HISTORICAL_COMPATIBILITY.md`](HISTORICAL_COMPATIBILITY.md) §5.

M0 is included deliberately. Reporting M1–M3 without a naive floor makes any
error figure uninterpretable.

---

## 4. Planned primary metrics

Declared in advance, in reporting order:

| Metric | Why |
|---|---|
| **MAE** — mean absolute error | Average error in µg/m³, directly interpretable against air-quality thresholds. Robust to the extreme pollution episodes this series contains. |
| **RMSE** — root mean squared error | Penalises large misses. The gap between RMSE and MAE is itself informative about the severe-episode behaviour that matters most here. |
| **R²** — coefficient of determination | Proportion of variance explained; comparable across models on the same split. |

Reported in the same units and on the same split for every model, so the
comparison is like-for-like.

**Implementation note (2019 API constraint).** scikit-learn 0.20.0's
`mean_squared_error` has no `squared=` keyword — that arrived in 0.22 — so
RMSE is computed as an explicit square root of MSE.

**Selection rule, declared now:** the model comparison is decided on
**MAE on the chronological held-out test set**, with RMSE and R² reported
alongside for interpretation. Fixing this before results exist prevents
choosing whichever metric happens to flatter a preferred model.

---

## 5. Planned supporting analysis

Descriptive and diagnostic work planned for Phase 3, before modelling:

1. **Descriptive statistics** — central tendency, spread and range for the
   target and every predictor.
2. **Missing-data analysis** — the 2,067 missing `pm2.5` values are not
   assumed to be randomly distributed; their distribution across years,
   months and hours is examined before any handling rule is chosen.
3. **PM2.5 distribution** — shape, skew and extreme values; whether a
   transformation is warranted.
4. **Temporal pollution patterns** — hour-of-day, day, month and year
   variation.
5. **Feature correlations** — pairwise relationships among predictors and
   with the target; collinearity that would affect linear-model
   interpretation.
6. **Seasonal patterns** — seasonal structure, including the winter heating
   season noted in the dataset's introductory paper.
7. **Meteorological relationships** — how dew point, temperature, pressure,
   wind direction, wind speed, snow and rain relate to PM2.5, particularly
   the dispersal effect of cumulative wind speed.

---

## 6. Evaluation design

The main scientific evaluation uses a **chronological split**: an earlier
period trains, a later period tests. A random split can make evaluation on a
time series artificially optimistic.

The full statement of this rule, its rationale, and the status of the random
split as a **secondary educational comparison only**, is in
[`V1_RESEARCH_PROTOCOL.md`](V1_RESEARCH_PROTOCOL.md) §3. Split boundaries
are set only after Phase 3, and only from the calendar — never by comparing
candidate splits on model performance.

---

## 7. Possible later extension

Classification of PM2.5 into **pollution bands** is a permitted classical
extension. It is not part of the V1 regression comparison, would be reported
separately with classification metrics, and is **not implemented** in this
phase.

---

## 8. Status

**No model has been trained. No metric has been computed. No numerical
result exists in this repository.**

This document is a pre-registration. Its value depends on having been
written before results were seen — so it is not to be edited to match
outcomes. If the plan must change once the data is understood, the change is
recorded as a dated amendment in
[`V1_RESEARCH_PROTOCOL.md`](V1_RESEARCH_PROTOCOL.md) §7, with its reason,
rather than by rewriting this text.
