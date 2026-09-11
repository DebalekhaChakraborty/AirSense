# AirSense V2 — Forecast Semantics

**Protocol Phase 0. Preregistered before any V2 model exists.**

This document defines what a V2 forecast *is*, what information it may use, and
how partitions are assigned. It is the leakage contract: every later dataset,
window builder, model and evaluation must satisfy it.

---

## 1. The task

V2 is **true future-horizon forecasting**.

| | |
|---|---|
| Forecast origin | `t` |
| Horizon | `h` ∈ {1, 6, 12, 24} hours |
| Target | PM2.5 at `t + h`, in µg/m³, at a named station |
| Frozen horizons | 1 h, 6 h, 12 h, 24 h |

This is the decisive break from V1. V1 estimated PM2.5 for hour `τ` from the
meteorology and calendar description *of hour `τ` itself*, used no PM2.5
history, and was therefore concurrent-hour estimation rather than forecasting.
V2 predicts a *future* hour from information that genuinely exists at the
origin.

## 2. Information contract

At origin `t`, a forecast for `t + h` **may** use:

- observations timestamped `<= t`, at the target station;
- past PM2.5 values at the target station (`<= t`);
- past co-pollutant values — PM10, SO2, NO2, CO, O3 (`<= t`);
- past meteorological observations — TEMP, PRES, DEWP, RAIN, wd, WSPM (`<= t`);
- observations from other stations timestamped `<= t`;
- calendar information that is inherently known in advance;
- deterministic calendar features **of the target time** `t + h` — hour of day,
  day of week, month, season. These are knowable at `t` without observing
  anything, so they are not leakage.

It **must never** use:

- observed meteorology after `t`;
- observed pollutants after `t`, at any station;
- the target value PM2.5(`t + h`), or any transform of it;
- interpolation, smoothing or gap-filling that consults future observations;
- centred rolling statistics;
- bidirectional transformations that see future timestamps;
- normalisation, scaling or encoding statistics fitted on data later than the
  training partition;
- any other station's information after `t`.

**Observed future weather is not available to the standard V2 forecast.** V2
does not assume a numerical weather prediction feed exists. A later, explicitly
separate experiment may study forecast-conditioned performance using known
future weather; it is outside the primary protocol and must be labelled as a
distinct study when reported.

## 3. Partition assignment

Partition membership is determined by the **target timestamp** `τ`, never by
the origin.

    origin  = τ - h
    target  = τ
    τ decides the partition

| Partition | Target timestamps `τ` |
|---|---|
| Train | 2013-03-01 00:00 → 2015-02-28 23:00 |
| Validation | 2015-03-01 00:00 → 2016-02-29 23:00 |
| Locked final test | 2016-03-01 00:00 → 2017-02-28 23:00 |

Inputs for a forecast of `τ` may only use timestamps `<= τ - h`.

A consequence worth stating: near the start of a partition, the origin and its
history lie in the *previous* partition. That is correct and intended — it is
how a real forecaster operates on the first day of a new period, and it does
not leak, because the model never sees anything at or after the origin's
future. What must never happen is the reverse: a target from a later partition
influencing anything fitted on an earlier one.

### Worked examples

Target station Dongsi, target `τ` = **2016-06-10 12:00** (validation period):

| Horizon `h` | Forecast origin `t = τ - h` | Latest usable observation | Target |
|---:|---|---|---|
| 1 h | 2016-06-10 11:00 | 2016-06-10 11:00 | PM2.5 @ 2016-06-10 12:00 |
| 6 h | 2016-06-10 06:00 | 2016-06-10 06:00 | PM2.5 @ 2016-06-10 12:00 |
| 12 h | 2016-06-10 00:00 | 2016-06-10 00:00 | PM2.5 @ 2016-06-10 12:00 |
| 24 h | 2016-06-09 12:00 | 2016-06-09 12:00 | PM2.5 @ 2016-06-10 12:00 |

At `h = 24` the model is blind to everything in the 24 hours preceding the
target, including the meteorology of the target hour itself. The same target
therefore appears once per horizon, with four different information sets. Cells
are never pooled across horizons before metrics are computed.

## 4. Rolling-origin evaluation

Validation and final testing are **rolling-origin forecasting without parameter
updating**.

- Model parameters are frozen before the partition is scored and do not change
  during it.
- As the origin advances, observations that have genuinely already occurred
  before the new origin may enter the model's historical context.

Example: forecasting the target 2016-06-10 12:00 from origin 2016-06-10 06:00,
observations up to 06:00 are legitimate context — including observations made
*after* an earlier forecast was issued. This is what a deployed forecaster
does.

This is **not** retraining on the test set. Two distinct things:

| Legitimate | Prohibited |
|---|---|
| Newly observed *past* context entering the input window as the origin advances | Updating weights, hyperparameters or preprocessing statistics using test data |
| Using PM2.5(`t-1`) once `t-1` has occurred | Using PM2.5(`t+h`) in any form |

Weights must remain unchanged across the entire locked-test pass.

## 5. Missing data and the contract

Missingness handling is **not decided in Phase 0** — it is decided in Phase 1
after the missingness audit is reviewed. Whatever is chosen must satisfy this
contract: any imputation, masking or gap-filling used at origin `t` may consult
only timestamps `<= t`. Backward fill, centred interpolation and
whole-series statistics are prohibited by construction, not by convention.

## 6. Sealed final test

PM2.5 values in the 2016-03-01 → 2017-02-28 partition are **sealed**. Before
the locked evaluation they may be counted (present/missing) and their
timestamps verified, but no mean, median, quantile, threshold, distribution,
plot or correlation of those values may be computed, and no model metric may be
produced on them. See `docs/V2_RESEARCH_PROTOCOL.md` §Target discipline.
