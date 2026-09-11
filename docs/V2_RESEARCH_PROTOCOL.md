# AirSense V2 — Research Protocol

**Preregistration. Protocol Phase 0, written before any V2 model exists.**

This document governs. Where prose elsewhere disagrees with it, this document
and the machine-readable `configs/study.json` win.

---

## 1. Study identity

| | |
|---|---|
| Study | AirSense V2 — Modern Temporal and Spatiotemporal Air-Quality Forecasting |
| Task | `multi_horizon_pm25_forecasting` |
| Branch | `master` |
| V1 | Complete and frozen on `legacy`; read-only, never modified |
| Data set | UCI Beijing Multi-Site Air-Quality Data (ID 501, DOI `10.24432/C5RK5G`) |
| Horizons | 1, 6, 12, 24 hours |
| Primary metric (provisional) | Macro station-horizon MAE |
| Final test | **Sealed** |

## 2. Chronological partitions

Complete March-to-February annual cycles, assigned by **target timestamp**:

| Partition | From | To |
|---|---|---|
| Train | 2013-03-01 00:00 | 2015-02-28 23:00 |
| Validation | 2015-03-01 00:00 | 2016-02-29 23:00 |
| Locked final test | 2016-03-01 00:00 | 2017-02-28 23:00 |

Rationale, all of it calendar-based and none of it performance-based:

- every training timestamp precedes every validation timestamp, which precedes
  every test timestamp;
- validation and test each span one complete seasonal cycle, so neither is
  dominated by a single season — decisive for a pollutant with a severe winter
  heating signature;
- the 2016-02-29 leap day falls naturally inside validation;
- the boundaries were fixed from the calendar before any model existed.

**Candidate split boundaries were not, and will not be, compared using model
performance.**

## 3. Forecast semantics

Defined in [`FORECAST_SEMANTICS.md`](FORECAST_SEMANTICS.md) and incorporated
here by reference. Summary of the binding rules:

- forecast origin `t`, horizon `h`, target PM2.5 at `t + h`;
- inputs may use only timestamps `<= t`;
- deterministic calendar features of the target time are permitted;
- observed future meteorology is **not** available;
- evaluation is rolling-origin **without parameter updating**;
- partition membership follows the target timestamp.

## 4. Target discipline

PM2.5 is the target. The test partition's target values are **sealed** from now
until the locked evaluation.

**Permitted at any time** — integrity operations that do not read target values
as data: hashing files, counting rows, verifying schema, verifying the
timestamp grid, counting present/missing PM2.5, counting non-finite or
impossible values.

**Prohibited before the locked evaluation**, on the test partition: mean,
median, quantiles, standard deviation, minimum, maximum, any severe threshold,
any target distribution plot, any target correlation, and any model metric.

**Training-period** target values may be analysed from Phase 1 onward.
**Validation** target values may be used only once the protocol permits model
selection. Phase 0 computed **no PM2.5 value statistic in any partition** —
`artifacts/dataset_audit.json` records
`"target_value_statistics_computed": false`.

After the locked test is opened, its target values may be inspected for
post-test error and shift analysis.

## 5. Phase sequence

| Phase | Content | Gate |
|---:|---|---|
| **0** | Foundation, dataset audit, research blueprint | No model, no metric, no target statistic |
| **1** | Training-only EDA; missingness and preprocessing policy | Training partition only |
| **2** | Forecast dataset and causal window construction; baseline definitions | Windows must satisfy the origin contract |
| **3** | Classical forecasting baselines (B0–B3) | Validation only |
| **4** | Temporal neural baselines (recurrent, convolutional) | Validation only |
| **5** | Modern Transformer and pretrained TSFM evaluation | Validation only |
| **6** | Spatiotemporal modelling (Track B) | Held-out station frozen in advance |
| **7** | Ablation and robustness across regimes R0–R3 | Validation only |
| **8** | Pre-test freeze: models, features, preprocessing, metric, threshold all sealed | Nothing may change afterwards |
| **9** | Locked final-test evaluation — opened once, scored once | Single pass, frozen weights |
| **10** | Post-test error analysis | Descriptive only; no model change |
| **11** | Final V2 report and evidence freeze | Complete |

This sequence may be refined **now**, before any result exists. After
Experiment 001 begins, every scientific amendment must be recorded explicitly,
with its reason and its date, in the phase record — never applied silently.

## 6. Preprocessing principles (frozen now; numbers deferred)

Principles are binding from this point. Numeric parameters are Phase 1/2 work.

1. All learned preprocessing — scaling, encoding, imputation statistics,
   embeddings — is fitted on **training data only**.
2. No target leakage of any kind.
3. No future-aware imputation. Any gap-filling at origin `t` may consult only
   timestamps `<= t`.
4. No centred rolling statistics. Trailing windows only.
5. No normalisation that uses validation or test data.
6. Timestamps stay explicit and are never inferred from row position.
7. Missingness masks are preserved where useful rather than silently filled.
8. Station identity must not encode partition or future information.
9. Window construction obeys the forecast-origin contract by construction.
10. Final-test target values remain sealed until Phase 9.

Specifically **not** decided in Phase 0: forward-fill vs interpolation vs
masking; maximum tolerable gap; window dropping; scaling family; context
length. Phase 1 decides these *after* the missingness audit is reviewed, and
records the reasoning.

## 7. Evaluation

Primary: **macro station-horizon MAE** — MAE per (station × horizon) cell, up
to 48 cells, averaged with equal weight. Secondary: micro MAE, horizon-wise
MAE, station-wise MAE, RMSE, R² where meaningful.

Severe tail: threshold derived from the **training** target distribution only
(pooled P95 primary, station-specific P95 robustness), frozen before Phase 3;
measures are severe-hour MAE, severe-hour RMSE, mean residual, underprediction
percentage, peak magnitude error.

Baselines B0–B3 as defined in the blueprint. A model that does not beat the
appropriate baseline at a horizon is reported as not beating it.

## 8. Model selection principle

The roster stays small and purposeful; each model must answer a stated research
question. V2 is not a survey of everything downloadable. Selection is decided
by the research question, hardware feasibility, and licence — never by
searching for whichever model scored best on the test set.

The exact roster is **not** frozen in Phase 0. It is frozen after the model
landscape, hardware plan, missingness policy and licence review are accepted.

## 9. Prohibited practices

- Reusing V1's 2014 test as unseen evidence.
- Copying V1 model implementations into V2 for continuity.
- Selecting split boundaries, thresholds, eligibility masks or model rosters by
  performance.
- Computing any test metric before Phase 9.
- Changing any model after the Phase 8 freeze.
- Reporting a horizon, station or regime selectively because it looks better.
- Asserting model-landscape facts that were not verified against an
  authoritative source.

## 10. Amendment rule

Before Experiment 001: this protocol may be edited freely, since no result
exists to bias the edit. From Experiment 001 onward: amendments must be
appended with date and reason, and the original text must remain legible.
