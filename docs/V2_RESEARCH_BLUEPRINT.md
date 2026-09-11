# AirSense V2 — Research Blueprint

**Modern Temporal and Spatiotemporal Air-Quality Forecasting**
**Protocol Phase 0. Written before any V2 model exists.**

---

## 1. What V2 is trying to predict

PM2.5 concentration (µg/m³) at a named Beijing monitoring station, at a
**future** hour `t + h`, forecast from an origin `t`, for
`h ∈ {1, 6, 12, 24}` hours.

Formally the estimand is `E[PM2.5(station, t + h) | I(t)]`, where `I(t)` is the
information set available at the origin — defined exactly in
[`FORECAST_SEMANTICS.md`](FORECAST_SEMANTICS.md).

## 2. Why V2 is scientifically motivated by V1

AirSense V1 (frozen on branch `legacy`) established, on the single-site 2010–14
Beijing data set, under a strictly chronological design:

| V1 finding | Value |
|---|---|
| Best model | M3 random forest |
| Final held-out MAE | ≈ 48.535 µg/m³ |
| Severe-tail MAE (top 5% hours) | ≈ 160.9 µg/m³ — 3.3× overall |
| Severe-hour underprediction rate | ≈ 96.3% |
| Residual autocorrelation at 1 h lag | ≈ 0.903 |
| PM2.5 history used as input | **none** |

These are **motivation, not baselines.** V1 solved a different problem
(concurrent-hour estimation) on a different data set (single site, 2010–2014).
No V1 number is a V2 baseline, and V1's 2014 test may never be presented as
unseen evidence again.

The inference that drives V2 is the residual autocorrelation. A residual
correlation of ≈ 0.903 at one hour means V1's errors were *predictable from
their own recent history*: the model was leaving a large, structured, and in
principle recoverable signal on the table — precisely the signal that
meteorology and calendar covariates cannot express. The obvious candidate for
that missing structure is the pollutant's own temporal memory, which V1 was
forbidden by design to use.

Two further observations shape V2's scope. V1's failure concentrated in the
high-concentration tail, where it under-predicted almost every severe hour;
severe episodes are the hours anyone actually cares about, so tail reliability
must be a first-class endpoint rather than a footnote. And V1 saw one station,
so it could not use the fact that pollution episodes are regional and advect
across a city — a neighbouring site upwind may carry an hour or more of warning.

## 3. Central research question

> **Can explicit temporal memory and cross-station information improve
> multi-horizon PM2.5 forecasting, particularly during high-concentration
> episodes, while preserving strict chronological generalisation?**

Adopted as given. No change of scientific meaning has been made.

## 4. Dataset

UCI Beijing Multi-Site Air-Quality Data (ID 501, DOI `10.24432/C5RK5G`):
420,768 hourly rows, 12 nationally controlled stations, 2013-03-01 00:00 →
2017-02-28 23:00, six pollutants plus meteorology. Verified from the official
files; full record in [`DATASET_PROVENANCE.md`](DATASET_PROVENANCE.md).

Fitness for V2, in one line each: the hourly grid is complete at every station
(0 missing timestamps, 0 duplicates), which makes causal window construction
clean; it carries PM2.5 history, which V1 lacked; it carries co-pollutants,
which V1 lacked; and it carries 12 sites, which makes the spatiotemporal track
possible at all.

## 5. Information regimes

The scientific decomposition of V2. Each regime adds exactly one class of
information, so any improvement is attributable.

| Regime | Information | Question it answers |
|---|---|---|
| **R0** | Target-station meteorology + calendar history | What can V1's information class do once it is used *causally* for a future horizon? |
| **R1** | R0 + target-station PM2.5 history | Does explicit temporal memory recover V1's autocorrelated residual structure? |
| **R2** | R1 + target-station co-pollutant history (PM10, SO2, NO2, CO, O3) | Does multivariate pollutant memory add beyond PM2.5's own past? |
| **R3** | R2 + cross-station history (the other 11 sites) | Does spatial context add beyond single-site temporal history? |

R0 is deliberately the closest V2 analogue to V1's information class, which
makes the V1 → V2 transition measurable rather than rhetorical — though it is
still not a like-for-like comparison, because the task and data set differ.

These regimes must not be collapsed into one model. The ablation *is* the
result.

## 6. Preregistered hypotheses

| ID | Hypothesis |
|---|---|
| **H1** | Adding causal PM2.5 history (R1) reduces future-horizon MAE relative to the meteorology/calendar-only regime (R0). |
| **H2** | The benefit of PM2.5 history is largest at short horizons, consistent with V1's strong short-lag residual autocorrelation, and decays as `h` grows. |
| **H3** | Adding co-pollutant history (R2) gives incremental improvement beyond PM2.5 history alone. |
| **H4** | Cross-station context (R3) gives additional benefit beyond single-station temporal history. |
| **H5** | Modern temporal and spatiotemporal models still degrade in the high-concentration tail, but a successful V2 model reduces the severe-tail error and the underprediction rate seen in V1. |

These are hypotheses, not expected conclusions. A refuted hypothesis is a
result and will be reported as one. In particular, H1–H4 may fail at long
horizons, and H5 is deliberately phrased so that "the tail is still bad" is a
publishable outcome.

## 7. Evaluation design

**Primary selection metric: macro station-horizon MAE.**

1. Compute MAE independently for each (station × horizon) cell — up to
   12 × 4 = 48 cells.
2. Average the valid cell MAEs with equal weight.

Equal weighting is the point: station-level target missingness already ranges
from 1.09% to 2.72%, so after any eligibility mask the cells will hold
different row counts. A micro average would silently let the most complete
stations set the score.

Secondary, always reported: micro MAE, horizon-wise MAE, station-wise MAE,
RMSE, and R² where meaningful.

The exact eligible-row mask is **not** fixed in Phase 0 — it depends on the
missingness policy, which is Phase 1 work.

## 8. Severe-tail endpoint

Upper-tail reliability is a first-class endpoint, declared before results.

- **V1's 282 µg/m³ threshold is not reused.** Different network, different
  period, different sampling.
- The primary threshold will be the **pooled training-period PM2.5 P95**,
  derived from the training partition only.
- A station-specific training P95 is the planned robustness variant.
- The numeric threshold is **not computed in Phase 0** — doing so would turn
  the dataset audit into target EDA before the policy is reviewed. It is frozen
  in the pre-model phase.

Planned tail measures: severe-hour MAE, severe-hour RMSE, mean residual
(forecast bias), underprediction percentage, and peak magnitude error.
Event-oriented metrics (recall, precision, onset timing error) may be added
only if their definitions are preregistered before any model result.

## 9. Baselines

V2 must clear simple forecasting baselines before any modern claim is credible.

| ID | Baseline |
|---|---|
| **B0** | Persistence — the latest observation at or before the origin |
| **B1** | Seasonal persistence — the value 24 h before the target, where appropriate |
| **B2** | Training climatology — calendar-conditioned mean/median from training only |
| **B3** | Gradient-boosted lag-feature model on causal features |

Persistence at `h = 1` on hourly PM2.5 is a genuinely strong forecaster. Any V2
model that fails to beat the appropriate baseline at a given horizon will be
reported as failing, not quietly dropped. Missingness-aware persistence
semantics (what "latest observation" means when the last hours are `NA`) are
deliberately deferred to Phase 1.

## 10. Model families under consideration

Shortlist and trade-offs: [`MODEL_LANDSCAPE_2026.md`](MODEL_LANDSCAPE_2026.md).
The roster is **not** frozen in Phase 0. It will be small and purposeful — each
model earns its place by answering a question, not by being available.

## 11. Two tracks

**Track A (primary)** — multi-horizon temporal forecasting across the 12 known
stations. Regimes R0–R2, plus R3 as a modelled input.

**Track B (secondary)** — spatial generalisation: graph and graph-temporal
networks, cross-station attention, leave-one-station-out transfer, station
embeddings. **Not implemented yet.** The held-out station will be chosen on
geographic and structural grounds and frozen *before* Track-B experiments, never
selected by model result.

## 12. What Phase 0 freezes, and what it does not

**Frozen now:** the task; horizons 1/6/12/24; the forecast-origin contract;
target-timestamp partition assignment; the three chronological boundaries; the
information-regime decomposition R0–R3; the hypotheses; the primary metric
*form*; the severe-threshold *definition* (training-only P95); the baseline
family; raw dataset identity; target-discipline rules.

**Deliberately not frozen:** missingness and imputation policy; the eligible-row
mask; context-window lengths; the numeric severe threshold; the model roster;
the V2 Python environment; Track-B design; whether a random-split secondary
comparison is worth running at all.
