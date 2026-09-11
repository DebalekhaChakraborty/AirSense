# AirSense V2 — Classical Baseline Results

> ## DEVELOPMENT VALIDATION — NOT FINAL TEST
>
> Every number on this page comes from the **2015-03-01 → 2016-02-29
> development validation** partition. The locked 2016-03-01 → 2017-02-28 test
> remains **sealed**: no test target was read, no test prediction was
> generated, no test metric exists. These are development figures used to
> select among predeclared candidates, and they are **not** V2 performance.

**Protocol Phase 3.** Primary metric: macro station-horizon MAE — MAE computed
independently in each of 48 (station × horizon) cells, averaged with equal
weight. Residual convention: `residual = actual − prediction`, so a positive
residual is an under-prediction. 413,148 validation samples; every model
predicted every one of them.

---

## 1. Primary comparison

| Model | macro station-horizon MAE | micro MAE | macro RMSE | macro R² | Rank |
|---|---:|---:|---:|---:|---:|
| **B3_R2** | **30.111722** | 30.114311 | 50.782684 | 0.543162 | **1** |
| B3_R1 | 30.472594 | 30.475081 | 51.193995 | 0.534980 | 2 |
| B0 persistence | 33.127699 | 33.131272 | 57.654653 | 0.404175 | 3 |
| B3_R0 | 40.623282 | 40.626598 | 63.315691 | 0.376350 | 4 |
| B1 seasonal naive | 51.205937 | 51.213139 | 85.073764 | −0.101770 | 5 |
| B2 climatology | 56.733945 | 56.741553 | 85.168816 | −0.101508 | 6 |

Ranking is by the pre-registered primary metric alone. Secondary metrics agree
here, but would not have been allowed to change the order.

**The most important line in this table is the fourth one.** A modern
gradient-boosted model with 207 engineered meteorological and calendar
features (B3_R0) is **worse than repeating the last observation** — 40.62
against 33.13, a 22.6% deficit. Everything the boosting machinery buys over
persistence comes from PM2.5's own history, not from the meteorology.

## 2. By horizon

Macro-station MAE, µg/m³:

| Model | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---:|---:|---:|---:|
| B0 persistence | **9.901** | 30.106 | 40.882 | 51.621 |
| B1 seasonal naive | 50.820 | 51.101 | 51.282 | 51.621 |
| B2 climatology | 56.734 | 56.734 | 56.734 | 56.734 |
| B3_R0 | 31.951 | 36.117 | 42.846 | 51.578 |
| B3_R1 | 9.245 | 26.713 | 37.298 | 48.635 |
| **B3_R2** | **9.022** | **26.457** | **36.763** | **48.204** |

The task hardens steeply with horizon: the best model's error rises 5.3× from
h = 1 to h = 24. Persistence at h = 1 (9.90) is close to unbeatable — the best
boosted model improves on it by 8.9%, which is real but modest against the
0.965 lag-1 autocorrelation Phase 1 measured. By h = 24 persistence has
degraded to 51.62 and the boosted models pull 6.6% ahead.

B2 is horizon-independent by construction: it uses no observation at all.

## 3. Information-regime increments

| Increment | Scope | From | To | Δ MAE | Relative |
|---|---|---:|---:|---:|---:|
| **R0 → R1** (add PM2.5 history) | overall | 40.623 | 30.473 | 10.151 | **25.0%** |
| | h = 1 | 31.951 | 9.245 | 22.707 | **71.1%** |
| | h = 6 | 36.117 | 26.713 | 9.404 | 26.0% |
| | h = 12 | 42.846 | 37.298 | 5.548 | 12.9% |
| | h = 24 | 51.578 | 48.635 | 2.943 | 5.7% |
| | severe | 176.973 | 138.851 | 38.123 | 21.5% |
| **R1 → R2** (add co-pollutants) | overall | 30.473 | 30.112 | 0.361 | **1.2%** |
| | h = 1 | 9.245 | 9.022 | 0.222 | 2.4% |
| | h = 6 | 26.713 | 26.457 | 0.256 | 1.0% |
| | h = 12 | 37.298 | 36.763 | 0.535 | 1.4% |
| | h = 24 | 48.635 | 48.204 | 0.430 | 0.9% |
| | severe | 138.851 | 136.966 | 1.884 | 1.4% |

The two increments differ by more than an order of magnitude. Adding the
target's own history is transformative; adding five co-pollutant histories on
top is a consistent but small refinement — positive at every horizon and on the
severe subset, never large.

No causal claim is made. These are associations measured on one development
partition.

## 4. Severe endpoint — where the classical models fail

Frozen threshold: **actual PM2.5 > 244.0 µg/m³**, derived in Phase 1 from the
pooled training P95 and never recomputed. **18,636** validation samples qualify
(4.51%), and all 48 cells contribute — no cell was empty, so nothing was
silently dropped from the denominator.

| Model | severe MAE | severe RMSE | mean residual | under-prediction | macro severe MAE |
|---|---:|---:|---:|---:|---:|
| **B0 persistence** | **101.273** | **152.283** | 70.344 | **71.6%** | **102.247** |
| B3_R2 | 136.612 | 188.469 | 132.018 | 88.3% | 136.966 |
| B3_R1 | 138.325 | 189.878 | 133.729 | 88.5% | 138.851 |
| B1 | 152.589 | 197.952 | 126.956 | 85.9% | 156.259 |
| B3_R0 | 176.945 | 216.103 | 175.348 | 97.0% | 176.973 |
| B2 | 278.796 | 299.027 | 278.796 | 100.0% | 279.012 |

**Persistence wins the severe tail outright, and every boosted model loses to
it.** B3_R2 is 34.9% worse than B0 on severe hours despite being 9.1% better
overall. The mean residual tells the story: B3_R2 under-shoots severe hours by
132 µg/m³ on average and under-predicts 88.3% of them, while persistence —
which simply repeats a recent, often already-elevated observation — under-shoots
by 70 and under-predicts 71.6%.

This is the same failure mode V1 documented, reproduced in a different data set
under a genuinely predictive task. A squared-or-absolute-error objective fitted
across 205k mostly-ordinary hours has little incentive to chase the tail, and
regression toward the middle is the result. **No V2 model so far reduces the
severe-tail error that motivated the whole programme.**

## 5. Baseline diagnostics

**B0 does not lean on its fallback.** The observation at the origin was
available for 98.26–99.22% of samples depending on horizon; the causal carry
covered 0.75–1.43%, and the station training median was needed for only
0.03–0.31%. Persistence's strength is genuine, not an artefact of the fallback
chain. Per station, the observed-at-origin share ranges 98.12% (Wanshouxigong)
to 99.02% (Guanyuan).

**B1 collapses into B0 at h = 24, in 100.00% of samples** — 103,287 of 103,287,
exactly as the arithmetic predicts (`τ − 24 h == t`). At shorter horizons it
used the direct τ−24 observation for 98.26% of samples and coincided with B0 by
chance in only 3.4–3.8%. Reported rather than hidden: at h = 24, B1 is not an
independent baseline.

**B2 never needed a fallback.** All 413,148 predictions came from the finest
level of the hierarchy, station × month × hour, so the training partition
supported every calendar cell the validation period asked for.

## 6. Negative predictions — reported, not clipped

| Model | count | % | minimum |
|---|---:|---:|---:|
| B3_R0 | 259 | 0.063% | −16.02 |
| B3_R1 | 146 | 0.035% | −24.88 |
| B3_R2 | 63 | 0.015% | −32.56 |
| B0 / B1 / B2 | 0 | 0% | ≥ 3.0 |

Physically impossible for a concentration, and **not clipped**: clipping would
improve the metric after the fact. The tree ensembles produce them because a
leaf value is a fitted L1 optimum, not a constrained one. The counts are tiny
and shrink as the regime gains information.

## 7. Per-station behaviour

B3_R2 beats B0 at **every one of the 12 stations**, by 1.43 (Dingling) to 4.57
(Shunyi) µg/m³ of macro-over-horizon MAE. Cleaner suburban sites — Dingling
25.74, Changping 26.09, Huairou 26.26 — carry lower error than central sites
such as Wanshouxigong 32.94 and Dongsi 32.50, consistent with the station level
differences Phase 1 measured. Stations are not ranked as "good" or "bad": the
differences track local concentration levels, not station quality.

## 8. Selected B3 candidates

| Regime | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---|---|---|---|
| R0 | B3_C03 | B3_C04 | B3_C03 | B3_C02 |
| R1 | B3_C07 | B3_C02 | B3_C02 | B3_C02 |
| R2 | B3_C07 | B3_C02 | B3_C01 | B3_C01 |

Selected independently per regime × horizon on macro-station validation MAE,
under the frozen tie-break order. All 96 candidate results are retained in
`results/models/b3/b3_candidate_metrics.csv` — none is hidden.

A pattern worth noting for later phases: the shorter horizons prefer the larger
`num_leaves = 63` candidates, while h = 12 and h = 24 prefer smaller trees and
the smallest learning rate — longer horizons carry less exploitable structure
and reward more regularisation.

## 9. Limitations

**Selection optimism.** Eight candidates were scored on validation and the best
was kept, independently in each of 12 cells. The selected values are therefore
optimistic estimates of held-out performance, in the same way V1's Phase 8–9
selections were. Only the locked test can settle that, and it stays sealed.

**Development, not evidence of generalisation.** One validation year, one city,
one network. Nothing here has survived a held-out test.

**The severe tail is unsolved**, and the model that wins overall is the one that
loses most conspicuously where it matters.

**No causal claim.** Feature importance is a descriptive model diagnostic, not
an attribution of physical cause.

## 10. Hypothesis status on development validation

| | Statement | Status |
|---|---|---|
| **H1** | Causal PM2.5 history reduces future-horizon MAE relative to meteorology/calendar only | **supported on development validation** — 25.0% overall |
| **H2** | The benefit is largest at short horizons | **supported on development validation** — 71.1% at h = 1 decaying monotonically to 5.7% at h = 24 |
| **H3** | Co-pollutant history adds incremental value beyond PM2.5 history | **weakly supported on development validation** — 1.2% overall, positive at every horizon and on the severe subset, but an order of magnitude smaller than H1 |
| H4 | Cross-station context adds further benefit | **not evaluated** — R3 is not implemented in Phase 3 |

No hypothesis is confirmed. Confirmation is reserved for the locked test, which
has not been opened.
