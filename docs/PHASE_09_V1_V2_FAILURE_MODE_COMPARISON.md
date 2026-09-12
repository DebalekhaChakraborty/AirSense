# V1 → V2 Failure-Mode Continuity

## QUALITATIVE / STRUCTURAL COMPARISON

**Classification: POST-TEST EXPLORATORY.** This document compares *failure
modes*, not scores.

## 1. Why no numeric comparison appears here

V1 and V2 are **not directly comparable** benchmark tasks. Reading across them
numerically would be meaningless, so no cross-study performance claim is made
anywhere in this document.

| | AirSense V1 | AirSense V2 |
|---|---|---|
| Task | concurrent-hour PM2.5 estimation from meteorology | future multi-horizon forecasting, h = 1/6/12/24 |
| Design | single-station historical setup | 12-station network |
| Test period | different chronological period | 2016-03-01 → 2017-02-28 |
| Severe rule | actual > **282.0** (2010–2013 development P95) | actual > **244.0** (2013–2015 training pooled P95) |
| Severe hours | 512 | 20,336 sample-rows (5,084 unique hours × 4 horizons) |
| Severe events | 41 episodes, median duration 7 h | 586 events, median duration 4 h |
| Best-known baseline | `M0`, a training-median constant (73.0) | `B0`, causal persistence |

The two severe definitions do not even use the same threshold, so the severe
populations are different populations. A model that estimates the *current*
hour from concurrent weather is solving a different problem from one that
forecasts 24 hours ahead. All values below are resolved from the frozen V1
evidence on the read-only `legacy` branch
(`16c9030cf74508b620cc6d28f90346aa379f29cd`), not from recollection.

## 2. A — Does severe under-prediction remain visible?

**Yes. It is the most persistent pathology in the study.**

| Study | Model | Severe under-prediction rate |
|---|---|---|
| V1 | `M0` constant | 100.0% |
| V1 | `M1` linear | 100.0% |
| V1 | `M2` tree | 94.73% |
| V1 | `M3` random forest | 96.29% |
| V2 | `B3_R2` | 88.60% |
| V2 | `GRU_R1` | 93.73% |
| V2 | `B0` persistence | 72.59% |

Every learned model in both studies under-calls severe pollution the large
majority of the time. In V2 the rate is strongly horizon-dependent: `B3_R2`
rises 63.77% → 91.88% → 98.84% → **99.90%** from h = 1 to h = 24. At a full
day's lead it under-predicts essentially every severe hour.

**Status: unresolved, and structurally the same failure in both studies.**

## 3. B — Does strong residual temporal dependence remain?

**Yes, and it is no weaker.**

| Study | Model | Residual ACF lag 1 | lag 24 |
|---|---|---|---|
| V1 | `M0` | 0.9722 | 0.4388 |
| V1 | `M3` | 0.9033 | 0.3442 |
| V2 (h = 24) | `B3_R2` | 0.9626 | +0.2669 |
| V2 (h = 24) | `GRU_R1` | 0.9500 | −0.2321 |
| V2 (h = 24) | `B0` | 0.9525 | −0.2971 |

Residuals in both studies remain almost perfectly correlated at one hour. Some
structure the models leave on the table is systematic, not noise, and neither
study's models absorb it.

One difference is worth recording. At lag 24 the three V2 models separate by
*sign*: `B3_R2` stays positive (+0.267) while `GRU_R1` and `B0` go negative
(−0.232, −0.297). A positive lag-24 residual correlation is associated with an
error that recurs at the same hour on consecutive days; a negative one with
over/under alternation. This is a descriptive difference in error structure,
not evidence about mechanism.

**Status: unresolved in both studies.**

## 4. C — Does persistence remain difficult to beat during extreme pollution?

**In V2, yes — decisively, and this is the sharpest new finding.**

V1 could not test this. Its task was concurrent-hour estimation, so it had no
persistence baseline at all; its reference was `M0`, a constant. V2 introduced
`B0` causal persistence as a frozen reference benchmark and the locked test
shows it beating both learned models in the severe tail (`B0` 93.22 against
`GRU_R1` 109.33 and `B3_R2` 131.90 µg/m³).

The advantage is concentrated at long horizons. At h = 1 the three are close
(`B0` 26.02, `B3_R2` 26.76); by h = 24 the gap is wide (`B0` 142.53, `GRU_R1`
152.63, `B3_R2` 229.79). In the severe stratum `B0` has the strictly smallest
absolute error on **52.88%** of samples — more than the two learned models
combined.

**Status: newly demonstrated in V2; not testable in V1.**

## 5. D — Which pathology weakened, which is still open?

**Weakened:**

- *Overall skill against a naive reference.* V1's learned models improved on a
  constant baseline; V2's improve on a much stronger one — `B3_R2` beats causal
  persistence at every horizon, by 9.47% at h = 1 and 12.30% at h = 24. Beating
  persistence at 24 hours is a materially harder bar than beating a constant.
- *Blanket under-prediction at all concentrations.* V1's simpler models
  under-predicted severe hours essentially always (100% for `M0` and `M1`). V2's
  `B3_R2` is at 63.77% at h = 1, so at short lead the error is closer to
  symmetric. The pathology is now concentrated at long horizons rather than
  present everywhere.

**Still open:**

- *Severe-tail forecasting.* No learned model in either study forecasts the
  severe tail better than the best available naive rule.
- *Residual autocorrelation.* Lag-1 residual correlation is ~0.95 in V2 and
  ~0.90–0.97 in V1. Unchanged in character.
- *Peak capture.* In V2, at h = 24 `B3_R2` under-predicts the event peak in
  **100.0%** of 586 events and flags the first severe hour in **0.3%** of them.
  `GRU_R1` is better (94.9% / 10.4%) and `B0` better still (88.9% / 16.2%), but
  none is usable as an early warning at that lead.

## 6. What this comparison does not establish

It does not establish that V2's architecture is better than V1's, that the
extra stations or co-pollutants helped, or that any V1 conclusion transfers.
The studies share a pollutant and a city and nothing else that would license
such a claim. What it does show is that two independently conducted studies, on
different tasks and different periods, land on the **same two unresolved
failure modes**: systematic severe under-prediction, and strongly
autocorrelated residuals. That recurrence is the finding worth carrying
forward.
