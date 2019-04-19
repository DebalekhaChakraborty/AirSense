# AirSense V1 Final Error Analysis

**Research-protocol mapping: Protocol Phase 11 — Error Analysis.**

> **This is post-evaluation characterisation, not model development.** The
> 2014 test was consumed in Phase 10. Nothing here changed a model, a
> prediction or a metric, and nothing here may.

---

## Purpose

Answer *where* the four models fail, now that the final ranking is settled.

The Phase-10 result — M3 best by MAE at 48.535 µg/m³ — says how much error
there is. It says nothing about *which hours* carry it. This phase locates
the error in season, time of day, concentration level, wind regime and
time-structure, so the final report can state honestly what V1 does and does
not do.

## Post-test status

The final test is **exhausted**. Every number below is derived from
artifacts frozen before or during Phase 10:

- predictions from `results/final_test/final_test_predictions.csv`,
  verified string-identical to the pre-opening blind vectors;
- actual PM2.5 from `results/final_test/final_test_target_snapshot.csv`;
- context from the frozen `X_test.csv` and the row manifest.

**`data/processed/airsense_test_2014.csv` was not reopened.** A guard in the
analysis module aborts if any code path reaches it, and a token-level audit
confirmed zero executable references.

**No estimator was imported. No `.fit()`. No `.predict()`.** The manifest
records `models_fitted: 0`, `predictions_generated: 0`,
`model_modification_status: none`.

## Frozen final models

M0 training-median constant (73.0); M1 ordinary least squares; M2 decision
tree (`max_depth=12`, `min_samples_leaf=10`); M3 random forest (100 trees,
`max_depth=12`, `min_samples_leaf=1`, `max_features="auto"`). All refitted
on 2010–2013, all scored on the same 8,661 observations.

## Analysis population

**8,661 hours of 2014**, identical for every model and every subgroup. Season
and hour reconstructed from the timestamp; wind direction reconstructed from
the frozen one-hot block (`cbwd_NW`/`cbwd_SE`/`cbwd_cv`, all-zero → the
omitted `NE` reference), verified mutually exclusive.

---

## Development-derived pollution thresholds

**Frozen from the 2010–2013 development target (33,096 observations) before
any 2014 subgroup metric was computed.** Deriving bands from the test
distribution would let the evaluation data shape its own analysis.

| Quantile | Value (µg/m³) |
|---|---:|
| P25 | 29 |
| P50 | 73 |
| P75 | 138 |
| P90 | 221 |
| **P95** | **282** |

Bands: `le_p25` (≤29), `p25_p50` (29–73], `p50_p75` (73–138],
`p75_p90` (138–221], `p90_p95` (221–282], `gt_p95` (>282).

These are **descriptive relative bands**, deliberately named neutrally. They
are **not** regulatory AQI categories and are not labelled good, moderate,
unhealthy or hazardous — no external regulatory definition was consulted.

---

## Absolute-error distributions

| Model | Mean | Median | Std | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 65.417 | 49.000 | 71.275 | 67.000 | 148.000 | 235.000 | 349.400 | 598.000 |
| M1 | 52.900 | 40.180 | 50.458 | 68.675 | **108.113** | **150.416** | 241.152 | 520.049 |
| M2 | 51.249 | 35.374 | 55.475 | 65.484 | 118.017 | 163.750 | 270.321 | 515.544 |
| **M3** | **48.535** | **34.220** | 50.829 | **62.731** | 109.561 | 153.127 | **240.386** | **504.744** |

M3 is best on the mean, the median, the 75th percentile, the 99th and the
maximum. **M1 is marginally better at the 90th and 95th percentiles**
(108.11 vs 109.56; 150.42 vs 153.13) — its mid-upper error tail is very
slightly tighter than M3's despite a clearly worse typical error. Recorded
because it is true, not because it changes anything: the pre-registered
metric is the mean, and on the mean M3 wins by 4.4 µg/m³.

The median absolute errors are far below the means for every model — 34.2
against 48.5 for M3 — so a minority of hours carries a disproportionate
share of the total error. The rest of this analysis identifies them.

## Bias and under/overprediction

Residual = actual − predicted, so **positive is underprediction**.

| Model | Under | Over | Exact | Mean resid. | Median resid. | Negative predictions |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 49.14% | 50.41% | 39 | +24.735 | −1.000 | 0 |
| M1 | 39.88% | 60.12% | 0 | +0.073 | −14.560 | **466 (5.38%)** |
| M2 | 38.04% | 61.88% | 7 | −2.810 | −9.810 | 0 |
| M3 | 36.44% | 63.56% | 0 | −2.845 | −11.568 | 0 |

**All three feature-based models over-predict the majority of hours while
their mean residual sits near zero.** M3 over-predicts 63.6% of hours, yet
its mean residual is only −2.8 µg/m³. Both facts hold because the many small
over-predictions are offset by fewer, far larger under-predictions. This is
the arithmetic signature of a model regressing toward the middle of a
right-skewed distribution.

**M0's +24.7 mean residual with a −1.0 median** is the same phenomenon in
its purest form: a constant chosen to minimise absolute error sits near the
distribution's centre and is badly wrong about its tail.

### Negative M1 predictions

M1 produced **466 physically impossible predictions (5.38%)**, minimum
**−162.14 µg/m³**. On those rows the actual PM2.5 averaged 21.2 µg/m³ and
M1's MAE was **58.99** — worse than its overall 52.90, so these are not
harmless. Unconstrained least squares on a non-negative target does this.

**They were not clipped.** Clipping now would be a post-test model
modification. Recorded as a V1 limitation.

---

## Seasonal error patterns

| Model | Winter | Spring | Summer | Autumn |
|---|---:|---:|---:|---:|
| M0 | 92.872 | 53.510 | 42.666 | 73.457 |
| M1 | 67.717 | 48.753 | 38.246 | 57.305 |
| M2 | 66.239 | 44.264 | 38.567 | 56.392 |
| **M3** | **62.738** | **39.624** | **38.014** | **54.241** |

**M3 has the lowest MAE in all four seasons**, so its advantage is broad
rather than concentrated in one part of the year.

**Winter is hardest for every model** — M3's winter MAE (62.7) is 65% higher
than its summer MAE (38.0). Phase 3 found winter has both the highest mean
and by far the largest spread, and the error follows the spread.

Mean residuals flip sign by season: all models under-predict in winter
(M3 +17.8) and over-predict in summer (M3 −15.6). No causal mechanism is
asserted; this is an observed association.

## Hour-of-day error patterns

| Model | Best hour | Worst hour | Spread |
|---|---|---|---:|
| M0 | 61.080 (h14) | 72.912 (h01) | 11.831 |
| M1 | 46.678 (h14) | 62.426 (h01) | 15.748 |
| M2 | 41.714 (h15) | 60.308 (h01) | 18.594 |
| M3 | **39.634 (h15)** | **56.281 (h01)** | 16.647 |

**The error retains the diurnal structure Phase 3 identified in PM2.5
itself** — worst overnight, best mid-afternoon, the same shape as the
concentration cycle. The models have not removed the diurnal pattern; they
have inherited it.

M3 is best at every hour. Its 16.6 µg/m³ spread across the day is roughly a
third of its overall MAE.

**No hour feature was added or re-encoded in response.**

---

## Error by pollution concentration

**This is the dominant structure in the whole analysis.**

| Model | `le_p25` | `p25_p50` | `p50_p75` | `p75_p90` | `p90_p95` | `gt_p95` |
|---|---:|---:|---:|---:|---:|---:|
| n | 2,212 | 2,193 | 2,238 | 1,160 | 346 | 512 |
| mean actual | 15.5 | 50.7 | 101.7 | 171.9 | 249.4 | 367.0 |
| M0 MAE | 57.490 | 22.348 | 28.664 | 98.882 | 176.382 | 293.990 |
| M1 MAE | 45.444 | 45.398 | 33.566 | 42.820 | 97.169 | 194.676 |
| M2 MAE | 32.171 | 40.187 | 40.553 | 63.945 | 102.712 | 164.260 |
| **M3 MAE** | **33.333** | **39.038** | **34.373** | **59.597** | **94.174** | **160.897** |

**M3's mean residual by band:** −33.1, −34.7, −9.2, +30.5, +87.2, **+155.9**.

**Every model over-predicts low concentrations and under-predicts high
ones, monotonically.** M3's error is nearly five times larger in the top
band (160.9) than in the bottom (33.3), and its under-prediction rate climbs
from 4.1% in `le_p25` to **96.3%** in `gt_p95`.

This is textbook regression toward the conditional mean, and it is not a bug
in any model — it is what squared-error-fitted models do on a skewed target
when the predictors do not separate the extremes. **It is, however, the
central practical limitation of V1**: the hours a user would most want
predicted are the hours the models get most wrong.

## Wind-direction error patterns

| Model | `NE` (993) | `NW` (2,426) | `SE` (3,186) | `cv` (2,056) |
|---|---:|---:|---:|---:|
| M0 | 67.053 | 69.070 | 55.093 | 76.317 |
| M1 | 52.518 | 53.683 | 46.847 | 61.541 |
| M2 | 51.890 | 46.906 | 48.984 | 59.572 |
| **M3** | **49.828** | **43.096** | **45.904** | **58.407** |

**`cv` is the hardest regime for every model** and `NW` the easiest for M3.
Phase 3 found `cv` carried the highest median PM2.5 (98 µg/m³) and `NW` the
lowest (31), so error again tracks concentration level.

Descriptive only. **No causal interpretation of wind direction is made.**

---

## Severe high-concentration hours

**512 test hours exceed the development P95 of 282 µg/m³** (5.9% of the test
set). This is a relative, study-specific tail definition — **not** a
regulatory severe-pollution threshold.

| Model | MAE | RMSE | Mean resid. | Underprediction | Max abs. error |
|---|---:|---:|---:|---:|---:|
| M0 | 293.990 | 302.226 | +293.990 | **100.0%** | 598.0 |
| M1 | 194.676 | 206.806 | +194.676 | **100.0%** | 520.0 |
| M2 | 164.260 | 188.734 | +157.839 | 94.7% | 515.5 |
| **M3** | **160.897** | **181.215** | +155.939 | 96.3% | **504.7** |

**Every model fails badly here.** M3 is the best of four, and its MAE in the
tail (160.9) is still **3.3 times its overall MAE** (48.5). It
under-predicts 96.3% of severe hours by an average of 156 µg/m³.

M0 and M1 under-predict **100%** of severe hours — M0 necessarily, being a
constant below the threshold; M1 because a linear surface cannot reach the
tail.

**No severe-event model exists and none was created.** These metrics are not
compared to a separately tuned tail model, and no event-detection or
classification claim is made: these models predict a concentration, not an
event class.

## Severe episode behaviour

Under the frozen definition — consecutive test timestamps exactly one hour
apart, all above the development P95, with gaps never bridged:

| Property | Value |
|---|---:|
| Episodes | **41** |
| Severe hours | 512 |
| Shortest episode | 1 hour |
| Median | 7 hours |
| Mean | 12.5 hours |
| **Longest** | **76 hours** |

Verified: every episode contains only severe hours; consecutive rows within
an episode are exactly one hour apart; episodes do not overlap; every severe
hour belongs to exactly one episode; no sub-threshold hour was included; and
episode hours sum to 512.

The longest episode ran **76 consecutive hours** — over three days of
sustained concentrations above 282 µg/m³. A model whose errors persist
across such a stretch is wrong for days at a time, not for isolated hours.

*Table:* `results/error_analysis/severe_episode_details.csv` — all 41
episodes, none selected manually.

---

## Residual autocorrelation

Computed by **exact timestamp matching**: residual at *t* is paired with
residual at *t + lag* only when that hour is actually observed. Row shifting
was deliberately not used — the supervised test set holds 8,661 of 8,760
hours, so an offset would have paired hours separated by a gap and corrupted
every lag.

| Model | 1 h | 6 h | 12 h | 24 h | 48 h | 168 h |
|---|---:|---:|---:|---:|---:|---:|
| n pairs | 8,625 | 8,582 | 8,556 | 8,542 | 8,519 | 8,394 |
| M0 | 0.972 | 0.785 | 0.614 | 0.439 | 0.139 | 0.076 |
| M1 | 0.942 | 0.699 | 0.530 | 0.432 | 0.222 | 0.089 |
| M2 | 0.845 | 0.518 | 0.381 | 0.321 | 0.207 | 0.094 |
| M3 | 0.903 | 0.592 | 0.438 | 0.344 | 0.193 | 0.103 |

Every lag has ample pairs (≥ 8,394); none is too sparse to interpret.

**Errors remain strongly temporally structured.** M3's residuals correlate
at **0.903 at one hour**, still 0.344 at a full day, and remain weakly
positive at a week. If a model's error this hour is known, its error next
hour is largely known too.

Stated carefully: this indicates **substantial predictable structure that
the concurrent-hour specification does not capture**. It does **not** mean
any model is invalid, and no specific causal mechanism is inferred. The
V1 task was deliberately defined as concurrent-hour estimation with no
lagged target — that choice was frozen in Phase 5, and this is the cost of
it, now measured.

**This motivates dedicated time-series modelling as future work.** **No V1
lag feature may be added**, and none was.

M2 shows the *lowest* one-hour autocorrelation (0.845) despite being the
weaker model, plausibly because its coarse piecewise-constant output
decorrelates adjacent errors slightly. Noted as an observation, not a
virtue.

---

## M3 versus M2

| Comparison | M3 lower error | M2 lower error | Ties |
|---|---:|---:|---:|
| Row-wise absolute error | **4,623 (53.4%)** | 4,038 (46.6%) | 0 |

Mean per-row difference **−2.71 µg/m³**; median **−1.03**.

**M3's advantage is narrow in frequency but real in magnitude.** It wins on
only 53.4% of hours — barely more than half — yet its mean error is 2.7
µg/m³ lower. The advantage comes from winning *larger*, not winning *more
often*: when M3 is better it is better by more than M2 is when M2 is better.

Full pairwise table: M3 beats M1 on 56.2% of rows and M0 on 62.6%.

**This is descriptive.** It is not a new selection criterion, and it must
not be used to build a per-row selector or a hybrid model. The final ranking
remains the pre-registered overall MAE ranking.

## Largest M3 errors

The 25 worst M3 hours (`results/error_analysis/m3_largest_errors.csv`), an
audit table only:

| Rank | Timestamp | Actual | M3 predicted | Abs. error | Band | Season |
|---:|---|---:|---:|---:|---|---|
| 1 | 2014-04-09 20:00 | 580.0 | 75.3 | 504.7 | `gt_p95` | Spring |
| 2 | 2014-01-16 01:00 | 659.0 | 167.3 | 491.7 | `gt_p95` | Winter |
| 3 | 2014-02-26 14:00 | 542.0 | 65.8 | 476.2 | `gt_p95` | Winter |
| 4 | 2014-01-16 02:00 | 648.0 | 176.9 | 471.1 | `gt_p95` | Winter |
| 5 | 2014-02-26 16:00 | 525.0 | 78.8 | 446.2 | `gt_p95` | Winter |

**24 of the 25 fall in `gt_p95`; 23 of 25 are winter; 24 of 25 are
under-predictions.** Several are consecutive hours of the same episode
(16 January, 26 February). The worst single hour, 9 April 2014 at 20:00, had
an actual of 580 µg/m³ against a prediction of 75.3 — the model was not
merely low, it was in a different regime entirely.

**These observations were not removed and no model was retrained around
them.** They are **not** called data errors: nothing independent establishes
them as such, and Phase 3 concluded high-concentration readings in this
dataset are plausible measurements.

---

## What M3 improved

- **Best MAE in every season, at every hour, and in five of six
  concentration bands** — the advantage is broad, not localised.
- **Best severe-hour MAE and the smallest maximum error** (504.7) of any
  model.
- **Zero physically impossible predictions**, against M1's 466.
- **Lowest 99th-percentile absolute error** (240.4), so its worst cases are
  less extreme than any competitor's.

## Where M3 still fails

- **The upper tail.** MAE 160.9 above P95 — 3.3× its overall MAE — with
  96.3% under-prediction.
- **Winter.** 62.7 MAE, 65% above its summer figure.
- **Overnight hours.** 56.3 at 01:00 against 39.6 at 15:00.
- **Calm-wind (`cv`) conditions.** 58.4, its worst regime.
- **Temporally structured residuals.** 0.903 at one hour.
- **Narrow row-wise margin over M2** — 53.4%.

---

## Implications

The single most consequential finding is that **V1's error is concentrated
exactly where prediction matters most**. A user interested in ordinary air
quality is served reasonably: M3's median absolute error is 34 µg/m³ and it
handles the middle bands well. A user interested in *when the air will be
dangerous* is served poorly: in the top band the model under-predicts
almost every hour by an average of 156 µg/m³.

The residual autocorrelation says something more hopeful for future work:
substantial structure remains that a concurrent-hour model cannot reach.
That structure is, by construction, available to a forecasting formulation
with lagged observations — which V1 deliberately excluded to keep the
question well-posed.

## Limitations

- **One site, one city, one evaluation year.** 2014's tail (max 671 µg/m³)
  is milder than 2013's (886); another year could shift the tail figures.
- **Unmeasured drivers.** Emissions, traffic and industrial activity are
  absent from the dataset. No model built from weather and calendar position
  can recover them, and 56% of variance stays unexplained.
- **Subgroup sizes are uneven.** `p90_p95` holds 346 observations, so its
  figures are less stable than the larger bands'.
- **Bands are relative, not regulatory.** They describe this study's
  development distribution only.
- **Severe episodes are defined post hoc for analysis**, from a frozen
  development threshold — but 2014 contains only 41 of them, a small sample
  for episode-level conclusions.
- **M2 and M3 used 2013 for hyperparameter selection**; their 2014 results
  already reflect the modest optimism quantified in Phase 10.
- **Descriptive only.** No significance test, no confidence interval, no
  bootstrap ranking inference was performed — none was pre-declared, and
  manufacturing one after seeing the results would be exactly the practice
  this protocol exists to prevent.

## What must NOT be inferred

- **Not causation.** Every relationship reported here is a predictive
  association measured on observational data. "Winter error is higher" does
  not mean winter causes error.
- **Not a mandate to change V1.** Findings such as tail under-prediction or
  residual autocorrelation suggest obvious-looking remedies — a lag feature,
  a tail-weighted loss, clipping negatives. **None may be implemented in
  V1.** They are hypotheses for a future study, and acting on them now would
  be tuning against a test set that has already been consumed.
- **Not a claim of event detection.** These models predict a concentration.
  No threshold-crossing classifier was built or evaluated.
- **Not a ranking change.** M3 remains best by the pre-registered 2014 MAE.
  Nothing in this phase re-ranks the models, and the row-wise win counts are
  not a competing criterion.

## Next phase: Final V1 report

Protocol Phase 12 consolidates the findings, states the limitations, and
closes V1. **It has not begun.**
