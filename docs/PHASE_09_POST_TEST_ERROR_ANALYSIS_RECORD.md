# Phase 9 — Post-Test Error Analysis Record

## 1. Status and classification

**POST-TEST EXPLORATORY. Confirmatory status: NONE.**

Every number in this document is exploratory. None is a confirmatory result,
none was frozen before the locked final test was opened, and none changes a
Phase-8 value. The frozen plan is
`docs/PHASE_09_POST_TEST_ANALYSIS_PROTOCOL.md` and
`artifacts/phase9_analysis_freeze.json`.

**No model was trained, refitted, loaded or run. `models_trained: 0`,
`predictions_generated: 0`.** Phase 9 consumed the three frozen Phase-8
prediction arrays and nothing else that a model could produce.

## 2. Immutable Phase-8 anchors

| | |
|---|---|
| Pre-test snapshot | `75494266792e09bbcfa51c00aaacec58c5b0fb4a` |
| Locked-evaluation snapshot | `9a787a6320a3eca2104abfcbf10c8dbd29bfc19b` |
| HEAD at Phase-9 start | `adf19874fb7863579e8b0ca5a1b45d3c7975c097`, clean tree |
| Phase-8 final validation receipt | `863d740776024e0ab97a599f0445fcc5b3cf90c9b1a69a79f5597df278d5f13a` (60 gates, 0 failures) |

Ancestry verified with `git merge-base --is-ancestor`, not log membership.
**Phase-8 drift: prediction 0, metric 0, scientific 0.**

The locked results, carried by reference:

| Role | Model | Endpoint | Value |
|---|---|---|---|
| PRIMARY CONFIRMATORY | `B3_R2` | macro station-horizon MAE | `31.98722559774534` |
| SECONDARY CONFIRMATORY | `GRU_R1` seed 42 | severe MAE > 244.0 | `109.3328897571946` (n = 20,336) |
| REFERENCE BENCHMARK | `B0` | macro / severe MAE | `36.138513538776856` / `93.22447875688434` |

All four were recomputed here from the frozen arrays and agreed to <1e-9. The
analysis script stops rather than proceeding on disagreement.

## 3. Analysis freeze

`artifacts/phase9_analysis_freeze.json` =
`66fda8ee9a5a9facfe3f243fd8fbfa74c24a8dff17c86e205aebe8bcb4ca5e9f`, written
before any Phase-9 number existed. It fixes the severe rule, both
concentration boundaries, the nine ACF lags, the event rule, the season and
hour definitions, the complementarity statistics, and the sixteen tables and
eleven figures produced.

Both concentration boundaries were verified against frozen config rather than
assumed: training pooled median **61.0**, training pooled P95 **244.0**
(`rule: pooled_training_pm25_p95`). 244.0 is a study-internal training
quantile, **not a regulatory or AQI threshold**.

## 4. Data used

411,012 canonical test samples; truth read from the raw station CSVs; the three
frozen prediction arrays, hash-verified against the prediction lock. **232
targets sit at exactly 244.0 and all are excluded from severe** — independently
re-verified, not taken from Phase 8. Under `>=` the severe count would have
been 20,568 rather than 20,336.

## 5. Development → locked-test transfer

| Model | macro MAE dev → test | severe MAE dev → test |
|---|---|---|
| `B3_R2` | 30.1117 → 31.9872 (**+6.23%**) | 136.612 → 131.901 (−3.45%) |
| `GRU_R1` | 30.9122 → 33.0821 (**+7.02%**) | 118.965 → 109.333 (−8.10%) |
| `B0` | 33.1277 → 36.1385 (**+9.09%**) | 101.273 → 93.224 (−7.95%) |

`GRU_R1`'s development analogue is its **seed-42** row, the artifact the locked
test evaluated. All three degraded on overall error and improved in the tail —
the 2016-17 tail was somewhat easier than 2015-16. Ordering is preserved on
both endpoints. Notably, the **reference baseline degraded most** and the
primary model least, so the learned advantage widened slightly out of sample.

## 6. Horizon structure

MAE, and improvement over persistence:

| Model | h=1 | h=6 | h=12 | h=24 |
|---|---|---|---|---|
| `B3_R2` | 9.304 (+9.47%) | 28.234 (+11.54%) | 39.437 (+10.82%) | 50.941 (+12.30%) |
| `GRU_R1` | 12.863 (**−25.16%**) | 29.244 (+8.37%) | 39.354 (+11.01%) | 50.833 (+12.49%) |
| `B0` | 10.277 | 31.916 | 44.223 | 58.088 |

`B3_R2` beats persistence at every horizon. `GRU_R1` is substantially **worse**
than persistence at h=1 — a one-hour-ahead forecast is nearly a copy of the
last observation, and a 24,641-parameter network has no way to be that
faithful — but overtakes it from h=6 and is marginally the best model at h=24.

`B0`'s micro R² at h=24 is **−0.131**: at a full day's lead persistence is
worse than predicting the test mean, while still winning the severe tail. Those
two facts are not in tension — they describe different parts of the
distribution.

## 7. Station structure

`B3_R2` ranges from **26.835** (Huairou) to **35.739** (Dongsi). The ordering is
near-identical for all three models: northern suburban sites (Huairou,
Dingling, Changping) are easiest, central urban sites (Dongsi, Wanshouxigong,
Nongzhanguan) hardest. Because the ranking barely moves across models, it is
associated with site character rather than anything a model is doing. Dongsi
also carries the most severe hours (2,328) and Dingling the fewest (936).

## 8. Residual temporal structure

Mean over 12 stations at h=24:

| Model | lag 1 | lag 24 | lag 168 |
|---|---|---|---|
| `B3_R2` | **0.9626** | **+0.2669** | 0.0266 |
| `GRU_R1` | **0.9500** | −0.2321 | 0.0001 |
| `B0` | **0.9525** | −0.2971 | 0.0126 |

Lag-1 residual correlation near 0.95 for every model is the single most
striking diagnostic here: a large share of the remaining error is systematic
temporal structure that none of these models absorbs.

The lag-24 sign split is a real difference in error character. `B3_R2` repeats
its error at the same hour the next day; `GRU_R1` and `B0` alternate. This is
descriptive — it says nothing about mechanism.

For context, the observed test PM2.5 series itself autocorrelates at 0.972
(1 h), 0.781 (6 h), 0.428 (24 h), 0.023 (168 h).

## 9. Concentration regimes (POST-TEST EXPLORATORY STRATA)

| Model | LOW mean residual | ELEVATED | SEVERE | LOW MAE | ELEVATED MAE | SEVERE MAE |
|---|---|---|---|---|---|---|
| `B3_R2` | −19.155 | +18.366 | +127.650 | 23.154 | 31.873 | 131.901 |
| `GRU_R1` | −22.121 | +14.259 | +106.595 | 26.883 | 32.225 | 109.333 |
| `B0` | −17.930 | +16.845 | +65.549 | 27.768 | 40.725 | 93.224 |

Textbook regression to the mean: **every model over-predicts when the air is
clean and under-predicts when it is dirty**, with the bias growing sharply with
concentration. `B3_R2` is best in LOW and ELEVATED; `B0` is best in SEVERE.
n = 228,284 / 162,392 / 20,336.

## 10. Severe tail

Severe MAE by horizon (5,084 severe samples per horizon):

| Model | h=1 | h=6 | h=12 | h=24 |
|---|---|---|---|---|
| `B3_R2` | **26.760** | 100.179 | 170.875 | 229.791 |
| `GRU_R1` | 56.011 | 99.678 | 129.012 | 152.631 |
| `B0` | 26.017 | 88.113 | 116.240 | **142.528** |

**The persistence advantage in the tail is entirely a long-horizon
phenomenon.** At h=1 `B3_R2` and `B0` are within 0.74 µg/m³. By h=24 `B3_R2` is
87 µg/m³ worse than persistence while `GRU_R1` sits between them. Averaged over
horizons this produces the locked headline — `B0` 93.22 < `GRU_R1` 109.33 <
`B3_R2` 131.90 — but the average conceals that the models are nearly
indistinguishable at short lead.

Severe under-prediction rate climbs with horizon: `B3_R2` 63.77% → 91.88% →
98.84% → **99.90%**. At h=24 it under-predicts essentially every severe hour.

## 11. Severe events

586 strict severe events, median duration **4 h**, mean 8.7 h, max 62 h; 19.8%
are single-hour. Median peak 292 µg/m³, maximum 835.

Behaviour at the event peak hour:

| Model | horizon | mean \|error\| at peak | under-predicted peak | flagged first severe hour |
|---|---|---|---|---|
| `B3_R2` | h=1 | 41.5 | 92.5% | 19.5% |
| `B3_R2` | h=24 | 225.4 | **100.0%** | **0.3%** |
| `GRU_R1` | h=1 | 77.1 | 97.1% | 13.7% |
| `GRU_R1` | h=24 | 169.1 | 94.9% | 10.4% |
| `B0` | h=1 | 33.3 | 99.1% | 2.2% |
| `B0` | h=24 | 169.9 | 88.9% | 16.2% |

At a day's lead `B3_R2` under-predicts the peak of **every one of 586 events**
and raises a severe flag at the onset of 2 of them. None of the three is usable
as a 24-hour early-warning system.

### Severe detection (frozen 244.0 on both sides)

Recall collapses with horizon for the learned models: `B3_R2` 0.892 → 0.530 →
0.050 → **0.0069**; `GRU_R1` 0.795 → 0.518 → 0.352 → 0.236; `B0` 0.887 → 0.618
→ 0.486 → **0.356**. At h=24 `B3_R2` produced 35 true positives out of 5,084
severe hours. Persistence is the best severe detector at long lead, at the cost
of the most false positives (3,286).

## 12. Seasonal and diurnal structure

Winter dominates. At h=24, DJF holds 3,120 of the severe hours against 28 in
JJA, and carries the largest MAE for every model (`B3_R2` 79.94 in DJF against
29.90 in JJA). Severe failure is overwhelmingly a winter phenomenon.

The diurnal effect is shallow: at h=24 error is lowest around 08:00 and highest
just after midnight, spread ~7–8 µg/m³ for all models — small beside the
horizon and concentration effects.

## 13. Model complementarity

Absolute errors correlate strongly: overall Pearson 0.801 (`B3_R2`/`GRU_R1`),
0.668 (`B3_R2`/`B0`), 0.873 (`GRU_R1`/`B0`). **The models largely fail on the
same samples.**

Strictly best of three — overall: `B3_R2` 36.89%, `B0` 33.46%, `GRU_R1` 29.64%.
Severe only: **`B0` 52.88%**, `B3_R2` 26.39%, `GRU_R1` 20.74%.

So persistence is not merely better on average in the tail; it is the best
individual model on a majority of severe samples. **No ensemble was
constructed, no oracle was scored, and no weights were fitted** — that is
future work, not a Phase-9 result.

## 14. Negative predictions

| Model | n | % | min |
|---|---|---|---|
| `B3_R2` | 53 | 0.0129 | −14.425 |
| `GRU_R1` | **1,358** | 0.3304 | −28.540 |
| `B0` | 0 | 0.0000 | +2.0 |

`GRU_R1` produces 26× more negatives, and 807 of its 1,358 are at h=1 — the
horizon where it is also weakest. Nothing was clipped. The unconstrained head
was a deliberate pre-registration; this is its visible cost.

## 15. V1 → V2 failure-mode continuity

See `docs/PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md`, labelled **QUALITATIVE /
STRUCTURAL COMPARISON**. V1 and V2 are not directly comparable — different
task, period, design, and a different severe threshold (282.0 vs 244.0) — so no
cross-study numeric claim is made. Both studies land on the same two open
pathologies: systematic severe under-prediction, and residual autocorrelation
near 0.9–0.97 at lag 1.

## 16. Hypothesis interpretation

| | Status |
|---|---|
| H1 target PM2.5 history helps | **development-supported**; not independently re-tested on the locked final test |
| H2 history benefit largest at short horizons | **development-supported**; consistent with the locked h=1 results but not a confirmatory test |
| H3 co-pollutants add value | **development-supported**; the locked test evaluated only R2 among B3 variants, so no contrast exists |
| H4 cross-station context adds value | **Phase-7 robustness-supported** (paired, 5/5 seeds); not independently re-tested — no cross-station model was in the confirmatory set |
| H5 modern methods improve the severe tail | **partially supported, and the unfavourable half must be stated** |

On H5: `GRU_R1` did improve severe MAE relative to `B3_R2` on the locked test
(109.33 against 131.90, a 17.1% reduction), which is the direction development
predicted. But **persistence `B0` remained better than both** at 93.22. H5 is
not redefined to make it pass: a learned model beat another learned model in
the tail, and the naive baseline beat them both.

The locked test did not evaluate the paired regime contrasts H1–H4 require. No
final-test hypothesis test is manufactured here.

## 17. Limitations

1. Single test period, one city, one pollutant.
2. `GRU_R1`'s confirmatory role rested on five-seed evidence while the
   evaluated artifact is the single seed-42 model; Phase 7 showed severe-tail
   metrics swinging 29–35 µg/m³ across seeds.
3. H1–H4 were not re-tested confirmatorily; only three models were evaluated.
4. No uncertainty quantification — none was frozen before the test was opened,
   so reporting one now would mean choosing a method after seeing the answers.
5. Severe strata are exploratory, not endpoints.
6. Bias/variance decomposition is only partly identifiable from point
   forecasts; the peak analysis addresses this obliquely.
7. Event analysis uses observed hours only; missing hours end an event by the
   frozen rule, so long episodes with gaps appear as several events.

## 18. Future research implications

Candidates, recorded as future work and **not** attempted here: quantile or
distributional forecasting for the upper tail; explicit severe-weighted or
asymmetric loss; residual models exploiting the remaining lag-1 structure; a
persistence-anchored hybrid given `B0`'s tail behaviour; event-onset detection
framed as classification rather than point regression; and separately frozen
post-test uncertainty quantification.

## 19. Reproducibility

Analysis `scripts/run_phase9_post_test_analysis.py`; independent recomputation
`scripts/verify_phase9_independently.py` (**18 checks, all agree**, via a
separate parser, separate truth read and reimplemented statistics); figures
`scripts/build_phase9_figures.py`; validator `scripts/validate_phase9.py`,
frozen before the analysis ran and pinned in
`artifacts/phase9_verification_tooling_registry.json`.

Two corrections were made during Phase 9, both before acceptance and neither
touching a scientific definition. First, the severe-event scanner initially
failed to advance past a completed event, re-detecting it indefinitely; it ran
19.5 hours before being stopped, and the fix was a one-line advance. The nine
tables produced before that point were byte-identical to the tables produced
after the fix, so no result was affected. Second, the validator's overclaiming
check matched substrings rather than whole words, so the honest word
"unresolved" tripped it because a banned term appears inside that word; it now
matches word boundaries.

## 20. No model was changed

No model was trained, refitted, calibrated, ensembled, reseeded or loaded. No
prediction was generated. No Phase-8 prediction array, metric table, lock,
receipt or manifest was modified. The confirmatory hierarchy is exactly as
frozen before the test was opened:

- `B3_R2` remains **PRIMARY CONFIRMATORY MODEL**
- `GRU_R1` remains **SECONDARY CONFIRMATORY SEVERE-TAIL MODEL**
- `B0` remains **REFERENCE BENCHMARK**

`B0` wins the severe tail and is not promoted. `GRU_R1` beats `B3_R2` in the
tail and is not promoted. Reporting a subgroup win is not the same as changing
a hierarchy, and Phase 9 changes nothing.
