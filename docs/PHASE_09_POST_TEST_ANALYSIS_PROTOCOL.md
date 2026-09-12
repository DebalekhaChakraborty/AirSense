# Phase 9 — Post-Test Exploratory Analysis Protocol

**Classification: POST-TEST EXPLORATORY. Confirmatory status: NONE.**

Frozen before any Phase-9 numerical output was computed. The machine-readable
form is `artifacts/phase9_analysis_freeze.json`; this document is its prose
companion. The findings themselves live in
`docs/PHASE_09_POST_TEST_ERROR_ANALYSIS_RECORD.md`.

## 1. What Phase 9 is, and is not

Phase 8 evaluated the sealed final test once, under a hierarchy frozen before
the test was opened. Those results are **immutable scientific evidence**:

| Role | Model | Endpoint | Value |
|---|---|---|---|
| PRIMARY CONFIRMATORY | `B3_R2` | macro station-horizon MAE | `31.98722559774534` |
| SECONDARY CONFIRMATORY | `GRU_R1` seed 42 | severe MAE, actual > 244.0 | `109.3328897571946` (n = 20,336) |
| REFERENCE BENCHMARK | `B0` | macro / severe MAE | `36.138513538776856` / `93.22447875688434` |

Phase 9 asks *why* those numbers look as they do. It produces **no confirmatory
result**. Nothing in it may change a Phase-8 value, reorder the hierarchy,
rescue a model, introduce a new winner, tune anything, or evaluate a model that
was not in the frozen three. If a Phase-9 recomputation disagrees with a
Phase-8 value, the response is to stop for human review — never to "correct"
Phase 8.

## 2. Inputs, and the prohibition on inference

Phase 9 consumes only frozen evidence:

- the three frozen final-test prediction arrays under `artifacts/phase8_predictions/`;
- the canonical Phase-2 test sample index;
- the final-test PM2.5 truth, read from the raw station CSVs;
- frozen development metrics from the Phase-7 pre-test freeze and addendum;
- training-only statistics from `configs/preprocessing.json`.

**No model is loaded, no inference is run, no prediction is generated.**
Recomputing metrics from frozen prediction arrays is permitted; producing a
prediction is not. Only `B3_R2`, `GRU_R1` and `B0` are analysed.

## 3. Frozen definitions

Everything below is fixed now so that no choice can be made after seeing a
result.

### Residuals

`residual = actual − prediction`. A **positive residual is an under-prediction**.

### Concentration regimes — training-frozen boundaries

Both boundaries are training-derived and verified against
`configs/preprocessing.json`: pooled training median **61.0 µg/m³**, pooled
training P95 **244.0 µg/m³** (`rule: pooled_training_pm25_p95`). No test
quantile is ever computed.

| Stratum | Rule |
|---|---|
| LOW | `actual <= 61.0` |
| ELEVATED | `61.0 < actual <= 244.0` |
| SEVERE | `actual > 244.0` |

These are **POST-TEST EXPLORATORY STRATA**, not endpoints.

### Severe rule

Strictly `actual > 244.0`. Never `>=`. Targets sitting exactly at 244.0 are
excluded, and the count of such targets is verified independently rather than
assumed. 244.0 is the **training pooled P95** — it is not a regulatory or AQI
threshold and must never be described as one.

### Autocorrelation

Residual and target series are placed on the hourly grid indexed by **target
timestamp**, with missing hours held as NaN. For lag *k*, the autocorrelation
is the Pearson correlation between `x[t]` and `x[t+k]` over **complete pairs
only** — index pairs where both entries are finite. Pairs are not imputed,
interpolated or gap-bridged, and a lag with fewer than 30 complete pairs is
reported as NA.

Lags are fixed at exactly: **1, 2, 3, 6, 12, 24, 48, 72, 168** hours. No lag
may be added or dropped after seeing a result.

### Severe events

A severe event is one or more **consecutive hourly observations** at a single
station where `actual > 244.0`, identified on that station's **unique hourly
truth series** — never on horizon-duplicated sample rows. No gap bridging, no
tolerance window, no minimum duration. A single non-severe hour ends an event,
including an hour whose observation is missing.

### Severe detection diagnostic

Actual severe: `actual > 244.0`. Predicted severe: `prediction > 244.0`. The
same frozen threshold on both sides. No threshold tuning, no ROC optimisation,
no alternative threshold, no AUROC-based selection. This is a descriptive view
of frozen point forecasts.

### Seasons

By target timestamp only. **DJF** = December, January, February; **MAM** =
March, April, May; **JJA** = June, July, August; **SON** = September, October,
November. Boundaries are not optimised.

### Hour of day

Exact target hour, 0–23. No adaptive binning.

### Model error complementarity

Pairwise over absolute errors: Pearson correlation, Spearman correlation, mean
and median paired absolute-error difference, and the fraction of samples where
A has strictly smaller absolute error than B. Computed overall and severe-only.
Also the fraction where each model is strictly best of three. **Ties are
counted explicitly** and reported as their own category rather than split.

No ensemble is constructed: no oracle prediction, no oracle score, no averaging
of predictions, no stacker, no fitted weights, no proposed ensemble result.

## 4. Deliverables

Tables (`artifacts/phase9_*`): `generalization_gap.csv`,
`horizon_error_analysis.csv`, `station_error_analysis.csv`,
`station_horizon_error_analysis.csv`, `residual_acf.csv`,
`residual_acf_summary.csv`, `test_target_acf.csv`,
`concentration_regime_analysis.csv`, `severe_tail_analysis.csv`,
`severe_events.csv`, `severe_event_model_analysis.csv`,
`severe_detection_analysis.csv`, `seasonal_error_analysis.csv`,
`hour_of_day_error_analysis.csv`, `error_complementarity.csv`,
`negative_prediction_analysis.csv`.

Figures (`figures/phase9_*`), eleven, drawn only after all tables exist, every
title or subtitle carrying **POST-TEST EXPLORATORY**: `generalization_gap`,
`mae_by_horizon`, `station_mae_heatmap`, `residual_acf`,
`concentration_regimes`, `severe_mae_by_horizon`, `severe_underprediction`,
`severe_event_peak_error`, `seasonal_mae`, `hour_of_day_mae`,
`model_error_correlation`. No decorative figure, and no extra figure added
because a pattern looked interesting without first recording an amendment.

Axes are not manipulated to exaggerate differences. Units are µg/m³ where
applicable. Residual figures state `actual − prediction`, positive =
under-prediction. Severe figures state `actual PM2.5 > 244.0 µg/m³`.

Documents: this protocol, `PHASE_09_POST_TEST_ERROR_ANALYSIS_RECORD.md`, and
`PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md`.

## 5. Hypotheses H1–H5

Phase 9 may discuss the development-era hypotheses but may not redefine them,
and may not manufacture final-test tests that were never run. The locked test
evaluated three models on two endpoints; it did not evaluate the paired regime
contrasts that H1–H4 would need. Those are labelled **development-supported**
or **Phase-7 robustness-supported** and **not independently re-tested on the
locked final test**.

H5 is reported honestly against the locked result, including the part that is
unfavourable: whichever way `GRU_R1` compares to `B3_R2` in the tail,
persistence `B0` was better than both, and H5 is not redefined to make it pass.

## 6. Uncertainty quantification is out of scope

No bootstrap, no p-value, no confidence interval, no Bayesian interval, no new
statistical test. None was frozen before the test was opened, and choosing one
after seeing these diagnostics would be selecting an inferential method to suit
the answer. If formal temporal uncertainty quantification is wanted, it needs
its own separately frozen post-test protocol.

## 7. No rescue experiment

An observed failure is a finding, not a prompt to act. Phase 9 will not train a
new model, change epochs or loss, add severe weighting or quantile loss, add
coordinates or a graph, ensemble, calibrate, move a threshold, add future
meteorology, change context, try a foundation model, or try another seed. Those
are candidate **future work** and are recorded as such.

## 8. Language discipline

Phase-8 results are "the locked final test". Phase-9 results are "post-test
exploratory analysis". Relationships are "associated with" unless causality is
genuinely established. The words *proved*, *solved* and *eliminated* are not
used. Attention in `SA_R3` is never described as geographic propagation or
spatial reasoning — Phase 7 found attention was not proximity-driven
(Spearman(distance, attention) = +0.136).

V1 and V2 are **not** comparable benchmark tasks: V1 was concurrent-hour
estimation on a different period and a single-station design; V2 is true future
multi-horizon forecasting across a 12-station network on a different
chronological test period. No cross-study numeric performance improvement claim
is permitted. The comparison is **qualitative and structural**, at the level of
failure modes only.

## 9. Hierarchy is unchanged regardless of findings

`B3_R2` remains PRIMARY CONFIRMATORY MODEL. `GRU_R1` remains SECONDARY
CONFIRMATORY SEVERE-TAIL MODEL. `B0` remains REFERENCE BENCHMARK. If `B0` wins
a subgroup, that is reported and `B0` is still not promoted. If `GRU_R1` beats
`B3_R2` in the tail, that is reported and `GRU_R1` is still not promoted.

## 10. Output namespace and git

Tables in `artifacts/phase9_*`, figures in `figures/phase9_*`, documents in
`docs/PHASE_09_*`. `results/final_test/` is never created. No Phase-8 output is
mutated. Git is read-only for the agent throughout; Phase-9 outputs are left
unstaged.
