# Phase 8 — Locked Final-Test Protocol

**Status: frozen before the sealed final test was numerically opened.**

This document states the execution contract. The chronological record of what
actually happened, with every hash and every number, is
`docs/PHASE_08_LOCKED_FINAL_TEST_RECORD.md`.

## 1. What Phase 8 is

A single, one-way confirmatory evaluation of three models that were selected
and frozen before any final-test value was read. Phase 8 does not select, tune,
refit, calibrate or explore. It predicts, locks, scores, locks again, and stops.

## 2. Confirmatory hierarchy — frozen, not revisable

| Rank | Model | Role | Prespecified endpoint |
|---|---|---|---|
| 1 | `B3_R2` | PRIMARY CONFIRMATORY MODEL | macro station-horizon MAE |
| 2 | `GRU_R1` (seed 42) | SECONDARY CONFIRMATORY SEVERE-TAIL MODEL | severe MAE, actual > 244.0 µg/m³ |
| 3 | `B0` | REFERENCE BENCHMARK | both, as reference only |

Every other model — `TCN_R1`, `SA_R3`, `SA_R2`, `iTransformer_R1/R2`, `B1`,
`B2`, `B3_R0`, `B3_R1`, `GRU_R0`, `GRU_R2`, `TCN_R0`, `TCN_R2`, and any
foundation model — is historical development evidence only. No model may be
promoted after test access, whatever the outcome. `assert_confirmatory_only`
in `src/evaluation/final_test.py` enforces the whitelist in code.

## 3. Sample universe

The canonical Phase-2 test partition, unchanged: **411,012 samples**, target
timestamps 2016-03-01 00:00 through 2017-02-28 23:00, partitioned by *target*
timestamp, 12 stations × 4 horizons (1/6/12/24 h) = 48 cells, 48-hour context.

Early test targets legitimately have forecast origins inside the validation
period; that is rolling-origin forecasting across a partition boundary, not
leakage.

## 4. The two stages, and why they cannot be merged

**Stage A — chronological prediction generation.** Forecast origins are
processed in strictly increasing timestamp order. No target is read and no
metric is computed; the scoring oracle is not even constructed. Stage A ends by
writing `artifacts/phase8_prediction_lock.json`, which hashes all three
prediction arrays.

**Stage B — scoring.** Refuses to start unless the prediction lock exists and
certifies the ordering. It reloads the prediction arrays from disk, verifies
their hashes against the lock, and only then opens the test actuals. No model
executes in Stage B.

The separation exists so that no test result can influence how a prediction was
produced. Because the arrays are hashed before any metric is seen, that claim is
checkable after the fact rather than merely asserted.

## 5. Causality: three independent mechanisms

1. **Unseal guard.** `CausalTargetHistory` drops PM2.5 values after its ceiling
   at load time, so they are not in memory to leak. Phase 8 is the first phase
   to raise that ceiling into the test period.
2. **Origin guard.** Every accessor refuses an index after the sample's
   forecast origin, and `RevealLedger` counts any breach elementwise against
   each row's own origin. The protocol requires
   `future_target_access_violations = 0`.
3. **Reveal ceiling with fatal masking.** Inputs are materialised only up to
   the current chronological block's ceiling; everything after is `NaN` (or a
   wind sentinel). A feature that read past the ceiling would therefore produce
   a non-finite column, and both model paths reject non-finite input —
   `verify_matrix` for B3, the frozen `SampleBundle.batch` guard for the GRU.

The decisive evidence is none of these assertions but the non-influence test in
`tests/test_phase8_final_test.py`: corrupting **every** observation strictly
after an origin leaves that origin's feature row bit-identical.

### Block granularity, stated honestly

Blocks are calendar months of the forecast origin. The ceiling therefore bounds
materialisation to the end of the sample's own month, while the per-row ledger
check bounds each individual sample to its own origin, which is strictly
tighter. Month blocks are an efficiency choice: the frozen Phase-3 feature code
computes trailing statistics over the whole timeline, and rebuilding them per
origin is infeasible. The causal guarantee does not rest on the block size — it
rests on the per-row bound and on the demonstrated non-influence.

## 6. Model execution contract

**B3_R2** uses the exact Phase-3 feature implementation (`build_matrix`, R2
regime, 405 columns) and the four frozen per-horizon boosters named in
`artifacts/phase7_pretest_freeze.json`. Scaling and fallback constants stay
training-derived; no test statistic is recomputed.

**GRU_R1** uses the canonical 48-hour R1 window, the frozen masks, gap-age
scaling, meteorology/calendar representation, station identity and horizon
conditioning, with the seed-42 weights from
`results/models/neural/GRU_R1.safetensors`. No co-pollutants, no architectural
change, no test-time adaptation.

**B0** is the frozen rule: PM2.5 observed at the origin, else a causal carry
within 6 hours, else the station's **training** median. No test-fitted median.

Nothing is retrained, no seed is reselected, no train+validation refit occurs,
and **no output is clipped** — negative predictions are reported, not repaired.

## 7. Endpoints

- **Primary:** macro station-horizon MAE for `B3_R2` — MAE within each of the
  48 cells, then an equal-weight mean over cells. Never sample-weighted.
- **Secondary:** severe MAE for `GRU_R1` over `actual > 244.0`. The threshold is
  the frozen training-derived constant; it is never re-derived as a test
  percentile, and the rule is strictly `>`, never `>=`.
- **Reference:** both endpoints for `B0`.
- **Descriptive:** micro MAE/RMSE, macro RMSE, R², per-horizon, per-station,
  per-cell, the full severe block, and negative-prediction counts. Descriptive
  metrics do not alter any model's frozen role.

Residual convention: `residual = actual - prediction`; a positive residual is an
under-prediction.

No new threshold, subgroup, weighting, ranking criterion or inferential
procedure may be introduced after opening.

## 8. Storage namespace

Under the human-approved amendment
(`artifacts/phase8_preopening_protocol_amendment.json`), Phase-8 outputs live in
the phase-owned `artifacts/phase8_*` namespace rather than `results/final_test/`.
The original protocol named `results/final_test/`, but the Phase-0 and Phase-1
validators do not recognise that path as later-phase-owned and would classify
legitimate Phase-8 tables as earlier-phase model results. Relocating the outputs
avoids modifying a frozen upstream validator. **This is a storage and provenance
correction only; no numerical content changes.**

| Content | Path |
|---|---|
| Prediction arrays | `artifacts/phase8_predictions/{B0,B3_R2,GRU_R1}.npy` |
| Metric tables | `artifacts/phase8_final_test_tables/` |
| Receipts and locks | `artifacts/phase8_*.json`, `artifacts/final_test_opening_receipt.json` |
| Figures | `figures/phase8_*` |

`results/final_test/` must not exist; `validate_phase8.py` gates on its absence.

## 9. Verification tooling

The Phase-7 registry `artifacts/verification_tooling_registry.json` is
**immutable** at `a1ec2652…79122f`, because
`artifacts/phase7_pretest_freeze.json` hashes it as frozen evidence. Phase 8
therefore adds an append-only child,
`artifacts/phase8_verification_tooling_registry.json`, which names the Phase-7
registry as its parent and pins all nine validators including
`validate_phase7.py` (completed after its parent registry was written) and
`validate_phase8.py` (frozen before test access).

No upstream validator source is modified in Phase 8.

## 10. Stop conditions

Before opening: any git mismatch, any freeze or registry hash mismatch, any
upstream validator failure, an unresolvable model artifact, a changed test
index, a pre-existing test artifact, or unfinalized Phase-8 code.

After opening: a future-target access violation, a prediction count that is not
411,012, a model artifact hash change, unexpected NaN or inf, an alignment
mismatch, an excluded model being evaluated, a prediction file changing after
the lock, or independent verification disagreeing with the scoring path. In all
cases the response is to record the incident in
`artifacts/phase8_execution_incident.json` and stop for human review — never to
alter scientific logic, substitute a model, or restart with changed code.

Phase 8 ends after the primary-results lock, the required tables, the minimal
figure set and independent verification. Post-test error analysis — residual
ACF, episode segmentation, peak/onset analysis, error-case mining, attention
diagnostics, conditioned slices — belongs to a later, clearly labelled phase.
