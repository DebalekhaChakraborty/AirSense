# AirSense V2 — Figure Plan

**Phase 11. No new scientific figure was created.**

Every figure below is an **existing frozen asset** produced in Phase 8 or Phase
9 and selected, unmodified, for the manuscript. Phase 11 created zero figures,
regenerated zero figures and modified zero figures. Redrawing for a journal's
house style, if wanted, belongs to a later phase and would require its own
freeze.

Figure numbering below is the proposed manuscript numbering. Placement between
main text and supplement is proposed in `manuscript/SUPPLEMENT_PLAN.md`; a
summary column appears here for convenience.

---

## Figure 1 — Development validation against the locked test

- **Source file:** `figures/phase9_generalization_gap.png`
- **Source SHA-256:** `17d93504835c1bc99d5fe284d43fadbdad9b9244499da6348ef80ac0c4f21e3d`
- **Source phase:** Phase 9
- **Evidence class:** C3
- **Proposed placement:** Supplement
- **What it demonstrates:** Development validation against the locked test.
- **Caption constraint carried from the frozen index:** Caption must state that development and locked-test values come from different partitions and that GRU_R1 uses its seed-42 row.
- **What it must NOT be claimed to demonstrate:** This figure must not be read as a confirmatory comparison; development and locked-test values come from different partitions, and GRU_R1 uses its seed-42 row.

## Figure 2 — Error growth with forecast horizon

- **Source file:** `figures/phase9_mae_by_horizon.png`
- **Source SHA-256:** `af737eee80c52c34f14a525d948686be60fcfd54e7b28acd39b06faa3d8d8d24`
- **Source phase:** Phase 9
- **Evidence class:** C3
- **Proposed placement:** Main text
- **What it demonstrates:** Error growth with forecast horizon.
- **Caption constraint carried from the frozen index:** Caption must not imply a confirmatory horizon-wise test.
- **What it must NOT be claimed to demonstrate:** This figure must not be read as a confirmatory horizon-wise hypothesis test. No horizon-wise contrast was predeclared.

## Figure 3 — Station-level error structure

- **Source file:** `figures/phase9_station_mae_heatmap.png`
- **Source SHA-256:** `98e78758080ebed529fef55c21a6619c84cbeecc1d56db72d02c13eb2c889a22`
- **Source phase:** Phase 9
- **Evidence class:** C3
- **Proposed placement:** Supplement
- **What it demonstrates:** Station-level error structure.
- **Caption constraint carried from the frozen index:** Descriptive; ordering is associated with site character.
- **What it must NOT be claimed to demonstrate:** This figure must not be read as evidence about model behaviour. The ordering is near-identical across models and is associated with site character.

## Figure 4 — Severe-tail error by horizon

- **Source file:** `figures/phase9_severe_mae_by_horizon.png`
- **Source SHA-256:** `ba18fc61ee4e314b293f4bad65fcaae907148c9fc7c33b6a3dc6ebab616be585`
- **Source phase:** Phase 9
- **Evidence class:** C3
- **Proposed placement:** Main text
- **What it demonstrates:** Severe-tail error by horizon.
- **Caption constraint carried from the frozen index:** Caption must state the severe rule and that 244.0 is the training pooled P95, not a regulatory threshold.
- **What it must NOT be claimed to demonstrate:** This figure must not be read as a regulatory exceedance analysis. 244.0 ug/m3 is the training pooled P95, a study-internal quantile.

## Figure 5 — Severe under-prediction rate by horizon

- **Source file:** `figures/phase9_severe_underprediction.png`
- **Source SHA-256:** `f7321c81084750276a6e5568544f43eb2e0c3dde06406b65b218dccb696eda3b`
- **Source phase:** Phase 9
- **Evidence class:** C3
- **Proposed placement:** Supplement
- **What it demonstrates:** Severe under-prediction rate by horizon.
- **Caption constraint carried from the frozen index:** Caption must state residual = actual - prediction.
- **What it must NOT be claimed to demonstrate:** This figure must not be read as a claim that any model is unfit for use. Operational criteria were never defined.

## Figure 6 — Residual autocorrelation at frozen lags

- **Source file:** `figures/phase9_residual_acf.png`
- **Source SHA-256:** `5b1e6c365198dd6cb60d3351eab707bb9c88961265867b54c010d8b280f1309d`
- **Source phase:** Phase 9
- **Evidence class:** C3
- **Proposed placement:** Supplement
- **What it demonstrates:** Residual autocorrelation at frozen lags.
- **Caption constraint carried from the frozen index:** Association only; no causal claim.
- **What it must NOT be claimed to demonstrate:** This figure must not be read as establishing a causal mechanism. These are associations at frozen lags.

## Figure 7 — Error by training-frozen concentration stratum

- **Source file:** `figures/phase9_concentration_regimes.png`
- **Source SHA-256:** `714a9ade4f0ca54fef8a071f9a04011482a9e094ac964151bf318d50e652248a`
- **Source phase:** Phase 9
- **Evidence class:** C3
- **Proposed placement:** Supplement
- **What it demonstrates:** Error by training-frozen concentration stratum.
- **Caption constraint carried from the frozen index:** Boundaries are training-derived; no test quantile was computed.
- **What it must NOT be claimed to demonstrate:** This figure must not be read as using test-derived strata. Stratum boundaries are training-derived constants.

## Figure 8 — Error at severe-event peaks

- **Source file:** `figures/phase9_severe_event_peak_error.png`
- **Source SHA-256:** `f0ad74156d29e96a6907df99a59fa79792028e63bba5449ed779f0ca3a532ba8`
- **Source phase:** Phase 9
- **Evidence class:** C3
- **Proposed placement:** Supplement
- **What it demonstrates:** Error at severe-event peaks.
- **Caption constraint carried from the frozen index:** Must not be framed as an operational early-warning evaluation.
- **What it must NOT be claimed to demonstrate:** This figure must not be framed as an operational early-warning evaluation.

## Figure 9 — Locked final-test primary endpoint

- **Source file:** `figures/phase8_final_test_primary_mae.png`
- **Source SHA-256:** `e56b90dc5a5e777baac72fe527aee49a0c79b5d0404301f486b38c0b5a12aec2`
- **Source phase:** Phase 8
- **Evidence class:** C1
- **Proposed placement:** Main text
- **What it demonstrates:** Locked final-test primary endpoint.
- **Caption constraint carried from the frozen index:** The only confirmatory figure; roles must be labelled.
- **What it must NOT be claimed to demonstrate:** This figure must not be annotated with any model not carrying a frozen locked-test role, and must not label any model a winner.

## Figure 10 — Locked final-test secondary endpoint

- **Source file:** `figures/phase8_final_test_severe_mae.png`
- **Source SHA-256:** `59edd43afcc250de63655482a731f7dde422c56ef6c30b2d28f7226149f05424`
- **Source phase:** Phase 8
- **Evidence class:** C1
- **Proposed placement:** Main text
- **What it demonstrates:** Locked final-test secondary endpoint.
- **Caption constraint carried from the frozen index:** Caption must note persistence is a reference benchmark, not a selected model.
- **What it must NOT be claimed to demonstrate:** This figure must not imply persistence is a recommended model. It is a reference benchmark, not a selected learned model.

---

## Rules binding every figure in this package

1. A C3 figure **must not** be placed adjacent to Table 5 without a visible
   exploratory label. Confirmatory and exploratory evidence must never read as
   one result set.
2. No figure **must not** carry an inferential annotation — no error bar
   presented as a sampling interval, no significance marker. None was
   predeclared.
3. Figures 9 and 10 are the only confirmatory figures (C1). Every other figure
   in this package is post-test exploratory (C3).
4. The main text is deliberately not crowded: four figures are proposed for the
   main paper and six for the supplement.
5. No figure may be regenerated during manuscript preparation. Any redraw
   invalidates the frozen provenance recorded above and requires a new freeze.
