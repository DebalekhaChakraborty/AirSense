# Phase 9 Error Analysis Record — AirSense V1

**Research-protocol mapping: Protocol Phase 11 — Error Analysis.**

> Record-sequence numbering (`PHASE_09_`) is independent of protocol phase
> numbering. Earlier records were not renamed. Mapping:
> `PHASE_00_FOUNDATION_RECORD.md` (protocol 1–2),
> `PHASE_00A_RUNTIME_RECORD.md` (runtime recovery),
> `PHASE_01_EDA_RECORD.md` (3), `PHASE_02_CLEANING_RECORD.md` (4),
> `PHASE_03_FEATURE_RECORD.md` (5), `PHASE_04_M0_BASELINE_RECORD.md` (6),
> `PHASE_05_M1_LINEAR_REGRESSION_RECORD.md` (7),
> `PHASE_06_M2_DECISION_TREE_RECORD.md` (8),
> `PHASE_07_M3_RANDOM_FOREST_RECORD.md` (9),
> `PHASE_08_FINAL_TEST_RECORD.md` (10), this record (11).

- **Recorded:** 2026-09-06 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE** — descriptive post-evaluation analysis
- **Analysis document:** [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md)
- **Manifest:** [`artifacts/error_analysis_manifest.json`](../artifacts/error_analysis_manifest.json)

---

## 1. Runtime

| Property | Value |
|---|---|
| **Interpreter** | `/home/debalekha_chakraborty/AirSense/venv/bin/python` |
| **Python** | CPython **3.6.7** |
| Preflight | **READY**, exit 0 |

numpy 1.15.4, pandas 0.23.4, scipy 1.1.0, scikit-learn 0.20.0,
matplotlib 3.0.2, seaborn 0.9.0 — all MATCH. No dependency changed.

## 2. Upstream integrity

**18 hashes referenced by `final_evaluation_manifest.json` independently
verified** — raw dataset, split manifest, feature schema, pre-test freeze,
blind freeze, opening receipt, target snapshot, blind predictions, plus all
10 `outputs_sha256` entries. All MATCH.

| Artifact | SHA-256 |
|---|---|
| `artifacts/final_evaluation_manifest.json` | `5f3270cc4d85781e6cea3bcd315649628c825a253e5cc0fa949627af1e7cb273` |
| `results/final_test/final_test_target_snapshot.csv` | `fc04c671599c3b09c9091518b6629d855b1d98de6e8ed462a3dc383372cf2314` |
| `results/final_test/final_test_predictions.csv` | `c09533f32406c5f3caf143631165ee4b0696fd7d5d7e589f308c9249e5617efd` |

Final predictions verified **string-identical** to the pre-opening blind
vectors. Final metrics and MAE ranking unchanged: M3 < M2 < M1 < M0.

## 3. Analysis population

**8,661 rows**, identical for all four models and every subgroup. Season and
hour reconstructed from the timestamp using the Phase-3 definition; wind
direction reconstructed from the frozen one-hot block and verified mutually
exclusive.

**The original Phase-4 2014 source was not reopened.** A guard aborts on any
attempt; a token-level audit found zero executable references.

---

## 4. Frozen definitions

### Development-derived thresholds

From the 2010–2013 development target (**33,096** observations), frozen
**before** any 2014 subgroup metric:

| Quantile | Value |
|---|---:|
| P25 | 29 |
| P50 | 73 |
| P75 | 138 |
| P90 | 221 |
| **P95** | **282** |

**No threshold was optimised on 2014 model performance.**

### Subgroup definitions

Six bands: `le_p25`, `p25_p50`, `p50_p75`, `p75_p90`, `p90_p95`,
`gt_p95`, with deterministic inclusive/exclusive boundaries. Descriptive
relative bands — **not** regulatory AQI categories.

### Severe-hour threshold

`actual_pm25 > 282` (development P95). A relative, study-specific tail
definition, **not** a regulatory severe-pollution threshold.

### Severe-episode definition

One or more consecutive test timestamps **exactly 1 hour apart**, all above
the threshold. **Gaps are never bridged.** Frozen before calculation.

### Residual lags

**1, 6, 12, 24, 48, 168 hours**, matched by **exact timestamp** —
`residual(t)` paired with `residual(t + lag)` only when that hour is
observed. **Row-offset shifting was not used**, because the supervised test
set holds 8,661 of 8,760 hours.

---

## 5. Principal results

**Absolute error (mean / median / max):** M0 65.417 / 49.000 / 598.0 ·
M1 52.900 / 40.180 / 520.0 · M2 51.249 / 35.374 / 515.5 ·
**M3 48.535 / 34.220 / 504.7**. M1 is marginally better than M3 at the 90th
and 95th percentiles (108.113 vs 109.561; 150.416 vs 153.127) — recorded
because it is true; the pre-registered metric is the mean.

**Bias:** all feature models over-predict the majority of hours while their
mean residual sits near zero (M3: 63.56% over, mean −2.845). M0 mean
residual +24.735 with median −1.000.

**M1 negative predictions:** **466 (5.38%)**, min −162.139, actual mean
21.217 there, MAE 58.989 on those rows. **Not clipped.**

**Season (MAE):** M3 best in all four — Winter 62.738, Spring 39.624,
Summer 38.014, Autumn 54.241.

**Hour (MAE):** all models worst at 01:00, best at 14:00–15:00. M3 39.634
(h15) to 56.281 (h01), spread 16.647.

**Concentration band (M3 MAE):** 33.333 → 39.038 → 34.373 → 59.597 →
94.174 → **160.897**. M3 mean residual by band: −33.1, −34.7, −9.2, +30.5,
+87.2, **+155.9**; under-prediction rate 4.07% → **96.29%**.

**Wind direction (M3 MAE):** NE 49.828, NW 43.096, SE 45.904, **cv 58.407**.

**Severe hours:** **512** above 282 µg/m³. MAE — M0 293.990, M1 194.676,
M2 164.260, **M3 160.897**. Under-prediction — M0 **100%**, M1 **100%**,
M2 94.73%, M3 **96.29%**. M3 max abs error 504.744.

**Episodes:** **41**, durations 1 / 7 / 12.49 / **76** hours
(min/median/mean/max).

**Residual autocorrelation (M3):** 1 h **0.903**, 6 h 0.592, 12 h 0.438,
24 h 0.344, 48 h 0.193, 168 h 0.103. Pairs 8,625 → 8,394; none too sparse.
**Errors remain strongly temporally structured.**

**Pairwise:** M3 beats M2 on **4,623 of 8,661 rows (53.4%)**, M2 on 4,038,
0 ties; mean per-row difference −2.714. M3 beats M1 on 56.18% and M0 on
62.65%. Descriptive only — not a selection criterion.

**M3 largest 25 errors:** 24 in `gt_p95`, 23 in Winter, 24
under-predictions. Worst: 2014-04-09 20:00, actual 580.0, predicted 75.3.

---

## 6. Artifacts

| Path | SHA-256 | Rows |
|---|---|---:|
| `results/error_analysis/absolute_error_summary.csv` | `7d932bc7737004aa12203fe201fc71f43b92be2f51019a0f9e21e11e54d80b40` | 4 |
| `results/error_analysis/error_by_concentration_band.csv` | `f201ae97a5cc09c06680446e34a280f24dfb234890ab7ea406a8ca6f630e7aa1` | 24 |
| `results/error_analysis/error_by_hour.csv` | `4489b488acd237f9d8a676ed771617fca167a49a8638b18fe649cff8c53973b4` | 96 |
| `results/error_analysis/error_by_season.csv` | `403e4ea1030f0cc3c9c881a4e932254760c71506b67b9fbc6c512164c08b8b01` | 16 |
| `results/error_analysis/error_by_wind_direction.csv` | `8de659fed3257a6d4ac0ceb9de6f94b6de749dd13f11c069479efabff1d21097` | 16 |
| `results/error_analysis/m3_largest_errors.csv` | `0e4029db7adb356d8c4f44cd8f9e8803cfd2a838b8b9ea78292daa476ec699e7` | 25 |
| `results/error_analysis/m3_vs_m2_error_difference.csv` | `d83b314c2784497fc1603e1f6bd99e33b664c6bdbce5a092693a15d6d8464571` | 8661 |
| `results/error_analysis/pairwise_error_wins.csv` | `d8d0bb2af2b43a10d054a96c25f4c46b13b467e3a910e8ae6e3ba82756db5afa` | 6 |
| `results/error_analysis/prediction_bias_summary.csv` | `40cad985bb126f6dbbe6b0df0080d5b958c6e376eb0e58512bbcdd36352e6066` | 4 |
| `results/error_analysis/residual_autocorrelation.csv` | `8a0c26babebad85eb9ef582ca444ae7d91d28e059494879d5103d9e74ff233eb` | 24 |
| `results/error_analysis/severe_episode_details.csv` | `8f114c221af6e48c648a6596b9deb709181d81e7c20b6642d5692fb06dc9a57c` | 41 |
| `results/error_analysis/severe_episode_summary.csv` | `153f123385ee6c236cc71b365b314d0f780e79df9d14211d878d6a126a7d384d` | 1 |
| `results/error_analysis/severe_hour_summary.csv` | `bc3179d667605e1f24ccb3e5a0b7407e8b61fe3744c2096c788a40775397ee27` | 4 |
| `artifacts/error_analysis_manifest.json` | `8b642d4865bbfc23afe0f011c6e8c6c0a24928bf1a0384d7dfbd0d368f0a84be` | — |

### Figures

| Path | SHA-256 |
|---|---|
| `figures/error_analysis_mae_by_season.png` | `21bc52ba79395c47e3b4cc2050c09a1ee1275e6738d0aa01230cdea3010146b3` |
| `figures/error_analysis_mae_by_hour.png` | `6e129aae66d1fba5ee7def53a3f8873f95758b62b5906217c7fd6fc6cc81c91c` |
| `figures/error_analysis_mae_by_concentration_band.png` | `1ba4754013a6a4c9143e5a4f5b3eeb1fb1ec11ed604eb851ce0488ccbd54f462` |
| `figures/error_analysis_absolute_error_cdf.png` | `323f949c5c1e2329d0724efcc255694ab376d31001129da1eb26e8db56dffd7d` |
| `figures/error_analysis_residual_autocorrelation.png` | `fdd79bfaa5c56d7428e11ab586c9d71f454c34b55656314ad7c050b3cf13fe4e` |
| `figures/error_analysis_severe_episodes.png` | `933a7b2e2f0a0d50bb854f9871327d06202b06d4abfddc186faeca9bb0b649e2` |

Every figure is labelled *2014 Final Held-Out Test — Post-Evaluation Error
Analysis*. The CDF was built directly from sorted absolute errors and
`rank / n`, with no post-2019 plotting helper.

---

## 7. Verification

### Deterministic rerun

Run twice; **all 20 artifacts byte-identical** (14 CSV + 1 JSON + 6 PNG —
the CSV count includes every table). No randomness, no execution timestamp.

### Independent recomputation

Recomputed in a **separate snippet** from the frozen artifacts, without the
Phase-11 pipeline:

| Check | Result |
|---|---|
| All five development thresholds | exact match |
| Complete Winter season table (MAE + RMSE, all 4 models) | max diff 1.42e-14 |
| Complete `gt_p95` band table (MAE, all 4 models) | max diff 2.84e-14 |
| Severe-hour M3 MAE | 2.84e-14 |
| Severe-hour M3 under-prediction rate | exact |
| M3 lag-24h Pearson r and n_pairs | exact (0.344170197257, 8,542) |
| M2 vs M3 row-wise win counts | exact (4,038 / 4,623) |

**All agree within 1e-10.**

### Invariants

**Episodes:** every episode contains only severe hours · within-episode gaps
are exactly 1 hour · no overlap · every severe hour in exactly one episode ·
no sub-threshold hour included · episode hours sum to 512.

**Lag pairs:** every matched pair verified to differ by exactly the declared
lag; no row-offset approximation.

**Cross-table reconciliation:** season, hour, concentration-band and
wind-direction counts each sum to **8,661 per model**; all four models
evaluated over identical rows in every subgroup.

### Python 3.6 compatibility

All project `.py` files compile under CPython 3.6.7. No post-3.6 syntax, no
new dependency, no post-cutoff API.

---

## 8. Files created and modified

**Created:** `src/evaluation/error_analysis.py`,
`scripts/run_error_analysis.py`, 14 `results/error_analysis/` tables,
`artifacts/error_analysis_manifest.json`, 6
`figures/error_analysis_*.png`, `docs/ERROR_ANALYSIS.md`,
`docs/PHASE_09_ERROR_ANALYSIS_RECORD.md`.

**Modified:** `README.md` and `docs/V1_RESEARCH_PROTOCOL.md` —
**administrative status only** (Phase 11 marked Complete).

**Unchanged:** every Phase-10 artifact, every model manifest, both freezes,
the raw dataset, all Phase-4 and Phase-5 artifacts.

## 9. Explicit confirmations

- **Phase 11 used only the frozen Phase-10 predictions.**
- **No model was fitted.** `models_fitted: 0`.
- **No `.fit()` occurred.** No estimator was imported; only
  `sklearn.metrics`.
- **No `.predict()` occurred.** `predictions_generated: 0`.
- **No prediction changed** — verified string-identical to the blind
  vectors.
- **Final MAE/RMSE/R² remained unchanged.**
- **Final MAE ranking remained unchanged** — M3 < M2 < M1 < M0.
- **Development-derived thresholds were frozen before subgroup
  calculations.**
- **No threshold was optimised on 2014 model performance.**
- **Severe episodes used the fixed development P95 rule.**
- **Residual autocorrelation used exact timestamp lags, not row shifts.**
- **The original Phase-4 2014 source was not reopened.**
- **Negative M1 predictions were not clipped.**
- **No severe-event model was created.**
- **No hybrid or ensemble model was created.**
- **No feature was added; no lag feature was added.**
- **No hyperparameter tuning occurred.**
- **No random split occurred.**
- **No post-test retraining occurred.**
- **No significance test, confidence interval or bootstrap ranking
  inference was performed** — none was pre-declared.
- **2014 remains an exhausted test set.**
- **All upstream frozen artifacts remained unchanged.**
- **No dependency changed.**
- **No git write operation was performed; history was not rewritten.**

## 10. Status

Protocol Phase 11 is **complete**. All legitimate changes are left
**unstaged**.

The next phase is Protocol Phase 12 — the final V1 report. **It has not
begun.**
