# Phase 8 Final Test Record — AirSense V1

**Research-protocol mapping: Protocol Phase 10 — Held-out Evaluation.**

> Record-sequence numbering (`PHASE_08_`) is independent of protocol phase
> numbering. Earlier records were not renamed. Mapping:
> `PHASE_00_FOUNDATION_RECORD.md` (protocol 1–2),
> `PHASE_00A_RUNTIME_RECORD.md` (runtime recovery),
> `PHASE_01_EDA_RECORD.md` (3), `PHASE_02_CLEANING_RECORD.md` (4),
> `PHASE_03_FEATURE_RECORD.md` (5), `PHASE_04_M0_BASELINE_RECORD.md` (6),
> `PHASE_05_M1_LINEAR_REGRESSION_RECORD.md` (7),
> `PHASE_06_M2_DECISION_TREE_RECORD.md` (8),
> `PHASE_07_M3_RANDOM_FOREST_RECORD.md` (9), this record (10).

- **Recorded:** 2026-09-06 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE** — the 2014 target has been opened and is exhausted
- **Analysis document:** [`FINAL_HELD_OUT_EVALUATION.md`](FINAL_HELD_OUT_EVALUATION.md)
- **Manifest:** [`artifacts/final_evaluation_manifest.json`](../artifacts/final_evaluation_manifest.json)

---

## 1. Execution runtime

| Property | Value |
|---|---|
| **Interpreter** | `/home/debalekha_chakraborty/AirSense/venv/bin/python` |
| **Python** | CPython **3.6.7** |
| **scikit-learn** | **0.20.0** |
| Preflight | **READY**, exit 0 |

numpy 1.15.4, pandas 0.23.4, scipy 1.1.0, scikit-learn 0.20.0,
matplotlib 3.0.2, seaborn 0.9.0 — all MATCH. System Python 3.11.2 not used.
No dependency changed.

## 2. Pre-test freeze verification

| Property | Value |
|---|---|
| `pretest_development_freeze.json` SHA-256 | `98568e88cccb6a64a6e52dc03a76b0e3b3e5b7d750ece3d81e3790fb0beb0bf9` |
| Expected | identical |
| Result | **MATCH** |
| Hashes independently verified | **51** |
| `final_test_status` in freeze | `sealed` |

51 = 16 referenced by the freeze + 22 referenced by the four development
manifests + 12 Phase-4/Phase-5 artifacts + the raw dataset. All matched.

## 3. Populations

| Population | Rows | Columns |
|---|---:|---:|
| `X_train` + `y_train` (2010–2012) | 24,418 | 43 |
| `X_validation` + `y_validation` (2013) | 8,678 | 43 |
| **Final development** | **33,096** | **43** |
| **Test predictors** (`X_test.csv`) | **8,661** | **43** |

Chronological concatenation, no shuffle, no further split, no feature
engineering re-run.

---

## 4. Stage A — blind final refit

| Model | Final specification | Structure |
|---|---|---|
| **M0** | training-median constant, recomputed on 2010–2013 | **constant = 73.0** |
| **M1** | `LinearRegression(fit_intercept=True)`, no scaling, no regularization | intercept 2435.4089340577; rank 44/44; condition 213,048 |
| **M2** | `DecisionTreeRegressor(criterion="mse", splitter="best", max_depth=12, min_samples_split=2, min_samples_leaf=10, max_features=None, random_state=42)` | depth 12; 1,787 nodes; 894 leaves |
| **M3** | `RandomForestRegressor(n_estimators=100, criterion="mse", bootstrap=True, max_depth=12, min_samples_split=2, min_samples_leaf=1, max_features="auto", random_state=42, n_jobs=1, oob_score=False)` | 100 trees; mean depth 12.00; mean 1,784.5 leaves; mean 3,567.9 nodes |

Each specification was read from the pre-test freeze and asserted
parameter-by-parameter. M0's constant was **recomputed**, not reused; the
2010–2013 median happens also to be 73.0.

### Stage-A artifacts and reproducibility

| Path | SHA-256 |
|---|---|
| `results/final_test/blind_final_predictions.csv` | `362b664ac840c450d2149e4bc02e0bf4f77b9d197adfb80dae4ad7470839e87d` |
| `artifacts/final_refit_manifest.json` | `847c35e7956799cc522c65053387cfe6a9221297dc82bb8d2dfc5e4d91ff53b8` |
| `results/final_test/refit/m1_final_coefficients.csv` | `943d4b2ed7bbebe2b00c459dc9b9824b677a72ded0520b717c5abf15d9ce80e3` |
| `results/final_test/refit/m2_final_feature_importances.csv` | `d3d0b82ca5ebf9b778fe28a05704adc8e70b8182766b5f0d5c73e9a10f733ba6` |
| `results/final_test/refit/m3_final_feature_importances.csv` | `0945405ead5207543da9febb0306f0b76fedb6b18f2678bce7fe224a5a2435a1` |
| `results/final_test/refit/final_refit_diagnostics.csv` | `47213a0308d11d9585937eb64c1436f49aa2da810222e9ffb3927ace8f264e22` |

**Stage A was run twice before the target was touched. All six artifacts
were byte-identical**, including the 100-tree forest's predictions.

Independent pre-opening checks: M0 emits one distinct value equal to the
development median; M1 reconstructs from `intercept + X_test · coefficients`
to 9.663 × 10⁻¹³; M2 and M3 are finite and inside the development target
range (M2 5.21–449.15, M3 5.78–459.90); all four vectors have 8,661 rows
with identifiers and timestamps aligned; the blind artifact contains no
`actual_pm25`, no `pm2.5`, no residual column.

`artifacts/final_refit_manifest.json` recorded `test_target_status: sealed`,
`test_metrics_status: not_computed`.

### Stage-A freeze

`artifacts/final_blind_prediction_freeze.json` — SHA-256
`4f59c710529e99b8c8a33517a34c503837d471871eb3f12ef987b34e41e7f1eb`.
`2014_target_status: sealed`, `blind_prediction_status: frozen`,
`final_models_status: frozen`. Its **7 referenced hashes were independently
verified**; it contains no metric-like token and no timestamp field.

**The target was still sealed when the predictions were frozen.**

### Stage-A quarantine audit

Token-level audit of `src/evaluation/final_refit.py` and
`scripts/prepare_final_predictions.py`: **zero executable references** to
`airsense_test_2014`, `y_test` or `actual_pm25`. The three textual mentions
of the Phase-4 test partition are the module docstring and the
`_forbidden_path_check` guard that aborts if that path is reached; every CSV
read in Stage A passes through it.

---

## 5. Test opening procedure

Gates in order: pre-test freeze (12 hashes re-verified in-script) → blind
freeze (7 hashes) → blind predictions verified → **no opening receipt
present** → single authorized read.

Stage B read `data/processed/airsense_test_2014.csv` with an explicit
`usecols` list of `No`, `year`, `month`, `day`, `hour`, `pm2.5` only.
Feature values were not loaded.

Verified before any metric: **8,661 rows**, 0 missing, 0 non-finite, and
**all 8,661 identifiers and timestamps matching the frozen predictions in
order**.

| Artifact | SHA-256 |
|---|---|
| `results/final_test/final_test_target_snapshot.csv` | `fc04c671599c3b09c9091518b6629d855b1d98de6e8ed462a3dc383372cf2314` |
| `artifacts/final_test_opening_receipt.json` | `4b57468b9b2ec4f211b0403fbcdd1c4d64f6c7902c4700f686edbe53c89617a8` |

A second Stage-B run confirmed the single-source-open policy: with the
receipt present it entered verification/report-only mode and **did not
reopen** the Phase-4 partition.

**Stage B fits nothing.** It imports no estimator and contains no `.fit(`
or `.predict(` — only `sklearn.metrics`.

---

## 6. 2014 held-out results

| Model | MAE | RMSE | R² | MAE rank |
|---|---:|---:|---:|---:|
| M0 | 65.417388292344995 | 96.741588760774960 | −0.069942695889054 | 4 |
| M1 | 52.900193501674970 | 73.103403258667839 | 0.389044922216543 | 3 |
| M2 | 51.248813500432064 | 75.522046099895277 | 0.347948973628056 | 2 |
| **M3** | **48.535341383781684** | **70.277632509152781** | **0.435364300995649** | **1** |

**Primary ranking by MAE: M3 < M2 < M1 < M0.**
**Best final model by MAE: M3.**

**Ranking disagreement, reported not resolved:** by RMSE and R² the order is
M3 < M1 < M2 < M0 — M1 beats M2 on both, while M2 beats M1 on MAE. MAE was
pre-registered as the decision metric and was not changed. M3 is best on all
three, so the headline does not depend on the choice.

## 7. Development-validation versus final test

| Model | Selected on 2013? | 2013 MAE | 2014 MAE | Δ | 2013 RMSE | 2014 RMSE | Δ | 2013 R² | 2014 R² | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | no | 67.591150 | 65.417388 | −2.173762 | 102.181753 | 96.741589 | −5.440164 | −0.085726 | −0.069943 | +0.015783 |
| M1 | no | 53.333361 | 52.900194 | −0.433167 | 76.558423 | 73.103403 | −3.455020 | 0.390520 | 0.389045 | −0.001475 |
| M2 | **yes** | 50.827853 | 51.248814 | **+0.420960** | 77.984115 | 75.522046 | −2.462069 | 0.367609 | 0.347949 | −0.019660 |
| M3 | **yes** | 47.938823 | 48.535341 | **+0.596518** | 73.090585 | 70.277633 | −2.812953 | 0.444484 | 0.435364 | −0.009120 |

**The selection-optimism pattern predicted in Phases 8 and 9 appeared
exactly as warned.** M0 and M1, which selected nothing on 2013, both did
*better* on 2014. M2 and M3, which used 2013 to choose hyperparameters, both
did *worse*. The optimism is ~1% of MAE — small enough that the ranking
survived, real enough to justify the locked test.

## 8. Aggregate residual summary

| Model | Mean | Median | Std | Min | Max | Negative predictions |
|---|---:|---:|---:|---:|---:|---:|
| M0 | +24.734557 | −1.000000 | 93.531528 | −71.000000 | +598.000000 | 0 |
| M1 | +0.072807 | −14.560446 | 73.107588 | −208.195503 | +520.049185 | **466 (5.38%)** |
| M2 | −2.809627 | −9.809524 | 75.474122 | −391.153846 | +515.544379 | 0 |
| M3 | −2.844763 | −11.567521 | 70.224087 | −364.183680 | +504.744036 | 0 |

Aggregate only. **No Phase-11 subgroup analysis was performed.**

---

## 9. Verification

### Independent 12-metric recomputation

All 4 models × MAE/RMSE/R² recomputed **without scikit-learn**, from the
frozen target snapshot and the frozen blind predictions — **the Phase-4
source was not reopened**. Worst absolute difference against both the
metrics CSV and the evaluation manifest: **1.421 × 10⁻¹⁴**, all within
1 × 10⁻¹².

### Blind-vs-final prediction identity

**All four prediction columns are string-identical and bitwise-identical**
between `blind_final_predictions.csv` and `final_test_predictions.csv`, as
are the identifier, timestamp and row-number columns.

**A defect was found and fixed here, and it is worth recording.** The first
Stage-B implementation re-serialised the parsed prediction floats, which
shifted the final digit on 3,500 of 8,661 M1 rows (and similarly for M2 and
M3) — a maximum numeric difference of **5.684 × 10⁻¹⁴**, about one ULP,
caused by pandas 0.23.4's `to_csv` not emitting shortest-round-trip floats.
The metrics were unaffected: they were computed from the parsed blind
values, and the independent recomputation confirmed agreement to 1e-12.
Rather than declare a serialization tolerance, Stage B was changed to carry
the prediction columns through **verbatim as strings**, and an in-script
identity gate now aborts if they ever differ. **No model, prediction or
metric changed** — only how an already-frozen number is written to a file.

### Grid, alignment and target integrity

8,661 rows · target 2.0–671.0 · 0 missing · 0 non-finite · `No` strictly
ascending · `matrix_row_number` sequential 0…8,660 · all four models scored
against the same snapshot.

### Python 3.6 compatibility

All project `.py` files compile under CPython 3.6.7. No post-3.6 syntax, no
post-cutoff API, no new dependency.

---

## 10. Phase-10 artifacts

| Path | SHA-256 |
|---|---|
| `artifacts/final_refit_manifest.json` | `847c35e7956799cc522c65053387cfe6a9221297dc82bb8d2dfc5e4d91ff53b8` |
| `artifacts/final_blind_prediction_freeze.json` | `4f59c710529e99b8c8a33517a34c503837d471871eb3f12ef987b34e41e7f1eb` |
| `artifacts/final_test_opening_receipt.json` | `4b57468b9b2ec4f211b0403fbcdd1c4d64f6c7902c4700f686edbe53c89617a8` |
| **`artifacts/final_evaluation_manifest.json`** | **`5f3270cc4d85781e6cea3bcd315649628c825a253e5cc0fa949627af1e7cb273`** |
| `results/final_test/blind_final_predictions.csv` | `362b664ac840c450d2149e4bc02e0bf4f77b9d197adfb80dae4ad7470839e87d` |
| `results/final_test/final_test_target_snapshot.csv` | `fc04c671599c3b09c9091518b6629d855b1d98de6e8ed462a3dc383372cf2314` |
| `results/final_test/final_test_predictions.csv` | `c09533f32406c5f3caf143631165ee4b0696fd7d5d7e589f308c9249e5617efd` |
| `results/final_test/final_test_metrics.csv` | `99c600ee87bcf27c818698316fc8e5e674e8bf12a16bae59847d8418235662ca` |
| `results/final_test/final_residual_summary.csv` | `f7b3d29917554dedf0dce20d3e223a00343b84397f497746de94afe4f1a2d139` |
| `results/final_test/development_to_test_comparison.csv` | `578bdb985876400e906cce158a8efc0d89dd566f1cdc2012012ad5b855c018cf` |
| `figures/final_test_model_comparison_mae.png` | `6bbc3ef49500a55efb0462ef45111055d6ed471ce14ae6c605f2dc603d8b4672` |
| `figures/final_test_actual_vs_predicted.png` | `72534d51b9e088d64d32b36ea4ea6293f5820731df79296ca52e20ff4fda2c1b` |
| `figures/final_test_rmse_r2.png` | `324e58a551a14ae1a14f46b670fa23d506485d8220b4545084d0911330aa19e6` |

## 11. Upstream integrity

Raw dataset, all Phase-4 partitions, all Phase-5 matrices, and all M0, M1,
M2 and M3 development artifacts re-verified after evaluation — **all
unchanged**. `pretest_development_freeze.json` and
`final_blind_prediction_freeze.json` both unchanged.

## 12. Files created and modified

**Created:** `src/evaluation/__init__.py`, `src/evaluation/final_refit.py`,
`scripts/prepare_final_predictions.py`, `scripts/open_final_test.py`, the
ten `results/final_test/` artifacts, the four Phase-10 `artifacts/*.json`,
the three `figures/final_test_*.png`,
`docs/FINAL_HELD_OUT_EVALUATION.md`, `docs/PHASE_08_FINAL_TEST_RECORD.md`.

**Modified:** `README.md`, and `docs/V1_RESEARCH_PROTOCOL.md` —
**administrative status only** (Phase 10 marked Complete). No methodological
declaration was altered.

---

## 13. Test-set exhaustion

**The 2014 target has been opened and is exhausted.** It is no longer an
untouched test set. Any model changed after this point must not claim a
fresh evaluation on the same 2014 split; a genuinely fresh evaluation would
require data outside 2010–2014. **The target cannot be resealed.**

## 14. Explicit confirmations

- **All development decisions were frozen before 2014 was opened.**
- **`pretest_development_freeze.json` matched its expected hash exactly.**
- **M0–M3 final refits used exactly 2010–2013.**
- **Final development row count was 33,096.**
- **All feature-based models used exactly the 43 frozen features.**
- **M2 used the previously selected parameters only.**
- **M3 used the previously selected parameters only.**
- **No hyperparameter search occurred in Phase 10.**
- **Stage-A blind predictions existed and were frozen before target access.**
- **Stage-A predictions were deterministic across two runs.**
- **The 2014 target was not accessed during Stage A.**
- **Stage B fitted no model** — no estimator import, no `.fit(`, no
  `.predict(`.
- **Final predictions were not changed after target opening** — verified
  byte-identical to the frozen blind artifact.
- **The same 8,661 test observations scored all four models.**
- **MAE remained the primary ranking metric.**
- **No model was dropped; no additional model was introduced.**
- **No post-test retraining occurred.**
- **No feature change occurred.**
- **No target transformation occurred.**
- **No random split occurred.**
- **No detailed Phase-11 error analysis occurred.**
- **The 2014 test is now exhausted and cannot honestly be called unseen
  again.**
- **All pre-test artifacts remained unchanged.**
- **No dependency changed.**
- **No git write operation was performed; history was not rewritten.**

## 15. Status

Protocol Phase 10 is **complete**. All legitimate changes are left
**unstaged**.

The next phase is Protocol Phase 11 — error analysis. **It has not begun.**

Awaiting instruction.
