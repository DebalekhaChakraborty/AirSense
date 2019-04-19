# Phase 7 M3 Random Forest Record — AirSense V1

**Research-protocol mapping: Protocol Phase 9 — Random Forest Regression (M3).**

> Record-sequence numbering (`PHASE_07_`) is independent of protocol phase
> numbering. Earlier records were not renamed. Mapping:
> `PHASE_00_FOUNDATION_RECORD.md` (protocol 1–2),
> `PHASE_00A_RUNTIME_RECORD.md` (runtime recovery),
> `PHASE_01_EDA_RECORD.md` (protocol 3),
> `PHASE_02_CLEANING_RECORD.md` (protocol 4),
> `PHASE_03_FEATURE_RECORD.md` (protocol 5),
> `PHASE_04_M0_BASELINE_RECORD.md` (protocol 6),
> `PHASE_05_M1_LINEAR_REGRESSION_RECORD.md` (protocol 7),
> `PHASE_06_M2_DECISION_TREE_RECORD.md` (protocol 8),
> this record (protocol 9).

- **Recorded:** 2026-09-06 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE** — final development-model phase
- **Decision document:** [`M3_RANDOM_FOREST.md`](M3_RANDOM_FOREST.md)
- **Manifest:** [`artifacts/m3_random_forest_manifest.json`](../artifacts/m3_random_forest_manifest.json)
- **Pre-test freeze:** [`artifacts/pretest_development_freeze.json`](../artifacts/pretest_development_freeze.json)

> **2014 was not evaluated, was never read, and did not influence
> selection. Protocol Phase 10 has not started.**

---

## 1. Execution runtime

| Property | Value |
|---|---|
| **Interpreter** | `/home/debalekha_chakraborty/AirSense/venv/bin/python` |
| **Python** | CPython **3.6.7** |
| **scikit-learn** | **0.20.0** |
| Preflight | **READY**, exit 0 |

Pins verified: numpy 1.15.4, pandas 0.23.4, scipy 1.1.0, scikit-learn
0.20.0, matplotlib 3.0.2, seaborn 0.9.0 — all MATCH. System Python 3.11.2
not used. No dependency added, removed, upgraded or downgraded.

---

## 2. Upstream integrity

**36 checks**, run before the phase and again afterwards.

| Input | Count | Result |
|---|---:|---|
| Raw dataset SHA-256 `4127f868…6d27f1c` | 1 | MATCH |
| Phase-4 partitions vs `split_manifest.json` | 4 | all MATCH |
| Phase-5 outputs vs `feature_schema.json` | 8 | all MATCH |
| M0 artifacts vs its manifest | 2 | all MATCH |
| M1 artifacts vs its manifest | 6 | all MATCH |
| M2 artifacts vs its manifest | 7 | all MATCH |
| Shapes, column order, schema, `y_test` absence | 8 | all MATCH |

`X_train` (24,418 × 43), `X_validation` (8,678 × 43), `y_train` 24,418,
`y_validation` 8,678, column order identical and equal to the frozen
43-column schema. No earlier phase was regenerated. **M0, M1 and M2 were not
altered.**

---

## 3. Search space — frozen before any M3 result

**Estimator:** `sklearn.ensemble.RandomForestRegressor`, scikit-learn 0.20.0.

**Searched — 3 × 4 × 2 = 24 candidates:**

| Parameter | Values |
|---|---|
| `max_depth` | 8, 12, None |
| `min_samples_leaf` | 1, 10, 50, 100 |
| `max_features` | "auto", 0.5 |

**Fixed, not searched:** `n_estimators=100`, `criterion="mse"` (the 0.20
name; `"squared_error"` did not exist until 1.0), `bootstrap=True`,
`min_samples_split=2`, `random_state=42`, `n_jobs=1`, `oob_score=False`.
Also not searched: `max_leaf_nodes`, `min_impurity_decrease`.

**Candidate order:** `max_depth` outer, `min_samples_leaf` middle,
`max_features` inner; `candidate_id` 1…24. **Method:** explicit
deterministic loop — `GridSearchCV`, `RandomizedSearchCV`, `KFold`,
`ShuffleSplit` and `cross_val_score` were all deliberately avoided, since
the design contains an explicit chronological validation year.

**`n_estimators` fixed at 100** so selection concerns tree structure and
feature subsampling, not ensemble size. **One seed only** (`random_state=42`,
`n_jobs=1`); no best-of-seed reporting. **`bootstrap=True` throughout.**
**OOB scoring off and not used for selection** — it would introduce a second
selection framework ignoring temporal ordering.

## 4. Selection and tie-break policy

**Primary: lowest 2013 validation MAE**, the Phase-0 decision metric.
Tie-break, frozen before fitting, applied only within 1 × 10⁻¹²:

1. lower validation RMSE;
2. smaller finite `max_depth` (`None` more complex than any finite value);
3. larger `min_samples_leaf`;
4. `max_features=0.5` before `"auto"`;
5. deterministic grid order.

R² never overrides. Training metrics were diagnostics only. **Exactly one
candidate tied on MAE — no tie-break was required.**

---

## 5. Selected configuration

| Property | Value |
|---|---|
| **Candidate ID** | **9** |
| `n_estimators` | 100 |
| `criterion` | `"mse"` |
| `bootstrap` | `True` |
| **`max_depth`** | **12** |
| `min_samples_split` | 2 |
| **`min_samples_leaf`** | **1** |
| **`max_features`** | **`"auto"`** |
| `random_state` | 42 |
| `n_jobs` | 1 |
| `oob_score` | `False` |

**A selection worth recording.** Candidate 11 achieved a *better* RMSE
(72.958450) and *better* R² (0.446491) than candidate 9 (73.090585,
0.444484). It was not selected because its MAE (47.962390) is higher than
candidate 9's (47.938823) — a margin of 0.024 µg/m³ — and **MAE is the
pre-registered decision metric**. The rule was applied as written.

### Forest structure — all 100 estimators

| Statistic | Min | Mean | Median | Max |
|---|---:|---:|---:|---:|
| Tree depth | 12 | 12.00 | 12.0 | 12 |
| Leaf count | 1,525 | 1,662.43 | 1,662.0 | 1,827 |
| Node count | 3,049 | 3,323.86 | 3,323.0 | 3,653 |

**Tree-count verification: 100 estimators fitted and verified.** Constraint
verification confirms all 100 trees respect `max_depth=12` and
`min_samples_leaf=1`. No tree was visualised.

---

## 6. Metrics

**Training (2010–2012) — diagnostic only, played no part in selection:**

| Metric | Value |
|---|---:|
| MAE | 32.181851771139407 |
| RMSE | 47.486814168837370 |
| R² | 0.716890505849367 |

**2013 DEVELOPMENT VALIDATION — not final held-out test performance:**

| Metric | Value |
|---|---:|
| **MAE** | **47.938822971054385** |
| **RMSE** | **73.090585077540226** |
| **R²** | **0.444484279813415** |

Train → validation MAE degradation: +15.756971 (×1.490).

## 7. Comparison with M0, M1 and M2

Prior values read from their manifests, not hard-coded. Sign convention
identical for all three: `improvement = prior − M3` (positive = M3 better);
`r2_delta = M3 − prior`.

| Metric | M0 | M1 | M2 | **M3** |
|---|---:|---:|---:|---:|
| MAE | 67.591150 | 53.333361 | 50.827853 | **47.938823** |
| RMSE | 102.181753 | 76.558423 | 77.984115 | **73.090585** |
| R² | −0.085726 | 0.390520 | 0.367609 | **0.444484** |

| Versus | MAE improvement | MAE % | RMSE improvement | RMSE % | R² delta |
|---|---:|---:|---:|---:|---:|
| **M0** | +19.652327063515791 | **+29.075296%** | +29.091167816838492 | +28.470022% | +0.530210119070021 |
| **M1** | +5.394537655079290 | **+10.114753%** | +3.467837831402463 | +4.529662% | +0.053964244206930 |
| **M2** | +2.889030375108668 | **+5.683951%** | +4.893529914837558 | +6.275034% | +0.076875413511532 |

**M3 is better than M0, M1 and M2 on all three metrics simultaneously** —
the only model in the project for which that holds. It **repairs the defect
M2 introduced**: M2 had beaten M1 on MAE while losing on RMSE (+1.43) and R²
(−0.023); M3 beats M1 on both.

---

## 8. Residual diagnostics and prediction range

| Statistic | Value |
|---|---:|
| Mean residual | +3.127329 |
| Median residual | −9.020883 |
| Residual std. dev. | 73.027858 |
| Minimum residual | −392.789071 |
| Maximum residual | +688.119063 |
| Pearson corr(predicted, residual) | **+0.001759** |
| **Negative predictions** | **0 (0.0000%)** |
| Prediction range | 10.312080 … 431.094649 |
| Training target range | 0.0 … 994.0 |
| **Predictions within training range** | **yes** |
| Predictions clipped | **no** |
| Distinct predictions | **7,632** (M2: 628) |

Residual spread is the tightest of any model, and the predicted-vs-residual
correlation (+0.0018) is the closest to zero — averaging has nearly removed
M2's −0.159 regression-to-leaf-mean pattern. Severe episodes remain hard
(max residual +688 µg/m³); **the selection metric was not changed to favour
extreme-error performance.**

## 9. Feature-importance validation

43 rows in frozen feature order, sum **1.000000000000000**, **all 43
non-zero** (M2: 31 of 43). Top: `DEWP` 0.290604, `TEMP` 0.157607,
`Iws` 0.120227, `PRES` 0.108188, `month_10` 0.078892, `cbwd_NW` 0.052027.

Impurity-based importance is predictive, not causal; it favours
continuous/high-cardinality variables and divides credit unpredictably among
the correlated `DEWP`/`TEMP`/`PRES` group. **No feature selection was
performed from it and the Phase 5 schema was not altered.**

---

## 10. Artifacts

| Path | SHA-256 | Rows |
|---|---|---:|
| `results/models/m3/m3_grid_results.csv` | `80b7c6957ff1370fb88db5eb3d07fe5167c882256ee1a1718a7a4cf973fd6200` | 24 |
| `results/models/m3/m3_validation_metrics.csv` | `349f6757a8a3088b05a495e517d7120de529b5b4f091d9fe9167ea045a7285e3` | 1 |
| `results/models/m3/m3_validation_predictions.csv` | `e88de20474616f95ed9e757feedc6f345ab6ed4f293e1ee0759ef896c765d5ce` | 8,678 |
| `results/models/m3/m3_feature_importances.csv` | `2670dbbeaf872fe515dd3e96c16186e9bfd9f728f7e5b8e32384466064d022fd` | 43 |
| `artifacts/m3_random_forest_manifest.json` | `04ea281d02afcf7a702381dc743d7236b64b6e2377aed9f0c31bf7f12ece8a8e` | — |
| `figures/m3_validation_actual_vs_predicted.png` | `2229b7bbf638d166558b51a2650d13b72679d8bc89250a361f5d56173f0e7e46` | — |
| `figures/m3_validation_residuals_vs_predicted.png` | `98d9ffc2c51fb1c20c3f9c34f23d6c738aacb7dac15f30b0c62bc5fb95247475` | — |
| `figures/m3_grid_mae_by_complexity.png` | `b81a1bd88dfff1d024e646fa31b320d5993bfa2f3a59b8efb95d6037c3c7438e` | — |
| **`artifacts/pretest_development_freeze.json`** | `98568e88cccb6a64a6e52dc03a76b0e3b3e5b7d750ece3d81e3790fb0beb0bf9` | — |

**No serialized model** — Phase 10 refits on 2010–2013.

---

## 11. Verification

### Independent metric recomputation

From `m3_validation_predictions.csv`, **without scikit-learn**:

| Metric | Pipeline | Recomputed | Abs. difference |
|---|---|---|---:|
| MAE | 47.938822971054392 | 47.938822971054385 | 7.105e-15 |
| RMSE | 73.090585077540240 | 73.090585077540226 | 1.421e-14 |
| R² | 0.444484279813415 | 0.444484279813415 | 0.000e+00 |

All within 1e-12; the manifest agrees.

### Independent selection recomputation

The frozen selection rule was **re-implemented from scratch** against
`m3_grid_results.csv` alone: best validation MAE 47.938822971054392,
**1 candidate tied within 1e-12**, independently selected **candidate 9** —
**matching the manifest**.

### Grid invariants

24 rows · 24 unique combinations · **combinations exactly equal the declared
grid** (no missing, no undeclared) · exactly one `selected=True` · every
candidate has `n_estimators=100`, `criterion="mse"`, `bootstrap=True`,
`min_samples_split=2`, `random_state=42`, `n_jobs=1` · all train and
validation metrics and all structural statistics finite.

### Prediction and row alignment

8,678 rows · actuals equal frozen `y_validation` **and** the Phase-4 source ·
predictions and residuals finite · residual == actual − predicted within
1e-9 · `matrix_row_number` sequential 0…8,677 · `No` aligned to both the row
manifest and the Phase-4 partition, strictly ascending · 7,632 distinct
predictions (≥ 2) · predictions inside the training target range · no
clipping.

### Deterministic rerun

M3 was run twice end to end; **all 8 artifacts byte-identical**, including
all three PNGs, and run 2 independently selected candidate 9 with identical
metrics. `random_state=42` and `n_jobs=1` on every candidate, figures pin
their PNG `Software` metadata, and no execution timestamp is written
anywhere. The pre-test freeze is likewise byte-identical across reruns.

### Python 3.6 compatibility

All project `.py` files compile under CPython 3.6.7. No post-3.6 syntax, no
post-cutoff API, no new dependency.

---

## 12. Test-quarantine verification

| Check | Result |
|---|---|
| Executable references to `X_test`/`y_test`/test partition/`m3_test`/2014 | **NONE** (token-level audit of both new files) |
| `y_test.csv` present | no |
| Files matching `*m3_test*`, `*test_prediction*`, `*y_test*` | **0** |
| `results/models/m3/` contents | 4 development artifacts only |
| Manifest `test_features_read` / `test_target_read` | `false` / `false` |
| Manifest `test_predictions_generated` | `false` |
| Manifest `selection_used_test_information` | **`false`** |
| Manifest `test_evaluation_status` / `final_test_status` | `not_evaluated` |
| Placeholder 2014 number | none |

Every textual occurrence of "2014" in the new source is a comment, docstring
or string; the one path string containing it is the raw dataset's own
filename. **No further hyperparameter search was performed after
selection** — no neighbouring depth, leaf or `max_features` value, no second
seed, no larger forest, no `bootstrap=False`.

---

## 13. Pre-test development freeze

Created **only after** M3 was selected, validated and shown reproducible.

| Property | Value |
|---|---|
| Path | `artifacts/pretest_development_freeze.json` |
| SHA-256 | `98568e88cccb6a64a6e52dc03a76b0e3b3e5b7d750ece3d81e3790fb0beb0bf9` |
| `freeze_type` | `pre_final_test` |
| `final_test_period` | `2014` |
| **`final_test_status`** | **`sealed`** |
| `test_opening_policy` | `Protocol Phase 10 only` |
| Models sealed | M0, M1, M2, M3 |
| Primary final metric | MAE |
| Secondary metrics | RMSE, R² |
| Generator | `scripts/create_pretest_freeze.py` |

**Independent verification: 16 referenced hashes re-verified from disk with
a separately implemented SHA-256 routine — all consistent.** No forbidden
2014 token (`y_test`, `test_mae`, `test_rmse`, `test_r2`, `test_metric`)
appears anywhere in the file; no volatile timestamp field exists; the file
is byte-identical across reruns.

Phase 10 can begin by validating this one artifact and know the entire
pre-test development state is unchanged.

### Immutability statement

**Once this freeze exists, Protocol Phase 10 may not change:** the feature
schema (the ordered 43 columns), the M0 strategy, the M1 specification, the
M2 selected hyperparameters, the M3 selected hyperparameters, the split
boundaries, the primary metric (MAE), or the test policy.

**If a later defect requires changing any of those, the planned final-test
evaluation must be invalidated and reset, and the deviation explicitly
documented, BEFORE the test is reopened. Nothing may be silently repaired
after seeing 2014.**

---

## 14. Files created and modified

**Created:** `src/models/random_forest.py`,
`scripts/run_m3_random_forest.py`, `scripts/create_pretest_freeze.py`, the
four `results/models/m3/` artifacts,
`artifacts/m3_random_forest_manifest.json`,
`artifacts/pretest_development_freeze.json`, the three `figures/m3_*.png`,
`docs/M3_RANDOM_FOREST.md`, `docs/PHASE_07_M3_RANDOM_FOREST_RECORD.md`.

**Modified:** `README.md` only.

**Not modified by this phase:** `V1_RESEARCH_PROTOCOL.md` (still shows as
modified in `git status` from the uncommitted Phase-4 administrative status
edit; this phase required and made no protocol change), `FEATURE_POLICY.md`,
`FEATURE_PREPARATION.md`, `CLEANING_AND_SPLIT_POLICY.md`, `M0_BASELINE.md`,
`M1_LINEAR_REGRESSION.md`, `M2_DECISION_TREE.md`, `RESEARCH_QUESTION.md`,
`EDA_ANALYSIS.md`, both requirements files, `data/raw/`, the Phase-4
partitions, the Phase-5 matrices, and all M0, M1 and M2 artifacts.

---

## 15. Warnings

**W1 — 2013 was used to select M3's hyperparameters** (as it was for M2), so
M3's 2013 figure is optimistic relative to M0 and M1, which selected
nothing. M3's apparent margin over M1 should not be taken at face value
until the locked test is opened. This must be restated when Phase 10 results
are reported.

**W2 — The MAE/RMSE winner differs within the grid.** Candidate 11 has
better RMSE and R² than the selected candidate 9. The pre-registered metric
decided; both readings are recorded.

**W3 — Interpretability cost.** 100 trees with ~166,000 leaves cannot be
inspected like M1's coefficients or M2's single tree; impurity importances
are a weak substitute.

**W4 — Severe episodes remain badly under-predicted** (max residual
+688 µg/m³). Full analysis belongs to Phase 11.

**W5 — Wider train/validation gap than M2** (×1.49 vs ×1.35), a consequence
of `min_samples_leaf=1`. It did not harm validation performance.

**No unresolved blockers.**

---

## 16. Explicit confirmations

- **M3 is `sklearn.ensemble.RandomForestRegressor`.**
- **Exactly 100 estimators per candidate**, verified on the selected forest.
- **Exactly 24 candidates were evaluated**, matching the declared grid
  exactly.
- **The complete M3 grid was frozen before any M3 result.**
- **`criterion="mse"`**, **`bootstrap=True`**, **`random_state=42`**,
  **`n_jobs=1`** for every candidate.
- **OOB selection was not used**; `oob_score=False`.
- **The selected candidate was chosen by 2013 validation MAE.**
- **RMSE was only the first tie-break** — and was not needed.
- **No test information influenced selection.**
- **M3 uses the same frozen 43 features as M1 and M2.**
- **No feature or target transformation occurred.**
- **No post-hoc hyperparameter was tried.**
- **No additional seed was tried.**
- **M3 was fitted only on 2010–2012.**
- **Validation used only 2013.**
- **`X_test` was not read.**
- **The 2014 target was never read.**
- **No 2014 prediction was generated.**
- **No 2014 metric was calculated.**
- **No `y_test` artifact exists.**
- **M0, M1 and M2 remained unchanged.**
- **`pretest_development_freeze.json` was created only after M3
  selection**, validation and reproducibility verification.
- **The 2014 final test remains sealed.**
- **Protocol Phase 10 has not started.**
- **No upstream artifact changed.**
- **No dependency changed.**
- **No git write operation was performed.**
- **Git history was not rewritten.**

---

## 17. Status

Protocol Phase 9 is **complete**, and with it the development programme. All
legitimate changes are left **unstaged**.

The next phase is Protocol Phase 10 — held-out evaluation of M0–M3 on the
locked 2014 partition, once. **It has not begun.**

Awaiting instruction.
