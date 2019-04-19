# Phase 5 M1 Linear Regression Record — AirSense V1

**Research-protocol mapping: Protocol Phase 7 — Linear Regression (M1).**

> Record-sequence numbering (`PHASE_05_`) is independent of protocol phase
> numbering. Earlier records were not renamed. Mapping:
> `PHASE_00_FOUNDATION_RECORD.md` (protocol 1–2),
> `PHASE_00A_RUNTIME_RECORD.md` (runtime recovery),
> `PHASE_01_EDA_RECORD.md` (protocol 3),
> `PHASE_02_CLEANING_RECORD.md` (protocol 4),
> `PHASE_03_FEATURE_RECORD.md` (protocol 5),
> `PHASE_04_M0_BASELINE_RECORD.md` (protocol 6),
> this record (protocol 7).

- **Recorded:** 2026-09-05 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE**
- **Decision document:** [`M1_LINEAR_REGRESSION.md`](M1_LINEAR_REGRESSION.md)
- **Manifest:** [`artifacts/m1_linear_regression_manifest.json`](../artifacts/m1_linear_regression_manifest.json)

> **2014 was not evaluated.** No test feature, no test target, no 2014
> prediction and no 2014 metric.

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
not used. No dependency added, removed, upgraded or downgraded; **statsmodels
was not installed for VIF**.

---

## 2. Upstream integrity

Verified before the phase and again afterwards.

| Input | Result |
|---|---|
| Raw dataset SHA-256 `4127f868…6d27f1c` | MATCH |
| 4 Phase-4 partitions vs `split_manifest.json` | all MATCH |
| 8 Phase-5 outputs vs `feature_schema.json` | all MATCH |
| 2 M0 artifacts vs `m0_baseline_manifest.json` | all MATCH |
| `X_train` shape (24,418 × 43) | MATCH |
| `X_validation` shape (8,678 × 43) | MATCH |
| `y_train` 24,418 · `y_validation` 8,678 | MATCH |
| `list(X_train.columns) == list(X_validation.columns)` | MATCH |
| Columns == `feature_schema.feature_names` | MATCH |
| Feature count == 43 | MATCH |
| `y_test.csv` absent | confirmed |

No earlier phase was regenerated. **M0 was not altered.**

---

## 3. Model specification

```
LinearRegression(fit_intercept=True)
```

| Property | Value |
|---|---|
| Estimator | `sklearn.linear_model.LinearRegression` |
| `fit_intercept` | **True** |
| Other options | none set |
| Features | the frozen 43, consumed as produced |
| Fit period | 2010-01-01 00:00 → 2012-12-31 23:00 (24,418 rows) |
| Evaluation period | 2013-01-01 00:00 → 2013-12-31 23:00 (8,678 rows) |
| Hyperparameter search | **none** |
| Feature ablation | **none** |
| Scaling | **none** |
| Regularization | **none** |

---

## 4. Training-design diagnostics

Computed on the **training matrix only**, via `numpy.linalg.svd`.
Condition number = largest singular value / smallest non-zero singular
value, with the non-zero threshold `max(m,n)·eps·σ_max`.

| Matrix | Shape | Rank | Deficient | σ_max | σ_min(nz) | Condition |
|---|---|---:|---|---:|---:|---:|
| `[1\|X]` (solved design) | 24,418 × 44 | **44** | **no** | 158,931 | 0.748356 | **212,373** |
| Raw `X_train` | 24,418 × 43 | 43 | no | 158,931 | 6.20043 | 25,632.2 |
| Centred `X_train` | 24,418 × 43 | 43 | no | 8,176.89 | 6.20024 | 1,318.8 |

scikit-learn attributes agree with the centred row: `rank_ = 43`, 43
singular values, largest 8,176.892470, smallest 6.200238.

**Full rank — not rank deficient.** The design is nonetheless
ill-conditioned at 2.1 × 10⁵, driven by `PRES` (~1016 hPa) sitting
alongside 0/1 dummies. Reported, not "fixed".

**Intercept: 2491.5560610308**

---

## 5. Validation results

**2013 DEVELOPMENT VALIDATION — not final held-out test performance.**

| Metric | M1 |
|---|---:|
| **MAE** | **53.333360626133683** |
| **RMSE** | **76.558422908942688** |
| **R²** | **0.390520035606485** |

## 6. Exact M0 comparison

M0 values read from `artifacts/m0_baseline_manifest.json`, not hard-coded.

| Metric | M0 | M1 | Absolute | Relative |
|---|---:|---:|---:|---:|
| **MAE** *(primary)* | 67.591150034570177 | 53.333360626133683 | **−14.257789408436494** | **−21.0942%** |
| RMSE | 102.181752894378718 | 76.558422908942688 | −25.623329985436030 | −25.0762% |
| R² | −0.085725839256606 | 0.390520035606485 | **+0.476245874863091** | — |

M0 baseline constant: 73.0. The primary comparison metric remains **MAE**;
it was not switched because another metric looked more favourable.

---

## 7. Prediction and residual diagnostics

| Property | Value |
|---|---:|
| Predictions | 8,678 |
| Distinct predictions | **8,675** |
| Prediction std. dev. | 59.942416 |
| Prediction range | −150.5374 … +264.5828 |
| Identical to M0 constant | **no** |
| **Predictions below 0 µg/m³** | **658 (7.5824%)** |

Residuals, defined as `actual − predicted`:

| Statistic | Value |
|---|---:|
| Mean | +6.560091 |
| Median | −7.353553 |
| Std. dev. | 76.281242 |
| Minimum | −175.470247 |
| Maximum | **+687.220866** |
| Pearson corr(predicted, residual) | +0.022515 |

Mean and median have opposite signs: M1 over-predicts the typical hour and
badly under-predicts severe episodes. 658 physically impossible negative
predictions were **retained, not clipped** — clipping would improve the
metrics and would be an undeclared post-hoc change. High-concentration
validation observations were not removed or separately refitted around.

---

## 8. Artifacts

| Path | SHA-256 | Rows |
|---|---|---:|
| `results/models/m1/m1_coefficients.csv` | `1e8bc028c9ea45cc5359e5f05ea4d6a431c604295a208b6df431874b1bcdf11b` | 43 |
| `results/models/m1/m1_training_diagnostics.csv` | `6d51c76e0b25f7d40a90735628a29662c5d7e3909db81b795455b9c58a5ab924` | 1 |
| `results/models/m1/m1_validation_metrics.csv` | `e05ff31ba284caaec24b77a9219f8981a13e0ed4e3bdce75901ce7ed3140c256` | 1 |
| `results/models/m1/m1_validation_predictions.csv` | `3310f6c083cb0972b39c8e54cf4dd586b0342201f7736b58c58a4e6e4c316ab5` | 8,678 |
| `artifacts/m1_linear_regression_manifest.json` | `395abc4cc8d9a749274d58160a54e372647f6033f78584208219bf8fb27cdfef` | — |
| `figures/m1_validation_actual_vs_predicted.png` | `6bf650deb1930c5a8a1423a97b3324599f363d1b8f1cab8469f7f45d266027a6` | — |
| `figures/m1_validation_residuals_vs_predicted.png` | `eb96f05ab578e3b6207550ec171c722eac3300bd70ab8abb6b3ebb67ff3c3bae` | — |

The coefficient file holds 43 predictor rows in the **frozen feature order,
not sorted by value**; the intercept lives in the diagnostics table and the
manifest. **No serialized model was written** — Phase 10 refits M1 on
2010–2013, so a pickled development model would not be the final model.

---

## 9. Verification

### Independent equation reconstruction

`intercept + X_validation · coefficients` rebuilt from the **coefficient
artifact alone** and compared with `model.predict`:

| Measure | Value |
|---|---|
| Max absolute difference | **9.379 × 10⁻¹³** |
| Tolerance | 1 × 10⁻¹⁰ |
| Result | **AGREES** |

This proves the saved coefficients completely represent the fitted linear
equation — nothing dropped, reordered or rescaled.

### Independent metric recomputation

Recomputed from `m1_validation_predictions.csv` **without scikit-learn**:

| Metric | Pipeline | Recomputed | Abs. difference |
|---|---|---|---:|
| MAE | 53.333360626133683 | 53.333360626133675 | 7.105e-15 |
| RMSE | 76.558422908942688 | 76.558422908942688 | 0.000e+00 |
| R² | 0.390520035606485 | 0.390520035606485 | 0.000e+00 |

All within the 1e-12 tolerance; the manifest agrees.

### Coefficient finiteness

43 coefficients + 1 intercept · no NaN · no infinity · none missing · order
matches the feature schema exactly.

### Prediction and row alignment

8,678 predictions == 8,678 actuals == 8,678 manifest rows · all finite ·
actuals equal the frozen `y_validation` and the Phase-4 source ·
`matrix_row_number` sequential 0…8,677 · `No` strictly ascending and
matching both the row manifest and the Phase-4 partition · ≥ 2 distinct
predictions.

**One precision note.** A bitwise-exact comparison of the stored `residual`
column against `actual − predicted` recomputed from the same CSV shows
differences on 4,772 rows, maximum **1.137 × 10⁻¹³** absolute
(3.5 × 10⁻¹³ relative). This is a decimal round-trip artifact — the column
is the CSV representation of a full-precision difference, while the
recomputation subtracts two separately round-tripped values. The sign
convention is identical and agreement holds well within 1e-9. Recorded
rather than glossed over.

### Deterministic rerun

M1 was run twice; **all 7 artifacts byte-identical**, including both PNGs.
Ordinary least squares has no random component, figures pin their PNG
`Software` metadata, and no execution timestamp is written anywhere.

### Python 3.6 compatibility

All project `.py` files compile under CPython 3.6.7. No post-3.6 syntax, no
post-cutoff API, no new dependency.

---

## 10. Test-quarantine verification

| Check | Result |
|---|---|
| Executable references to `X_test`/`y_test`/test partition/`m1_test`/2014 | **NONE** (token-level audit of both new files) |
| `y_test.csv` present | no |
| Files matching `*m1_test*`, `*test_prediction*`, `*y_test*` | **0** |
| `results/models/m1/` contents | 4 development artifacts only |
| Manifest `test_features_read` / `test_target_read` | `false` / `false` |
| Manifest `test_predictions_generated` | `false` |
| Manifest `test_evaluation_status` / `final_test_status` | `not_evaluated` |
| Placeholder 2014 number | none |

Every textual occurrence of "2014" in the new source is inside a comment,
docstring or string; the one path string containing it is the raw dataset's
own filename `PRSA_data_2010.1.1-2014.12.31.csv`.

---

## 11. Files created and modified

**Created:** `src/models/linear_regression.py`,
`scripts/run_m1_linear_regression.py`, the four
`results/models/m1/` artifacts, `artifacts/m1_linear_regression_manifest.json`,
the two `figures/m1_validation_*.png`, `docs/M1_LINEAR_REGRESSION.md`,
`docs/PHASE_05_M1_LINEAR_REGRESSION_RECORD.md`.

**Modified:** `README.md` only.

**Not modified by this phase:** `V1_RESEARCH_PROTOCOL.md` (it still shows as
modified in `git status` from the uncommitted Phase-4 administrative status
edit; this phase required and made no protocol change), `FEATURE_POLICY.md`,
`FEATURE_PREPARATION.md`, `CLEANING_AND_SPLIT_POLICY.md`, `M0_BASELINE.md`,
`RESEARCH_QUESTION.md`, `EDA_ANALYSIS.md`, both requirements files,
`data/raw/`, the Phase-4 partitions, the Phase-5 matrices and all M0
artifacts.

---

## 12. Warnings

**W1 — Ill-conditioned design.** Condition number 2.1 × 10⁵ with the
intercept column. Full rank, but individual coefficients are numerically
fragile. Reported, not corrected.

**W2 — 658 negative predictions (7.58%).** Physically impossible for a
concentration. An unavoidable consequence of unconstrained OLS on a
non-negative target; retained deliberately.

**W3 — Severe episodes badly under-predicted.** Maximum residual
+687 µg/m³. Full analysis belongs to Phase 11.

**W4 — Coefficients are not an importance ranking.** Strong
multicollinearity (Phase 3: |r| 0.78–0.83 among `DEWP`/`TEMP`/`PRES`),
unscaled heterogeneous units, and reference-relative dummies. The
coefficient file is stored in frozen order, unsorted, for this reason.

**No unresolved blockers.**

---

## 13. Explicit confirmations

- **M1 is ordinary `sklearn.linear_model.LinearRegression`.**
- **`fit_intercept=True`.**
- **M1 uses exactly the frozen 43 features**, verified against
  `feature_schema.json` before fitting.
- **M1 was fitted only on 2010–2012.**
- **Validation uses only 2013.**
- **The primary comparison remains MAE.**
- **The feature schema was not changed after seeing results.**
- **No scaling was introduced.**
- **No regularization was introduced** — no Ridge, Lasso or ElasticNet.
- **No feature was removed because of multicollinearity.**
- **Coefficient interpretation is treated cautiously** and no causal or
  importance claim is made.
- **No hyperparameter search occurred.**
- **No feature ablation occurred.**
- **`X_test` was not read.**
- **The 2014 target was never read.**
- **No 2014 prediction was generated.**
- **No 2014 metric was calculated.**
- **No `y_test` artifact exists.**
- **No M2 Decision Tree was trained.**
- **No M3 Random Forest was trained.**
- **No upstream artifact changed.**
- **No dependency changed.**
- **No git write operation was performed.**
- **Git history was not rewritten.**

---

## 14. Status

Protocol Phase 7 is **complete**. All legitimate changes are left
**unstaged**.

The next phase is Protocol Phase 8 — M2 Decision Tree Regressor. Nothing
about it has been implemented.

Awaiting instruction.
