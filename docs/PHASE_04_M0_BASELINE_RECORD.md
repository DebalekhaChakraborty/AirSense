# Phase 4 M0 Baseline Record — AirSense V1

**Research-protocol mapping: Protocol Phase 6 — Baseline Regression (M0).**

> Record-sequence numbering (`PHASE_04_`) is independent of protocol phase
> numbering. Earlier records were not renamed. Mapping:
> `PHASE_00_FOUNDATION_RECORD.md` (protocol 1–2),
> `PHASE_00A_RUNTIME_RECORD.md` (runtime recovery),
> `PHASE_01_EDA_RECORD.md` (protocol 3),
> `PHASE_02_CLEANING_RECORD.md` (protocol 4),
> `PHASE_03_FEATURE_RECORD.md` (protocol 5),
> this record (protocol 6).

- **Recorded:** 2026-09-05 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE**
- **Decision document:** [`M0_BASELINE.md`](M0_BASELINE.md)
- **Manifest:** [`artifacts/m0_baseline_manifest.json`](../artifacts/m0_baseline_manifest.json)

> **2014 was not evaluated.** The test target was never read, no 2014
> prediction was generated, and no 2014 metric exists.

---

## 1. Execution runtime

| Property | Value |
|---|---|
| **Interpreter** | `/home/debalekha_chakraborty/AirSense/venv/bin/python` |
| **Python** | CPython **3.6.7** (`sys.version_info[:3] == (3, 6, 7)`) |
| `sys.prefix` | `<repo>/venv` |
| Preflight | **READY**, exit 0 |

Frozen pins verified: numpy 1.15.4, pandas 0.23.4, scipy 1.1.0,
scikit-learn 0.20.0, matplotlib 3.0.2, seaborn 0.9.0 — all MATCH. System
Python 3.11.2 not used. No dependency added, removed, upgraded or
downgraded.

`src/models/baseline.py` imports `sklearn.metrics` only. **No estimator
class is imported anywhere** — no `LinearRegression`, `DecisionTreeRegressor`,
`RandomForestRegressor` or `DummyRegressor`.

---

## 2. Frozen-contract check

`RESEARCH_QUESTION.md` §3 defines M0 as *"a constant predictor (mean **or**
median of the training target)"*. The median is therefore a **resolution
within** the already-declared option set, not a redefinition, and the
resolution follows from the pre-registered MAE decision metric. **No
conflict with any frozen document was found**, and nothing was overwritten.

---

## 3. Upstream integrity

Verified before the phase and again afterwards.

| Input | Result |
|---|---|
| Raw dataset SHA-256 `4127f868…6d27f1c` | MATCH |
| 4 Phase-4 partitions vs `split_manifest.json` | all MATCH |
| 8 Phase-5 outputs vs `feature_schema.json` | all MATCH |
| `y_train.csv` `fbcea1e4…fa7b53fa` | MATCH |
| `y_validation.csv` `9dc04233…91c1b0be` | MATCH |
| `y_train` rows = 24,418 | MATCH |
| `y_validation` rows = 8,678 | MATCH |
| `y_test.csv` absent | confirmed |

No earlier phase was regenerated.

### Target integrity

| Partition | Count | NaN | Inf | Min | Max | Zeros retained |
|---|---:|---:|---:|---:|---:|---:|
| development_train | 24,418 | 0 | 0 | 0 | 994 | 2 |
| validation | 8,678 | 0 | 0 | 2 | 886 | 0 |

Values are used exactly as written by Phase 5 — no clipping, transformation,
imputation or filtering. High-concentration observations retained.

---

## 4. M0 strategy

**Training-median constant baseline**, frozen before its validation result
was observed.

```
c = median(y_train)                fitted on 2010-2012 only
prediction(row) = c                for every 2013 validation row
```

| Property | Value |
|---|---|
| Fit partition | development_train, 2010-01-01 00:00 → 2012-12-31 23:00 |
| Evaluation partition | validation, 2013-01-01 00:00 → 2013-12-31 23:00 |
| Training rows | **24,418** |
| Validation rows | **8,678** |
| **Baseline constant** | **73.0** µg/m³ |
| Hyperparameters | none |
| Predictors used | **none** — `X_train` / `X_validation` / `X_test` not read |

**Exactly one baseline strategy was implemented.** No mean baseline, no
M0a/M0b, no mean-versus-median comparison. The training mean (97.8234) was
computed only as descriptive context for the write-up, never as a competing
predictor.

---

## 5. Metric implementations

| Metric | Definition | Implementation |
|---|---|---|
| MAE | `mean(abs(y_true - y_pred))` | `sklearn.metrics.mean_absolute_error` |
| RMSE | `sqrt(mean((y_true - y_pred)^2))` | `sqrt(sklearn.metrics.mean_squared_error(...))` |
| R² | `1 - SS_res / SS_tot` | `sklearn.metrics.r2_score` |

scikit-learn 0.20's `mean_squared_error` has no `squared=` keyword (added
in 0.22), so RMSE is an explicit `numpy.sqrt` of MSE. No modern API was
introduced. No metric beyond these three was computed.

---

## 6. Validation results

**2013 DEVELOPMENT VALIDATION — not final held-out test performance.**

| Metric | Value |
|---|---:|
| **MAE** | **67.591150034570177** |
| **RMSE** | **102.181752894378718** |
| **R²** | **−0.085725839256606** |

Context recorded for interpretation, not as competing models: the 2013
target's own median is 71.5 (oracle MAE 67.581009) and its mean is
101.7124 (the R² = 0 reference, RMSE 98.064826). M0's MAE is within
0.010 µg/m³ of the best constant achievable on 2013, and its negative R²
reflects that the median — not the mean — minimises the declared decision
metric.

---

## 7. Artifacts

| Path | SHA-256 | Rows |
|---|---|---:|
| `results/models/m0/m0_validation_metrics.csv` | `34a64d8ab662edcb0f3f1054e20991cf8cd6c7e87c0e947ad6c3c2c5754e56eb` | 1 |
| `results/models/m0/m0_validation_predictions.csv` | `2bd6c78ae64a28970558c5c67e80c4f2f68e1e08ed32bb0ecc852693ccc3fba6` | 8,678 |
| `artifacts/m0_baseline_manifest.json` | `bd76763ca030da2cc4cb7ee4861dc79bc9668e6c4c256ffe48a6dcc1cd1c5dd9` | — |

Prediction columns: `matrix_row_number`, `No`, `timestamp`, `actual_pm25`,
`predicted_pm25`. No error-analysis columns (`season_error`,
`pollution_band`, `severe_event`, `hour_group`, `residual_class`) — those
belong to Phase 11.

---

## 8. Prediction invariants

| Check | Result |
|---|---|
| Prediction rows == 8,678 | **yes** |
| Unique predicted values == 1 | **yes** — `[73.0]` |
| Unique value == training median | **yes** |
| Actuals match `y_validation` exactly | **yes** |
| Actuals match the Phase-4 validation source exactly | **yes** |
| No missing predictions / actuals | **0 / 0** |
| No infinite values | **yes** |
| Rows not reordered (`No` strictly ascending) | **yes** |

## 9. Row alignment

| Check | Result |
|---|---|
| `y_validation` rows | 8,678 |
| Validation rows in `feature_row_manifest.csv` | 8,678 |
| Prediction rows | 8,678 |
| `matrix_row_number` sequential 0…8,677 | **yes** |
| `No` matches the Phase-4 validation partition, in order | **yes** |
| `No` matches the row manifest, in order | **yes** |

---

## 10. Validation commands executed

| # | Check | Result |
|---|---|---|
| 1 | Runtime gate — version + six pins | 3.6.7, all MATCH |
| 2 | `venv/bin/python scripts/preflight.py` | **READY**, exit 0 |
| 3 | Frozen-contract M0 definition check | no conflict |
| 4 | Upstream gate — raw + 4 Phase-4 + 8 Phase-5 digests | all MATCH |
| 5 | M0 run 1 | exit 0 |
| 6 | M0 run 2 | exit 0 |
| 7 | Determinism — SHA-256 of all 3 artifacts | **byte-identical** |
| 8 | Independent metric recomputation | agrees within 1e-12 |
| 9 | Prediction invariants | all PASS |
| 10 | Row alignment | all PASS |
| 11 | Test-quarantine token audit | **0 executable references** |
| 12 | `py_compile` on all project `.py` | all OK |
| 13 | Final upstream gate | all MATCH |
| 14 | Git status | nothing staged |

### Deterministic rerun

M0 was run twice; all three artifacts are **byte-identical**. No random
operation is used anywhere and no execution timestamp is written into any
artifact.

### Independent metric recomputation

Recomputed from `m0_validation_predictions.csv` **without scikit-learn**,
directly from the metric definitions:

| Metric | Pipeline | Recomputed | Abs. difference |
|---|---|---|---:|
| MAE | 67.591150034570177 | 67.591150034570177 | 0.000e+00 |
| RMSE | 102.181752894378718 | 102.181752894378718 | 0.000e+00 |
| R² | −0.085725839256606 | −0.085725839256606 | 1.388e-17 |

All within the 1e-12 tolerance. The manifest values agree with both.

### Python 3.6 compatibility

All project `.py` files compile under CPython 3.6.7, including the two
added here. No post-3.6 syntax, no post-cutoff API, no new dependency.

---

## 11. Test-quarantine verification

| Check | Result |
|---|---|
| Executable references to `X_test` / `y_test` / test partition / 2014 in the new source | **NONE** (token-level audit) |
| `y_test.csv` in `data/processed/features/` | absent |
| Files matching `*y_test*`, `*test_prediction*`, `m0_test*` | **0** |
| `results/models/m0/` contents | metrics + predictions only |
| Manifest `test_target_read` | `false` |
| Manifest `test_features_read` | `false` |
| Manifest `test_predictions_generated` | `false` |
| Manifest `test_evaluation_status` | `not_evaluated` |
| Manifest `final_test_status` | `not_evaluated` |
| Placeholder 2014 number anywhere | none |

The audit tokenised both new files and classified every occurrence of
`test`, `2014`, `X_test` and `y_test`: **all are inside comments,
docstrings or explanatory strings; none is executable.** The one path
string containing "2014" is the raw dataset's own filename,
`PRSA_data_2010.1.1-2014.12.31.csv`, which spans 2010–2014 by name and is
the Phase-0 provenance file, not the test partition.

**2014 was not evaluated.** The final M0 score does not exist until
Protocol Phase 10.

---

## 12. Final-refit policy

Documented in [`M0_BASELINE.md`](M0_BASELINE.md) and recorded in the
manifest as `executed: false`. No combined 2010–2013 target was created.
The constant reported here is fitted on 2010–2012 only and is **not**
automatically the final M0 constant.

---

## 13. Files created and modified

**Created**

| Path | Purpose |
|---|---|
| `src/models/baseline.py` | M0 implementation |
| `scripts/run_m0_baseline.py` | thin entry point |
| `results/models/m0/m0_validation_metrics.csv` | one-row metrics table |
| `results/models/m0/m0_validation_predictions.csv` | 8,678-row prediction audit |
| `artifacts/m0_baseline_manifest.json` | M0 manifest |
| `docs/M0_BASELINE.md` | decision and interpretation document |
| `docs/PHASE_04_M0_BASELINE_RECORD.md` | this record |

**Modified**

| Path | Change |
|---|---|
| `README.md` | Phase 6 status and 2013-validation figures, labelled as such |

**Not modified by this phase:** `V1_RESEARCH_PROTOCOL.md` (it still shows
as modified in `git status`, but that is the uncommitted administrative
status edit from Phase 4 — this phase did not touch it and required no
protocol change), `FEATURE_POLICY.md`, `FEATURE_PREPARATION.md`,
`CLEANING_AND_SPLIT_POLICY.md`, `RESEARCH_QUESTION.md`, `EDA_ANALYSIS.md`,
`requirements-v1-2019.txt`, `requirements-v1-2019-lock.txt`, `data/raw/`,
the Phase-4 partitions and the Phase-5 feature matrices.

---

## 14. Warnings

**W1 — Negative validation R² is expected, not a fault.** R² is measured
against variance about the evaluation period's own mean, and M0 is
optimised for MAE via the median. A median-based constant necessarily
explains less squared variance than a mean-based one. Explained in
[`M0_BASELINE.md`](M0_BASELINE.md).

**W2 — These are single-year development-validation figures.** They are not
final performance and support no cross-model comparison, because no other
model exists.

**No unresolved blockers.**

---

## 15. Explicit confirmations

- **M0 is exactly one constant training-median predictor.**
- **Its constant was calculated from 2010–2012 training only** — the
  validation target was not consulted during fitting.
- **Validation metrics use 2013 only.**
- **The M0 strategy was fixed before its validation result was observed**,
  and was not revised afterwards.
- **`X_train` and `X_validation` were not used for prediction.**
- **`X_test` was not used.**
- **The 2014 target was never read.**
- **No 2014 prediction was generated.**
- **No 2014 metric was calculated.**
- **No `y_test` artifact exists.**
- **No random split, shuffle or `train_test_split` was used.**
- **No M1 Linear Regression was trained.**
- **No Decision Tree was trained.**
- **No Random Forest was trained.**
- **No feature was modified**; the Phase-5 matrices are byte-identical.
- **No target was transformed, clipped or imputed.**
- **No upstream artifact changed.**
- **No dependency changed.**
- **No git write operation was performed** — no `add`, `commit`, `push`,
  `reset`, `rebase`, `merge`, `cherry-pick`, `tag`, or branch
  creation/deletion. Every git command was read-only inspection.
- **Git history was not rewritten.**

---

## 16. Status

Protocol Phase 6 is **complete**. All legitimate changes are left
**unstaged**.

The next phase is Protocol Phase 7 — M1 Linear Regression. Nothing about it
has been implemented.

Awaiting instruction.
