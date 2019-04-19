# Phase 6 M2 Decision Tree Record — AirSense V1

**Research-protocol mapping: Protocol Phase 8 — Decision Tree Regression (M2).**

> Record-sequence numbering (`PHASE_06_`) is independent of protocol phase
> numbering. Earlier records were not renamed. Mapping:
> `PHASE_00_FOUNDATION_RECORD.md` (protocol 1–2),
> `PHASE_00A_RUNTIME_RECORD.md` (runtime recovery),
> `PHASE_01_EDA_RECORD.md` (protocol 3),
> `PHASE_02_CLEANING_RECORD.md` (protocol 4),
> `PHASE_03_FEATURE_RECORD.md` (protocol 5),
> `PHASE_04_M0_BASELINE_RECORD.md` (protocol 6),
> `PHASE_05_M1_LINEAR_REGRESSION_RECORD.md` (protocol 7),
> this record (protocol 8).

- **Recorded:** 2026-09-05 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE**
- **Decision document:** [`M2_DECISION_TREE.md`](M2_DECISION_TREE.md)
- **Manifest:** [`artifacts/m2_decision_tree_manifest.json`](../artifacts/m2_decision_tree_manifest.json)

> **2014 was not evaluated and did not influence selection.**

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

Verified before the phase and again afterwards.

| Input | Result |
|---|---|
| Raw dataset SHA-256 `4127f868…6d27f1c` | MATCH |
| 4 Phase-4 partitions vs `split_manifest.json` | all MATCH |
| 8 Phase-5 outputs vs `feature_schema.json` | all MATCH |
| 2 M0 artifacts vs `m0_baseline_manifest.json` | all MATCH |
| 6 M1 artifacts vs `m1_linear_regression_manifest.json` | all MATCH |
| `X_train` (24,418 × 43) · `X_validation` (8,678 × 43) | MATCH |
| `y_train` 24,418 · `y_validation` 8,678 | MATCH |
| Column order identical and equal to frozen schema | MATCH |
| `y_test.csv` absent | confirmed |

No earlier phase was regenerated. **M0 and M1 were not altered.**

---

## 3. Model specification and search space

**Estimator:** `sklearn.tree.DecisionTreeRegressor`, scikit-learn 0.20.0.

**Searched (5 × 4 = 20 candidates):**

| Parameter | Values |
|---|---|
| `max_depth` | 3, 5, 8, 12, None |
| `min_samples_leaf` | 1, 10, 50, 100 |

**Fixed, not searched:** `criterion="mse"` (the 0.20 name;
`"squared_error"` did not exist until 1.0), `splitter="best"`,
`min_samples_split=2`, `max_features=None`, `random_state=42`. Also not
searched: `max_leaf_nodes`, `min_impurity_decrease`.

**Candidate ordering:** `max_depth` outer, `min_samples_leaf` inner, both in
declared order. **Method:** explicit deterministic loop, **not
`GridSearchCV`** — the design is a single declared validation year, not
cross-validation folds. **The grid was frozen in code and documentation
before any tree was fitted**, and no candidate was added or removed
afterwards.

## 4. Selection and tie-break policy

**Primary: lowest 2013 validation MAE**, the Phase-0 decision metric.
Tie-break, frozen before fitting and applied only within 1 × 10⁻¹²:

1. lower validation RMSE;
2. smaller configured `max_depth` (`None` > every finite depth);
3. larger `min_samples_leaf`;
4. deterministic grid order.

R² never overrides the rule. **Training metrics played no part in
selection.** **Exactly one candidate tied on MAE — no tie-break was
required.**

---

## 5. Selected configuration

| Property | Value |
|---|---|
| **Candidate ID** | **14** |
| `criterion` | `"mse"` |
| `splitter` | `"best"` |
| **`max_depth`** | **12** |
| `min_samples_split` | 2 |
| **`min_samples_leaf`** | **10** |
| `max_features` | `None` |
| `random_state` | 42 |

### Tree structure

| Property | Value |
|---|---:|
| Observed tree depth | **12** |
| Node count | **1,597** |
| Leaf count | **799** |
| Root split feature | **`DEWP`** |
| Non-zero importances | **31 of 43** |
| Importance sum | 1.0000000000000002 |

No full tree diagram was drawn — 1,597 nodes would be unreadable.

---

## 6. Metrics

**Training (2010–2012) — diagnostic only, not used for selection:**

| Metric | Value |
|---|---:|
| MAE | 37.559165 |
| RMSE | 56.169988 |
| R² | 0.603889 |

**2013 DEVELOPMENT VALIDATION — not final held-out test performance:**

| Metric | Value |
|---|---:|
| **MAE** | **50.827853346163046** |
| **RMSE** | **77.984114992377783** |
| **R²** | **0.367608866301883** |

Train → validation MAE degradation: +13.268689 (×1.35). Overfitting is
reduced but not eliminated at the selected setting.

## 7. Comparison with M0 and M1

Prior values read from their manifests, not hard-coded.

| Metric | M0 | M1 | **M2** | vs M0 | vs M1 |
|---|---:|---:|---:|---:|---:|
| **MAE** *(primary)* | 67.591150 | 53.333361 | **50.827853** | **−16.763297 (−24.8010%)** | **−2.505507 (−4.6978%)** |
| RMSE | 102.181753 | 76.558423 | 77.984115 | −24.197638 (−23.6813%) | **+1.425692 (+1.8623% worse)** |
| R² | −0.085726 | 0.390520 | 0.367609 | +0.453335 | **−0.022911 (worse)** |

**M2 wins on the pre-registered decision metric (MAE) and loses to M1 on
RMSE and R².** By the declared criterion M2 is the better model; the
disagreement is reported rather than resolved by switching metrics. M2
improves the typical hour while making worse extreme errors — visible in its
wider residual range.

---

## 8. Residual diagnostics and prediction range

| Statistic | Value |
|---|---:|
| Mean residual | +2.777733 |
| Median residual | −6.133929 |
| Residual std. dev. | 77.939120 |
| Minimum residual | −329.723077 |
| Maximum residual | +698.802198 |
| Pearson corr(predicted, residual) | −0.158984 |
| **Negative predictions** | **0 (0.0000%)** |
| Prediction range | 8.454545 … 377.900000 |
| Training target range | 0.0 … 994.0 |
| **Predictions within training range** | **yes** |
| Predictions clipped | **no** |

Zero negative predictions, against M1's 658 — structural, since a regression
tree outputs leaf means of observed training targets and cannot leave their
convex range. The −0.159 predicted-vs-residual correlation indicates
regression toward leaf means; recorded as an observation for Phase 11.

---

## 9. Artifacts

| Path | SHA-256 | Rows |
|---|---|---:|
| `results/models/m2/m2_grid_results.csv` | `c98464463b11be52177f70e84f0f5bd8ec85387da5f856ce6794a949ee4e7aaa` | 20 |
| `results/models/m2/m2_validation_metrics.csv` | `9293fe849e58397a9b7443c6e7094e030d1087514e67668b923996870f9e45c0` | 1 |
| `results/models/m2/m2_validation_predictions.csv` | `49074ef654278858de71b95a99bb5a0290d69ebfdd3f4bab821888a94fba90e4` | 8,678 |
| `results/models/m2/m2_feature_importances.csv` | `a4c94b707500eec596e528e643d5c21b3f02ef13194b08b230775343c6f0edf5` | 43 |
| `artifacts/m2_decision_tree_manifest.json` | `d05632ee3db6c1f746aaf4f41f08812cf2bf2c8feb2bf4d2be7f0b70ef379af0` | — |
| `figures/m2_validation_actual_vs_predicted.png` | `866ea8932e001aad697aeb11fbd0d4708c3d284bd2afa68c44c4fbba53bb7fc4` | — |
| `figures/m2_validation_residuals_vs_predicted.png` | `b678748222362d9598cd6e52ae9c40079be96ea0c51d7d12e0b8b1faf510b3f9` | — |
| `figures/m2_grid_mae_by_complexity.png` | `36de8461e449c43f8e485f624cf4bae4a07a7e1257b064d6aa27b93ca4b6d7f0` | — |

**No serialized model** — Phase 10 refits on 2010–2013.

---

## 10. Verification

### Independent metric recomputation

From `m2_validation_predictions.csv`, **without scikit-learn**:

| Metric | Pipeline | Recomputed | Abs. difference |
|---|---|---|---:|
| MAE | 50.827853346163046 | 50.827853346163053 | 7.105e-15 |
| RMSE | 77.984114992377783 | 77.984114992377769 | 1.421e-14 |
| R² | 0.367608866301883 | 0.367608866301883 | 2.776e-16 |

All within 1e-12; the manifest agrees.

### Independent selection recomputation

The frozen selection rule was **re-implemented from scratch** against
`m2_grid_results.csv` alone: best validation MAE 50.827853346163046,
**1 candidate tied within 1e-12**, independently selected
**candidate 14** — **matching the manifest**.

### Grid invariants

20 rows · 20 unique hyperparameter combinations · exactly one `selected=True`
· `max_depth` values {3, 5, 8, 12, None} · `min_samples_leaf` values
{1, 10, 50, 100} · `criterion` only `mse` · `splitter` only `best` ·
`random_state` only 42 · no undeclared configuration · no candidate missing
· all train/validation metrics and structural diagnostics finite.

### Prediction and row alignment

8,678 rows · actuals equal frozen `y_validation` **and** the Phase-4 source ·
predictions and residuals finite · `matrix_row_number` sequential 0…8,677 ·
`No` aligned to both the row manifest and the Phase-4 partition, strictly
ascending · **628 distinct predictions** (≥ 2) · predictions within the
training target range · no clipping.

### Deterministic rerun

M2 was run twice; **all 8 artifacts byte-identical**, including all three
PNGs. `random_state=42` on every candidate, figures pin their PNG `Software`
metadata, and no execution timestamp is written anywhere.

### Python 3.6 compatibility

All project `.py` files compile under CPython 3.6.7. No post-3.6 syntax, no
post-cutoff API, no new dependency.

---

## 11. Test-quarantine verification

| Check | Result |
|---|---|
| Executable references to `X_test`/`y_test`/test partition/`m2_test`/2014 | **NONE** (token-level audit of both new files) |
| `y_test.csv` present | no |
| Files matching `*m2_test*`, `*test_prediction*`, `*y_test*` | **0** |
| `results/models/m2/` contents | 4 development artifacts only |
| Manifest `test_features_read` / `test_target_read` | `false` / `false` |
| Manifest `test_predictions_generated` | `false` |
| Manifest `selection_used_test_information` | **`false`** |
| Manifest `test_evaluation_status` / `final_test_status` | `not_evaluated` |
| Placeholder 2014 number | none |

Every textual occurrence of "2014" in the new source is a comment, docstring
or string; the one path string containing it is the raw dataset's own
filename. **No further hyperparameter search was performed after
selection.**

---

## 12. Files created and modified

**Created:** `src/models/decision_tree.py`,
`scripts/run_m2_decision_tree.py`, the four `results/models/m2/` artifacts,
`artifacts/m2_decision_tree_manifest.json`, the three `figures/m2_*.png`,
`docs/M2_DECISION_TREE.md`, `docs/PHASE_06_M2_DECISION_TREE_RECORD.md`.

**Modified:** `README.md` only.

**Not modified by this phase:** `V1_RESEARCH_PROTOCOL.md` (still shows as
modified in `git status` from the uncommitted Phase-4 administrative status
edit; this phase required and made no protocol change), `FEATURE_POLICY.md`,
`FEATURE_PREPARATION.md`, `CLEANING_AND_SPLIT_POLICY.md`, `M0_BASELINE.md`,
`M1_LINEAR_REGRESSION.md`, `RESEARCH_QUESTION.md`, `EDA_ANALYSIS.md`, both
requirements files, `data/raw/`, the Phase-4 partitions, the Phase-5
matrices, and all M0 and M1 artifacts.

---

## 13. Warnings

**W1 — Metrics disagree between M1 and M2.** M2 wins on MAE, loses on RMSE
and R². Reported as such; the pre-registered decision metric governs.

**W2 — Residual overfitting at the selected setting.** Training MAE is 26%
below validation MAE (×1.35 degradation).

**W3 — 2013 was used to select M2's hyperparameters**, so its 2013 figure is
mildly optimistic relative to M0 and M1, neither of which selected anything.
This must be restated when Phase 10 results are reported.

**W4 — Feature importances are not mechanism.** Impurity-based importance
favours high-cardinality predictors and splits credit arbitrarily among the
correlated `DEWP`/`TEMP`/`PRES` group. The root split on `DEWP` is a
property of this partition, not a causal finding.

**No unresolved blockers.**

---

## 14. Explicit confirmations

- **M2 is `sklearn.tree.DecisionTreeRegressor`.**
- **`criterion="mse"`** (the scikit-learn 0.20 name).
- **`splitter="best"`.**
- **Exactly 20 candidates were evaluated.**
- **The grid was frozen before any M2 result was observed.**
- **The selected candidate was chosen by 2013 validation MAE.**
- **RMSE was only the first tie-break** — and was not needed, since one
  candidate tied.
- **No test information influenced selection.**
- **M2 uses exactly the same 43 features as M1.**
- **No scaling was introduced.**
- **No target transformation occurred.**
- **No feature was added or removed.**
- **No post-hoc hyperparameter was tried** after selection.
- **`random_state=42` for all candidates**; only one seed was used.
- **M2 was fitted only on 2010–2012.**
- **Validation used only 2013.**
- **`X_test` was not read.**
- **The 2014 target was never read.**
- **No 2014 prediction was generated.**
- **No 2014 metric was calculated.**
- **No `y_test` artifact exists.**
- **No M3 Random Forest was trained.**
- **No upstream artifact changed.**
- **No dependency changed.**
- **No git write operation was performed.**
- **Git history was not rewritten.**

---

## 15. Status

Protocol Phase 8 is **complete**. All legitimate changes are left
**unstaged**.

The next phase is Protocol Phase 9 — M3 Random Forest Regressor. Nothing
about it has been implemented.

Awaiting instruction.
