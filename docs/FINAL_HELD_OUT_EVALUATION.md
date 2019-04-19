# AirSense V1 Final Held-Out Evaluation

**Research-protocol mapping: Protocol Phase 10 — Held-out Evaluation.**

> **2014 is the first and only planned final held-out model evaluation in
> this project.** Every result below was produced against a target that was
> sealed until all four prediction vectors had already been generated and
> frozen.

---

## Purpose

Score the four pre-registered models — M0, M1, M2, M3 — once, on data no
model or decision has ever seen, and let that result stand.

Everything that could influence the models was frozen beforehand: the
feature schema, the split, each model's specification, the selected
hyperparameters, and the primary metric. Phase 10 adds no development. It
reveals a number.

---

## Pre-test freeze

`artifacts/pretest_development_freeze.json`, SHA-256
`98568e88cccb6a64a6e52dc03a76b0e3b3e5b7d750ece3d81e3790fb0beb0bf9`,
matched its expected value exactly.

**51 hashes were independently verified before anything else ran**: the 16
referenced by the freeze, 22 referenced by the four development manifests,
12 Phase-4 and Phase-5 artifacts, and the raw dataset. All matched. The
freeze reported `final_test_status: sealed`.

A single differing hash would have stopped the phase. None did.

---

## Blind final refit

Phase 10 was executed in two mandatory stages.

**Stage A** (`scripts/prepare_final_predictions.py`) refitted all four
models on the full permitted development period and generated 2014
predictions **from predictors only**. It never loaded
`data/processed/airsense_test_2014.csv` — every CSV read passes through a
guard that aborts the run if that path is ever reached, and a token-level
audit confirmed zero executable references to the test partition, `y_test`
or `actual_pm25`.

**Stage B** (`scripts/open_final_test.py`) is the only Phase-10 source path
permitted to read the 2014 target. It imports no estimator and contains no
`.fit(` or `.predict(` call — **it is structurally incapable of training a
model.**

## Final development population

**2010-01-01 00:00 → 2013-12-31 23:00**, formed by concatenating the frozen
Phase-5 matrices in chronological order:

| Component | Rows |
|---|---:|
| `X_train` / `y_train` (2010–2012) | 24,418 |
| `X_validation` / `y_validation` (2013) | 8,678 |
| **Final development population** | **33,096 × 43** |

**No feature engineering was re-run.** No `get_dummies`, no changed
reference category, no 2013-derived level, no scaling. The representation
was frozen before the test existed, so the matrices were concatenated
unchanged. No shuffle, no further split. 2013's role as validation is over;
it is development data now.

Test predictors: `X_test.csv`, **8,661 × 43**, hash-verified against
`feature_schema.json`. That file is predictors-only by construction — Phase 5
wrote it with an explicit `usecols` list omitting `pm2.5`.

---

## Frozen M0 specification

Training-median constant predictor, **recomputed** on the combined
2010–2013 target as the pre-registered refit rule requires.

**Final constant: 73.0 µg/m³.**

This coincidentally equals the Phase-6 value fitted on 2010–2012 alone. It
was recomputed, not reused — the median of all 33,096 development targets
is also 73.0.

## Frozen M1 specification

`LinearRegression(fit_intercept=True)`, no scaling, no regularization, the
frozen 43 features.

Final intercept **2435.4089340577**. Design rank **44/44** (not deficient),
condition number **213,048** — ill-conditioned, as in development, and
reported rather than corrected.

## Frozen M2 specification

`DecisionTreeRegressor(criterion="mse", splitter="best", max_depth=12,
min_samples_split=2, min_samples_leaf=10, max_features=None,
random_state=42)`.

Exactly the Phase-8 selection. **No retuning.** Final tree: depth 12, 1,787
nodes, 894 leaves.

## Frozen M3 specification

`RandomForestRegressor(n_estimators=100, criterion="mse", bootstrap=True,
max_depth=12, min_samples_split=2, min_samples_leaf=1, max_features="auto",
random_state=42, n_jobs=1, oob_score=False)`.

Exactly the Phase-9 selection. **No retuning, no extra seed, no additional
trees.** Final forest: 100 trees, mean depth 12.00, mean 1,784.5 leaves,
mean 3,567.9 nodes.

Each specification was read out of the pre-test freeze rather than restated
in code, and asserted parameter-by-parameter, so a divergence between the
frozen record and the refit would have stopped the run.

---

## Blind-prediction freeze

Stage A was run **twice before the target was touched**. All six
deterministic artifacts were byte-identical across runs, including the
100-tree forest's predictions.

Independent pre-opening checks confirmed: M0's predictions are a single
value equal to the development median; M1's predictions reconstruct from
`intercept + X_test · coefficients` to within 9.663 × 10⁻¹³; M2's and M3's
predictions are finite and inside the development target range; all four
vectors have 8,661 rows aligned by identifier and timestamp; and the blind
artifact contains **no `actual_pm25`, no `pm2.5`, no residual column**.

`artifacts/final_blind_prediction_freeze.json`, SHA-256
`4f59c710529e99b8c8a33517a34c503837d471871eb3f12ef987b34e41e7f1eb`, then
sealed the state with `2014_target_status: sealed`,
`blind_prediction_status: frozen`, `final_models_status: frozen`. Its seven
referenced hashes were independently verified.

**The target was still sealed at the moment the predictions were frozen.**

---

## Test opening procedure

With every gate passed and no opening receipt present, Stage B performed
the single authorized read of `data/processed/airsense_test_2014.csv`,
loading only `No`, `year`, `month`, `day`, `hour` and `pm2.5`. Feature
values were not read — they are irrelevant to scoring and there is no route
by which that file could reach a model.

Verified before any metric was computed: **8,661 rows**, no missing target,
no non-finite value, and **all 8,661 identifiers and timestamps matching the
frozen predictions in order**.

`results/final_test/final_test_target_snapshot.csv` (SHA-256
`fc04c671599c3b09c9091518b6629d855b1d98de6e8ed462a3dc383372cf2314`) froze
the revealed target, and
`artifacts/final_test_opening_receipt.json` (SHA-256
`4b57468b9b2ec4f211b0403fbcdd1c4d64f6c7902c4700f686edbe53c89617a8`)
recorded the opening. All later verification used the snapshot; the Phase-4
source has not been reopened. A second Stage-B run confirmed this — with the
receipt present it entered verification/report-only mode and left the source
untouched.

For context, the revealed 2014 target: n = 8,661, mean 97.73, median 72.0,
std 93.53, range 2–671 µg/m³.

---

## 2014 held-out results

**All four models scored against the same 8,661 observations, fitted on
2010–2013.**

| Model | MAE | RMSE | R² | MAE rank |
|---|---:|---:|---:|---:|
| **M0** naive constant | 65.417388 | 96.741589 | −0.069943 | 4 |
| **M1** linear regression | 52.900194 | 73.103403 | 0.389045 | 3 |
| **M2** decision tree | 51.248814 | 75.522046 | 0.347949 | 2 |
| **M3** random forest | **48.535341** | **70.277633** | **0.435364** | **1** |

Every model is reported. None was dropped, and none was added.

## Primary MAE ranking

**M3 < M2 < M1 < M0.**

**Best final model by MAE: M3, the random forest, at 48.535341 µg/m³.**

Against the naive floor, M3 reduces mean absolute error by 16.88 µg/m³, a
**25.8% improvement over M0**. Against the linear model it improves by 4.36
µg/m³ (**8.2%**), and against the single tree by 2.71 µg/m³ (**5.3%**).

## Secondary RMSE and R²

**The rankings disagree, and the disagreement is reported rather than
resolved.**

By RMSE and R², the order is **M3 < M1 < M2 < M0** — M1 (73.103, 0.389)
beats M2 (75.522, 0.348) on both, while M2 beats M1 on MAE.

This is the same pattern Phase 8 found on validation, and it has the same
explanation: the tree improves the typical hour while making worse extreme
errors, and RMSE squares errors so it is dominated by the largest misses.
M2's residuals span −391 to +516 against M1's −208 to +520.

**The primary metric was not changed.** MAE was pre-registered in Phase 0
as the decision metric, and by MAE M2 outranks M1. A reader who prefers
squared-error criteria should read the RMSE column and reach the opposite
conclusion about that pair — both readings are on the table.

For M3 the question does not arise: **it is best on all three metrics
simultaneously**, so the headline result does not depend on which metric
you privilege.

---

## Development-validation versus final-test behaviour

| Model | Selected on 2013? | 2013 val MAE | 2014 test MAE | Δ (test − val) |
|---|---|---:|---:|---:|
| M0 | no | 67.591150 | 65.417388 | **−2.173762** |
| M1 | no | 53.333361 | 52.900194 | **−0.433167** |
| M2 | **yes** | 50.827853 | 51.248814 | **+0.420960** |
| M3 | **yes** | 47.938823 | 48.535341 | **+0.596518** |

**The pattern is exactly the one Phases 8 and 9 warned about, and it shows
up cleanly.**

M0 and M1 selected nothing on 2013 — their 2013 figures were honest
estimates, and both did *better* on 2014. M2 and M3 used 2013 to choose
hyperparameters — their 2013 figures were optimistic, and both did *worse*
on 2014.

The optimism is modest: **+0.42 µg/m³ for M2 and +0.60 for M3**, roughly 1%
of their MAE. That is small enough that the development ranking survives
intact — M3 was best on 2013 and is best on 2014 — but it is real, it is in
the predicted direction, and it is why the locked test existed.

RMSE improved for every model on 2014 (−2.46 to −5.44), and R² was close to
flat (−0.020 to +0.016). The RMSE improvement is consistent with 2014 having
a less extreme upper tail than 2013 — maximum 671 µg/m³ against 886 — which
affects squared-error metrics more than absolute ones. That is a property of
the years, not of the models, and it is a reminder that a single evaluation
year carries the character of that year.

*Table:* `results/final_test/development_to_test_comparison.csv`

---

## Final model ranking

1. **M3 — Random Forest**, MAE 48.535341. Best on MAE, RMSE and R².
2. **M2 — Decision Tree**, MAE 51.248814. Second on MAE, third on RMSE/R².
3. **M1 — Linear Regression**, MAE 52.900194. Third on MAE, second on
   RMSE/R².
4. **M0 — Naive constant**, MAE 65.417388. Last on every metric.

The pre-registered expectation — recorded in `RESEARCH_QUESTION.md` before
any data was modelled — was that the random forest would be the strongest of
the four. **It was.** The pre-registration was not adjusted to fit.

---

## Interpretation

**The predictors carry real, substantial information.** M3 beats the naive
floor by 25.8% of MAE and explains 43.5% of the variance in hourly PM2.5
that a constant explains none of. The answer to the V1 research question is
that classical 2019 methods *can* predict PM2.5 from meteorology and
calendar position, meaningfully better than a trivial baseline.

**Most of the variance remains unexplained.** R² of 0.435 means roughly 56%
of the variation in hourly PM2.5 is not captured. That is consistent with
what Phase 3 found: emissions, traffic and industrial activity are not in
this dataset, and no model built from weather and time alone can recover
them.

**Ensembling repaid its cost, modestly.** M3 improves on M2 by 5.3% of MAE
and repairs the RMSE and R² regression a single tree suffered against the
linear model. The price is interpretability: 100 trees with ~178,000 leaves
cannot be read the way M1's 43 coefficients can.

**M0 is more informative than its rank suggests.** Its mean residual is
+24.73 µg/m³ — the development median of 73.0 sits 24.73 below the 2014
mean of 97.73 — while its *median* residual is −1.0. A constant tuned for
absolute error lands near the middle of a right-skewed distribution and is
badly wrong about the top of it. That is precisely the behaviour a naive
floor should expose.

**M1 still produces impossible predictions.** 466 of 8,661 test predictions
(**5.38%**) are below zero µg/m³, minimum −208.20. Unconstrained least
squares on a non-negative target does this, and it was not corrected. M2 and
M3 produce **zero** negative predictions, structurally, because a tree
outputs averages of observed non-negative targets.

**No causal claim is made anywhere.** These are predictive associations
measured on one site over five years.

---

## Test-set exhaustion statement

**The 2014 target has now been opened. It is no longer available as an
untouched test set.**

Any model changed after this point must **not** claim a fresh evaluation on
the same 2014 split. Future exploratory variants may be compared on 2014
only with the explicit understanding that the test has become known, and any
such comparison must say so.

**The target cannot be resealed.** Pretending otherwise would be the exact
failure the whole protocol was built to prevent. A genuinely fresh
evaluation would require data outside 2010–2014.

`artifacts/final_evaluation_manifest.json` records
`post_test_model_changes: none`. No model was retrained, retuned, clipped or
dropped after the metrics became visible, and no fifth model was added.

---

## Limitations

- **One site, one city, five years.** Nothing here generalises to other
  locations, and 2014 is a single evaluation year whose particular character
  — a less extreme upper tail than 2013 — visibly moved the RMSE figures.
- **Meteorology is an incomplete system.** The unexplained 56% of variance
  is bounded by what the dataset contains, not by the model family.
- **M2 and M3 carry residual selection optimism**, quantified above at
  +0.42 and +0.60 µg/m³ of MAE. Their 2014 numbers are the honest ones;
  their 2013 numbers were not.
- **Severe episodes remain badly predicted** by every model — maximum
  residuals of +505 to +598 µg/m³. Aggregate metrics hide this.
- **The grids were spanning, not exhaustive.** Better configurations may
  exist outside the declared search spaces; they were deliberately not
  sought.
- **Interpretability declines with rank.** The best model is the least
  explicable one.
- **The test is now spent.**

---

## Next phase: error analysis

Protocol Phase 11 examines *where* the models fail: residual structure by
season, hour and pollution level, and behaviour during severe episodes.

**No such analysis was performed here.** Phase 10 computed only aggregate
MAE, RMSE, R² and an overall residual summary. No subgroup ranking by
season, month, hour, wind direction, pollution severity or extreme-event
threshold exists, and the optional random-split educational comparison was
not run — it remains explicitly secondary and outside this phase.
