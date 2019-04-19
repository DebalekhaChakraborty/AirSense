# AirSense V1 M3 Random Forest

**Research-protocol mapping: Protocol Phase 9 — Random Forest Regression (M3).**

> **All metrics below are 2013 development-validation metrics. They are not
> final held-out test results.** The 2014 partition remains locked, was not
> read, and did not influence hyperparameter selection.

---

## Purpose

Fit the last of the four pre-registered models, and test whether averaging
many decorrelated trees repairs the specific weakness M2 exposed.

M2 improved on M1's MAE but **regressed on RMSE and R²** — better on the
typical hour, worse on the extremes. That is the classic signature of a
single high-variance tree. Bagging is the standard remedy, so M3 is not a
generic "try something stronger" step: it targets a defect the previous
phase measured.

---

## Protocol position

M3 was pre-registered in [`RESEARCH_QUESTION.md`](RESEARCH_QUESTION.md) §3
as *"Random Forest Regressor — ensemble of trees; expected to be the
strongest of the four, and the test of whether ensembling repays its
interpretability cost."*

It is the **final development-model phase**. After it, the pre-test
development freeze seals every model specification, and Protocol Phase 10
opens 2014 exactly once.

---

## Frozen 43-feature representation

The **exact same matrix M1 and M2 used**, consumed as produced and verified
against [`artifacts/feature_schema.json`](../artifacts/feature_schema.json)
before fitting: 6 raw meteorological + 11 month + 23 hour + 3 wind-direction
one-hot columns.

**No model-specific feature engineering was permitted.** No scaling, no
transformation, no feature added, removed or reordered. The M0–M3 comparison
is about model class, not different feature sets.

## Training period

**2010-01-01 00:00 → 2012-12-31 23:00** — 24,418 observations.

## Validation period

**2013-01-01 00:00 → 2013-12-31 23:00** — 8,678 observations. Used for both
evaluation and M3 hyperparameter selection.

---

## Why Random Forest

Three properties matter here, and each addresses something measured earlier:

1. **Variance reduction by averaging.** A single tree's partition is
   unstable — small changes to the training data move the split points. M2's
   worse extreme errors are a symptom. Averaging 100 trees fitted to
   different bootstrap samples cancels much of that variance.
2. **Non-linearity and interactions for free.** Like M2, and unlike M1, a
   forest can represent `TEMP`'s non-monotone relationship and any
   interaction, without manual feature construction.
3. **Predictions stay in range.** Each tree outputs a leaf mean of observed
   training targets and the forest averages those, so — as with M2, and
   unlike M1 — the output cannot be negative.

The cost is interpretability: 100 trees cannot be read the way a single
tree or a coefficient vector can.

---

## Predeclared 24-candidate grid

Frozen in `src/models/random_forest.py` before any forest was fitted.

**Searched — 3 × 4 × 2 = 24 candidates:**

| Parameter | Values | Rationale |
|---|---|---|
| `max_depth` | **8** | moderately constrained trees |
| | **12** | complex trees comparable to the selected M2 structural regime |
| | **None** | unrestricted-depth reference |
| `min_samples_leaf` | **1** | fully local leaves permitted |
| | **10** | light smoothing |
| | **50** | moderate smoothing |
| | **100** | strong smoothing |
| `max_features` | **"auto"** | all 43 predictors available at each split, under scikit-learn 0.20 regression semantics |
| | **0.5** | roughly half the predictors per split, increasing random-subspace diversity between trees |

The grid spans bias, variance and ensemble diversity while staying small
enough to be transparent. **It is not claimed to exhaust Random Forest
design space.**

**Candidate order:** `max_depth` outer, `min_samples_leaf` middle,
`max_features` inner, `candidate_id` 1…24. That ordering is the final
tie-break.

**Method:** an explicit deterministic loop. `GridSearchCV`,
`RandomizedSearchCV`, `KFold`, `ShuffleSplit` and `cross_val_score` were all
deliberately avoided — the design already contains an explicit chronological
validation year, and random folds would break the temporal ordering the
protocol exists to protect.

### Fixed, not searched

`n_estimators=100`, `criterion="mse"`, `bootstrap=True`,
`min_samples_split=2`, `random_state=42`, `n_jobs=1`, `oob_score=False`.

`criterion="mse"` is the **scikit-learn 0.20 name**; `"squared_error"` did
not exist until 1.0 and would be a post-cutoff API.

### Why `n_estimators` was fixed at 100

100 trees is a conventional, computationally manageable ensemble size that
was entirely available in 2019, and gives substantially more averaging than
scikit-learn 0.20's small default.

Fixing it **before results** means selection concerns tree structure and
feature subsampling rather than arbitrary ensemble size. More trees in a
bagged ensemble reduce variance monotonically and essentially never
overfit — so tuning `n_estimators` on validation would mostly measure
compute budget, and would tempt an unbounded search. **50, 200, 500 and 1000
were not tried.**

### Randomness and reproducibility policy

`random_state=42` for **every** candidate, with `n_jobs=1` so the result
cannot depend on thread scheduling. The seed exists solely to make the
historical experiment reproducible. **Only one seed was used**, and no
best-of-seed result is reported.

`bootstrap=True` for every candidate — the classical Random Forest
formulation. `bootstrap=False` was not searched.

**Out-of-bag scoring is off and was not used for selection.** OOB samples
are drawn from across the whole training period, so an OOB criterion would
introduce a second selection framework that ignores temporal ordering. 2013
validation MAE remains the sole primary criterion.

---

## Selection rule

**Primary: lowest 2013 validation MAE** — the decision metric frozen in
Phase 0, long before M3 existed.

Tie-break policy, also frozen before fitting, applied only within 1 × 10⁻¹²:

1. lower validation RMSE;
2. smaller finite `max_depth`, with `None` considered more complex than
   every finite value;
3. larger `min_samples_leaf`;
4. `max_features=0.5` before `"auto"`, since fewer candidate predictors are
   exposed at each split;
5. deterministic grid order.

R² never overrides the rule. Training metrics were computed for every
candidate as an overfitting diagnostic and **played no part in selection**.

**Only one candidate tied on MAE, so no tie-break was needed.**

---

## Selected configuration

**Candidate 9** — the unique lowest-MAE candidate.

| Parameter | Value |
|---|---|
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

### A selection worth stating plainly

**Candidate 11 has a better RMSE (72.958450) and a better R² (0.446491)
than the selected candidate 9 (73.090585, 0.444484).** It was not selected,
because its MAE (47.962390) is higher than candidate 9's (47.938823), and
**MAE is the pre-registered decision metric**.

The difference is small — 0.024 µg/m³ of MAE — and a project willing to
choose its metric after the fact could easily have justified candidate 11.
That is exactly why the metric was fixed in Phase 0. The rule was applied as
written.

### `max_features` result

Averaged across the grid, `"auto"` outperformed `0.5` (mean validation MAE
50.026 against 50.959), and it won at every depth/leaf combination. With
only 43 predictors — 37 of them sparse one-hot columns — halving the
candidate set per split appears to cost more in tree quality than it gains
in inter-tree decorrelation. Recorded as an observation from this grid, not
a general claim about random subspace sizing.

---

## Full grid results

**2013 DEVELOPMENT VALIDATION.** All 24 candidates reported — none dropped.

| # | `max_depth` | `leaf` | `max_features` | Train MAE | **Val MAE** | Val RMSE | Val R² | Mean depth | Mean leaves |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1 | auto | 42.242591 | 51.217858 | 76.276592 | 0.394999 | 8.00 | 235.8 |
| 2 | 8 | 1 | 0.5 | 44.060324 | 52.416378 | 77.975382 | 0.367750 | 8.00 | 230.5 |
| 3 | 8 | 10 | auto | 42.815835 | 51.060622 | 75.992870 | 0.399491 | 8.00 | 176.1 |
| 4 | 8 | 10 | 0.5 | 44.601437 | 52.397459 | 77.930418 | 0.368479 | 8.00 | 165.4 |
| 5 | 8 | 50 | auto | 44.477918 | 51.500540 | 76.866194 | 0.385610 | 8.00 | 108.2 |
| 6 | 8 | 50 | 0.5 | 46.092152 | 52.825562 | 78.495166 | 0.359293 | 8.00 | 100.1 |
| 7 | 8 | 100 | auto | 45.911106 | 52.229571 | 77.763159 | 0.371187 | 8.00 | 75.6 |
| 8 | 8 | 100 | 0.5 | 47.313728 | 53.519807 | 79.217552 | 0.347446 | 8.00 | 70.7 |
| **9** | **12** | **1** | **auto** | **32.181852** | **47.938823** | **73.090585** | **0.444484** | **12.00** | **1662.4** |
| 10 | 12 | 1 | 0.5 | 34.411670 | 49.137569 | 74.158369 | 0.428135 | 12.00 | 1571.3 |
| 11 | 12 | 10 | auto | 36.141625 | 47.962390 | **72.958450** | **0.446491** | 12.00 | 604.4 |
| 12 | 12 | 10 | 0.5 | 38.207416 | 48.963194 | 74.060908 | 0.429637 | 12.00 | 538.6 |
| 13 | 12 | 50 | auto | 41.709151 | 49.010980 | 74.374721 | 0.424793 | 12.00 | 191.4 |
| 14 | 12 | 50 | 0.5 | 43.139831 | 50.678756 | 76.233790 | 0.395678 | 12.00 | 180.6 |
| 15 | 12 | 100 | auto | 44.268204 | 50.770480 | 76.506261 | 0.391350 | 12.00 | 107.8 |
| 16 | 12 | 100 | 0.5 | 45.632446 | 52.139172 | 77.857919 | 0.369654 | 11.99 | 102.9 |
| 17 | None | 1 | auto | **12.463122** | 49.984486 | 76.601747 | 0.389830 | 42.89 | 14691.9 |
| 18 | None | 1 | 0.5 | 12.619274 | 49.163884 | 74.913456 | 0.416430 | 43.99 | 14985.8 |
| 19 | None | 10 | auto | 31.553507 | 48.780885 | 74.447926 | 0.423660 | 26.54 | 1188.6 |
| 20 | None | 10 | 0.5 | 33.625510 | 48.226082 | 73.582412 | 0.436983 | 26.47 | 1107.2 |
| 21 | None | 50 | auto | 41.076545 | 49.095025 | 74.526187 | 0.422448 | 18.50 | 232.6 |
| 22 | None | 50 | 0.5 | 42.233325 | 50.146309 | 75.690242 | 0.404265 | 18.37 | 232.0 |
| 23 | None | 100 | auto | 44.052069 | 50.758127 | 76.532761 | 0.390929 | 15.38 | 117.0 |
| 24 | None | 100 | 0.5 | 45.300743 | 51.896234 | 77.690919 | 0.372355 | 15.14 | 111.9 |

**Bagging visibly tames the overfitting that destroyed the equivalent single
tree.** The unrestricted, `min_samples_leaf=1` forests (candidates 17 and
18) reach training MAE ≈ 12.5 and still achieve validation MAE ≈ 49.2–50.0.
The comparable single tree in M2 reached training MAE **0.056** and
validation MAE **64.41** — the worst of its grid. Averaging 100 bootstrap
trees converts a catastrophic overfit into a merely mediocre one. Every M3
candidate beats every M2 candidate on validation MAE.

*Figure:* `figures/m3_grid_mae_by_complexity.png`, faceted by
`max_features`.

---

## Training vs validation behaviour

| Metric | Training (2010–2012) | Validation (2013) | Degradation |
|---|---:|---:|---:|
| MAE | 32.181852 | **47.938823** | +15.76 (×1.49) |
| RMSE | 47.486814 | 73.090585 | +25.60 |
| R² | 0.716890 | 0.444484 | −0.272 |

Validation MAE is 49% above training MAE, a wider gap than M2's ×1.35. That
is expected: the selected forest uses `min_samples_leaf=1`, so individual
trees fit training data very closely and the ensemble's remaining
generalisation gap is carried by averaging rather than by pruning.

**The larger training-validation gap did not make M3 worse.** It has the
best validation figures of any model in the project. Training error remains
a poor guide to generalisation here, which is why selection used validation
MAE and training metrics were diagnostics only.

---

## 2013 validation results

**2013 DEVELOPMENT VALIDATION — not final test performance.**

| Metric | Value |
|---|---:|
| **MAE** | **47.938823** µg/m³ |
| **RMSE** | **73.090585** µg/m³ |
| **R²** | **0.444484** |

Independently recomputed from the prediction file without scikit-learn;
agreement within 1.4 × 10⁻¹⁴.

---

## Comparison with M0

| Metric | M0 (constant 73.0) | **M3** | Change | Relative |
|---|---:|---:|---:|---:|
| **MAE** *(primary)* | 67.591150 | **47.938823** | **−19.652327** | **−29.08%** |
| RMSE | 102.181753 | 73.090585 | −29.091168 | −28.47% |
| R² | −0.085726 | 0.444484 | +0.530210 | — |

## Comparison with M1

| Metric | M1 (linear) | **M3** | Change | Relative |
|---|---:|---:|---:|---:|
| **MAE** *(primary)* | 53.333361 | **47.938823** | **−5.394538** | **−10.11%** |
| RMSE | 76.558423 | 73.090585 | −3.467838 | −4.53% |
| R² | 0.390520 | 0.444484 | +0.053964 | — |

## Comparison with M2

| Metric | M2 (single tree) | **M3** | Change | Relative |
|---|---:|---:|---:|---:|
| **MAE** *(primary)* | 50.827853 | **47.938823** | **−2.889030** | **−5.68%** |
| RMSE | 77.984115 | 73.090585 | −4.893530 | −6.28% |
| R² | 0.367609 | 0.444484 | +0.076875 | — |

Sign convention, identical for all three: `improvement = prior − M3`, so
positive means M3 improved; `r2_delta = M3 − prior`.

**M3 is better than M0, M1 and M2 on all three metrics simultaneously.** It
is the only model in the project for which that is true, and it is the first
to be unambiguously best without needing the decision metric to adjudicate.

**M3 repairs the specific defect M2 introduced.** M2 had beaten M1 on MAE
while *losing* on RMSE (+1.43) and R² (−0.023). M3 beats M1 on RMSE by 3.47
and on R² by 0.054 — so averaging did exactly what it was expected to do,
recovering the extreme-error performance a single tree gave away while
keeping the tree family's advantage on typical hours.

---

## Forest complexity

Across all **100 estimators** of the selected forest:

| Statistic | Min | Mean | Median | Max |
|---|---:|---:|---:|---:|
| Tree depth | 12 | 12.00 | 12.0 | 12 |
| Leaf count | 1,525 | 1,662.43 | 1,662.0 | 1,827 |
| Node count | 3,049 | 3,323.86 | 3,323.0 | 3,653 |

Every tree hit the configured depth limit of 12, so depth was the binding
constraint throughout. Constraint verification confirmed all 100 trees
respect both `max_depth` and `min_samples_leaf`.

The ensemble contains roughly **166,000 leaves in total**. **No tree was
visualised** — 100 tree diagrams would be unreadable and add nothing.
Structural summaries are sufficient.

---

## Feature importance

`results/models/m3/m3_feature_importances.csv` — 43 rows in the **frozen
feature order**, with a separate `rank` column. Importances sum to
1.000000000000000.

| Feature | Importance | Rank |
|---|---:|---:|
| `DEWP` | 0.290604 | 1 |
| `TEMP` | 0.157607 | 2 |
| `Iws` | 0.120227 | 3 |
| `PRES` | 0.108188 | 4 |
| `month_10` | 0.078892 | 5 |
| `cbwd_NW` | 0.052027 | 6 |
| `month_3` | 0.029177 | 7 |
| `month_2` | 0.019837 | 8 |

**All 43 features have non-zero importance**, against 31 of 43 for the
single M2 tree. That is a property of bagging rather than a discovery: with
100 bootstrap samples, even a weak predictor is eventually chosen for some
split in some tree.

The top four are the same continuous meteorological variables M2 ranked
first, in the same order.

### Feature-importance limitations

Random-forest impurity-based importance must not be read as scientific
mechanism:

- **It is predictive, not causal.** It measures variance reduction from
  splitting, nothing more.
- **It favours continuous and high-cardinality variables.** `DEWP`, `TEMP`,
  `PRES` and `Iws` offer hundreds or thousands of candidate split points; a
  binary dummy offers exactly one. That structural advantage alone partly
  explains the top four.
- **It divides importance unpredictably across correlated predictors.**
  Phase 3 measured |r| between 0.78 and 0.83 among `DEWP`, `TEMP` and
  `PRES`. Where several correlated variables carry the same signal, the
  split of credit between them is close to arbitrary.
- **It describes this fitted ensemble**, not an environmental mechanism. A
  different seed, depth or training period could redistribute it.

Statements such as *"`DEWP` causes PM2.5"* are **not supported** and are not
made. **No feature selection was performed from these values, and the
Phase 5 schema was not altered.**

---

## Residual diagnostics

Descriptive only, on 2013 validation. **Residual = actual − predicted.**

| Statistic | **M3** | M2 | M1 |
|---|---:|---:|---:|
| Mean residual | +3.127329 | +2.777733 | +6.560091 |
| Median residual | −9.020883 | −6.133929 | −7.353553 |
| Residual std. dev. | 73.027858 | 77.939120 | 76.281242 |
| Minimum residual | −392.789071 | −329.723077 | −175.470247 |
| Maximum residual | +688.119063 | +698.802198 | +687.220866 |
| Pearson corr(predicted, residual) | **+0.001759** | −0.158984 | +0.022515 |
| Negative predictions | **0** | 0 | 658 |

Three observations.

**The predicted-versus-residual correlation is +0.0018 — essentially
zero**, and the best of the three feature-based models. M2's −0.159 showed
systematic regression toward leaf means; averaging 100 trees has almost
entirely removed that pattern.

**Residual spread is the tightest of any model** (std 73.03 against M1's
76.28 and M2's 77.94), consistent with M3's RMSE advantage.

**Zero negative predictions.** Predictions span **10.312080 to 431.094649
µg/m³**, entirely inside the training target range of 0–994. This is
structural — a forest averages leaf means of observed non-negative targets —
and required no clipping. **Nothing was clipped anywhere in this phase.**

Prediction granularity improved sharply: **7,632 distinct predicted values**
across 8,678 validation hours, against M2's 628. Averaging 100 trees turns a
coarse step function into something close to continuous.

**Severe episodes remain hard.** The maximum residual is still +688 µg/m³.
M3 has not solved extreme-event prediction; it has reduced the penalty. Full
severe-episode analysis belongs to Protocol Phase 11, and **the selection
metric was not changed to favour extreme-error performance.**

**Figures:** `figures/m3_validation_actual_vs_predicted.png` (with a y = x
reference line) and `figures/m3_validation_residuals_vs_predicted.png`, both
labelled *2013 Development Validation*.

---

## Test-set quarantine

**The 2014 partition was not read, not predicted and not evaluated, and no
2014 information influenced hyperparameter selection.**

A token-level audit of both new source files found **zero executable
references** to `X_test`, `y_test`, the test partition file, `m3_test` or
2014 evaluation. No `m3_test_predictions.csv`, `test_predictions.csv` or
`y_test.csv` exists anywhere in the repository.

The manifest records `test_evaluation_status: not_evaluated`,
`final_test_status: not_evaluated` and
`selection_used_test_information: false`, with no placeholder 2014 number.

**No further search was performed after selection.** Depth 12 winning is not
a reason to try 10, 11, 13 or 14; leaf 1 winning is not a reason to try 2 or
5; `"auto"` winning is not a reason to try 0.7 or 0.9; and no second seed,
larger forest or `bootstrap=False` variant was fitted.

---

## Final-refit policy

**Documented; deliberately not executed.**

At Protocol Phase 10:

1. The selected hyperparameters — `max_depth=12`, `min_samples_leaf=1`,
   `max_features="auto"`, `n_estimators=100`, `random_state=42` and the rest
   — **remain frozen**.
2. Combine the permitted 2010–2013 development data.
3. Refit `RandomForestRegressor` with exactly those parameters, retaining
   `random_state=42`.
4. Generate 2014 predictions **once**.
5. Open the locked 2014 target **once**.
6. Calculate MAE, RMSE and R² **once**.
7. **Do not retune after seeing 2014.**

**No serialized model was written.** Phase 10 refits on a different period,
so a pickled development forest would not be the final model; the
configuration, metrics, predictions, importances and structural diagnostics
describe this fit deterministically.

---

## Limitations

- **Development validation, not generalization.** 2013 figures only.
  Nothing here says how M3 behaves on 2014.
- **2013 was used to select M3's hyperparameters**, so its 2013 figure is
  optimistic relative to M0 and M1, which selected nothing. M2 shares this
  caveat. It must be restated when Phase 10 results are reported — and it is
  the reason M3's apparent margin over M1 should not be taken at face value
  until the locked test is opened.
- **Interpretability is the price.** 100 trees with ~166,000 leaves cannot
  be inspected the way M1's coefficient vector or M2's single tree can.
  Impurity importances are a weak substitute, for the reasons above.
- **Severe episodes remain badly under-predicted** — maximum residual
  +688 µg/m³.
- **Meteorology is an incomplete system.** Emissions, traffic and
  industrial activity are unmeasured. R² of 0.44 means well over half the
  variance in hourly PM2.5 is unexplained by weather and calendar position
  alone, and no model in the frozen set can change that.
- **One seed.** The forest's variance across seeds was not measured, by
  design.
- **The grid is spanning, not exhaustive.** A different region of Random
  Forest design space might do better; it was not searched, and that is a
  deliberate constraint rather than an oversight.

---

## Pre-test development freeze

M3 completes development. Immediately after its selection was validated and
shown reproducible, `artifacts/pretest_development_freeze.json` was written,
sealing the complete pre-test state: the raw dataset digest, the split and
feature-schema digests, and each of the four models' specification, selected
hyperparameters, manifest digest and Phase-10 refit rule.

**From this point, Protocol Phase 10 may not change:** the feature schema,
the M0 strategy, the M1 specification, the M2 selected hyperparameters, the
M3 selected hyperparameters, the split boundaries, the primary metric, or
the test policy.

If a later defect requires changing any of those, the planned final-test
evaluation must be **invalidated and reset**, and the deviation documented
**before** the test is reopened. **Nothing may be silently repaired after
seeing 2014.**

Phase 10 can begin by validating that single artifact and know the entire
pre-test development state is unchanged.

**Protocol Phase 10 has not started. The 2014 final test remains sealed.**
