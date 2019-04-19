# AirSense V1 M2 Decision Tree

**Research-protocol mapping: Protocol Phase 8 — Decision Tree Regression (M2).**

> **All metrics below are 2013 development-validation metrics. They are not
> final held-out test results.** The 2014 partition remains locked and was
> not read, and no 2014 information influenced hyperparameter selection.

---

## Purpose

Fit the first model that can represent non-linearity and interactions
without manual feature engineering, and measure what that buys over the
linear form.

M1 was constrained in ways Phase 3 predicted would matter: it could not
represent `TEMP`'s non-monotone relationship with PM2.5, could not model any
interaction, and produced 658 physically impossible negative predictions. M2
is structurally free of all three constraints. Whether that translates into
better validation error is the question this phase answers.

---

## Protocol position

M2 was pre-registered in [`RESEARCH_QUESTION.md`](RESEARCH_QUESTION.md) §3
as *"Decision Tree Regressor — single tree; captures non-linearity and
interactions, at a known risk of overfitting."* It follows M1 and precedes
M3 Random Forest.

It is also **the first model in the project requiring hyperparameter
selection**, which is why the search space and selection rule below were
frozen in code and documentation before a single tree was fitted.

---

## Frozen feature representation

The **exact same 43-column matrix M1 used**, consumed as produced and
verified against
[`artifacts/feature_schema.json`](../artifacts/feature_schema.json) before
fitting.

| Block | Columns | Count |
|---|---|---:|
| Raw meteorological | `DEWP`, `TEMP`, `PRES`, `Iws`, `Is`, `Ir` | 6 |
| Month one-hot | `month_2` … `month_12` | 11 |
| Hour one-hot | `hour_1` … `hour_23` | 23 |
| Wind direction one-hot | `cbwd_NW`, `cbwd_SE`, `cbwd_cv` | 3 |
| | **Total** | **43** |

**Nothing was altered to suit a tree.** No scaling, no transformation, no
feature added or removed, no reordering. The reference-category omissions
are unnecessary for a tree and were kept anyway — the point of a shared
matrix is that the M1-vs-M2 comparison reflects **model class**, not hidden
differences in feature engineering.

## Training period

**2010-01-01 00:00 → 2012-12-31 23:00** — 24,418 observations.

## Validation period

**2013-01-01 00:00 → 2013-12-31 23:00** — 8,678 observations. Used for both
evaluation and hyperparameter selection.

---

## Why a complexity search is required

A decision tree has no natural stopping point. Grown without constraint it
will keep splitting until every leaf is nearly pure, memorising the training
data and generalising poorly. Grown too shallow it is a coarse step function
barely better than a few group means.

Unlike M0 (nothing to tune) and M1 (an analytic solution with nothing to
tune), M2's behaviour is **entirely governed by where you stop growing it**.
That makes a complexity search unavoidable — and makes it essential that the
search space be declared in advance, since the temptation to keep probing
around a good value is exactly how a validation set gets overfitted.

The grid below is spanning, not exhaustive. It is **not** presented as
optimisation.

---

## Predeclared 20-candidate grid

Frozen in `src/models/decision_tree.py` before any tree was fitted.

**Searched — 5 × 4 = 20 candidates:**

| Parameter | Values | Rationale |
|---|---|---|
| `max_depth` | **3** | very shallow, high-bias tree |
| | **5** | moderately shallow |
| | **8** | moderate complexity |
| | **12** | relatively flexible |
| | **None** | unrestricted-depth reference |
| `min_samples_leaf` | **1** | fully local leaves permitted |
| | **10** | light regularization |
| | **50** | moderate regularization |
| | **100** | stronger smoothing |

**Fixed, not searched:** `criterion="mse"`, `splitter="best"`,
`min_samples_split=2`, `max_features=None`, `random_state=42`. Also not
searched: `max_leaf_nodes`, `min_impurity_decrease`.

`criterion="mse"` is the **scikit-learn 0.20 name**. `"squared_error"` did
not exist until 1.0 and would be a post-cutoff API.

`random_state=42` is fixed for every candidate. `splitter="best"` is largely
deterministic, but scikit-learn may break ties randomly; the fixed seed
removes that ambiguity. **Only one seed was used** — testing several would
be an undeclared search.

**Method:** an explicit deterministic loop, **not `GridSearchCV`**. The
evaluation design is a single declared validation *year*, not
cross-validation folds, and a plain loop keeps the candidate order and
per-candidate diagnostics auditable.

**Candidate order:** `max_depth` outer, `min_samples_leaf` inner, both in
declared order. This ordering is the final tie-break.

---

## Selection rule

**Primary: lowest 2013 validation MAE** — the decision metric frozen in
Phase 0, long before M2 existed.

Tie-break policy, also frozen before fitting, applied only to candidates
whose MAE ties the best within 1 × 10⁻¹²:

1. lower 2013 validation RMSE;
2. simpler tree — smaller configured `max_depth`, with `None` treated as
   greater than every finite depth;
3. if `max_depth` is equal — larger `min_samples_leaf`;
4. deterministic grid order.

R² is reported for every candidate but **never overrides** the MAE/RMSE
rule. Training metrics are computed for every candidate as an overfitting
diagnostic and **played no part in selection**.

**In this run only one candidate tied on MAE, so no tie-break was needed.**

---

## Full grid results

**2013 DEVELOPMENT VALIDATION.** All 20 candidates reported — none dropped.

| # | `max_depth` | `min_samples_leaf` | Train MAE | **Val MAE** | Val RMSE | Val R² | Depth | Leaves |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 1 | 56.544363 | 60.693377 | 86.160477 | 0.228049 | 3 | 8 |
| 2 | 3 | 10 | 56.544363 | 60.693377 | 86.160477 | 0.228049 | 3 | 8 |
| 3 | 3 | 50 | 56.544363 | 60.693377 | 86.160477 | 0.228049 | 3 | 8 |
| 4 | 3 | 100 | 56.544363 | 60.693377 | 86.160477 | 0.228049 | 3 | 8 |
| 5 | 5 | 1 | 50.776277 | 56.573156 | 82.518362 | 0.291932 | 5 | 32 |
| 6 | 5 | 10 | 50.780423 | 56.565594 | 82.517111 | 0.291954 | 5 | 32 |
| 7 | 5 | 50 | 50.781654 | 56.568572 | 82.517629 | 0.291945 | 5 | 31 |
| 8 | 5 | 100 | 50.858282 | 56.726335 | 82.743979 | 0.288055 | 5 | 27 |
| 9 | 8 | 1 | 43.920737 | 53.518392 | 80.115044 | 0.332576 | 8 | 246 |
| 10 | 8 | 10 | 44.570246 | 53.247627 | 78.844406 | 0.353579 | 8 | 194 |
| 11 | 8 | 50 | 45.864761 | 53.369367 | 78.838897 | 0.353670 | 8 | 129 |
| 12 | 8 | 100 | 46.525692 | 53.351415 | 79.272570 | 0.346539 | 8 | 101 |
| 13 | 12 | 1 | 32.935523 | 52.710904 | 82.616227 | 0.290252 | 12 | 1,980 |
| **14** | **12** | **10** | **37.559165** | **50.827853** | **77.984115** | **0.367609** | **12** | **799** |
| 15 | 12 | 50 | 42.300000 | 50.936087 | 78.232410 | 0.363575 | 12 | 281 |
| 16 | 12 | 100 | 44.439318 | 51.210764 | 77.874662 | 0.369383 | 12 | 159 |
| 17 | None | 1 | **0.055860** | **64.414035** | 96.978477 | 0.022033 | 45 | 23,075 |
| 18 | None | 10 | 31.371321 | 54.109051 | 81.907606 | 0.302375 | 31 | 1,879 |
| 19 | None | 50 | 41.188589 | 51.771866 | 79.051385 | 0.350181 | 21 | 377 |
| 20 | None | 100 | 43.961920 | 51.690769 | 78.210512 | 0.363932 | 17 | 188 |

Three patterns are worth recording.

**Candidate 17 is a textbook overfit.** Unrestricted depth with
`min_samples_leaf=1` drives training MAE to **0.056 µg/m³** — the tree has
essentially memorised 24,418 observations into 23,075 leaves — yet its
validation MAE of **64.41** is the **worst of all twenty**, worse than the
depth-3 stump. It is retained in the results, not hidden.

**Candidates 1–4 are numerically identical.** At `max_depth=3` a tree has at
most 8 leaves, each holding thousands of samples, so `min_samples_leaf` of
1, 10, 50 or 100 never binds. The constraint is inactive, not ignored.

**`min_samples_leaf` matters more as depth increases.** At depth 3 it does
nothing; at unrestricted depth it is the difference between 64.41 and 51.69.
It is the parameter doing the regularising once depth stops binding.

*Figure:* `figures/m2_grid_mae_by_complexity.png`.

---

## Selected configuration

**Candidate 14** — the unique lowest-MAE candidate.

| Parameter | Value |
|---|---|
| `criterion` | `"mse"` |
| `splitter` | `"best"` |
| **`max_depth`** | **12** |
| `min_samples_split` | 2 |
| **`min_samples_leaf`** | **10** |
| `max_features` | `None` |
| `random_state` | 42 |

---

## Training vs validation behaviour

| Metric | Training (2010–2012) | Validation (2013) | Degradation |
|---|---:|---:|---:|
| MAE | 37.559165 | **50.827853** | +13.27 (×1.35) |
| RMSE | 56.169988 | 77.984115 | +21.81 |
| R² | 0.603889 | 0.367609 | −0.236 |

The selected tree fits training data noticeably better than validation data —
training MAE is 26% lower. **Some overfitting remains at the selected
setting**, which is expected: the search chose the depth-12 tree because it
generalised best among the twenty, not because it stopped overfitting.

The grid makes the trade-off visible. Candidate 13 (same depth,
`min_samples_leaf=1`) reaches training MAE 32.94 — better than the selected
tree — but validation MAE 52.71, worse. Candidate 17 takes it to the
extreme. Training error is a poor guide to generalisation here, which is
precisely why selection used validation MAE and training metrics were
recorded as diagnostics only.

---

## 2013 validation results

**2013 DEVELOPMENT VALIDATION — not final test performance.**

| Metric | Value |
|---|---:|
| **MAE** | **50.827853** µg/m³ |
| **RMSE** | **77.984115** µg/m³ |
| **R²** | **0.367609** |

Independently recomputed from the prediction file without scikit-learn;
agreement within 1.4 × 10⁻¹⁴.

---

## Comparison with M0

M0's figures are read from its manifest, not transcribed.

| Metric | M0 (constant 73.0) | **M2** | Change | Relative |
|---|---:|---:|---:|---:|
| **MAE** *(primary)* | 67.591150 | **50.827853** | **−16.763297** | **−24.80%** |
| RMSE | 102.181753 | 77.984115 | −24.197638 | −23.68% |
| R² | −0.085726 | 0.367609 | +0.453335 | — |

M2 beats the naive floor decisively on all three metrics.

## Comparison with M1

**This is the more interesting comparison, and the metrics disagree.**

| Metric | M1 (linear) | **M2** | Change | Verdict |
|---|---:|---:|---:|---|
| **MAE** *(primary)* | 53.333361 | **50.827853** | **−2.505507** | **M2 better by 4.70%** |
| RMSE | 76.558423 | 77.984115 | **+1.425692** | **M1 better by 1.86%** |
| R² | 0.390520 | 0.367609 | **−0.022911** | **M1 better** |

**M2 wins on the pre-registered decision metric and loses on the other two.**

This is exactly the situation the pre-registration exists to handle. The
decision metric was fixed as **MAE** in Phase 0, before any model existed,
so **by the declared criterion M2 is the better model** — and that judgement
does not change because RMSE and R² point the other way.

The disagreement is informative rather than awkward. MAE and RMSE weight
errors differently: RMSE squares them, so it is dominated by the largest
misses. M2 improves the *typical* hour — its leaf means track local
conditions better than a global linear surface — while making *worse
extreme* errors, as its residual range shows (−329.72 to +698.80, against
M1's −175.47 to +687.22). A tree assigns each validation hour the mean of
its leaf; when an hour is unlike anything in that leaf, the error can be
large in either direction, whereas a linear surface degrades more smoothly.

**No metric was switched to make either model look better.** Both readings
are reported, and the ranking that counts is the declared one.

---

## Tree complexity

| Property | Value |
|---|---:|
| Configured `max_depth` | 12 |
| **Observed tree depth** | **12** |
| Configured `min_samples_leaf` | 10 |
| **Node count** | **1,597** |
| **Leaf count** | **799** |
| **Root split feature** | **`DEWP`** |
| Features with non-zero importance | **31 of 43** |

The tree reached its configured depth limit, so depth was the binding
constraint rather than leaf purity.

**The root split feature is not a causal claim.** `DEWP` is where this
fitted partition happened to achieve the largest first-split variance
reduction. It is a property of this tree, not evidence that dew point is a
dominant environmental driver.

**No full tree diagram was drawn.** With 1,597 nodes it would be unreadable
and scientifically useless. The structural statistics above serve the
purpose.

---

## Feature importance

`results/models/m2/m2_feature_importances.csv` — 43 rows in the **frozen
feature order**, with a separate `rank` column. Importances sum to 1.0
(observed 1.0000000000000002, floating-point exact).

| Feature | Importance | Rank |
|---|---:|---:|
| `DEWP` | 0.302131 | 1 |
| `TEMP` | 0.165750 | 2 |
| `Iws` | 0.118759 | 3 |
| `PRES` | 0.105750 | 4 |
| `month_10` | 0.093894 | 5 |
| `cbwd_NW` | 0.059635 | 6 |

**12 of the 43 features have zero importance.** They were **retained in the
model** — zero importance in this tree is not grounds for removing a feature
from a frozen schema, and these values were **not used to redesign Phase 5**.

### Feature-importance limitations

Impurity-based decision-tree importance must not be read as scientific
mechanism:

- **It is predictive, not causal.** It measures variance reduction achieved
  by splitting, nothing more.
- **It favours variables offering many possible split points.** The four
  continuous meteorological variables take hundreds or thousands of distinct
  values; a binary dummy offers exactly one split. That structural advantage
  alone partly explains why `DEWP`, `TEMP`, `Iws` and `PRES` occupy the top
  four positions.
- **It distributes importance unpredictably among correlated predictors.**
  Phase 3 measured |r| between 0.78 and 0.83 among `DEWP`, `TEMP` and
  `PRES`. When several correlated variables carry the same signal, whichever
  is chosen first absorbs the credit and the others appear less important
  than they are.
- **It reflects only this fitted partition.** A different seed, depth or
  training period could redistribute it.

Statements such as *"`DEWP` causes PM2.5"* or *"`DEWP` is definitively the
most important environmental driver"* are **not supported** and are not made
here.

---

## Residual diagnostics

Descriptive only, on 2013 validation. **Residual = actual − predicted.**

| Statistic | M2 | (M1 for context) |
|---|---:|---:|
| Mean residual | +2.777733 | +6.560091 |
| Median residual | −6.133929 | −7.353553 |
| Residual std. dev. | 77.939120 | 76.281242 |
| Minimum residual | **−329.723077** | −175.470247 |
| Maximum residual | **+698.802198** | +687.220866 |
| Pearson corr(predicted, residual) | **−0.158984** | +0.022515 |

**M2 produced zero negative predictions**, against M1's 658 (7.58%).
Predictions span **8.454545 to 377.900000 µg/m³**, entirely inside the
training target range of 0–994. This is structural, not luck: a regression
tree predicts the mean of the training targets in a leaf, so its output
cannot leave the convex range of what it was trained on. It is a real
advantage of the tree over unconstrained OLS, and it required no clipping —
**no prediction was clipped anywhere in this phase.**

**The predicted-versus-residual correlation is −0.159**, materially further
from zero than M1's +0.023. Negative correlation means the tree tends to
*over*-predict where it predicts high and *under*-predict where it predicts
low — the classic regression-toward-the-leaf-mean pattern. Recorded as an
observation; full analysis belongs to Phase 11.

High-concentration validation observations were **not removed and not
separately refitted around**.

**Figures:** `figures/m2_validation_actual_vs_predicted.png` (with a y = x
reference line) and `figures/m2_validation_residuals_vs_predicted.png`, both
labelled *2013 Development Validation*.

---

## Test-set quarantine

**The 2014 partition was not read, not predicted and not evaluated, and no
2014 information influenced hyperparameter selection.**

A token-level audit of both new source files found **zero executable
references** to `X_test`, `y_test`, the test partition file, `m2_test` or
2014 evaluation. No `m2_test_predictions.csv`, `test_predictions.csv` or
`y_test.csv` exists anywhere in the repository.

The selected configuration was chosen **exclusively from 2013 development
validation**. The manifest records `test_evaluation_status: not_evaluated`,
`final_test_status: not_evaluated` and
`selection_used_test_information: false`, with no placeholder 2014 number.

**No further search was performed after selection.** Depth 12 winning is not
a reason to try 11 or 13; leaf 10 winning is not a reason to try 5 or 20.
Doing either would make the declared grid meaningless.

---

## Final-refit policy

**Documented; deliberately not executed.**

At Protocol Phase 10:

1. The selected hyperparameters (`max_depth=12`, `min_samples_leaf=10`, and
   the fixed parameters) **remain frozen**.
2. Combine the 2010–2013 development data.
3. Refit `DecisionTreeRegressor` with exactly those parameters.
4. Generate 2014 predictions **once**.
5. Open the locked 2014 target **once**.
6. Calculate MAE, RMSE and R² **once**.

**M2 is not retuned after 2014 becomes visible.**

**No serialized model was written.** Phase 10 refits on a different period,
so a pickled development tree would not be the final model; the
configuration, metrics, predictions, importances and structural diagnostics
describe this fit deterministically.

---

## Limitations

- **Development validation, not generalization.** 2013 figures only. Nothing
  here says how M2 behaves on 2014.
- **The validation year was used to select hyperparameters**, so 2013
  performance is mildly optimistic for M2 in a way it is not for M0 or M1 —
  neither of which had anything to select. This is why the locked 2014 test
  exists, and it should be restated when Phase 10 results are reported.
- **Overfitting is reduced, not eliminated** — training MAE remains 26%
  below validation MAE at the selected setting.
- **A single tree is high-variance.** Small changes to the training data can
  substantially change the partition. M3 exists to address exactly this.
- **Piecewise-constant predictions.** Only 628 distinct values across 8,678
  validation hours; the tree cannot interpolate within a leaf.
- **Worse extreme errors than M1**, despite better typical error.
- **Feature importances are not mechanism**, for the reasons above.
- **Meteorology is an incomplete system** — emissions, traffic and
  industrial activity are unmeasured, bounding what any V1 model achieves.

---

## What remains for M3

Protocol Phase 9 fits **M3, a Random Forest Regressor**, on the same
2010–2012 partition using the same frozen 43-feature matrix, evaluated on
2013 with the same three metrics.

M3 is of interest precisely where M2 is weakest: averaging many decorrelated
trees is the standard remedy for the high variance of a single tree, and it
should reduce the extreme errors that cost M2 its RMSE and R² advantage over
M1. Whether it does is an open question, and M3's own hyperparameter search
will need declaring in advance in the same way.

Nothing about M3 has been implemented. No ensemble estimator was imported or
instantiated in this phase.
