# AirSense V1 Feature Preparation

**Research-protocol mapping: Protocol Phase 5 — Feature Preparation.**

This document records how the frozen Phase 0 feature policy was resolved
into a concrete schema, after the Phase 3 exploratory analysis and the
Phase 4 split freeze.

> **All feature decisions recorded here were fixed before observing any
> predictive model performance.** No model has been fitted, no prediction
> generated, and no MAE, RMSE or R² computed anywhere in this repository.

---

## Purpose

Turn the Phase 4 partitions into one shared numeric feature matrix that
M1, M2 and M3 will all consume, and record every inclusion and exclusion
with its reason.

Companion documents: [`FEATURE_POLICY.md`](FEATURE_POLICY.md) is the
Phase 0 declaration and is **unchanged** — it states what was permitted and
what still had to be decided. This document states what was decided, and
why. Machine-readable form:
[`artifacts/feature_schema.json`](../artifacts/feature_schema.json); tabular
form: `results/features/feature_decisions.csv`.

---

## Frozen split context

Inherited unchanged from
[`CLEANING_AND_SPLIT_POLICY.md`](CLEANING_AND_SPLIT_POLICY.md):

| Partition | Period | Supervised rows |
|---|---|---:|
| Development train | 2010-01-01 → 2012-12-31 | 24,418 |
| Validation | 2013-01-01 → 2013-12-31 | 8,678 |
| Locked final test | 2014-01-01 → 2014-12-31 | 8,661 |

Upstream digests were verified against
[`artifacts/split_manifest.json`](../artifacts/split_manifest.json) before
this phase began and again after it finished. Nothing upstream was
regenerated or modified.

---

## Training-only preprocessing rule

**Categorical vocabularies are established from the 2010–2012
development-training predictors and then frozen.** Validation and test are
*applied* against that frozen schema — they are compatibility checks, never
contributors to it.

The implementation does **not** call `get_dummies` separately on each
partition and reconcile the columns afterwards. That pattern lets a
partition's own contents shape the representation, which is exactly what
this rule forbids. Instead the frozen vocabulary is applied by explicit
comparison, so the column set is identical by construction.

An unseen categorical level in validation or test **raises an error and
stops the phase**. It does not silently add a feature column. In this run
no unseen level occurred in either partition.

The same principle applies to everything else: there is no scaler, no
encoder state and no imputer whose parameters could have been fitted
outside the training period. **The only learned preprocessing state in V1 is
the three categorical vocabularies**, and all three come from training
alone.

---

## Shared M1–M3 feature representation

One matrix serves **M1 (Linear Regression)**, **M2 (Decision Tree
Regressor)** and **M3 (Random Forest Regressor)**.

This is deliberate. If each model received features tuned to its own
strengths, the Phase 10 comparison would confound model class with feature
engineering, and it would no longer answer the pre-registered question of
how far *classical model families* get on this problem. A shared matrix
makes the comparison about the models.

The cost is accepted knowingly: one-hot blocks and unscaled skewed
predictors are not what one would choose for a linear model alone, and the
reference-category constraint is unnecessary for trees. Both are tolerated
so the inputs stay identical.

---

## Included meteorological variables

Six, used **exactly as distributed**:

| Feature | Description | Units |
|---|---|---|
| `DEWP` | dew point | °C |
| `TEMP` | temperature | °C |
| `PRES` | pressure | hPa |
| `Iws` | cumulated wind speed | m/s |
| `Is` | cumulated hours of snow | hours |
| `Ir` | cumulated hours of rain | hours |

No standardisation, normalisation, log or square-root transform, clipping,
winsorizing, binning or polynomial expansion.

`DEWP`, `TEMP` and `PRES` are direct meteorological measurements. `Iws`,
`Is` and `Ir` are cumulative or sparse quantities whose unusual
distributions Phase 3 documented — `Iws` has a median of 5.37 m/s against a
maximum of 585.60, and `Is` and `Ir` are zero in more than 75% of hours —
but **no evidence establishes those values as erroneous**. Skew is a
property of a cumulative quantity, not a defect.

Keeping them raw preserves a simple, period-authentic baseline. It also
means M1 is deliberately allowed to expose the limits of a linear
functional form on non-linear data, while M2 and M3 can represent that
structure without any manual numerical transformation. That contrast is
part of what the model comparison is for.

---

## Month encoding

`month` enters as a **one-hot categorical**, not as a raw integer.

- Training vocabulary verified from 2010–2012: **all twelve months 1–12
  present**.
- Reference category: **`month_1`**, omitted from the matrix.
- Columns produced: `month_2` … `month_12` — **11 features**.

Phase 3 found monthly means ranging from 80.00 µg/m³ (August) to
125.74 (February) with no monotone ordering — February and October are both
high while the months between them are not. An integer month would force a
linear model to fit a straight line through that pattern. Categorical
encoding lets it fit each month independently.

---

## Hour encoding

`hour` enters as a **one-hot categorical**, not as a raw integer.

- Training vocabulary verified from 2010–2012: **all twenty-four hours 0–23
  present**.
- Reference category: **`hour_0`**, omitted.
- Columns produced: `hour_1` … `hour_23` — **23 features**.

Phase 3 found a clear diurnal cycle: mean PM2.5 lowest at hour 15
(85.53 µg/m³) and highest at hour 01 (113.70). An integer encoding would
assert `23 > 22 > … > 0`, which is both false as an ordering and unable to
represent a cycle. Categorical encoding avoids imposing either.

---

## Wind-direction encoding

`cbwd` enters as a **one-hot categorical**.

- Training vocabulary **verified, not assumed**, from 2010–2012:
  **`NE`, `NW`, `SE`, `cv`** — four levels, matching the declared
  expectation. Training counts: `NW` 8,368 · `SE` 8,362 · `cv` 4,972 ·
  `NE` 2,716.
- Reference category: **`cbwd_NE`**, omitted.
- Columns produced: `cbwd_NW`, `cbwd_SE`, `cbwd_cv` — **3 features**.
- Validation and test contained **no unseen level**.

Phase 3 found this variable separates the target more than any other
examined — median 31 µg/m³ under `NW` against 98 under `cv`.

### Why a reference category is dropped

The shared matrix will be used by an unregularized `LinearRegression` **with
an intercept**. Retaining every dummy level of a categorical makes that
block sum to a column of ones, exactly collinear with the intercept, so the
coefficients are not uniquely determined.

The three references — `month_1`, `hour_0`, `cbwd_NE` — were **declared in
advance** and are not selected for predictive advantage. `month_1` and
`hour_0` are simply the first level of each; `cbwd_NE` was declared before
the category frequencies were inspected.

M2 and M3 do not need this constraint, but they use the same matrix anyway,
because keeping the representation identical across M1–M3 matters more than
giving the trees three extra redundant columns.

---

## Excluded identifiers and calendar fields

| Field | Decision | Reason |
|---|---|---|
| `No` | **excluded** | observation identifier and traceability field, not a physical predictor |
| `year` | **excluded** | see below |
| `day` | **excluded** | see below |
| `pm2.5` | **excluded from X** | the target; supplied only as `y_train` / `y_validation` |

`No`, `year`, `month`, `day` and `hour` remain available for timestamp
reconstruction and appear in the alignment manifest. Availability for
traceability is not admission as a predictor.

### Why `year` was excluded

The evaluation periods are **disjoint later years**: the model trains on
2010–2012 and is evaluated on 2013 and 2014. A numeric `year` would
therefore only ever appear at test time with values the model never saw
during fitting.

For a linear model that means extrapolating an arbitrary calendar counter
beyond its fitted range. For a tree model it is worse: a tree can only split
on thresholds it observed, so every 2014 row would fall into whichever
terminal region the largest training year occupied — an arbitrary
assignment, not a learned relationship.

Phase 3 also found **no monotonic annual trend** — annual means sit in a
91–104 µg/m³ band with 2012 lowest and 2013 higher again — so there is no
trend for such extrapolation to follow even in principle.

This decision rests on protocol reasoning and the Phase 3 evidence. **No
model score was consulted**, and none exists.

### Why day-of-month was excluded

Day-of-month has no predeclared physical interpretation in the V1
hypothesis. It is not naturally ordinal across month boundaries — the 31st
is adjacent to the 1st — and Phase 3 identified no defensible day-of-month
relationship.

Including it would give the tree models an opportunity to fit
calendar-position artefacts with no stated scientific rationale behind them.

---

## Why `season` was not duplicated

**No `season` feature was created.**

Phase 3 documented real seasonal structure, and `FEATURE_POLICY.md` listed
`season` as a candidate. But season is a deterministic function of month,
and the month one-hot block already carries a **more granular** form of the
same calendar information: any seasonal grouping is representable as a sum
of month indicators, so adding `season` supplies no information the matrix
does not already contain, while adding exact linear dependence among the
predictors.

`season` remains useful for descriptive work and for Phase 11 error
analysis. It is not a model input.

---

## Weekend admission rule outcome

**No `weekend`, `weekday` or `day_of_week` feature was created.**

`FEATURE_POLICY.md` §1 Class D admitted a weekend indicator **"only if
Phase 3 shows an actual weekday/weekend difference in PM2.5. Not included on
plausibility alone."**

Phase 3 did not perform that comparison. The predeclared admission
condition was therefore **not met**, and the feature is excluded.

The EDA was **not** reopened to test the criterion after the fact. Running a
new target-based comparison specifically to make a feature eligible would
convert a pre-registered condition into a post-hoc search, which is the
failure mode the policy exists to prevent.

**This records adherence to the frozen policy. It is not evidence that
weekends have no effect on PM2.5** — that question is simply unanswered in
V1, and it stays open.

---

## Why no cyclical sine/cosine features

**No `sin_month`, `cos_month`, `sin_hour` or `cos_hour` was created.**

Such encodings existed well before 2019 and would have been permissible.
They are unnecessary here: month and hour are already represented
categorically, and one-hot encoding is strictly more flexible — it lets a
linear model fit an arbitrary non-monotone shape, whereas a sine/cosine pair
imposes a smooth sinusoid.

Creating both would duplicate the same calendar information in two forms and
introduce a redundancy with no stated benefit.

---

## Why no scaling was used

**No `StandardScaler`, `MinMaxScaler`, `RobustScaler`, manual z-scoring or
normalisation.**

- **M1** is ordinary unregularized least squares. Predictor scaling does not
  change the fitted model or its predictions; it only rescales the
  coefficients. There is no penalty term whose behaviour would depend on
  feature magnitude.
- **M2 and M3** split on thresholds and are invariant to any monotonic
  rescaling of an individual feature.

Avoiding scaling also keeps meteorological features in their original units,
so coefficients and feature importances stay interpretable in µg/m³ per °C
and similar; eliminates another piece of learned preprocessing state that
would have to be fitted on training data and applied correctly to the
locked test set; and simplifies historical reproducibility.

Fixed before any model result existed.

---

## Why no numerical transformations were used

Beyond scaling, no log, square-root, clipping, winsorizing, binning or
polynomial expansion was applied to any predictor, and **no transformation
at all was applied to the target**.

`pm2.5` is written to `y_train` and `y_validation` exactly as it appears in
the Phase 4 data — the two zero-valued observations and the 994 µg/m³
maximum are all present and unmodified. Transforming the target would change
the quantity being predicted and would make MAE and RMSE no longer
interpretable in µg/m³, which the pre-registration fixed as the reporting
units.

---

## Test-target quarantine

**The 2014 target was never read in this phase.**

Enforcement, at three levels:

1. `data/processed/airsense_test_2014.csv` is read with an explicit
   `usecols` list of twelve predictor and metadata columns that **omits
   `pm2.5`**, so the target is never materialised into a DataFrame.
2. A code-level assertion requires `'pm2.5' not in test_dataframe.columns`
   and raises `QUARANTINE BREACH` otherwise.
3. After writing outputs the pipeline checks that no `y_test.csv` exists and
   aborts if one is found.

Verified in this run: `target_column_in_test_usecols: false`,
`target_column_in_test_dataframe: false`, `assertion_holds: true`,
`y_test_artifact_created: false`.

Test **predictors** were inspected — for schema conformity, categorical
compatibility, predictor completeness and feature-construction validity.
That uses no labels and is what makes the matrix verifiably usable at
Phase 10.

`results/features/feature_matrix_summary.csv` records a target row count for
train and validation, and leaves it **empty for test**: the count was never
computed.

The 2014 target remains solely inside the frozen Phase 4 test dataset until
Protocol Phase 10.

---

## Final 43-feature schema

One deterministic column order, identical in `X_train`, `X_validation` and
`X_test`:

| # | Block | Columns | Count |
|---|---|---|---:|
| 1–6 | Raw meteorological | `DEWP`, `TEMP`, `PRES`, `Iws`, `Is`, `Ir` | 6 |
| 7–17 | Month one-hot | `month_2` … `month_12` (ref `month_1`) | 11 |
| 18–40 | Hour one-hot | `hour_1` … `hour_23` (ref `hour_0`) | 23 |
| 41–43 | Wind direction one-hot | `cbwd_NW`, `cbwd_SE`, `cbwd_cv` (ref `cbwd_NE`) | 3 |
| | | **Total** | **43** |

| Matrix | Shape |
|---|---|
| `X_train` | 24,418 × 43 |
| `X_validation` | 8,678 × 43 |
| `X_test` | 8,661 × 43 |
| `y_train` | 24,418 × 1 |
| `y_validation` | 8,678 × 1 |
| `y_test` | **does not exist** |

Every column is numeric, with 41 integer and 2 floating-point columns
(`PRES`, `Iws` — the rest happen to be integral in these partitions). Zero
missing cells, zero non-numeric columns, zero infinite values, and every
dummy column strictly 0/1.

---

## Limitations

- **A shared matrix is a compromise.** It is not the representation one
  would choose for any single model in isolation. That is the price of a
  comparison that isolates model class.
- **Unscaled skewed predictors** mean M1's coefficients on `Iws`, `Is` and
  `Ir` will be dominated by rare large values; that is expected, and the
  limitation belongs to the linear functional form rather than to a
  preprocessing mistake.
- **43 one-hot columns for 34 calendar levels** is a wide, sparse block
  relative to the six physical measurements. The models may find calendar
  position easier to exploit than meteorology.
- **No interaction terms**, so M1 cannot represent any conditional
  relationship — for example wind speed mattering more in winter. M2 and M3
  can. This asymmetry is inherent in the pre-registered model set.
- **Meteorology alone is an incomplete system.** Emissions, traffic and
  industrial activity are unmeasured. Phase 3's small correlation
  magnitudes bound what any V1 model can achieve, and no feature choice here
  changes that.
- **Excluding `year`** means no model can represent a genuine multi-year
  shift, should one exist beyond the 2010–2014 window.

---

## What remains for Phase 6

Protocol Phase 6 establishes **M0**, the naive baseline predictor — a
constant, computed from the training target alone, giving the error floor
that M1–M3 must beat.

Nothing about M0 has been implemented here. No estimator has been imported
or instantiated, no prediction produced, and no metric computed. The feature
schema above is frozen input to that phase, not a result of it.
