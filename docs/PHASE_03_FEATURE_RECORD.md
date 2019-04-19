# Phase 3 Feature Record — AirSense V1

**Research-protocol mapping: Protocol Phase 5 — Feature Preparation.**

> Record-sequence numbering (`PHASE_03_`) is independent of protocol phase
> numbering. Earlier records were **not** renamed. Mapping:
> `PHASE_00_FOUNDATION_RECORD.md` (protocol Phases 1–2),
> `PHASE_00A_RUNTIME_RECORD.md` (runtime recovery),
> `PHASE_01_EDA_RECORD.md` (protocol Phase 3),
> `PHASE_02_CLEANING_RECORD.md` (protocol Phase 4),
> this record (protocol Phase 5).

- **Recorded:** 2026-09-05 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE**
- **Decision document:** [`FEATURE_PREPARATION.md`](FEATURE_PREPARATION.md)
- **Machine-readable schema:** [`artifacts/feature_schema.json`](../artifacts/feature_schema.json)

> No model was trained, no prediction generated, and no predictive metric
> calculated. The 2014 target was never read.

---

## 1. Execution runtime

| Property | Value |
|---|---|
| **Interpreter** | `/home/debalekha_chakraborty/AirSense/venv/bin/python` |
| **Python** | CPython **3.6.7** (`sys.version_info[:3] == (3, 6, 7)`) |
| `sys.prefix` | `<repo>/venv` |
| Preflight | **READY**, exit 0 |

Frozen pins verified before any work: numpy 1.15.4, pandas 0.23.4,
scipy 1.1.0, scikit-learn 0.20.0, matplotlib 3.0.2, seaborn 0.9.0 — all
MATCH. System Python 3.11.2 was not used for scientific execution. No
dependency was added, removed, upgraded or downgraded.

`src/features/preparation.py` imports only pandas, numpy and the standard
library. **No scikit-learn estimator is imported anywhere in this phase.**

---

## 2. Upstream integrity

Verified before the phase and again after it.

| Input | Expected | Result |
|---|---|---|
| Raw dataset SHA-256 | `4127f868…6d27f1c` | MATCH |
| `airsense_supervised.csv` | manifest digest | MATCH |
| `airsense_train_2010_2012.csv` | manifest digest, 24,418 rows | MATCH |
| `airsense_validation_2013.csv` | manifest digest, 8,678 rows | MATCH |
| `airsense_test_2014.csv` | manifest digest, 8,661 rows | MATCH |
| `artifacts/split_manifest.json` | present and parsed | OK |

No upstream phase was regenerated. `data/raw/` and the Phase 4 partitions
are byte-identical to their recorded digests.

---

## 3. Input row counts

| Partition | Rows loaded | Expected |
|---|---:|---:|
| development_train | 24,418 | 24,418 |
| validation | 8,678 | 8,678 |
| test (predictors only) | 8,661 | 8,661 |

Predictor completeness re-verified: **0 missing values** across all selected
predictors in every partition. No imputer was introduced.

---

## 4. Inclusion decisions

| Source field | Representation | Features |
|---|---|---:|
| `DEWP` | raw continuous, unchanged | 1 |
| `TEMP` | raw continuous, unchanged | 1 |
| `PRES` | raw continuous, unchanged | 1 |
| `Iws` | raw cumulative, unchanged | 1 |
| `Is` | raw sparse cumulative, unchanged | 1 |
| `Ir` | raw sparse cumulative, unchanged | 1 |
| `month` | one-hot, reference `month_1` omitted | 11 |
| `hour` | one-hot, reference `hour_0` omitted | 23 |
| `cbwd` | one-hot, reference `cbwd_NE` omitted | 3 |
| | **Total** | **43** |

## 5. Exclusion decisions

| Source field | Reason |
|---|---|
| `No` | observation identifier, not a physical predictor; retained only in alignment manifests |
| `year` | evaluation periods are disjoint later years; numeric year would require extrapolating a calendar counter into unseen values, and Phase 3 found no monotonic annual trend |
| `day` | no predeclared physical interpretation, not ordinal across month boundaries, no Phase 3 evidence |
| `pm2.5` | the target; supplied only as `y_train` / `y_validation` |
| `season` | deterministic function of month; the month one-hot block already carries a more granular form of the same information |
| `weekend` / `day_of_week` | the Phase 0 admission condition (a weekday/weekend PM2.5 difference established by pre-modelling EDA) was **not met**, because Phase 3 did not perform that comparison. The EDA was not reopened to make the feature eligible. |
| sin/cos month and hour | month and hour are already categorical; a cyclical form would duplicate the same information under an imposed sinusoidal shape |
| lags, rolling means, interactions, polynomials, target encodings, AQI bands | outside the frozen V1 feature policy; a lagged target would change the task to forecasting |

Full table with reasons: `results/features/feature_decisions.csv` (17 rows).

---

## 6. Training-derived categorical vocabularies

Established from the 2010–2012 development-training predictors **only**.
Verified against the declared expectation, never assumed.

| Field | Vocabulary | Count | Reference level |
|---|---|---:|---|
| `month` | 1, 2, …, 12 | 12 | **`1`** |
| `hour` | 0, 1, …, 23 | 24 | **`0`** |
| `cbwd` | `NE`, `NW`, `SE`, `cv` | 4 | **`NE`** |

Training `cbwd` frequencies: `NW` 8,368 · `SE` 8,362 · `cv` 4,972 ·
`NE` 2,716.

Reference levels were declared before the vocabularies were inspected and
are not chosen for predictive advantage. They are dropped so the dummy
blocks are not exactly collinear with the intercept of an unregularized
`LinearRegression`.

**Vocabularies were not unioned across partitions.** `get_dummies` was not
called per partition; the frozen vocabulary is applied by explicit
comparison, so the column set is identical by construction. An unseen level
would raise and stop the phase — **none occurred** in validation or test.

---

## 7. Ordered feature schema

```
DEWP, TEMP, PRES, Iws, Is, Ir,
month_2 … month_12,
hour_1 … hour_23,
cbwd_NW, cbwd_SE, cbwd_cv
```

**Feature count: 43.** Identical order in `X_train`, `X_validation` and
`X_test`, shared by M1, M2 and M3.

---

## 8. Output dimensions and hashes

| Artifact | Shape | SHA-256 |
|---|---|---|
| `data/processed/features/X_train.csv` | 24,418 × 43 | `2c61894b4ca4e7657cc12c45f8bfe551a2c7121a90ca1ac0102742bd2ba3353c` |
| `data/processed/features/X_validation.csv` | 8,678 × 43 | `21c125778d535c124dc7642b0447086d4aa52aac4573d1fa0419ace725700069` |
| `data/processed/features/X_test.csv` | 8,661 × 43 | `def2ef73167e43a0bc4fb7521bb228b560cec047faab84e39724dd6416ee67c8` |
| `data/processed/features/y_train.csv` | 24,418 × 1 | `fbcea1e4611e69cf707776e8925daed511539ee08a9809b54c796754fa7b53fa` |
| `data/processed/features/y_validation.csv` | 8,678 × 1 | `9dc04233f80940865e9ae25e8380b458437fd0dece16b5c9727d72f891c1b0be` |
| `artifacts/feature_row_manifest.csv` | 41,757 rows | `5acddf85355b0daddf5c20dbcbf9867236dab3baddcdac316e5775d950846329` |
| `artifacts/feature_schema.json` | — | `a5d7cd37ca1872f299e42e219369b5b79c084d1be679214d96d9878c62446b2f` |
| `results/features/feature_decisions.csv` | 17 rows | `64c11b6a05cf3cc30cd2ce8de9d954af0d71050624c7e1e8f8a9ebc78906e02a` |
| `results/features/feature_matrix_summary.csv` | 3 rows | `e8fbb475bef967bc4f39e7f6f2314ea67bc250aabb93a2d6cc930a4f301464bf` |

**`y_test.csv` does not exist and was never created.**

---

## 9. Test-target quarantine verification

| Check | Result |
|---|---|
| `pm2.5` present in `TEST_USECOLS` | **false** |
| `pm2.5` present in the test DataFrame | **false** |
| Assertion `'pm2.5' not in test_dataframe.columns` | **holds** |
| `y_test` artifact created | **false** |
| `y_test.csv` exists on disk | **no** |
| Any file matching `*y_test*` / `*test*target*` | **none** |
| Test target row count recorded in the summary | **empty — never computed** |
| Quarantine status | **ACTIVE** |

The test file is read with an explicit twelve-column `usecols` list
(`No`, `year`, `month`, `day`, `hour`, `DEWP`, `TEMP`, `PRES`, `cbwd`,
`Iws`, `Is`, `Ir`) that omits the target, so `pm2.5` is never materialised.
The pipeline additionally refuses to finish if a `y_test.csv` is found.

Test **predictors** were inspected for schema conformity, categorical
compatibility, completeness and construction validity. That uses no labels.

---

## 10. Validation results

| Check | Result |
|---|---|
| Cross-partition column order identical | **yes** — all three |
| Feature count == 43 in every partition | **yes** |
| Missing feature cells | **0** in all partitions |
| Non-numeric / object-dtype columns | **0** |
| Infinite values | **0** |
| Dummy columns strictly 0/1 | **yes** |
| Dummy blocks mutually exclusive | **yes** |
| Unseen categorical levels | **none** in validation or test |
| Meteorological features match source exactly | **yes**, all six, all partitions |
| Month dummies map back to original month | **yes** (all-zero → `month_1`) |
| Hour dummies map back to original hour | **yes** (all-zero → `hour_0`) |
| `cbwd` dummies map back to original level | **yes** (all-zero → `NE`) |
| Row order preserved | **yes** |
| Target leakage token scan on feature names | **clean** |
| `y_train` values == source `pm2.5`, in order | **yes** |
| `y_validation` values == source `pm2.5`, in order | **yes** |
| Target untransformed | range 0–994, **2 zero values retained** |

### Row-alignment manifest

`artifacts/feature_row_manifest.csv` — **41,757 rows**, columns
`partition`, `matrix_row_number`, `No`, `timestamp`. It does **not** contain
`pm2.5`.

| Partition | Manifest rows | First `No` | Last `No` |
|---|---:|---:|---:|
| development_train | 24,418 | 25 | 26,304 |
| validation | 8,678 | 26,305 | 35,064 |
| test | 8,661 | 35,065 | 43,824 |

Verified that manifest row *N* of each partition names the observation at
matrix row *N*, and that `matrix_row_number` is sequential from 0.

---

## 11. Validation commands executed

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Runtime gate | `venv/bin/python --version` + pin verification | 3.6.7, all MATCH |
| 2 | Preflight | `venv/bin/python scripts/preflight.py` | **READY**, exit 0 |
| 3 | Upstream gate | digests vs `split_manifest.json` | all MATCH |
| 4 | Phase 5 run 1 | `venv/bin/python scripts/prepare_features.py` | exit 0 |
| 5 | Phase 5 run 2 | same | exit 0 |
| 6 | Determinism | SHA-256 of all 9 artifacts, run 1 vs run 2 | **all byte-identical** |
| 7 | Independent round-trip | recomputed outside the pipeline | all PASS |
| 8 | `y_test` absence | filesystem scan | none found |
| 9 | Python 3.6 compile | `py_compile` on all project `.py` | all OK |
| 10 | Final upstream gate | raw + Phase 4 digests | all MATCH |
| 11 | Git status | read-only inspection | nothing staged |

### Deterministic rerun result

| Artifact class | Count | Run 1 vs run 2 |
|---|---:|---|
| Feature matrices and targets | 5 | **byte-identical** |
| Manifests (`feature_row_manifest.csv`, `feature_schema.json`) | 2 | **byte-identical** |
| Audit tables | 2 | **byte-identical** |

No execution timestamp is written into any Phase 5 artifact, and no random
operation is used anywhere, which is what makes the byte-level guarantee
achievable.

### Python 3.6 compatibility

All project `.py` files compile under CPython 3.6.7, including the two added
here. No post-3.6 syntax, no API beyond the frozen historical stack, and no
new dependency.

---

## 12. Files created and modified

**Created**

| Path | Purpose |
|---|---|
| `src/features/preparation.py` | feature preparation implementation |
| `scripts/prepare_features.py` | thin entry point |
| `data/processed/features/X_train.csv` | 24,418 × 43 |
| `data/processed/features/X_validation.csv` | 8,678 × 43 |
| `data/processed/features/X_test.csv` | 8,661 × 43 (predictors only) |
| `data/processed/features/y_train.csv` | 24,418 × 1 |
| `data/processed/features/y_validation.csv` | 8,678 × 1 |
| `artifacts/feature_row_manifest.csv` | 41,757-row alignment manifest |
| `artifacts/feature_schema.json` | machine-readable frozen schema |
| `results/features/feature_decisions.csv` | 17-row decision table |
| `results/features/feature_matrix_summary.csv` | per-partition validation |
| `docs/FEATURE_PREPARATION.md` | decision document |
| `docs/PHASE_03_FEATURE_RECORD.md` | this record |

**Modified**

| Path | Change |
|---|---|
| `README.md` | Phase 5 status and high-level summary |
| `docs/FEATURE_POLICY.md` | **navigation link only** — one pointer to `FEATURE_PREPARATION.md`. No original scientific statement was altered. |

**Not modified by this phase:** `V1_RESEARCH_PROTOCOL.md`
(it still shows as modified in `git status`, but that is the uncommitted
administrative status edit made in Phase 4 — Phase 5 did not touch it),
`CLEANING_AND_SPLIT_POLICY.md`,
`EDA_ANALYSIS.md`, `RESEARCH_QUESTION.md`, `DATASET_AUDIT.md`,
`requirements-v1-2019.txt`, `requirements-v1-2019-lock.txt`, `data/raw/`,
and the Phase 4 partitions.

`FEATURE_POLICY.md` remains the Phase 0 **declaration**: its candidate
statements were deliberately left as written rather than being rewritten
into final decisions, so the record of what was open before EDA stays
intact.

---

## 13. Warnings

**W1 — The shared matrix is a deliberate compromise.** It is not the
representation one would choose for any single model alone. Accepted so the
Phase 10 comparison isolates model class rather than feature engineering.

**W2 — Calendar features outnumber physical ones 34 to 6.** The one-hot
blocks are wide and sparse relative to the meteorological measurements. This
is a property of the chosen encoding and is noted for Phase 11 error
analysis.

**W3 — `weekend` was excluded on a procedural ground, not an empirical
one.** The predeclared admission condition was never tested. This is not
evidence that weekends do not matter; the question is unanswered in V1.

**No unresolved blockers.**

---

## 14. Explicit confirmations

- **M1, M2 and M3 use the same feature schema** — one 43-column matrix.
- **Exactly 43 features** were produced, in one deterministic order.
- **`No` is not a predictor** — excluded from every X matrix.
- **`year` is not a predictor.**
- **`day` is not a predictor.**
- **No `season` feature was created.**
- **No `weekend` / `weekday` / `day_of_week` feature was created.**
- **No lag or rolling feature was created.**
- **No target-derived feature was created**; the X transformation never
  reads `pm2.5`.
- **No scaling was performed.**
- **No target transformation was performed** — `pm2.5` written unmodified,
  zeros and the 994 maximum retained.
- **Categorical vocabulary came from training only.**
- **Test `pm2.5` was never loaded** by the Phase 5 pipeline.
- **No `y_test` artifact was created.**
- **No model was trained**; no estimator was imported or instantiated.
- **No prediction was generated.**
- **No MAE, RMSE or R² was calculated.**
- **No upstream dataset was modified.**
- **No dependency changed.**
- **No git write operation was performed** — no `add`, `commit`, `push`,
  `reset`, `rebase`, `merge`, `cherry-pick`, `tag`, or branch
  creation/deletion. Every git command was read-only inspection.
- **Git history was not rewritten.**

---

## 15. Status

Protocol Phase 5 is **complete**. All legitimate changes are left
**unstaged**.

The next phase is Protocol Phase 6 — the M0 naive baseline. Nothing about it
has been implemented.

Awaiting instruction.
