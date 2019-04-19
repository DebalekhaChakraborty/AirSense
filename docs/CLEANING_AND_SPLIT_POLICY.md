# AirSense V1 Cleaning and Split Policy

**Research-protocol mapping: Protocol Phase 4 — Data Cleaning and
Preprocessing.**

This document states the supervised-target cleaning rule and freezes the
chronological evaluation design.

> **Both were fixed before any model was fitted and before any predictive
> metric existed.** No MAE, RMSE or R² has been computed anywhere in this
> repository. The boundaries below were chosen from the calendar, not from
> performance.

---

## Purpose

Phase 3 characterised the dataset. This phase converts that characterisation
into two decisions that everything downstream depends on:

1. **Which rows are eligible for supervised modelling**, and what happens to
   the ones that are not.
2. **Where the chronological boundaries sit**, fixed now so they cannot
   later be moved in response to results.

The phase produces derived datasets. It does not produce features, and it
does not produce a model.

---

## Evidence from Phase 3

The decisions below rest on findings already recorded in
[`EDA_ANALYSIS.md`](EDA_ANALYSIS.md):

- Missing values occur in **`pm2.5` only** — 2,067 of 43,824 (4.7166%). All
  eleven predictors and the index column are complete.
- **Missing-target frequency is strongly year-dependent**: 7.64% (2010),
  8.31% (2011), 5.57% (2012), 0.94% (2013), 1.13% (2014). 2010–2012 hold
  1,886 of the 2,067 missing targets.
- The **hourly timeline is complete**: 43,824 of 43,824 expected timestamps,
  no gap, no duplicate, already in chronological order.
- **Zero-valued PM2.5 observations were not proven erroneous.**
- **High-concentration PM2.5 observations were not proven erroneous.**
- There are **no duplicate observations**.

No additional cleaning has been introduced beyond what these findings
support.

---

## Missing-target decision

**A row is eligible for supervised modelling if and only if `pm2.5` is
non-missing.**

Rows failing that test are **excluded from the supervised derived datasets
only**. They are not deleted from anything. Specifically:

- `data/raw/` is untouched and remains the authoritative 43,824-row
  timeline, byte-identical and read-only.
- All 2,067 excluded rows are recorded permanently in
  [`artifacts/target_missing_exclusions.csv`](../artifacts/target_missing_exclusions.csv),
  with identifier, calendar components, reconstructed timestamp, assigned
  partition and reason.
- The exclusion reason is exactly `missing_pm25_target` for every row.

No invented target value appears anywhere in that manifest or in any derived
dataset.

---

## Why target imputation was rejected

For these 2,067 hours the **predictors exist but the ground truth does
not**. Imputing the target would manufacture the very quantity the model is
supposed to learn.

The concrete failure modes:

- **Mean or median fill** would insert a constant the model can partly
  recover from the calendar, teaching it a relationship that exists only
  because it was inserted.
- **Interpolation, forward-fill or backward-fill** would borrow information
  from neighbouring hours. Because hourly PM2.5 is strongly autocorrelated,
  such a value looks plausible and is therefore worse than an obvious one:
  it would inflate apparent performance while adding no real information,
  and in the test period it would let a neighbouring hour's observed value
  leak into a scored prediction.
- **Any imputation at all** would blur the distinction between *predicting*
  a value and *reconstructing* one, which is the distinction the whole
  project rests on.

Imputing a **predictor** would be a different and more defensible question.
It does not arise here: every predictor is complete.

**The 2,067 rows retain their missing targets in the raw data. Nothing was
filled, interpolated, forward-filled, backward-filled, averaged or
synthesised.**

---

## Treatment of zero PM2.5 values

**Retained unchanged.** Two observations with `pm2.5 == 0` are present in
the supervised data.

Phase 3 established no evidence that they are erroneous. Whether they are
genuine near-zero readings or a reporting floor cannot be determined from
this file, and "the value is unusual" is not evidence of error. Removing
them would be a silent modelling assumption disguised as cleaning. They are
flagged for Phase 11 error analysis, where the question can be asked
properly.

---

## Treatment of high-concentration values

**Retained unchanged, including the maximum of 994 µg/m³.**

Nothing was removed, clipped, winsorized, capped or reclassified as an
outlier. These are described as **high-concentration observations**, not
outliers: severe pollution episodes in Beijing over 2010–2014 are
well-documented, and discarding them would remove exactly the events a
pollution model most needs to represent. A model that predicts ordinary
hours well and severe episodes badly is a model whose weakness must be
*visible*, not one whose hard cases have been deleted.

---

## Calendar split

Membership is a property of **when** an observation was taken.

| Partition | Start | End | Role |
|---|---|---|---|
| **Development train** | 2010-01-01 00:00 | 2012-12-31 23:00 | fitting and development |
| **Validation** | 2013-01-01 00:00 | 2013-12-31 23:00 | time-aware model/feature/hyperparameter selection |
| **Locked final test** | 2014-01-01 00:00 | 2014-12-31 23:00 | used once, at Protocol Phase 10 |

Rationale, recorded before any result existed:

1. Every partition is a **whole number of complete calendar years**.
2. Every partition therefore contains **complete seasonal cycles** — no
   partial winter split across a boundary, which matters given the seasonal
   structure Phase 3 found.
3. **No future observation precedes a training observation.**
4. The boundary is **simple and predetermined** — calendar years, not a
   tuned cut point.
5. **Model performance was not consulted**; none exists.
6. The final held-out year contains **genuinely later** observations.
7. 2013 gives a **full-year** time-aware validation period before 2014.

This satisfies the frozen rule in
[`V1_RESEARCH_PROTOCOL.md`](V1_RESEARCH_PROTOCOL.md) §3.2, which requires
calendar boundaries covering complete seasonal cycles and forbids choosing a
boundary by comparing model performance. The 2013 validation year is the
"chronological validation slice" that protocol §2 already anticipated for
Phases 6–9.

> **This boundary must not later move because another boundary gives better
> metrics.** Doing so would tune the split on the evaluation data, which is
> the leakage the temporal rule exists to prevent.

### Split assignment happens *before* target exclusion

The ordering is mandatory and is enforced in code:

```
raw 43,824-row chronological dataset
        |
        v  assign partition by calendar timestamp   <-- all 43,824 rows
        |
        v  record target availability within each partition
        |
        v  exclude missing-target rows separately inside each partition
        |
        v  write supervised derived datasets
```

Dropping missing targets first and then slicing by row position would let
the year-dependent missing rate shift the effective boundaries away from the
calendar dates frozen above. Calendar time — never cleaned row number —
defines membership.

---

## Development-training period

**2010-01-01 00:00 to 2012-12-31 23:00.** 26,304 raw rows, 24,418 with an
observed target. Three complete years.

This is the only period permitted to inform development decisions:
transformations, encodings, scalers and any target-based feature
justification must be learned here.

---

## Validation period

**2013-01-01 00:00 to 2013-12-31 23:00.** 8,760 raw rows, 8,678 with an
observed target.

2013 is the only explicit development validation year. The workflow it
supports, declared now and not implemented in this phase:

- **Phases 6–9:** fit on 2010–2012, validate on 2013 where a choice must be
  made.
- **After all choices for a model are frozen:** 2010–2013 may be used as the
  final development fit.
- **Phase 10:** 2014, used once.

**The final refit has not been performed, and train and validation have not
been concatenated.** This phase freezes the policy only.

---

## Locked test period

**2014-01-01 00:00 to 2014-12-31 23:00.** 8,760 raw rows, 8,661 with an
observed target. File:
`data/processed/airsense_test_2014.csv`, SHA-256
`9f273de4ae9243c269d6f211229e5d017341e50ee8e7093add3a04dedd903702`.

---

## Missingness asymmetry by partition

| Partition | Raw rows | Missing targets | Supervised rows | Missing rate |
|---|---:|---:|---:|---:|
| Development train (2010–2012) | 26,304 | 1,886 | 24,418 | **7.1700%** |
| Validation (2013) | 8,760 | 82 | 8,678 | **0.9361%** |
| Test (2014) | 8,760 | 99 | 8,661 | **1.1301%** |
| **Overall** | **43,824** | **2,067** | **41,757** | **4.7166%** |

Ratios:

- development train / validation = **7.66×**
- development train / test = **6.34×**

### What this is, and what it is not

This is a **target-availability / label-missingness shift**. It is
explicitly **not** predictor covariate shift: every predictor is complete in
every partition, so the input distribution is fully observed throughout.
What differs across partitions is only how often the *label* is present.

### Implication

**The earlier supervised training sample is more selectively observed than
the later validation and test periods.** Roughly one training hour in
fourteen is absent from the supervised training set, against about one in a
hundred for validation and test.

Consequences to carry forward:

- The training period is not a uniform sample of 2010–2012; it is the subset
  of hours whose target happened to be recorded. If that recording process
  was related to conditions — which is unknown — the training sample is not
  a random subset of the period.
- Model performance measured on 2013 or 2014 is measured on a period whose
  labels are far more completely observed than the period the model learned
  from. This is a genuine and asymmetric property of the evaluation, and it
  should be stated when results are eventually reported.
- Per-year training counts differ for the same reason and should not be read
  as changes in data collection volume: the raw hourly coverage is complete
  and identical in every year.

**No causal explanation is asserted.** Phase 3 found the pattern; nothing in
this dataset establishes why the earlier years have more absent targets, and
this document does not speculate.

---

## Data reconciliation

| Quantity | Rows |
|---|---:|
| Raw total | 43,824 |
| Excluded (target missing) | 2,067 |
| Supervised total | 41,757 |
| — development train | 24,418 |
| — validation | 8,678 |
| — test | 8,661 |

Verified in code on every run:

- supervised + excluded = 43,824 ✓
- train + validation + test = 41,757 = supervised total ✓
- partition overlap = 0 rows ✓
- partitions exactly cover the supervised dataset ✓
- every retained row matches its raw source verbatim ✓
- every excluded row genuinely has a missing raw target ✓
- chronological ordering preserved in every file ✓
- no retained row carries a missing target ✓

Retained targets span 0 to 994 µg/m³, with 2 zero-valued observations kept.

*Tables:* `results/preprocessing/partition_summary.csv`,
`results/preprocessing/data_reconciliation.csv`
*Manifest:* [`artifacts/split_manifest.json`](../artifacts/split_manifest.json)

---

## Test-set quarantine

**From the completion of this phase until Protocol Phase 10, the 2014 target
is quarantined.**

It must **not** be used for:

feature selection · encoding decisions · scaling decisions · hyperparameter
selection · model comparison · threshold selection · transformation choice ·
debugging model quality · choosing the split · deciding which model to keep

**Permitted** before Phase 10:

- confirming the test file's identity, hash, row count and schema;
- confirming the target is present and non-missing for retained rows;
- confirming temporal boundaries;
- applying transformations learned **exclusively** from development data.

**Not permitted:** calculating predictive performance on 2014. No MAE, RMSE,
R² or any other model score may be computed on the test partition before
Phase 10.

---

## Leakage controls

1. **Chronological membership.** Every training observation precedes every
   validation observation, which precedes every test observation. No
   shuffling anywhere.
2. **Split before exclusion.** Enforced in code, so the year-dependent
   missing rate cannot move the boundaries.
3. **Fit-on-development-only.** Every encoding, scaling or imputation
   parameter must be fitted on the permitted development data and then
   applied unchanged. A scaler fitted on the full dataset has already read
   the test period.
4. **The EDA boundary.** The Phase 3 exploratory analysis was performed on
   the full dataset, deliberately and in the order the original protocol
   declared. That stands and is not retroactively changed. **But from this
   point forward it is descriptive characterisation only — it is not
   permission to optimise features against the 2014 target.** Any feature
   decision requiring target-based evidence must use only the development
   data the phase permits. Concretely: if Phase 5 weighs the optional
   weekend indicator, the weekday/weekend target comparison that admits or
   rejects it must be computed on 2010–2012 development-train data alone.
5. **No lag features.** Unchanged from
   [`FEATURE_POLICY.md`](FEATURE_POLICY.md) §4–5; a lagged variant would be
   a separate forecasting experiment.
6. **The test set is used once**, at Phase 10.

---

## What remains undecided for Phase 5

Nothing below has been decided, and `FEATURE_POLICY.md` has **not** been
modified to pretend otherwise:

- whether `year` is a model input;
- how `month` is encoded;
- how `hour` is encoded;
- whether cyclical sine/cosine features are used;
- whether `season` is admitted;
- whether a weekend indicator is admitted;
- how `cbwd` is encoded;
- whether `Iws` is transformed;
- whether `Is` or `Ir` are transformed;
- whether scaling is used;
- which features are used by which model;
- whether `No` is excluded from the feature matrix.

**`No` is retained in every derived dataset for row traceability.** It is
**not** thereby a legitimate model predictor — it is a distributor's running
index, perfectly collinear with time. Its exclusion from the feature matrix
is a Phase 5 implementation task.

No feature matrix exists. No model exists.
