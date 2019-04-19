# V1 Research Protocol — AirSense

This protocol freezes the intended sequence of work **before any result is
known**. It was written during Phase 0, when no model had been trained and no
metric computed.

Its purpose is to constrain the author. A protocol written after seeing
results can always be made to look like a plan that succeeded; one written
before results cannot.

---

## 1. Pre-registration statement

**Model choices and primary metrics are declared before final-test
performance is observed, in order to reduce result-driven selection.**

Fixed in advance, and not revisable in response to outcomes:

- the four models compared — M0 baseline, M1 Linear Regression,
  M2 Decision Tree, M3 Random Forest;
- the primary metrics — **MAE**, **RMSE**, **R²**;
- the decision metric — **MAE on the chronological held-out test set**;
- that the chronological split is the primary evaluation, and any random
  split is secondary;
- that every model is reported, including ones that perform poorly.

Concretely this rules out four common failure modes: adding a fifth model
after seeing that M0–M3 disappointed and presenting it as the plan; swapping
the headline metric to whichever flatters the preferred model; quietly
dropping a weak model from the results table; and moving the split boundary
until the numbers improve.

Deviations are permitted where they are honest, but must be recorded in §7
with a date and a reason.

---

## 2. Phase sequence

| Phase | Name | Purpose | Status |
|---|---|---|---|
| 1 | Environment reconstruction | Define and verify the 2019 reference environment; record host divergence | **Complete (with blocker)** |
| 2 | Dataset provenance and audit | Acquire the original dataset; record identity, dimensions, schema, missingness, digest | **Complete** |
| 3 | Exploratory data analysis | Understand distributions, temporal patterns, correlations, missingness structure | Not started |
| 4 | Data cleaning and preprocessing | Decide and apply handling of missing `pm2.5`; write to `data/processed/` | Not started |
| 5 | Feature preparation | Apply `FEATURE_POLICY.md`; encode `cbwd`; decide engineered features | Not started |
| 6 | Baseline regression (M0) | Establish the naive error floor | Not started |
| 7 | Linear regression (M1) | Fit and evaluate | Not started |
| 8 | Decision tree regression (M2) | Fit and evaluate | Not started |
| 9 | Random forest regression (M3) | Fit and evaluate | Not started |
| 10 | Held-out evaluation | Apply M0–M3 to the chronological test set; produce the comparison | Not started |
| 11 | Error analysis | Where and when models fail; residual structure; severe-episode behaviour | Not started |
| 12 | Final V1 report | Consolidate findings and limitations | Not started |

Phases 1 and 2 are complete as recorded in
[`PHASE_00_FOUNDATION_RECORD.md`](PHASE_00_FOUNDATION_RECORD.md). Phase 1
carries an unresolved environment blocker (see
[`HISTORICAL_COMPATIBILITY.md`](HISTORICAL_COMPATIBILITY.md) §4): the pinned
2019 stack cannot be installed on the current host, so Phases 3 onward
cannot execute until a conforming environment exists.

### Phase notes

**Phase 3 — EDA.** Descriptive statistics; missing-data analysis across
years, months and hours; PM2.5 distribution and skew; temporal patterns
(hour, day, month, year); feature correlations and collinearity; seasonal
patterns including the winter heating season; meteorological relationships.
Figures to `figures/`. **No modelling.**

**Phase 4 — Cleaning.** The handling rule for the 2,067 missing `pm2.5`
values is chosen here, from what Phase 3 showed, and recorded with its
reason. Since missingness affects only the target, the realistic options are
to exclude those rows from training and evaluation, or to impute — and
imputing a target is a strong choice requiring explicit justification.
`data/raw/` is read-only; all output goes to `data/processed/`.

**Phase 5 — Features.** Governed entirely by
[`FEATURE_POLICY.md`](FEATURE_POLICY.md). Every encoding, scaling or
imputation parameter is fitted on the **training period only**.

**Phases 6–9 — Model fitting.** Each model is fitted on the training period.
Any hyperparameter selection uses **only** training-period data, via a
time-aware validation scheme (e.g. a chronological validation slice, or
scikit-learn 0.20's `TimeSeriesSplit`). The test set is not consulted.

**Phase 10 — Held-out evaluation.** The test set is used **once**, at the
end, for all four models. It is not used to choose models, features,
hyperparameters or the split boundary. If a result prompts a change to an
earlier phase, the run becomes exploratory and is labelled as such — it does
not silently become the reported result.

**Phase 11 — Error analysis.** Residuals by season, hour and pollution
level; behaviour during severe episodes; whether errors are structured or
noise-like; where the models are unreliable.

**Phase 12 — Report.** Findings, the M0–M3 comparison, limitations, and an
explicit statement of what the V1 method cannot answer.

---

## 3. Temporal leakage rule

**This section governs all evaluation in AirSense V1.**

### 3.1 The problem

The observations form a chronological environmental time series: 43,824
consecutive hourly records from 2010-01-01 to 2014-12-31.

**A random train/test split can make forecasting evaluation artificially
optimistic, because future observations may be used indirectly when learning
patterns for earlier observations.**

With hourly environmental data the mechanism is concrete. Randomly assigning
rows sends 14:00 on a given day to training and 15:00 of that same day to
test. Consecutive hours share a weather system and a pollution episode, so
the test row is nearly a copy of a training row. The model is scored largely
on interpolation between neighbours it has already seen — not on
generalisation to conditions it has not. The resulting metrics are
optimistic in a way that does not survive contact with genuinely unseen
future data.

Leakage also enters through preprocessing: any scaler, encoder or imputer
fitted on the full dataset before splitting has already read the test period.
See [`FEATURE_POLICY.md`](FEATURE_POLICY.md) §2.

### 3.2 The rule

**The main scientific evaluation uses a chronological split.**

```
   earlier period  ->  training
   later period    ->  testing
```

Training and test periods are contiguous and non-overlapping, and every
training observation precedes every test observation.

**The exact boundaries are determined only after inspecting the dataset**
(Phase 3), and are set from the calendar — a whole number of years or a
clean seasonal boundary — so that both periods cover complete seasonal
cycles. Given the strong seasonality of PM2.5, a boundary mid-season would
put a partial winter in one period and its remainder in the other and
distort the comparison.

**The boundary is never chosen by comparing model performance across
candidate boundaries.** Doing so would tune the split on the test set, which
is the leakage this rule exists to prevent.

### 3.3 Status of the random split

A random split **may** be included later as a **secondary educational
comparison**. Such workflows were common in introductory 2019 Data Science
training, and showing the gap between the two evaluation designs is itself a
useful result — arguably the most instructive finding V1 can offer.

**But it must not silently become the strongest evidence.** Therefore:

1. The chronological split is the **primary** evaluation. Headline figures,
   the model comparison, and every conclusion come from it.
2. Random-split results, if reported, are **explicitly labelled** as an
   optimistically biased secondary comparison.
3. The two are **never** presented in a single undifferentiated table, and
   the better random-split numbers are never quoted as the project's
   performance.
4. If random-split numbers are materially better — which is expected — the
   gap is reported and discussed as evidence of the leakage effect, not
   presented as achievement.

**This distinction is important.** A reader who takes a random-split score
as this project's predictive performance has been misled, and the protocol
is written to make that impossible.

---

## 4. Immutability of raw data

`data/raw/` is never modified. All derived data is written to
`data/processed/`. The raw SHA-256 recorded in
[`DATASET_AUDIT.md`](DATASET_AUDIT.md) must remain verifiable at every phase.

---

## 5. Historical constraint

Every phase is bound by
[`HISTORICAL_COMPATIBILITY.md`](HISTORICAL_COMPATIBILITY.md): only
technologies, APIs and versions available on or before **2019-04-26**. If a
task seems to need a post-cutoff tool, the task is solved with 2019 tools or
narrowed — the cutoff is not relaxed for convenience.

---

## 6. Reporting standards

- No result is reported without stating the split that produced it.
- Every model in §1 is reported, including poor performers.
- Metrics are given in the units of the target (µg/m³) where applicable.
- Figures carry the phase and split that generated them.
- Limitations are stated in Phase 12, including single-location data, a
  fixed 2010–2014 window, and the deliberate exclusion of lagged PM2.5.
- No placeholder, illustrative or expected number is ever written into a
  results file. A metric appears only once it has actually been computed.

---

## 7. Protocol amendments

Changes to this protocol after Phase 0 are recorded here with date, what
changed, why, and the effect on comparability.

**No amendments.** The protocol stands as originally written.
