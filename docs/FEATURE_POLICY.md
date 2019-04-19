# Feature Policy — AirSense V1

Declared in Phase 0, before any feature was built and before any model was
fitted. It governs what may become a model input, what may not, and why.

**Nothing in this document has been implemented.** No feature has been
engineered, encoded or written to `data/processed/`.

---

## 1. Classification of candidate features

### Class A — raw meteorological features

Measured atmospheric conditions, used as distributed.

| Feature | Description | Units |
|---|---|---|
| `DEWP` | Dew point | °C |
| `TEMP` | Temperature | °C |
| `PRES` | Pressure | hPa |
| `Iws` | Cumulated wind speed | m/s |
| `Is` | Cumulated hours of snow | hours |
| `Ir` | Cumulated hours of rain | hours |

**Caution on `Iws`, `Is`, `Ir`.** These are *cumulative* within a spell and
reset between spells — they are not instantaneous readings. Their
distributions are strongly skewed and their meaning is path-dependent.
Phase 3 must examine them before Phase 5 decides whether to use them raw,
transform them, or derive an instantaneous companion. They are not to be
treated as ordinary continuous measurements without that check.

### Class B — calendar / time features

Present in the raw file as integers.

| Feature | Description | Range |
|---|---|---|
| `year` | Year of observation | 2010–2014 |
| `month` | Month | 1–12 |
| `day` | Day of month | 1–31 |
| `hour` | Hour of day | 0–23 |

**Caution on `year`.** Under the chronological split, training years and
test years are disjoint by construction. A model given `year` as a numeric
input can only extrapolate it into a range it never saw, which is a known
way to damage a tree-based model at test time. Whether `year` is admitted as
a predictor is a Phase 5 decision, and is decided on reasoning about the
split — **not** by comparing test scores.

**Caution on `month` and `hour`.** Both are cyclical: month 12 is adjacent
to month 1, hour 23 to hour 0. Integer encoding does not represent that.
Tree models handle it acceptably via splits; linear models do not. The
options (leave integer, one-hot, or cyclical sine/cosine encoding — all
standard and available well before 2019) are weighed in Phase 5.

### Class C — categorical wind direction

| Feature | Description |
|---|---|
| `cbwd` | Combined wind direction, categorical |

Requires encoding before use in scikit-learn 0.20.0. Categories are
enumerated from the data in Phase 3 — not assumed.

**2019 API constraint.** `sklearn.preprocessing.OneHotEncoder` in 0.20
does not accept string categories in the way later versions do, and
`ColumnTransformer` had only just arrived (0.20). The period-appropriate
route is `pandas.get_dummies` on the categorical column. Encoding categories
must be fixed from the **training** period and applied unchanged to the test
period, so that a direction absent from training cannot silently create a
new column at test time.

### Class D — historically reasonable engineered features

Simple derivations a 2019 practitioner would plausibly have built. All are
derived **only** from Class A and B values of the same observation.

| Candidate | Derivation | Rationale |
|---|---|---|
| `season` | from `month` | Compact seasonal signal; the winter heating season is a documented driver in the dataset's introductory paper |
| hour-of-day grouping | from `hour` | Coarse diurnal pattern (e.g. night / morning / afternoon / evening) |
| `month` as categorical | from `month` | Lets a linear model represent non-monotone seasonality |
| weekend indicator | from `year`/`month`/`day` | Traffic and industrial activity differ at weekends — **admitted only if Phase 3 shows an actual weekday/weekend difference in PM2.5.** Not included on plausibility alone. |

Each candidate must earn admission in Phase 5 with a stated reason. None is
included by default.

---

## 2. Prohibited: future information

**No feature may use information unavailable at the moment being predicted.**

Specifically forbidden:

- values from later hours, days, months or years;
- rolling, expanding or aggregate statistics computed over the full dataset
  (a full-series mean encodes the test period into the training features);
- any normalisation, scaling, encoding or imputation parameter fitted on the
  full dataset. **Every such parameter is fitted on the training period
  only and then applied to the test period.**

This last point is where leakage most often enters a project quietly: a
scaler fitted before the split looks harmless and is not.

---

## 3. Prohibited: target-derived features

**No feature may be derived from the `pm2.5` value being predicted.**

Forbidden: the target itself in any transformed form; target encoding of a
categorical using target means; any group statistic of `pm2.5`; any feature
whose computation reads the target of the row being predicted.

Such a feature would produce excellent metrics that mean nothing.

---

## 4. Prohibited in this phase: lag features

**No lag feature is created in this phase.** Not `pm2.5` at t−1, not a
rolling mean of past PM2.5, not any autoregressive term.

---

## 5. If lagged PM2.5 is considered later

Lagged PM2.5 is not forbidden in principle — it is standard and entirely
period-appropriate. But adding it **changes the question being asked**.

| | V1 as declared | A lagged variant |
|---|---|---|
| Question | Can weather and time explain PM2.5? | Can recent PM2.5 plus weather predict the next PM2.5? |
| Task | Concurrent-hour estimation | Forecasting |
| Inputs | Meteorology + calendar | The above, plus prior target values |
| Expected difficulty | Harder | Much easier — hourly PM2.5 is strongly autocorrelated |

Therefore, if introduced later, lagged PM2.5 must be:

1. run as a **separate, explicitly labelled forecasting experiment**;
2. **never** merged into the M0–M3 comparison table;
3. given its own baseline — for a forecasting task the honest floor is
   persistence (predict the previous observed value), not the training mean;
4. constructed strictly backward in time, respecting the chronological
   split at the boundary so that no test-period row draws a lag value from
   across it.

A forecasting model's metrics are not comparable to V1's. Presenting them
side by side would misrepresent both.

---

## 6. Summary

| Class | Content | Status |
|---|---|---|
| A | Raw meteorological | Candidate; `Iws`/`Is`/`Ir` need Phase 3 review |
| B | Calendar / time | Candidate; `year` and cyclical encoding pending Phase 5 |
| C | Categorical wind direction | Candidate; encoding fitted on training period only |
| D | Simple engineered | Candidate; each requires a stated Phase 5 justification |
| — | Future information | **Prohibited** |
| — | Target-derived | **Prohibited** |
| — | Lag features | **Not in this phase**; separate experiment if ever used |

**Implementation status: none.** `src/features/` contains only an empty
`__init__.py`.
