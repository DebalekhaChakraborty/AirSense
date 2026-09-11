# AirSense V2 Preprocessing Policy

**Protocol Phase 1. Frozen before any V2 model exists.**

Machine-readable form: [`configs/preprocessing.json`](../configs/preprocessing.json).
Evidence: [`TRAINING_EDA.md`](TRAINING_EDA.md).

Every rule below distinguishes two things that are routinely confused:

> **FITTED ON** — the partition whose data may be used to *learn* a constant,
> vocabulary, statistic or parameter. This is always **training only**.
>
> **APPLIED TO** — the partitions where the already-fitted rule is *executed*.
> This is train, validation and test alike.

Nothing learned from validation or test may ever enter a rule.

---

## Scope

Covers the transformation from raw hourly station CSVs to model-ready causal
sequences: target-label eligibility, missing-value handling, masks, scaling,
calendar and station encoding, and sequence eligibility. It does **not** cover
model architecture or hyperparameters, which are out of Phase-1 scope.

## Leakage principles

Binding. A violation of any of these invalidates an experiment.

1. For a forecast at origin `t`, no input may derive from a timestamp later
   than `t` — including via imputation, smoothing, scaling or encoding.
2. **Backward fill is prohibited.** No value may be filled from the future.
3. **Interpolation is prohibited** where it consults observations after the
   origin. Centred rolling statistics and any bidirectional transform are
   prohibited outright.
4. No statistic may be fitted on the full series before partitioning.
5. No full-dataset or validation/test mean, median, quantile or vocabulary.
6. The target at `t + h` never appears in any input, in any transformed form.
7. Calendar values at `t + h` are permitted: they are deterministic and
   knowable at the origin without observing anything.
8. Observed future weather is not available. A forecast-conditioned study would
   be a separate, explicitly labelled experiment.

## Target-label missingness

**Never impute a target.** A sample is eligible only if PM2.5 at the target
timestamp is observed in the raw data. Missing labels are excluded, never
filled, never interpolated, never carried forward.

- FITTED ON: nothing — this is a rule, not a statistic.
- APPLIED TO: train, validation, test identically.

Exclusion counts are recorded in
`results/eda_training/target_eligibility_by_partition_horizon.csv` by partition
× horizon. Training excludes 4,251 targets for missingness plus 12/72/144/288
at horizons 1/6/12/24 whose origin would precede the dataset start. Validation
and test exclude 2,121 and 2,367 targets respectively — **counts only; no
target value was read for either.**

## Historical PM2.5 missingness (model inputs)

Two-tier, because the gap distribution is bimodal: the median gap is 1 h and
91.6% of gaps are ≤ 6 h, but 23 gaps exceed 24 h and hold 40% of all missing
PM2.5 hours.

| Gap age at origin | Rule |
|---|---|
| ≤ **6 hours** | causal forward fill — carry the most recent observation at or before the origin |
| > 6 hours | do **not** carry; substitute the station's training median and mark the value as not observed |

Always emitted alongside: the observed/missing mask and
`time_since_last_observed`.

**Why 6 hours.** It is the P90 gap length (5 h) rounded to the nearest horizon
boundary, it covers 1,019 of 1,112 training gaps, and it is where the evidence
stops supporting the carry: training autocorrelation is still 0.752 at lag 6
but only 0.567 at 12 h and 0.368 at 24 h. Carrying a day-old value forward as
if current would assert information the data does not contain.

**Long spans are not dropped.** Outages concentrate by station — 343 h at
Aotizhongxin, 208 h at Huairou, against 13 h at Wanliu — so dropping them would
remove data unevenly across stations and seasons and quietly bias the macro
metric.

- FITTED ON: station-specific training medians (`train`).
- APPLIED TO: all partitions, using those training constants.

## Co-pollutant missingness

PM10, SO2, NO2, CO, O3 follow the same two-tier rule with the same 6-hour
ceiling and the same mask + gap-age companions.

CO deserves a warning label: 7.80% missing, longest outage 1,517 h (63 days),
and 60% of its missing hours in runs over 24 h. It will spend long stretches on
fallback. Whether CO earns its place in R2 is a modelling question for the
ablation, not a preprocessing decision.

- FITTED ON: station-specific training medians.
- APPLIED TO: all partitions.

## Meteorological missingness

TEMP, PRES, DEWP, RAIN, WSPM are effectively complete: under 0.09% missing,
longest run 7 h, and **zero** runs over 24 h. The same causal forward fill with
a 6-hour ceiling covers essentially every case; the training-median fallback
will rarely fire. Masks are still emitted, for uniformity and so that the rare
case is visible.

- FITTED ON: station-specific training medians.
- APPLIED TO: all partitions.

## Wind-direction missingness

`wd` is categorical and is **never** interpolated numerically.

- Causal forward fill of the category, ceiling 6 hours.
- Beyond that, an explicit `MISSING` category — a state, not a guess.
- Vocabulary is the 16 compass categories observed in training; a reserved
  `UNKNOWN` slot exists for robustness although a structural check found the
  same 16 categories at every station across the whole dataset period, so it is
  not expected to fire.
- No target-based encoding of any kind.

- FITTED ON: training vocabulary.
- APPLIED TO: all partitions.

## Missingness masks

For every variable subject to imputation, emit a binary channel: **1 = observed
in the raw data, 0 = imputed or absent**. The mask is computed from the raw
value *before* any fill, so the model can always distinguish a measurement from
a carried value.

Masked variables: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM,
wd.

- FITTED ON: nothing — masks are observations about the raw data.
- APPLIED TO: all partitions.

## Time-since-observation features

For PM2.5 and the five co-pollutants, emit `time_since_last_observed` in hours
at the origin, capped at 168 h, with the cap used where a variable has never
been observed within the window.

This exists because gap age spans orders of magnitude — 1 h for most gaps,
1,517 h at worst — and a mask alone cannot distinguish a value that is one hour
stale from one that is sixty days stale.

- FITTED ON: nothing.
- APPLIED TO: all partitions.

## Scaling

**Global training robust scaling**: `(x − median) / IQR`, with the median and
IQR pooled across all 12 stations over the training partition. Values are
frozen in `configs/preprocessing.json` and tabulated in
`results/eda_training/training_scaling_statistics.csv`.

- **Robust, not z-score**, because training PM2.5 has skewness 1.64 and excess
  kurtosis 3.64; a standard-deviation scale is dominated by the tail.
- **Global, not per-station.** Rejected deliberately: station medians range
  45.0 → 68.0 and station P95 221 → 259, and those level differences are the
  signal the macro station-horizon metric exists to compare fairly. Per-station
  centring would erase them, and would make Track-B leave-one-station-out
  transfer ill-defined because an unseen station has no fitted centre.
- **Zero-IQR special case: RAIN.** Its training median, P25 and P75 are all
  0.0, so the robust denominator is zero and the formula is undefined. RAIN
  falls back to `(x − median) / std`. A companion binary "rain occurred"
  feature is a Phase-2 option, not assumed here.
- **Target transform: none.** PM2.5 stays in native µg/m³ so that MAE is
  directly interpretable. A log transform is an explicit Phase-2 decision, not
  a silent default.

- FITTED ON: training partition, pooled.
- APPLIED TO: validation and test using **the training constants only**, never
  refitted.

## Calendar features

Frozen, all deterministic and knowable at the origin:

`origin_hour`, `origin_day_of_year`, `origin_month`, `target_hour`,
`target_day_of_year`, `target_month`, with cyclic sin/cos encodings of hour and
day-of-year.

Target-time calendar features are explicitly permitted — see leakage principle
7.

**Demoted to optional: `day_of_week` and `weekend`.** Training PM2.5
autocorrelation at a 168-hour lag is 0.008, which gives no support for a weekly
cycle. They are retained only as a labelled Phase-2 ablation, not in the frozen
set.

- FITTED ON: nothing — calendar features are deterministic.
- APPLIED TO: all partitions.

## Station identity

A learned embedding or one-hot over the 12 training stations. Identity must
encode station, never partition membership or anything derived from a later
period.

Track-B caveat recorded now: one-hot identity cannot transfer to an unseen
station, so leave-one-station-out generalisation will require an identity-free
or attribute-based representation. The official UCI package ships **no station
coordinates**, so any geometric representation depends on an external
authoritative source acquired in the spatiotemporal phase.

- FITTED ON: the 12 training stations.
- APPLIED TO: all partitions.

## Sequence eligibility

A sample is the triple `(station, target_timestamp τ, horizon h)`. It is
eligible when **all** of the following hold:

1. `τ` lies inside the assigned partition (partition membership follows the
   target timestamp);
2. PM2.5 at `τ` is observed — never imputed;
3. the forecast origin `t = τ − h` exists in the dataset timeline;
4. every input timestamp is `≤ t`;
5. the required context window `[t − context + 1, t]` lies inside the dataset
   timeline;
6. missing inputs inside the window are handled by the causal rules above, and
   by nothing else.

Context-length-dependent eligibility counts are finalised in Phase 2, once the
context set is approved. The counts published now assume only rules 1–3.

## Boundary context

For targets near the start of validation or test, the context window legitimately
extends into the **preceding** partition, because those timestamps precede the
forecast origin. **Such samples are kept, not discarded** — a real forecaster on
the first day of a new period has yesterday's observations, and using them is
rolling-origin history, not leakage.

The constraint that makes this safe: model parameters and every preprocessing
statistic remain fitted on the training partition only. Newly observed *past*
context entering the input window is legitimate; updating anything from it is
not.

## Final-test quarantine

The locked test partition `2016-03-01 00:00 → 2017-02-28 23:00` remains sealed.
Permitted before Phase 9: row counts, timestamps, missingness counts, schema,
station identity, feature availability, integrity hashes. Prohibited: any
target mean, median, quantile, threshold, distribution, correlation, figure or
metric.

Phase 1 enforced this structurally rather than by intention: `load_partition`
in `src/data/eda.py` raises unless `mode="presence"` for any partition other
than training, and presence mode discards numeric values inside the parser.

## Decisions deferred to Phase 2

- Final context length, chosen from the approved candidate set {24, 48, 72} h
  (optional control 168 h) — **never** by validation performance.
- Exact eligible-row mask once context length is fixed.
- Whether PM2.5 inputs are log-transformed.
- Whether CO stays in R2 given its outage burden.
- Whether a binary rain-occurred companion feature is added.
- Whether `day_of_week` / `weekend` are worth the ablation.
- Batch and window-sampling strategy, and any class-balancing of severe hours.
- Track-B station representation and the external coordinate source.
