# AirSense V2 — Windowing Contract

**Protocol Phase 2. Frozen before any model result exists.**

Machine-readable form: [`configs/windowing.json`](../configs/windowing.json).
Implementation: `src/data/windowing.py`, `src/data/preprocessing.py`,
`src/data/target_history.py`.

This is the canonical definition of a V2 forecasting sample. Every model, every
baseline and every information regime consumes exactly this.

---

## 1. Sample key

A sample is the triple

    (station, target_timestamp τ, horizon h)

carried by a deterministic identifier:

    <partition>|<station>|<YYYYMMDDHH>|h<horizon>

for example `train|Dongsi|2014090712|h6`. The identifier is a pure function of
its parts — the same sample always receives the same id, across rebuilds and
machines. No random identifier is ever issued.

## 2. Geometry

| Quantity | Definition |
|---|---|
| Horizon `h` | one of 1, 6, 12, 24 hours |
| Target timestamp `τ` | the hour being forecast |
| Forecast origin `t` | `τ − h` |
| Context length | **48 hours** (primary) |
| Context start | `t − 47 h` |
| Context end | `t` |
| Context timestamps | `t−47, t−46, …, t−1, t` — exactly 48, spaced one hour |

**Hard invariants**, asserted on every window built:

```
len(timestamps) == 48
timestamps are hourly-contiguous
max(timestamps) == context_end == forecast_origin == τ − h
context_start    == forecast_origin − 47 h
forecast_origin  <  τ
```

No observed variable later than `t` may enter a sample, ever.

### Worked examples — target 2016-06-10 12:00, station Dongsi

| `h` | Forecast origin `t` | Context window | Target |
|---:|---|---|---|
| 1 | 2016-06-10 11:00 | 2016-06-08 12:00 → 2016-06-10 11:00 | 2016-06-10 12:00 |
| 6 | 2016-06-10 06:00 | 2016-06-08 07:00 → 2016-06-10 06:00 | 2016-06-10 12:00 |
| 12 | 2016-06-10 00:00 | 2016-06-08 01:00 → 2016-06-10 00:00 | 2016-06-10 12:00 |
| 24 | 2016-06-09 12:00 | 2016-06-07 13:00 → 2016-06-09 12:00 | 2016-06-10 12:00 |

The same target appears once per horizon with four different information sets.
At `h = 24` the model is blind to the entire day preceding the target.

## 3. Context length

**Primary: 48 hours.** Frozen from Phase-1 training evidence, before any
validation performance exists: PM2.5 autocorrelation runs 0.965 (1 h), 0.752
(6 h), 0.567 (12 h), 0.368 (24 h), 0.098 (48 h), −0.004 (72 h). Forty-eight
hours spans essentially the useful empirical memory while avoiding dead
history.

**Predeclared robustness contexts: 24 h and 72 h.** These may later be run as a
declared context-length robustness experiment. They **may not replace 48 h as
the primary context because they score better on validation.** 168 h is
excluded from the primary study; admitting it would require an explicit
protocol amendment.

## 4. Partition rule

Partition membership follows the **target timestamp** `τ` — never the origin,
never the context start.

| Partition | Target timestamps |
|---|---|
| Train | 2013-03-01 00:00 → 2015-02-28 23:00 |
| Validation | 2015-03-01 00:00 → 2016-02-29 23:00 |
| Locked test | 2016-03-01 00:00 → 2017-02-28 23:00 |

## 5. Boundary rule

Near the start of validation or test the context legitimately reaches into the
**preceding** partition, because those timestamps precede the forecast origin.
**Such samples are kept.**

Worked case, the first eligible test sample at `h = 24`:

```
target          2016-03-01 00:00   (test)
origin          2016-02-29 00:00   (validation)
context         2016-02-27 01:00 → 2016-02-29 00:00   (validation)
```

This is a *test* sample because `τ` is in test. Its inputs are historical
observations that had genuinely occurred before the origin — rolling-origin
history, not leakage. Discarding it would throw away a legitimate forecast a
real operator could make.

What keeps this safe: every learned constant — scaler, median fallback,
vocabulary, and later model weights — is fitted on the training partition only.

## 6. Target-label eligibility

A sample is eligible when **all** hold:

1. `τ` lies inside the assigned partition;
2. **PM2.5 at `τ` is observed** — labels are never imputed, filled or
   interpolated;
3. the origin `t = τ − h` exists in the dataset timeline;
4. the full 48-hour context lies inside the dataset timeline
   (`τ − h − 47 ≥ 2013-03-01 00:00`).

Only these structural rules exclude a sample. **Input-variable missingness
never changes eligibility** — the causal policy handles it — so no model can
look better by silently skipping difficult rows.

## 7. Causal fill

For every numeric input, at each hour, in chronological order:

| Condition | Value used | `fill_source` |
|---|---|---|
| Observed at this hour | the observation | `0` |
| Last observation ≤ 6 h ago | that observation, carried forward | `1` |
| Otherwise | station-specific **training** median | `2` |
| PM2.5 outside training | not processed in Phase 2 | `−1` (deferred) |

No backward fill. No interpolation. No centred statistic. Nothing later than
the current hour is consulted. Forward-fill state is **chronological and not
reset at partition boundaries**: the first validation hour may inherit an
observation from the last training hours, because that observation is in the
past. Learned fallback constants remain training-fitted regardless.

## 8. Mask semantics

For every imputable variable a binary channel records the **raw** state:

```
1 = originally observed at that exact timestamp
0 = missing in the raw source
```

Masks are computed **before** any fill, so an imputed value can never
masquerade as observed.

## 9. Gap-age semantics

For PM2.5, PM10, SO2, NO2, CO and O3, `time_since_last_observed_hours` at each
hour, capped at **168**:

| Situation | Value |
|---|---|
| Observed at this hour | `0` |
| Last observed `k` hours ago | `min(k, 168)` |
| Never observed before this hour | `168` (the cap) |

Never negative, never above the cap. Gap age is derived from presence alone, so
it is available for every partition without touching a sealed value.

## 10. Calendar semantics

Deterministic and knowable at the origin, therefore legal for both the origin
and the target time:

- origin: hour sin/cos, day-of-year sin/cos, month sin/cos
- target time: hour sin/cos, day-of-year sin/cos, month sin/cos

Raw integer hour, day-of-year and month are kept alongside for diagnostics and
for the climatology baseline lookup. **`day_of_week` and `weekend` are not
primary features** — training autocorrelation at 168 h was ≈ 0.008. They may
not be added later merely because a model performs poorly.

Calendar channels are station-independent and are therefore stored once, not
twelve times.

## 11. Target quarantine

PM2.5 **values** are materialised for training timestamps only. For validation
and test the canonical layer holds presence, masks and gap age and nothing
else:

- `pm25_scaled_train_only.npy` is NaN beyond 2015-02-28 23:00;
- `target_series/<station>_train_pm25.npy` is NaN beyond the same hour;
- the sample indices for validation and test carry `target_observed` only —
  no value, no bin, no severity label, no quantile.

Numeric validation and test target history reaches models later through
`CausalTargetHistory`, which refuses any timestamp after the requested origin
and refuses every value beyond its explicit unseal ceiling. In Phase 2 that
ceiling is the end of training, so sealed values are never even read off disk.

## 12. Regime channel sets

| Regime | Channels |
|---|---|
| **R0** | meteorology (TEMP, PRES, DEWP, RAIN, WSPM), `wd_code`, `rain_occurred`, calendar, masks. **No pollutant history.** |
| **R1** | R0 + target-station PM2.5 history + its mask + its gap age |
| **R2** | R1 + PM10, SO2, NO2, CO, O3 history + their masks + their gap ages |
| **R3** | R2 + the same synchronised history from the other 11 stations |

R3's spatial tensor encoding is deliberately **not** fixed here; it belongs to
the spatiotemporal phase. The regime determines which channels are exposed —
never which samples exist.

## 13. Common sample universe

One canonical universe, keyed by `(partition, station, target_timestamp,
horizon)`, shared by every model, every baseline and every regime. Its counts
and file hashes are frozen in
[`artifacts/sample_universe_manifest.json`](../artifacts/sample_universe_manifest.json)
before Experiment 001.

**Fairness invariant.** Every primary model must produce a prediction for every
sample in the canonical validation and test universe, unless a documented
catastrophic runtime failure occurs. No model may silently drop missing-heavy
samples, severe samples, particular stations or particular horizons. If an
architecture cannot consume the canonical representation, the *model wrapper*
is adapted — never the evaluation universe.

## 14. Storage architecture

Windows are **generated on demand**, never stored. Each station's series is
stored once (35,064 hours × compact channels); samples are stored as metadata
rows; the 48-hour window is assembled at access time. Materialising
`samples × context × features` would duplicate the same history millions of
times across four horizons and four regimes.

There are no model-specific dataset forks. One canonical layer; model adapters
consume it.
