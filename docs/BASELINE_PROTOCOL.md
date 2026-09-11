# AirSense V2 — Baseline Protocol

**Protocol Phase 2. Definitions frozen before any prediction exists.**

Machine-readable form: [`configs/baselines.json`](../configs/baselines.json).

**No baseline prediction was generated in this phase, and no metric of any kind
was computed.** This document defines the baselines mathematically so that
Phase 3 can run them without a single downstream choice being informed by
observed performance.

All four baselines operate on the canonical sample universe and the canonical
48-hour window defined in [`WINDOWING_CONTRACT.md`](WINDOWING_CONTRACT.md).
Every baseline must produce a prediction for **every** eligible sample.

---

## Why baselines come first

V1 ended with a random forest that beat a constant by 25.8%, which sounds
impressive until you ask what a one-line rule would have scored. V2 fixes that
ordering: the naive rules are defined before the modern ones are built, so no
sophisticated model can be credited for work that persistence already does.

Phase-1 evidence makes this concrete. Training PM2.5 autocorrelation at a
one-hour lag is **0.965**. At `h = 1`, persistence is not a straw man — it is a
genuinely strong forecaster, and beating it is the minimum bar for any claim
about temporal memory.

---

## B0 — causal persistence

> **B0(station, τ, h) = the latest usable PM2.5 observation at the target
> station available at the forecast origin `t = τ − h`.**

Resolution order, applying the frozen causal policy:

1. if PM2.5 is **observed at `t`**, predict that value;
2. else carry the most recent observation within the **6-hour** ceiling;
3. else predict the **station training PM2.5 median**.

Step 3 guarantees a prediction for every eligible sample, so B0 can never
improve its score by declining the hard cases. The median is a training-only
constant.

Implemented later on top of
`CausalTargetHistory.latest_value_at_or_before(origin, 6)`, which cannot return
anything after the origin.

## B1 — daily seasonal naive

> **B1(station, τ, h) = PM2.5 observed at `τ − 24 h`,** used only when that
> observation is causally available at the origin.

Usable only if both hold:

- `τ − 24 h ≤ t` — the seasonal lag is not in the future relative to the
  origin;
- the observation at `τ − 24 h` exists (it is not imputed for this purpose).

Otherwise **fall back to B0**.

**At `h = 24`, `τ − 24 h == t`, so B1 collapses into B0.** That is arithmetic,
not a defect, and it will be reported explicitly rather than presented as an
independent baseline at that horizon.

At `h = 1, 6, 12` the seasonal lag lies strictly before the origin, so B1 is a
genuinely different rule that asks whether yesterday's same-hour value beats
the most recent observation.

**No weekly seasonal naive is included in the primary study**: training PM2.5
autocorrelation at 168 h was ≈ **0.008**, so a weekly rule has no empirical
support here.

## B2 — training climatology

> **B2(station, τ) = a training-only median looked up by the target
> timestamp's calendar position.**

Deterministic fallback hierarchy, first available wins:

1. `median(PM2.5 | station, month(τ), hour(τ))` — training only;
2. `median(PM2.5 | station, hour(τ))` — training only;
3. `median(PM2.5 | station)` — training only.

Properties fixed now: the statistic is the **median** (MAE is the primary
metric, and the median minimises absolute error for a constant predictor);
every cell is fitted on the **training partition only**; the lookup key is the
**target** timestamp's month and hour, which is deterministic and knowable at
the origin; no validation or test statistic may enter any cell.

The climatology table is persisted before any metric is seen, so it cannot be
quietly retuned afterwards.

B2 is horizon-independent by construction — it uses no observation at all. It
is the "how much does knowing the season and hour alone buy you" reference, and
the natural companion to R0.

## B3 — gradient-boosted causal lag-feature baseline

> **B3 = supervised gradient boosting over causal features derived from the
> canonical 48-hour window.**

**Frozen now: the feature family.**
**Explicitly not frozen: the library and the hyperparameter grid.**

### Predeclared lag positions

Hours before the forecast origin:

```
0, 1, 2, 3, 6, 12, 24, 47
```

Lag 0 means the value **at** the forecast origin.

**Lag 48 is deliberately excluded.** The canonical window is
`[t−47, …, t]` — exactly 48 timestamps. An exact lag-48 value would sit at
`t−48`, one hour outside that window, and honouring it would mean quietly
reading a 49th hour. Rather than hide an extra hour, the oldest lag is **47**,
the genuine first element of the window. No hidden 49th hour exists anywhere in
V2.

### Predeclared causal summary windows

Trailing windows ending at the origin, all inside the 48-hour context:

```
3 h, 6 h, 12 h, 24 h, 48 h
```

Candidate statistics: mean, min, max, standard deviation. **No centred rolling
window** — every summary ends at the origin and looks only backwards.

### Variables and extras

Lags and summaries over PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN,
WSPM, plus observed masks, gap ages, `wd_code`, `rain_occurred`, origin and
target calendar channels, station identity and the horizon. `day_of_week` and
`weekend` are excluded, consistent with the primary feature contract.

### Recorded obligation

**The boosting library and the hyperparameter grid must be frozen and written
down at the start of Phase 3, before any validation metric is computed.**
Neither may be chosen from observed performance. No boosting package is
installed in Phase 2 and nothing is fitted.

---

## Evaluation obligations (defined, not executed)

- Primary metric form remains **macro station-horizon MAE**; it is not computed
  in this phase and is not replaced.
- Every baseline predicts every sample in the common universe.
- B1's collapse into B0 at `h = 24` is reported, not hidden.
- Severe-tail behaviour is measured against the frozen **244.0 µg/m³**
  training-derived threshold, as an evaluation endpoint — never as a dataset
  balancing trick. No oversampling, undersampling, weighting or duplication of
  severe targets exists in the primary dataset.
