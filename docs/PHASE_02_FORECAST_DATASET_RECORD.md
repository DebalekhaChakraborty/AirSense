# AirSense V2 — Phase 2 Record

**Forecast dataset, windowing contract, sample-universe freeze and baseline
protocol.**
**Executed on branch `master` from `09942928…` ("Start V2 on an empty master
branch").**

No model was fitted. No prediction was generated. No forecast metric of any
kind was computed. No validation or test target value was materialised. The
`legacy` branch was not touched.

---

## 1. Gates

`git branch --show-current` returned `master`. `validate_foundation.py` and
`validate_phase1.py` both returned **PASS** before anything was written, and
every hash referenced by `v2_foundation_manifest.json` and
`v2_phase1_manifest.json` was re-verified unchanged — 19 Phase-1 tables and
figures plus the policy, config and summary documents. Nothing upstream was
regenerated.

### 1.1 Environment — the Phase-0 disk blocker is resolved

Free disk was measured independently at this phase rather than assumed:
**17.89 GB before** the Phase-2 build and **17.63 GB after** it.

The full history, recorded here because it is an operational fact of the V2
environment rather than a scientific result:

- **Phase 0 observed approximately 1.9 GB free and correctly recorded a disk
  blocker.** That reading was accurate for the machine as it stood at that
  phase, and the judgement that a modern forecasting environment could not be
  built in 1.9 GB was right.
- **Before and during Phase 2, re-downloadable caches were cleaned** — a 13 GB
  Hugging Face benchmark cache holding three Qwen checkpoints, the Docker build
  cache, Playwright browser binaries, and uv/pip/poetry/npm/apt package caches.
- **Free disk increased to approximately 17.9 GB.**
- **No V1 runtime, no AirSense scientific artifact and no project source was
  deleted.** The V1 CPython 3.6.7 `venv/` prefix is intact; so are `data/raw/`,
  every Phase-0 and Phase-1 artifact, the Phase-2 processed layer, and all of
  `/home/AI_POC`. Only re-downloadable caches outside the repository were
  removed.
- **The disk blocker is therefore resolved for the current V2 environment.**
  Roughly 17.6 GB is enough for the Phase 4–5 environment: PyTorch CPU
  (~1.5 GB), pandas/scikit-learn (~0.5 GB), one foundation-model checkpoint
  (0.25–1.5 GB) and processed experiment outputs.
- **`docs/ENVIRONMENT_PLAN.md` is deliberately left unchanged**, exactly as
  frozen in Phase 0 at SHA-256 `8414e6db8ab7…`. It is historical evidence of
  the environment observed at that phase, and it is not amended to reflect a
  later state. This record is the current environment statement; the Phase-0
  plan remains the Phase-0 observation.

The other two Phase-0 environment findings are unchanged: this machine still
has **no GPU**, and the TimesFM 3.0 non-commercial weight licence remains an
open decision deferred to the foundation-model phase. No heavy ML dependency
has been installed and no V2 environment has been created.

## 2. What was frozen

| Decision | Value |
|---|---|
| Primary context | **48 hours** |
| Robustness contexts | 24 h, 72 h (predeclared, may never replace 48 h on validation) |
| 168 h | excluded from the primary study |
| PM2.5 input log transform | **none** |
| CO | **kept** in R2 |
| RAIN | continuous value **plus** `rain_occurred` binary |
| `day_of_week`, `weekend` | **excluded** from the primary representation |
| Severe-hour balancing | **none** |
| Severe threshold | 244.0 µg/m³, training pooled P95 |

The context length was fixed from Phase-1 training autocorrelation — 0.965 at
1 h decaying to 0.098 at 48 h and −0.004 at 72 h — **before any validation
performance exists**, and the robustness contexts are declared in advance
precisely so that a better 24 h or 72 h score later cannot be used to
retroactively rename the primary.

## 3. Architecture — windows are generated, not stored

Materialising `samples × context × features` across 4 horizons and 4 regimes
would duplicate the same station history millions of times. Instead:

```
data/processed/phase2/
  station_series/<Station>/   numeric_scaled, observed_mask, gap_age,
                             wd_code, rain_occurred, fill_source,
                             pm25_scaled_train_only, meta.json
  target_series/              <Station>_train_pm25.npy   (training only)
  calendar/                   calendar_cyclic, calendar_raw  (shared, once)
  climatology/                b2_training_climatology.csv
  sample_index/               train.csv, validation.csv, test.csv
```

Total **267,265,583 bytes (254.9 MB)** — 220 MB of that is the three sample
indexes, 34 MB the station series. No window tensor exists anywhere. Calendar
channels are station-independent and are therefore stored once rather than
twelve times.

## 4. Sample universe

One canonical universe keyed by `(partition, station, target_timestamp,
horizon)`, identified as `<partition>|<station>|<YYYYMMDDHH>|h<horizon>`.

| Partition | h=1 | h=6 | h=12 | h=24 | Total |
|---|---:|---:|---:|---:|---:|
| Train | 205,413 | 205,353 | 205,281 | 205,137 | **821,184** |
| Validation | 103,287 | 103,287 | 103,287 | 103,287 | **413,148** |
| Test | 102,753 | 102,753 | 102,753 | 102,753 | **411,012** |

Exclusions are structural only: **17,004 / 8,484 / 9,468** targets excluded for
missing labels (never imputed), and **2,772** training samples excluded because
a full 48-hour context would start before 2013-03-01. Validation and test lose
**nothing** to history, because their context legitimately reaches into the
preceding partition.

Training counts fall below the Phase-1 provisional figures exactly as
predicted, and by exactly the amount the 48-hour context requires: 48, 53, 59
and 71 hours per station at h = 1, 6, 12, 24.

**Input-variable missingness never changes eligibility.** The causal policy
handles it, so no model can look better by silently skipping hard rows.

## 5. Target quarantine, enforced structurally

PM2.5 values are materialised for training timestamps only. The parser drops
validation and test target values *inside* `load_station_raw`, so they never
reach an array:

- `pm25_scaled_train_only.npy` and `target_series/*.npy` are NaN beyond
  2015-02-28 23:00, verified at all 12 stations;
- validation and test indexes carry `target_observed` only — no value, bin,
  quantile or severity label;
- `CausalTargetHistory` refuses any timestamp after the requested origin, and
  refuses every value beyond its unseal ceiling, which in Phase 2 is the end of
  training. Sealed values are **not even read off disk**, so a bug elsewhere
  cannot leak them.

## 6. Verification

- **44 unit tests** across `tests/test_preprocessing.py`,
  `tests/test_windowing.py` and `tests/test_target_history.py` — all pass.
- **`validate_phase2.py`: 59 gates, 0 failures, exit 0.**
- **108 deterministic window-boundary audit cases** (first/middle/last sample ×
  3 stations × 4 horizons × 3 partitions): every one exactly 48 hourly
  timestamps, ending at the origin, starting at origin − 47 h, with no input
  after the origin.
- **12 independent recomputations** through a separate pure-standard-library
  path that does not import the Phase-2 modules — sample counts, first sample
  metadata, a window sequence, a short-gap carry, a beyond-ceiling fallback, a
  mask sequence, a gap-age sequence, a scaled value, the RAIN exception, a
  climatology cell, and the target-history refusal. All agree.
- **Determinism: 122 files byte-identical** across two complete rebuilds —
  every `.npy`, every CSV, every JSON.

One defect was found and fixed during verification: the independent probe
initially indexed 100 hours *before* the first eligible training sample, which
is a negative index. That was an error in the verification code, not in the
data layer; the probe now sits comfortably inside the timeline.

## 7. Baselines defined, none executed

B0 causal persistence, B1 daily seasonal naive with B0 fallback, B2
training-only station×month×hour climatology with a frozen fallback hierarchy,
and B3 a gradient-boosted causal lag-feature baseline whose **feature family is
frozen now** while its library and hyperparameter grid are explicitly deferred
to the start of Phase 3, before validation opens.

B3's lag list is `0, 1, 2, 3, 6, 12, 24, 47`. Lag 48 was **rejected**: the
canonical window is `[t−47, …, t]`, so an exact lag-48 value would sit one hour
outside it. Rather than quietly read a 49th hour, the oldest lag is the genuine
first element of the window.

B1 collapses into B0 at h = 24 by arithmetic (`τ − 24 h == t`). This is
recorded now so it is reported later rather than presented as an independent
baseline.

The B2 climatology table is persisted before any metric exists. Its Dongsi
station-level median is 66.0 µg/m³, matching the Phase-1 by-station median
exactly — an independent cross-check that the training partition is being read
consistently across phases.

## 8. Deliberately not decided

The R3 spatial tensor encoding; the B3 library and grid; batching and sampling
strategy; any model architecture; whether the 24 h / 72 h robustness contexts
are run at all. None of these may be settled by looking at validation
performance.

## 9. Git

No write operation: no `git add`, `git commit`, `git push`, reset, rebase,
merge, tag, branch creation or history rewrite. All Phase-2 changes were left
unstaged. Inspection was read-only.
