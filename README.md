# AirSense V2

**Modern Temporal and Spatiotemporal Air-Quality Forecasting**

AirSense V2 forecasts PM2.5 concentration at 12 Beijing monitoring stations at
**future** horizons of 1, 6, 12 and 24 hours, using explicit temporal memory and
cross-station context under strict chronological generalisation.

> **Status: Phase 6 complete — cross-station development modelling
> evaluated.** Development validation is open; the locked 2016-03-01 →
> 2017-02-28 test remains **SEALED** — no test target has been read, no test
> prediction generated, no test metric computed. Nothing below is V2
> performance.

**V1 is complete and frozen on the [`legacy`](../../tree/legacy) branch. V2
lives on `master`.** V1 is historical evidence and is never modified; V2 is a
clean modern implementation that does not reuse V1 code.

## Why V2 exists

V1 showed that concurrent meteorology and calendar information leaves a large,
*temporally structured* PM2.5 error unexplained. Its best model's residuals
correlated at ≈ 0.903 at a one-hour lag — the errors were predictable from their
own recent history — while the model was forbidden by design to use any PM2.5
history at all. It also failed hardest where it mattered most, under-predicting
≈ 96.3% of severe-pollution hours.

V2 asks the obvious follow-on question:

> **Can explicit temporal memory and cross-station information improve
> multi-horizon PM2.5 forecasting, particularly during high-concentration
> episodes, while preserving strict chronological generalisation?**

V1's numbers motivate V2. They are **not** V2 baselines: different task,
different data set, different period. V1's 2014 test is spent and will never be
presented as unseen evidence again.

## What makes V2 different

**V2 is true future-horizon forecasting.** At origin `t` it predicts PM2.5 at
`t + h`, using only information available at or before `t`. V1 was
concurrent-hour estimation — it described the same hour it predicted. Observed
future weather is not available to a V2 forecast, and the full leakage contract
is in [`docs/FORECAST_SEMANTICS.md`](docs/FORECAST_SEMANTICS.md).

## Dataset

UCI **Beijing Multi-Site Air-Quality Data** (ID 501, DOI `10.24432/C5RK5G`) —
420,768 hourly rows, 12 stations, 2013-03-01 → 2017-02-28, six pollutants plus
meteorology. Acquired from the official UCI source only and verified: the
hourly grid is complete at every station, with no missing timestamps, no
duplicate `(station, timestamp)` pairs and no duplicated rows. All missingness
is value missingness; PM2.5 is 2.077% missing.

Full record, including a contamination finding in the official archive:
[`docs/DATASET_PROVENANCE.md`](docs/DATASET_PROVENANCE.md).

## Design

| | |
|---|---|
| Horizons | 1, 6, 12, 24 hours |
| Train | 2013-03-01 → 2015-02-28 |
| Validation | 2015-03-01 → 2016-02-29 |
| Locked final test | 2016-03-01 → 2017-02-28 (**sealed**) |
| Partition rule | by target timestamp; origin = target − horizon |
| Primary metric (provisional) | macro station-horizon MAE |
| Severe threshold (frozen, training-only P95) | **244.0 µg/m³** |
| Primary context length | **48 h** (24 h and 72 h predeclared as robustness) |
| Common sample universe | 821,184 train · 413,148 validation · 411,012 test |
| Windows | built **on demand**; no window tensor is stored |

### Development validation — NOT FINAL TEST

Macro station-horizon MAE (µg/m³) on 2015-03-01 → 2016-02-29, used to select
among predeclared candidates. **These are development figures, not held-out
performance.**

| Model | Family | macro station-horizon MAE | Rank |
|---|---|---:|---:|
| B3_R2 gradient boosting, + co-pollutants | classical | **30.11** | 1 |
| B3_R1 gradient boosting, + PM2.5 history | classical | 30.47 | 2 |
| TCN_R1 causal temporal convnet | neural | 30.69 | 3 |
| GRU_R1 recurrent | neural | 30.91 | 4 |
| SA_R3 station attention, all 12 stations | spatial | 31.53 | 5 |
| GRU_R2 recurrent, + co-pollutants | neural | 31.84 | 6 |
| SA_R2 station attention, self-only control | spatial | 32.24 | 7 |
| iTransformer_R1 attention across variates | transformer | 32.48 | 8 |
| iTransformer_R2, + co-pollutants | transformer | 32.72 | 9 |
| B0 causal persistence | classical | 33.13 | 10 |
| TCN_R2, TCN_R0, GRU_R0, B3_R0, B1, B2 | mixed | 34.08 – 56.73 | 11–16 |

Findings worth stating plainly. A boosted model with 207 meteorological and
calendar features is **worse than repeating the last observation**, so nearly
all of its advantage comes from PM2.5's own history. Learned sequence models
did **not** beat those engineered features overall — though they win on
meteorology-only inputs and at the 24-hour horizon. And on the 18,636 severe
hours above the frozen 244.0 µg/m³ threshold, **persistence still beats every
learned model**: 101.3 against GRU_R1's 119.0 and B3_R2's 136.6. The
severe-tail failure that motivated V2 is not solved, now across four model
families.

Phase 5 added an inverted Transformer whose attention runs **across variates**
rather than across time, to ask whether the co-pollutant penalty seen in
Phase 4 was a property of the data or of the architecture. The answer is
mostly the architecture, but not entirely: the penalty shrinks from −3.0%
(GRU) and −11.1% (TCN) to −0.75%, and in the severe tail from about −21% to
−3.6%, yet it never turns positive. Co-pollutants still add nothing to a
learned sequence model. **No foundation model could be evaluated**: Chronos-2
was stopped by a pretraining-overlap audit — its corpus contains Beijing air
quality overlapping the sealed test window — and TimesFM 3.0 and
Moirai-2.0-R-small by non-commercial weight licences.

Phase 6 ran the study's first **paired information experiment**: one
station-attention architecture, identical capacity and seed, evaluated with a
self-only mask against full attention over all 12 stations. Cross-station
context improved the primary metric by **2.19%**, with **12 of 12 stations**
and 36 of 48 station-horizon cells improving, and by **15.0% on the severe
tail** — though it is *negative* at the 24-hour horizon. **H4 is supported on
development validation**, the first hypothesis here settled by holding capacity
fixed and changing only the information. Gradient boosting still leads overall,
and persistence still owns the severe tail. No station coordinates exist in
this dataset, so nothing spatial or geographical is claimed. Details:
[`CLASSICAL_BASELINE_RESULTS.md`](docs/CLASSICAL_BASELINE_RESULTS.md),
[`TEMPORAL_NEURAL_RESULTS.md`](docs/TEMPORAL_NEURAL_RESULTS.md),
[`MODERN_MODEL_RESULTS.md`](docs/MODERN_MODEL_RESULTS.md),
[`SPATIOTEMPORAL_RESULTS.md`](docs/SPATIOTEMPORAL_RESULTS.md).

**Planned progression — temporal, then multivariate, then spatial:**

| Regime | Adds |
|---|---|
| R0 | target-station meteorology + calendar history |
| R1 | + target-station PM2.5 history |
| R2 | + co-pollutant history (PM10, SO2, NO2, CO, O3) |
| R3 | + cross-station history |

Each regime adds exactly one class of information, so any gain is attributable.

## Documentation

| Topic | Document |
|---|---|
| Research blueprint | [`V2_RESEARCH_BLUEPRINT.md`](docs/V2_RESEARCH_BLUEPRINT.md) |
| Preregistered protocol | [`V2_RESEARCH_PROTOCOL.md`](docs/V2_RESEARCH_PROTOCOL.md) |
| Forecast semantics and leakage contract | [`FORECAST_SEMANTICS.md`](docs/FORECAST_SEMANTICS.md) |
| Dataset provenance and audit | [`DATASET_PROVENANCE.md`](docs/DATASET_PROVENANCE.md) |
| Model landscape, audited 2026 | [`MODEL_LANDSCAPE_2026.md`](docs/MODEL_LANDSCAPE_2026.md) |
| Environment plan | [`ENVIRONMENT_PLAN.md`](docs/ENVIRONMENT_PLAN.md) |
| Phase 0 record | [`PHASE_00_V2_FOUNDATION_RECORD.md`](docs/PHASE_00_V2_FOUNDATION_RECORD.md) |
| Training-only EDA (Phase 1) | [`TRAINING_EDA.md`](docs/TRAINING_EDA.md) |
| Preprocessing contract (Phase 1) | [`PREPROCESSING_POLICY.md`](docs/PREPROCESSING_POLICY.md) |
| Windowing contract (Phase 2) | [`WINDOWING_CONTRACT.md`](docs/WINDOWING_CONTRACT.md) |
| Baseline protocol (Phase 2) | [`BASELINE_PROTOCOL.md`](docs/BASELINE_PROTOCOL.md) |
| Phase 2 record | [`PHASE_02_FORECAST_DATASET_RECORD.md`](docs/PHASE_02_FORECAST_DATASET_RECORD.md) |
| Classical baseline results (Phase 3) | [`CLASSICAL_BASELINE_RESULTS.md`](docs/CLASSICAL_BASELINE_RESULTS.md) |
| Phase 3 record | [`PHASE_03_CLASSICAL_BASELINES_RECORD.md`](docs/PHASE_03_CLASSICAL_BASELINES_RECORD.md) |
| Temporal neural results (Phase 4) | [`TEMPORAL_NEURAL_RESULTS.md`](docs/TEMPORAL_NEURAL_RESULTS.md) |
| Phase 4 record | [`PHASE_04_TEMPORAL_NEURAL_RECORD.md`](docs/PHASE_04_TEMPORAL_NEURAL_RECORD.md) |
| Modern model results (Phase 5) | [`MODERN_MODEL_RESULTS.md`](docs/MODERN_MODEL_RESULTS.md) |
| Phase 5 record | [`PHASE_05_MODERN_MODELS_RECORD.md`](docs/PHASE_05_MODERN_MODELS_RECORD.md) |
| Phase 5 model audit | [`PHASE5_MODEL_AUDIT.md`](docs/PHASE5_MODEL_AUDIT.md) |
| Spatiotemporal results (Phase 6) | [`SPATIOTEMPORAL_RESULTS.md`](docs/SPATIOTEMPORAL_RESULTS.md) |
| Phase 6 record | [`PHASE_06_SPATIOTEMPORAL_RECORD.md`](docs/PHASE_06_SPATIOTEMPORAL_RECORD.md) |
| Chronos-2 pretraining-overlap audit | [`CHRONOS2_PRETRAINING_AUDIT.md`](docs/CHRONOS2_PRETRAINING_AUDIT.md) |

Machine-readable contracts: [`configs/study.json`](configs/study.json),
[`configs/preprocessing.json`](configs/preprocessing.json),
[`configs/windowing.json`](configs/windowing.json),
[`configs/baselines.json`](configs/baselines.json).

## Reproduce Phases 0 and 1

No ML dependency and nothing to install — the pipeline runs on the standard
library plus the numpy and Pillow already present on the host:

```sh
python3 scripts/acquire_dataset.py           # official UCI download, hash, extract
python3 scripts/audit_dataset.py             # deterministic dataset audit
python3 scripts/validate_foundation.py       # Phase-0 gates; exit 0 on pass
python3 scripts/run_training_eda.py          # training-only EDA tables + figures
python3 scripts/build_preprocessing_policy.py --write-manifest
python3 scripts/validate_phase1.py           # Phase-1 gates + independent recompute
python3 scripts/build_forecast_dataset.py    # canonical series, sample indexes, audits
python3 scripts/validate_phase2.py           # Phase-2 gates + independent recompute
python3 -m unittest discover -s tests        # unit tests (stdlib + numpy)

# Phase 3 needs the modern environment (.venv-v2, CPython 3.11 + LightGBM 4.7.0)
uv venv --python 3.11 .venv-v2
uv pip install --python .venv-v2/bin/python -r requirements-v2-core-lock.txt
.venv-v2/bin/python scripts/build_phase3_freeze.py --freeze --receipt
.venv-v2/bin/python scripts/run_phase3_development.py   # 96 candidate fits
python3 scripts/build_phase3_report.py                  # tables + figures
.venv-v2/bin/python scripts/validate_phase3.py          # Phase-3 gates

# Phase 4 adds CPU-only PyTorch to the same environment
uv pip install --python .venv-v2/bin/python torch==2.14.0 \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-v2/bin/python safetensors==0.8.0
.venv-v2/bin/python scripts/build_phase4_internal_split.py
.venv-v2/bin/python scripts/run_phase4_internal_selection.py  # 24 fits
.venv-v2/bin/python scripts/run_phase4_full_refit.py          # 6 models
python3 scripts/build_phase4_report.py                        # tables + figures
.venv-v2/bin/python scripts/validate_phase4.py                # Phase-4 gates

# Phase 5 needs no new package: the iTransformer is trained from scratch
.venv-v2/bin/python scripts/run_phase5_itransformer_selection.py   # 8 fits
.venv-v2/bin/python scripts/run_phase5_itransformer_full_refit.py  # 2 models
.venv-v2/bin/python scripts/run_phase5_itransformer_full_refit.py --determinism
python3 scripts/build_phase5_report.py                        # tables + figures
.venv-v2/bin/python scripts/validate_phase5.py                # Phase-5 gates

# Phase 6 groups the 12 stations by shared forecast origin
.venv-v2/bin/python scripts/run_phase6_benchmark.py               # batch size
.venv-v2/bin/python scripts/run_phase6_internal_selection.py      # 8 fits
.venv-v2/bin/python scripts/run_phase6_full_refit.py              # 2 models
.venv-v2/bin/python scripts/run_phase6_full_refit.py --determinism
python3 scripts/build_phase6_report.py                        # tables + figures
.venv-v2/bin/python scripts/validate_phase6.py                # Phase-6 gates
```

`scripts/acquire_dataset.py --verify-only` re-hashes what is on disk without
writing anything.

## Current limits, stated plainly

Phases 0–6 established a design, a data layer, classical baselines, two
temporal neural baselines, one modern Transformer and one cross-station
spatiotemporal model — not a result on unseen data. The disk blocker recorded in Phase 1 has been resolved by documented
cache cleanup — see [`docs/ENVIRONMENT_PLAN.md`](docs/ENVIRONMENT_PLAN.md).
This machine still has **no GPU**.

The foundation-model question is now settled for this study rather than open:
TimesFM 3.0 and Moirai-2.0-R-small are excluded on non-commercial weight
licences, and Chronos-2 — permissively licensed and otherwise suitable — is
excluded because its pretraining corpus overlaps the sealed test window by
1,416 hours per station. Every Phase-5 result comes from weights trained here
from scratch.
