# AirSense V2

**Modern Temporal and Spatiotemporal Air-Quality Forecasting**

AirSense V2 forecasts PM2.5 concentration at 12 Beijing monitoring stations at
**future** horizons of 1, 6, 12 and 24 hours, using explicit temporal memory and
cross-station context under strict chronological generalisation.

> **Status: Phase 11 complete — manuscript construction. The study is
> scientifically complete.**
>
> The locked 2016-03-01 → 2017-02-28 test was **opened exactly once**, at
> Phase 8, under a confirmatory hierarchy frozen beforehand. Predictions were
> generated chronologically with metrics unseen, hashed before scoring, then
> scored once. `final_test_status` is **evaluated** and can never return to
> sealed. Phases 9 and 10 performed post-test exploratory analysis and final
> synthesis, adding no new experiment.

### Headline result — locked final test

Both halves belong together; reporting either alone misrepresents the study.

A gradient-boosted model over causal pollutant, meteorological and temporal
history (`B3_R2`) achieved a macro station-horizon MAE of **31.99 µg/m³**
against **36.14** for causal persistence (`B0`) — an improvement of
**11.49%** — and improved on persistence **at every horizon** from 1 to 24
hours, over 411,012 samples and 48 equally weighted station-horizon cells.

**In the severe tail, persistence won.** Restricted to observed PM2.5 above the
training pooled P95 of 244.0 µg/m³ (n = 20,336), the prespecified secondary
model `GRU_R1` reached **109.33 µg/m³** and `B3_R2` **131.90**, while the
persistence reference benchmark recorded the lowest severe MAE at **93.22**.
Reliable forecasting of extreme episodes remains the principal open problem.

Frozen model roles: `B3_R2` primary confirmatory · `GRU_R1` (seed 42) secondary
confirmatory severe-tail · `B0` reference benchmark. Thirteen further models are
development-only. No foundation model was executed. **No overall winner label is
assigned**, because the primary model leads overall while the reference
benchmark leads in the severe tail.

Every number above traces to
[`artifacts/phase8_primary_results_lock.json`](artifacts/phase8_primary_results_lock.json).

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
| Locked final test | 2016-03-01 → 2017-02-28 (**opened once at Phase 8; evaluated**) |
| Partition rule | by target timestamp; origin = target − horizon |
| Primary metric | macro station-horizon MAE |
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
| Phase 7 record — robustness and pre-test freeze | [`PHASE_07_ROBUSTNESS_RECORD.md`](docs/PHASE_07_ROBUSTNESS_RECORD.md) |
| Phase 7 coordinate acquisition audit | [`PHASE7_COORDINATE_ACQUISITION_AUDIT.md`](docs/PHASE7_COORDINATE_ACQUISITION_AUDIT.md) |
| Phase 8 final-test protocol | [`PHASE_08_FINAL_TEST_PROTOCOL.md`](docs/PHASE_08_FINAL_TEST_PROTOCOL.md) |
| Phase 8 record — locked final test | [`PHASE_08_LOCKED_FINAL_TEST_RECORD.md`](docs/PHASE_08_LOCKED_FINAL_TEST_RECORD.md) |
| Phase 9 post-test analysis protocol | [`PHASE_09_POST_TEST_ANALYSIS_PROTOCOL.md`](docs/PHASE_09_POST_TEST_ANALYSIS_PROTOCOL.md) |
| Phase 9 record — post-test error analysis | [`PHASE_09_POST_TEST_ERROR_ANALYSIS_RECORD.md`](docs/PHASE_09_POST_TEST_ERROR_ANALYSIS_RECORD.md) |
| V1/V2 failure-mode comparison (qualitative) | [`PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md`](docs/PHASE_09_V1_V2_FAILURE_MODE_COMPARISON.md) |
| Phase 10 — final scientific synthesis | [`PHASE_10_FINAL_SCIENTIFIC_SYNTHESIS.md`](docs/PHASE_10_FINAL_SCIENTIFIC_SYNTHESIS.md) |
| Publication claim guide (binding wording) | [`PUBLICATION_CLAIM_GUIDE.md`](docs/PUBLICATION_CLAIM_GUIDE.md) |
| Phase 12 reproducibility notes | [`PHASE_12_REPRODUCIBILITY_NOTES.md`](docs/PHASE_12_REPRODUCIBILITY_NOTES.md) |
| Proposed README update (Phase 11, superseded) | [`README_PHASE11_PROPOSED_UPDATE.md`](docs/README_PHASE11_PROPOSED_UPDATE.md) |

The manuscript draft and its evidence apparatus are in
[`manuscript/`](manuscript/): the draft itself, six tables, a ten-figure plan,
supplementary methods, a language audit, a readiness review, and a
claim-traceability matrix mapping 235 quantitative claims to hash-pinned
artifacts.

Machine-readable contracts: [`configs/study.json`](configs/study.json),
[`configs/preprocessing.json`](configs/preprocessing.json),
[`configs/windowing.json`](configs/windowing.json),
[`configs/baselines.json`](configs/baselines.json).

## Reproduce

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

# Phase 7 — multi-seed robustness, LOSO, spatial diagnostics, pre-test freeze
.venv-v2/bin/python scripts/build_phase7_candidate_freeze.py
.venv-v2/bin/python scripts/run_phase7_multiseed.py               # 16 refits, ~4.3 h
.venv-v2/bin/python scripts/run_phase7_leave_one_station_out.py   # 12 folds, ~2.5 h
python3 scripts/run_phase7_spatial_analysis.py                    # analysis only
.venv-v2/bin/python scripts/build_phase7_pretest_freeze.py        # freezes model roles
.venv-v2/bin/python scripts/validate_phase7.py

# Phase 8 — THE LOCKED FINAL TEST.  READ THE WARNING BELOW BEFORE RUNNING.
.venv-v2/bin/python scripts/build_phase8_receipts.py --preopening
.venv-v2/bin/python scripts/run_phase8_final_test.py              # opens the test
.venv-v2/bin/python scripts/verify_phase8_independently.py        # separate code path
python3 scripts/build_phase8_figures.py
.venv-v2/bin/python scripts/build_phase8_manifest.py
.venv-v2/bin/python scripts/validate_phase8_postopening.py        # current-state gates

# Phase 9 — post-test exploratory analysis, from the frozen prediction arrays
.venv-v2/bin/python scripts/run_phase9_post_test_analysis.py
.venv-v2/bin/python scripts/verify_phase9_independently.py
python3 scripts/build_phase9_figures.py
.venv-v2/bin/python scripts/build_phase9_summary_and_manifest.py
.venv-v2/bin/python scripts/validate_phase9.py

# Phase 10 — final synthesis (no new experiment)
.venv-v2/bin/python scripts/build_phase10_synthesis.py
.venv-v2/bin/python scripts/verify_phase10_independently.py
.venv-v2/bin/python scripts/validate_phase10.py

# Phase 11 — manuscript package
python3 scripts/validate_phase11.py                               # 289 gates
python3 scripts/verify_phase11_independently.py                   # 54 checks

# Phase 12 — release hardening
python3 scripts/validate_phase12a.py
```

Or, in one command each:

```sh
make env          # reconstruct the frozen environment
make data         # rebuild the derived Phase-2 layer from tracked raw data
make check        # unit suite + every validator, expected failures annotated
```

> **Running Phase 8 on a fresh clone re-opens the locked test.**
>
> In this repository the test was opened once, on 2026-09, and the result is
> frozen. Re-running `run_phase8_final_test.py` regenerates predictions and
> metrics; it does not and cannot un-freeze what is recorded. If your rerun
> disagrees with `artifacts/phase8_primary_results_lock.json`, that is a finding
> to report — **not** a reason to update the lock.

### Three validators fail by design

`validate_phase3.py`, `validate_phase8.py` and `validate_phase8_postopening.py`
each assert a repository state that later phases legitimately advanced past: a
still-sealed test, an unmoved HEAD, and an un-started Phase 9 respectively.
Their sources are preserved byte-for-byte as historical evidence and must never
be patched to go green. Two further gates are *conditional* rather than
by-design — one fails while any tracked file is uncommitted, one while a commit
is unpushed. `make validate` names every expected and conditional failure, and
exits non-zero only on a failure that is *not* documented.
See [`CONTRIBUTING.md`](CONTRIBUTING.md).

`scripts/acquire_dataset.py --verify-only` re-hashes what is on disk without
writing anything.

## Current limits, stated plainly

Phases 0–6 established a design, a data layer, classical baselines, two
temporal neural baselines, one modern Transformer and one cross-station
spatiotemporal model, all on development data. Phases 7–11 added robustness
replication, the single locked final-test evaluation, post-test exploratory
analysis, synthesis and the manuscript package; the headline result above is
the locked-test outcome. What remains genuinely open is the severe tail: no
model evaluated here improved on persistence during extreme episodes.
The disk blocker recorded in Phase 1 has been resolved by documented
cache cleanup — see [`docs/ENVIRONMENT_PLAN.md`](docs/ENVIRONMENT_PLAN.md).
This machine still has **no GPU**.

The foundation-model question is now settled for this study rather than open:
TimesFM 3.0 and Moirai-2.0-R-small are excluded on non-commercial weight
licences, and Chronos-2 — permissively licensed and otherwise suitable — is
excluded because its pretraining corpus overlaps the sealed test window by
1,416 hours per station. Every Phase-5 result comes from weights trained here
from scratch.

## Licensing, citation and contributing

| | |
|---|---|
| Code (`src/ scripts/ tests/ configs/`) | [MIT](LICENSE) |
| Docs, manuscript, artifacts, figures, results | CC BY 4.0 |
| Dataset (`data/raw/`) | **CC BY 4.0 as presented by UCI — NOT covered by the code licence** |

The MIT licence on the code does not, and cannot, relicense the dataset. The
repository authors do not own it. Full breakdown, including the third-party
station coordinates: [`LICENSES.md`](LICENSES.md).

Citation metadata: [`CITATION.cff`](CITATION.cff). No tagged release exists
yet — cite the commit SHA. The manuscript is unpublished and in preparation, so
the citation file deliberately asserts no venue, year or DOI for it.

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md) — read the frozen-evidence
rule first. Security policy: [`SECURITY.md`](SECURITY.md).
