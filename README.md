# AirSense

**Air Quality Prediction Using Statistical Analysis and Machine Learning**

---

## Provenance notice — read first

> **AirSense V1 is a period-authentic reconstruction of a classical Data
> Science project that could reasonably have been implemented using tools
> available during a six-week Data Science training completed in 2019. The
> reconstruction intentionally limits its V1 implementation to methods and
> software available by 26 April 2019. A later V2 research extension will
> investigate modern AI methods separately.**

To be explicit about what is and is not claimed:

**This repository was created recently, not in 2019.** It is a
reconstruction. It does **not** claim that this repository existed in 2019,
that its commits were authored in 2019, that any result here was produced in
2019, or that this work was submitted to any training provider or platform.
No historical commit, timestamp, certificate, experiment date or result has
been fabricated to suggest otherwise.

What *is* period-authentic is the **method and technology choice**: the
algorithms, libraries, versions, APIs and workflow are constrained to what
existed on or before **2019-04-26**. That constraint is the point of the
exercise.

Where a document records a date, it is labelled as either a **historical**
date (e.g. a package's actual PyPI release date, the dataset's actual
donation date) or a **reconstruction** date (e.g. when the dataset was
downloaded — 2026-09-05). The two are never blurred.

---

## Motivation

Airborne particulate matter under 2.5 micrometres (PM2.5) penetrates deep
into the respiratory system and is among the most consequential air-quality
measurements for public health. Beijing's PM2.5 record from 2010 to 2014
covers a period of severe and widely studied pollution episodes.

Pollution concentration is not driven by emissions alone. Meteorology
governs how pollutants disperse or accumulate: wind clears the air, still
cold conditions trap it, and pressure and humidity shape the boundary layer.
That makes the relationship between weather and PM2.5 a genuine statistical
question rather than a bookkeeping one — and a well-suited subject for
classical regression methods.

AirSense asks how far the classical toolkit gets on that question, and is
equally interested in where it fails.

---

## V1 research question

> **How accurately can classical statistical and machine-learning methods
> available in 2019 predict PM2.5 concentration from meteorological and
> temporal information?**

The V1 task is **regression** on the target `pm2.5`, in µg/m³.

It is **concurrent-hour estimation**, not forecasting: PM2.5 for an hour is
predicted from the meteorological and calendar description of that same hour.
No past PM2.5 value is used as an input, so the model must explain pollution
from weather and time alone. That is the harder and more interesting
question, and it is fixed deliberately — see
[`docs/RESEARCH_QUESTION.md`](docs/RESEARCH_QUESTION.md).

---

## Dataset

**UCI Beijing PM2.5 Data Set** (repository ID 381, DOI `10.24432/C5JS49`),
donated 2017-01-18 by Song Xi Chen, Peking University.

| | |
|---|---|
| Period | 2010-01-01 to 2014-12-31, hourly |
| Rows | 43,824 |
| Columns | 13 (1 index + 11 features + 1 target) |
| Target | `pm2.5` (µg/m³) |
| Missing | 2,067 `pm2.5` values (4.7166%); all other columns complete |
| PM2.5 source | US Embassy, Beijing |
| Meteorology source | Beijing Capital International Airport |
| License | CC BY 4.0 as currently presented by UCI |

The original UCI-distributed CSV is used, not a pre-cleaned derivative. It is
stored read-only in `data/raw/` with SHA-256
`4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`.

Full provenance: [`data/README.md`](data/README.md).
Full audit: [`docs/DATASET_AUDIT.md`](docs/DATASET_AUDIT.md).

The dataset's 2017 donation date places it comfortably before the 2019
cutoff, so its use is period-consistent.

---

## Historical technology constraints

**Hard cutoff: 2019-04-26.**

Reference environment (what V1 source is written against):

| Component | Pin |
|---|---|
| OS | Ubuntu 18.04 LTS |
| Python | CPython 3.6.7 |
| numpy | 1.15.4 |
| pandas | 0.23.4 |
| scipy | 1.1.0 |
| scikit-learn | 0.20.0 |
| matplotlib | 3.0.2 |
| seaborn | 0.9.0 |
| jupyter | 1.0.0 |

Every pin was verified against PyPI as published before the cutoff, and every
one ships a CPython 3.6 wheel. Frozen in
[`requirements-v1-2019.txt`](requirements-v1-2019.txt).

**Excluded from V1:** TensorFlow, PyTorch, Keras, XGBoost, LightGBM,
CatBoost, transformers, Hugging Face, LLM APIs, foundation models,
generative AI, agent frameworks, SHAP, MLflow, Weights & Biases, Optuna,
AutoML, modern time-series foundation models, deep neural networks,
cloud-managed AI services, and any API introduced after 2019-04-26.

Reasoning and the full list: [`docs/HISTORICAL_COMPATIBILITY.md`](docs/HISTORICAL_COMPATIBILITY.md).

> **Known environment blocker.** The current host runs Debian 12 with
> CPython 3.11, on which the pinned 2019 stack cannot be installed —
> verified, not assumed: no pinned numeric release resolves for CPython 3.11,
> and a source build of numpy 1.15.4 fails to compile against the 3.11 C API.
> Docker is not usable by this account and no 3.6 interpreter is available.
> Rather than modernise the pins, the host is recorded as a non-conforming
> execution environment and `scripts/preflight.py` reports a
> **HOST-COMPATIBILITY WARNING**. Modelling phases cannot run here.
> Details in [`docs/HISTORICAL_COMPATIBILITY.md`](docs/HISTORICAL_COMPATIBILITY.md) §4.

---

## Planned methodology

Four models, declared in advance:

| ID | Model |
|---|---|
| **M0** | Simple baseline predictor (constant) |
| **M1** | Linear Regression |
| **M2** | Decision Tree Regressor |
| **M3** | Random Forest Regressor |

Primary metrics, declared in advance: **MAE**, **RMSE**, **R²**. The
comparison is decided on MAE.

**Evaluation uses a chronological split** — an earlier period trains, a later
period tests. A random split on an hourly time series lets adjacent hours
land on both sides, so the model is scored largely on interpolation between
near-copies and looks better than it is. A random split may appear later as a
clearly labelled *secondary* comparison, because such workflows were common
in 2019 introductory training and the gap between the two designs is
instructive — but it never becomes the headline result. Full rule:
[`docs/V1_RESEARCH_PROTOCOL.md`](docs/V1_RESEARCH_PROTOCOL.md) §3.

Twelve phases, from environment reconstruction to the final report, are fixed
in [`docs/V1_RESEARCH_PROTOCOL.md`](docs/V1_RESEARCH_PROTOCOL.md).

---

## Project structure

```
AirSense/
├── README.md
├── requirements-v1-2019.txt      frozen 2019 dependency set
├── data/
│   ├── README.md                 dataset provenance
│   ├── raw/                      original UCI CSV, read-only, never modified
│   └── processed/                derived data (empty)
├── notebooks/                    analysis notebooks (empty)
├── src/
│   ├── data/                     data loading and auditing
│   ├── features/                 feature preparation (empty)
│   ├── models/                   model code (empty)
│   └── visualization/            plotting (empty)
├── scripts/
│   ├── preflight.py              read-only foundation validation
│   └── audit_dataset.py          writes artifacts/data_audit.json
├── artifacts/
│   └── data_audit.json           machine-readable dataset audit
├── results/                      model results (empty)
├── figures/                      generated figures (empty)
├── docs/
│   ├── HISTORICAL_COMPATIBILITY.md
│   ├── DATASET_AUDIT.md
│   ├── RESEARCH_QUESTION.md
│   ├── FEATURE_POLICY.md
│   ├── V1_RESEARCH_PROTOCOL.md
│   └── PHASE_00_FOUNDATION_RECORD.md
└── tests/                        (empty)
```

---

## Current status

**Phase 0 — foundation. Complete.**

| Phase | Status |
|---|---|
| 1. Environment reconstruction | Defined and verified; host blocker recorded |
| 2. Dataset provenance and audit | Complete |
| 3–12. EDA through final report | Not started |

Done: project structure; historical compatibility contract with verified
package release dates; frozen requirements; dataset acquired from the
original UCI source, audited and hashed; research question, feature policy
and research protocol pre-registered; read-only preflight utility.

**Not done, deliberately:** no exploratory analysis, no cleaning, no feature
engineering, **no model trained, no metric computed**. `results/`,
`figures/`, `notebooks/` and `data/processed/` are empty because nothing has
legitimately been produced for them yet.

`artifacts/data_audit.json` contains measurements of the raw file only. There
are no model results anywhere in this repository, and no placeholder values
that could be mistaken for results.

Verify the foundation at any time:

```sh
python scripts/preflight.py
```

---

## Reproducibility philosophy

1. **Raw data is immutable.** `data/raw/` is read-only and never modified.
   Derived data goes to `data/processed/`. The raw SHA-256 is recorded so
   any later change is detectable.
2. **Provenance is recorded, not remembered.** Source URL, retrieval date,
   donation date, license and digest are all written down, and the
   reconstruction date is never presented as a historical one.
3. **Decisions are pre-registered.** Models, metrics and evaluation design
   were fixed before any result was seen, so they cannot be quietly retrofitted
   to the outcome.
4. **The test set is used once.** It does not inform model choice, features,
   hyperparameters or the split boundary.
5. **Negative results are reported.** Every declared model is reported,
   including poor performers.
6. **No fabricated evidence.** No invented history, no illustrative metrics,
   no placeholder numbers in results files. A number appears only after it
   has actually been computed.
7. **Constraints are documented rather than dissolved.** Where the historical
   environment could not be reproduced, that is recorded as a blocker instead
   of resolved by modernising the pins.

---

## Future V2 boundary

A separate **V2** research extension will investigate modern AI methods.

V2 is deliberately kept out of scope here, and out of this repository's V1
history. It is not designed, described or implemented in V1, and no V1
document anticipates its technical content. The boundary exists so that V1's
results stand as an honest account of what 2019 classical methods achieve on
this problem, uninfluenced by knowledge of later techniques — which is
exactly what makes a later comparison meaningful.

---

## Citation

Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5JS49

Liang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H., &
Chen, S. X. (2015). Assessing Beijing's PM2.5 pollution: severity, weather
impact, APEC and winter heating. *Proceedings of the Royal Society A*,
471(2182).
