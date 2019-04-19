# Historical Compatibility Contract — AirSense V1

**Hard historical cutoff: `2019-04-26`**

This document defines what "period-authentic" means for AirSense V1, records
the verification that was actually performed, and states honestly where exact
reconstruction was **not** possible on the current host.

> **Provenance notice.** This repository was created as a present-day
> reconstruction. Nothing in this document asserts that this code, these
> files, or any result existed in 2019. What is constrained to 2019 is the
> *technology and method choice* of the V1 implementation, not the repository
> itself.

---

## 1. The contract

AirSense V1 may only use technologies, algorithms, APIs, workflows and
software versions that were publicly available **on or before 2019-04-26**.

This applies to:

- language level and standard library usage,
- third-party package versions,
- the APIs called from those packages,
- the modelling families used,
- the general workflow and tooling idioms.

It does **not** apply to the version-control system, the operating system that
happens to host the reconstruction, or the fact that the reconstruction is
being carried out now. Those are recorded truthfully rather than disguised.

---

## 2. Reference historical environment

This is the environment V1 source code is written **against**.

| Component | Reference pin |
|---|---|
| Operating system | Ubuntu 18.04 LTS |
| Interpreter | CPython 3.6.7 |
| numpy | 1.15.4 |
| pandas | 0.23.4 |
| scipy | 1.1.0 |
| scikit-learn | 0.20.0 |
| matplotlib | 3.0.2 |
| seaborn | 0.9.0 |
| jupyter | 1.0.0 |

Frozen in [`requirements-v1-2019.txt`](../requirements-v1-2019.txt).

### 2.1 Verification (a): every pin predates the cutoff

Release dates were read from the PyPI JSON API during this reconstruction
(earliest upload timestamp of any distribution for that exact version):

| Package | Version | First released | Before 2019-04-26? | `requires_python` declared |
|---|---|---|---|---|
| numpy | 1.15.4 | 2018-11-04 | yes | `>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*` |
| pandas | 0.23.4 | 2018-08-04 | yes | `>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*` |
| scipy | 1.1.0 | 2018-05-05 | yes | `>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*` |
| scikit-learn | 0.20.0 | 2018-09-26 | yes | unspecified |
| matplotlib | 3.0.2 | 2018-11-11 | yes | `>=3.5` |
| seaborn | 0.9.0 | 2018-07-16 | yes | unspecified |
| jupyter | 1.0.0 | 2015-08-12 | yes | unspecified |

All seven pins are confirmed pre-cutoff. The oldest cutoff margin is
matplotlib 3.0.2 at roughly five months before 2019-04-26; all others are
older still.

### 2.2 Verification (b): interpreter compatibility of the pinned set

Published distribution tags for each pinned version:

| Package | Distribution tags published for that version |
|---|---|
| numpy 1.15.4 | cp27, cp34, cp35, **cp36**, cp37, source |
| pandas 0.23.4 | cp27, cp35, **cp36**, cp37, source |
| scipy 1.1.0 | cp27, cp34, cp35, **cp36**, cp37, source |
| scikit-learn 0.20.0 | cp27, cp34, cp35, **cp36**, cp37, source |
| matplotlib 3.0.2 | cp35, **cp36**, cp37, source |
| seaborn 0.9.0 | py3, source |
| jupyter 1.0.0 | py3 (universal), source |

Every pinned package ships a CPython 3.6 wheel, and matplotlib's
`requires_python>=3.5` is satisfied. **The reference set is internally
consistent for CPython 3.6.7 on Ubuntu 18.04 LTS.**

### 2.3 Verification (c): no post-cutoff APIs in project source

V1 source is written to the 2019 API surface. Concretely, the following
conventions are enforced by review:

- Language level is CPython 3.6. f-strings are permitted (3.6). Dataclasses
  (3.7), the walrus operator (3.8), PEP 585 builtin generics (3.9),
  positional-only parameters (3.8), and `subprocess.run(capture_output=...)`
  (3.7) are **not** used.
- pandas usage stays within the 0.23 API. No `DataFrame.explode` (0.25),
  no `.convert_dtypes` (1.0), no nullable extension dtypes, no
  `pd.NamedAgg` (0.25), no `.attrs`.
- scikit-learn usage stays within the 0.20 API. Model selection is imported
  from `sklearn.model_selection`; `sklearn.metrics.mean_squared_error` is
  used **without** the `squared=` keyword (added in 0.22), so RMSE is
  computed as an explicit square root. No `sklearn.inspection`
  (0.22), no `HistGradientBoosting*` (0.21), no `plot_confusion_matrix`
  (0.22), no `sklearn.experimental` iterative imputation.
- The two modules written in this phase, `src/data/audit.py` and
  `scripts/preflight.py`, use the **standard library only** and are
  3.6-compatible by construction, so that they run on both the reference
  and the host interpreter.

---

## 3. Host execution environment (observed)

This is the machine on which the reconstruction is being carried out. It is
recorded as observed, not adjusted.

| Component | Observed on host |
|---|---|
| Operating system | Debian GNU/Linux 12 (bookworm) |
| Kernel | Linux 6.1.0-52-cloud-amd64, x86_64 |
| Interpreter | CPython 3.11.2 (`/usr/bin/python3`) |
| pip | 23.0.1 |
| numpy | 2.4.6 (system-wide; **not** the V1 pin) |
| pandas / scipy / scikit-learn / matplotlib / seaborn / jupyter | not installed |
| Other interpreters present | none (no 3.6–3.10, no pyenv, no conda) |
| Docker | binary present (20.10.24), **daemon not usable by this user** |

---

## 4. Verified incompatibility: the reference set cannot be installed here

This is the central compatibility finding of Phase 1. It was established by
running the installers, not by assumption.

### 4.1 No pinned binary distribution resolves on CPython 3.11

`pip download <pin> --only-binary=:all: --python-version 311 --platform
manylinux2014_x86_64 --no-deps` was executed for each pin:

| Package pin | Result on CPython 3.11 |
|---|---|
| numpy==1.15.4 | **not resolvable** — pip offers only 1.23.2 and newer |
| pandas==0.23.4 | **not resolvable** — pip offers only 1.5.0 and newer |
| scipy==1.1.0 | **not resolvable** — pip offers only 1.9.2 and newer |
| scikit-learn==0.20.0 | **not resolvable** — pip offers only 1.1.3 and newer |
| matplotlib==3.0.2 | **not resolvable** — pip offers only 3.6.0rc1 and newer |
| seaborn==0.9.0 | resolvable (pure-python wheel) |
| jupyter==1.0.0 | resolvable (universal wheel) |

The two that resolve are pure-Python metapackages; they are useless in
isolation because their runtime dependencies are exactly the five that do
not resolve.

### 4.2 Source build of the numeric core fails on CPython 3.11

A clean virtual environment was created on the host interpreter and
`pip install numpy==1.15.4` was attempted from source. It failed:

```
x86_64-linux-gnu-gcc ... -c numpy/core/src/multiarray/scalarapi.c ...
    failed with exit status 1
error: legacy-install-failure
```

The build emits deprecation errors against `PyUnicode_AS_UNICODE` and
`PyUnicode_GET_DATA_SIZE` from `/usr/include/python3.11/cpython/
unicodeobject.h` and then fails to compile the multiarray module. numpy
1.15.4 predates CPython 3.11 by roughly four years and cannot be compiled
against its C API. Because numpy is the base of the entire pinned stack,
every other numeric pin is transitively unbuildable here.

### 4.3 Isolation routes evaluated

The contract prefers a container or otherwise isolated historical
environment. Each available route was checked:

| Route | Status | Evidence |
|---|---|---|
| Docker container (e.g. `python:3.6.7-slim`) | **unavailable** | `docker ps` → `permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock` |
| Distribution package `python3.6` | **unavailable** | `apt-cache policy python3.6` → `Candidate: (none)` on Debian 12 |
| pyenv-built CPython 3.6.7 | **not attempted** | pyenv absent; host filesystem is at 98% capacity (~2.5 GB free), which is not a safe margin for a source interpreter build plus a full scientific stack |
| Host venv on CPython 3.11 | **rejected** | would require modernising every pin, i.e. abandoning the contract |

### 4.4 Resolution

Per the contract, the V1 source is **not** modernised to make it install.

- `requirements-v1-2019.txt` continues to describe the reference historical
  environment, unchanged.
- The host is formally recorded as a **non-conforming execution
  environment**.
- `scripts/preflight.py` reports this as an explicit
  `HOST-COMPATIBILITY WARNING`, never as a historical-environment `PASS`.
- No V1 modelling code will be executed until a conforming environment is
  available. Establishing one requires either Docker daemon access for this
  user, or disk headroom plus pyenv to build CPython 3.6.7. Both are
  environment-level actions outside this phase.

This is an open blocker, tracked in
[`PHASE_00_FOUNDATION_RECORD.md`](PHASE_00_FOUNDATION_RECORD.md).

---

## 5. Prohibited in V1

AirSense V1 must **not** use any of the following. Several postdate the
cutoff outright; the remainder are excluded because they fall outside the
declared classical-methods scope of V1.

**Deep learning frameworks:** TensorFlow, PyTorch, Keras, and any deep
neural network architecture (MLP, CNN, RNN, LSTM, GRU, attention).

**Gradient-boosting libraries:** XGBoost, LightGBM, CatBoost. XGBoost and
LightGBM did exist before the cutoff, so they are excluded on *scope*
grounds rather than availability grounds: V1's declared modelling family is
linear and tree/forest based. If either is ever introduced it must be
declared as a deliberate, separately justified historical extension, dated,
and reported apart from the M0–M3 comparison — never folded into it
silently.

**Foundation models and generative AI:** transformers, Hugging Face
libraries and hubs, LLM APIs of any vendor, foundation models, generative
AI, agent frameworks, and modern time-series foundation models.

**Modern interpretability:** SHAP, and any successor attribution library.
Feature importance in V1 is limited to what scikit-learn 0.20 itself
exposes (linear coefficients, tree/forest `feature_importances_`).

**Modern experiment and tuning infrastructure:** MLflow, Weights & Biases,
Optuna, Ray Tune, AutoML of any form, and modern experiment-tracking
frameworks.

**Cloud-managed AI services:** vendor-managed training, tuning, feature
store or inference services.

**Any API introduced after 2019-04-26**, including in packages that are
themselves permitted.

### 5.1 Permitted V1 modelling family

- **M0** — naive / statistical baseline
- **M1** — Linear Regression
- **M2** — Decision Tree Regression
- **M3** — Random Forest Regression

A later classical extension to **classification of pollution bands** is
permitted in principle, but is not part of the declared V1 comparison and is
not implemented in this phase.

---

## 6. Change control

Versions are not changed for convenience. Any deviation from
`requirements-v1-2019.txt` requires an entry here recording: what changed,
why the original pin was untenable, evidence that the replacement also
predates 2019-04-26, and what effect the change has on comparability.

**No deviations recorded.** The pinned set stands as written; the recorded
issue is with the *host*, not with the pins.
