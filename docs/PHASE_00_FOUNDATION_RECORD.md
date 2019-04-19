# Phase 0 Foundation Record — AirSense V1

Evidence record for the foundation / preflight phase.

- **Recorded:** 2026-09-05 (reconstruction date — **not** a historical date)
- **Phase:** 0 — foundation, covering Phase 1 (environment reconstruction)
  and Phase 2 (dataset provenance and audit)
- **Historical cutoff enforced:** 2019-04-26

---

## 1. Repository state before this work

| Property | Observed |
|---|---|
| Path | `/home/debalekha_chakraborty/AirSense` |
| Contents | **empty** — 0 files, 0 subdirectories, no hidden entries |
| Git | **not a git repository** (`git status` → `fatal: not a git repository`) |
| Existing commits | none |
| Python/runtime files | none |
| Dataset files | none |
| Conflicting project | none |

No pre-existing work was present, so nothing was deleted, overwritten or
displaced. No git history existed, so none was rewritten. A fresh repository
was initialised on branch `main`.

---

## 2. What was created

### Documentation
| Path | Purpose |
|---|---|
| `README.md` | Project overview with explicit provenance notice |
| `docs/HISTORICAL_COMPATIBILITY.md` | Cutoff contract, verification evidence, prohibited technologies |
| `docs/DATASET_AUDIT.md` | Human-readable dataset audit |
| `docs/RESEARCH_QUESTION.md` | Pre-registered question, models M0–M3, metrics |
| `docs/FEATURE_POLICY.md` | Feature classes A–D, leakage prohibitions |
| `docs/V1_RESEARCH_PROTOCOL.md` | 12-phase protocol, temporal-leakage rule |
| `docs/PHASE_00_FOUNDATION_RECORD.md` | This record |
| `data/README.md` | Dataset provenance, license, acquisition |

### Code
| Path | Purpose |
|---|---|
| `src/data/audit.py` | Read-only CSV audit; stdlib only; CPython 3.6-compatible |
| `scripts/audit_dataset.py` | Writes `artifacts/data_audit.json` |
| `scripts/preflight.py` | Read-only PASS/WARN/FAIL validation |
| `src/__init__.py`, `src/{data,features,models,visualization}/__init__.py` | Package markers (empty) |

### Configuration and data
| Path | Purpose |
|---|---|
| `requirements-v1-2019.txt` | Frozen 2019 dependency pins |
| `.gitignore` | Python/Jupyter artefacts; raw data deliberately **not** ignored |
| `data/raw/PRSA_data_2010.1.1-2014.12.31.csv` | Original UCI dataset, mode `444` |
| `artifacts/data_audit.json` | Machine-readable audit record |

### Directories
`data/{raw,processed}`, `notebooks`, `src/{data,features,models,visualization}`,
`scripts`, `artifacts`, `results`, `figures`, `docs`, `tests` — 15 in total.
Empty directories carry `.gitkeep`.

**Files modified:** none. Every file listed above is new; nothing
pre-existing was edited, because nothing pre-existed.

---

## 3. Environment observed (host)

| Component | Observed |
|---|---|
| OS | Debian GNU/Linux 12 (bookworm) |
| Kernel | Linux 6.1.0-52-cloud-amd64, x86_64 |
| Interpreter | CPython 3.11.2 (`/usr/bin/python3`) |
| pip | 23.0.1 |
| numpy | 2.4.6 (system-wide) |
| pandas, scipy, scikit-learn, matplotlib, seaborn, jupyter | not installed |
| Other interpreters | none (no 3.6–3.10; no pyenv; no conda) |
| Docker | binary 20.10.24 present; **daemon socket permission denied** |
| Disk (`/`) | 99G total, ~2.5G free (98% used) |
| git | repository initialised, branch `main` |

---

## 4. Historical target environment

| Component | Reference pin | Verified released | Pre-cutoff? |
|---|---|---|---|
| OS | Ubuntu 18.04 LTS | 2018-04-26 | yes |
| Python | CPython 3.6.7 | 2018-10-20 | yes |
| numpy | 1.15.4 | 2018-11-04 | yes |
| pandas | 0.23.4 | 2018-08-04 | yes |
| scipy | 1.1.0 | 2018-05-05 | yes |
| scikit-learn | 0.20.0 | 2018-09-26 | yes |
| matplotlib | 3.0.2 | 2018-11-11 | yes |
| seaborn | 0.9.0 | 2018-07-16 | yes |
| jupyter | 1.0.0 | 2015-08-12 | yes |

**Verification sources.** Package release dates were read from the PyPI JSON
API during this reconstruction. The CPython 3.6.7 date came from the
python.org downloads API (`release_date: 2018-10-20T12:00:00Z`). The Ubuntu
18.04 date came from the Launchpad API milestone `ubuntu-18.04`
(`date_targeted: 2018-04-26`); Launchpad exposes no `date_released` value for
that series, so the targeted date is what is recorded.

Every pinned version publishes a CPython 3.6 distribution, so the reference
set is internally consistent for CPython 3.6.7.

---

## 5. Dataset status

| Property | Value |
|---|---|
| Dataset | UCI Beijing PM2.5 (ID 381, DOI `10.24432/C5JS49`) |
| Donated to UCI | 2017-01-18 (historical) |
| **Retrieved** | **2026-09-05 (reconstruction)** |
| Source URL | `https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv` |
| HTTP status | 200 |
| Form | original UCI-distributed CSV — not a Kaggle or cleaned derivative |
| Stored at | `data/raw/PRSA_data_2010.1.1-2014.12.31.csv`, mode `444` |
| License | CC BY 4.0 as currently presented by UCI |

The modern `ucimlrepo` convenience API was deliberately **not** used; it
postdates the cutoff. Retrieval was a plain HTTP fetch.

### 5.1 Dataset hash and dimensions

| Property | Value |
|---|---|
| **SHA-256** | `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c` |
| Byte size | 2,010,494 |
| Rows (excl. header) | 43,824 |
| Columns | 13 |
| Ragged rows | 0 |
| Header | `No, year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir` |
| Schema vs expected | exact match, including order |
| Target `pm2.5` | present |
| Missing `pm2.5` | 2,067 (4.7166%) |
| Missing, all other columns | 0 |
| Usable target rows | 41,757 (95.2834%) |
| Duplicates (all columns) | 0 |
| Duplicates (excl. index) | 0 |

Row count matches the UCI-published instance count of 43,824, and matches
1,826 days × 24 hours exactly — no hourly timestamp is absent.

**No cleaning was performed. No rows were removed. No missing value was
imputed.**

---

## 6. Checks executed

| # | Check | Command / method | Result |
|---|---|---|---|
| 1 | Repository state inspection | `ls -la`, `find`, `git status` | empty, non-repo — confirmed |
| 2 | Host environment survey | `/etc/os-release`, `uname -a`, `python3 --version`, import probes | recorded |
| 3 | Network reachability | `curl -I` against UCI | HTTP 200 |
| 4 | Pin release dates vs cutoff | PyPI JSON API, all 7 pins | all pre-cutoff |
| 5 | Pin interpreter compatibility | PyPI distribution tags | all publish cp36 |
| 6 | cp311 wheel availability | `pip download --only-binary=:all: --python-version 311` × 7 | 5 of 7 unresolvable |
| 7 | Real install attempt | `pip install numpy==1.15.4` in a clean venv | **failed to compile** |
| 8 | Container route | `docker ps` | permission denied |
| 9 | 3.6 interpreter availability | `apt-cache policy python3.6`, `command -v pyenv` | unavailable |
| 10 | Dataset download | `curl -L` from UCI legacy path | 200, 2,010,494 bytes |
| 11 | Dataset provenance | UCI dataset page 381 fetched and read | recorded |
| 12 | Dataset audit | `python scripts/audit_dataset.py` | wrote `artifacts/data_audit.json` |
| 13 | Python syntax compilation | `python -m py_compile` on all new `.py` | all compile |
| 14 | Preflight | `python scripts/preflight.py` | exit 1 — HOST-COMPATIBILITY WARNING |
| 15 | Preflight read-only proof | digest + mtime compared before/after | unchanged |
| 16 | Project tree inspection | `find` | 15/15 dirs, 8/8 docs |
| 17 | Git status | `git status` | recorded |

---

## 7. Checks passed

**Passed (8 of 10 preflight checks):** project structure (15/15 directories,
8/8 documents); raw dataset present and read-only; dataset schema exact
match; dataset integrity (digest matches the recorded audit); dataset
dimensions (43,824 × 13, 0 ragged); missing-value summary recorded without
removal; duplicate rows 0; target column present with 41,757 usable values.

Also passed outside preflight: all pins verified pre-cutoff; all pins
verified cp36-compatible; all new Python files compile; the audit is
reproducible and byte-stable.

---

## 8. Warnings

**W1 — Runtime (WARN).** Host CPython 3.11.2 is newer than the reference
3.6.7. Reported explicitly as a non-conforming execution environment, never
as a historical-environment PASS.

**W2 — Dependencies (WARN).** The pinned 2019 stack is not installed. numpy
is present at 2.4.6, which is not the pin; the remaining six are absent. The
pins were **not** modernised to resolve this.

**W3 — Disk pressure (informational).** The host filesystem is at 98%
capacity (~2.5 GB free), which is an unsafe margin for building a source
interpreter plus a full scientific stack. This is why the pyenv route was not
attempted.

**W4 — License label provenance (informational).** The CC BY 4.0 label is as
displayed by UCI today, not verified 2019 license text. Recorded as such in
`data/README.md`.

---

## 9. Unresolved blockers

**B1 — The frozen 2019 environment cannot be reconstructed on this host.**

> **STATUS: RESOLVED in Phase 0A (2026-09-05).** A conforming CPython 3.6.7
> environment was built in user space with micromamba and verified: all
> seven direct pins at their frozen versions, the transitive closure locked
> to pre-cutoff releases, 57 distributions with **0** postdating the cutoff,
> and preflight reporting **READY** with exit code 0. No pin was modernised.
> Evidence: [`PHASE_00A_RUNTIME_RECORD.md`](PHASE_00A_RUNTIME_RECORD.md).
>
> The findings below remain accurate for the host *system* interpreter and
> are retained unchanged as the original Phase 0 record.

Verified, not assumed:
- No pinned numeric release resolves for CPython 3.11 — pip offers only
  numpy ≥ 1.23.2, pandas ≥ 1.5.0, scipy ≥ 1.9.2, scikit-learn ≥ 1.1.3,
  matplotlib ≥ 3.6.0rc1.
- A source build of numpy 1.15.4 against CPython 3.11 fails with
  `error: legacy-install-failure`, the compiler failing on
  `numpy/core/src/multiarray/scalarapi.c` amid deprecated-`PyUnicode_*`
  errors from the 3.11 headers.
- Docker is unusable: `permission denied` on
  `unix:///var/run/docker.sock`.
- Debian 12 offers no `python3.6` package; no pyenv is installed; disk
  headroom is insufficient for a source interpreter build.

**Impact (as assessed in Phase 0).** Phases 3–12 cannot be executed in a
period-authentic environment on this host. The foundation itself is
unaffected — it was built with the standard library only.

**Impact (revised after Phase 0A).** Phases 3–12 are unblocked. They must be
run with `venv/bin/python` (in-repo), never with the
host system interpreter, which still reports
`HOST-COMPATIBILITY WARNING`.

**Resolution options** (environment-level, outside this phase): grant this
user Docker daemon access and run `python:3.6`-based image; or free disk
space and install pyenv to build CPython 3.6.7; or move the project to a host
that already provides a 3.6 interpreter.

**Not an acceptable resolution:** modernising `requirements-v1-2019.txt`.
That would silently abandon the contract.

---

## 10. Explicit confirmations

- **No model was trained.** `src/models/` contains only an empty
  `__init__.py`. No estimator was instantiated or fitted.
- **No metric was computed.** No MAE, RMSE or R² value exists anywhere in
  this repository.
- **No result was fabricated.** `results/` and `figures/` are empty. No
  placeholder, illustrative or expected number was written anywhere.
- **No exploratory analysis was performed.** No descriptive statistics,
  distributions, correlations or figures were produced.
- **No cleaning or feature engineering was performed.** `data/processed/` is
  empty; `src/features/` contains only an empty `__init__.py`.
- **No data was modified.** The raw CSV is byte-identical to the downloaded
  file, stored read-only, and its digest was re-verified after the preflight
  run.
- **No historical commit, timestamp, experiment date, certificate or
  evidence was fabricated.** The git repository was initialised now, with no
  backdating. Every date in this repository is labelled as either historical
  (verifiable externally) or reconstruction (2026-09-05).
- **No post-cutoff AI technology was introduced into V1.** No deep learning
  framework, gradient-boosting library, transformer, LLM API, foundation
  model, agent framework, SHAP, MLflow, W&B, Optuna or AutoML tool is
  referenced, imported or depended upon. The only code written uses the
  Python standard library.

---

## 11. Phase gate

Phase 0 is **complete**, with blocker **B1** open.

The next phase (3 — exploratory data analysis) requires pandas, matplotlib
and seaborn at their pinned versions and therefore **cannot begin until B1 is
resolved**. Proceeding on the host's modern stack would produce results that
are not period-authentic, and would contradict the contract this phase
exists to establish.

Awaiting instruction.
