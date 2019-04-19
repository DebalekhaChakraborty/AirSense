# Phase 0A Runtime Record — AirSense V1

Evidence record for the execution-environment reconstruction that resolves
blocker **B1** from
[`PHASE_00_FOUNDATION_RECORD.md`](PHASE_00_FOUNDATION_RECORD.md) §9.

- **Recorded:** 2026-09-05 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Historical cutoff enforced:** 2019-04-26
- **Outcome:** **VERIFIED READY**
- **Machine-readable evidence:**
  [`artifacts/runtime_snapshot.json`](../artifacts/runtime_snapshot.json)

> No exploratory analysis, cleaning, feature engineering, model training or
> metric calculation was performed in this phase.

---

## 1. Initial blocker

Recorded in Phase 0 and **re-verified at the start of this phase rather than
assumed**:

| Condition | Phase 0 | Rechecked now | Still true? |
|---|---|---|---|
| Host interpreter | CPython 3.11.2 | CPython 3.11.2 | yes |
| Any CPython 3.6 present | none | none | yes |
| `python3.6` in Debian 12 apt | unavailable | unavailable | yes |
| Docker daemon usable | no — permission denied | no — `docker ps` still fails | yes |
| Podman | absent | absent | yes |
| conda / mamba / micromamba | absent | absent | yes |
| pyenv | absent | absent | yes |
| Free disk on `/` | ~2.5 GB | **2.4 GB** | yes, still tight |

**One material change since Phase 0:** `uv 0.9.2` is now present at
`~/.local/bin/uv`. It was evaluated and rejected for this purpose —
`uv python list --all-versions` offers no CPython 3.6 build at all (its
managed distributions begin above 3.6).

The blocker was therefore confirmed genuine before any attempt to solve it.

---

## 2. Routes evaluated

Worked through in the preferred order, least invasive first.

| # | Route | Outcome | Evidence |
|---|---|---|---|
| A | Existing CPython 3.6.7 installation | **not available** | no `python3.6` on `PATH` or on disk |
| B | Existing conda/mamba environment | **not available** | no `~/miniconda3`, `~/anaconda3`, `~/.conda`, `~/micromamba`; no env roots at all |
| C | **User-space conda-style install** | **CHOSEN** | conda-forge publishes `python 3.6.7`, confirmed via the anaconda.org API |
| D | pyenv source build of CPython 3.6.7 | **rejected** | pyenv absent; a source interpreter build plus a full scientific stack is not safe against 2.4 GB free |
| E | Container runtime | **not available** | Docker daemon socket permission denied; podman absent |
| F | uv-managed interpreter | **not viable** | uv publishes no CPython 3.6 build |

No `sudo` was used. No system Python was altered. No OS package was
installed, replaced or removed. No unrelated user file was deleted and no
system cleanup command was run.

---

## 3. Chosen isolation method

**micromamba 2.9.0**, a single statically-linked user-space binary
(18 MB), installed to `~/.local/bin/micromamba`. It was chosen over a full
Miniconda installation specifically because of the disk constraint: it needs
no base environment.

| Property | Value |
|---|---|
| Environment name | `airsense-v1-2019` |
| Environment path | `~/micromamba/envs/airsense-v1-2019` |
| Root prefix | `~/micromamba` |
| Channel | `conda-forge` (with `--override-channels`) |
| Environment size | 527 MB |
| Free disk after | 1.8 GB |

The environment lives **outside** the tracked project source, so nothing
about it enters the repository except the recorded evidence.

---

## 4. Interpreter obtained

| Property | Value |
|---|---|
| Implementation | CPython |
| **Version** | **3.6.7** |
| `sys.version_info[:3]` | `(3, 6, 7)` |
| Path | `~/micromamba/envs/airsense-v1-2019/bin/python` |
| conda build string | `h357f687_1008_cpython` |

**Exact match to the reference interpreter.** Not 3.6.8, not 3.6.9, not
3.6.15 — 3.6.7, as the contract requires.

---

## 5. Bootstrap tooling versus V1 runtime

The contract distinguishes these, and so does this record.

**(A) Bootstrap tooling** — used only to *obtain* the historical runtime.
Not part of V1 modelling methodology, and permitted to be modern:

| Tool | Version | Role |
|---|---|---|
| micromamba | 2.9.0 | acquires CPython 3.6.7 |
| conda-forge channel | — | source of the interpreter |
| Host CPython 3.11.2 | 3.11.2 | ran the PyPI date-audit queries only |

**(B) V1 runtime** — what actually executes AirSense code. Frozen to
pre-cutoff releases.

The conda-forge environment shipped pip 21.3.1, setuptools 58.0.4 and wheel
0.37.1 — all post-cutoff. These were replaced with period-consistent
releases contemporaneous with CPython 3.6.7 (October–December 2018):

| Tool | Installed | Released | Pre-cutoff |
|---|---|---|---|
| pip | **18.1** | 2018-10-05 | yes |
| setuptools | **40.6.3** | 2018-12-11 | yes |
| wheel | **0.32.3** | 2018-11-19 | yes |

`pip==18.1` is the candidate named in the contract, and it was **verified to
work against the modern package index** before being relied upon: a
`pip download six==1.12.0` probe succeeded. No fallback to a newer pip was
required.

*Note:* the conda metadata still lists its original pip/setuptools/wheel
records, because the pip-installed copies shadow them in `site-packages`.
The versions that actually import and execute are 18.1 / 40.6.3 / 0.32.3,
and those are what
[`artifacts/runtime_snapshot.json`](../artifacts/runtime_snapshot.json)
records.

---

## 6. Direct dependency versions

Installed from `requirements-v1-2019.txt`, **which was not edited**.

| Package | Required | Installed | Result |
|---|---|---|---|
| numpy | 1.15.4 | 1.15.4 | MATCH |
| pandas | 0.23.4 | 0.23.4 | MATCH |
| scipy | 1.1.0 | 1.1.0 | MATCH |
| scikit-learn | 0.20.0 | 0.20.0 | MATCH |
| matplotlib | 3.0.2 | 3.0.2 | MATCH |
| seaborn | 0.9.0 | 0.9.0 | MATCH |
| jupyter | 1.0.0 | 1.0.0 | importable |

No direct dependency was modernised.

---

## 7. Transitive-dependency audit

**This was the substantive finding of the phase.**

### 7.1 The unlocked install was not historical

Installing the seven direct pins alone produced **69 distributions, of which
57 postdated the cutoff**. Every version was checked against its PyPI
release date. Representative examples:

| Package | Resolved unlocked | Released |
|---|---|---|
| pytz | 2026.3.post1 | 2026-07-25 |
| six | 1.17.0 | 2024-12-04 |
| python-dateutil | 2.9.0.post0 | 2024-03-01 |
| tornado | 6.1 | 2020-10-30 |
| ipython | 7.16.3 | 2022-01-19 |
| notebook | 6.4.10 | 2022-03-15 |
| Pygments | 2.14.0 | 2023-01-01 |

**Correct direct pins do not imply a historical runtime.** Had this audit
been skipped, V1 would have executed on a 2026 support stack while appearing
period-authentic on the surface.

### 7.2 Four packages did not exist in 2019 at all

These have **no** pre-cutoff release. They are required only by post-2019
versions of other packages, and are absent from the locked environment:

| Package | Pulled in by |
|---|---|
| `argon2-cffi-bindings` | notebook 6.x |
| `comm` | ipywidgets 8.x |
| `jupyterlab-pygments` | nbconvert 6.x |
| `nbclient` | nbconvert 6.x |

### 7.3 Resolution

`requirements-v1-2019-lock.txt` was created, pinning the **complete
transitive closure** to pre-cutoff releases. The environment was then torn
down and rebuilt from the lock.

Selection rule: *the newest release published on or before 2019-04-26 that
is CPython 3.6-compatible **and** satisfies the constraints declared by the
other pinned packages.* Where newest-by-date would break a declared
constraint, the constraint wins. Two such overrides were needed and are
documented inline in the lock file:

| Package | Newest pre-cutoff by date | Used instead | Reason |
|---|---|---|---|
| `prompt-toolkit` | 1.0.16 (2019-04-12) | **2.0.9** (2019-02-19) | ipython 7.5.0 declares `prompt_toolkit>=2.0.0,<2.1.0` |
| `parso` | 0.4.0 (2019-04-05) | **0.3.4** (2019-02-12) | jedi 0.13.3 declares `parso>=0.3.0,<0.4.0` |

One pin sits exactly on the boundary: `pyrsistent==0.15.1`, released
**2019-04-26**, the cutoff date itself. "On or before" admits it; it is
flagged here and in the lock file so the judgement is visible rather than
buried.

### 7.4 Result after locking

| Measure | Before lock | After lock |
|---|---|---|
| Distributions installed | 69 | 54 |
| Pre-cutoff | 12 | **54** |
| **Post-cutoff** | **57** | **0** |
| Unknown | 0 | 0 |

`pip check` reports **"No broken requirements found."**

Adding the three bootstrap tools (pip 18.1, setuptools 40.6.3, wheel
0.32.3 — all pre-cutoff), the full inventory is **57 distributions, none
postdating 2019-04-26**.

`requirements-v1-2019.txt` was **not modified**. The lock file is additive.

---

## 8. Smoke tests

Executed in the 3.6.7 environment. No AirSense data was touched and no model
was fitted.

| Library | Operation | Result |
|---|---|---|
| numpy 1.15.4 | build array, compute mean | shape `(4,)`, mean 2.50 |
| pandas 0.23.4 | build DataFrame, `describe()` | shape `(3, 2)`, 8 summary rows |
| scipy 1.1.0 | `import scipy.stats` | normal pdf at 0 = 0.3989 |
| scikit-learn 0.20.0 | import `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, MAE/MSE/R², `TimeSeriesSplit` | all imported, **none fitted** |
| matplotlib 3.0.2 | create and close a figure (Agg) | in-memory only, nothing saved |
| seaborn 0.9.0 | import | OK |
| jupyter 1.0.0 | import; CLI; kernelspec | `jupyter --version` 4.4.0; `python3` kernel registered |

**A contract claim was verified, not merely asserted.**
`HISTORICAL_COMPATIBILITY.md` §2.3 states that scikit-learn 0.20's
`mean_squared_error` has no `squared=` keyword, so RMSE must be computed as
an explicit square root. Introspection confirms the signature is
`(y_true, y_pred, sample_weight, multioutput)` — `squared` is absent.

---

## 9. Python 3.6 compilation and post-cutoff API scan

`python -m py_compile` under **CPython 3.6.7** on every project `.py` file:

```
OK   scripts/audit_dataset.py      OK   src/data/__init__.py
OK   scripts/preflight.py          OK   src/data/audit.py
OK   scripts/runtime_snapshot.py   OK   src/features/__init__.py
OK   src/__init__.py               OK   src/models/__init__.py
                                   OK   src/visualization/__init__.py
```

All compile. Source scan for post-cutoff syntax and APIs:

| Checked for | Found |
|---|---|
| `dataclasses` (3.7) | none |
| walrus operator `:=` (3.8) | none |
| builtin generics `list[…]`, `dict[…]` (3.9) | none |
| `subprocess.run(capture_output=…)` (3.7) | none |
| pandas post-0.23.4 (`explode`, `convert_dtypes`, `NamedAgg`, `attrs`, `pd.NA`) | none |
| sklearn post-0.20.0 (`inspection`, `HistGradientBoosting*`, `plot_confusion_matrix`, `experimental`, `squared=`) | none |

---

## 10. Dataset integrity recheck

| Property | Expected | Observed | Result |
|---|---|---|---|
| SHA-256 | `4127f868…6d27f1c` | `4127f868…6d27f1c` | MATCH |
| Rows | 43,824 | 43,824 | MATCH |
| Columns | 13 | 13 | MATCH |
| Missing `pm2.5` | 2,067 | 2,067 | MATCH |
| Duplicates (excl. index) | 0 | 0 | MATCH |
| Schema | exact | exact | MATCH |

Full digest:
`4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`

**No raw dataset byte was modified.** The file remains mode `444`.

---

## 11. Preflight

Run with the 3.6.7 interpreter:

```
Runtime .................................. PASS
Dependencies ............................. PASS
Project structure ........................ PASS
Raw dataset .............................. PASS
Dataset schema ........................... PASS
Dataset integrity ........................ PASS
Dataset dimensions ....................... PASS
Missing-value summary .................... PASS
Duplicate rows ........................... PASS
Target column ............................ PASS

Overall: READY
```

**Exit code: 0.**

The runtime line reads PASS only because the interpreter is genuinely
3.6.7. On the host interpreter the same script still reports
`HOST-COMPATIBILITY WARNING` with exit 1 — the check was not weakened to
produce a green result.

---

## 12. Residual differences from the Ubuntu 18.04 reference

Recorded honestly. The Python layer is period-exact; the layer beneath it is
not, and cannot be made so without root access.

| Aspect | Reference | Actual | Assessment |
|---|---|---|---|
| Distribution | Ubuntu 18.04 LTS | Debian 12 (bookworm) | different distro |
| Kernel | ~4.15 | 6.1.0-52-cloud-amd64 | modern |
| System glibc | 2.27 | 2.36 | modern; manylinux wheels are built against older glibc and run correctly |
| CPython 3.6.7 build | 2018 upstream build | conda-forge rebuild `h357f687_1008` | **version exact, build modern** |
| OpenSSL | 1.1.0/1.1.1 era | 1.1.1w (2023) | modern rebuild, supplied by the env |
| libsqlite / ncurses / readline / tk | 2018-era | 3.46.0 / 6.6 / 8.3 / 8.6.13 | modern rebuilds supplied by the env |
| ca-certificates | 2018 bundle | 2026.7.22 | modern; required for TLS to the package index |

**Interpretation.** Everything that determines *numerical and API behaviour*
of V1 — the interpreter version and all 57 Python distributions — is
period-exact and pre-cutoff. What remains modern is the C-library substrate
that conda-forge rebuilds for current systems, plus the host kernel. These
affect neither the scikit-learn/numpy/pandas APIs available nor the
algorithms V1 may use.

The honest characterisation is: **a period-exact Python environment running
on a modern substrate**, not a bit-for-bit 2018 Ubuntu machine. A true
2018 substrate would require a container or a VM, and Docker is still
unavailable to this account.

This residual difference is recorded rather than resolved, and it does not
block Phase 3.

---

## 13. Files created and modified

**Created**

| Path | Purpose |
|---|---|
| `requirements-v1-2019-lock.txt` | pre-cutoff transitive closure, 54 pins |
| `scripts/runtime_snapshot.py` | runtime evidence recorder, stdlib-only, 3.6-compatible |
| `artifacts/runtime_snapshot.json` | machine-readable runtime evidence |
| `docs/PHASE_00A_RUNTIME_RECORD.md` | this record |

**Modified**

| Path | Change |
|---|---|
| `docs/HISTORICAL_COMPATIBILITY.md` | §3 host table and §4 blocker status updated to reflect the resolved environment |
| `docs/PHASE_00_FOUNDATION_RECORD.md` | §9 blocker B1 marked resolved, with a pointer here |

**Not modified:** `requirements-v1-2019.txt`, `scripts/preflight.py`,
`src/data/audit.py`, `scripts/audit_dataset.py`, `README.md`,
`data/README.md`, `docs/DATASET_AUDIT.md`, `docs/RESEARCH_QUESTION.md`,
`docs/FEATURE_POLICY.md`, `docs/V1_RESEARCH_PROTOCOL.md`, and every byte of
`data/raw/`.

---

## 14. Reproducing the environment

```sh
# bootstrap tooling (not part of the V1 runtime)
micromamba create -y -n airsense-v1-2019 -c conda-forge --override-channels \
  "python=3.6.7" pip

PY=~/micromamba/envs/airsense-v1-2019/bin/python

# period-consistent packaging tooling
"$PY" -m pip install --no-cache-dir pip==18.1 setuptools==40.6.3 wheel==0.32.3

# the V1 runtime, transitively locked to pre-cutoff releases
"$PY" -m pip install --no-cache-dir -r requirements-v1-2019-lock.txt

# verify
"$PY" scripts/preflight.py          # expect: Overall: READY, exit 0
"$PY" scripts/runtime_snapshot.py   # rewrites artifacts/runtime_snapshot.json
```

---

## 15. Status

**Runtime gate: VERIFIED READY.** Blocker B1 is resolved.

Phase 3 (exploratory data analysis) is now unblocked but **was not started**.
No EDA, no cleaning, no split, no feature engineering, no model training and
no metric calculation was performed in this phase.

Awaiting instruction.
