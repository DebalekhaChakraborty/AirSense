# Phase 1 EDA Record — AirSense V1

Evidence record for the exploratory data analysis phase (Phase 3 of the
[research protocol](V1_RESEARCH_PROTOCOL.md); "Phase 01" in the record
sequence).

- **Recorded:** 2026-09-05 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE**
- **Analysis document:** [`EDA_ANALYSIS.md`](EDA_ANALYSIS.md)
- **Machine-readable summary:**
  [`artifacts/eda_summary.json`](../artifacts/eda_summary.json)

> No model was trained, no predictive metric was calculated, no train/test
> split was created, and no stored dataset was cleaned or modified.

---

## 1. Execution runtime

| Property | Value |
|---|---|
| **Interpreter path** | `/home/debalekha_chakraborty/AirSense/venv/bin/python` |
| **Python version** | CPython **3.6.7** |
| `sys.prefix` | `<repo>/venv` |
| Platform | Linux 6.1.0-52-cloud-amd64, x86_64, Debian 12 |
| Preflight | **READY**, exit code 0 |

**Path correction.** The task specified
`~/micromamba/envs/airsense-v1-2019/bin/python`. That path no longer exists:
the environment was relocated into the repository as `venv/` during Phase
0A, and both `~/micromamba` and the micromamba binary were deleted at the
same time. The in-repo `venv/bin/python` **is** the same verified runtime —
identical interpreter version and byte-identical package set — and is what
every command in this phase used. The host system interpreter (CPython
3.11.2) was used only to state that it was *not* used for V1 execution.

### Runtime verification

| Package | Required | Observed |
|---|---|---|
| numpy | 1.15.4 | 1.15.4 |
| pandas | 0.23.4 | 0.23.4 |
| scipy | 1.1.0 | 1.1.0 |
| scikit-learn | 0.20.0 | 0.20.0 |
| matplotlib | 3.0.2 | 3.0.2 |
| seaborn | 0.9.0 | 0.9.0 |
| jupyter | 1.0.0 | importable |

No runtime dependency was added, removed, upgraded or downgraded.
`requirements-v1-2019.txt` and `requirements-v1-2019-lock.txt` are
unmodified.

---

## 2. Input dataset

| Property | Value |
|---|---|
| Path | `data/raw/PRSA_data_2010.1.1-2014.12.31.csv` |
| **SHA-256 before analysis** | `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c` |
| **SHA-256 after analysis** | `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c` |
| **Match** | **YES — byte-identical** |
| File mode | `444` (read-only), unchanged |
| Rows / columns | 43,824 × 13 |
| Missing `pm2.5` | 2,067 |
| Duplicate rows | 0 |
| Schema | exact match to expected |

The frozen-input gate was run **before** any analysis and refused to
proceed on mismatch. `run_analysis()` additionally recomputes the digest at
the end of every run and raises `CRITICAL DATA INTEGRITY FAILURE` if it has
changed, so the guarantee is enforced in code rather than only by
convention.

---

## 3. Files created

**Code**

| Path | Purpose |
|---|---|
| `src/analysis/__init__.py` | package marker |
| `src/analysis/eda.py` | all EDA logic; 3.6.7-compatible; deterministic |
| `scripts/run_eda.py` | thin entry point |

**Numerical results** — `results/eda/` (15 tables)

`dataset_profile.csv`, `missing_values.csv`,
`pm25_descriptive_statistics.csv`, `pm25_by_year.csv`, `pm25_by_month.csv`,
`pm25_by_hour.csv`, `pm25_by_season.csv`,
`numerical_feature_summary.csv`, `pm25_by_wind_direction.csv`,
`pearson_correlation.csv`, `pm25_spearman_correlations.csv`,
`pm25_missingness_by_year.csv`, `pm25_missingness_by_month.csv`,
`pm25_missingness_by_hour.csv`, `time_series_integrity.csv`

**Figures** — `figures/` (9)

`eda_pm25_distribution.png`, `eda_pm25_yearly.png`,
`eda_pm25_monthly.png`, `eda_pm25_hourly.png`, `eda_pm25_seasonal.png`,
`eda_pm25_wind_direction.png`, `eda_correlation_heatmap.png`,
`eda_pm25_meteorology.png`, `eda_pm25_missingness.png`

**Other**

| Path | Purpose |
|---|---|
| `artifacts/eda_summary.json` | machine-readable summary |
| `notebooks/01_exploratory_data_analysis.ipynb` | presentation layer, 27 cells |
| `docs/EDA_ANALYSIS.md` | written analysis |
| `docs/PHASE_01_EDA_RECORD.md` | this record |

**Files modified:** `README.md` only (Phase 1 status and validated
high-level findings). No previously existing code, data or document was
otherwise altered.

---

## 4. Calculations performed

| Section | Content |
|---|---|
| Dataset overview | dimensions, dtypes, unique counts, observations per year, in-memory datetime construction |
| Data quality | missing counts and rates per column, duplicate counts by two definitions |
| PM2.5 descriptives | count, missing, mean, median, std, min, max, quartiles, IQR, p90/p95/p99, skewness, excess kurtosis |
| Temporal | count / mean / median / std by year, calendar month, hour of day |
| Seasonal | count / mean / median / std by meteorological season |
| Meteorological | count / missing / mean / median / std / min / quartiles / max for six variables; 20-bin summaries of PM2.5 against `TEMP`, `DEWP`, `PRES`, `Iws` |
| Wind direction | category counts, shares, and PM2.5 count / mean / median / std per level |
| Correlation | Pearson matrix (pairwise-complete) over 7 variables; Spearman ρ and p-value of PM2.5 against each of 6 meteorological variables |
| Missingness | missing-PM2.5 count and rate by year, month and hour |
| Time-series integrity | first/last timestamp, expected vs actual vs unique counts, duplicates, absent timestamps, largest gap, monotonicity, strict hourliness, validity |

**Statistical restraint.** No hypothesis testing was performed beyond the
Spearman coefficients, which were declared in advance as a robustness check
against the skew of PM2.5. scipy returns a p-value with each ρ, so those
values are recorded, but **no significance threshold was applied and no
selection decision was taken from them**. No broad or uncontrolled testing
was carried out.

---

## 5. Method notes

**Determinism by construction.** No random sampling is used anywhere,
including for visualisation. Meteorological relationships are shown as
binned summaries over *all* available observations rather than as a
subsampled scatter, so no seed, sample size or "statistics computed from the
full data, figure from a sample" caveat applies. No execution timestamp is
written into any EDA artifact.

**Historical API discipline.** seaborn's estimator plots (`barplot`,
`pointplot`) were deliberately avoided: their confidence intervals are
computed by bootstrap resampling, which would make figures non-deterministic
across runs. Plain matplotlib is used for all bar and line figures, and
seaborn only for the correlation heatmap. No API introduced after
2019-04-26 is used.

**In-memory only.** The `datetime` and `season` columns are constructed on a
copy and never persisted. `dropna` is applied in memory for individual
statistics. `data/processed/` remains empty.

---

## 6. Validation executed

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Branch | `git branch --show-current` | `legacy` |
| 2 | Git status | `git status` (read-only) | recorded in §8 |
| 3 | Preflight | `venv/bin/python scripts/preflight.py` | **READY**, exit 0 |
| 4 | Frozen-input gate | SHA-256 + dimensions + schema | all MATCH |
| 5 | EDA run 1 | `venv/bin/python scripts/run_eda.py` | exit 0, 15 tables + 9 figures |
| 6 | EDA run 2 | same | exit 0 |
| 7 | Reproducibility | SHA-256 of all artifacts, run 1 vs run 2 | **all byte-identical** |
| 8 | Output existence | 15 CSV + 9 PNG + 1 JSON present | all present |
| 9 | Non-empty | `find -type f -empty` | none (except intentional `.gitkeep`) |
| 10 | JSON parse | `json.load` on `eda_summary.json` | OK |
| 11 | Notebook validate | `nbformat.validate` | OK, 27 cells |
| 12 | Notebook execute | `jupyter nbconvert --execute` | 13 code cells, **0 errors** |
| 13 | Python 3.6 compile | `py_compile` on all project `.py` | all OK |
| 14 | Raw hash gate | recompute SHA-256 | **MATCH** |

### Reproducibility result

The EDA was run twice and every artifact hashed:

| Artifact class | Count | Run 1 vs run 2 |
|---|---:|---|
| CSV tables | 15 | **byte-identical** |
| JSON summary | 1 | **byte-identical** |
| PNG figures | 9 | **byte-identical** |

Figures reproduce byte-for-byte because `savefig` is given an explicit
`metadata={"Software": "AirSense V1"}`, which suppresses the matplotlib
version string that would otherwise be embedded in the PNG `tEXt` chunk. No
weakening of the check was needed — the stronger byte-level guarantee holds.

### Notebook policy

The notebook is committed **without executed outputs**: 0 embedded outputs,
all `execution_count` values `None`. It was nonetheless executed end-to-end
in the historical environment to prove it works, and the executed copy was
written to a scratch location and discarded. This keeps the tracked file
deterministic and small (11,641 bytes) while still verifying it runs. The
script-generated artifacts remain authoritative; the notebook reads them
rather than recomputing them.

---

## 7. Warnings

**W1 — Interpreter path in the task instructions was stale.** Resolved as
described in §1. Not a defect in the environment.

**W2 — Missingness is strongly year-dependent.** 8.31% of 2011 targets are
missing against 0.94% of 2013; 91.2% of all missing targets fall in
2010–2012. This is a genuine analytical finding with a direct consequence
for the chronological split, and it is carried into
[`EDA_ANALYSIS.md`](EDA_ANALYSIS.md) rather than being treated as a defect.

**W3 — `pm2.5` minimum is 0 µg/m³.** Whether these are genuine readings or a
reporting floor cannot be determined from this file. Flagged for Phase 4;
nothing was removed.

**W4 — Host disk pressure.** The filesystem was between 99% and 100% used
throughout. Not caused by this project — the whole repository including the
environment is ~530 MB. All EDA artifacts together are under 0.5 MB.

**No unresolved blockers.**

---

## 8. Git status at completion

Branch `legacy`. **Nothing staged.** All changes left unstaged for manual
handling, as instructed.

No `git add`, `git commit`, `git push`, `git reset`, `git rebase`,
`git merge`, `git cherry-pick`, `git tag`, or branch creation/deletion was
performed. Every git command used was read-only inspection
(`status`, `diff`, `log`, `branch --show-current`, `check-ignore`).

---

## 9. Explicit confirmations

- **No stored dataset cleaning occurred.** `data/raw/` is byte-identical;
  `data/processed/` is empty.
- **No imputation occurred.** All 2,067 missing PM2.5 values remain missing.
- **No row was removed from any stored dataset.** `dropna` was applied in
  memory only, for individual statistics.
- **No train/test or validation split was created.** No split boundary is
  defined anywhere in this phase.
- **No model was trained.** No estimator was instantiated or fitted.
  `src/models/` still contains only an empty `__init__.py`.
- **No predictive metric was calculated.** No MAE, RMSE or model R² exists.
- **No winsorizing, standardisation, normalisation or encoding** was
  applied. `cbwd` is untouched.
- **No lag or rolling feature was engineered.**
- **No raw dataset byte was changed** — digest verified before and after.
- **No runtime dependency was changed**, and no post-2019 dependency was
  introduced.
- **No git write operation was performed.**

---

## 10. Status

Phase 3 (exploratory data analysis) is **complete**.

The next phase is Phase 4 (data cleaning and preprocessing). The decisions
handed forward are listed under *Implications for Preprocessing* in
[`EDA_ANALYSIS.md`](EDA_ANALYSIS.md). Nothing from that list has been acted
on.

Awaiting instruction.
