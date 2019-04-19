# Dataset Audit — AirSense V1

Phase 2 record of the raw dataset as it exists in this repository. Its
purpose is to make it verifiable at any later point that the raw inputs have
not changed.

- **Audit generated (UTC):** 2026-09-05T21:27:43Z — timestamp of this
  reconstruction run, **not** a historical date.
- **Audit tool:** [`src/data/audit.py`](../src/data/audit.py), Python
  standard library only.
- **Machine-readable record:**
  [`artifacts/data_audit.json`](../artifacts/data_audit.json)
- **Cleaning performed:** none. **Rows removed:** 0.

---

## 1. File identity

| Property | Value |
|---|---|
| Filename | `PRSA_data_2010.1.1-2014.12.31.csv` |
| Path | `data/raw/PRSA_data_2010.1.1-2014.12.31.csv` |
| Byte size | 2,010,494 |
| **SHA-256** | `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c` |
| File mode | `444` (read-only) |
| Source | UCI ML Repository, dataset 381 — see [`../data/README.md`](../data/README.md) |
| Retrieved | 2026-09-05 |

Verify at any time:

```sh
sha256sum data/raw/PRSA_data_2010.1.1-2014.12.31.csv
```

A digest other than the one above means the raw input is not the file this
project was audited against.

---

## 2. Dimensions

| Property | Value |
|---|---|
| Data rows (excluding header) | **43,824** |
| Columns | **13** |
| Ragged rows (field count ≠ 13) | 0 |
| Header present | yes |

The row count matches the UCI-published instance count of 43,824 exactly.

It also matches the period exactly: 2010-01-01 through 2014-12-31 is 1,826
days, and 1,826 × 24 = 43,824 hourly slots. **No hourly timestamp is
absent from the file.**

---

## 3. Columns

Observed header, in file order:

```
No, year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir
```

**Schema check: PASS.** The observed header matches the expected UCI schema
exactly, in both membership and order. No columns missing, none unexpected.

**Target column `pm2.5`: present.**

---

## 4. Missing values per column

Counted against the missing tokens `NA`, `` (empty), `NaN`, `nan`, `N/A`,
`null`. The raw file uses `NA`.

| Column | Missing | % of 43,824 |
|---|---:|---:|
| `No` | 0 | 0.0000% |
| `year` | 0 | 0.0000% |
| `month` | 0 | 0.0000% |
| `day` | 0 | 0.0000% |
| `hour` | 0 | 0.0000% |
| **`pm2.5`** | **2,067** | **4.7166%** |
| `DEWP` | 0 | 0.0000% |
| `TEMP` | 0 | 0.0000% |
| `PRES` | 0 | 0.0000% |
| `cbwd` | 0 | 0.0000% |
| `Iws` | 0 | 0.0000% |
| `Is` | 0 | 0.0000% |
| `Ir` | 0 | 0.0000% |

**Summary.** Missingness is confined entirely to the target variable. All
eleven predictor columns and the ID column are fully populated. 41,757 rows
(95.2834%) carry a usable `pm2.5` value.

**These values have not been removed or imputed.** How to handle missing
targets is a Phase 4 decision, deliberately deferred so it is made
explicitly. Note for that phase: because missingness affects only the
target, no predictor imputation is required for the V1 regression task, and
the missing hours are not randomly scattered in time — their temporal
clustering must be examined in Phase 3 before any handling rule is fixed.

---

## 5. Duplicate rows

| Measure | Count |
|---|---:|
| Duplicate rows, all 13 columns compared | 0 |
| Duplicate rows, `No` index column excluded | 0 |

Duplicates are counted twice deliberately. The `No` column is a unique
running index, so comparing whole rows would guarantee zero duplicates and
tell us nothing. Excluding `No` and still finding **0** duplicates is the
meaningful result: no observation is genuinely repeated.

---

## 6. Audit status

| Check | Result |
|---|---|
| File present | PASS |
| SHA-256 recorded | PASS |
| Row count matches UCI published count (43,824) | PASS |
| Column count = 13 | PASS |
| Schema matches expected order | PASS |
| Target column `pm2.5` present | PASS |
| Ragged rows = 0 | PASS |
| Duplicates (excluding index) = 0 | PASS |
| Missing values recorded, not removed | PASS |
| Raw file unmodified | PASS |

**Dataset audit: PASS.** The raw dataset is present, schema-conforming,
complete in its temporal coverage, free of duplicates, and recorded in a way
that permits later verification.

---

## 7. What this audit does not do

- It does **not** clean, filter, impute or transform anything.
- It does **not** compute descriptive statistics, distributions or
  correlations. Those belong to Phase 3 (exploratory data analysis).
- It does **not** split the data or define any train/test boundary.
- It does **not** train or evaluate any model.

To regenerate:

```sh
python scripts/audit_dataset.py    # rewrites artifacts/data_audit.json
python scripts/preflight.py        # read-only verification
```
