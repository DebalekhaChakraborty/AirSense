# Phase 2 Cleaning Record — AirSense V1

**Research-protocol mapping: Protocol Phase 4 — Data Cleaning and
Preprocessing.**

> Record-sequence numbering (`PHASE_02_`) is independent of protocol phase
> numbering. Earlier records were **not** renamed to align the two; the
> mapping is stated explicitly instead. Previous records:
> `PHASE_00_FOUNDATION_RECORD.md` (protocol Phases 1–2),
> `PHASE_00A_RUNTIME_RECORD.md` (runtime recovery),
> `PHASE_01_EDA_RECORD.md` (protocol Phase 3).

- **Recorded:** 2026-09-05 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE**
- **Policy document:** [`CLEANING_AND_SPLIT_POLICY.md`](CLEANING_AND_SPLIT_POLICY.md)
- **Split manifest:** [`artifacts/split_manifest.json`](../artifacts/split_manifest.json)

> No model was trained, no prediction was produced, and no predictive metric
> was calculated.

---

## 1. Execution runtime

| Property | Value |
|---|---|
| **Interpreter path** | `/home/debalekha_chakraborty/AirSense/venv/bin/python` |
| **Python version** | CPython **3.6.7** (`sys.version_info[:3] == (3, 6, 7)`) |
| `sys.prefix` | `<repo>/venv` |
| Preflight | **READY**, exit code 0 |

Frozen pins verified before any work: numpy 1.15.4, pandas 0.23.4,
scipy 1.1.0, scikit-learn 0.20.0, matplotlib 3.0.2, seaborn 0.9.0, jupyter
importable — all MATCH.

The host system interpreter (CPython 3.11.2) was used only to state that it
was **not** used for AirSense V1 execution. No runtime dependency was added,
removed, upgraded or downgraded; `requirements-v1-2019.txt` and
`requirements-v1-2019-lock.txt` are unmodified.

The implementation imports only pandas, numpy and the standard library. No
modelling library is imported anywhere in this phase.

---

## 2. Input integrity

| Property | Expected | Observed | Result |
|---|---|---|---|
| SHA-256 **before** | `4127f868…6d27f1c` | `4127f868…6d27f1c` | MATCH |
| Rows | 43,824 | 43,824 | MATCH |
| Columns | 13 | 13 | MATCH |
| Missing `pm2.5` | 2,067 | 2,067 | MATCH |
| Duplicate rows | 0 | 0 | MATCH |
| Schema | exact | exact | MATCH |
| First timestamp | 2010-01-01 00:00 | 2010-01-01 00:00 | MATCH |
| Last timestamp | 2014-12-31 23:00 | 2014-12-31 23:00 | MATCH |
| Unique timestamps | 43,824 | 43,824 | MATCH |
| Absent hourly slots | 0 | 0 | MATCH |

Full digest:
`4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`

`run_preparation()` re-verifies the digest before starting and again at the
end, raising `IntegrityError` on any change, so the guarantee is enforced in
code rather than by convention.

---

## 3. Cleaning rule

**A row is eligible for supervised modelling if and only if `pm2.5` is
non-missing.**

Ineligible rows are excluded from the **supervised derived datasets only**.
Nothing is deleted from `data/raw/`, which remains the authoritative
43,824-row timeline.

- **No target was imputed**, interpolated, forward-filled, backward-filled,
  mean/median-filled or synthesised. Rationale in
  [`CLEANING_AND_SPLIT_POLICY.md`](CLEANING_AND_SPLIT_POLICY.md).
- **Zero-valued PM2.5 retained** — 2 observations.
- **High-concentration PM2.5 retained** — maximum 994 µg/m³. Nothing
  clipped, winsorized or capped.
- No scaling, encoding, transformation, feature engineering, lag, rolling
  statistic or shuffling.

**Verbatim write-back.** The raw file is read a second time as text
(`dtype=str`, `na_filter=False`) and retained rows are written from that
view, so each output line reproduces its original representation exactly
rather than passing through a float round-trip that would render `129` as
`129.0`. The two reads are cross-checked: the typed view and the text view
must agree about which targets are missing, or the run aborts.

---

## 4. Split boundaries — frozen

| Partition | Start | End |
|---|---|---|
| `development_train` | 2010-01-01 00:00:00 | 2012-12-31 23:00:00 |
| `validation` | 2013-01-01 00:00:00 | 2013-12-31 23:00:00 |
| `test` (locked) | 2014-01-01 00:00:00 | 2014-12-31 23:00:00 |

Frozen **before** any model was fitted and before any predictive metric
existed. Membership is assigned from calendar timestamp on the full
43,824-row timeline **before** any target-based exclusion, enforced by the
order of operations in `src/data/cleaning.py`.

---

## 5. Counts

Declared in advance and verified in code; the pipeline aborts rather than
continuing on any mismatch.

| Partition | Raw rows | Missing targets | Supervised rows | Missing rate |
|---|---:|---:|---:|---:|
| `development_train` | 26,304 | 1,886 | 24,418 | 7.170012% |
| `validation` | 8,760 | 82 | 8,678 | 0.936073% |
| `test` | 8,760 | 99 | 8,661 | 1.130137% |
| **overall** | **43,824** | **2,067** | **41,757** | **4.716594%** |

All twelve declared values matched exactly.

**Label-missingness shift:** development train / validation = **7.66×**;
development train / test = **6.34×**. This is a target-availability shift,
not predictor covariate shift — every predictor is complete in every
partition.

Identifier and timestamp boundaries:

| Partition | Raw `No` | Supervised `No` | Supervised timestamps |
|---|---|---|---|
| `development_train` | 1–26,304 | 25–26,304 | 2010-01-02 00:00 → 2012-12-31 23:00 |
| `validation` | 26,305–35,064 | 26,305–35,064 | 2013-01-01 00:00 → 2013-12-31 23:00 |
| `test` | 35,065–43,824 | 35,065–43,824 | 2014-01-01 00:00 → 2014-12-31 23:00 |

The development-train supervised series begins at `No` 25 because the first
24 hours of 2010-01-01 all carry a missing target in the raw file.

---

## 6. Artifact digests

| File | SHA-256 | Rows |
|---|---|---:|
| `data/processed/airsense_supervised.csv` | `c86415f427e49a535e45e45b14ce069533a17b8d3f36a9755f5a575064e76076` | 41,757 |
| `data/processed/airsense_train_2010_2012.csv` | `0d5a14ecf079a661e837759c2334cbca40ac9223d39f0af28df39370d289d140` | 24,418 |
| `data/processed/airsense_validation_2013.csv` | `41f5c7a733daa6f6617ceb18c0a53b1a1056616482257a50992145565c425f82` | 8,678 |
| `data/processed/airsense_test_2014.csv` | `9f273de4ae9243c269d6f211229e5d017341e50ee8e7093add3a04dedd903702` | 8,661 |
| `artifacts/target_missing_exclusions.csv` | `df93dda088ad35d563ae97f7f6f7e9597b551ae10b63fedb7a25cfe2ef9b84c9` | 2,067 |
| `artifacts/split_manifest.json` | `79678b839a9d0e178b681daba56482c948b27d68e308d5fcd7d2c40580550f33` | — |
| `results/preprocessing/partition_summary.csv` | `00e30347ec0159190918ba6b9235ee9418996ea5d2f36aa38000680fed5ed2ab` | 4 |
| `results/preprocessing/data_reconciliation.csv` | `dbb8816eb1ffd187f4b3ba0b1525949d983d5fcdd38a3603c6c90d9054590cb5` | 8 |

The processed CSVs keep the original 13-column schema and original column
names. No timestamp column is persisted in the modelable files — timestamp
information lives in the manifests, per the phase specification. `No` is
retained for traceability and is **not** thereby a legitimate predictor.

---

## 7. Round-trip and reconciliation

Verified in code on every run:

| Check | Result |
|---|---|
| Retained rows match raw source verbatim (all 4 files) | PASS |
| Schema unchanged in every processed file | PASS |
| Chronological ordering preserved (`No` strictly increasing) | PASS |
| No retained row carries a missing target | PASS |
| Every excluded row genuinely has a missing raw target | PASS — 2,067 |
| supervised (41,757) + excluded (2,067) = 43,824 | PASS |
| train + validation + test = 41,757 = supervised file | PASS |
| Partition overlap | 0 rows |
| Partitions exactly cover the supervised dataset | True |
| Retained target range | 0 to 994 µg/m³ |
| Zero-valued targets retained | 2 |

---

## 8. Validation commands executed

| # | Check | Command | Result |
|---|---|---|---|
| 1 | Runtime gate | `venv/bin/python --version` + pin verification | 3.6.7, all pins MATCH |
| 2 | Preflight | `venv/bin/python scripts/preflight.py` | **READY**, exit 0 |
| 3 | Raw input gate | digest, dimensions, schema, timestamps | all MATCH |
| 4 | Phase 4 run 1 | `venv/bin/python scripts/prepare_data.py` | exit 0 |
| 5 | Phase 4 run 2 | same | exit 0 |
| 6 | Determinism | SHA-256 of all 8 artifacts, run 1 vs run 2 | **all byte-identical** |
| 7 | Verbatim fidelity | raw line vs processed line spot-check | identical text |
| 8 | Exclusion breakdown | manifest grouped by partition | 1,886 / 82 / 99 |
| 9 | Python 3.6 compile | `venv/bin/python -m py_compile` on all project `.py` | all OK |
| 10 | Raw hash gate | recompute SHA-256 | **MATCH** |
| 11 | Git status | read-only inspection | nothing staged |

### Deterministic rerun result

The pipeline was run twice and every artifact hashed:

| Artifact class | Count | Run 1 vs run 2 |
|---|---:|---|
| Processed CSVs | 4 | **byte-identical** |
| Exclusion manifest | 1 | **byte-identical** |
| Split manifest (JSON) | 1 | **byte-identical** |
| Preprocessing audit CSVs | 2 | **byte-identical** |

No execution timestamp is written into any Phase 4 artifact, which is what
makes the byte-level guarantee achievable.

### Python 3.6 compatibility

All project `.py` files compile under CPython 3.6.7, including the two added
here. No post-3.6 syntax and no post-cutoff pandas or scikit-learn API is
used. `src/data/cleaning.py` requires only pandas, numpy and the standard
library.

---

## 9. Raw dataset integrity

| | |
|---|---|
| **SHA-256 before** | `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c` |
| **SHA-256 after** | `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c` |
| **Match** | **YES — byte-identical** |
| File mode | `444`, unchanged |

---

## 10. Files created and modified

**Created**

| Path | Purpose |
|---|---|
| `src/data/cleaning.py` | cleaning and split implementation |
| `scripts/prepare_data.py` | thin entry point |
| `data/processed/airsense_supervised.csv` | 41,757 supervised rows |
| `data/processed/airsense_train_2010_2012.csv` | 24,418 rows |
| `data/processed/airsense_validation_2013.csv` | 8,678 rows |
| `data/processed/airsense_test_2014.csv` | 8,661 rows — **locked** |
| `artifacts/target_missing_exclusions.csv` | 2,067-row audit trail |
| `artifacts/split_manifest.json` | frozen split record |
| `results/preprocessing/partition_summary.csv` | per-partition audit |
| `results/preprocessing/data_reconciliation.csv` | reconciliation arithmetic |
| `docs/CLEANING_AND_SPLIT_POLICY.md` | policy document |
| `docs/PHASE_02_CLEANING_RECORD.md` | this record |

**Modified**

| Path | Change |
|---|---|
| `README.md` | Phase 4 status and high-level summary |
| `docs/V1_RESEARCH_PROTOCOL.md` | **administrative status only** — the phase table's Status column and the stale runtime-blocker sentence. No scientific rule altered. |

**Not modified:** `FEATURE_POLICY.md` (no Phase 5 decision has been taken),
`RESEARCH_QUESTION.md`, `EDA_ANALYSIS.md`, `DATASET_AUDIT.md`,
`requirements-v1-2019.txt`, `requirements-v1-2019-lock.txt`, and every byte
of `data/raw/`.

### Protocol amendment check

`docs/V1_RESEARCH_PROTOCOL.md` was edited for execution status only. **None
of the following was altered:** the M0–M3 model list, the primary metrics
(MAE / RMSE / R²), the decision metric, the chronological-test rule, the
random-split rule, the leakage rule, or the model reporting rule. §7
(Protocol amendments) still records no amendments, because none was made.

---

## 11. Warnings

**W1 — Label-missingness asymmetry between partitions.** Development-train
missing-target rate is 7.66× validation and 6.34× test. This is a genuine
property of the data, documented in
[`CLEANING_AND_SPLIT_POLICY.md`](CLEANING_AND_SPLIT_POLICY.md) and to be
restated when results are reported. It is not a defect and was not
"corrected".

**W2 — Two zero-valued PM2.5 observations retained.** Not established as
either genuine or artefactual. Carried to Phase 11 error analysis.

**W3 — One implementation defect found and fixed during the run.** The first
execution aborted in round-trip verification with a pandas `TypeError`: the
excluded-row check indexed the text view (where `No` is a string) with
identifiers taken from the typed view (where `No` is an integer). Replaced
with a vectorised comparison over the text view. The failure occurred
*before* any verification claim was made — the run aborted rather than
reporting success, which is the behaviour intended. No artifact from the
failed run survived; all current artifacts come from complete successful
runs.

**No unresolved blockers.**

---

## 12. Test-set quarantine — now in force

From the completion of this phase until Protocol Phase 10, the 2014 target
is quarantined. Full terms in
[`CLEANING_AND_SPLIT_POLICY.md`](CLEANING_AND_SPLIT_POLICY.md).

No predictive performance has been computed on 2014, or on any partition.

---

## 13. Explicit confirmations

- **Missing targets were excluded only from supervised derived data.**
  `data/raw/` is byte-identical and still holds all 43,824 rows.
- **No target was imputed**, interpolated, forward-filled, backward-filled,
  averaged or synthesised.
- **Zero PM2.5 values were retained** — 2 observations.
- **High-concentration PM2.5 values were retained** — maximum 994 µg/m³;
  nothing clipped, winsorized or capped.
- **Split membership was assigned by calendar time before target
  exclusion.**
- **2014 is now the locked final test partition.**
- **No feature encoding or scaling occurred.** `cbwd` is untouched; no
  one-hot, ordinal, sine/cosine, season, weekend, lag or rolling feature
  exists.
- **No model was trained.** `src/models/` still contains only an empty
  `__init__.py`; no estimator was imported or instantiated.
- **No prediction was produced.**
- **No MAE, RMSE or R² was calculated.**
- **No raw dataset byte changed** — digest verified before and after.
- **No runtime dependency changed.**
- **No git write operation was performed** — no `add`, `commit`, `push`,
  `reset`, `rebase`, `merge`, `cherry-pick`, `tag`, or branch
  creation/deletion. Every git command used was read-only inspection.
- **Git history was not rewritten.**

---

## 14. Status

Protocol Phase 4 is **complete**. All legitimate changes are left
**unstaged**.

The next phase is Protocol Phase 5 (feature preparation), governed by
[`FEATURE_POLICY.md`](FEATURE_POLICY.md), which is unchanged. Nothing from
its open-decision list has been acted on.

Awaiting instruction.
