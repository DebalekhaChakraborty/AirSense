# AirSense V2 — Phase 0 Record

**Foundation, research blueprint and dataset audit.**
**Executed 2026-09-06 on branch `master`, starting from
`09942928eef31808c27e621ae26c7ff2c82ed3ed` ("Start V2 on an empty master
branch").**

No model was fitted. No prediction was generated. No validation or test metric
was computed. No PM2.5 value statistic was computed in any partition. No heavy
dependency was installed. The `legacy` branch was not modified.

---

## 1. Branch gate

`git branch --show-current` returned `master`; HEAD matched the declared
starting point `09942928…`; the working tree was clean and the only tracked
file was `.gitignore`, confirming the intentionally empty V2 branch. Work began
only after that gate passed.

Two V1 side effects were removed from the working directory before scaffolding:
`src/__pycache__/` and `src/data/__pycache__/`, containing `cpython-36` byte
code left by an earlier V1 verifier run. Both were untracked and gitignored.

## 2. What was created

```
README.md
configs/study.json
docs/  V2_RESEARCH_BLUEPRINT.md, V2_RESEARCH_PROTOCOL.md,
       DATASET_PROVENANCE.md, FORECAST_SEMANTICS.md,
       MODEL_LANDSCAPE_2026.md, ENVIRONMENT_PLAN.md,
       PHASE_00_V2_FOUNDATION_RECORD.md
scripts/  acquire_dataset.py, audit_dataset.py, validate_foundation.py
src/  __init__.py, data/__init__.py, data/audit.py
artifacts/  raw_dataset_manifest.json, dataset_audit.json,
            v2_foundation_manifest.json
results/dataset_audit/  five CSV tables
data/raw/  official archive, inner archive, 12 station CSVs (read-only)
```

Empty planned directories (`data/processed/`, `figures/`, `notebooks/`,
`tests/`, `src/features/`, `src/models/`, `src/evaluation/`,
`src/visualization/`) carry a `.gitkeep` so the declared structure survives.
**No model implementation was written.**

## 3. Dataset acquisition

Downloaded once from the official UCI source. The acquisition script initially
**refused** the archive: its first structural check found members that were not
station CSVs. Inspection showed the official archive is nested and carries
extra files, including two — `data.csv` and `test.csv` — that are **stock-price
series, not air-quality data**. The script was corrected to encode the verified
real structure, and both files are now recorded as auxiliary members and
excluded. See [`DATASET_PROVENANCE.md`](DATASET_PROVENANCE.md) §3.

That the strict check fired on first contact is the intended behaviour: an
audit that silently accepted the archive would have carried unrelated data into
the study.

## 4. Audit findings

All verified from the downloaded files, not from documentation:

- 12 stations, 420,768 rows, 2013-03-01 00:00 → 2017-02-28 23:00 — all three
  match UCI's documented figures.
- Identical 18-column schema in every file; station column agrees with filename
  everywhere.
- **Timestamp grid perfect**: 35,064 hourly rows per station, 0 missing
  timestamps, 0 duplicates, 0 rows outside the grid, 0 duplicated rows, 0
  malformed rows.
- PM2.5 missing 8,739 (2.077%); CO worst at 4.920%; meteorology below 0.1%.
- Station-level PM2.5 missingness ranges 1.09% (Wanliu) to 2.72% (Huairou).
- No negative values in non-negative columns, no non-finite values, no sentinel
  codes; covariate ranges physically plausible.
- **Nothing was repaired.** No imputation, no cleaning, no coercion.

## 5. Target discipline

No mean, median, quantile, threshold, distribution or correlation of PM2.5 was
computed in **any** partition — not even training. The audit reports the target
by presence and absence only, and `artifacts/dataset_audit.json` records
`"target_value_statistics_computed": false`. Covariates received an ordinary
range audit, since they are inputs rather than the sealed target.

The severe threshold is **defined** (pooled training-period P95) but
deliberately **not computed**, so the audit does not become target EDA before
the policy is reviewed.

## 6. Determinism

`scripts/audit_dataset.py` was run twice; all six machine-readable outputs were
byte-identical. Artefacts are sorted, contain no wall-clock timestamp and no
filesystem mtime, and hash file content rather than metadata.

## 7. Hardware and environment

Audited, nothing installed. Debian 12, kernel 6.1, Intel Xeon @ 2.20 GHz,
20 logical cores, 31 GiB RAM, CPython 3.11.2, uv 0.9.2.

**Two blockers:** there is **no GPU** of any kind, and the root filesystem has
**1.9 GB free (99% full)** — less than a minimal PyTorch environment needs, let
alone foundation-model checkpoints. Phase 0 completed within the constraint by
using the standard library only. Details and options:
[`ENVIRONMENT_PLAN.md`](ENVIRONMENT_PLAN.md).

## 8. Model landscape

Audited against official repositories and model cards on 2026-09-06, with
unverifiable claims marked `UNVERIFIED` rather than asserted. Headline
findings: **TimesFM 3.0 (Aug 2026) ships non-commercial weights** while 2.5
remains Apache-2.0; **Chronos-2 (120M, Apache-2.0) explicitly supports CPU
inference** and native covariates, which matters on GPU-less hardware; Moirai's
latest is 2.0-R-small (Aug 2025); THUML's MIT-licensed Time-Series-Library
covers both PatchTST and iTransformer and targets Python 3.11. The GIFT-Eval
leaderboard could not be read directly, so ranking claims are recorded as
vendor-sourced. **No model was selected.** See
[`MODEL_LANDSCAPE_2026.md`](MODEL_LANDSCAPE_2026.md).

## 9. Frozen in Phase 0

Task and target; horizons 1/6/12/24; the forecast-origin information contract;
target-timestamp partition assignment; the three chronological boundaries;
regimes R0–R3; hypotheses H1–H5; the primary metric *form*; the severe
threshold *definition*; the baseline family B0–B3; raw dataset identity;
target-discipline rules; preprocessing principles.

## 10. Deliberately not frozen

Missingness and imputation policy; the eligible-row mask; context lengths; the
numeric severe threshold; the model roster; the V2 Python environment and its
pins; Track-B spatial design and the held-out station; whether a random-split
secondary comparison is worth running.

## 11. Git

No write operation was performed: no `git add`, `git commit`, `git push`,
reset, rebase, merge, tag, branch creation or history rewrite. All Phase-0
changes were left unstaged for manual review. Inspection was read-only.
