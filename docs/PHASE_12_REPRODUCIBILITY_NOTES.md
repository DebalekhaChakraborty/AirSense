# AirSense V2 — Phase 12A Reproducibility Notes

**Classification: retrospective release metadata, written after the study
completed. Nothing here modifies, supersedes or reinterprets any Phase 0–11
scientific result.**

These notes record what the repository can and cannot promise about
reconstructing the study environment. They exist because a static sweep found
one dependency used by the figure pipeline that no frozen record pins.

---

## 1. The modelling runtime is historically frozen

Everything that produced a number is pinned, twice — direct pins and a
resolved closure:

| File | Scope |
|---|---|
| `requirements-v2-core.txt` | direct pins, Phase 3 |
| `requirements-v2-core-lock.txt` | full resolved closure: lightgbm 4.7.0, narwhals 2.25.0, numpy 2.4.6, scipy 1.17.1 |
| `requirements-v2-neural.txt` | direct pins, Phase 4 |
| `requirements-v2-neural-lock.txt` | full closure incl. torch 2.14.0+cpu, safetensors 0.8.0 |

**These four files are frozen historical environment records and were not
modified by Phase 12.** Every model fit, every prediction and every metric in
Phases 3–9 came from the environment they describe.

## 2. Pillow is a rendering dependency, and its historical version is not recorded

`src/visualization/render.py:15` imports `PIL` (Pillow). Eight scripts reach it:

```
scripts/build_phase3_report.py     scripts/build_phase8_figures.py
scripts/build_phase4_report.py     scripts/build_phase9_figures.py
scripts/build_phase5_report.py     scripts/run_training_eda.py
scripts/build_phase6_report.py     scripts/validate_phase1.py
```

Two facts, stated precisely because the distinction matters:

- **Pillow appears in none of the four requirements files**, and a search of
  all 78 tracked artifacts finds no Pillow version in any environment block.
  The exact version used to render the committed PNGs is therefore **not
  recorded anywhere in the repository**.
- The interpreter that currently has Pillow available is the **system**
  `python3`, at version **12.3.0**. `.venv-v2` has no Pillow at all — figure
  scripts run under system Python, the modelling scripts under `.venv-v2`.

**12.3.0 is what is installed now. It is not evidence of what rendered the
figures, and this document does not claim it is.** Inventing a historical
version would be worse than recording the gap.

A consequence worth being explicit about: **do not claim that an arbitrary
fresh environment reproduces the committed PNG bytes bit-for-bit.** PNG output
depends on the Pillow version, its zlib build and font rasterisation. A
rebuild may be visually identical and byte-different.

One further consequence: `scripts/validate_phase1.py` imports the renderer, so
that validator inherits an unpinned dependency.

## 3. The frozen PNG hashes remain the authoritative figure evidence

All **50 tracked figures have their SHA-256 recorded in a `v2_*_manifest.json`
(100%)**. Those digests, not any re-render, are the historical record of what
the study produced. A regenerated figure that does not match its frozen digest
is a new artifact, not a reproduction — and under the study's own append-only
discipline it would need its own freeze rather than replacing the original.

The scientific content of the figures is independently reproducible: every
plotted value traces to a frozen CSV or manifest. It is the *byte-level PNG
encoding* that is not guaranteed, not the data.

## 4. What a consolidated release requirements file could and could not say

A single `requirements-release.txt` is **proposed, not created**, because it
cannot be written without deciding how to represent Pillow, and every option
carries a caveat:

| Option | Honest? | Problem |
|---|---|---|
| Pin `Pillow==12.3.0` | **No** | States a version the study never recorded using. Falsifies the historical claim. |
| Pin a floor, `Pillow>=10` | Yes | Truthful but does not reproduce PNG bytes; must say so. |
| List Pillow unpinned, annotated | Yes | Most honest; weakest as a lockfile. |
| Omit Pillow | No | Reproduces the present gap. |

**Recommended, pending human approval:** a new `requirements-release.txt` that
re-states the four frozen pins verbatim and adds Pillow as an explicitly
*unpinned, rendering-only, version-not-historically-recorded* dependency, with
a header stating it is a convenience file for re-running figure code and is
**not** a historical environment record. The existing lock files stay
authoritative and untouched.

## 5. Summary

| Claim | Status |
|---|---|
| Modelling runtime reconstructible from frozen locks | **Yes** |
| Model artifacts and predictions hash-pinned | **Yes** |
| Figure *data* reproducible from frozen tables | **Yes** |
| Figure *PNG bytes* reproducible in a fresh environment | **Not guaranteed** — Pillow version unrecorded |
| Frozen PNG digests authoritative | **Yes**, 50/50 pinned |
| Historical Pillow version | **Not recorded; not inferred here** |
