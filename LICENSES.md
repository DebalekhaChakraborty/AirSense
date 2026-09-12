# Licensing

This repository contains three kinds of material under three different terms.
They are listed separately because conflating them would misstate what you may
do with the data.

---

## 1. Code — MIT

**Covers:** `src/`, `scripts/`, `tests/`, `configs/`, `Makefile`,
`pyproject.toml`, `.github/`

Full text: [`LICENSE`](LICENSE). Copyright (c) 2026 Debalekha Chakraborty.

`LICENSE` is the unmodified MIT text so that automated licence detection
identifies it correctly. This file carries the scope notes instead.

## 2. Documentation, manuscript and derived analysis content — CC BY 4.0

**Covers:** `docs/`, `manuscript/`, `artifacts/`, `figures/`, `results/`,
`README.md`

Licensed under [Creative Commons Attribution 4.0 International](
https://creativecommons.org/licenses/by/4.0/). You may share and adapt this
material, including commercially, with attribution.

**Caveat for the manuscript.** `manuscript/` is a draft prepared for journal
submission. A publisher may later require a copyright transfer or impose its
own licence on the published version. This CC BY 4.0 grant applies to the
draft as distributed here and does not bind, or predict, whatever terms a
future publisher applies.

## 3. The dataset — NOT covered by either licence above

**Covers:** `data/raw/`

> **The MIT licence on the code does not, and cannot, relicense the dataset.**

| | |
|---|---|
| Name | Beijing Multi-Site Air-Quality Data |
| Source | UCI Machine Learning Repository, dataset 501 |
| DOI | 10.24432/C5RK5G |
| Terms | CC BY 4.0 as presented by UCI |

The dataset is redistributed here under its own terms, unmodified and
read-only by convention, so that its SHA-256 digests can be verified against
the files this repository actually carries. **The authors of this repository do
not own this dataset and claim no rights over it.** Anyone reusing
`data/raw/` is bound by the dataset's terms, not by this repository's code
licence.

The derived layer `data/processed/` is not tracked in git. It is regenerated
deterministically from `data/raw/`, so it inherits the dataset's terms, not the
code licence.

## 4. Third-party content incorporated by reference

| What | Source | Terms |
|---|---|---|
| Station coordinates, used for descriptive diagnostics only | Wu, L., Xie, J. & Kang, K., *npj Urban Sustainability* **2**, 23 (2022), doi:10.1038/s42949-022-00070-0 | CC BY 4.0 — attribution required |

Coordinates were used for analysis only: never a model input, never a graph,
never a selection criterion, and acquired after every model was frozen.

---

## Summary

| Material | Terms | Owner |
|---|---|---|
| Code | MIT | repository author |
| Docs, manuscript, artifacts, figures, results | CC BY 4.0 | repository author |
| `data/raw/` dataset | CC BY 4.0 as presented by UCI | **UCI / original providers** |
| Station coordinates | CC BY 4.0 | Wu, Xie & Kang (2022) |
