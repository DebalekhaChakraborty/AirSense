# AirSense V2 — References

Two sections, and the distinction between them is binding. An entry moves from
NEEDS VERIFICATION to VERIFIED only when every required field has been checked
against an authoritative source. **No entry is upgraded silently, and no
bibliographic entry in this package was generated without verification.**

Required fields for a VERIFIED entry: Authors, Title, Venue, Year, DOI or
canonical publisher URL, and Verification source.

---

## VERIFIED

### [V1] Beijing Multi-Site Air-Quality Data (dataset)

- **Authors:** Not asserted here. The UCI distribution page is the citable
  artifact used by this study; the associated source publication has not been
  verified and is listed under NEEDS VERIFICATION as `N1`.
- **Title:** Beijing Multi-Site Air-Quality Data
- **Venue:** UCI Machine Learning Repository, dataset 501
- **Year:** Distribution accessed and hash-pinned by this study
- **DOI:** 10.24432/C5RK5G
- **Canonical URL:** https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- **Licence:** CC BY 4.0 as presented by UCI
- **Verification source:** Repository provenance record
  `docs/DATASET_PROVENANCE.md`, which pins the resolved download URL, the raw
  archive SHA-256 and the SHA-256 of all twelve station CSVs. The dataset
  identity used in the manuscript is verified against those digests rather than
  against a secondary description.
- **Used in manuscript:** Abstract; 3. Data and protocol; 16. Data availability.

### [V2] Wu, Xie & Kang (2022) — station coordinates

- **Authors:** Wu, L., Xie, J. & Kang, K.
- **Title:** Changing weekend effects of air pollutants in Beijing under 2020
  COVID-19 lockdown controls
- **Venue:** npj Urban Sustainability, volume 2, article 23
- **Year:** 2022
- **DOI:** 10.1038/s42949-022-00070-0
- **Licence:** CC BY 4.0
- **Verification source:** Repository audit
  `docs/PHASE7_COORDINATE_ACQUISITION_AUDIT.md`. All twelve station names were
  matched by exact case-sensitive comparison against Table 1 of the paper with
  no alias inference, all twelve latitude/longitude pairs agreed exactly with
  the published table, and the licence statement was confirmed.
- **Used in manuscript:** 8. Robustness results (attention-distance
  diagnostics); 16. Data availability.
- **Scope note:** Coordinates were used for descriptive diagnostics only. They
  were never a model input, never a graph, never a selection criterion, and
  were acquired after every model was frozen.

### [V3] Liu et al. (2024) — iTransformer

- **Authors:** Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L. & Long, M.
- **Title:** iTransformer: Inverted Transformers Are Effective for Time Series
  Forecasting
- **Venue:** International Conference on Learning Representations (ICLR),
  spotlight presentation (venue id `ICLR.cc/2024/Conference`)
- **Year:** 2024 (conference); arXiv preprint posted 2023
- **DOI:** 10.48550/arXiv.2310.06625
- **Canonical URLs:** https://arxiv.org/abs/2310.06625 ·
  https://openreview.net/forum?id=JePfAI8fah
- **Verification source:** Two authoritative pages, because neither alone
  carried every required field. The arXiv abstract page supplied the verbatim
  title and the complete seven-author list; it states **no venue**, so the
  conference could not be confirmed there. The OpenReview record supplied the
  venue string "ICLR 2024 spotlight" under venue id `ICLR.cc/2024/Conference`,
  forum `JePfAI8fah`. Neither field was taken from a search snippet or from the
  repository source comment.
- **Used in manuscript:** 2.2 Deep temporal forecasting; 4.4 iTransformer-style
  model.
- **Scope note:** `src/models/itransformer_forecaster.py` implements the
  inverted formulation as an **in-repository adaptation**, not a reproduction of
  the authors' implementation, and the manuscript says so wherever the model is
  described. The citation establishes the design this study adapted; it makes no
  claim of equivalence.

---

## NEEDS VERIFICATION

These are **not** citable in the manuscript in their current state. Each appears
in the manuscript only as a `[REF-NEEDED: …]` placeholder. Nothing below has
been checked against an authoritative source, and no author list, year, venue,
DOI or URL below should be treated as established.

| ID | What is needed | Required before upgrade | Manuscript location |
|---|---|---|---|
| N1 | Canonical source publication associated with the UCI Beijing Multi-Site dataset | Authors, title, venue, year, DOI; cross-check against publisher record | 16. Data availability |
| N2 | Authoritative review or cohort study establishing PM2.5 health associations | Authors, title, venue, year, DOI; effect sizes must match the cited claim | 1.1 |
| N3 | Recent survey of machine-learning PM2.5 forecasting | Authors, title, venue, year, DOI | 2.1 |
| N4 | Methodological critique of baseline reporting in air-quality forecasting | Authors, title, venue, year, DOI | 2.1 |
| N5 | Representative recurrent and temporal-convolutional forecasting architectures | Authors, title, venue, year, DOI for each | 2.2 |
| N7 | Study comparing deep forecasting models against simple baselines | Authors, title, venue, year, DOI | 2.2 |
| N8 | Representative spatiotemporal graph forecasting work | Authors, title, venue, year, DOI | 2.3 |
| N9 | Chronos-2 paper | Authors, title, venue, year; arXiv identifier recorded in the repository audit as `arXiv:2510.15821`, **not independently verified here** | 2.4 |
| N10 | TimesFM paper | Authors, title, venue, year, DOI | 2.4 |
| N11 | Moirai paper | Authors, title, venue, year, DOI | 2.4 |
| N12 | Work on data leakage in time-series evaluation | Authors, title, venue, year, DOI | 2.5 |
| N13 | Work on benchmark or pretraining contamination in foundation models | Authors, title, venue, year, DOI | 2.5 |

### Verification rule

A reference may not be upgraded from a search-result snippet. Each entry
requires the publisher record or an equivalently authoritative source, and the
verification source must be recorded alongside the entry. If verification
cannot be completed, the placeholder stays in the manuscript.
