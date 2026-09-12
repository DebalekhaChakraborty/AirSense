# AirSense V2 — Related Work Citation Needs

Every `[REF-NEEDED: …]` placeholder in the manuscript, what evidence would
discharge it, and why the claim needs external support at all. **No placeholder
may be replaced by a fabricated entry.** If verification cannot be completed,
the placeholder remains visible in the submitted manuscript and the
corresponding sentence is softened or removed.

---

## Section 1.1 — Motivation

**`[REF-NEEDED: authoritative review or cohort study establishing PM2.5 health
associations, with effect sizes]`**

- **Claim supported:** that PM2.5 is associated with cardiovascular and
  respiratory morbidity in dense urban populations.
- **Required:** a peer-reviewed review or cohort study with authors, title,
  venue, year and DOI, verified against the publisher record.
- **Risk if unfilled:** this is the only epidemiological claim in the
  manuscript. It must not be supported by a broad unsourced number. If no
  verified citation is obtained, the sentence should be reduced to a
  non-quantitative statement of motivation.

## Section 2.1 — Classical and machine-learning air-quality forecasting

**`[REF-NEEDED: recent survey or review of machine-learning PM2.5 forecasting,
covering feature-engineering approaches]`** — establishes that engineered lag
and rolling-window features over pollutant and meteorological series are a
standard approach, so this study's `B3` family is positioned rather than
presented as new.

**`[REF-NEEDED: methodological critique of baseline reporting in air-quality
forecasting]`** — supports the claim that naive references are frequently
omitted or under-specified. This claim motivates a core design decision of this
study and should not rest on assertion. If no verified critique is found, the
sentence must be rewritten as a statement about this study's own design rather
than about the field.

## Section 2.2 — Deep temporal forecasting

**`[REF-NEEDED: representative recurrent and temporal-convolutional forecasting
architectures]`** — positions the GRU and TCN families.

**RESOLVED — now cited as [V3].** Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S.,
Ma, L. & Long, M., *iTransformer: Inverted Transformers Are Effective for Time
Series Forecasting*, ICLR 2024 (spotlight), arXiv:2310.06625,
DOI 10.48550/arXiv.2310.06625.

Verified against two authoritative pages: the arXiv abstract page for the
verbatim title and the full author list, and the OpenReview record for the
venue, which arXiv does not state. The repository source comment in
`src/models/itransformer_forecaster.py` named this paper all along, but a code
comment is not a verification source under this study's citation policy, so it
was treated as a lead and checked independently.

The manuscript continues to state, in both 2.2 and 4.4, that the implementation
is an in-repository adaptation and not an official reference implementation.

**`[REF-NEEDED: study comparing deep forecasting models against simple
baselines]`** — supports the claim that added capacity does not always
translate into gains. This study's own development results are consistent with
that, but the general claim needs external support.

## Section 2.3 — Spatial and multi-station forecasting

**`[REF-NEEDED: representative spatiotemporal graph forecasting work, e.g. a
graph-convolutional or diffusion-convolutional recurrent architecture]`** —
positions the station-attention model by contrast. The frozen architecture
record explicitly states the model is **not** an implementation of Graph
WaveNet, DCRNN, STGCN, GAT or ASTGCN, and the citation exists to make that
contrast legible, never to imply equivalence.

## Section 2.4 — Time-series foundation models

**`[REF-NEEDED: Chronos-2 paper]`**, **`[REF-NEEDED: TimesFM paper]`**,
**`[REF-NEEDED: Moirai paper]`**

- **Claim supported:** that these models exist, what they are, and — for
  Chronos-2 — what its published pretraining corpus enumerates.
- **Required:** authors, title, venue, year, DOI or canonical URL. The
  repository audit records consulting the Chronos-2 paper at
  `arXiv:2510.15821`, Appendix A, Table 6; that identifier is recorded but has
  **not** been independently verified in this phase.
- **Binding constraint:** none of these three models was executed. The citations
  establish eligibility context only. No sentence citing them may describe their
  performance, because no performance was measured.

## Section 2.5 — Leakage and contamination

**`[REF-NEEDED: work on data leakage in time-series or applied machine-learning
evaluation]`** and **`[REF-NEEDED: work on benchmark or pretraining
contamination in foundation models]`** — support the framing that partitioning,
preprocessing discipline and corpus provenance are evaluation-validity
concerns. The Chronos-2 exclusion decision rests on this framing, so the
citations matter more than usual here.

## Section 16 — Data availability

**`[REF-NEEDED: canonical citation for the UCI Beijing Multi-Site Air-Quality
dataset, including the associated source publication]`** — the DOI and landing
page are verified and hash-pinned; the associated source publication is not.
The manuscript already cites the dataset by DOI, so this placeholder affects
citation formatting completeness rather than provenance.

---

## Summary

| Section | Placeholders |
|---|---:|
| 1.1 Motivation | 1 |
| 2.1 Classical / ML forecasting | 2 |
| 2.2 Deep temporal forecasting | 2 |
| 2.3 Spatial forecasting | 1 |
| 2.4 Foundation models | 3 |
| 2.5 Leakage and contamination | 2 |
| 16. Data availability | 1 |
| **Total** | **12** |

All twelve are literature-positioning citations. **No result, table, figure or
quantitative claim in this manuscript depends on any of them**; every number
traces to a hash-pinned repository artifact recorded in
`manuscript/MANUSCRIPT_CLAIM_TRACEABILITY.csv`.
