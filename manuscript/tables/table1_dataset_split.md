# Table 1 — Dataset and chronological split

**Evidence class:** C5 (methodological / governance)
**Source artifact:** `artifacts/phase10_dataset_ledger.json`
**Source SHA-256:** `08676a8c49d636e5ac7fd0ac892ff600afa8344e412b108f0d33c1e459a8a5f9`
**Manuscript section:** 3. Data and protocol

| Partition | Target window | Eligible samples | Status |
|---|---|---|---|
| Train | 2013-03-01 to 2015-02-28 | 821,184 | Open throughout |
| Validation | 2015-03-01 to 2016-02-29 | 413,148 | Development only |
| Locked final test | 2016-03-01 to 2017-02-28 | 411,012 | Sealed; opened once |

**Caption (draft).** Chronological partitioning of the Beijing Multi-Site
Air-Quality dataset (420,768 hourly station records,
12 stations, 2013-03-01 to
2017-02-28). Partitions are contiguous blocks assigned by
**target timestamp**, not by forecast origin.

**Footnotes and qualifiers.**

- Partitioning is by target timestamp. A target early in validation or test
  legitimately has its forecast origin, and therefore its context window, inside
  the preceding partition. Such samples are retained: this is rolling-origin
  history, not leakage. Model parameters and every preprocessing statistic
  remain fitted on training data alone.
- Eligibility requires an **observed** PM2.5 value at the target timestamp.
  Targets are never imputed.
- The locked test was sealed from the start of the programme and opened exactly
  once, under an authorisation receipt.
- No random splitting was used anywhere in this study.
