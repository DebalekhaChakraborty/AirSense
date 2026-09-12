# Development-validation model comparison

**Evidence class:** C2
**Source artifact:** `artifacts/phase10_table_development_models.csv`
**Source SHA-256:** `387417f6d6ef4652a48d4e6aaacd11d079815cb0822ee1479aa651c367d8f273`
**Manuscript section:** 7. Development-validation results

| model | family | regime | validation_macro_MAE | validation_severe_MAE | validation_severe_underprediction_pct | evidence_phase | seed_provenance | role |
|---|---|---|---|---|---|---|---|---|
| B0 | classical | n/a | 33.127699 | 101.273235 | 71.624812 |  | single run (deterministic) | REFERENCE BENCHMARK |
| B1 | classical | n/a | 51.205937 | 152.58886 | 85.882164 |  | single run (deterministic) | development only |
| B2 | classical | n/a | 56.733945 | 278.796416 | 100.0 |  | single run (deterministic) | development only |
| B3_R0 | classical | R0 | 40.623282 | 176.944868 | 97.048723 |  | single run (deterministic) | development only |
| B3_R1 | classical | R1 | 30.472594 | 138.324841 | 88.506117 |  | single run (deterministic) | development only |
| B3_R2 | classical | R2 | 30.111722 | 136.611861 | 88.286113 |  | single run (deterministic) | PRIMARY CONFIRMATORY MODEL |
| GRU_R1 | neural | R1 | 30.912193 | 118.964535 | 93.555484 |  | single seed 42 | SECONDARY CONFIRMATORY SEVERE-TAIL MODEL |
| GRU_R2 | neural | R2 | 31.842957 | 145.125893 | 95.12771 |  | single seed 42 | development only |
| TCN_R1 | neural | R1 | 30.688629 | 152.530079 | 95.422838 |  | single seed 42 | development only |
| TCN_R2 | neural | R2 | 34.079881 | 183.212217 | 98.288259 |  | single seed 42 | development only |
| iTransformer_R1 | transformer | R1 | 32.480136 | 140.986383 | 84.030908 |  | single seed 42 | development only |
| iTransformer_R2 | transformer | R2 | 32.723452 | 145.136876 | 89.187594 |  | single seed 42 | development only |
| SA_R2 | spatial | R2 | 32.239759 | 142.703386 | 92.010088 |  | single seed 42 | development only |
| SA_R3 | spatial | R3 | 31.532702 | 121.270043 | 88.044645 |  | single seed 42 | development only |

**Caption (draft).** Development-validation comparison across model families
and information regimes. All values are development evidence obtained before the
locked final test was opened.

**Footnotes and qualifiers.**

- **These are not final-test results and must not be read as such.** Only
  `B3_R2`, `GRU_R1` and `B0` ever carried a locked-test role; the other eleven
  rows are development-only models.
- Severe columns use the frozen threshold: observed PM2.5 **> 244.0** ug/m3,
  the training pooled P95. It is not a regulatory or air-quality-index boundary.
- Neural rows are single-seed (seed 42). Severe-tail metrics are the least
  seed-stable quantities in this study; see Table 4.
- `B0` is a reference benchmark, not a selected learned model.
- Classical rows are deterministic single runs.
