# Multi-seed robustness

**Evidence class:** C2
**Source artifact:** `artifacts/phase10_table_multiseed_robustness.csv`
**Source SHA-256:** `47f1a13334816ef097cc7403606386ca2917c8501abd8752c9100926c736367a`
**Manuscript section:** 8. Robustness results

| model | seed_count | macro_MAE_mean | macro_MAE_sd | macro_MAE_min | macro_MAE_max | macro_MAE_range | severe_MAE_mean | severe_MAE_sd | severe_MAE_min | severe_MAE_max | severe_MAE_range | role_in_phase7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRU_R1 | 5 | 30.75906 | 0.123331 | 30.619271 | 30.912193 | 0.292922 | 115.793 | 3.601889 | 111.641105 | 119.077182 | 7.436077 | candidate |
| TCN_R1 | 5 | 30.543738 | 0.338707 | 30.162691 | 30.964026 | 0.801335 | 137.758679 | 10.429503 | 123.102834 | 152.530079 | 29.427245 | candidate |
| SA_R3 | 5 | 31.545205 | 0.424712 | 31.221119 | 32.273309 | 1.05219 | 136.818892 | 13.968082 | 121.270043 | 156.431747 | 35.161704 | candidate |
| SA_R2 | 5 | 32.518538 | 0.551844 | 31.850858 | 33.213047 | 1.362189 | 154.436906 | 14.7961 | 140.399427 | 175.192043 | 34.792616 | paired control |

**Caption (draft).** Spread of development-validation metrics across five
training seeds (42-46) for the stochastic candidates, with `SA_R2` carried as the
paired control for the cross-station contrast.

**Footnotes and qualifiers.**

- **Spreads are descriptive, not inferential.** No interval reported here is a
  sampling-theoretic interval, and none was predeclared.
- Severe-tail metrics are far less seed-stable than overall metrics: severe MAE
  ranges span roughly 7 to 35 ug/m3 against roughly 0.3 to 1.4 for overall macro
  MAE. Single-seed severe claims should be treated as provisional.
- `SA_R2` and `SA_R3` differ only in information access; architecture, capacity,
  training protocol and seed are identical.
- Seed-42 artifacts were not retrained; their frozen prediction arrays were
  re-scored through the robustness code path and reproduced exactly.
