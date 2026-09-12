# Post-test severe and error diagnostics

**Evidence class:** C3
**Source artifact:** `artifacts/phase10_table_posttest_diagnostics.csv`
**Source SHA-256:** `05e01cf31b064124cec445bd7266451aa0bb745b758f9beafcdb2a6c57f5896b`
**Manuscript section:** 10. Post-test exploratory error analysis

| subject | diagnostic | value | evidence_class | artifact_source |
|---|---|---|---|---|
| B3_R2 | residual_acf_lag1_h24 | 0.962625 | C3 | artifacts/phase9_residual_acf_summary.csv |
| B3_R2 | residual_acf_lag24_h24 | 0.266948 | C3 | artifacts/phase9_residual_acf_summary.csv |
| B3_R2 | severe_MAE_h24 | 229.790698 | C3 | artifacts/phase9_severe_tail_analysis.csv |
| B3_R2 | severe_underprediction_pct_h24 | 99.901652 | C3 | artifacts/phase9_severe_tail_analysis.csv |
| B3_R2 | severe_detection_recall_h24 | 0.006884 | C3 | artifacts/phase9_severe_detection_analysis.csv |
| B3_R2 | negative_prediction_count | 53 | C3 | artifacts/phase9_negative_prediction_analysis.csv |
| GRU_R1 | residual_acf_lag1_h24 | 0.949952 | C3 | artifacts/phase9_residual_acf_summary.csv |
| GRU_R1 | residual_acf_lag24_h24 | -0.232078 | C3 | artifacts/phase9_residual_acf_summary.csv |
| GRU_R1 | severe_MAE_h24 | 152.630723 | C3 | artifacts/phase9_severe_tail_analysis.csv |
| GRU_R1 | severe_underprediction_pct_h24 | 96.144768 | C3 | artifacts/phase9_severe_tail_analysis.csv |
| GRU_R1 | severe_detection_recall_h24 | 0.235641 | C3 | artifacts/phase9_severe_detection_analysis.csv |
| GRU_R1 | negative_prediction_count | 1358 | C3 | artifacts/phase9_negative_prediction_analysis.csv |
| B0 | residual_acf_lag1_h24 | 0.952522 | C3 | artifacts/phase9_residual_acf_summary.csv |
| B0 | residual_acf_lag24_h24 | -0.297138 | C3 | artifacts/phase9_residual_acf_summary.csv |
| B0 | severe_MAE_h24 | 142.528324 | C3 | artifacts/phase9_severe_tail_analysis.csv |
| B0 | severe_underprediction_pct_h24 | 85.464201 | C3 | artifacts/phase9_severe_tail_analysis.csv |
| B0 | severe_detection_recall_h24 | 0.356216 | C3 | artifacts/phase9_severe_detection_analysis.csv |
| B0 | negative_prediction_count | 0 | C3 | artifacts/phase9_negative_prediction_analysis.csv |
| observed_PM2.5 | target_acf_lag1 | 0.971949 | C3 | artifacts/phase9_test_target_acf.csv |
| observed_PM2.5 | target_acf_lag24 | 0.427467 | C3 | artifacts/phase9_test_target_acf.csv |
| all_models | severe_event_count | 586 | C3 | artifacts/phase9_severe_events.csv |
| B0 | severe_strictly_best_of_three_pct | 52.876573 | C3 | artifacts/phase9_error_complementarity.csv |
| all_models | DJF_severe_hours_h24 | 3120 | C3 | artifacts/phase9_seasonal_error_analysis.csv |

**Caption (draft).** Post-test exploratory severe and error diagnostics,
computed from the frozen locked-test prediction arrays after scoring.

**Footnotes and qualifiers.**

- **Exploratory only.** Every value here is C3. None was predeclared, and none
  is confirmatory. This table must be rendered visibly separate from Table 5.
- These diagnostics explain the locked result; they do not add to it, and no
  value here changes any role, endpoint or hypothesis status.
- Residual ACF values are associations. No causal mechanism is established.
- Detection recall is reported under the study's fixed 244.0 ug/m3 threshold and
  is insufficient to establish early-warning capability at 24 hours.
- Negative-prediction counts are reported unclipped and deliberately so.
- No ensemble, oracle or weighting scheme was constructed from the
  complementarity figures.
