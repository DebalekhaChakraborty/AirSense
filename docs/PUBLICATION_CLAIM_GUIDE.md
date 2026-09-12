# AirSense V2 — Publication Claim Guide

Binding wording rules for the manuscript. Every claim must trace to
`artifacts/phase10_claim_evidence_matrix.csv` and
`artifacts/phase10_master_evidence_index.json`.

Phase 10 froze this guide. Phase 11 drafts the manuscript against it.

---

## F. Evidence classification rules

Every claim carries exactly one class.

| Class | Meaning | May be described as |
|---|---|---|
| **C1** | LOCKED CONFIRMATORY — obtained under the Phase-8 frozen final-test hierarchy | "on the locked final test" |
| **C2** | DEVELOPMENT / ROBUSTNESS SUPPORTED — development validation, training-only tuning, multi-seed robustness, LOSO; all before the test was opened | "on development validation", "across five seeds" |
| **C3** | POST-TEST EXPLORATORY — first generated in Phase 9, after the test was scored | "post-test exploratory analysis indicates" |
| **C4** | HISTORICAL V1 — from the frozen AirSense V1 program | "in the earlier V1 study" |
| **C5** | METHODOLOGICAL / GOVERNANCE — data integrity, leakage control, sealing, provenance | "by protocol", "by audit" |

**The hard rule: no C3 finding may ever be presented as C1.** Phase 9 explains
the locked result; it does not add to it. A reviewer must be able to tell, from
the sentence alone, which class a claim belongs to.

Only three models have a locked-test role: `B3_R2` primary confirmatory,
`GRU_R1` (seed 42) secondary confirmatory severe-tail, `B0` reference
benchmark. Thirteen further models are development-only. No foundation model
was executed.

---

## A. Claims we can make strongly

These are C1 or C5 and rest on frozen, hash-verified evidence.

1. **The test was sealed and opened once.** The final test was never
   numerically opened before Phase 8; opening required an authorization
   receipt; predictions were generated and hashed before any metric was
   computed; there were zero future-target access violations across 1,233,036
   index checks. *(C5)*

2. **The primary confirmatory result.** On the locked 2016–2017 test, `B3_R2`
   achieved a macro station-horizon MAE of **31.98722559774534** over 411,012
   samples and 48 equally weighted station-horizon cells. *(C1)*

3. **Improvement over persistence.** `B3_R2` improved on causal persistence
   `B0` (36.138513538776856) by **11.487%**, and did so at every horizon from
   1 to 24 hours. *(C1)*

4. **The secondary confirmatory result.** `GRU_R1` seed 42 achieved a severe
   MAE of **109.3328897571946** over 20,336 severe samples, lower than
   `B3_R2`'s 131.901121. *(C1)*

5. **Persistence held the severe tail.** `B0` recorded the lowest severe MAE
   at **93.22447875688434**. *(C1)*

6. **Causal construction.** No model input ever postdates its forecast origin;
   future observed meteorology and pollutants are prohibited by construction,
   and only the deterministic calendar is available for future timestamps.
   *(C5)*

7. **Roles were frozen before opening and never changed.** *(C5)*

---

## B. Claims requiring qualifiers

Each must carry its qualifier in the same sentence, not in a distant footnote.

| Claim | Required qualifier |
|---|---|
| Target PM2.5 history improves forecasting (H1) | "on development validation"; **not** re-tested on the locked test |
| History helps most at short horizons (H2) | "on development validation"; **not** re-tested |
| Co-pollutants add value (H3) | "for the gradient-boosted model"; it **degraded** the neural and transformer variants |
| Cross-station context adds value (H4) | "in a paired comparison across five seeds on development validation"; **not** confirmed on the locked test; h=24 direction unreliable |
| GRU_R1's severe advantage (H5) | relative to `B3_R2` only; persistence remained the strongest severe benchmark |
| LOSO transfer | "held-out target-station supervision transfer **with causal historical observations from the held-out station available**" |
| Any single-seed severe number | Phase 7 showed severe-tail metrics spanning roughly 29–35 µg/m³ across seeds |
| The 244.0 threshold | "the training pooled P95", a study-internal quantile |
| Severe-tail comparisons | the gap is concentrated at long horizons; at h=1 the three models are close |

---

## C. Post-test exploratory findings

All C3. Introduce each with an exploratory framing, never as evidence *for* a
model.

- **Residual dependence.** "Substantial short-lag residual dependence remains
  after forecasting, indicating unresolved temporal structure in the locked
  predictions." (lag-1 ≈ 0.95–0.96 at h=24, all three models.)
- **Severe error shape.** "The exploratory error analysis is consistent with
  regression toward the conditional mean during extreme episodes." Severe
  errors are strongly one-directional: the mean residual accounts for most of
  the severe MAE.
- **Why persistence holds up.** "At long horizons, persistence retains an
  advantage during sustained severe episodes because recently observed extreme
  values remain informative." Observed PM2.5 autocorrelation is ≈0.97 at 1 h
  and ≈0.43 at 24 h.
- **Event behaviour.** "At 24 h, the frozen severe-event detection diagnostics
  are insufficient to establish reliable early-warning capability under the
  study's fixed 244 µg/m³ threshold."
- **Seasonality, diurnal pattern, complementarity, negative predictions** —
  all descriptive stratifications, none predeclared.

---

## D. Claims we must NOT make

Each is prohibited outright.

| Prohibited | Why |
|---|---|
| "AirSense solves severe PM2.5 forecasting" | The reference baseline outperformed both learned models in the tail. |
| "GRU beats persistence in severe pollution" | Factually contradicted: 109.33 against 93.22. |
| "Cross-station attention learns pollution propagation" | Phase-7 diagnostics found attention was not proximity-driven (Spearman = 0.136). |
| "Foundation models failed" | None was executed; this is a governance outcome, not a measurement. |
| "All hypotheses were confirmed on the final test" | None of H1–H5 was re-tested confirmatorily there. |
| "V2 improves V1 by X%" | Different tasks, periods, designs and severe thresholds; the comparison is invalid. |
| "The model is operationally ready for early warning" | Operational usability was never defined or tested. |
| "A bias–variance decomposition shows…" | No formal decomposition was performed. |
| "The neural network cannot learn extremes" | Unsupported mechanistic claim. |
| "B3_R2 wins because it uses 405 features" | Feature count is not an established cause. |
| Any p-value, confidence interval or significance claim | None was predeclared before the test was opened. |

---

## E. Exact terminology

| Use | Not |
|---|---|
| "locked final test" | "test set", "holdout" |
| "post-test exploratory analysis" | "further results", "additional findings" |
| "reference benchmark" for `B0` | "baseline model we beat" |
| "primary confirmatory model" | "best model", "winner" |
| "associated with" | "causes", "drives", "because of" |
| "training pooled P95 (244.0 µg/m³)" | "AQI threshold", "regulatory limit" |
| "held-out target-station supervision transfer" | "unseen-station forecasting" |
| "no eligible foundation model could be evaluated" | "foundation models were worse" |
| "under-prediction" | "failure", "collapse" |

**On feature attribution.** Write: "`B3_R2`'s advantage is consistent with
richer causal pollutant, meteorological and temporal history providing useful
information beyond persistence." Do not attribute the advantage to feature
count alone.

**On severe error composition.** Write: "Severe errors are strongly
one-directional: the mean residual accounts for most of the severe MAE, while
residuals also retain substantial temporal dependence." Do not describe this as
a variance component result.

**On the headline framing.** The defensible headline is that engineered causal
history beat persistence on average across 1–24 h, while the extreme tail
remained the open problem. Both halves must appear together; reporting either
alone misrepresents the study.
