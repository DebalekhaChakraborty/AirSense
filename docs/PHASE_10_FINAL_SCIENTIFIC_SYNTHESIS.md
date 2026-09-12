# AirSense V2 — Final Scientific Synthesis

**Phase 10. Classification: FINAL_SCIENTIFIC_SYNTHESIS. Status:
SYNTHESIS_FROZEN.**

No experiment was run in this phase. No model was trained or loaded, no
prediction was generated, no metric was newly defined. Every number below is
parsed from a frozen artifact or is an arithmetic restatement of frozen values.

Evidence classes are marked throughout: **C1** LOCKED CONFIRMATORY, **C2**
development/robustness, **C3** POST-TEST EXPLORATORY, **C4** historical V1,
**C5** methodological/governance. Wording is bound by
`docs/PUBLICATION_CLAIM_GUIDE.md`.

---

## 1. Research question

Can a forecasting model, restricted to strictly causal inputs, predict Beijing
PM2.5 at 1–24 hour horizons better than carrying the last observation forward —
and does that hold during the severe pollution episodes that matter most?

The second half of the question is the one the study answers least comfortably,
and it is the more important half.

## 2. Study design *(C5)*

A pre-registered, phase-gated design. Each phase froze its claims and artifact
hashes before producing results; a validator per phase re-derived them and
exited non-zero on drift. The final test was sealed from Phase 0 and opened
exactly once, in Phase 8, after an authorization receipt.

The hierarchy evaluated on the locked test was frozen before opening:
`B3_R2` primary confirmatory, `GRU_R1` seed 42 secondary confirmatory
severe-tail, `B0` reference benchmark. No model could be promoted afterwards,
whatever the outcome.

Opening was a one-way transition. Predictions for all three models were
generated chronologically and hashed into
`artifacts/phase8_prediction_lock.json` **before** any target was read, so the
claim that results did not influence prediction is checkable after the fact
rather than asserted.

## 3. Dataset and chronological protocol *(C5)*

UCI Beijing Multi-Site Air-Quality (ID 501, DOI 10.24432/C5RK5G, CC BY 4.0):
12 stations, 420,768 hourly rows, 2013-03-01 to 2017-02-28, every source CSV
hash-pinned.

| Partition | Interval | Samples |
|---|---|---|
| Train | 2013-03-01 → 2015-02-28 | 821,184 |
| Validation | 2015-03-01 → 2016-02-29 | 413,148 |
| **Locked test** | 2016-03-01 → 2017-02-28 | **411,012** |

Partitioning is by **target timestamp**. Context 48 h; horizons 1/6/12/24 h;
12 stations × 4 horizons = 48 equally weighted cells. Severe threshold 244.0
µg/m³ — the **training pooled P95**, a study-internal quantile and not a
regulatory or AQI boundary. The training pooled median, 61.0 µg/m³, bounds the
exploratory concentration strata.

Early test targets legitimately have forecast origins inside the validation
period. That is rolling-origin forecasting across a boundary, not leakage.

## 4. Information regimes *(C5)*

| Regime | Inputs |
|---|---|
| R0 | meteorology, wind direction, rain occurrence, calendar, masks |
| R1 | R0 + target-station PM2.5 history, mask, gap age |
| R2 | R1 + co-pollutants (PM10, SO2, NO2, CO, O3) with masks and gap ages |
| R3 | R2 + cross-station history |

Future observed meteorology and pollutants are prohibited. Only the
deterministic calendar is available for future timestamps. Historical
observations are restricted to timestamps at or before the forecast origin, and
a PM2.5 value becomes available as history only once it is past relative to the
origin. Enforcement was measured: **1,233,036 index checks, zero violations.**

## 5. Model families

Classical baselines (`B0` causal persistence, `B1`, `B2`), gradient-boosted
engineered features (`B3_R0/R1/R2`), recurrent and convolutional sequence
models (`GRU_*`, `TCN_*`), a transformer (`iTransformer_*`) and a cross-station
attention model (`SA_R2`, `SA_R3`). Sixteen models were evaluated on
development validation; three carried locked-test roles.

**No foundation model was executed** *(C5)*. Chronos-2 was excluded on a
material pretraining-overlap risk with the sealed test period identified by
provenance audit — byte-level contamination was neither demonstrated nor
claimed. TimesFM-3.0 and Moirai-2.0-R-small were excluded on weight licensing.
This is a governance outcome, not a measurement.

## 6. Development findings *(C2)*

Best development macro station-horizon MAE: `B3_R2` 30.112, `B3_R1` 30.473,
`TCN_R1` 30.689, `GRU_R1` 30.912, `SA_R3` 31.533, `B0` 33.128.

- Adding target PM2.5 history moved `B3_R0` 40.623 → `B3_R1` 30.473 — the
  largest single effect in the study (H1).
- Adding co-pollutants helped only the gradient-boosted family
  (`B3_R1` → `B3_R2`, +0.361) and **degraded** every learned family: `GRU_R2`,
  `TCN_R2` and `iTransformer_R2` were all worse than their R1 counterparts
  (H3).
- In the severe stratum, `B0` was already the strongest at 101.273 against
  `B3_R2`'s 136.612 — the tail pattern was visible before the test was opened.

## 7. Robustness findings *(C2)*

Five seeds (42–46) on `GRU_R1`, `TCN_R1`, `SA_R3` and the paired control
`SA_R2`. Overall macro MAE was stable (SD 0.12–0.55); **severe-tail metrics
were not**, spanning roughly 29–35 µg/m³ across seeds. Any single-seed severe
number is therefore provisional.

Cross-station context (H4): `SA_R3` improved on `SA_R2` by 0.707 MAE overall
(+2.193%) and in **5/5 paired seeds**, but the h=24 direction was not reliably
signed across seeds. Leave-one-station-out showed a mean penalty of +0.134%
with 5 of 12 stations improving when held out — a held-out **target-station
supervision transfer** result, since the held-out station's causal historical
observations remained available to the shared encoder.

Spatial diagnostics found attention was **not** proximity-driven
(Spearman(distance, attention) = 0.136), while distance and training PM2.5
correlation were strongly related (−0.938). Coordinates were used descriptively
and never as a model input.

## 8. Locked final-test findings *(C1)*

| Model | Role | Macro station-horizon MAE | Severe MAE > 244.0 |
|---|---|---|---|
| `B3_R2` | PRIMARY CONFIRMATORY MODEL | **31.98722559774534** | 131.901121 |
| `GRU_R1` seed 42 | SECONDARY CONFIRMATORY SEVERE-TAIL MODEL | 33.082098 | **109.3328897571946** |
| `B0` | REFERENCE BENCHMARK | 36.138513538776856 | **93.22447875688434** |

n = 411,012; severe n = 20,336; 232 targets sitting exactly at 244.0 are
excluded by the strict `>` rule.

- `B3_R2` improved on persistence by **11.487%**, and at every horizon:
  +9.47% (h=1), +11.54% (h=6), +10.82% (h=12), +12.30% (h=24).
- `GRU_R1` reduced severe MAE relative to `B3_R2` by 17.1%, the direction
  development predicted.
- **Persistence remained the strongest severe benchmark**, ahead of both
  learned models. `B0` is not promoted; it remains the reference benchmark.

All three models transferred with modest degradation on overall error (+6.2%,
+7.0%, +9.1%) and improved severe error, the 2016–17 tail being somewhat easier
than 2015–16. The reference degraded most and the primary model least.

## 9. Post-test exploratory findings *(C3)*

**These explain the locked result; they do not add to it.**

- **Residual structure.** Substantial short-lag residual dependence remains
  after forecasting — lag-1 ≈ 0.95–0.96 at h=24 for all three models —
  indicating unresolved temporal structure in the locked predictions.
- **Severe error shape.** Severe errors are strongly one-directional: the mean
  residual accounts for most of the severe MAE, while residuals also retain
  substantial temporal dependence. Under-prediction rises with horizon, from
  63.8% to 99.9% for `B3_R2`.
- **Why the tail favours carrying forward.** The analysis is consistent with
  regression toward the conditional mean during extreme episodes. At long
  horizons, persistence retains an advantage during sustained severe episodes
  because recently observed extreme values remain informative — observed PM2.5
  autocorrelation is ≈0.97 at 1 h and ≈0.43 at 24 h.
- **Horizon dependence.** The tail gap is concentrated at long lead. At h=1
  `B3_R2` (26.76) and `B0` (26.02) are close; by h=24 they are 229.79 against
  142.53.
- **Events.** Across 586 strict severe events, at h=24 `B3_R2` under-predicted
  the peak in 100% and flagged onset in 0.3%. At 24 h the frozen severe-event
  detection diagnostics are insufficient to establish reliable early-warning
  capability under the study's fixed 244 µg/m³ threshold.
- **Concentration strata.** Every model over-predicts below the training median
  and under-predicts above it.
- **Seasonality.** DJF carries 3,120 of the h=24 severe hours against 28 in
  JJA.
- **Complementarity.** Absolute errors correlate 0.67–0.87 — the models largely
  fail together — yet `B0` holds the strictly smallest error on 52.88% of
  severe samples. No ensemble was constructed.
- **Negative predictions.** `GRU_R1` 1,358 (807 at h=1), `B3_R2` 53, `B0` none.
  Nothing was clipped.

## 10. H1–H5 evidence ledger

Full detail in `artifacts/phase10_hypothesis_ledger.json`.

| | Status | Class | Re-tested on locked test |
|---|---|---|---|
| H1 target history helps | SUPPORTED ON DEVELOPMENT VALIDATION | C2 | **No** |
| H2 benefit largest at short horizons | SUPPORTED ON DEVELOPMENT VALIDATION | C2 | **No** |
| H3 co-pollutants add value | WEAK / MODEL-DEPENDENT DEVELOPMENT SUPPORT | C2 | **No** |
| H4 cross-station context adds value | ROBUSTLY SUPPORTED ON DEVELOPMENT / ROBUSTNESS EVIDENCE, BUT NOT INDEPENDENTLY CONFIRMED ON THE LOCKED TEST | C2 | **No** |
| H5 modern methods reduce severe-tail error | PARTIALLY SUPPORTED AGAINST SOME LEARNED AND ENGINEERED MODELS, BUT NOT SUPPORTED AS SUPERIORITY OVER PERSISTENCE ON THE LOCKED TEST | C2/C3 | **No** |

The confirmatory set held three models and no paired regime variant, so none of
these contrasts could be re-tested there. No final-test hypothesis test is
manufactured, and H5 is not redefined to make it pass.

## 11. V1 → V2 structural continuity *(C4)*

V1 performed concurrent-hour estimation on a different period with a
single-station design and a different severe threshold (282.0). V2 performs
future multi-horizon forecasting across 12 stations. **Direct metric comparison
is invalid and none is made.**

Two failure modes recur in both studies: systematic severe under-prediction,
and residual autocorrelation near 0.9–0.97 at lag 1. Structural progress in V2:
strict chronological protocol, multi-station evidence, explicit information
regimes, multi-horizon evaluation, a sealed test opened once, multi-seed
robustness, and a paired cross-station controlled experiment.

## 12. Limitations

Seventeen recorded in `artifacts/phase10_limitations_ledger.json`. The ones
that most constrain interpretation: one city and one pollutant with no external
validation; the secondary artifact is a single seed while its role rested on
five-seed evidence; H1–H4 were never re-tested confirmatorily; no inferential
interval was predeclared, so no significance claim is available; the severe
threshold is a training quantile; and operational usability was never defined
or tested.

## 13. Future work

Twelve untested directions in `artifacts/phase10_future_work_ledger.json` —
tail-aware and quantile objectives, probabilistic forecasting, persistence/model
hybrid gating, forecast meteorology as input, cross-city validation, a
clean-pretraining foundation-model benchmark, and predeclared block-based
uncertainty quantification. **None was attempted and no claim is made that any
would improve results.**

## 14. Reproducibility *(C5)*

Four immutable git anchors: pre-test `75494266`, locked evaluation `9a787a63`,
post-evaluation validation `7ccfc7e5`, Phase-9 closure `8cc3a7ab`, with V1
frozen at `16c9030c`. Raw data hash-pinned; `data/processed` regenerable from
`scripts/build_forecast_dataset.py` with every digest in the Phase-2 manifest.
Twelve validators, each pinned in an append-only registry chain. Environment,
seeds and the full test-access chronology are recorded in
`artifacts/phase10_reproducibility_ledger.json`.

Zero future-target access violations; zero retraining after test access; zero
model selection after test access.

## 15. Canonical conclusion

On the sealed 2016–2017 test, opened once under a hierarchy frozen beforehand, a
gradient-boosted model over causal pollutant, meteorological and temporal
history improved macro station-horizon MAE relative to causal persistence by
11.487%, and improved on it at every horizon from 1 to 24 hours *(C1)*. In the
severe stratum the prespecified secondary model reduced severe MAE relative to
the primary model, while persistence remained the strongest severe benchmark
*(C1)*.

Development and multi-seed experiments showed substantial value from target
PM2.5 history, weak and model-dependent value from co-pollutants, and a paired
cross-station benefit in all five seeds — none of which was independently
re-tested on the locked final test *(C2)*.

Post-test exploratory analysis is consistent with regression toward the
conditional mean during extreme episodes, with strongly one-directional severe
errors and substantial unresolved short-lag residual structure *(C3)*.

**The principal open problem is reliable forecasting of extreme pollution
episodes rather than average multi-horizon accuracy.** Both halves of that
sentence belong together: the study demonstrates a real average-case gain over
a strong naive rule, and it demonstrates, on the same sealed data, that the
gain does not extend to the tail.
