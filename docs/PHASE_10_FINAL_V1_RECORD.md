# Phase 10 Final V1 Record — AirSense V1

**Research-protocol mapping: Protocol Phase 12 — Final V1 Report.**

> Record-sequence numbering (`PHASE_10_`) is independent of protocol phase
> numbering. Earlier records were not renamed. Mapping:
> `PHASE_00_FOUNDATION_RECORD.md` (protocol 1–2),
> `PHASE_00A_RUNTIME_RECORD.md` (runtime recovery),
> `PHASE_01_EDA_RECORD.md` (3), `PHASE_02_CLEANING_RECORD.md` (4),
> `PHASE_03_FEATURE_RECORD.md` (5), `PHASE_04_M0_BASELINE_RECORD.md` (6),
> `PHASE_05_M1_LINEAR_REGRESSION_RECORD.md` (7),
> `PHASE_06_M2_DECISION_TREE_RECORD.md` (8),
> `PHASE_07_M3_RANDOM_FOREST_RECORD.md` (9),
> `PHASE_08_FINAL_TEST_RECORD.md` (10),
> `PHASE_09_ERROR_ANALYSIS_RECORD.md` (11), this record (12).

- **Recorded:** 2026-09-06 (reconstruction date — **not** a historical date)
- **Branch:** `legacy`
- **Outcome:** **COMPLETE** — AirSense V1 is closed and frozen
- **Final report:** [`V1_FINAL_REPORT.md`](V1_FINAL_REPORT.md)
- **Reproducibility:** [`V1_REPRODUCIBILITY.md`](V1_REPRODUCIBILITY.md)

> **Phase 12 is synthesis and audit only.** No model was fitted, no
> prediction generated, no scientific metric changed, and no new analysis
> performed.

---

## 1. Runtime

| Property | Value |
|---|---|
| **Interpreter** | `/home/debalekha_chakraborty/AirSense/venv/bin/python` |
| **Python** | CPython **3.6.7** |
| Preflight | **READY**, exit 0 |

numpy 1.15.4, pandas 0.23.4, scipy 1.1.0, scikit-learn 0.20.0,
matplotlib 3.0.2, seaborn 0.9.0 — all MATCH. No dependency changed. No
package installed or upgraded.

## 2. Upstream scientific integrity

**95 hashes across the complete evidence chain were independently verified
before any Phase-12 output was written**, and all matched:

| Group | Artifacts |
|---|---:|
| Raw dataset | 1 |
| Phase-4 split | 4 |
| Phase-5 features | 8 |
| M0 | 2 |
| M1 | 6 |
| M2 | 7 |
| M3 | 7 |
| Pre-test freeze | 12 |
| Blind-prediction freeze | 7 |
| Final evaluation | 16 |
| Error analysis | 25 |
| **Total** | **95** |

Raw dataset SHA-256
`4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c` —
**MATCH**, mode `444`.

---

## 3. Phase-12 outputs

| Artifact | SHA-256 |
|---|---|
| `docs/V1_FINAL_REPORT.md` | `5c87f8d1407cfffebfea5d978741a3ddb075780f8f2ada1bb889274f3cda45f5` |
| `docs/V1_REPRODUCIBILITY.md` | `dd7a86786b39c6a2534916cbb417300c24b56cb960e9f85eaf9337c55f9d37f9` |
| `results/v1_summary/final_model_comparison.csv` | `17c26a1a05b8ec2b70fe6983d478acf39e080f74b24055c777ca327265a45f3a` |
| `results/v1_summary/key_findings.csv` | `46ffeddf39e7bf200f34b7450bc3768c5fba8ee608b39252818296ff18a35810` |
| `artifacts/v1_claims_ledger.csv` | `1dbd0b8f17f3c9f5150eea3ae643f92ef2987c6b8cdcf67d69694433f3f78fc0` |

**Claims ledger: 43 claims, all traced and verified** — 40 by literal
lookup in the named frozen artifact, 3 by recomputing a stated derivation
(the MAE ranking string and the two M3-versus-M0 improvement figures).

### A note on ordering

The three closing artifacts —
`artifacts/v1_evidence_registry.csv`,
`artifacts/v1_final_evidence_manifest.json` and
`artifacts/v1_final_freeze_receipt.json` — are created **after** this
record, in that order, so that each references only artifacts that already
exist. That keeps the chain acyclic:

```
this record  ->  evidence registry  ->  final evidence manifest  ->  freeze receipt
```

Consequently this record cannot state their digests: doing so would make the
registry's hash depend on content that contains the registry's hash. **Their
digests are recorded in the freeze receipt and in the phase's final
report-out.** The registry covers this record; the manifest records this
record's digest.

---

## 4. Frozen final results (unchanged)

| Model | MAE | RMSE | R² | MAE rank |
|---|---:|---:|---:|---:|
| M0 | 65.417388292344995 | 96.741588760774960 | −0.069942695889054 | 4 |
| M1 | 52.900193501674970 | 73.103403258667839 | 0.389044922216543 | 3 |
| M2 | 51.248813500432064 | 75.522046099895277 | 0.347948973628056 | 2 |
| **M3** | **48.535341383781684** | **70.277632509152781** | **0.435364300995649** | **1** |

**Ranking M3 < M2 < M1 < M0. Winner: M3.** Read from
`results/final_test/final_test_metrics.csv`; no value was recomputed from
raw source except as an integrity check.

Frozen headline limitations, unchanged: severe-hour (>282 µg/m³) M3 MAE
**160.897392** with **96.289062%** under-prediction across **512 hours** in
**41 episodes**; M3 residual autocorrelation **0.903292** at one hour; M1
**466** negative predictions.

---

## 5. Verification performed

| Check | Result |
|---|---|
| Runtime is exactly CPython 3.6.7 with the frozen pins | PASS |
| Preflight | READY, exit 0 |
| 95 upstream evidence hashes | all MATCH |
| Summary table agrees numerically with frozen final metrics | PASS |
| Claims ledger — every claim traced and verified | PASS (43/43) |
| Prose/evidence conflict scan | none found |
| Python 3.6 compilation of every project `.py` | PASS |
| No `.fit(`, no `.predict(`, no estimator import in Phase-12 code | PASS |
| Deterministic rerun of Phase-12 generated artifacts | byte-identical |
| Final verifier `scripts/verify_v1_final.py` | PASS, exit 0 |
| Upstream scientific artifacts unchanged after Phase 12 | PASS |

The final verifier and the deterministic rerun are executed after the freeze
artifacts exist; their outcomes are reported in the phase's final
report-out and the freeze receipt records the state they verify.

---

## 6. Files created and modified

**Created:** `docs/V1_FINAL_REPORT.md`, `docs/V1_REPRODUCIBILITY.md`,
`docs/PHASE_10_FINAL_V1_RECORD.md`,
`results/v1_summary/final_model_comparison.csv`,
`results/v1_summary/key_findings.csv`, `artifacts/v1_claims_ledger.csv`,
`artifacts/v1_evidence_registry.csv`,
`artifacts/v1_final_evidence_manifest.json`,
`artifacts/v1_final_freeze_receipt.json`,
`scripts/build_v1_summary.py`, `scripts/freeze_v1_evidence.py`,
`scripts/verify_v1_final.py`.

**Modified:** `README.md` (final V1 state) and
`docs/V1_RESEARCH_PROTOCOL.md` — **administrative status only**, marking
Phase 12 complete.

**No frozen scientific evidence was modified.**

## 7. Protocol status

`docs/V1_RESEARCH_PROTOCOL.md` phase table now records Phases 1–12 as
complete. **No scientific declaration was altered**: §1 pre-registration,
§3 temporal-leakage and random-split rules, §4 raw immutability, §5
historical constraint, §6 reporting standards and §7 amendments are all
byte-identical to their committed form.

**§7 still reads "No amendments."** There has been no scientific deviation
requiring one, and an amendment is not added merely because results are now
known. **Administrative status changes are not scientific amendments.**

The optional random-split educational comparison permitted by the protocol
was **not** performed, and V1 closes without it.

---

## 8. Declarations

### V1 completion

**AirSense V1 is COMPLETE.** All twelve protocol phases are finished, every
declared gate passed, and the evidence package is internally consistent and
hash-linked end to end.

### Test exhaustion

**The 2014 target has been opened and scored once. It is exhausted.** It is
no longer an unseen test set. Any model changed after this point must not
claim a fresh evaluation on the same split, and the target cannot be
resealed. A genuinely fresh evaluation would require data outside 2010–2014.

### No scientific change

**Phase 12 changed no model, feature, split, hyperparameter, prediction,
metric, threshold or ranking.** It introduced no new metric, subgroup,
correlation, significance test, bootstrap interval, random-split experiment,
model or ensemble. Its only calculations were integrity checks, file hashes,
audit counts, and direct arithmetic on already-frozen values.

### Post-freeze policy

Once `artifacts/v1_final_freeze_receipt.json` exists and verifies, V1 is
immutable scientific evidence. Future work must not silently edit V1 to
improve its results, methodology, features, models or conclusions. A genuine
factual or documentation correction must be recorded transparently as a
post-freeze correction that leaves the original evidence intact. **Modern
AirSense research must proceed separately as V2**, and must not reuse the
2014 partition as an unseen test.

---

## 9. Status

Protocol Phase 12 is **complete**. All legitimate changes are left
**unstaged** for manual handling.

**AirSense V1 is complete and frozen. Awaiting human review.**
