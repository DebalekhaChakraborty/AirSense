# Phase 8 — Locked Final-Test Record

Chronological record of the one-way opening of the AirSense V2 sealed final
test. The execution contract is `docs/PHASE_08_FINAL_TEST_PROTOCOL.md`.

Every number below was produced after the opening receipt was written, and no
code changed after the first metric was seen.

---

## 1. Pre-test git snapshot

| | |
|---|---|
| Branch | `master` |
| HEAD | `75494266792e09bbcfa51c00aaacec58c5b0fb4a` |
| origin/master | identical |
| Working tree before any Phase-8 artifact | clean, 0 entries |
| `legacy` (frozen V1) | `16c9030cf74508b620cc6d28f90346aa379f29cd`, untouched |

No git write of any kind was performed in Phase 8.

## 2. Pre-test freeze verification

| Artifact | SHA-256 | Result |
|---|---|---|
| `phase7_pretest_freeze.json` | `78a0e360…ade60` | exact match |
| `phase7_pretest_freeze_addendum.json` | `6a1342b7…13708e2` | exact match |
| `phase7_candidate_freeze.json` | `0b6b0518182503425aff92a83a8f5d0f9f3b4721e8e667e052c87785339520c8` | full digest resolved |

Hierarchy confirmed from the addendum: primary `B3_R2` on macro station-horizon
MAE, secondary `GRU_R1` on severe MAE at 244.0, reference `B0`, and
`promotion_after_test_access_permitted: false`.

## 3. Two contradictions found, and the human-approved correction

Executing the locked prompt as written was impossible. Both blockers were
reported before anything was opened, and resolved by human review as **Option
D**, recorded in `artifacts/phase8_preopening_protocol_amendment.json`
(`abfc1d50e027f132…`).

**Blocker 1 — the tooling registry could not be appended to.**
`phase7_pretest_freeze.json` pins `artifacts/verification_tooling_registry.json`
inside `evidence_sha256` at `a1ec2652…79122f`, and
[validate_phase7.py:372](../scripts/validate_phase7.py#L372) re-hashes every
evidence path. Adding the completed Phase-7 validator to that registry, as the
original prompt directed, would change its bytes and fail the gate "All frozen
Phase-7 evidence hashes match". Rewriting the freeze was forbidden and would
have failed its own hash gate. Simulated in a scratch copy before reporting:
the updated registry hashed `1a1d7aaf…92f1`, producing exactly that failure.

*Resolution:* the Phase-7 registry is untouched. An append-only child,
`artifacts/phase8_verification_tooling_registry.json` (`c59abf2765fbb078…`),
names it as parent and pins all nine validators, including `validate_phase7.py`
(`72b2b335…f74ba`) and `validate_phase8.py`.

**Blocker 2 — the mandated output paths would fail Phase-0 and Phase-1.**
`owned_by_a_later_phase()` does not recognise `results/final_test`, and the
mandated table names match the forbidden patterns `model_comparison`, `metric`
and `prediction`. Creating them there fails "No model-result file exists" in
both validators; the alternative fix would have modified frozen validator
source.

*Resolution:* Phase-8 outputs live in the phase-owned `artifacts/phase8_*`
namespace. `results/final_test/` was never created. Storage only — no numerical
content changed.

**A third question resolved without a stop.** The addendum ranks `GRU_R1` on
five-seed means and marks it stochastic without naming a seed, which reads like
the ambiguity the prompt says to stop on. It is not: exactly one GRU_R1 weight
artifact exists, because Phase-7 seeds 43–46 saved validation prediction arrays
only. Human review confirmed seed 42 as the evaluable artifact.

## 4. Upstream validators

All eight run immediately before the receipt, by
`scripts/build_phase8_receipts.py --preopening`:

| Validator | Gates | Pass | Fail | Exit |
|---|---|---|---|---|
| `validate_foundation.py` | 24 | 24 | 0 | 0 |
| `validate_phase1.py` | 31 | 31 | 0 | 0 |
| `validate_phase2.py` | 58 | 58 | 0 | 0 |
| `validate_phase3.py` | 47 | 47 | 0 | 0 |
| `validate_phase4.py` | 74 | 74 | 0 | 0 |
| `validate_phase5.py` | 95 | 95 | 0 | 0 |
| `validate_phase6.py` | 97 | 97 | 0 | 0 |
| `validate_phase7.py` | 32 | 32 | 0 | 0 |

458 gates, zero failures — identical to the baseline measured before the
amendment existed. **The registry evolution caused no upstream failure.**

Independent hash sweep over all seven manifests plus the Phase-7 freezes, by a
separate code path: **228 files re-hashed, zero drift**.

Unit tests: **193 passed** (149 pre-existing plus 44 new Phase-8 tests).

## 5. Verification tooling registry

| | SHA-256 |
|---|---|
| Phase-7 registry (parent, unmodified) | `a1ec2652df1c6c58bce666ccb8c8bda30743ab1209a5aa7469351ca23d79122f` |
| Phase-8 registry (append-only child) | `c59abf2765fbb0782ea2f52761be3a60aca7e9892242ba4b3c1787fe9facc021` |

All seven inherited validator digests match the parent exactly.
`upstream_validator_sources_changed: false`, `gates_removed: 0`,
`gates_disabled: 0`.

## 6. Environment

CPython 3.11.2, numpy 2.4.6, scipy 1.17.1, lightgbm 4.7.0, torch 2.14.0+cpu,
safetensors 0.8.0, CPU only, 16 torch threads, deterministic algorithms on.
Nothing was installed or upgraded. Every frozen model loaded without error.

## 7. Exact model artifacts

**B3_R2** — four frozen per-horizon boosters, all matching the pre-test freeze:

| Horizon | Candidate | SHA-256 |
|---|---|---|
| 1 h | `B3_C07` | `347b8de0734ebba4…` |
| 6 h | `B3_C02` | `c5a1266a1935dfe5…` |
| 12 h | `B3_C01` | `910c62f2344ac65b…` |
| 24 h | `B3_C01` | `3133a922b953b63f…` |

**GRU_R1** — `results/models/neural/GRU_R1.safetensors`
`37e0d900206a1a9e…b6dd7`, config `0cc119982f02e0fb…56bb35`, **seed 42**, regime
R1, context 48, 31 dynamic / 28 static channels, 24,641 parameters, output
unconstrained. Matches `v2_phase4_manifest.json`.

**B0** — frozen rule from `src/models/baselines.py` (`1f06f357d121b323…`):
observed at origin, else causal carry within 6 h, else station **training**
median.

Nothing retrained, refitted, calibrated, ensembled or reseeded.

## 8. Pipeline fidelity, established before opening

The Phase-8 execution path was validated against frozen **development**
evidence by reproducing the Phase-3/4 validation predictions:

| Check | Result |
|---|---|
| B3_R2 vs frozen validation array | exact float32 match |
| GRU_R1 masked vs unmasked, identical batching | **0.000e+00 — bitwise identical** |
| GRU_R1 vs frozen array, different batch composition | 1.53e-05 max abs |

The masking layer introduces **zero** numerical change. The residual 1.5e-05 is
float32 reduction order from different batch composition inside the frozen
`predict()` path, and appears unmasked too. It is an inherent property of any
run that does not reproduce the identical global batch layout, and it is far
below any reported precision.

## 9. Opening

| Artifact | SHA-256 |
|---|---|
| `phase8_preopening_integrity_receipt.json` | `f5874620337843092a11d31b1cb79ab8d299444608a9ef121579a512ee864c67` |
| `final_test_opening_receipt.json` | `52c5e68caf1b5e40a206b290995a921654b8ee6f9f7f4f22a077633f97c8599b` |

The pre-opening receipt declares `final_test_status: sealed`,
`test_target_accessed: false`, predictions 0, metrics 0, and carries no
performance value. The opening receipt records `AUTHORIZED_TO_OPEN`.

**The first numerical final-test PM2.5 value was accessed only after both
receipts existed.** Status moved to `opened_for_locked_evaluation` and can never
return to sealed.

## 10. Stage A — chronological prediction generation

13 calendar-month blocks, ceilings 26303 → 35062, strictly increasing. All
411,012 canonical samples predicted per model, in 102 seconds.

Reveal ledger:

| | |
|---|---|
| Index checks | 793 |
| Indices checked | 1,233,036 |
| Max index read | 35062 |
| Max forecast origin | 35062 |
| **Future-target access violations** | **0** |

`max_index_read` equals `max_forecast_origin` exactly: no observation later
than a sample's own origin ever entered feature construction.

B0 resolution: 404,852 observed at origin, 4,939 short causal carry, 1,221
station training-median fallback.

## 11. Prediction lock

`artifacts/phase8_prediction_lock.json` — `25751d669a3ff3096728c049db9b593d…`

| Model | Length | dtype | Bytes | SHA-256 |
|---|---|---|---|---|
| `B3_R2` | 411,012 | float32 | 1,644,176 | `cfd1d13cf0f5568f6c3f7b20babfe4ed…` |
| `GRU_R1` | 411,012 | float32 | 1,644,176 | `8656e6a6104074eca01f40724c2aa784…` |
| `B0` | 411,012 | float32 | 1,644,176 | `a2792b5456f5284b6edecdb7d388e62c…` |

Written with `test_metrics_seen: 0`. **No metric existed when the arrays were
frozen.**

## 12. Stage B — scoring

Refused to start until the lock existed and certified the ordering. Reloaded
all three arrays from disk, verified their hashes against the lock, and only
then constructed the target oracle. No model executed during scoring.

---

# CONFIRMATORY LOCKED TEST RESULT

## 13. PRIMARY CONFIRMATORY RESULT

**`B3_R2` — macro station-horizon MAE over 48 equally weighted cells:**

### **31.987226 µg/m³**

## 14. SECONDARY CONFIRMATORY RESULT

**`GRU_R1` (seed 42) — severe MAE, actual > 244.0 µg/m³:**

### **109.332890 µg/m³**  (severe n = **20,336**)

## 15. REFERENCE BENCHMARK

**`B0` persistence** — macro station-horizon MAE **36.138514**, severe MAE
**93.224479**.

---

## 16. Descriptive metrics

| Model | Role | macro MAE | micro MAE | macro RMSE | micro RMSE | macro R² |
|---|---|---|---|---|---|---|
| `B3_R2` | PRIMARY CONFIRMATORY | 31.987226 | 31.979019 | 51.277412 | 55.896373 | 0.540125 |
| `GRU_R1` | SECONDARY CONFIRMATORY | 33.082098 | 33.073378 | 52.378488 | 55.825520 | 0.539579 |
| `B0` | REFERENCE BENCHMARK | 36.138514 | 36.126157 | 58.467386 | 63.936495 | 0.396206 |

### Severe tail (n = 20,336, all 48 cells contributing)

| Model | severe MAE | severe RMSE | mean residual | under-prediction % | severe macro MAE |
|---|---|---|---|---|---|
| `B3_R2` | 131.901121 | 173.423318 | 127.650421 | 88.60 | 131.557564 |
| `GRU_R1` | 109.332890 | 141.985607 | 106.594564 | 93.73 | 109.960239 |
| `B0` | 93.224479 | 134.656962 | 65.549321 | 72.59 | 92.572546 |

### By horizon (macro station MAE)

| Model | 1 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|
| `B3_R2` | 9.306152 | 28.242612 | 39.447567 | 50.952572 |
| `GRU_R1` | 12.865730 | 29.252080 | 39.365864 | 50.844720 |
| `B0` | 10.279825 | 31.929450 | 44.241292 | 58.103486 |

Micro R² at 24 h: B3_R2 0.132, GRU_R1 0.174, B0 **−0.131**.

### By station

B3_R2 ranges from 26.835 (Huairou) to 35.739 (Dongsi). GRU_R1 ranges from
28.378 (Dingling) to 36.700 (Dongsi). Cleanest stations are the northern
suburban sites, hardest the central urban ones, for every model.

### Negative predictions — reported, not clipped

| Model | n negative | % | minimum |
|---|---|---|---|
| `B3_R2` | 53 | 0.0129 | −14.424702 |
| `GRU_R1` | 1,358 | 0.3304 | −28.539839 |
| `B0` | 0 | 0.0000 | +2.0 |

No output was clipped at any point.

## 17. Development → test transfer

| Model | macro MAE dev → test | severe MAE dev → test |
|---|---|---|
| `B3_R2` | 30.1117 → 31.9872 (+6.2%) | 136.612 → 131.901 (−3.4%) |
| `GRU_R1` | 30.7591 → 33.0821 (+7.6%) | 115.793 → 109.333 (−5.6%) |
| `B0` | 33.1277 → 36.1385 (+9.1%) | 101.273 → 93.224 (−7.9%) |

Overall error rose modestly for all three; severe-tail error fell for all
three, the 2016–17 tail being slightly easier than 2015–16. Relative ordering
is preserved on both endpoints.

## 18. Independent verification

`scripts/verify_phase8_independently.py` — **24 checks, all agree.** It
re-parses the index, reads actuals straight from the raw station CSVs,
reimplements every metric from first principles, and reimplements the B0
fallback chain, none of it through the scoring path.

The B3_R2 spot predictions at all four horizons and the GRU_R1 spot prediction
were rebuilt with the reveal ceiling at the sample's **own origin** rather than
the calendar-month block Stage A used, and agree — so the block structure
carried no information.

Also verified independently: the target at τ was refused at its forecast origin
and available once it became history; the severe rule is strictly `>` 244.0;
the residual convention is `actual − prediction`; violations are zero.

## 19. Phase-8 evidence chain

| Artifact | SHA-256 (first 32) |
|---|---|
| `phase8_preopening_protocol_amendment.json` | `abfc1d50e027f132b213ac826b4d2635` |
| `phase8_verification_tooling_registry.json` | `c59abf2765fbb0782ea2f52761be3a60` |
| `phase8_preopening_integrity_receipt.json` | `f5874620337843092a11d31b1cb79ab8` |
| `final_test_opening_receipt.json` | `52c5e68caf1b5e40a206b290995a9216` |
| `phase8_prediction_lock.json` | `25751d669a3ff3096728c049db9b593d` |
| `phase8_primary_results_lock.json` | `5c92451317f8348dbac978a23bd11246` |
| `phase8_final_test_metrics.json` | `eaff81fd1c237671a4a47f36f27da9ad` |
| `confirmatory_endpoints.json` | `d5276586785e29e1339d035943ce90ab` |
| `phase8_figure_manifest.json` | `5dde23098acbcdb772cdfa27e7095bc9` |
| `v2_phase8_manifest.json` | `a83fd1556c2f4a173c4aded4f0d54ecd` |

Figures, drawn only after the primary-results lock, every title carrying LOCKED
FINAL TEST: `phase8_final_test_primary_mae.png` `e56b90dc…`,
`phase8_final_test_severe_mae.png` `59edd43a…`,
`phase8_final_test_mae_by_horizon.png` `f46450dd…`.

## 20. Limitations

1. **The severe tail belongs to persistence.** B0 beats both learned models by
   a wide margin (93.22 against 109.33 and 131.90). Development H5 predicted
   this and the final test confirms it. No learned model in this study forecasts
   severe pollution better than carrying the last observation forward.
2. **Severe residuals are overwhelmingly under-predictions** — 88.6% for B3_R2,
   93.7% for GRU_R1. All three models systematically under-call episodes, the
   neural model worst.
3. **The secondary model's evidence base and its evaluated artifact differ.**
   GRU_R1's rank-2 role rests on five-seed robustness means; the evaluable
   artifact is the single pre-robustness seed-42 model. Its test severe MAE
   (109.33) is better than the seed-42 development value (118.96) and than the
   five-seed development mean (115.79), but one seed cannot carry a seed-spread
   claim, and Phase 7 showed severe-tail metrics swinging 29–35 µg/m³ across
   seeds.
4. **GRU_R1 produces 1,358 negative predictions**, 26× B3_R2's count, minimum
   −28.54. The unconstrained head was a deliberate pre-registration; the cost
   is visible here rather than hidden by clipping.
5. **B0's 24-hour micro R² is negative** (−0.131): at a full day's lead,
   persistence is worse than predicting the test mean, even while it wins the
   severe tail.
6. **Float32 batch-composition sensitivity.** GRU predictions carry an
   order-1e-5 dependence on batch layout, quantified in section 8. Immaterial at
   reported precision, but a full bitwise re-run requires the identical block
   structure.
7. **Single test period, one city.** 2016-03 → 2017-02, twelve Beijing stations.
   Nothing here licenses a claim about other years, cities or pollutants.
8. A stale comment in `src/data/target_history.py` says "only Phase 9 may raise
   it into the test period", from an earlier phase numbering. The file is frozen
   upstream source and was deliberately not edited.

## 21. Discipline preserved

- No model was selected, promoted or demoted after test access. The hierarchy is
  exactly as frozen: `B3_R2` primary, `GRU_R1` secondary, `B0` reference.
- Only those three models were evaluated. No excluded development model, and no
  foundation model, has a final-test artifact.
- No new threshold, subgroup, weighting, ranking criterion or inferential
  procedure was introduced after opening. No p-value, bootstrap or confidence
  interval was invented; none was frozen beforehand, so none is reported.
- No post-test error analysis was performed. Residual ACF, episode segmentation,
  peak/onset analysis, error-case mining, attention diagnostics and conditioned
  slices all belong to a later, clearly labelled phase and are held for human
  review.

---

# POST-OPENING VALIDATOR SEMANTICS

*Added after the locked final test was opened, predicted, locked, scored and
independently verified. It changes no scientific conclusion and no metric
value recorded above.*

## Why five upstream validators return non-zero

`validate_phase3.py` carries a gate called **"No test prediction or test metric
artifact exists"**. It scans `results/validation/predictions`,
`results/validation`, `results/models/b3` and `artifacts/` recursively and
flags any filename containing `test`, excluding only `latest` and `pretest`.

That gate was written while the final test was sealed, and it asserts one
thing: *the locked final test has never been opened.*

- Through Phases 0–7 that assertion was **true**, and the gate passed.
- The authorized Phase-8 one-way opening makes it **intentionally false**.

Two artifacts trip it:

```
artifacts/final_test_opening_receipt.json
artifacts/phase8_final_test_metrics.json
```

The first is the receipt that *authorizes* the transition, mandated by the
locked Phase-8 protocol. **The opening receipt is itself one of the artifacts
that trips the gate.** Its presence is precisely the evidence that the
transition happened under authorization.

`validate_phase4.py` through `validate_phase7.py` nest Phase 3 and therefore
also return non-zero. Verified by parsing every gate line: their *only*
failures are the nested cascade labels — `Upstream validate_phase3.py passes`
and its onward propagation. **No gate owned by Phase 4, 5, 6 or 7 fails.**

## What was deliberately not done

- No upstream validator was modified.
- `scripts/validate_phase8.py`, frozen before the opening, was preserved
  byte-for-byte. Its full-mode 65/70 result is kept as historical evidence
  rather than patched after metrics were seen.
- **No artifact was renamed.** Renaming `final_test_opening_receipt.json` would
  have turned every gate green at zero content cost, since a rename preserves
  the bytes and therefore every recorded hash. It was rejected: the gate is a
  seal tripwire, and silencing a tripwire while the seal stays broken is
  evasion, not repair.
- No gate was removed, disabled, suppressed or reinterpreted.

The failure is evidence of the one-way transition, not evidence of scientific
invalidity.

## The separate post-opening validator

`scripts/validate_phase8_postopening.py` checks the state that now exists. It
is deliberately narrow about what it will tolerate:

| Scope | Requirement |
|---|---|
| Phases 0–2 | PASS outright, at their existing 24 / 31 / 58 gates |
| Phase 3 | **exactly one** failing gate — the seal gate — triggered by **exactly** the two authorized artifacts and nothing else |
| Phases 4–7 | no independently owned failure; only the nested Phase-3 cascade |
| Phase 8 | every non-upstream gate of the frozen validator still passes |
| Results | all four confirmatory values reconcile independently against the frozen artifacts |

Phase 3 is **not** relabelled PASS. It is recorded as
`EXPECTED_HISTORICAL_SEAL_TRANSITION`, and Phases 4–7 as
`SCIENTIFIC_GATES_PASS_WITH_EXPECTED_PHASE3_SEAL_CASCADE`. The distinction is
preserved on purpose.

## State transition, asserted chronologically

| Stage | State |
|---|---|
| Phases 0–7 historical | `sealed` |
| Opening receipt | `AUTHORIZED_TO_OPEN` |
| Stage A | `opened_for_locked_evaluation`, metrics unseen |
| Prediction lock | predictions frozen before scoring |
| Primary-results lock | `evaluated` |
| Current | `evaluated` |

Historical Phase 0–7 files keep their original sealed wording, unchanged. No
current-state Phase-8 artifact describes the final test as sealed, and none
ever will again.

## One correction made to the new validator, before acceptance

The first draft of the post-opening validator asserted the severe rule
backwards: it required targets sitting exactly on the threshold to be counted
as severe, which is `>=`, not the frozen `>`. The gate failed, and the failure
was the validator's, not the study's.

The underlying data settles it: **232 test targets sit at exactly 244.0 µg/m³,
and all 232 are excluded from the severe set.** Under `>=` the severe count
would have been 20,568 instead of 20,336. The frozen strict rule is applied
correctly, and the assertion was corrected to test that boundary samples are
excluded rather than included.

No scientific value changed. The correction is recorded in the post-opening
tooling registry.

## Provenance chain

| Artifact | Role |
|---|---|
| `artifacts/phase8_postopening_validator_correction.json` | the decision and its reasoning |
| `artifacts/phase8_postopening_tooling_registry.json` | append-only; parent is the pre-opening Phase-8 registry, whose parent is the frozen Phase-7 registry |
| `artifacts/phase8_postopening_validation_receipt.json` | what the post-opening run measured |

Neither existing registry was rewritten. The Phase-7 registry remains
`a1ec2652…79122f`, exactly as `artifacts/phase7_pretest_freeze.json` hashes it.

---

## POST-EVALUATION GIT CHECKPOINT SEMANTICS

*Added after the locked final-test evaluation was committed and pushed. No
result is reinterpreted and no metric value above changes.*

The study now has two immutable evidence anchors in `master`:

| Commit | Subject | Role |
|---|---|---|
| `7549426` | Freeze AirSense V2 pre-test research state | the repository exactly as it stood before the sealed test was ever opened |
| `9a787a6` | Freeze AirSense V2 locked final-test evaluation | the locked evaluation, purely additive: 37 files, 5,413 insertions, 0 deletions, 0 modifications |

Neither commit was rewritten, and neither may be.

### Two validators, two correct semantics

`scripts/validate_phase8.py` was frozen before numerical test opening and
asserts **exact equality** between HEAD and the pre-test snapshot. Once the
locked-evaluation checkpoint exists, HEAD has necessarily moved, so its two git
gates — `HEAD is the pre-test evidence snapshot` and `origin/master matches the
pre-test snapshot` — now fail.

That source is preserved byte-for-byte. It is evidence of the protocol as it
stood before the test was opened, and its failing gates are a historical
transition, exactly like the Phase-3 seal gate. It is **not** relabelled as
currently applicable, and it is **not** patched.

`scripts/validate_phase8_postopening.py` validates the repository as it evolves
and therefore uses **immutable freeze-point ancestry** instead:

| | Assertion |
|---|---|
| A | the pre-test snapshot commit exists |
| B | the locked-evaluation commit exists |
| C | the pre-test snapshot is an ancestor of the locked-evaluation commit |
| D | the locked-evaluation commit is an ancestor of current HEAD |
| E | history contains both freeze points, in that order |
| F | `origin/master` equals current HEAD |

Ancestry is tested with `git merge-base --is-ancestor`, which requires a real
ancestry relation. Finding a SHA somewhere in `git log` is explicitly not
accepted, so a rewritten or grafted history cannot satisfy these gates. Both
commit subjects are verified too.

This is stronger than log membership and weaker than equality in exactly one
dimension — the tip — which is the dimension that must be free to move so that
Phase 9 and later legitimate work can descend from the locked-evaluation
checkpoint without either anchor becoming unverifiable.

The post-opening validator also tolerates **exactly** the two historical
exact-HEAD gates of the frozen validator and requires every other gate it owns
to pass, the same treatment already given to the Phase-3 seal cascade.

### Provenance

| Artifact | Role |
|---|---|
| `artifacts/phase8_postcommit_validator_correction.json` | the decision, both validator SHAs, and the semantics change |
| `artifacts/phase8_postcommit_tooling_registry.json` | append-only; parent is the post-opening registry, whose lineage runs back through the pre-opening Phase-8 registry to the frozen Phase-7 registry |
| `artifacts/phase8_postcommit_validation_receipt.json` | what the corrected run measured |

The earlier `phase8_postopening_validator_correction.json` and all three prior
registries were left untouched; they remain historical evidence of the state
they described. Change scope: **Git state semantics only.**
`scientific_assertion_changed: false`, `scientific_result_changed: false`,
`gates_removed: 0`, `scientific_detection_weakened: false`.
