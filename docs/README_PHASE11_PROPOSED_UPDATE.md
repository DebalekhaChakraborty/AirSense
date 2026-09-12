# Proposed README Update — Phase 11

**This is a proposal. It has NOT been applied to `README.md`.**
Human review is required before any part of it is used.

---

## Why this is urgent rather than cosmetic

The committed `README.md` currently carries this status block:

> **Status: Phase 6 complete — cross-station development modelling
> evaluated.** Development validation is open; the locked 2016-03-01 →
> 2017-02-28 test remains **SEALED** — no test target has been read, no test
> prediction generated, no test metric computed. Nothing below is V2
> performance.

**Every factual assertion in that block became false at Phase 8.** The test was
opened once, under an authorisation receipt; predictions were generated and
hashed; metrics were computed; `final_test_status` is `evaluated` and can never
return to `sealed`.

This is not a missing status line. It is a published, pushed statement that the
repository's own frozen evidence contradicts. Phases 9 and 10 each deferred
presentation changes to Phase 11, which is why it survived this long. It should
be corrected before the repository is shown to anyone in connection with the
manuscript.

---

## Proposed replacement status block

> **Status: Phase 11 — manuscript construction. The study is scientifically
> complete.**
>
> The locked 2016-03-01 → 2017-02-28 test was **opened exactly once**, at Phase
> 8, under a confirmatory hierarchy frozen beforehand, and scored once.
> `final_test_status` is **evaluated**; it can never return to sealed. Phases 9
> and 10 performed post-test exploratory analysis and final synthesis, adding no
> new experiment. Phase 11 drafts the manuscript from frozen evidence and runs
> no new science.

---

## Proposed headline result block

Both halves are mandatory. Reporting either alone misrepresents the study.

> **Headline result (locked final test, C1).** A gradient-boosted model over
> causal pollutant, meteorological and temporal history (`B3_R2`) achieved a
> macro station-horizon MAE of **31.99 µg/m³** against **36.14** for causal
> persistence (`B0`) — an improvement of **11.49%** — and improved on
> persistence **at every horizon** from 1 to 24 hours, over 411,012 samples and
> 48 equally weighted station-horizon cells.
>
> **In the severe tail, persistence won.** Restricted to observed PM2.5 above
> the training pooled P95 of 244.0 µg/m³ (n = 20,336), the prespecified
> secondary model `GRU_R1` reached **109.33 µg/m³** and `B3_R2` **131.90**,
> while the persistence reference benchmark recorded the lowest severe MAE at
> **93.22**. Reliable forecasting of extreme episodes remains the principal open
> problem.

---

## Proposed additional sections

### Research question

> Under strict causal, chronologically partitioned evaluation on a sealed test
> block opened exactly once: how much does a learned model improve on a naive
> causal reference across forecast horizons of 1 to 24 hours, and does that
> improvement extend to severe pollution episodes?

### Frozen model roles

> `B3_R2` primary confirmatory model · `GRU_R1` (seed 42) secondary confirmatory
> severe-tail model · `B0` reference benchmark. Thirteen further models are
> development-only. No foundation model was executed. **No overall winner label
> is assigned**, because the primary model leads overall while the reference
> benchmark leads in the severe tail.

### Reproducibility anchors

> ```
> pre-test research freeze          75494266792e09bbcfa51c00aaacec58c5b0fb4a
> locked final-test evaluation      9a787a6320a3eca2104abfcbf10c8dbd29bfc19b
> post-evaluation semantics         7ccfc7e54f7d9008421cfcf764fcfaacc353e1db
> post-test exploratory closure     8cc3a7ab8a069b9a187904be68f14b9040928ab4
> final synthesis closure           9a1685355713dd56d7cf2e912cb16681eabc0531
> ```
>
> Every commit after the pre-test snapshot is strictly additive. Future-target
> access violations: 0 across 1,233,036 index checks.

### A note on two validators that now fail by design

> `validate_phase3.py` asserts that no test prediction or metric artifact
> exists. That was true through Phase 7 and is intentionally false now; the
> artifacts that trip it include the opening receipt that authorised the
> transition. `validate_phase8.py` was frozen before opening and asserts an
> exact repository state that has legitimately advanced. **Neither is a defect
> and neither may be "fixed".** Renaming artifacts to silence the Phase-3 gate
> would be evading a tripwire. Current-state verification is carried by
> `validate_phase8_postopening.py`.

### Manuscript status

> A venue-neutral manuscript draft is in `manuscript/`. It is **not ready for
> submission**: thirteen literature citations remain unverified and appear as
> `REF-NEEDED` placeholders. No result, table, figure or quantitative claim
> depends on any of them — all 235 quantitative claims trace to hash-pinned
> artifacts.

---

## What must NOT change

- The existing statement that V1 is complete, frozen on `legacy`, and never
  modified.
- Any wording implying V2 performance is comparable to V1 performance. It is
  not: different tasks, periods, designs and severe thresholds.
- The absence of any claim that the study is operationally ready.

---

## Application instructions

Do not apply automatically. A human should:

1. Confirm the replacement status block is accurate as of the commit where it
   lands.
2. Decide how much of the headline result belongs in the README versus the
   manuscript.
3. Update the final-synthesis-closure SHA in the anchors block if the README
   change is committed on top of a later commit.
4. Commit the README change separately from any manuscript change, so the
   correction of a false published status is legible in the history on its own.
