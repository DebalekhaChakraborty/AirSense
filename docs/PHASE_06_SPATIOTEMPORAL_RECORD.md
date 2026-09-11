# AirSense V2 — Phase 6 Record

**Cross-station spatiotemporal modelling: the H4 experiment.**
**Executed on branch `master` from `09942928…`.**

Development research only. The locked 2016-03-01 → 2017-02-28 test remains
**sealed**: no test target was read, no test prediction generated, no test
metric computed. `legacy` was not touched. No foundation model, no external
station coordinates, no validation-derived graph.

Results: [`SPATIOTEMPORAL_RESULTS.md`](SPATIOTEMPORAL_RESULTS.md).

---

## 1. Why H4 is being evaluated now

H4 — *cross-station context gives additional benefit beyond single-station
temporal history* — has been preregistered since Phase 0 and **unevaluated
through five phases**. Every model so far has seen exactly one station.

It is being evaluated now, and not earlier, because the three temporal phases
had to come first for the comparison to mean anything. Phase 3 established
what engineered features on one station achieve, Phase 4 what learned sequence
models achieve, and Phase 5 what attention across variables achieves. Phase 5
in particular closed the co-pollutant question: adding more *variables* at the
target station does not help a learned model. That makes the remaining
question sharp — if more variables at one site do not help, does the same site
observed from eleven other places?

## 2. Why the primary experiment needs no coordinates

The official UCI package contains **no station coordinates**. They were not
scraped and not invented.

That constraint turned out to be clarifying rather than limiting. There are two
different questions hiding inside "does space help":

1. **relational** — does synchronized information from other monitoring sites
   improve the forecast?
2. **geometric** — does *distance-structured* information, a graph built from
   station positions, improve it?

Phase 6 answers the first, which is prior to the second and needs no geometry:
learned attention across stations discovers its own weighting from training
data. Coordinates, graph construction and leave-one-station-out spatial
generalisation are Phase-7 work.

Consequently **no geographical claim appears anywhere in this phase** — no
nearby station, no distance effect, no propagation, no upwind or downwind. The
vocabulary is cross-station, network context and relational context, and the
validator scans Phase-6 source for coordinate and distance vocabulary as a
gate.

## 3. Gates

Branch `master`, HEAD `09942928…`, nothing staged. All six upstream validators
passed before any Phase-6 model work — Phase-0 24 gates, Phase-1 31, Phase-2
58, Phase-3 47, Phase-4 74, Phase-5 95 — and **182 recorded digests across the
six phase manifests and the tooling registry were independently re-verified
unchanged** through a code path separate from the validators. Free disk 16.2
GB, above the 8 GB floor.

**No upstream validator was modified in Phase 6.** No Phase-6 artifact
triggered an earlier-phase ownership false positive: `artifacts/phase6_`,
`figures/phase6_` and `results/models/spatiotemporal/` all fall inside
ownership boundaries the Phase-5 correction had already declared, and this was
verified by execution rather than assumed.

## 4. Grouped network representation

Phase 6 needs all 12 stations at one shared forecast origin. The canonical
samples are therefore **regrouped, never rebuilt**:

    group = (target timestamp, horizon)  ->  up to 12 station slots
    group id = <partition>|<YYYYMMDDHH>|h<horizon>

| Partition | Canonical samples | Groups | Slots | Fill |
|---|---:|---:|---:|---:|
| train | 821,184 | 69,745 | 836,940 | 98.12% |
| validation | 413,148 | 35,052 | 420,624 | 98.22% |

**The grouping changes nothing about which predictions count.** Every group
carries the canonical sample row of each observed station target; a slot with
no canonical sample contributes to neither loss nor metric and is never
imputed; a group is never dropped because one of its 12 targets is missing.
The internal split regroups to 34,733 core groups holding exactly **410,696**
samples and 35,012 tuning groups holding exactly **410,488** — identical to the
Phase-4 counts, which is the reconciliation that proves the representation is
faithful.

Grouping is also what made the phase affordable: the network is encoded once
per origin rather than once per station sample, so a full refit costs 764 s.

## 5. Causality across twelve stations

The cross-station requirement is stricter than anything earlier phases needed,
because a leak could now enter through **any** of 12 stations.

* every window is the 48 hours **ending at the shared origin**, for all 12
  stations, so no station can contribute an observation later than the origin;
* validation PM2.5 reaches the loader only through the Phase-3 guarded rolling
  series, forward-filled and never built beyond the unseal ceiling, so the
  locked test period is not on disk in this process at all;
* `GuardedNetworkHistory` keeps the network tensor **private** and refuses any
  origin beyond the ceiling, so the unrestricted validation network is never
  handed to the model loader.

This is asserted rather than assumed. The validator re-derives, for three
sampled groups and again for one probe group, that **every one of the 12
stations' PM2.5 windows equals the Phase-3 guarded accessor's window ending at
that origin**, and that a sealed test origin raises `TargetAccessError`.

## 6. The paired design — the methodological core

Phase 6 must isolate **information** from **capacity**. One architecture is
evaluated in two modes:

* **SA_R2** — a station may attend only to itself;
* **SA_R3** — a station may attend to all 12.

Everything else is identical: the same shared temporal encoder, the same
multi-head attention module with the same trainable parameters, the same
hidden dimension, optimizer, budget, feature channels, station vocabulary,
horizon conditioning, seed and sample universe. The self-only mask is a
registered **buffer**, not a parameter.

Both modes hold **144,001 parameters** and initialise identically from seed 42.
Verified numerically, not asserted:

- SA_R2's prediction is **exactly unchanged** — `torch.equal`, not a tolerance
  — when every non-target station's history is perturbed;
- SA_R2's attention matrix **is the identity**;
- SA_R3's prediction does move under the same perturbation;
- the two state dicts are element-wise equal at initialisation.

These four checks appear in the unit tests, in the validator, and again in the
independent verification against the *saved* models.

## 7. Architecture

**AirSense Station-Attention Forecaster**,
`src/models/station_attention_forecaster.py`. It is **not** Graph WaveNet,
DCRNN, STGCN, GAT or ASTGCN, does not implement any of them, and is not
claimed to. It is a purpose-built controlled spatiotemporal baseline.

1. a **shared** single-layer unidirectional GRU encodes each station's frozen
   48-hour R2 window — the same weights process all 12 stations;
2. multi-head attention across the **station** dimension, 4 heads, dropout 0.1;
3. residual fusion `Z = LayerNorm(H + A)` — no graph convolution, no second
   attention layer;
4. a shared head over each fused station representation concatenated with the
   frozen 28-channel static conditioning, emitting all 12 station predictions
   at once. Output unconstrained; negative predictions stay visible.

**Why the encoder is shared**, documented before results: it isolates the value
of station context from the value of 12 station-specific models, it controls
parameter count, it encourages transfer across sites, it makes SA_R2 and SA_R3
exactly comparable, and it is the only form that supports later
station-generalisation work.

## 8. Feasibility benchmark and the frozen batch size

Required before any candidate fit, and computed on **training data only** with
no predictive metric. The rule was declared before it was run: try 512, step
deterministically to 256 then 128 if peak resident memory exceeds 60% of
physical RAM.

Batch 512 measured 0.6884 s per forward/backward at the worst-case hidden 128,
peak RSS 3.10 GB against a 20.2 GB ceiling — safe on the first try, so **512
was frozen**. Projected total runtime 1.72 h against the 36 h hard gate.
Batch size was never chosen by predictive performance.

## 9. Fair selection

The Phase-4 training-only internal split was reused unchanged — core
2013-03-01 → 2014-02-28, tuning 2014-03-01 → 2015-02-28, both entirely inside
the training partition. Four candidates crossing hidden size {64, 128} with
learning rate {3e-4, 1e-3}, everything else fixed including the 8-epoch
budget, scored by the equal-weight mean of the SA_R2 and SA_R3 internal
macro MAEs. **Eight fits, 2,396 s.**

`SA_C04` (hidden 128, lr 1e-3) won at 35.290346 against SA_C02's 35.601739 —
no tie-break needed — and was used unchanged for both modes, so the
cross-station mode never received a wider search than its own control.

**SA_R3 beat SA_R2 in all four candidates on the internal split**, before
external validation was opened. The gain was near zero for the clearly
under-trained SA_C01 (0.05%) and 1.6–3.2% for the other three.

The external validation partition was not read during selection.

## 10. Full refit and external validation

Both modes were refit **from clean seeded initialisation** on the complete
821,184-sample training universe — 763.8 s and 765.7 s — and the parameter
counts were re-checked equal after training. Only then was
`artifacts/phase6_external_validation_receipt.json` written, and only then was
validation unsealed. Each model predicted every grouped validation origin,
scattered back into the frozen canonical sample order, producing exactly
**413,148** predictions each with no sample dropped and none added.

## 11. Determinism

Both selected models were refit a second time from scratch into a scratch
location and compared against the primary run byte for byte — weight files and
prediction arrays. The refit script performs the comparison itself and exits
non-zero on any difference, reporting the maximum prediction deviation if the
arrays differ.

## 12. What was found

Detailed in [`SPATIOTEMPORAL_RESULTS.md`](SPATIOTEMPORAL_RESULTS.md). Five
facts of the phase:

1. **H4 is supported.** SA_R3 31.532702 against SA_R2 32.239759 — **+2.19%**,
   with **12 of 12 stations** and **36 of 48 cells** improving, under identical
   capacity and seed. The first hypothesis in V2 supported by a fully paired
   experiment.
2. **The benefit is horizon-shaped, and reverses.** +3.06% at h = 1, **+6.55%
   at h = 6**, +2.69% at h = 12, and **−1.31% at h = 24**. At h = 12, SA_R3 is
   the best model in the entire study.
3. **The severe tail is where it matters.** +15.0%, from 142.70 to 121.27 —
   six times the overall gain. SA_R3 becomes the second-best learned model in
   the tail, improving B3_R2 by 11.2%. **Persistence still wins by 16.5%.**
4. **Gradient boosting still leads overall.** SA_R3 places fifth of sixteen,
   4.7% behind B3_R2.
5. **The attention is nearly uniform.** Self-weight 0.0882 against a uniform
   0.0833; the whole matrix spans 0.0609–0.1410. The model uses the network,
   but by something much closer to averaging than to relational discovery.

## 13. Attention diagnostics, and what they are not

Attention weights are exported as **descriptive model diagnostics**. They are
not causal influence estimates and carry no geographical meaning.

Two rank diagnostics were predeclared, both against training-only statistics.
Against the frozen Phase-1 pairwise PM2.5 correlation matrix, Spearman is
**−0.137** across 132 directed pairs — essentially null. Against training-period
mean PM2.5 at the source station, Spearman is **−0.762**: the model attends
most to the *least polluted* stations.

The second suggests the low-background sites serve as a regional baseline
reference. That is a **hypothesis produced by a diagnostic**, not a finding of
this phase, and it is recorded as such.

## 14. Budget limitation, stated rather than corrected

Grouped batching processes ~6,144 canonical samples per optimizer step, so the
frozen 8-epoch budget yields roughly **544 optimizer steps** against Phase 4's
**~3,208**. Both curves are still descending steeply at epoch 8. The Phase-6
models are materially under-trained relative to the models they are tabled
beside, and their absolute ranking understates the architecture.

The budget was frozen before any fit precisely so it could not be tuned after
seeing results, and §29 of the phase brief forbids changing epochs to make
numbers look better. It is therefore reported as a limitation and not
adjusted. **It does not affect the primary H4 result**, where both modes
receive exactly the same budget — which is why H4 is defined as the paired
contrast rather than a comparison against B3.

## 15. Seed limitation

Every result uses **seed 42**. The overall gain of 0.707 µg/m³ with 12-of-12
station agreement is unlikely to be a seed artifact, but the h = 24 reversal of
−0.597 is small enough that one seed cannot settle its sign. Multi-seed
robustness is reserved for **Phase 7**, and the h = 24 cell is the specific
thing to re-run.

## 16. Pre-test discipline

No model was refit on validation. The test was not opened, no test prediction
exists, no test metric was computed, and no test target was accessed. A new
pre-test freeze is created only after all development is complete.

## 17. Git

No write operation: no `git add`, `git commit`, `git push`, reset, rebase,
merge, tag, branch creation or history rewrite. All Phase-6 changes were left
unstaged. Inspection was read-only. `legacy` remains at `16c9030c`, recorded in
the Phase-6 manifest so later drift is detectable.
