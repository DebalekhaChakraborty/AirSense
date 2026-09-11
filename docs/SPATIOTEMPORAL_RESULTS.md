# AirSense V2 — Phase 6 Results

**Cross-station spatiotemporal modelling on development validation.**

Development validation only: **2015-03-01 → 2016-02-29**, 413,148 samples per
model. **This is not the final test.** The 2016-03-01 → 2017-02-28 test
partition remains sealed and unread.

Method and gates:
[`PHASE_06_SPATIOTEMPORAL_RECORD.md`](PHASE_06_SPATIOTEMPORAL_RECORD.md).

Phase 6 evaluates **H4**: does synchronized information from the other Beijing
stations add forecasting value beyond a model restricted to the target
station's own history?

---

## 1. The primary H4 test

One architecture, two station-attention masks, **identical capacity** (144,001
parameters each), identical seed, identical budget, identical inputs. The only
primary difference is which stations are reachable.

| | Macro station-horizon MAE |
|---|---:|
| **SA_R2** — a station may attend only to itself | 32.239759 |
| **SA_R3** — a station may attend to all 12 | **31.532702** |
| **Absolute gain** | **+0.707058** |
| **Relative gain** | **+2.193%** |

**H4 is supported on development validation.** The result is unusually
consistent for this study:

- **12 of 12 stations improve.** Not one station is worse under cross-station
  context.
- **36 of 48 station-horizon cells improve**, 12 worsen, none tie.
- The direction held in **all four** internal candidates before external
  validation was ever opened.

This is the first hypothesis in AirSense V2 to be supported by a *paired*
experiment in which capacity, seed, budget and features are all held fixed and
only the information changes. H1 and H2 were supported by comparing regimes
within a model; H3 was refuted the same way. H4 now joins H1 and H2, and it is
the cleanest of the three by construction.

## 2. Where the gain comes from — it is not uniform in horizon

| Horizon | SA_R2 | SA_R3 | Gain | Relative |
|---|---:|---:|---:|---:|
| 1 h | 17.667878 | 17.127256 | +0.540622 | +3.06% |
| 6 h | 28.919698 | 27.025475 | +1.894223 | **+6.55%** |
| 12 h | 36.768381 | 35.778187 | +0.990194 | +2.69% |
| 24 h | 45.603081 | 46.199889 | **−0.596808** | **−1.31%** |

The benefit **peaks at 6 hours and reverses at 24**. Cross-station context is
worth 6.6% at h = 6 and is actively *harmful* at h = 24.

That is a specific, falsifiable shape rather than a general "spatial helps"
claim, and it is the opposite of the H2 pattern for PM2.5's own history, whose
benefit is largest at h = 1 and decays. The two information sources appear to
be useful over different ranges: a station's own recent history dominates the
first hours, the network contributes most in the middle range, and by a day
ahead neither carries much.

At h = 12, **SA_R3 is the single best model in the entire study**: 35.778
against B3_R2's 36.763, TCN_R1's 36.874 and GRU_R1's 36.894. That is the first
time any V2 model has taken a horizon outright from gradient boosting other
than at h = 24.

## 3. The severe tail — the largest effect in the phase

18,636 severe hours, threshold frozen at 244.0 µg/m³ from the training pooled
P95.

| Model | Severe MAE | Mean residual | Under-prediction |
|---|---:|---:|---:|
| B0 persistence | **101.27** | 70.34 | **71.6%** |
| GRU_R1 | 118.96 | 116.40 | 93.6% |
| **SA_R3** | **121.27** | 116.47 | 88.0% |
| B3_R2 | 136.61 | 132.02 | 88.3% |
| iTransformer_R1 | 140.99 | 131.18 | 84.0% |
| SA_R2 | 142.70 | 139.71 | 92.0% |

**Cross-station context improves the severe tail by 15.0%** — 142.70 to
121.27, and on the macro station-horizon severe measure 143.599 to 121.410,
+15.45%. That is by far the largest H4 effect anywhere in the phase, six times
the overall gain.

The interpretation is plausible and worth stating carefully: a severe episode
in Beijing is a regional event, and eleven other stations observing the same
build-up carry information a single station's own history does not. Note this
is a statement about *information*, not about geography or transport — no
coordinates exist in this study and no propagation was measured.

SA_R3 is now the **second-best learned model in the severe tail**, ahead of
B3_R2 by 11.2% and behind GRU_R1 by 1.9%. But **persistence still wins by
16.5%**, and it has now done so in every phase of this study, against
classical, recurrent, convolutional, attention and spatiotemporal models
alike.

## 4. Broader development ranking

| # | Model | Family | Macro station-horizon MAE |
|---:|---|---|---:|
| 1 | B3_R2 | classical | 30.111722 |
| 2 | B3_R1 | classical | 30.472594 |
| 3 | TCN_R1 | neural | 30.688629 |
| 4 | GRU_R1 | neural | 30.912193 |
| **5** | **SA_R3** | **spatial** | **31.532702** |
| 6 | GRU_R2 | neural | 31.842957 |
| **7** | **SA_R2** | **spatial** | **32.239759** |
| 8 | iTransformer_R1 | transformer | 32.480136 |
| 9 | iTransformer_R2 | transformer | 32.723452 |
| 10 | B0 persistence | classical | 33.127699 |
| 11–16 | TCN_R2, TCN_R0, GRU_R0, B3_R0, B1, B2 | mixed | 34.08 – 56.73 |

Explicit comparisons, unfavourable ones included:

| Comparison | Outcome |
|---|---|
| SA_R3 vs SA_R2 | **SA_R3 better by 2.19%** — the primary H4 result |
| SA_R3 vs B3_R2 | SA_R3 **worse** by 4.72% |
| SA_R3 vs TCN_R1 | SA_R3 **worse** by 2.75% |
| SA_R3 vs GRU_R1 | SA_R3 **worse** by 2.01% |
| SA_R3 vs iTransformer_R1 | SA_R3 better by 2.92% |
| SA_R3 vs B0 | SA_R3 better by 4.82% |
| SA_R3 vs B0, severe | SA_R3 **worse** by 19.7% |
| SA_R3 vs GRU_R1, severe | SA_R3 **worse** by 1.94% |
| SA_R3 vs B3_R2, severe | SA_R3 better by 11.23% |

**Gradient boosting still leads overall.** Cross-station information did not
change that. What it did was move a spatiotemporal model from below
persistence to fifth place, and to within 2% of the best learned temporal
model.

### An important caveat on the absolute level

SA_R2 and SA_R3 sit lower in this table than the paired result alone would
suggest, and there is a structural reason that is **not** a property of the
architecture. Grouped batching processes 512 groups — about 6,144 canonical
samples — per optimizer step. Under the frozen 8-epoch budget that yields
roughly **544 optimizer steps**, where Phase 4's sample-batched models received
about **3,208**. Both training curves are still descending steeply at epoch 8
(SA_R2 34.78 → 34.20, SA_R3 33.35 → 32.66).

The Phase-6 models are therefore materially under-trained relative to the
Phase-4 and Phase-5 models they are tabled beside, and their absolute position
in this ranking should be read with that in mind. This does **not** touch the
primary H4 result: SA_R2 and SA_R3 receive exactly the same budget, so the
paired contrast is unaffected. It is precisely why the brief defines H4 as the
paired contrast rather than a comparison against B3.

## 5. Attention diagnostics — descriptive only

**These are model diagnostics, not causal influence estimates, and no
geographical claim is made anywhere: this study has no station coordinates.**

**The model does use the network, but almost uniformly.** Mean attention on a
station's own representation is **0.0882**; mean total attention on the other
eleven is **0.9118**. Against a uniform baseline of 1/12 = 0.0833, self-weight
is barely elevated. The full weight matrix spans only 0.0609 to 0.1410.

So SA_R3's advantage does not come from discovering a few strong relationships.
It behaves much closer to a **network-wide average** than to selective
relational routing. The share is also flat across horizons — self-attention
0.0882 at every horizon — even though the *benefit* varies from +6.55% to
−1.31%. Whatever produces the horizon-dependent gain, it is not a
horizon-dependent shift in where the model looks.

Two predeclared rank diagnostics, both computed against **training-only**
statistics:

| Diagnostic | Spearman |
|---|---:|
| Mean off-diagonal attention vs frozen training pairwise PM2.5 correlation (132 directed pairs) | **−0.137** |
| Mean attention received vs training-period mean PM2.5 at the source station | **−0.762** |

The first is the comparison the protocol asked for, and it is essentially
null: attention does **not** track which station pairs are most correlated.

The second is more informative. The model attends most to the **least polluted
stations** — Dingling (training mean 70.97 µg/m³) receives the most attention
at 0.1238, Wanliu (90.87 µg/m³) the least at 0.0636. A consistent reading is
that the low-background stations act as a regional baseline reference, which
the model uses to place the target station's own reading in context. That
reading is a **hypothesis suggested by a descriptive diagnostic**, not a
result: nothing here was designed to test it, and it should not be reported as
a mechanism until something is.

In the severe subset, self-attention rises slightly, from 0.0882 to 0.0953.

## 6. Negative predictions — reported, not clipped

| Model | Negatives | % | Most negative |
|---|---:|---:|---:|
| SA_R2 | 149 | 0.036% | −10.69 |
| SA_R3 | **4** | **0.001%** | **−0.68** |

Cross-station context nearly eliminates physically impossible predictions —
149 down to 4, and the worst excursion from −10.69 to −0.68. Nothing was
clipped; this is a property of the fitted model, not of a post-hoc constraint.
It is a side effect, not something the phase set out to produce, but it is
consistent with the network view stabilising the estimate.

## 7. Cost

| Item | Value |
|---|---|
| Parameters, each mode | 144,001 |
| Internal selection | 8 fits, 2,396 s |
| Full refit | SA_R2 763.8 s, SA_R3 765.7 s |
| Inference | 15.2 s / 15.1 s for 413,148 predictions |
| Throughput | ≈ 27,100 predictions/s |
| Serialized model | 577,156 bytes each |
| Peak RSS | 3.18 GB |
| Grouped batch | 512 groups ≈ 6,144 samples |

CPU-only on one host; no hardware-independent efficiency claim is made.

Grouping is what made the phase affordable. A naive representation would have
encoded the 12-station network once per station sample; grouping encodes it
once per origin, so a full refit costs 764 s — comparable to a single-station
Phase-4 GRU — while seeing all 12 stations.

## 8. Hypothesis status on development validation

| | Statement | Status |
|---|---|---|
| H1 | Causal PM2.5 history improves forecasting | **supported** — unchanged from Phases 3–4 |
| H2 | The benefit is larger at short horizons | **supported** — unchanged from Phases 3–4 |
| H3 | Co-pollutant history adds incremental value | **not supported for learned models** — unchanged from Phase 5 |
| **H4** | **Cross-station context adds benefit** | **supported on development validation** — +2.19% overall, 12 of 12 stations, 36 of 48 cells, +15.0% in the severe tail; but **negative at h = 24** (−1.31%) |
| **H5** | Modern methods reduce severe-tail error and underprediction | **weakly supported, unchanged in conclusion** — SA_R3 is the second-best learned model in the tail and improves B3_R2 by 11.2%, but persistence still beats every learned model by 16.5% |

No hypothesis is confirmed. Confirmation is reserved for the locked test,
which has not been opened.

## 9. Limitations

**One seed.** Seed 42 throughout. The overall gain of 0.707 µg/m³ is
comfortably larger than anything a seed is likely to move, and 12-of-12
station agreement is hard to produce by chance — but the h = 24 reversal of
−0.597 is small enough that one seed cannot settle its sign. Multi-seed
robustness is reserved for Phase 7, and the h = 24 cell is the specific thing
to re-run.

**Budget-limited, and more so than earlier phases.** See §4. The 8-epoch
budget was frozen before any fit precisely so it could not be tuned after
seeing results; the consequence is that Phase-6 absolute numbers understate
the architecture, and are reported that way rather than adjusted.

**Attention is nearly uniform.** A model that averages the network is a weaker
claim than a model that learns structure. The 2.19% gain is real, but the
mechanism looks closer to regional averaging than to relational discovery, and
§5 should not be read as evidence of learned inter-station structure.

**No coordinates, therefore no spatial claim.** Nothing here establishes
distance effects, upwind/downwind behaviour or transport. The experiment is
about *relational context*, and external coordinates, graph construction and
leave-one-station-out spatial generalisation remain Phase-7 work.

**Development only.** One validation year, one city, one network, no held-out
test.
