# AirSense V2 — Temporal Neural Baseline Results

> ## DEVELOPMENT VALIDATION — NOT FINAL TEST
>
> Every number here comes from the **2015-03-01 → 2016-02-29** development
> validation partition. The locked 2016-03-01 → 2017-02-28 test remains
> **sealed**: no test target was read, no test prediction generated, no test
> metric computed. Neural hyperparameters were chosen on a training-only
> internal split and never on this partition.

**Protocol Phase 4.** Two temporal neural families — a GRU (N1) and a causal
TCN (N2) — trained jointly across horizons, one model per architecture ×
information regime. Primary metric: macro station-horizon MAE over 48 cells.
Residual convention `actual − prediction`, so positive means under-prediction.
413,148 validation samples; every model predicted all of them.

---

## 1. Combined development ranking

| Rank | Model | Family | macro station-horizon MAE | micro MAE | macro RMSE | macro R² |
|---:|---|---|---:|---:|---:|---:|
| 1 | **B3_R2** | classical | **30.111722** | 30.114311 | 50.782684 | 0.543162 |
| 2 | B3_R1 | classical | 30.472594 | 30.475081 | 51.193995 | 0.534980 |
| 3 | **TCN_R1** | neural | **30.688629** | 30.691581 | 53.763204 | 0.526118 |
| 4 | GRU_R1 | neural | 30.912193 | 30.915224 | 51.961150 | 0.544183 |
| 5 | GRU_R2 | neural | 31.842957 | 31.845706 | 55.399789 | 0.502827 |
| 6 | B0 persistence | classical | 33.127699 | 33.131272 | 57.654653 | 0.404175 |
| 7 | TCN_R2 | neural | 34.079881 | 34.083725 | 60.111196 | 0.412328 |
| 8 | TCN_R0 | neural | 38.402739 | 38.403929 | 62.970165 | 0.388305 |
| 9 | GRU_R0 | neural | 38.436216 | 38.440617 | 62.994225 | 0.389219 |
| 10 | B3_R0 | classical | 40.623282 | 40.626598 | 63.315691 | 0.376350 |
| 11 | B1 | classical | 51.205937 | 51.213139 | 85.073764 | −0.101770 |
| 12 | B2 | classical | 56.733945 | 56.741553 | 85.168816 | −0.101508 |

Phase-3 values are reused verbatim from frozen artifacts; only the six neural
rows are new.

**Gradient boosting still wins.** The best neural model, TCN_R1, is **1.9%
worse** than B3_R2 and 0.7% worse than B3_R1. On the primary metric, an
end-to-end learned sequence representation did **not** beat hand-specified lag
and rolling features.

## 2. Where the neural models do win

Two places, and both are informative.

**Meteorology-only (R0):** GRU_R0 38.436 and TCN_R0 38.403 against B3_R0
**40.623** — a **5.5%** improvement. When there is no PM2.5 history to lag, the
engineered feature set has little to work with, and a learned representation of
the raw meteorological sequence does measurably better. The advantage of
hand-built features is specific to the case where the single most useful
feature — the pollutant's own recent value — is trivially expressible as a lag.

**Long horizons:** the ordering inverts with horizon.

| Model | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---:|---:|---:|---:|
| B0 persistence | **9.901** | 30.106 | 40.882 | 51.621 |
| B3_R2 | **9.022** | **26.457** | 36.763 | 48.204 |
| GRU_R1 | 12.372 | 27.744 | 36.894 | 46.639 |
| GRU_R2 | 14.724 | 29.048 | 37.525 | 46.075 |
| TCN_R1 | 12.640 | 28.094 | **36.874** | **45.147** |
| TCN_R2 | 14.618 | 31.148 | 41.280 | 49.273 |
| GRU_R0 | 30.706 | 34.781 | 40.216 | 48.041 |
| TCN_R0 | 30.496 | 35.349 | 40.185 | 47.581 |

At **h = 1** the neural models are far behind — 12.4–12.6 against B3_R2's 9.02
and even persistence's 9.90. At **h = 24** they are ahead: TCN_R1 **45.147**
beats B3_R2 by **6.3%** and persistence by **12.5%**.

The crossover sits between h = 6 and h = 12. Its shape is consistent with what
Phase 1 measured: at one hour the answer is essentially "the last observation",
which a lag-0 feature encodes exactly and a recurrent encoder can only
approximate; by a day ahead that shortcut is worth little (lag-24
autocorrelation 0.368) and a learned representation of the whole 48-hour
trajectory pays off.

## 3. Information-regime increments

| Architecture | Increment | Overall | h = 1 | h = 6 | h = 12 | h = 24 | Severe |
|---|---|---:|---:|---:|---:|---:|---:|
| GRU | R0 → R1 | **+19.6%** | +59.7% | +20.2% | +8.3% | +2.9% | **+30.2%** |
| GRU | R1 → R2 | **−3.0%** | −19.0% | −4.7% | −1.7% | +1.2% | **−21.3%** |
| TCN | R0 → R1 | **+20.1%** | +58.6% | +20.5% | +8.2% | +5.1% | +7.3% |
| TCN | R1 → R2 | **−11.1%** | −15.7% | −10.9% | −12.0% | −9.1% | **−20.4%** |

Positive means the added information helped.

**Adding PM2.5 history is transformative** — about 20% overall for both
families, and nearly 60% at h = 1, reproducing the Phase-3 pattern almost
exactly.

**Adding co-pollutants actively hurts both neural models.** This is the sharpest
disagreement with Phase 3, where the same increment gave B3 a small consistent
*gain* of +1.2%. Here it costs the GRU 3.0% and the TCN 11.1%, and roughly 20%
on the severe tail for both. The plausible reading is capacity and budget, not
physics: R2 raises the input from 31 to 46 channels while the parameter count
barely moves (GRU 24,641 → 27,521) and the epoch budget is fixed at 8, so the
extra channels dilute rather than inform. A boosted tree ensemble can ignore an
unhelpful feature almost for free; a small recurrent or convolutional encoder
must spend representation on it. **This is a development observation about
these models under this budget, not evidence that co-pollutants are
uninformative.**

## 4. Severe endpoint — the central question

Frozen threshold: **actual PM2.5 > 244.0 µg/m³**, unchanged from Phase 1.
**18,636** severe samples; all 48 cells contribute.

| Model | severe MAE | severe RMSE | mean residual | under-prediction |
|---|---:|---:|---:|---:|
| **B0 persistence** | **101.273** | 152.283 | 70.344 | **71.6%** |
| **GRU_R1** | **118.965** | 163.659 | 116.404 | 93.6% |
| B3_R2 | 136.612 | 188.469 | 132.018 | 88.3% |
| B3_R1 | 138.325 | 189.878 | 133.729 | 88.5% |
| GRU_R2 | 145.126 | 188.702 | 143.345 | 95.1% |
| TCN_R1 | 152.530 | 195.453 | 150.733 | 95.4% |
| B1 | 152.589 | 197.952 | 126.956 | 85.9% |
| TCN_R0 | 164.968 | 208.238 | 162.265 | 94.9% |
| GRU_R0 | 172.043 | 209.364 | 171.524 | 97.5% |
| B3_R0 | 176.945 | 216.103 | 175.348 | 97.0% |
| TCN_R2 | 183.212 | 223.627 | 182.671 | 98.3% |
| B2 | 278.796 | 299.027 | 278.796 | 100.0% |

**GRU_R1 is the best learned model on the severe tail** — 118.97 against
B3_R2's 136.61, a **12.9% improvement**, and the first V2 model of any kind to
improve on the boosted baseline where it matters most.

**Persistence still wins.** B0's 101.27 is **17.5% better** than GRU_R1. And the
under-prediction rates say why: persistence under-shoots 71.6% of severe hours,
every learned model 88–98%. Repeating an already-elevated observation is simply
a better severe-episode strategy than any of these fitted models, all of which
regress toward the middle.

Note also that the two architectures diverge here far more than they do
overall: GRU_R1 118.97 versus TCN_R1 152.53, a 28% gap, despite being within
0.7% of each other on the primary metric. **The overall metric and the severe
endpoint rank these models differently** — the same lesson Phase 3 recorded,
now reproduced across architectures.

## 5. Negative predictions — reported, not clipped

| Model | count | % | minimum |
|---|---:|---:|---:|
| TCN_R2 | 1,212 | 0.293% | −45.70 |
| GRU_R2 | 818 | 0.198% | −44.21 |
| GRU_R1 | 795 | 0.192% | −20.56 |
| TCN_R1 | 346 | 0.084% | −41.12 |
| TCN_R0 | 81 | 0.020% | −5.47 |
| GRU_R0 | 10 | 0.002% | −2.59 |
| B3_R2 (frozen) | 63 | 0.015% | −32.56 |

The neural heads are unconstrained by design, and they produce roughly an order
of magnitude more physically impossible values than the boosted models. **None
was clipped**: a non-negative head was not preregistered and adding one after
seeing performance would be exactly the kind of post-hoc repair the protocol
forbids. The counts remain tiny — under 0.3% at worst.

## 6. Cost

| Model | Parameters | Full-refit training | Inference (413k) |
|---|---:|---:|---:|
| GRU_R0 | 24,065 | 795 s | 19 s |
| GRU_R1 | 24,641 | 797 s | 19 s |
| GRU_R2 | 27,521 | 759 s | 19 s |
| TCN_R0 | 29,345 | 1,594 s | 9 s |
| TCN_R1 | 29,729 | 1,576 s | 9 s |
| TCN_R2 | 31,649 | 1,605 s | 9 s |

CPU-only, 16 threads, no GPU. The TCN costs about twice the GRU to train and
about half to run. All six are small — under 32k parameters — against B3's
400-tree ensembles.

## 7. Hypothesis status on development validation

| | Statement | Status |
|---|---|---|
| **H1** | Causal PM2.5 history improves forecasting | **supported on development validation** — +19.6% (GRU) and +20.1% (TCN), reproducing Phase 3's +25.0% for B3 |
| **H2** | The benefit is larger at short horizons | **supported on development validation** — ≈ +59% at h = 1 decaying monotonically to +2.9% (GRU) and +5.1% (TCN) at h = 24 |
| **H3** | Co-pollutant history adds incremental value | **not supported on development validation for neural models** — −3.0% (GRU) and −11.1% (TCN), the opposite sign to B3's +1.2%. The Phase-3 weak support for B3 stands unchanged |
| **H5** | Modern temporal models reduce upper-tail error | **weakly supported on development validation** — GRU_R1 improves on B3_R2 by 12.9% in the severe tail, the first V2 model to do so, but remains 17.5% worse than plain persistence. The tail is not solved |
| H4 | Cross-station context adds benefit | **not evaluated** — R3 was not implemented |

No hypothesis is confirmed. Confirmation is reserved for the locked test, which
has not been opened.

## 8. Limitations

**One seed.** Every result uses seed 42. A single seed cannot establish
stability, and the GRU/TCN gap on the severe tail is exactly the kind of
finding that could move under reseeding. Multi-seed robustness is reserved for
Phase 7.

**Fixed 8-epoch budget.** Chosen in advance for a CPU-only host, deliberately
not tuned. The training curves are still descending at epoch 8 for several
models, so these are budget-limited results, and the R2 regressions in
particular may partly reflect that budget rather than the information.

**Selection optimism, twice over.** B3's Phase-3 numbers were selected on this
same validation partition; the neural numbers were not, having been selected on
a training-only split. The comparison is therefore mildly *unfavourable* to the
neural models — B3 has had a tuning advantage on this specific year that they
have not.

**Development only.** One validation year, one city, one network, no held-out
test.
