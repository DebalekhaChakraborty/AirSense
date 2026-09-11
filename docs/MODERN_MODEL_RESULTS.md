# AirSense V2 — Phase 5 Results

**Modern Transformer evaluation on development validation.**

Development validation only: **2015-03-01 → 2016-02-29**, 413,148 samples per
model. **This is not the final test.** The 2016-03-01 → 2017-02-28 test
partition remains sealed and unread.

Method and gates: [`PHASE_05_MODERN_MODELS_RECORD.md`](PHASE_05_MODERN_MODELS_RECORD.md).

Phase 5 executed **one** model family. Chronos-2, the intended zero-shot
foundation model, was stopped by the pretraining-overlap audit and never
downloaded or run — see
[`CHRONOS2_PRETRAINING_AUDIT.md`](CHRONOS2_PRETRAINING_AUDIT.md).

---

## 1. Combined development ranking

Macro station-horizon MAE, the pre-registered primary metric, 48 equally
weighted cells. Phase-3 and Phase-4 values are **reused from frozen
artifacts**, not recomputed.

| # | Model | Family | Macro station-horizon MAE | Source |
|---:|---|---|---:|---|
| 1 | B3_R2 | classical | 30.111722 | Phase 3 |
| 2 | B3_R1 | classical | 30.472594 | Phase 3 |
| 3 | TCN_R1 | neural | 30.688629 | Phase 4 |
| 4 | GRU_R1 | neural | 30.912193 | Phase 4 |
| 5 | GRU_R2 | neural | 31.842957 | Phase 4 |
| **6** | **iTransformer_R1** | **transformer** | **32.480136** | **Phase 5** |
| **7** | **iTransformer_R2** | **transformer** | **32.723452** | **Phase 5** |
| 8 | B0 persistence | classical | 33.127699 | Phase 3 |
| 9 | TCN_R2 | neural | 34.079881 | Phase 4 |
| 10 | TCN_R0 | neural | 38.402739 | Phase 4 |
| 11 | GRU_R0 | neural | 38.436216 | Phase 4 |
| 12 | B3_R0 | classical | 40.623282 | Phase 3 |
| 13 | B1 | classical | 51.205937 | Phase 3 |
| 14 | B2 | classical | 56.733945 | Phase 3 |

**The iTransformer does not win.** It is 7.9% worse than B3_R2, 5.8% worse
than TCN_R1, and 3.1% worse than GRU_R1. It beats persistence by 2.0%, and it
beats TCN_R2 — but sixth place out of fourteen is the result.

It is also the largest model in the study: **109,121 parameters**, roughly
3.4× the GRU and TCN, which are both under 32k. More capacity and a more
modern architecture bought nothing on this metric.

The loss is consistent rather than mixed. Cell by cell, iTransformer_R1 beats
TCN_R1 in **9 of 48** station-horizon cells and B3_R2 in **6 of 48**.

## 2. By horizon

Macro-station MAE at each frozen horizon.

| Model | h = 1 | h = 6 | h = 12 | h = 24 |
|---|---:|---:|---:|---:|
| B0 persistence | **9.901** | 30.106 | 40.882 | 51.621 |
| B3_R2 | 9.022 | **26.457** | **36.763** | 48.204 |
| GRU_R1 | 12.372 | 27.744 | 36.894 | 46.639 |
| TCN_R1 | 12.640 | 28.094 | 36.874 | **45.147** |
| iTransformer_R1 | 12.519 | 29.516 | 39.309 | 48.577 |
| iTransformer_R2 | 12.239 | 30.009 | 39.880 | 48.765 |

Phase 4 found that the neural models lose badly at h = 1 and take the lead at
h = 24. **The iTransformer inherits the first half of that pattern and not the
second.** At h = 1 it is 38.7% worse than B3_R2, the familiar short-horizon
penalty of a learned sequence model against an engineered lag feature. At
h = 24 it does beat persistence by 5.9% and is within 0.8% of B3_R2 — but it
is 7.6% worse than TCN_R1, so the long-horizon crown stays with the causal
convolution.

At h = 6, 12 and 24 it beats persistence; at h = 1 it does not.

## 3. The co-pollutant question — the central result of the phase

Phase 4 found that adding co-pollutant history **hurt** both the GRU and the
TCN, while helping B3 slightly. That left an open question the phase could not
answer: is the R2 penalty a property of *the data* or of *the architecture*?

The iTransformer is the natural instrument for that question, because its
attention runs **across variates** rather than across time. If the penalty
were purely architectural — recurrent and convolutional stacks mishandling
extra channels — attention between variables should remove it.

R1 → R2 increment in macro station-horizon MAE. Positive means co-pollutants
help.

| Architecture | Overall | Severe tail (> 244) |
|---|---:|---:|
| B3 (gradient boosting) | **+1.18%** | **+1.36%** |
| GRU | −3.01% | −21.26% |
| TCN | −11.05% | −20.36% |
| **iTransformer** | **−0.75%** | **−3.61%** |

**The penalty shrinks by a lot and does not disappear.** Cross-variate
attention cuts the overall co-pollutant penalty from −3.0% and −11.1% down to
−0.75%, and in the severe tail from around −21% down to −3.6%. That is a
five-fold to six-fold reduction in the tail.

So the answer is *both*, and mostly architectural in magnitude but not in
sign. Most of what looked in Phase 4 like "co-pollutants are actively
poisonous" was the recurrent and convolutional stacks handling extra channels
badly. What survives the architectural fix is a small, consistent penalty:
even a model built to attend between variables gets no incremental value from
PM10, SO2, NO2, CO and O3 once it already has PM2.5's own history.

The penalty is not uniform — iTransformer_R2 beats iTransformer_R1 in 17 of 48
cells — but it loses in the majority, and it loses on the aggregate.

**H3 remains unsupported for learned sequence models**, now across three
architectures with genuinely different inductive biases. Only gradient
boosting extracts anything from co-pollutants, and only about 1%.

## 4. Severe endpoint

18,636 severe hours (actual > 244.0 µg/m³, threshold frozen from the training
pooled P95).

| Model | Severe MAE | Mean residual | Under-prediction |
|---|---:|---:|---:|
| B0 persistence | **101.27** | 70.34 | **71.6%** |
| GRU_R1 | 118.96 | 116.40 | 93.6% |
| B3_R2 | 136.61 | 132.02 | 88.3% |
| B3_R1 | 138.32 | 133.73 | 88.5% |
| **iTransformer_R1** | **140.99** | 131.18 | **84.0%** |
| GRU_R2 | 145.13 | 143.34 | 95.1% |
| iTransformer_R2 | 145.14 | 140.26 | 89.2% |
| TCN_R1 | 152.53 | 150.73 | 95.4% |

Two things, and they point in opposite directions.

**On accuracy, the iTransformer does not help.** 140.99 is 3.2% worse than
B3_R2, 18.5% worse than GRU_R1, and 39.2% worse than doing nothing at all.
**Persistence has now beaten every learned model in this study for three
consecutive phases**, across gradient boosting, recurrent, convolutional and
attention architectures. That is no longer a quirk of one model class.

**On bias, it is the best learned model in the study.** Its 84.0%
under-prediction rate is the lowest of any learned model — below B3_R2's
88.3%, well below GRU_R1's 93.6% and TCN_R1's 95.4%. Only persistence, at
71.6%, is less biased.

Those two facts together say the iTransformer is *less systematically timid*
in the tail but *not more accurate*: it misses in both directions rather than
almost always low. For an air-quality warning system that trade is not
obviously bad, but on the pre-registered endpoint it is not an improvement.

## 5. Negative predictions — reported, not clipped

The output is unconstrained by design and nothing was clipped.

| Model | Negative predictions | % | Most negative |
|---|---:|---:|---:|
| iTransformer_R1 | 376 | 0.091% | **−74.45** |
| iTransformer_R2 | 670 | 0.162% | −71.94 |
| TCN_R2 | 1,212 | 0.293% | −45.70 |
| GRU_R2 | 818 | 0.198% | −44.21 |
| B3_R2 | 63 | 0.015% | −32.56 |

The iTransformer produces fewer negatives than the Phase-4 R2 models but the
**most extreme single negative value in the whole study**, −74.45 µg/m³. Both
facts are reported rather than clipped away; a physically impossible
prediction is diagnostic information about the model, and hiding it behind a
`max(0, ·)` would only conceal it.

## 6. Cost

| Stage | Fits | Wall clock |
|---|---:|---:|
| Internal candidate selection | 8 | 27,513 s (7.6 h) |
| Full refit, both regimes | 2 | 9,628 s (2.7 h) |
| Determinism refit | 2 | ≈ 9,600 s (2.7 h) |

R1 refit 3,596 s, R2 refit 6,032 s: the 46-variate R2 model costs 68% more
than the 31-variate R1 model, since attention is quadratic in the token count
and tokens *are* variates in this formulation. Inference is cheap — 413,148
predictions in 15 s (R1) and 23 s (R2).

For comparison, Phase 4 spent 9.5 h of CPU for six models. Phase 5 spent
13 h for two, and they placed sixth and seventh.

## 7. Hypothesis status on development validation

| | Statement | Status |
|---|---|---|
| H1 | Causal PM2.5 history improves forecasting | **not tested in Phase 5** — no R0 iTransformer was trained; the Phase-3 and Phase-4 support stands unchanged |
| H2 | The benefit is larger at short horizons | **not tested in Phase 5** — requires the R0 arm |
| **H3** | Co-pollutant history adds incremental value | **not supported on development validation, now across three learned architectures** — iTransformer −0.75% overall and −3.61% severe. The penalty is far smaller than the GRU's −3.0% / −21.3% or the TCN's −11.1% / −20.4%, so most of its *magnitude* was architectural, but the *sign* survives an architecture built to attend across variables |
| **H5** | Modern temporal models reduce upper-tail error and underprediction | **split on development validation** — the underprediction clause is supported: 84.0% is the lowest rate of any learned model in the study. The error clause is not: 140.99 severe MAE is worse than B3_R2, GRU_R1 and persistence |
| H4 | Cross-station context adds benefit | **not evaluated** — R3 was not implemented |

No hypothesis is confirmed. Confirmation is reserved for the locked test,
which has not been opened.

## 8. What Phase 5 could not evaluate

**No foundation model was run at all.** Three of the four 2026 TSFM candidates
surveyed are unusable in a study that intends a genuinely held-out final test:

- **Chronos-2** — licence fine (Apache-2.0), capability fine, CPU inference
  supported. Stopped by the **pretraining-overlap audit**: its corpus names
  KDD Cup (2018), which is Beijing air quality over the same stations and the
  same hourly PM2.5 variable, overlapping the sealed test window by 1,416
  hours per station. No checkpoint was downloaded and no inference was run.
- **TimesFM 3.0** — non-commercial weight licence.
- **Moirai-2.0-R-small** — cc-by-nc-4.0 weights.

That a 120M-parameter zero-shot forecaster cannot be honestly evaluated here
*because it may already have seen the answers* is a finding about the state of
time-series foundation models in 2026, and it is recorded rather than worked
around by quietly using the model anyway.

## 9. Limitations

**One seed.** Seed 42 throughout. The 0.24 µg/m³ R1-to-R2 gap is small enough
that a single seed cannot settle its sign. Multi-seed robustness is reserved
for Phase 7, and this particular result is a strong candidate for it.

**Fixed 8-epoch budget.** Chosen in advance and deliberately not tuned. Both
training curves are still descending at epoch 8 — R1 ends at 29.766 having
come from 30.438, R2 at 29.850 from 30.509 — so these are budget-limited
results. A Transformer may need more epochs than a GRU to reach its own
plateau, and if so the sixth-place finish partly reflects the budget rather
than the architecture. That budget was fixed before any fit precisely so it
could not be tuned into a better-looking answer, and the constraint is
reported rather than relaxed after the fact.

**Small internal search.** Four candidates, varying only d_model and learning
rate, on a CPU-only host. Depth, head count, feed-forward ratio and dropout
were fixed in advance and never searched. A wider search might do better; it
was not affordable and was not attempted.

**Selection asymmetry, unchanged from Phase 4.** B3's Phase-3 numbers were
selected on this same validation year; the iTransformer's were selected on a
training-only split. The comparison is therefore mildly unfavourable to the
iTransformer, which makes its seventh-place finish no worse than it looks —
and no better.

**Development only.** One validation year, one city, one network, no held-out
test.
