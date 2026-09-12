# AirSense V2 — Supplementary Methods

Technical detail too dense for the main manuscript. This document does not
duplicate the repository; it records the decisions a reader would need in order
to reproduce or audit the study, and points to the hash-pinned artifact that
holds each one.

---

## S1. Sample identity and eligibility

A sample is the triple `(station, target_timestamp, horizon)`. Eligibility
requires all of:

1. the target timestamp lies inside the assigned partition, where partition
   membership follows the **target** timestamp;
2. PM2.5 at the target timestamp is **observed** in the raw data, never imputed;
3. the forecast origin `t = target − horizon` exists in the dataset timeline;
4. every input timestamp is at or before `t`;
5. the context window `[t − 47, t]` lies inside the dataset timeline;
6. missing inputs inside the window are handled by the causal rules in S2 and by
   nothing else.

Partition sample counts are 821,184 train, 413,148 validation and 411,012
locked test. The sample universe is hash-pinned and was not rewritten by any
later phase.

**Boundary context.** For targets near the start of validation or test the
context window legitimately extends into the preceding partition, because those
timestamps precede the forecast origin. Such samples are kept. What makes this
safe is that parameters and preprocessing statistics stay fitted on training
data only: newly observed *past* context may enter an input window, but nothing
is refitted from it.

---

## S2. Missingness logic

Two-tier, because the training gap distribution is bimodal.

| Gap age at origin | Rule |
|---|---|
| ≤ 6 hours | causal forward fill from the most recent observation at or before the origin |
| > 6 hours | station training median, marked as not observed |

The six-hour ceiling is the P90 gap length rounded to a horizon boundary. It is
where the evidence stops supporting the carry: training autocorrelation is still
high at lag 6 and much weaker at 12 and 24 hours.

Long outages are **not dropped**. Outages concentrate by station, so dropping
them would remove data unevenly across stations and seasons and would quietly
bias the macro metric.

Applies identically to PM2.5, the five co-pollutants and the meteorological
channels. Wind direction is forward-filled as a **category** with the same
ceiling, then assigned an explicit missing state — a state, not a guess. No
target-based encoding of any kind is used.

**Prohibited outright:** backward fill; interpolation consulting observations
after the origin; centred rolling statistics; any bidirectional transform; any
statistic fitted on validation or test data.

---

## S3. Masks and gap-age channels

For every imputed variable a binary mask is emitted, computed from the raw value
**before** any fill, so a measurement is always distinguishable from a carried
value. A `time_since_last_observed` channel in hours, capped at 168, accompanies
PM2.5 and each co-pollutant, because gap age spans orders of magnitude and a
mask alone cannot separate a one-hour-stale value from a sixty-day-stale one.

---

## S4. Scaling

Global training robust scaling, `(x − median) / IQR`, with constants pooled
across all twelve stations over the training partition and frozen in the
preprocessing configuration.

- **Robust rather than standardised**, because training PM2.5 is right-skewed
  and a standard-deviation scale is dominated by the tail.
- **Global rather than per-station**, deliberately. Station medians and upper
  quantiles differ materially, and those level differences are precisely the
  signal the macro station-horizon metric exists to compare fairly. Per-station
  centring would erase them and would make held-out-station transfer
  ill-defined.
- **RAIN exception.** Its training median and quartiles are all zero, so the
  robust denominator is undefined; RAIN falls back to a standard-deviation
  scale.
- **No target transform.** PM2.5 remains in native µg/m³ so that MAE is directly
  interpretable.

---

## S5. Calendar and station representation

Frozen calendar channels, all deterministic and knowable at the origin: origin
hour, origin day-of-year, origin month, target hour, target day-of-year, target
month, with cyclic sine/cosine encodings.

`day_of_week` and `weekend` are **excluded** from the frozen set: training PM2.5
autocorrelation at a 168-hour lag gives no support for a weekly cycle.

Station identity is a 12-way encoding over the training stations. It encodes
station, never partition membership or anything derived from a later period.

---

## S6. Gradient-boosting feature schema

Lag positions, in hours before the origin: 0, 1, 2, 3, 6, 12, 24, 47.

**Lag 48 is deliberately excluded.** The canonical window spans exactly 48
timestamps ending at the origin, so an exact 48-hour lag would sit one hour
outside it. Admitting it would mean silently reading a 49th hour. No hidden
49th hour exists anywhere in this study.

Trailing windows, all ending at the origin and looking only backwards: 3, 6, 12,
24, 48 hours, with mean, population standard deviation, minimum and maximum. No
centred rolling window and no exponentially weighted feature.

Additional channels: observed mask at origin and observed fraction over 6, 12
and 48 hours for every numeric variable; gap age at origin for each pollutant in
the regime; `rain_occurred` at origin and its trailing means; a 17-way wind
encoding (sixteen frozen training categories plus MISSING); a 12-way station
encoding; frozen origin and target cyclic calendar.

Horizon is not a feature column, because the boosted models are trained per
horizon.

Feature counts: **R0 = 207, R1 = 240, R2 = 405**.

---

## S7. Hyperparameter grids and selection

**Gradient boosting.** Exactly eight predeclared candidates from a documented
nested loop: learning rate ∈ {0.03, 0.07}, leaves ∈ {31, 63}, minimum data per
leaf ∈ {50, 200}. Fixed across all candidates: 400 boosting rounds, unlimited
depth, bagging and feature fractions 1.0, L1 regularisation 0.0, L2
regularisation 1.0, L1 objective and metric, **no early stopping**. Selection is
macro-station validation MAE at that horizon, with a frozen tie-break order and
a 1e-12 comparison tolerance. Micro MAE was never used for selection.

An early-stopped iteration count was excluded on principle: it is one more
quantity chosen by looking at validation data.

**Neural families.** Hyperparameters were selected on a **training-only internal
split** rather than on the development validation partition, which had already
been spent selecting classical candidates. The split is calendar logic, not
performance: an internal fit block of 410,696 samples over 2013-03-01 to
2014-02-28 and an internal scoring block of 410,488 samples over 2014-03-01 to
2015-02-28. Both are complete annual cycles, the scoring block is strictly
later, and together they partition the training universe exactly with zero
overlap.

Four candidates per family, with everything else fixed at 8 epochs, batch 1024,
AdamW, weight decay 1e-4, gradient clipping at global norm 1.0, L1 loss, head
width 64, dropout 0.1, seed 42, no scheduler, no warmup, no early stopping.

| Family | Grid | Selected |
|---|---|---|
| GRU | hidden ∈ {64, 128} × lr ∈ {3e-4, 1e-3} | hidden 64, lr 3e-4 |
| TCN | channels ∈ {32, 64} × lr ∈ {3e-4, 1e-3} | channels 32, lr 1e-3 |
| iTransformer-style | width ∈ {…} × lr ∈ {…} | width 64, lr 1e-3 |
| Station attention | hidden ∈ {…} × lr ∈ {…} | hidden 128, lr 1e-3, 144,001 parameters |

---

## S8. Architecture detail

**GRU.** Single-layer unidirectional GRU over the 48-hour window; final hidden
state concatenated with the static vector; head `Linear(→64) → ReLU →
Dropout(0.1) → Linear(→1)`.

**TCN.** Four residual blocks, two kernel-3 causal convolutions each, dilations
1/2/4/8, **left padding only**. Receptive field 61 timesteps, covering the whole
context. Causality is asserted twice: by formula, and by a test that perturbs
future inputs and requires bit-identical output at earlier positions.

**iTransformer-style.** In-repository adaptation, not an official reference
implementation. Attention across variates rather than time steps; 2 encoder
layers, 4 heads, feedforward ratio 4, GELU, layer normalisation, dropout 0.1, no
positional encoding across variates, no decoder. R1 and R2 only.

**Station attention.** Shared unidirectional GRU temporal encoder; one 4-head
attention layer along the **station** axis with dropout 0.1; fusion by
`LayerNorm(H + Attention(H))`. No graph convolution, no distance mask, no chosen
neighbour set, no coordinate input. Grouped batch size 512.

Attention masks define the paired contrast: `SA_R2` forbids every off-diagonal
edge, `SA_R3` is unmasked across all twelve stations.

Both neural heads are unconstrained — no ReLU, softplus or clipping — so
negative predictions remain visible.

---

## S9. Joint horizon conditioning

Neural models are trained once per architecture × regime rather than once per
horizon, with horizon supplied as a four-way one-hot static channel in the
frozen order 1/6/12/24. Each canonical sample still receives exactly one
prediction, sample identities are unchanged, and metrics are still reported
separately by horizon over the same 48 cells. The rationale was recorded before
results existed: all horizons share one physical process, representation sharing
is cheap, and the model count drops substantially on a CPU-only host.

---

## S10. Determinism controls

All seeds — Python, NumPy and framework — set to 42. Deterministic algorithm
mode enabled. Thread counts pinned and held identical across candidates.
DataLoader workers zero, with a seeded per-epoch permutation generator. No
mixed precision, no graph compilation, no GPU kernels. float32 training with
float64 metric accumulation.

**One documented caveat.** GRU inference carries an order-1e-5 dependence on
inference batch composition, arising from floating-point reduction order in the
underlying linear algebra. Masking itself is bitwise exact. The effect is
immaterial at the precision reported, but exact bitwise reproduction of the
neural predictions requires the identical batch structure.

---

## S11. Target reveal semantics

Validation and test PM2.5 history reaches feature code only through a rolling
history component whose series is built by forward fill alone, so the value at
any index depends only on observations at or before it, and whose every accessor
asserts that the window ends exactly at the requested forecast origin. During
development the unseal ceiling was set to the end of validation, so test values
were never read from disk at all.

Scoring is deliberately separated: a single module is the only place a target is
read as evaluation truth, it refuses any timestamp outside the permitted window,
and it requires the caller to present already-produced predictions.

---

## S12. Locked-test opening protocol

1. All pre-opening gates green; receipt written; status still sealed.
2. Opening receipt issued, recording authorisation and the freeze hash.
3. Stage A: chronological prediction generation with **metrics unseen**.
4. Prediction lock: arrays hashed **before** any scoring.
5. Stage B: scoring; status becomes evaluated.
6. Post-test analysis works only from those frozen arrays.

Audit counts: future-target access violations **0** across 1,233,036 index
checks; models retrained after opening **0**; models selected after opening
**0**; foundation models executed **0**.

---

## S13. Validator and evidence chain

Each phase carries a validator that exits non-zero on any failure, and an
append-only verification-tooling registry recording the SHA-256 of the code
doing the verifying. Manifests carry no volatile timestamps, so a manifest
difference always means a content difference.

Two validators now fail **by design**, and this is recorded rather than fixed:

- The Phase-3 validator asserts that no test prediction or metric artifact
  exists anywhere. That was true through Stage 5 and is intentionally false once
  the test has been opened. The artifacts that trip it include the opening
  receipt itself — the artifact that authorised the transition. Renaming files
  to silence it would be evading a tripwire, and is prohibited.
- The Phase-8 validator was frozen before opening and asserts an exact repository
  state that has legitimately advanced. Its source is preserved byte-for-byte as
  historical evidence and is never patched; a separate post-opening validator
  carries current-state verification using immutable freeze-point ancestry.

Independent recomputation of headline quantities runs through a **separate code
path** from the code that produced them, in every phase that produces results.

---

## S14. Foundation-model eligibility audit

Conducted **before** any checkpoint download or inference.

- **Chronos-2.** The published pretraining corpus enumerates a dataset whose
  Beijing air-quality content overlaps the sealed evaluation period and network.
  Under the preregistered contamination policy this is a disqualifying overlap
  risk, and the model was not downloaded and not run. The audit established
  overlap risk **by provenance**. It did not demonstrate that any specific
  record entered training, and no such demonstration is claimed.
- **TimesFM-3.0.** Weight licence incompatible with the study's intended use.
- **Moirai-2.0-R-small.** Non-commercial weight licence.

Zero foundation models executed; zero foundation-model predictions generated.
This is a governance outcome and carries no information about how these models
would have performed.
