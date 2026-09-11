# AirSense V2 — Phase 5 Record

**Modern Transformer and pretrained TSFM evaluation.**
**Executed on branch `master` from `09942928…`.**

Development research only. The locked 2016-03-01 → 2017-02-28 test remains
**sealed**: no test target was read, no test prediction generated, no test
metric computed. `legacy` was not touched. **No foundation model was
downloaded or run.** No R0 iTransformer, no R3 spatial model.

Results: [`MODERN_MODEL_RESULTS.md`](MODERN_MODEL_RESULTS.md).

---

## 1. Motivation

Phase 4 left one question sharply posed. Adding co-pollutant history **hurt**
both learned sequence models — the GRU by 3.0% and the TCN by 11.1% overall,
and both by roughly 21% in the severe tail — while helping gradient boosting
slightly. Two readings fit that evidence equally well:

1. the co-pollutant channels genuinely carry no incremental information once
   PM2.5's own history is present, and the extra dimensions only add noise; or
2. a recurrent stack and a causal convolution simply handle 46 interleaved
   channels badly, and the penalty is an artifact of the architecture.

These have different consequences. Under the first, R2 should be dropped from
the roster. Under the second, it should be kept and given to a model built for
it.

iTransformer separates them. Its defining move is that **variates are tokens**
and self-attention runs *across variables* rather than across time. It is the
natural instrument for asking whether attention between variables recovers
what the GRU and TCN lost, and it needs no pretrained weights, so it carries
no licence exposure and no contamination exposure.

## 2. Gates

Branch `master`. All five upstream validators passed before any Phase-5 model
work — Phase-0 24 gates, Phase-1 31, Phase-2 58, Phase-3 47, Phase-4 74 — and
**90 recorded artifact digests across Phases 3 and 4 were independently
re-verified unchanged** (44 Phase-3, 46 Phase-4), alongside the five upstream
manifest digests. Free disk 15.4 GB, above the 8 GB floor.

## 3. Verification-tooling scope correction

Phase 5 tripped the same class of defect Phase 4 recorded in its §13a, and it
was handled the same way: **no validator was touched until a human approved
the correction.** The correction was applied **before any Phase-5 result
existed** — while internal selection was still running and no external metric
had been computed — so no result could bias the edit.

Two false positives, both ownership defects rather than genuine violations:

1. **The Phase-4 architecture scanner read Phase-5 source.** Its assertion is
   that *Phase 4* implemented no Transformer, TSFM or LSTM, but the scan
   walked all of `src/` and matched
   `src/models/itransformer_forecaster.py` and `itransformer_grid.py` by
   filename. Those files are the declared subject of Phase 5. The scan is now
   scoped to the eleven Phase-4-owned source files, exactly as the Phase-1 and
   Phase-2 import scans already were.
2. **The Phase-0 and Phase-2 ownership boundaries stopped at Phase 4.** The
   Phase-2 forbidden-artifact patterns include `_mae` and `_rmse`, which match
   Phase-5 figure names. `artifacts/phase5_`,
   `artifacts/v2_phase5_manifest.json` and `figures/phase5_` were added to the
   existing later-phase boundaries — declared paths, not a wildcard.

What was **not** done: no gate was removed, none was disabled, no detection
rule was weakened, and no Phase-5 artifact was renamed to slip past a scanner.
Detection was strengthened: a **new** gate now asserts from the saved configs
that every Phase-4 model is a GRU or a TCN and that no Phase-4 model artifact
names another family. `validate_phase1.py` and `validate_phase3.py` needed no
change, verified by execution rather than assumed.

**A recording error was found while verifying this.** The registry recorded
74 gates for the pre-correction Phase-4 validator; executing an unmodified
copy of that file shows it emits **73** gate lines. The same off-by-one
affected all five recorded counts, and its cause is mundane: the trailing
`V2 … VALIDATION: PASS` summary line also ends in `PASS` and was counted with
the gates. No gate was ever missing. The registry now records the exact
counts, the previously recorded ones beside them, and the counting method.
Phase 4 legitimately reads 74 after the correction because one gate became
two.

Old and new validator digests, with reasons, are in
`artifacts/verification_tooling_registry.json` under
`correction_summary_phase5`.

## 4. What was not run, and why

The phase surveyed four modern candidates and executed **one**. The three
exclusions are recorded in full in
[`PHASE5_MODEL_AUDIT.md`](PHASE5_MODEL_AUDIT.md).

**Chronos-2 — stopped by the pretraining-overlap audit, classification C.**
Its licence (Apache-2.0), capability and CPU support all passed. Its
pretraining corpus names **KDD Cup (2018)**, which is a Beijing air-quality
dataset covering the same stations and the same hourly PM2.5 variable, and
overlapping AirSense's sealed final-test window by **1,416 hours per
station**. Validation does not overlap. Under the protocol's §12 the run was
stopped: **no checkpoint was downloaded and no inference was executed**, so
this phase deliberately records no weight digest, revision pin or checkpoint
size for it. There is nothing to pin. Full reasoning:
[`CHRONOS2_PRETRAINING_AUDIT.md`](CHRONOS2_PRETRAINING_AUDIT.md).

**TimesFM 3.0** — excluded on its `timesfm-non-commercial-license-v1.0`
weights. **Moirai-2.0-R-small** — excluded on `cc-by-nc-4.0` weights. Both
verified from the model cards, not assumed. Neither was downloaded.
Substituting an older permissively-licensed checkpoint to dodge a licence was
considered and rejected: it would be a performance-irrelevant change made for
convenience.

The absence of a foundation-model result is therefore a **finding**, not a
gap in execution.

## 5. Environment

No package was installed for Phase 5. The existing `.venv-v2` (CPython 3.11.2,
PyTorch 2.14.0+cpu, NumPy 2.4.6, safetensors 0.8.0) was sufficient, because
the model is implemented in-repository and trained from scratch. No
`transformers`, no `chronos-forecasting`, no `timesfm`, no `uni2ts`, no
`gluonts` — asserted as gates, not as intentions. `~/.cache/huggingface`
contains 28 KB of xet logs predating the phase, no `models--*` tree and no
weight blob: verified, because the audit's claim that nothing was downloaded
has to be checkable.

Determinism: `torch.use_deterministic_algorithms(True)` succeeded, seeds
(Python/NumPy/torch) all 42, `set_num_threads(16)`,
`set_num_interop_threads(1)`, DataLoader workers 0, a seeded permutation
generator per epoch, no AMP, no `torch.compile`, no GPU kernels.
`torch.cuda.is_available()` is **False**. float32 training, float64 metric
accumulation.

## 6. Architecture

**AirSense iTransformer-backbone forecaster** (`T1`),
`src/models/itransformer_forecaster.py`. It adopts the core principle of
iTransformer (Liu et al., ICLR 2024, arXiv 2310.06625): each dynamic channel's
entire 48-hour history is projected into **one token**, and self-attention
runs across those variate tokens. R1 carries 31 tokens, R2 carries 46.

**This is an in-repository adaptation, not a byte-for-byte reproduction**, and
it is named accordingly. The deliberate differences:

- a regression head over the encoded **PM2.5 variate token** concatenated with
  AirSense's frozen static conditioning, since AirSense predicts one scalar at
  one canonical horizon rather than a whole future sequence per variate;
- **no decoder**, and no future sequence enters the encoder;
- **no positional encoding across variate tokens** — token order is channel
  order and carries no sequential meaning; time order lives inside each
  token's 48-value embedding, exactly as the inverted formulation intends;
- AirSense's frozen 48-hour context and channel order rather than the paper's
  benchmark configurations;
- output **unconstrained** — no ReLU, softplus or clipping, so negative
  predictions stay visible.

The validator asserts this **structurally rather than by name**: that the
embedding consumes 48 hours and emits d_model, that the encoder maps
(N, C, d) → (N, C, d) so attention is over variates, that the module has
exactly three children and no decoder, that no parameter is a positional
encoding, and that the head reads the PM2.5 token plus the static vector. One
independent check perturbs the CO channel and confirms the PM2.5 prediction
moves — cross-variate attention is demonstrated to be live, not assumed.

## 7. Frozen contract

Unchanged from Phases 2–4 and re-verified by digest: 48-hour context, horizons
1/6/12/24, the frozen channel order and wind one-hot, the 244.0 µg/m³ severe
threshold from the training pooled P95, residual = actual − prediction, macro
station-horizon MAE over 48 equally weighted cells, no clipping, no severe
balancing, no log target. The Phase-2 sample universe was not rewritten and
the Phase-4 neural feature schema was reused byte-identically.

Only **R1 and R2** were evaluated. R0 was not run: the phase's question is
about co-pollutants, and a meteorology-only arm would have cost 2.7 h of CPU
to answer a question Phases 3 and 4 already answered twice.

## 8. Fair selection

The Phase-4 **training-only internal split** was reused unchanged — core
2013-03-01 → 2014-02-28 (410,696 samples), tuning 2014-03-01 → 2015-02-28
(410,488) — both entirely inside the training partition. Four candidates
crossing d_model {64, 128} with learning rate {3e-4, 1e-3}, everything else
including the 8-epoch budget fixed in advance, scored by the equal-weight mean
of the R1 and R2 internal-tuning macro MAEs. **Eight fits, 7.6 h.**

`IT_C02` (d_model 64, lr 1e-3) won outright at 35.287346 — no tie-break was
needed — and was then used unchanged for both regimes, so neither regime
received a wider search. Notably the **smaller model won at both learning
rates**: d_model 128 was worse in all four comparisons despite 3.8× the
parameters.

The external validation partition was never read during selection.

## 9. Full refit and external validation

Both models were refit **from clean seeded initialisation** — never from
candidate weights — on the complete 821,184-sample training universe, R1 in
3,596 s and R2 in 6,032 s. Only after both refits was
`artifacts/phase5_prevalidation_freeze.json` written, and only then was
validation history unsealed through the Phase-3 guarded rolling interface.
Each model predicted exactly **413,148** samples aligned to the canonical
index. Both models hold 109,121 parameters.

## 10. Determinism

Both models were refit a second time from scratch into a scratch location and
compared against the primary run **byte for byte** — weight files and
prediction arrays both. The comparison is made by the refit script itself,
which exits non-zero on any difference, rather than by eye.

## 11. What was found

Detailed in [`MODERN_MODEL_RESULTS.md`](MODERN_MODEL_RESULTS.md). Four facts:

1. **The iTransformer does not win.** 32.480 (R1) and 32.723 (R2) place sixth
   and seventh of fourteen — 7.9% behind B3_R2, 5.8% behind TCN_R1, ahead of
   persistence by 2.0%. It is the largest model in the study at 109,121
   parameters, 3.4× the GRU and TCN, and it loses to both.
2. **The co-pollutant penalty is mostly architectural in magnitude, but its
   sign survives.** Cross-variate attention cuts the R1→R2 penalty from −3.0%
   (GRU) and −11.1% (TCN) to **−0.75%**, and in the severe tail from ≈ −21% to
   **−3.6%**. It does not turn it positive. **H3 is now unsupported across
   three learned architectures with different inductive biases.**
3. **Phase 4's long-horizon advantage does not transfer.** The iTransformer
   inherits the neural short-horizon penalty at h = 1 but not the h = 24
   lead: 48.577 against TCN_R1's 45.147.
4. **Least biased learned model in the tail, still not accurate there.** Its
   84.0% under-prediction rate is the lowest of any learned model, below
   B3_R2's 88.3% and GRU_R1's 93.6% — but its 140.99 severe MAE is worse than
   both, and **persistence still beats every learned model, now for the third
   consecutive phase.**

## 12. Compute

8 internal fits (7.6 h) + 2 full refits (2.7 h) + 2 determinism refits
(≈ 2.7 h) ≈ **13 h** of CPU for two models. R2 costs 68% more than R1 per fit,
because attention is quadratic in the token count and tokens are variates.

## 13. Seed and budget limitations

Every result uses **seed 42**. The R1-to-R2 gap of 0.243 µg/m³ is small enough
that one seed cannot settle its sign, and it is a prime candidate for the
multi-seed robustness work reserved for **Phase 7**.

Both training curves are **still descending at epoch 8** (R1 30.438 → 29.766,
R2 30.509 → 29.850). These are budget-limited results. A Transformer may need
a longer budget than a GRU to reach its own plateau, so the sixth-place finish
may partly reflect the fixed budget rather than the architecture. That budget
was frozen before any fit precisely so it could not be tuned into a
better-looking answer; it is reported as a limitation rather than relaxed
after seeing the result.

## 14. Pre-test discipline

No model was refit on validation. The test was not opened, no test prediction
exists, and no test metric was computed. A new pre-test freeze is created only
after all development is complete.

## 15. Git

No write operation: no `git add`, `git commit`, `git push`, reset, rebase,
merge, tag, branch creation or history rewrite. All Phase-5 changes were left
unstaged. Inspection was read-only.
