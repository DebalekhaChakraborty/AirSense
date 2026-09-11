# AirSense V2 — Phase 5 Model Audit

**Protocol Phase 5, §10. Verified against authoritative sources before any
installation or execution.**

Audit date: 2026-09-08. Every fact below was read from an official paper,
repository, model card or package metadata during this phase. Nothing is
written from memory or from the phase brief.

---

## 1. Candidates and outcome

| Model | Role in Phase 5 | Outcome |
|---|---|---|
| **iTransformer-backbone forecaster** | T1, supervised temporal Transformer | **executed** |
| **Chronos-2** | F1, zero-shot TSFM | **NOT executed** — pretraining-overlap gate, classification C |
| TimesFM 3.0 | landscape only | **excluded** — weight licence |
| Moirai-2.0-R-small | landscape only | **excluded** — weight licence |

---

## 2. iTransformer (T1) — executed

| Field | Value |
|---|---|
| Origin | *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, Liu et al., ICLR 2024 (arXiv 2310.06625) |
| Reference implementation | `thuml/iTransformer`, also carried in `thuml/Time-Series-Library` |
| Reference licence | MIT (Time-Series-Library) |
| Implementation used here | **in-repository adaptation**, `src/models/itransformer_forecaster.py` |
| Core principle adopted | **time-series variates are tokens**: each variable's whole 48-hour history is embedded into one token, and self-attention runs *across variate tokens* rather than across time steps |
| Parameter count | reported in the results document |
| Runtime | CPU-only, existing `.venv-v2` (PyTorch 2.14.0+cpu) |
| Checkpoint download | none — trained from scratch |

**This is not a byte-for-byte reproduction of the official implementation and
is not claimed to be.** It is called an *AirSense iTransformer-backbone
forecaster*. Adaptations, all documented in the Phase-5 record: a
regression head over the encoded PM2.5 variate token concatenated with
AirSense's frozen static conditioning; no decoder; no positional encoding
across variate tokens (time order lives inside each token's 48-value
embedding); AirSense's frozen 48-hour context and channel order rather than the
paper's benchmark configurations.

**Selection rationale.** Phase 4 showed learned sequence models help overall
but that co-pollutants *hurt* the GRU and TCN. iTransformer's inverted
attention is specifically an attention mechanism *across variables*, which is
the natural test of whether that R2 regression is a property of the data or of
the architecture. It also needs no pretrained weights, so it carries no
licence or contamination exposure.

---

## 3. Chronos-2 (F1) — NOT executed

| Field | Value |
|---|---|
| Model id | `amazon/chronos-2` |
| Licence | **Apache-2.0** — permissive, would have satisfied §11 |
| Parameters | ~120M |
| Architecture | encoder-only, T5-encoder-inspired, group attention for in-context learning across related series and covariates |
| Context | 8,192 |
| Max horizon | 1,024 |
| Covariates | past-only and known-future, real and categorical, natively |
| Package | `chronos-forecasting >= 2.0` |
| CPU inference | explicitly supported per the model card |
| **Checkpoint downloaded** | **NO** |
| **Inference run** | **NO** |

The licence and capability checks passed. The model was stopped by a different
gate: the **pretraining-overlap audit** returned **classification C — direct or
material overlap found**. Chronos-2's pretraining corpus names **KDD Cup
(2018)**, which is a Beijing air-quality dataset covering the same stations and
the same hourly PM2.5 variable, and overlapping AirSense's sealed final-test
window by 1,416 hours per station.

Full reasoning and sources:
[`CHRONOS2_PRETRAINING_AUDIT.md`](CHRONOS2_PRETRAINING_AUDIT.md).

Because §12 requires it, no checkpoint was fetched — so this audit deliberately
records **no** weight SHA-256, config SHA-256, revision pin or checkpoint size.
There is nothing to pin.

---

## 4. TimesFM 3.0 — excluded on licence

| Field | Value |
|---|---|
| Release | TimesFM 3.0, August 2026 |
| Checkpoint | `google/timesfm-3.0-pytorch` |
| Parameters | 200M (+ optional 30M quantile head) |
| Context / horizon | up to 16k / continuous quantile forecast to 1k |
| Code licence | Apache-2.0 |
| **Weights licence** | **`timesfm-non-commercial-license-v1.0`** |

**Excluded from the primary experiment under §11**: its pretrained weights are
restricted to non-commercial use. Weights up to TimesFM 2.5 remain Apache-2.0,
but substituting an older checkpoint to dodge a licence would be a
performance-irrelevant change made for convenience, and was not done. **No
checkpoint was downloaded.**

---

## 5. Moirai-2.0-R-small — excluded on licence

| Field | Value |
|---|---|
| Checkpoint | `Salesforce/moirai-2.0-R-small` |
| Parameters | **11.4M** |
| Architecture | decoder-only universal forecasting transformer |
| Code licence | Apache-2.0 (`uni2ts`) |
| **Weights licence** | **`cc-by-nc-4.0`** |

**Excluded from the primary experiment under §11**: the checkpoint's weights
are non-commercial. This was **verified from the model card**, not assumed from
the phase brief. **No checkpoint was downloaded.**

Worth recording for a later phase: Moirai-2's own card states it was trained on
"a subset of GIFT-Eval Pretrain and Train datasets", "mixup data from
non-leaking subsets of the Chronos Dataset", KernelSynth synthetic series and
internal Salesforce data. If Moirai-2 is ever reconsidered, it needs the same
pretraining-overlap audit Chronos-2 received — a corpus described as a subset
of the Chronos dataset inherits the same question.

---

## 6. Summary of exclusions

| Model | Licence gate | Overlap gate | Executed |
|---|---|---|---|
| iTransformer (in-repo) | n/a — own implementation | n/a — no pretraining | **yes** |
| Chronos-2 | **pass** (Apache-2.0) | **FAIL — classification C** | no |
| TimesFM 3.0 | **fail** (non-commercial weights) | not reached | no |
| Moirai-2.0-R-small | **fail** (cc-by-nc-4.0 weights) | not reached | no |

Three of the four foundation-model candidates surveyed since Phase 0 are
unusable in this study as it stands: two on licence, one on pretraining
contamination. That is itself a finding about the 2026 TSFM landscape for
research that intends a genuinely held-out final test, and it is recorded
rather than worked around.
