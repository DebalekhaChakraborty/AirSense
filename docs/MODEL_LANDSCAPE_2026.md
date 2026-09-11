# AirSense V2 — Model Landscape Audit

**Protocol Phase 0. Current-tool audit, verified against authoritative sources.**
**Audit conducted: 2026-09-06.**

Every factual claim below was read from an official repository, model card or
vendor research page during this audit. Claims that could **not** be verified
are marked `UNVERIFIED` and must not be relied on. No release date, version,
licence or hardware figure here is written from memory.

**No model is selected in this document.** This is a shortlist and a trade-off
analysis. The roster is frozen later, after hardware and missingness review.

---

## 1. Pretrained time-series foundation models

### TimesFM (Google)

Source: <https://github.com/google-research/timesfm>

| Field | Value |
|---|---|
| Latest release | **TimesFM 3.0, August 2026**; previous 2.5, September 2025 |
| Checkpoint | `google/timesfm-3.0-pytorch` (Hugging Face) |
| Parameters | 200M, plus an optional 30M quantile head |
| Context | up to **16k** ("up from 2048") |
| Horizon | "continuous quantile forecast up to 1k horizon" |
| Multivariate | supported |
| Covariates | "native support for past-only and past-and-future dynamic covariates" |
| Probabilistic | yes — 9 quantiles, 0.1 … 0.9 |
| Zero-shot | yes |
| Fine-tuning | via Hugging Face Transformers + PEFT (LoRA) |
| Framework | PyTorch (primary), Flax alternative; `pip install timesfm[torch]` |
| **Licence** | code Apache-2.0; **3.0 weights are `timesfm-non-commercial-license-v1.0`** — non-commercial only. Weights up to 2.5 are Apache-2.0 |
| GPU/VRAM | not stated in the repository |
| Python | not stated in the repository |

**The licence split is the decisive trade-off.** TimesFM 3.0 is the newest and
longest-context option, but its weights are non-commercial; TimesFM 2.5 is
Apache-2.0. For an open research repository this is a real fork in the road and
must be decided deliberately, not by defaulting to "latest".

### Chronos-2 (Amazon)

Source: <https://huggingface.co/amazon/chronos-2>

| Field | Value |
|---|---|
| Release | model card updated 2026-06-05; original paper 2025-10-17 (arXiv 2510.15821) |
| Parameters | **120M**, encoder-only |
| Architecture | T5-encoder-inspired, with a "group attention mechanism for efficient in-context learning across related series and covariates" |
| Context | 8,192 |
| Horizon | 1,024 steps |
| Multivariate | yes — univariate, multivariate and covariate-informed in one architecture |
| Covariates | past-only and **known-future**, real and categorical, natively |
| Probabilistic | yes — multi-step quantile forecasts, configurable levels |
| Zero-shot | yes; 6 fine-tuned variants documented |
| Framework | `pip install chronos-forecasting>=2.0`, PyTorch |
| **Licence** | **Apache-2.0** |
| Hardware | "supports both GPU and CPU inference"; ~300+ forecasts/second quoted on an A10G |

**The best fit for AirSense on paper**: smallest of the three, permissively
licensed, explicitly multivariate with covariate support, and the only one whose
card explicitly states CPU inference is supported — which matters a great deal
on this machine (§3).

### Moirai (Salesforce)

Source: <https://github.com/SalesforceAIResearch/uni2ts>

| Field | Value |
|---|---|
| Latest release | **Moirai-2.0-R-small, August 2025** |
| Other checkpoints | `moirai-1.1-R-{small,base,large}` (June 2024); `moirai-moe-1.0-R-{small,base}` (October 2024) |
| Parameters | **not disclosed** in the repository content read |
| Context | "any positive integer" per user configuration; example configs up to 1,680 |
| Patch sizes | auto, 8, 16, 32, 64, 128 |
| Multivariate | supported |
| Covariates | via `feat_dynamic_real_dim` / `past_feat_dynamic_real_dim` |
| Probabilistic | yes — 100 samples by default |
| Zero-shot | yes; fine-tuning workflows supported |
| **Licence** | code Apache-2.0 |
| Python / PyTorch | **not stated** in the repository content read |
| Hardware | **not stated** |

### Ranking evidence — with a caveat

`UNVERIFIED (vendor-sourced).` The GIFT-Eval leaderboard
(<https://huggingface.co/spaces/Salesforce/GIFT-Eval>) is a JavaScript-rendered
Space and **could not be read directly during this audit**. Ranking claims
therefore come from vendor material — the Chronos-2 paper (arXiv 2510.15821)
and the Amazon Science announcement — which report Chronos-2 first among
pretrained models on GIFT-Eval, ahead of TimesFM-2.5 and TiRex, with Moirai-2
also placing highly among non-data-leaking entries.

These are **the model author's own claims about their own model**. They are
recorded as claims, not as independent evidence. Before any V2 model choice
rests on relative ranking, the leaderboard must be read directly.

### Leads not verified in this pass

`UNVERIFIED` — named in sources encountered but not independently checked:
**TiRex**, **Sundial**, **Time-MoE**, **Lag-Llama**, **TabPFN-TS**,
**Chronicle**. No claim is made here about their release dates, licences,
capabilities or performance. If any enters the roster, it gets its own verified
entry first.

## 2. Train-from-scratch architectures

Source: <https://github.com/thuml/Time-Series-Library> (THUML)

| Field | Value |
|---|---|
| Licence | **MIT** |
| Python | `conda create -n tslib python=3.11` |
| PyTorch | `torch==2.5.1` recommended; CUDA 12.1 / 11.8 builds referenced |
| Implements | PatchTST, iTransformer, TimesNet, DLinear, TimeMixer, TSMixer, TiDE, Crossformer, MICN, SCINet, Mamba, SegRNN, Koopa, FreTS, FiLM, Autoformer, FEDformer, Informer, Reformer, Pyraformer, Non-stationary Transformer, ETSformer, LightTS, TFT, TimeXer |
| Also wraps | Chronos2, TiRex, Sundial, Time-MoE, Chronos, TimesFM |
| Exogenous variables | yes — TimeXer, "Empowering Transformers for Time Series Forecasting with Exogenous Variables", `./scripts/exogenous_forecast/` |
| Hardware | no explicit requirement stated; CUDA setup documented throughout |

This single MIT-licensed library covers **both** train-from-scratch candidates
named in the V2 brief (PatchTST, iTransformer) *and* offers wrappers for several
foundation models — which would let one harness serve Phases 4–5 instead of
three incompatible ones. Its documented Python 3.11 target also matches this
machine's system interpreter (§3).

Recurrent and convolutional baselines (LSTM, GRU, TCN) need no external
checkpoint and would be implemented directly in `src/models/`.

## 3. Trade-offs that actually decide this

**Hardware dominates.** This machine has **no GPU** and **1.9 GB of free
disk** (see [`ENVIRONMENT_PLAN.md`](ENVIRONMENT_PLAN.md)). That reorders
everything:

- A 120M-parameter model with documented CPU inference (Chronos-2) is
  plausible; a 200M model with unstated CPU behaviour (TimesFM 3.0) is a risk;
  large checkpoints are out of reach until disk is resolved.
- Train-from-scratch Transformers on 420k rows are feasible on 20 CPU cores but
  slow; recurrent and boosted baselines are comfortable.
- Every foundation-model checkpoint must be counted against the disk budget
  before anything is downloaded.

**Licence matters for an open repository.** Chronos-2 (Apache-2.0), Moirai
(Apache-2.0 code), TSLib (MIT) are unencumbered. TimesFM 3.0 weights are
non-commercial; TimesFM 2.5 is not. If TimesFM is used, the version choice is a
licence decision as much as a capability one.

**Covariates are not optional for AirSense.** V2's whole design is regime-based
(R0 → R3): a model that cannot ingest exogenous covariates cannot express R0,
R2 or R3 at all. Chronos-2 and TimesFM state native covariate support; Moirai
exposes covariate dimensions; TSLib provides TimeXer. Univariate-only
forecasters could serve at most as R1 comparators.

**Zero-shot is a scientific control, not a shortcut.** A zero-shot TSFM that has
never seen Beijing data is a genuinely interesting comparator against models
fitted on the training partition — provided its pretraining corpus is checked
for overlap with this data set. **`UNVERIFIED`: whether any candidate's
pretraining corpus includes the UCI Beijing multi-site data must be established
before a zero-shot result is reported**, or an apparent zero-shot win may simply
be memorisation. This is a Phase-5 gate.

## 4. Provisional shortlist

Not a roster. A shortlist to evaluate against hardware and licence review.

| Role | Candidate | Why |
|---|---|---|
| Persistence / seasonal / climatology | own implementation | B0–B2; no dependency |
| Boosted lag-feature baseline | gradient boosting library, TBD | B3; strong, cheap, CPU-friendly |
| Classical sequence model | LSTM or GRU, own implementation | tests temporal memory with minimal machinery |
| Train-from-scratch Transformer | PatchTST and/or iTransformer via TSLib (MIT) | the two named candidates, one harness |
| Foundation model, primary | **Chronos-2** (Apache-2.0, 120M, CPU inference stated, covariates) | best capability-per-byte on this hardware |
| Foundation model, secondary | TimesFM 2.5 (Apache-2.0) or 3.0 (non-commercial) | longest context; licence decision required |
| Spatiotemporal (Track B) | graph-temporal family, **not yet surveyed** | deferred with Track B |

## 5. What this audit could not establish

1. GIFT-Eval standings — leaderboard not directly readable; vendor claims only.
2. Moirai parameter counts, Python/PyTorch requirements, hardware needs.
3. TimesFM GPU/VRAM and Python requirements.
4. Any fact about TiRex, Sundial, Time-MoE, Lag-Llama, TabPFN-TS, Chronicle.
5. Whether any candidate's pretraining corpus overlaps this data set.
6. Actual CPU-only throughput for any candidate on this machine.
7. The Track-B spatiotemporal landscape, not surveyed in this phase.

Items 1–5 must be closed before the roster is frozen; item 6 needs a measured
smoke test once an environment exists; item 7 belongs to Track B.
