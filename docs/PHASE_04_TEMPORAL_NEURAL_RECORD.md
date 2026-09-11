# AirSense V2 — Phase 4 Record

**Temporal neural baselines: GRU and causal TCN.**
**Executed on branch `master` from `09942928…`.**

Development research only. The locked 2016-03-01 → 2017-02-28 test remains
**sealed**: no test target was read, no test prediction generated, no test
metric computed. `legacy` was not touched. No Transformer, no foundation
model, no R3 spatial model.

Results: [`TEMPORAL_NEURAL_RESULTS.md`](TEMPORAL_NEURAL_RESULTS.md).

---

## 1. Motivation

Phase 3 left two questions that only a different model class could answer.
Gradient boosting on engineered lag and rolling features beat persistence
overall (30.11 against 33.13) but **lost the severe tail badly** (136.61
against 101.27), and its entire advantage came from PM2.5's own history rather
than from meteorology. Phase 4 asks whether a *learned* sequence
representation — one that is not told which lags matter — changes either
result.

Two families with genuinely different inductive biases, and only two: a
recurrent GRU and a causal dilated TCN. No model zoo.

## 2. Gates

Branch `master`. All four upstream validators passed before any model work —
Phase-0 25 gates, Phase-1 32, Phase-2 59, Phase-3 48 — and **199 recorded
hashes across Phases 0–3 were independently re-verified unchanged**. Free disk
17.05 GB, above the 8 GB floor.

## 3. Verification-tooling registry

Phase 3 required a scope correction to the Phase-0/1/2 validators once
legitimate later-phase model artifacts existed. That correction was documented,
but validator *source* had never belonged to any manifest, so the code doing
the verifying carried no provenance of its own.

`artifacts/verification_tooling_registry.json` closes that gap: SHA-256,
purpose, scope, interpreter and current PASS status for all four validators. It
is explicitly **not** a new scientific freeze of Phases 0–3 — no scientific
artifact is re-hashed by it. No validator was modified during Phase 4.

## 4. Training-only internal selection split

**This is the methodological core of the phase.** Phase 3 already spent the
development validation partition selecting among classical candidates. Tuning
every subsequent architecture on that same year would compound selection
optimism until the partition stopped being informative.

So Phase 4 selects hyperparameters on a chronological split **inside training**:

| Split | Window | Samples |
|---|---|---:|
| Internal core (fit) | 2013-03-01 00:00 → 2014-02-28 23:00 | **410,696** |
| Internal tuning (score) | 2014-03-01 00:00 → 2015-02-28 23:00 | **410,488** |

Both are complete March-to-February cycles, tuning is strictly after core, and
they partition the 821,184-sample training universe exactly with zero overlap.
The choice is calendar logic, not performance. Context for an early
internal-tuning sample may reach back into core — both lie inside training, so
no external data is involved.

**The Phase-2 sample universe was not rewritten.** `train.csv`,
`validation.csv`, `test.csv` and `sample_universe_manifest.json` are untouched;
the split is a filter on target timestamp recorded separately in
`artifacts/phase4_internal_split.json`.

The data source used for Stage A was constructed with validation **not
unsealed**, so external validation values were not merely unused — they were
never loaded.

## 5. Environment

`.venv-v2` extended, V1's `venv/` untouched.

| Component | Version |
|---|---|
| CPython | 3.11.2 |
| PyTorch | **2.14.0+cpu** |
| safetensors | 0.8.0 |
| numpy / scipy / lightgbm | 2.4.6 / 1.17.1 / 4.7.0 (Phase-3 pins unchanged) |

PyTorch 2.14.0 was verified as current stable from the authoritative PyPI
metadata. The default linux-x86_64 wheels are **CUDA-enabled** and pull in the
CUDA 13 toolkit; this host has no GPU, so torch was installed from PyTorch's
official CPU index. `torchvision` and `torchaudio` were deliberately not
installed — nothing in Phase 4 needs them. Disk 17.05 → 16.30 GB.

Determinism: `torch.use_deterministic_algorithms(True)` succeeded, seeds
(Python/NumPy/torch) all 42, `set_num_threads(16)`, `set_num_interop_threads(1)`,
DataLoader workers 0, a seeded permutation generator per epoch, no AMP, no
`torch.compile`, no GPU kernels. `torch.cuda.is_available()` is **False**.
float32 training, float64 metric accumulation.

## 6. Architectures

**GRU (N1)**: single-layer unidirectional GRU over the 48-hour window, final
hidden state concatenated with the static vector, head
`Linear(hidden+static → 64) → ReLU → Dropout(0.1) → Linear(64 → 1)`.

**TCN (N2)**: four residual blocks, two kernel-3 causal convolutions each,
dilations 1/2/4/8, **left padding only**. Receptive field
`1 + 2·2·(1+2+4+8) = 61` timesteps, covering the whole 48-hour context; asserted
by formula and by a unit test that perturbs future inputs and requires
bit-identical output at earlier positions. Same head shape.

Both outputs are **unconstrained** — no ReLU, softplus or clipping — so negative
predictions stay visible.

## 7. Joint horizon-conditioned training

One model per architecture × regime rather than per horizon, with horizon as a
4-way one-hot static feature in the frozen order 1/6/12/24. Each canonical
sample still receives exactly one prediction, the Phase-2 sample identities are
unchanged, and metrics are still reported separately by horizon over the same
48 cells. Rationale recorded before results: all horizons share one physical
process, representation sharing is cheap, and the model count drops from 24 to
6 on a CPU-only host.

**Separate-horizon neural models may only appear later as a predeclared
robustness experiment** — not as a reaction to one horizon performing poorly.

## 8. Frozen contract

Feature schema (`artifacts/neural_feature_schema.json`, frozen before any fit):
dynamic channels **R0 = 28, R1 = 31, R2 = 46**; static = 28 (12 station one-hot,
4 horizon one-hot, 6 origin calendar, 6 target calendar). Wind is 17-way
one-hot — 16 frozen training categories plus MISSING — never ordinal. Gap age
is divided by its frozen 168-hour cap. `day_of_week` and `weekend` excluded.

Grids, four candidates each and nothing else varying: GRU hidden {64, 128} ×
lr {3e-4, 1e-3}; TCN channels {32, 64} × lr {3e-4, 1e-3}. Fixed for every
candidate: 8 epochs, batch 1024, AdamW, weight decay 1e-4, gradient clip 1.0,
L1 loss, head 64, dropout 0.1, seed 42, no scheduler, no warmup, **no early
stopping**.

## 9. Fair selection across regimes

Each candidate was scored under **all three regimes** on internal tuning and
ranked by the equal-weight mean of its three macro station-horizon MAEs. This
keeps the search budget identical across R0, R1 and R2 — no regime gets a wider
search — and the winning configuration is then used unchanged for all three.
24 internal fits, 5.4 hours.

| Architecture | Selected | Capacity | LR | Mean internal macro MAE |
|---|---|---:|---:|---:|
| GRU | **GRU_C01** | hidden 64 | 3e-4 | 38.230279 |
| TCN | **TCN_C02** | channels 32 | 1e-3 | 38.079681 |

Smaller capacity won in both families, on MAE — the tie-break's capacity clause
never had to fire. Under an 8-epoch budget, extra capacity mostly bought slower
convergence.

`artifacts/phase4_architecture_selection_freeze.json` was written before any
external prediction. After it, capacity, learning rate, epochs, batch size,
loss and optimizer could not change.

## 10. Full refit and external validation

The six models were refit **from clean seeded initialisation** — never from
candidate weights — on the complete 821,184-sample training universe, then
`artifacts/phase4_external_validation_receipt.json` was written, and only then
was validation history unsealed through the Phase-3 guarded rolling interface.
Each model predicted exactly **413,148** samples, aligned to the canonical
index, and every prediction array was saved.

## 11. What was found

Detailed in [`TEMPORAL_NEURAL_RESULTS.md`](TEMPORAL_NEURAL_RESULTS.md). Four
facts of the phase:

1. **Gradient boosting still wins overall.** Best neural TCN_R1 30.689 against
   B3_R2 30.112 — 1.9% worse. A learned sequence representation did not beat
   engineered lag features on the primary metric.
2. **Neural wins where features are weakest.** On meteorology-only, GRU_R0 and
   TCN_R0 (≈ 38.4) beat B3_R0 (40.62) by 5.5%.
3. **The ordering inverts with horizon.** Neural is far behind at h = 1 (12.4
   against 9.0) and ahead at h = 24 (TCN_R1 45.15 against B3_R2 48.20).
4. **GRU_R1 improves the severe tail by 12.9% over B3_R2** — the first V2 model
   to do so — but persistence still beats every learned model by 17.5%.

## 12. Compute

24 internal fits (5.4 h) + 6 full refits (2.0 h) + 6 determinism refits ≈ 9.5 h
of CPU. GRU ≈ 795 s per full fit, TCN ≈ 1,590 s. All models are under 32k
parameters.

## 13. Seed limitation

Every result uses **seed 42**, for controlled development screening. One seed
does not establish stability, and the 28% GRU/TCN gap on the severe tail is
precisely the sort of finding that could move under reseeding.
**Multi-seed robustness is reserved for Phase 7**, after the development
shortlist is known. Five seeds per candidate now would have cost days of CPU
for evidence the shortlist does not yet justify.

## 13a. Verification-tooling scope correction

After the Phase-4 scientific work was complete, `validate_phase4.py` reported
69 Phase-4 gates passing and **four upstream-validator failures**. All four
were false positives, and all four appeared only because legitimate Phase-4
artifacts now existed. Under the phase's own rule, no validator was touched
until a human approved the correction.

The four observed false positives:

1. **R2 regime versus R² metric — a name collision.** The Phase-0 validator's
   placeholder-metric gate lowercased every dictionary key and treated `r2` as
   the coefficient of determination. `artifacts/neural_feature_schema.json`
   legitimately keys `dynamic_channel_counts` by information regime, so
   `"R2": 46` was read as a stray R² score.
2. **Phase-1 import scanner reading Phase-4 torch code.** Its assertion is that
   *Phase-1 implementation* imports no modelling library, but the scan walked
   all of `src/` and `scripts/`, so `gru_forecaster.py`, `tcn_forecaster.py`,
   `neural_training.py` and the Phase-4 runners tripped it.
3. **Phase-2 result scanner reading Phase-4 MAE figures.** Its assertion is
   that *Phase 2* produced no model result, but its filename patterns matched
   `figures/phase4_validation_mae_by_model.png` and
   `figures/phase4_severe_mae_comparison.png`.
4. **Phase-3 cascade.** `validate_phase3.py` runs the three upstream validators
   and inherited their exit codes. It had no defect of its own and **required
   no edit**: it returned to PASS automatically once 1–3 were corrected.

What was changed, and what was not:

- Metric detection in the Phase-0 validator is now **context-aware** rather
  than broadened: a bare `r2` key still counts as a metric *unless* its
  siblings are exactly the regime vocabulary `{R0, R1, R2, R3}`, and qualified
  spellings (`r2_score`, `macro_r2`, `validation_r2`, `test_r2`,
  `metric_value`, `prediction_score`) are always caught. The correctness of
  this was demonstrated during the fix: the corrected detector immediately and
  correctly identified `macro_R2` in a Phase-4 summary as a genuine metric,
  which was then excluded by ownership rather than by weakening detection.
- The Phase-1 and Phase-2 import scans are now scoped **by ownership** — they
  inspect the files those phases actually created, rather than the whole
  repository. This is a bounded rule, not an ever-growing ignore list.
- Phase-4-owned artifact and figure paths were added to the existing
  later-phase ownership boundaries, each one an exact existing path.

**No scientific assertion was removed, no gate was deleted or disabled, and no
science changed.** Every earlier-phase assertion still holds against its own
phase: Phase 0 still contains no model performance, Phase-1 source still
imports no modelling library, Phase 2 still produced no model result. Gate
counts rose only because ownership scoping made two gates report the file
counts they inspect.

The correction was applied **after** Phase-4 results were complete, on human
approval, and no model was retrained, no prediction regenerated and no metric
recomputed as a consequence. Old and new validator digests, with reasons, are
recorded in `artifacts/verification_tooling_registry.json`.

## 14. Pre-test discipline

No model was refit on validation. The test was not opened. A new pre-test
freeze is created only after all development is complete.

## 15. Git

No write operation: no `git add`, `git commit`, `git push`, reset, rebase,
merge, tag, branch creation or history rewrite. All Phase-4 changes were left
unstaged. Inspection was read-only.
