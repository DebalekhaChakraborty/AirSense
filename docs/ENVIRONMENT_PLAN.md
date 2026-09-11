# AirSense V2 — Environment Plan

**Protocol Phase 0. Hardware audit and environment recommendation.**
**Audited: 2026-09-06. Nothing was installed.**

---

## 1. Machine audit

| Component | Observed |
|---|---|
| OS | Debian GNU/Linux 12 (bookworm) |
| Kernel | Linux 6.1.0-52-cloud-amd64, x86_64 |
| CPU | Intel Xeon @ 2.20 GHz |
| Cores | 20 logical = 1 socket × 10 physical × 2 threads |
| RAM | 31 GiB total, ~23 GiB available, **no swap** |
| Disk | single root filesystem `/dev/sda1`, 99 G total, 93 G used, **1.9 G available (99% full)** |
| **GPU** | **none** — `nvidia-smi` absent, no `/dev/nvidia*`, no CUDA (`nvcc` absent), no ROCm, no MPS. `lspci` shows only a Virtio SCSI controller |
| System Python | CPython 3.11.2 (`/usr/bin/python3`) |
| Installed packages | numpy 2.4.6, requests 2.28.1, urllib3 1.26.12, PyYAML 6.0. **pandas is not installed** |
| pip | 23.0.1 |
| uv | **0.9.2** — can provision CPython 3.8 … 3.14 |
| conda / mamba | absent |
| Docker | binary present, but its config is unreadable by this user — treat as unavailable |

## 2. Two findings that govern everything

### 2.1 There is no GPU — this is CPU-only work

Every modern-forecasting assumption has to be re-examined. Consequences:

- PyTorch must be the **CPU wheel**. CUDA builds are pointless here and cost
  gigabytes that this machine does not have.
- Transformer training on ~420k rows is feasible on 20 cores but slow;
  epoch budgets and context lengths must be planned for CPU, not GPU.
- Foundation-model use should lean toward **inference**, ideally zero-shot or
  light fine-tuning. Chronos-2 (120M) is the only shortlisted TSFM whose card
  explicitly states CPU inference is supported.
- Track B graph-temporal models must be scoped realistically or deferred.

This is a constraint to design around and report honestly, not a reason to
weaken the protocol.

### 2.2 Disk is 99% full — **1.9 GB free. This is a blocker.**

Rough budget for a minimal V2 environment:

| Item | Approximate |
|---|---|
| CPython 3.11 via uv | ~50 MB |
| PyTorch CPU + deps | ~1.0–1.5 GB installed |
| pandas / pyarrow / scikit-learn | ~300–500 MB |
| Chronos-2 checkpoint (120M) | ~250–500 MB |
| A second TSFM checkpoint (200M+) | ~500 MB–1.5 GB |
| Processed windows and experiment outputs | ~1–5 GB, design-dependent |

Even the leanest viable environment exceeds the free space; several candidates
exceed it on their own. **Phase 2 onward cannot start until this is resolved.**

Phase 0 was completed entirely within the constraint: standard library only,
zero installs, ~48 MB of raw data written.

Options for the human — none taken unilaterally:

1. Extend or add a volume (cleanest; `/home/AI_POC` alone occupies 50 GB).
2. Free space elsewhere on the host.
3. Reclaim the V1 `venv/` (~530 MB, untracked, in this repository). **Not
   recommended and not done**: it is V1 historical evidence and §26 of the V2
   brief forbids modifying it. It is rebuildable from V1's two requirements
   files, but that is a decision for the operator, not for this phase.
4. Keep model checkpoints on a separate mount and point the framework cache at
   it.

## 3. Recommended Python version — evidence, not preference

**Recommendation: CPython 3.11**, provisioned per-project.

The evidence, rather than the assumption:

| Evidence | Bearing |
|---|---|
| System interpreter is already CPython 3.11.2 | no download needed; matches the host |
| THUML Time-Series-Library documents `python=3.11` with `torch==2.5.1` | the harness that carries PatchTST + iTransformer targets exactly this |
| `uv` 0.9.2 can pin any of 3.8–3.14 | if a candidate demands otherwise, switching is cheap |
| Chronos-2 requires `chronos-forecasting>=2.0` on PyTorch; no Python floor stated | no conflict identified |
| TimesFM and Moirai state no Python requirement in their repositories | **`UNVERIFIED`** — must be confirmed before either enters the roster |

3.11 is recommended because two independent lines of evidence point at it, not
because it is conventional. It is **not frozen**: the freeze happens after the
roster is chosen, and if a selected model demands 3.10 or 3.12, that wins.

## 4. Proposed environment shape

- Location: **`.venv-v2/`** at the repository root — clearly distinct from V1's
  `venv/`, and **not created yet**.
- Tool: `uv venv --python 3.11` plus `uv pip install`, for fast, reproducible,
  hash-pinned installs.
- Note: the inherited `.gitignore` ignores `venv/`, `.venv/` and `env/` but
  **would not ignore `.venv-v2/`**. That entry must be added when the
  environment is created.
- Pins: an explicit `requirements-v2.txt` plus a fully resolved lock, so the
  environment is reconstructible — the same discipline V1 used, applied to
  modern packages.
- Installation order once unblocked: numpy/pandas/scikit-learn first, then
  PyTorch CPU, then exactly one foundation-model package, measuring disk after
  each step.

## 5. Explicitly not done

- **Nothing was installed.** No PyTorch, TensorFlow, JAX, Transformers, TimesFM,
  Chronos, Moirai, GluonTS, NeuralForecast, XGBoost, LightGBM — and no pandas.
- `.venv-v2/` was **not** created; the environment is not frozen.
- V1's `venv/` was neither used nor modified. Phase-0 code ran on
  `/usr/bin/python3` (3.11.2) with the standard library only.

## 6. Open questions for review

1. **How is the disk constraint resolved?** Blocking for Phase 2+.
2. Given CPU-only execution, is the intended scope one strong TSFM plus
   train-from-scratch models, rather than a broad TSFM survey?
3. Is a non-commercial weight licence (TimesFM 3.0) acceptable, or is the study
   restricted to permissive weights?
4. Should checkpoints live outside the repository on a separate mount?
