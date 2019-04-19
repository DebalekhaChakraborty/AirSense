# AirSense V1 Reproducibility

How an independent reader reproduces the V1 evidence, and what
"reproduction" can and cannot mean now that the final test is exhausted.

---

## Reference environment

**Hard historical cutoff: 2019-04-26.**

| Component | Reference pin |
|---|---|
| Operating system | Ubuntu 18.04 LTS |
| Interpreter | CPython **3.6.7** |
| numpy | 1.15.4 |
| pandas | 0.23.4 |
| scipy | 1.1.0 |
| scikit-learn | 0.20.0 |
| matplotlib | 3.0.2 |
| seaborn | 0.9.0 |
| jupyter | 1.0.0 |

Frozen in [`requirements-v1-2019.txt`](../requirements-v1-2019.txt). Every
pin was verified against the PyPI JSON API as published before the cutoff,
and every one ships a CPython 3.6 distribution.

### The transitive closure also matters

Installing the seven direct pins **today** resolves their dependencies to
whatever is newest and still importable on CPython 3.6 — in practice 2020–2026
releases. An unlocked install produced **57 post-cutoff distributions out of
69**. Correct direct pins alone do **not** give a historical runtime.

[`requirements-v1-2019-lock.txt`](../requirements-v1-2019-lock.txt) therefore
pins the complete transitive closure to pre-cutoff releases. The verified
runtime contains **57 distributions, none postdating 2019-04-26**, plus
period-consistent packaging tooling (pip 18.1, setuptools 40.6.3,
wheel 0.32.3).

## The current verified runtime

`venv/bin/python` — an in-repo CPython **3.6.7** prefix. Every V1 scientific
command in this document must be run with it. The host system interpreter is
CPython 3.11 and `scripts/preflight.py` will report
`HOST-COMPATIBILITY WARNING` if it is used by mistake.

`venv/` is **not tracked by git**: at ~530 MB it is a build artifact, not
source. What makes it reproducible is the two requirements files; what makes
it verifiable is `artifacts/runtime_snapshot.json`.

### Honest description of the substrate

**This is a period-exact Python environment running on a modern substrate —
not a preserved 2019 Ubuntu machine.**

The interpreter version is exactly 3.6.7 and all 57 Python distributions
predate the cutoff. But the interpreter binary is a modern conda-forge
rebuild (`h357f687_1008`), and the C-library substrate beneath it is current
— OpenSSL 1.1.1w, libsqlite 3.46.0, glibc 2.36, Linux 6.1, on Debian 12
rather than Ubuntu 18.04.

Everything determining the numerical and API behaviour of V1 is
period-exact. What remains modern is the layer below it. A true 2018
substrate would need a container or VM; Docker was not available to this
account. This is recorded rather than glossed over, and it does not block
reproduction.

## Rebuilding the environment

Run from the repository root.

```sh
# 1. bootstrap tooling (not part of the V1 runtime, not kept afterwards)
curl -sSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj bin/micromamba

# 2. the interpreter prefix
./bin/micromamba create -y -p ./venv -c conda-forge --override-channels \
  "python=3.6.7" pip

PY=./venv/bin/python

# 3. period-consistent packaging tooling
"$PY" -m pip install --no-cache-dir pip==18.1 setuptools==40.6.3 wheel==0.32.3

# 4. the V1 runtime, transitively locked to pre-cutoff releases
"$PY" -m pip install --no-cache-dir -r requirements-v1-2019-lock.txt

# 5. verify
"$PY" scripts/preflight.py          # expect: Overall: READY, exit 0
"$PY" scripts/runtime_snapshot.py   # rewrites artifacts/runtime_snapshot.json

# 6. micromamba is not needed again
rm -rf ./bin
```

---

## Phase execution sequence

The authoritative order. Every command uses `venv/bin/python`.

| # | Protocol phase | Command |
|---|---|---|
| 0 | Integrity gate (run before anything) | `venv/bin/python scripts/preflight.py` |
| 3 | Exploratory data analysis | `venv/bin/python scripts/run_eda.py` |
| 4 | Cleaning + chronological split freeze | `venv/bin/python scripts/prepare_data.py` |
| 5 | Feature preparation | `venv/bin/python scripts/prepare_features.py` |
| 6 | M0 naive baseline | `venv/bin/python scripts/run_m0_baseline.py` |
| 7 | M1 linear regression | `venv/bin/python scripts/run_m1_linear_regression.py` |
| 8 | M2 decision tree | `venv/bin/python scripts/run_m2_decision_tree.py` |
| 9 | M3 random forest | `venv/bin/python scripts/run_m3_random_forest.py` |
| 9 | Pre-test development freeze | `venv/bin/python scripts/create_pretest_freeze.py` |
| 10A | Blind final refit and predictions | `venv/bin/python scripts/prepare_final_predictions.py` |
| 10B | Open the 2014 target and evaluate | `venv/bin/python scripts/open_final_test.py` |
| 11 | Post-evaluation error analysis | `venv/bin/python scripts/run_error_analysis.py` |
| 12 | Consolidated summary and claims ledger | `venv/bin/python scripts/build_v1_summary.py` |
| 12 | Evidence registry, manifest, freeze receipt | `venv/bin/python scripts/freeze_v1_evidence.py` |
| 12 | **Final verification** | `venv/bin/python scripts/verify_v1_final.py` |

Supporting utilities: `scripts/audit_dataset.py` (rewrites the raw dataset
audit) and `scripts/runtime_snapshot.py` (rewrites the runtime evidence).

---

## What "reproduction" means now

**This repository is a reproduction after test exhaustion, not a fresh
blinded evaluation.**

The 2014 target was opened once, in Phase 10, and
`artifacts/final_test_opening_receipt.json` records that. Re-running
`scripts/open_final_test.py` **does not** reopen the Phase-4 test partition:
because the receipt exists, it enters verification/report-only mode and
scores the frozen predictions against the frozen target snapshot.

**Do not treat a re-run of Phase 10 as though 2014 were still unseen.** It is
not. Anyone reproducing this work is re-deriving a known result, which is
exactly what reproduction should be — but it is not an independent blinded
test, and no future model variant may claim one on this split.

A genuinely fresh evaluation would require data outside 2010–2014.

## Determinism

Every phase produces byte-identical artifacts across runs, verified in each
phase record:

| Phase | Artifacts | Result |
|---|---:|---|
| 3 — EDA | 15 CSV + 1 JSON + 9 PNG | byte-identical |
| 4 — Cleaning/split | 8 | byte-identical |
| 5 — Features | 9 | byte-identical |
| 6 — M0 | 3 | byte-identical |
| 7 — M1 | 7 | byte-identical |
| 8 — M2 | 8 | byte-identical |
| 9 — M3 | 8 | byte-identical |
| 10A — Blind refit | 6 | byte-identical |
| 11 — Error analysis | 20 | byte-identical |
| 12 — Summary/registry | 5 | byte-identical |

Achieved by: fixing `random_state=42` wherever any randomness exists;
`n_jobs=1` for the forest so results cannot depend on thread scheduling;
writing **no execution timestamp** into any scientific artifact; and pinning
the PNG `Software` metadata field so figures reproduce byte-for-byte rather
than merely pixel-for-pixel.

## Verifying the frozen package

```sh
venv/bin/python scripts/verify_v1_final.py
```

It checks the interpreter version and frozen pins, the raw dataset digest and
dimensions, the evidence registry, the final evaluation and error-analysis
manifests, agreement between the summary table and the frozen final metrics,
the final ranking, the test row count, and the presence and integrity of
every freeze artifact. **It trains nothing, predicts nothing, and repairs
nothing** — it exits non-zero if any gate fails.

## Evidence integrity model

The chain is hash-linked end to end. The raw digest is carried into the
split manifest, the feature schema, each model manifest, the pre-test
development freeze, the blind-prediction freeze, the opening receipt, the
final evaluation manifest and the error-analysis manifest. Each phase
re-verified its predecessors before running and again afterwards.

Three artifacts close the package:

- `artifacts/v1_evidence_registry.csv` — every canonical research file with
  its size and SHA-256, in deterministic path order;
- `artifacts/v1_final_evidence_manifest.json` — the V1 identity, frozen
  specifications, final results and the digests of the closing documents;
- `artifacts/v1_final_freeze_receipt.json` — the freeze declaration.

The registry deliberately excludes those three files, since they are created
after it and cannot self-reference. It also excludes `.git/`, `venv/`,
`__pycache__/`, `*.pyc` and editor or OS metadata.

## Post-freeze policy

Once `artifacts/v1_final_freeze_receipt.json` exists and verifies,
**AirSense V1 is immutable scientific evidence**.

Future work must not silently edit V1 to improve its results, methodology,
features, models or conclusions. If a genuine factual or documentation
correction becomes necessary, it must be recorded transparently as a
**post-freeze correction** that leaves the original scientific evidence
intact.

**Modern AirSense research must proceed separately as V2**, and must not
reuse the 2014 partition as an unseen test set.
