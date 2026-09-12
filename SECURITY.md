# Security Policy

## Scope, stated honestly

AirSense V2 is an **offline scientific analysis repository**. It has a narrow
attack surface, and overstating it would waste your time:

- no network service, no server, no listening socket;
- no authentication, no credential handling, no secret material — the
  repository is scanned for credential-shaped content and carries none;
- no user-supplied input path at runtime; scripts read tracked data files and
  frozen artifacts;
- no database, no deserialisation of untrusted input.

The realistic risks are supply-chain (a compromised dependency) and the
ordinary hazard of running downloaded code.

## Reporting a vulnerability

Open a **private security advisory** through GitHub's *Security → Report a
vulnerability* on this repository. Please do not open a public issue for
something exploitable.

Include what you ran, what happened, and the commit SHA. A response should be
expected in weeks rather than hours: this is a completed research project, not
a maintained service.

## Dependencies

Runtime dependencies are pinned twice — direct pins plus a resolved closure:

| File | Scope |
|---|---|
| `requirements-v2-core-lock.txt` | lightgbm, narwhals, numpy, scipy |
| `requirements-v2-neural-lock.txt` | the above plus torch (CPU) and safetensors |

**These are frozen historical environment records and are never updated**, not
even for a security advisory. They describe the environment that produced the
study's results, and editing them would falsify that record.

If a pinned dependency carries a known vulnerability, the correct response is
to say so in a release note and, if needed, ship a *separate* file for fresh
environments — not to rewrite history. `requirements-release.txt` exists for
that purpose and may be updated freely.

One known gap, documented rather than hidden: **Pillow** is a rendering
dependency of the figure pipeline whose historical version was never recorded.
See [`docs/PHASE_12_REPRODUCIBILITY_NOTES.md`](docs/PHASE_12_REPRODUCIBILITY_NOTES.md).

## Running this code

Everything runs locally and offline. `scripts/acquire_dataset.py` is the only
script that performs a network fetch, and it verifies the archive against a
recorded SHA-256 before use.
