# Contributing to AirSense V2

This is a **completed scientific study**, not an evolving project. That changes
what contribution means here, so please read this before opening a pull
request.

---

## The one rule that matters

> **The frozen evidence chain is immutable. Phases 0–11 artifacts, model
> weights, prediction arrays, metric tables and manifests are historical
> records of what was done, when it was done. They are never edited, never
> regenerated, never "corrected".**

The study's central claim is that a sealed final test was opened exactly once
under a hierarchy frozen beforehand. That claim is only worth anything because
the evidence can be shown not to have moved afterwards. A commit that rewrites
a frozen artifact destroys the thing the repository exists to demonstrate —
even if the new value is more accurate.

If you believe a frozen result is wrong, **open an issue. Do not open a pull
request that changes it.**

## Three validators fail on purpose

A fresh contributor running the validators will see failures. Three are
expected, and "fixing" any of them would silence a tripwire that is working
correctly.

| Validator | Why it fails | What NOT to do |
|---|---|---|
| `validate_phase3.py` | Asserts no test prediction or metric artifact exists. True through Phase 7; intentionally false once the test was opened. The artifacts that trip it include the opening receipt that *authorised* the transition. | **Do not rename artifacts to silence it.** That is evading a seal tripwire. |
| `validate_phase8.py` | Frozen before the test was opened; asserts HEAD equals the pre-test snapshot. HEAD has legitimately advanced. | **Do not patch it.** Its source is preserved byte-for-byte as historical evidence. `validate_phase8_postopening.py` carries current-state verification. |
| `validate_phase8_postopening.py` | Frozen after the opening but **before Phase 9**. Asserts that post-test exploration has not started and that only two artifacts trip the seal gate. Phases 9 and 10 legitimately advanced past both. | **Do not widen `AUTHORIZED_SEAL_TRIGGERS`** or relax the `phase9*` glob. |

Phases 4–7 fail only by nesting Phase 3; they own no independent failures.

Two further gates are **conditional**, not by-design, and `make validate`
labels them as such:

- `validate_phase10.py: Working tree was clean at Phase-10 start` — fails
  whenever any tracked file is modified and uncommitted, including your own
  work in progress. It is *not* detecting drift in frozen evidence;
  `validate_phase12a.py` gates that specifically.
- `validate_phase8_postopening.py: origin/master equals current HEAD` — fails
  only while a local commit is unpushed.

Both pass again on a clean, pushed tree.

A validator that can pass without establishing anything is worse than no
validator. If you add a gate, make it fail when it verifies nothing — see the
non-vacuity floors in `validate_phase11.py` for the pattern.

## What contributions are welcome

- **Release engineering**: packaging, CI, documentation, developer ergonomics.
- **Tests against existing behaviour.** New tests are always welcome. They must
  test the code as frozen; if a test only passes after changing production
  code, that is a finding to report, not a change to make.
- **Bug reports** against the code, with a reproduction.
- **Independent reproduction attempts**, and reports of where they diverge.

## What will be declined

- Retraining, refitting or rescoring any model.
- New predictions or metrics presented as study evidence.
- Changes to hypothesis status or the confirmatory hierarchy.
- Backdated artifacts, or anything shaped to look like it predates the test
  opening. Retrospective metadata is fine and must say so — see
  `artifacts/phase12_phase7_source_provenance.json` for the required framing.
- Post-hoc promotion of a development-only model.

## House style for new phases

Each phase in this study follows the same discipline, and additions should too:

1. Freeze the plan **before** producing results.
2. Finalise the validator **before** producing outputs.
3. Verify through a **separate code path** from the one that produced the values.
4. Ship a validator that exits non-zero on failure.
5. Write manifests with no volatile timestamps.
6. Append to registries; never rewrite an earlier one.

## Before you open a pull request

```bash
make test        # unit suite
make validate    # all validators, with the three expected failures annotated
```

Both must behave as documented. `make validate` reports the expected failures
explicitly rather than hiding them; a *new* failure is a real one.

## Git

The repository owner manages all commits to `master`. Please work on a branch
and open a pull request rather than pushing.

## Licensing of contributions

Code contributions are accepted under [MIT](LICENSE); documentation and content
under CC BY 4.0. See [`LICENSES.md`](LICENSES.md) — note especially that the
dataset in `data/raw/` is **not** covered by either and remains under its own
UCI terms.
