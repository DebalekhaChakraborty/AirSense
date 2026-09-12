#!/usr/bin/env python3
"""AirSense V2 - run every validator and interpret the result.

Phase 12B release engineering. Read-only.

Three validators fail BY DESIGN. A runner that simply reported "FAILED" would
train a reader to ignore it, or worse, to go and "fix" a working tripwire. This
script therefore knows which gates are expected to fail, by exact label, and
exits non-zero only when something FAILS THAT SHOULD NOT.

Matching is on gate labels, never on failure counts: a count can coincide while
the underlying failure is different.

    python3 scripts/run_all_validators.py            # everything
    python3 scripts/run_all_validators.py --fast     # skip nested upstream runs
    python3 scripts/run_all_validators.py --metadata-only   # no processed data

Does not write anything.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-v2" / "bin" / "python"
SYS = sys.executable

GATE = re.compile(r"^(.*?) \.{3,} (PASS|FAIL)$")

# Gate labels that are expected to fail, with the reason. Anything else is real.
EXPECTED: dict[str, dict[str, str]] = {
    "validate_phase3.py": {
        "No test prediction or test metric artifact exists":
            "historical seal gate; true through Phase 7, intentionally false "
            "once the test was opened. The artifacts that trip it include the "
            "opening receipt that authorised the transition.",
    },
    "validate_phase4.py": {
        "Upstream validate_phase3.py passes": "nests Phase 3",
    },
    "validate_phase5.py": {
        "Upstream validate_phase3.py passes": "nests Phase 3",
        "Upstream validate_phase4.py passes": "nests Phase 4, which nests Phase 3",
    },
    "validate_phase6.py": {
        "Upstream validate_phase3.py passes": "nests Phase 3",
        "Upstream validate_phase4.py passes": "cascade",
        "Upstream validate_phase5.py passes": "cascade",
    },
    "validate_phase7.py": {
        "Upstream validate_phase3.py passes": "nests Phase 3",
        "Upstream validate_phase4.py passes": "cascade",
        "Upstream validate_phase5.py passes": "cascade",
        "Upstream validate_phase6.py passes": "cascade",
    },
    "validate_phase8.py": {
        "HEAD is the pre-test evidence snapshot":
            "frozen before the opening; HEAD has legitimately advanced",
        "origin/master matches the pre-test snapshot": "same reason",
        "Upstream validate_phase3.py passes": "cascade",
        "Upstream validate_phase4.py passes": "cascade",
        "Upstream validate_phase5.py passes": "cascade",
        "Upstream validate_phase6.py passes": "cascade",
        "Upstream validate_phase7.py passes": "cascade",
    },
    "validate_phase10.py": {
        "Working tree was clean at Phase-10 start":
            "CONDITIONAL - fails whenever ANY tracked file is modified and "
            "uncommitted, including your own work in progress. It is not "
            "detecting drift in frozen evidence: validate_phase12a.py gates "
            "that specifically, with dedicated checks that no frozen Phase "
            "0-11 artifact and no src/ results/ figures/ data/ file is "
            "modified. Expected to pass again on a clean tree.",
    },
    "validate_phase8_postopening.py": {
        "Seal gate is triggered only by the authorized Phase-8 artifacts":
            "frozen after the opening but BEFORE Phase 9. Its authorized "
            "trigger list holds two entries; Phase-9 and Phase-10 artifacts "
            "legitimately added more.",
        "Post-test exploration still not started":
            "same reason: the gate asserts no artifacts/phase9* exist, and "
            "Phase 9 ran.",
        "origin/master equals current HEAD":
            "CONDITIONAL - fails only while a local commit is unpushed. This "
            "one is expected to pass again once master is pushed.",
    },
}

ORDER = [
    ("validate_foundation.py", SYS, [], "metadata"),
    ("validate_phase1.py", SYS, [], "data"),
    ("validate_phase2.py", SYS, [], "data"),
    ("validate_phase3.py", VENV, [], "data"),
    ("validate_phase4.py", VENV, [], "nested"),
    ("validate_phase5.py", VENV, [], "nested"),
    ("validate_phase6.py", VENV, [], "nested"),
    ("validate_phase7.py", VENV, [], "nested"),
    ("validate_phase8.py", VENV, ["--skip-upstream"], "data"),
    ("validate_phase8_postopening.py", VENV, ["--skip-upstream"], "data"),
    ("validate_phase9.py", VENV, [], "data"),
    ("validate_phase10.py", VENV, [], "metadata"),
    ("validate_phase11.py", SYS, [], "metadata"),
    ("validate_phase12a.py", SYS, [], "metadata"),
]


def run(name: str, interpreter: Path | str, extra: list[str]):
    script = ROOT / "scripts" / name
    if not script.is_file():
        return None
    proc = subprocess.run([str(interpreter), str(script), *extra],
                          capture_output=True, text=True, cwd=str(ROOT))
    gates, failures = 0, []
    for line in proc.stdout.splitlines():
        m = GATE.match(line.strip())
        if not m:
            continue
        gates += 1
        if m.group(2) == "FAIL":
            failures.append(m.group(1).strip().lstrip("F ").strip())
    return {"exit": proc.returncode, "gates": gates, "failures": failures}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip validators that re-run the whole upstream chain")
    ap.add_argument("--metadata-only", action="store_true",
                    help="only validators that need no derived data layer")
    args = ap.parse_args()

    if not Path(VENV).exists():
        print(f"note: {VENV} not found; falling back to {SYS} where possible\n")

    selected = []
    for name, interp, extra, kind in ORDER:
        if args.metadata_only and kind != "metadata":
            continue
        if args.fast and kind == "nested":
            continue
        selected.append((name, interp, extra, kind))

    print("AirSense V2 - validator sweep")
    print("Three validators fail by design; they are annotated below.\n")
    print(f"  {'validator':34s} {'gates':>6s} {'fail':>5s}  status")
    print("  " + "-" * 68)

    unexpected: dict[str, list[str]] = {}
    skipped = []
    for name, interp, extra, kind in selected:
        if interp == VENV and not Path(VENV).exists() and kind != "metadata":
            skipped.append(name)
            print(f"  {name:34s} {'-':>6s} {'-':>5s}  SKIPPED (no .venv-v2)")
            continue
        res = run(name, interp, extra)
        if res is None:
            skipped.append(name)
            print(f"  {name:34s} {'-':>6s} {'-':>5s}  ABSENT")
            continue
        exp = EXPECTED.get(name, {})
        new = [f for f in res["failures"] if f not in exp]
        if new:
            unexpected[name] = new
            status = f"*** {len(new)} UNEXPECTED ***"
        elif res["failures"]:
            status = f"ok ({len(res['failures'])} expected)"
        else:
            status = "ok"
        print(f"  {name:34s} {res['gates']:6d} {len(res['failures']):5d}  {status}")

    print()
    if skipped:
        print(f"  skipped: {', '.join(skipped)}\n")
    print("  Expected failures, by design:")
    for name, gates in EXPECTED.items():
        if any(name == s[0] for s in selected):
            for label, why in gates.items():
                print(f"    {name}: {label}")
                print(f"        -> {why}")

    print("\n" + "=" * 70)
    if unexpected:
        print("UNEXPECTED FAILURES - investigate, do not silence:")
        for name, labels in unexpected.items():
            for lab in labels:
                print(f"  {name}: {lab}")
        print("=" * 70)
        return 1
    print("All validators behaved as documented.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
