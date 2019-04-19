"""Record the AirSense V1 execution environment as verifiable evidence.

Writes artifacts/runtime_snapshot.json describing the interpreter, the
platform, the direct frozen dependencies, the complete installed
distribution inventory, the preflight outcome, and the digests of the raw
dataset and both requirements files.

Purpose: later experiments must be attributable to a specific, checkable
runtime. This snapshot is what makes that possible.

Constraints, deliberately matching the rest of the project:

* Python standard library only. No new dependency is introduced for this
  script, and the installed-distribution inventory is read from
  ``*.dist-info`` / ``*.egg-info`` metadata directly rather than through
  ``pkg_resources`` or ``importlib.metadata`` (the latter is 3.8+).
* CPython 3.6-compatible syntax. No dataclasses, no walrus operator, no
  PEP 585 builtin generics, no ``subprocess.run(capture_output=...)``
  (3.7+).
* Read-only with respect to the dataset. The only file written is the
  snapshot artifact itself.

Usage:
    python scripts/runtime_snapshot.py
"""

import io
import json
import os
import platform
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.audit import sha256_of_file  # noqa: E402

HISTORICAL_CUTOFF = "2019-04-26"
REFERENCE_PYTHON = "3.6.7"
REFERENCE_OS = "Ubuntu 18.04 LTS"

DIRECT_MODULES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("jupyter", "jupyter"),
]

# Bootstrap tooling. Recorded separately from the V1 scientific runtime,
# because environment-management and packaging tools are not part of the
# V1 modelling methodology.
BOOTSTRAP_MODULES = [("pip", "pip"), ("setuptools", "setuptools"),
                     ("wheel", "wheel")]

RAW_CSV = os.path.join(
    PROJECT_ROOT, "data", "raw", "PRSA_data_2010.1.1-2014.12.31.csv")
REQUIREMENTS = os.path.join(PROJECT_ROOT, "requirements-v1-2019.txt")
LOCKFILE = os.path.join(PROJECT_ROOT, "requirements-v1-2019-lock.txt")
SNAPSHOT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "runtime_snapshot.json")
PREFLIGHT = os.path.join(PROJECT_ROOT, "scripts", "preflight.py")

PREFLIGHT_MEANING = {
    0: "READY",
    1: "HOST-COMPATIBILITY WARNING",
    2: "NOT READY",
}


def module_version(import_name):
    """Import a module and return its __version__, or a status string."""
    try:
        module = __import__(import_name)
    except ImportError:
        return None
    return getattr(module, "__version__", "importable (no __version__)")


def parse_metadata_name_version(metadata_path):
    """Read Name/Version from a METADATA or PKG-INFO file."""
    name = version = None
    try:
        with io.open(metadata_path, "r", encoding="utf-8",
                     errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    break  # headers end at the first blank line
                if line.startswith("Name:") and name is None:
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("Version:") and version is None:
                    version = line.split(":", 1)[1].strip()
                if name and version:
                    break
    except (IOError, OSError):
        return None, None
    return name, version


def installed_distributions():
    """Inventory every installed distribution, from site-packages metadata.

    Uses only the standard library: importlib.metadata is 3.8+, and
    pkg_resources would tie this script to setuptools.
    """
    found = {}
    for entry in sys.path:
        if not entry or not os.path.isdir(entry):
            continue
        try:
            children = os.listdir(entry)
        except OSError:
            continue
        for child in children:
            if child.endswith(".dist-info"):
                metadata_path = os.path.join(entry, child, "METADATA")
            elif child.endswith(".egg-info"):
                candidate = os.path.join(entry, child)
                metadata_path = (
                    os.path.join(candidate, "PKG-INFO")
                    if os.path.isdir(candidate) else candidate
                )
            else:
                continue
            name, version = parse_metadata_name_version(metadata_path)
            if name and version and name not in found:
                found[name] = version

    inventory = OrderedDict()
    for name in sorted(found, key=lambda s: s.lower()):
        inventory[name] = found[name]
    return inventory


def run_preflight():
    """Execute the preflight check and capture its exit code."""
    result = OrderedDict()
    if not os.path.isfile(PREFLIGHT):
        result["executed"] = False
        result["reason"] = "scripts/preflight.py not found"
        return result
    try:
        # subprocess.run(..., stdout=PIPE) is 3.6-compatible;
        # capture_output= is 3.7+ and is deliberately not used.
        completed = subprocess.run(
            [sys.executable, PREFLIGHT],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
    except Exception as exc:                       # pragma: no cover
        result["executed"] = False
        result["reason"] = "%s: %s" % (type(exc).__name__, exc)
        return result

    code = completed.returncode
    result["executed"] = True
    result["exit_code"] = code
    result["overall"] = PREFLIGHT_MEANING.get(code, "UNKNOWN")
    return result


def digest_or_none(path):
    return sha256_of_file(path) if os.path.isfile(path) else None


def main():
    snapshot = OrderedDict()

    snapshot["project"] = "AirSense"
    snapshot["phase"] = "V1 / Phase 0A - runtime reconstruction"
    snapshot["snapshot_generated_utc"] = datetime.utcnow().strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    snapshot["snapshot_generated_note"] = (
        "Timestamp of this reconstruction run, not a historical date.")
    snapshot["historical_cutoff"] = HISTORICAL_CUTOFF

    reference = OrderedDict()
    reference["operating_system"] = REFERENCE_OS
    reference["python"] = REFERENCE_PYTHON
    snapshot["reference_environment"] = reference

    interpreter = OrderedDict()
    interpreter["implementation"] = platform.python_implementation()
    interpreter["version"] = platform.python_version()
    interpreter["version_info"] = list(sys.version_info[:3])
    interpreter["executable"] = sys.executable
    interpreter["prefix"] = sys.prefix
    interpreter["environment_name"] = os.path.basename(sys.prefix)
    interpreter["matches_reference"] = (
        platform.python_version() == REFERENCE_PYTHON
        and platform.python_implementation() == "CPython")
    snapshot["interpreter"] = interpreter

    host = OrderedDict()
    host["platform"] = platform.platform()
    host["system"] = platform.system()
    host["release"] = platform.release()
    host["machine"] = platform.machine()
    host["processor"] = platform.processor()
    snapshot["host"] = host

    direct = OrderedDict()
    for dist_name, import_name in DIRECT_MODULES:
        direct[dist_name] = module_version(import_name)
    snapshot["direct_dependencies"] = direct

    bootstrap = OrderedDict()
    for dist_name, import_name in BOOTSTRAP_MODULES:
        bootstrap[dist_name] = module_version(import_name)
    snapshot["bootstrap_tooling"] = bootstrap
    snapshot["bootstrap_tooling_note"] = (
        "Packaging and environment-management tooling. Recorded separately "
        "from the V1 scientific runtime because such tooling is not part of "
        "the V1 modelling methodology.")

    inventory = installed_distributions()
    snapshot["installed_distributions"] = inventory
    snapshot["installed_distribution_count"] = len(inventory)

    digests = OrderedDict()
    digests["requirements-v1-2019.txt"] = digest_or_none(REQUIREMENTS)
    digests["requirements-v1-2019-lock.txt"] = digest_or_none(LOCKFILE)
    digests["data/raw/PRSA_data_2010.1.1-2014.12.31.csv"] = digest_or_none(
        RAW_CSV)
    snapshot["sha256"] = digests

    snapshot["preflight"] = run_preflight()

    with io.open(SNAPSHOT_PATH, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, indent=2))
        handle.write("\n")

    sys.stdout.write("Wrote %s\n" % SNAPSHOT_PATH)
    sys.stdout.write("  interpreter        : %s %s\n"
                     % (interpreter["implementation"], interpreter["version"]))
    sys.stdout.write("  matches reference  : %s\n"
                     % interpreter["matches_reference"])
    sys.stdout.write("  distributions      : %d\n" % len(inventory))
    sys.stdout.write("  preflight          : %s\n"
                     % snapshot["preflight"].get("overall", "not executed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
