"""Acquire the official UCI Beijing Multi-Site Air Quality data set (ID 501).

AirSense V2, Protocol Phase 0.

Downloads the archive from the **official UCI source only**, hashes it,
extracts it deterministically, hashes every extracted file, and records
provenance. It performs no cleaning, no imputation and no analysis.

Observed official structure (verified, not assumed):

    beijing+multi+site+air+quality+data.zip      outer archive
      PRSA2017_Data_20130301-20170228.zip        inner archive: the dataset
        PRSA_Data_20130301-20170228/             one directory
          PRSA_Data_<Station>_20130301-20170228.csv   x 12
      <uuid>.JPG                                 UCI page illustration
      data.csv, test.csv                         UNRELATED stock-price files

The two loose CSVs in the outer archive carry the schema
``Date,Open,High,Low,Close,Adj Close,Volume`` with 2018 dates: they are
unrelated financial series that are present in the official distribution but
are not part of this data set. They are recorded as auxiliary members for
provenance completeness and are never extracted into the dataset or used.

Guarantees:
  * an existing raw archive is never silently overwritten or re-downloaded;
  * the structure is validated before anything is extracted, and any
    unexpected or ambiguous layout is refused;
  * extracted raw files are made read-only;
  * the manifest is deterministic: entries are sorted and no wall-clock time
    or filesystem mtime is recorded.

Standard library only, by design: the V2 environment is not frozen until the
Phase-0 environment review, so Phase-0 tooling takes no third-party
dependency.

Usage:
    python3 scripts/acquire_dataset.py
    python3 scripts/acquire_dataset.py --verify-only
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_NAME = "Beijing Multi-Site Air-Quality Data"
UCI_ID = 501
DOI = "10.24432/C5RK5G"
LANDING_PAGE = (
    "https://archive.ics.uci.edu/dataset/501/"
    "beijing+multi+site+air+quality+data")
DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing+multi+site+air+quality+data.zip")
LICENSE = "CC BY 4.0 as presented by UCI"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "beijing+multi+site+air+quality+data.zip"
INNER_NAME = "PRSA2017_Data_20130301-20170228.zip"
INNER_PATH = RAW_DIR / INNER_NAME
STATION_DIR_NAME = "PRSA_Data_20130301-20170228"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "raw_dataset_manifest.json"

EXPECTED_CSV_COUNT = 12
CSV_NAME_RE = re.compile(
    r"^PRSA_Data_(?P<station>[A-Za-z]+)_20130301-20170228\.csv$")

CHUNK = 1 << 20


class AcquisitionError(RuntimeError):
    """Raised when the source is missing, ambiguous or unexpected."""


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive():
    """Download the official archive unless it is already present."""
    if ARCHIVE_PATH.exists():
        print("archive already present, not re-downloading: %s"
              % ARCHIVE_PATH.relative_to(PROJECT_ROOT))
        return False
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE_PATH.with_name(ARCHIVE_PATH.name + ".partial")
    if partial.exists():
        partial.unlink()
    print("downloading from official UCI source:\n  %s" % DOWNLOAD_URL)
    request = Request(DOWNLOAD_URL, headers={"User-Agent": "AirSense-V2/0"})
    try:
        with urlopen(request, timeout=180) as response:
            if response.status != 200:
                raise AcquisitionError(
                    "official source returned HTTP %s" % response.status)
            with open(str(partial), "wb") as handle:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    handle.write(block)
    except AcquisitionError:
        raise
    except Exception as error:                       # noqa: BLE001
        if partial.exists():
            partial.unlink()
        raise AcquisitionError(
            "could not obtain the official archive: %s. Do not substitute "
            "another source; report the blocker." % error)
    partial.rename(ARCHIVE_PATH)
    os.chmod(str(ARCHIVE_PATH), 0o444)
    return True


def safe_members(archive):
    names = [n for n in archive.namelist() if not n.endswith("/")]
    for name in names:
        if name.startswith("/") or ".." in Path(name).parts:
            raise AcquisitionError("unsafe archive member: %r" % name)
    return names


def inspect_outer():
    """Validate the outer archive and return (inner_bytes, auxiliary)."""
    if not zipfile.is_zipfile(str(ARCHIVE_PATH)):
        raise AcquisitionError("%s is not a zip archive" % ARCHIVE_PATH)
    with zipfile.ZipFile(str(ARCHIVE_PATH)) as archive:
        names = safe_members(archive)
        inner = [n for n in names if n.lower().endswith(".zip")]
        if len(inner) != 1:
            raise AcquisitionError(
                "expected exactly one inner archive, found %d: %s"
                % (len(inner), sorted(inner)))
        if inner[0] != INNER_NAME:
            raise AcquisitionError(
                "unexpected inner archive name %r (expected %r)"
                % (inner[0], INNER_NAME))
        auxiliary = []
        for name in sorted(set(names) - set(inner)):
            payload = archive.read(name)
            if CSV_NAME_RE.match(Path(name).name):
                raise AcquisitionError(
                    "station-like CSV found outside the inner archive: %r"
                    % name)
            auxiliary.append({
                "member": name,
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "part_of_dataset": False,
            })
        return archive.read(inner[0]), auxiliary


def inspect_inner(inner_bytes):
    """Validate the inner archive and return its station CSV members."""
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as archive:
        names = safe_members(archive)
        csvs = sorted(n for n in names if n.lower().endswith(".csv"))
        others = sorted(set(names) - set(csvs))
        if others:
            raise AcquisitionError(
                "inner archive holds unexpected members: %s" % others)
        if len(csvs) != EXPECTED_CSV_COUNT:
            raise AcquisitionError(
                "expected %d station CSVs, inner archive holds %d: %s"
                % (EXPECTED_CSV_COUNT, len(csvs), csvs))
        parents = {str(Path(n).parent) for n in csvs}
        if parents != {STATION_DIR_NAME}:
            raise AcquisitionError(
                "expected all CSVs under %r, found %s"
                % (STATION_DIR_NAME, sorted(parents)))
        stations = {}
        for name in csvs:
            match = CSV_NAME_RE.match(Path(name).name)
            if match is None:
                raise AcquisitionError(
                    "unexpected CSV filename, refusing ambiguous structure: %r"
                    % Path(name).name)
            stations[name] = match.group("station")
        if len(set(stations.values())) != EXPECTED_CSV_COUNT:
            raise AcquisitionError(
                "station names are not unique: %s" % sorted(stations.values()))
        return csvs, stations


def write_readonly(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print("raw file already present, not overwriting: %s"
              % path.relative_to(PROJECT_ROOT))
        return False
    with open(str(path), "wb") as handle:
        handle.write(payload)
    os.chmod(str(path), 0o444)
    return True


def extract(inner_bytes, csvs, verify_only):
    if not verify_only:
        write_readonly(INNER_PATH, inner_bytes)
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as archive:
            for name in csvs:
                write_readonly(RAW_DIR / name, archive.read(name))
    paths = [RAW_DIR / name for name in csvs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise AcquisitionError(
            "missing extracted files: %s"
            % [str(p.relative_to(PROJECT_ROOT)) for p in missing])
    return paths


def build_manifest(archive_sha, inner_bytes, auxiliary, csvs, stations, paths):
    entries = []
    for path in sorted(paths, key=str):
        member = str(path.relative_to(RAW_DIR))
        entries.append({
            "member": member,
            "relative_path": str(path.relative_to(PROJECT_ROOT)),
            "station": stations[member],
            "size_bytes": path.stat().st_size,
            "sha256": sha256_of_file(path),
        })
    aggregate = sha256_bytes(
        "".join("%s:%s\n" % (e["member"], e["sha256"]) for e in entries)
        .encode("utf-8"))
    return {
        "dataset": DATASET_NAME,
        "uci_id": UCI_ID,
        "doi": DOI,
        "landing_page": LANDING_PAGE,
        "resolved_download_url": DOWNLOAD_URL,
        "license": LICENSE,
        "outer_archive": {
            "filename": ARCHIVE_PATH.name,
            "relative_path": str(ARCHIVE_PATH.relative_to(PROJECT_ROOT)),
            "size_bytes": ARCHIVE_PATH.stat().st_size,
            "sha256": archive_sha,
        },
        "inner_archive": {
            "filename": INNER_NAME,
            "relative_path": str(INNER_PATH.relative_to(PROJECT_ROOT)),
            "size_bytes": len(inner_bytes),
            "sha256": sha256_bytes(inner_bytes),
        },
        "station_directory": STATION_DIR_NAME,
        "source_csv_count": len(entries),
        "source_csv_aggregate_sha256": aggregate,
        "source_csvs": entries,
        "auxiliary_members_not_part_of_dataset": auxiliary,
        "auxiliary_note": (
            "data.csv and test.csv in the outer archive carry the schema "
            "Date,Open,High,Low,Close,Adj Close,Volume with 2018 dates. They "
            "are unrelated financial series shipped inside the official UCI "
            "distribution. They are hashed here for provenance completeness, "
            "are not extracted into data/raw/, and are not used by AirSense "
            "V2. In particular the file named test.csv has nothing to do "
            "with the V2 locked final test."),
        "cleaning_applied": False,
        "imputation_applied": False,
        "raw_files_modified": False,
        "note": (
            "Raw material is immutable. Derived data belongs under "
            "data/processed/ and never overwrites data/raw/."),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true",
                        help="hash and validate what is already on disk")
    args = parser.parse_args()

    try:
        if not args.verify_only:
            download_archive()
        if not ARCHIVE_PATH.exists():
            raise AcquisitionError(
                "no archive at %s and --verify-only was given" % ARCHIVE_PATH)
        archive_sha = sha256_of_file(ARCHIVE_PATH)
        inner_bytes, auxiliary = inspect_outer()
        csvs, stations = inspect_inner(inner_bytes)
        paths = extract(inner_bytes, csvs, args.verify_only)
        by_member = {str(p.relative_to(RAW_DIR)): stations[str(
            p.relative_to(RAW_DIR))] for p in paths}
        manifest = build_manifest(archive_sha, inner_bytes, auxiliary,
                                  csvs, by_member, paths)
    except AcquisitionError as error:
        print("\nACQUISITION FAILED: %s" % error, file=sys.stderr)
        return 2

    write_json(MANIFEST_PATH, manifest)
    print("")
    print("outer archive sha256 : %s" % manifest["outer_archive"]["sha256"])
    print("outer archive bytes  : %d" % manifest["outer_archive"]["size_bytes"])
    print("inner archive sha256 : %s" % manifest["inner_archive"]["sha256"])
    print("source CSVs          : %d" % manifest["source_csv_count"])
    print("aggregate sha256     : %s"
          % manifest["source_csv_aggregate_sha256"])
    print("auxiliary members    : %d (recorded, not used)"
          % len(manifest["auxiliary_members_not_part_of_dataset"]))
    print("manifest             : %s"
          % MANIFEST_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
