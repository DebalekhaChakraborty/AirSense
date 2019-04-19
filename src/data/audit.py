"""Raw-dataset provenance and integrity auditing for AirSense V1.

Deliberately implemented with the Python standard library only:

* it must run identically on the reference historical interpreter
  (CPython 3.6.7) and on the modern host interpreter, so that the audit
  itself never depends on the scientific stack that V1 pins;
* it must be usable before any third-party package is installed, which is
  what makes the preflight check meaningful;
* it performs read-only operations exclusively. Nothing in this module
  writes to, moves, or mutates a raw data file.

Language level: CPython 3.6-compatible (f-strings are permitted, since they
were introduced in 3.6; no dataclasses, no walrus operator, no PEP 585
builtin generics).
"""

import csv
import hashlib
import io
import os
from collections import OrderedDict

# Tokens treated as "missing" in the original UCI distribution of the
# Beijing PM2.5 data set. The raw file encodes absent PM2.5 readings as the
# literal string "NA".
DEFAULT_MISSING_TOKENS = ("NA", "", "NaN", "nan", "N/A", "null")

# Expected schema of the original UCI file
# PRSA_data_2010.1.1-2014.12.31.csv, in original column order.
EXPECTED_COLUMNS = [
    "No",      # row index as distributed by UCI
    "year",
    "month",
    "day",
    "hour",
    "pm2.5",   # target variable
    "DEWP",    # dew point
    "TEMP",    # temperature
    "PRES",    # pressure
    "cbwd",    # combined wind direction (categorical)
    "Iws",     # cumulated wind speed
    "Is",      # cumulated hours of snow
    "Ir",      # cumulated hours of rain
]

TARGET_COLUMN = "pm2.5"

# Column carrying the distributor's row index; it is not a predictor and is
# excluded when judging whether two observations are genuinely duplicated.
INDEX_COLUMN = "No"


def sha256_of_file(path, chunk_size=1024 * 1024):
    """Return the hex SHA-256 digest of a file, read in binary chunks."""
    digest = hashlib.sha256()
    with io.open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audit_csv(path, missing_tokens=DEFAULT_MISSING_TOKENS):
    """Audit a raw CSV file and return an OrderedDict describing it.

    The returned record is designed so that a later run can prove the raw
    file has not changed: it captures the byte size, the SHA-256 digest, the
    dimensions, the column names, the per-column missing counts and the
    duplicate-row counts.

    No cleaning, imputation or row removal is performed. Missing values are
    counted, never dropped.
    """
    missing_set = set(missing_tokens)

    record = OrderedDict()
    record["filename"] = os.path.basename(path)
    record["relative_path"] = path
    record["byte_size"] = os.path.getsize(path)
    record["sha256"] = sha256_of_file(path)

    with io.open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV file is empty: %s" % path)

        header = [name.strip() for name in header]
        n_columns = len(header)

        missing_counts = OrderedDict((name, 0) for name in header)
        ragged_rows = 0
        row_count = 0

        # Duplicate accounting is done two ways, because the UCI file ships
        # with a unique running index in column "No" which would otherwise
        # mask genuinely repeated observations.
        seen_full = set()
        seen_without_index = set()
        duplicate_full = 0
        duplicate_without_index = 0

        index_position = header.index(INDEX_COLUMN) if INDEX_COLUMN in header else None

        for row in reader:
            if not row:
                continue
            row_count += 1

            if len(row) != n_columns:
                ragged_rows += 1

            for position, name in enumerate(header):
                if position < len(row):
                    value = row[position].strip()
                else:
                    value = ""
                if value in missing_set:
                    missing_counts[name] += 1

            full_key = tuple(cell.strip() for cell in row)
            if full_key in seen_full:
                duplicate_full += 1
            else:
                seen_full.add(full_key)

            if index_position is None:
                trimmed_key = full_key
            else:
                trimmed_key = tuple(
                    cell for position, cell in enumerate(full_key)
                    if position != index_position
                )
            if trimmed_key in seen_without_index:
                duplicate_without_index += 1
            else:
                seen_without_index.add(trimmed_key)

    record["row_count"] = row_count
    record["column_count"] = n_columns
    record["column_names"] = header
    record["ragged_row_count"] = ragged_rows
    record["missing_values_per_column"] = missing_counts
    record["missing_value_tokens"] = list(missing_tokens)
    record["duplicate_rows_including_index_column"] = duplicate_full
    record["duplicate_rows_excluding_index_column"] = duplicate_without_index
    record["target_column"] = TARGET_COLUMN
    record["target_column_present"] = TARGET_COLUMN in header
    record["schema_matches_expected"] = header == EXPECTED_COLUMNS
    record["expected_columns"] = list(EXPECTED_COLUMNS)

    return record


def schema_differences(header):
    """Return (missing_expected, unexpected_present) for a header list."""
    missing_expected = [c for c in EXPECTED_COLUMNS if c not in header]
    unexpected_present = [c for c in header if c not in EXPECTED_COLUMNS]
    return missing_expected, unexpected_present
