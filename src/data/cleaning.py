"""Data cleaning and chronological split freeze for AirSense V1.

Protocol Phase 4. Runs after the Phase 3 exploratory analysis and before any
feature preparation or model fitting.

Two things happen here, in a mandatory order:

1. **Split membership is assigned from calendar time** on the complete
   43,824-row raw timeline.
2. **Only then** are rows whose ``pm2.5`` target is missing excluded, and
   the exclusion happens separately within each partition.

The ordering matters. Dropping missing targets first and then slicing by
row position would let the missing-target rate — which Phase 3 showed is
strongly year-dependent — silently shift the boundaries away from the
calendar dates that were frozen.

Scope boundaries, enforced by construction:

* The raw CSV is opened read-only and never written, renamed or re-saved.
* No target value is imputed, interpolated, filled or synthesised. Missing
  targets are *excluded from the supervised derived datasets only*; the raw
  43,824-row timeline remains authoritative, and every excluded row is
  recorded in a permanent manifest.
* Zero-valued and high-concentration PM2.5 observations are retained
  unchanged. Nothing is clipped, winsorized, capped or reclassified.
* No scaling, encoding, transformation, feature engineering, lag, rolling
  statistic, shuffling or model of any kind.
* Retained rows are written back **verbatim**: the raw file is read a
  second time as text so that each retained line reproduces its original
  representation exactly, rather than being re-formatted by a float
  round-trip.
* No execution timestamp is written into any artifact, so repeated runs
  produce byte-identical output.

Historical constraint: CPython 3.6.7 with pandas 0.23.4 and numpy 1.15.4.
Only the frozen stack and the standard library are used; no modelling
library is imported.
"""

import io
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.data.audit import sha256_of_file

# ----------------------------------------------------------------------
# Frozen configuration
# ----------------------------------------------------------------------

RAW_FILENAME = "PRSA_data_2010.1.1-2014.12.31.csv"
EXPECTED_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")

TARGET = "pm2.5"
IDENTIFIER = "No"

RAW_COLUMNS = [
    "No", "year", "month", "day", "hour", "pm2.5",
    "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir",
]

MISSING_TOKEN = "NA"

EXCLUSION_REASON = "missing_pm25_target"

# ----------------------------------------------------------------------
# THE CHRONOLOGICAL DESIGN - FROZEN
#
# These boundaries were fixed before any model was fitted and before any
# predictive metric existed. Every partition is a whole number of complete
# calendar years, so each covers complete seasonal cycles, and every
# training observation precedes every validation observation, which in turn
# precedes every test observation.
#
# This boundary MUST NOT move because another boundary yields better
# metrics. Doing so would tune the split on the evaluation data, which is
# precisely the leakage the protocol's temporal rule exists to prevent.
# ----------------------------------------------------------------------

PARTITIONS = [
    ("development_train", "2010-01-01 00:00:00", "2012-12-31 23:00:00"),
    ("validation",        "2013-01-01 00:00:00", "2013-12-31 23:00:00"),
    ("test",              "2014-01-01 00:00:00", "2014-12-31 23:00:00"),
]
PARTITION_ORDER = [name for name, _, _ in PARTITIONS]

# The 2014 partition is the locked final test set. From the completion of
# this phase until Protocol Phase 10 its target must not inform any
# development decision.
LOCKED_TEST_PARTITION = "test"

PROCESSED_FILENAMES = OrderedDict([
    ("supervised", "airsense_supervised.csv"),
    ("development_train", "airsense_train_2010_2012.csv"),
    ("validation", "airsense_validation_2013.csv"),
    ("test", "airsense_test_2014.csv"),
])

# Counts declared in advance from the Phase 3 findings. The pipeline stops
# rather than continuing silently if the data disagrees.
EXPECTED_RAW_ROWS = OrderedDict([
    ("development_train", 26304), ("validation", 8760), ("test", 8760)])
EXPECTED_MISSING_TARGETS = OrderedDict([
    ("development_train", 1886), ("validation", 82), ("test", 99)])
EXPECTED_SUPERVISED_ROWS = OrderedDict([
    ("development_train", 24418), ("validation", 8678), ("test", 8661)])
EXPECTED_TOTAL_RAW = 43824
EXPECTED_TOTAL_MISSING = 2067
EXPECTED_TOTAL_SUPERVISED = 41757

FLOAT_FORMAT = "%.6f"


class IntegrityError(RuntimeError):
    """Raised when the data does not match what the phase declared."""


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_raw_typed(path):
    """Load the raw CSV with numeric types, ``NA`` treated as missing."""
    frame = pd.read_csv(path, na_values=[MISSING_TOKEN], keep_default_na=True)
    if list(frame.columns) != RAW_COLUMNS:
        raise IntegrityError(
            "raw schema mismatch: expected %s, found %s"
            % (RAW_COLUMNS, list(frame.columns)))
    return frame


def load_raw_text(path):
    """Load the raw CSV as verbatim strings, with no NA interpretation.

    Retained rows are written out from this frame so that every value
    reproduces its original representation byte-for-byte. Reading typed and
    writing back would re-format integers as floats and silently alter the
    file even though the numbers were unchanged.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False,
                        na_filter=False)
    if list(frame.columns) != RAW_COLUMNS:
        raise IntegrityError("raw schema mismatch in text read")
    return frame


def build_timestamps(typed):
    """Construct the hourly timestamp index IN MEMORY. Never persisted."""
    return pd.to_datetime(typed[["year", "month", "day", "hour"]])


def verify_temporal_integrity(timestamps):
    """Confirm the hourly timeline is complete, unique and ordered."""
    problems = []
    if timestamps.isnull().any():
        problems.append("some timestamps could not be constructed")
    if int(timestamps.nunique()) != int(len(timestamps)):
        problems.append("duplicate timestamps present")
    if not bool(timestamps.is_monotonic_increasing):
        problems.append("timestamps are not monotonic in file order")
    expected = pd.date_range(timestamps.min(), timestamps.max(), freq="H")
    absent = expected.difference(pd.DatetimeIndex(timestamps.unique()))
    if len(absent):
        problems.append("%d hourly slots absent" % len(absent))
    if problems:
        raise IntegrityError("temporal integrity failed: %s"
                             % "; ".join(problems))
    return OrderedDict([
        ("first_timestamp", str(timestamps.min())),
        ("last_timestamp", str(timestamps.max())),
        ("expected_hourly_slots", int(len(expected))),
        ("actual_rows", int(len(timestamps))),
        ("unique_timestamps", int(timestamps.nunique())),
        ("absent_hourly_slots", 0),
        ("duplicate_timestamps", 0),
        ("monotonic_increasing", True),
    ])


# ----------------------------------------------------------------------
# Step 1 - assign partitions from CALENDAR TIME, on the full raw timeline
# ----------------------------------------------------------------------

def assign_partitions(timestamps):
    """Return a Series naming each row's partition, from calendar time only.

    Assignment happens on all 43,824 raw rows, before any target-based
    exclusion. Membership is a property of *when* an observation was taken,
    never of whether its label happens to be present.
    """
    membership = pd.Series(index=timestamps.index, dtype=object)
    for name, start, end in PARTITIONS:
        selector = ((timestamps >= pd.Timestamp(start)) &
                    (timestamps <= pd.Timestamp(end)))
        membership[selector] = name

    unassigned = int(membership.isnull().sum())
    if unassigned:
        raise IntegrityError(
            "%d rows fall outside every declared partition" % unassigned)
    return membership


# ----------------------------------------------------------------------
# Step 2 - target availability within each partition
# ----------------------------------------------------------------------

def target_missing_mask(typed, text):
    """Boolean mask of rows whose target is missing, cross-checked.

    The mask is derived independently from the typed read and the text
    read; disagreement would mean the two views of the file differ, which
    must never pass silently.
    """
    from_typed = typed[TARGET].isnull()
    from_text = text[TARGET].str.strip() == MISSING_TOKEN
    if not bool((from_typed == from_text).all()):
        raise IntegrityError(
            "typed and text reads disagree about which targets are missing")
    return from_typed


def partition_statistics(membership, missing_mask, timestamps):
    """Per-partition raw / missing / supervised counts and boundaries."""
    rows = []
    for name in PARTITION_ORDER:
        selector = membership == name
        raw_rows = int(selector.sum())
        missing_rows = int((selector & missing_mask).sum())
        supervised_rows = raw_rows - missing_rows
        part_ts = timestamps[selector]
        rows.append(OrderedDict([
            ("partition", name),
            ("raw_rows", raw_rows),
            ("missing_target_rows", missing_rows),
            ("supervised_rows", supervised_rows),
            ("missing_target_rate_percent",
             100.0 * missing_rows / raw_rows if raw_rows else 0.0),
            ("start_timestamp", str(part_ts.min())),
            ("end_timestamp", str(part_ts.max())),
        ]))

    raw_total = int(sum(r["raw_rows"] for r in rows))
    missing_total = int(sum(r["missing_target_rows"] for r in rows))
    rows.append(OrderedDict([
        ("partition", "overall"),
        ("raw_rows", raw_total),
        ("missing_target_rows", missing_total),
        ("supervised_rows", raw_total - missing_total),
        ("missing_target_rate_percent",
         100.0 * missing_total / raw_total if raw_total else 0.0),
        ("start_timestamp", str(timestamps.min())),
        ("end_timestamp", str(timestamps.max())),
    ]))
    return pd.DataFrame(rows)


def verify_expected_counts(summary):
    """Stop rather than continue silently if counts differ from declared."""
    indexed = summary.set_index("partition")
    problems = []
    for name in PARTITION_ORDER:
        for column, expected_map in (
                ("raw_rows", EXPECTED_RAW_ROWS),
                ("missing_target_rows", EXPECTED_MISSING_TARGETS),
                ("supervised_rows", EXPECTED_SUPERVISED_ROWS)):
            observed = int(indexed.loc[name, column])
            expected = expected_map[name]
            if observed != expected:
                problems.append("%s.%s expected %d, observed %d"
                                % (name, column, expected, observed))
    totals = [("raw_rows", EXPECTED_TOTAL_RAW),
              ("missing_target_rows", EXPECTED_TOTAL_MISSING),
              ("supervised_rows", EXPECTED_TOTAL_SUPERVISED)]
    for column, expected in totals:
        observed = int(indexed.loc["overall", column])
        if observed != expected:
            problems.append("overall.%s expected %d, observed %d"
                            % (column, expected, observed))
    if problems:
        raise IntegrityError(
            "declared counts do not match the data; investigate before "
            "continuing: " + "; ".join(problems))


def missingness_shift(summary):
    """Ratios describing the label-missingness asymmetry across partitions.

    This is a TARGET-AVAILABILITY / LABEL-MISSINGNESS shift, not predictor
    covariate shift: every predictor is complete in every partition. No
    causal explanation for the pattern is asserted.
    """
    indexed = summary.set_index("partition")
    train_rate = float(indexed.loc["development_train",
                                   "missing_target_rate_percent"])
    val_rate = float(indexed.loc["validation",
                                 "missing_target_rate_percent"])
    test_rate = float(indexed.loc["test", "missing_target_rate_percent"])
    return OrderedDict([
        ("development_train_missing_rate_percent", train_rate),
        ("validation_missing_rate_percent", val_rate),
        ("test_missing_rate_percent", test_rate),
        ("train_over_validation_ratio",
         train_rate / val_rate if val_rate else float("nan")),
        ("train_over_test_ratio",
         train_rate / test_rate if test_rate else float("nan")),
        ("shift_type", "target_availability_label_missingness"),
        ("note", "Predictors are complete in every partition; only target "
                 "availability differs. No causal explanation is asserted."),
    ])


# ----------------------------------------------------------------------
# Step 3 - supervised outputs and manifests
# ----------------------------------------------------------------------

def write_supervised_outputs(text, membership, missing_mask, processed_dir):
    """Write the supervised CSVs verbatim from the text view of the raw file.

    Retained rows keep their original order, their original 13 columns and
    their original textual representation. No timestamp column is added:
    per the phase specification, timestamp information belongs in the
    manifests, so the modelable CSVs keep the raw schema exactly.
    """
    keep = ~missing_mask
    written = OrderedDict()

    combined = text.loc[keep]
    path = os.path.join(processed_dir, PROCESSED_FILENAMES["supervised"])
    combined.to_csv(path, index=False)
    written["supervised"] = PROCESSED_FILENAMES["supervised"]

    for name in PARTITION_ORDER:
        selector = keep & (membership == name)
        part = text.loc[selector]
        path = os.path.join(processed_dir, PROCESSED_FILENAMES[name])
        part.to_csv(path, index=False)
        written[name] = PROCESSED_FILENAMES[name]

    return written


def build_exclusion_manifest(typed, membership, missing_mask, timestamps):
    """Permanent audit trail of every row excluded from supervised data.

    One row per excluded observation. No target value is invented: the
    excluded rows have no observed target, and none is written here.
    """
    excluded = typed.loc[missing_mask, [IDENTIFIER, "year", "month", "day",
                                        "hour"]].copy()
    excluded["timestamp"] = timestamps[missing_mask].astype(str).values
    excluded["partition"] = membership[missing_mask].values
    excluded["exclusion_reason"] = EXCLUSION_REASON
    return excluded.reset_index(drop=True)


def build_reconciliation(summary):
    """Auditable arithmetic tying the raw total to the derived datasets."""
    indexed = summary.set_index("partition")
    rows = [
        ("raw_total_rows", int(indexed.loc["overall", "raw_rows"])),
        ("excluded_target_missing",
         int(indexed.loc["overall", "missing_target_rows"])),
        ("supervised_total_rows",
         int(indexed.loc["overall", "supervised_rows"])),
        ("supervised_development_train",
         int(indexed.loc["development_train", "supervised_rows"])),
        ("supervised_validation",
         int(indexed.loc["validation", "supervised_rows"])),
        ("supervised_test", int(indexed.loc["test", "supervised_rows"])),
    ]
    table = pd.DataFrame(OrderedDict([
        ("quantity", [name for name, _ in rows]),
        ("rows", [value for _, value in rows]),
    ]))

    raw_total = rows[0][1]
    excluded = rows[1][1]
    supervised = rows[2][1]
    parts_sum = rows[3][1] + rows[4][1] + rows[5][1]
    checks = pd.DataFrame(OrderedDict([
        ("quantity", [
            "check_supervised_plus_excluded_equals_raw",
            "check_partition_sum_equals_supervised",
        ]),
        ("rows", [
            int(supervised + excluded == raw_total),
            int(parts_sum == supervised),
        ]),
    ]))
    return pd.concat([table, checks], ignore_index=True)


# ----------------------------------------------------------------------
# Step 4 - round-trip verification against the raw source
# ----------------------------------------------------------------------

def verify_round_trip(text, typed, membership, missing_mask, timestamps,
                      processed_dir):
    """Verify every processed row against the raw source, value by value.

    Checks performed:

    * every retained row reproduces its raw ``No``, timestamp components,
      predictor values and observed target exactly;
    * every excluded row genuinely has a missing raw target;
    * supervised + excluded == 43,824;
    * the three partitions sum to the combined supervised file;
    * the partitions do not overlap, and every supervised row belongs to
      exactly one of them.
    """
    report = OrderedDict()
    keep = ~missing_mask

    def check_file(filename, expected_selector):
        path = os.path.join(processed_dir, filename)
        loaded = pd.read_csv(path, dtype=str, keep_default_na=False,
                             na_filter=False)
        if list(loaded.columns) != RAW_COLUMNS:
            raise IntegrityError("%s: schema changed" % filename)
        expected = text.loc[expected_selector].reset_index(drop=True)
        if len(loaded) != len(expected):
            raise IntegrityError(
                "%s: row count %d, expected %d"
                % (filename, len(loaded), len(expected)))
        if not loaded.equals(expected):
            raise IntegrityError(
                "%s: retained values differ from the raw source" % filename)
        # order preservation: identifiers must be strictly increasing
        ids = loaded[IDENTIFIER].astype(int)
        if not bool((ids.diff().dropna() > 0).all()):
            raise IntegrityError(
                "%s: chronological ordering not preserved" % filename)
        # no retained row may carry a missing target
        if bool((loaded[TARGET].str.strip() == MISSING_TOKEN).any()):
            raise IntegrityError(
                "%s: a retained row carries a missing target" % filename)
        return len(loaded), ids

    n_supervised, ids_supervised = check_file(
        PROCESSED_FILENAMES["supervised"], keep)
    report["supervised_rows_verified"] = int(n_supervised)

    partition_ids = OrderedDict()
    for name in PARTITION_ORDER:
        selector = keep & (membership == name)
        count, ids = check_file(PROCESSED_FILENAMES[name], selector)
        report["%s_rows_verified" % name] = int(count)
        partition_ids[name] = set(ids.tolist())

    # Every excluded row really is missing in the raw source. Checked
    # vectorially against the text view; the two views were already proven
    # to agree about the identifiers in target_missing_mask().
    excluded_targets = text.loc[missing_mask, TARGET].str.strip()
    present = excluded_targets != MISSING_TOKEN
    if bool(present.any()):
        offending = text.loc[missing_mask].loc[present, IDENTIFIER].tolist()
        raise IntegrityError(
            "%d rows were excluded but their raw target is present "
            "(e.g. No=%s)" % (int(present.sum()), offending[:5]))
    excluded_ids = typed.loc[missing_mask, IDENTIFIER].tolist()
    report["excluded_rows_verified"] = int(len(excluded_ids))

    # arithmetic
    total = n_supervised + len(excluded_ids)
    if total != EXPECTED_TOTAL_RAW:
        raise IntegrityError(
            "supervised (%d) + excluded (%d) = %d, expected %d"
            % (n_supervised, len(excluded_ids), total, EXPECTED_TOTAL_RAW))
    report["supervised_plus_excluded"] = int(total)

    parts_sum = sum(len(v) for v in partition_ids.values())
    if parts_sum != n_supervised:
        raise IntegrityError(
            "partition rows sum to %d, combined supervised file has %d"
            % (parts_sum, n_supervised))
    report["partition_sum"] = int(parts_sum)

    # zero overlap, exact cover
    names = PARTITION_ORDER
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = partition_ids[names[i]] & partition_ids[names[j]]
            if overlap:
                raise IntegrityError(
                    "partitions %s and %s overlap on %d rows"
                    % (names[i], names[j], len(overlap)))
    union = set()
    for value in partition_ids.values():
        union |= value
    if union != set(ids_supervised.tolist()):
        raise IntegrityError(
            "partitions do not exactly cover the supervised dataset")
    report["partition_overlap"] = 0
    report["partitions_exactly_cover_supervised"] = True

    # zero and extreme targets retained
    retained_targets = pd.to_numeric(
        text.loc[keep, TARGET], errors="coerce")
    report["retained_target_min"] = float(retained_targets.min())
    report["retained_target_max"] = float(retained_targets.max())
    report["zero_valued_targets_retained"] = int(
        (retained_targets == 0).sum())

    return report


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def run_preparation(project_root, verbose=True):
    """Execute Phase 4 end to end and write every artifact."""
    raw_path = os.path.join(project_root, "data", "raw", RAW_FILENAME)
    processed_dir = os.path.join(project_root, "data", "processed")
    artifacts_dir = os.path.join(project_root, "artifacts")
    results_dir = os.path.join(project_root, "results", "preprocessing")
    for directory in (processed_dir, artifacts_dir, results_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    def log(message):
        if verbose:
            print(message)

    # -- input gate ------------------------------------------------------
    sha_before = sha256_of_file(raw_path)
    if sha_before != EXPECTED_SHA256:
        raise IntegrityError(
            "raw dataset SHA-256 mismatch: expected %s, found %s. Refusing "
            "to proceed." % (EXPECTED_SHA256, sha_before))
    log("Raw dataset verified: %s" % sha_before)

    typed = load_raw_typed(raw_path)
    text = load_raw_text(raw_path)
    if len(typed) != len(text) or len(typed) != EXPECTED_TOTAL_RAW:
        raise IntegrityError("raw row count mismatch between reads")

    timestamps = build_timestamps(typed)
    temporal = verify_temporal_integrity(timestamps)
    log("Temporal integrity verified: %s to %s, %d hourly rows, no gaps"
        % (temporal["first_timestamp"], temporal["last_timestamp"],
           temporal["actual_rows"]))

    # -- split BEFORE exclusion -----------------------------------------
    membership = assign_partitions(timestamps)
    log("Partition membership assigned from calendar time "
        "(before any target exclusion)")

    missing_mask = target_missing_mask(typed, text)
    summary = partition_statistics(membership, missing_mask, timestamps)
    verify_expected_counts(summary)
    log("Partition counts match the declared expectations")

    shift = missingness_shift(summary)

    # -- outputs ---------------------------------------------------------
    written = write_supervised_outputs(text, membership, missing_mask,
                                       processed_dir)
    log("Wrote %d supervised CSV files to data/processed/" % len(written))

    exclusions = build_exclusion_manifest(typed, membership, missing_mask,
                                          timestamps)
    exclusions_path = os.path.join(artifacts_dir,
                                   "target_missing_exclusions.csv")
    exclusions.to_csv(exclusions_path, index=False)
    log("Wrote exclusion manifest: %d rows" % len(exclusions))

    summary_path = os.path.join(results_dir, "partition_summary.csv")
    summary.to_csv(summary_path, index=False, float_format=FLOAT_FORMAT)
    reconciliation = build_reconciliation(summary)
    reconciliation_path = os.path.join(results_dir,
                                       "data_reconciliation.csv")
    reconciliation.to_csv(reconciliation_path, index=False)
    log("Wrote preprocessing audit tables")

    # -- verification ----------------------------------------------------
    round_trip = verify_round_trip(text, typed, membership, missing_mask,
                                   timestamps, processed_dir)
    log("Round-trip verification passed")

    # -- split manifest --------------------------------------------------
    manifest = OrderedDict()
    manifest["project"] = "AirSense"
    manifest["phase"] = "V1 / Protocol Phase 4 - cleaning and split freeze"
    manifest["historical_cutoff"] = "2019-04-26"
    manifest["note"] = (
        "Split boundaries were frozen before any model was fitted and "
        "before any predictive metric existed. No execution timestamp is "
        "recorded here so that repeated runs are byte-identical.")

    source = OrderedDict()
    source["raw_dataset_path"] = os.path.join("data", "raw", RAW_FILENAME)
    source["raw_dataset_sha256"] = sha_before
    source["raw_rows"] = int(len(typed))
    source["raw_columns"] = int(len(typed.columns))
    source["temporal_integrity"] = temporal
    manifest["source"] = source

    policy = OrderedDict()
    policy["split_policy"] = "chronological_calendar_year_boundaries"
    policy["membership_rule"] = (
        "assigned from calendar timestamp on the full raw timeline, before "
        "any target-based exclusion")
    policy["supervised_eligibility_rule"] = (
        "a row is eligible for supervised modelling if and only if pm2.5 "
        "is non-missing")
    policy["target_imputation"] = "rejected - no target value is imputed, "\
        "interpolated, filled or synthesised"
    policy["zero_and_extreme_targets"] = "retained unchanged"
    policy["locked_final_test_partition"] = LOCKED_TEST_PARTITION
    policy["test_quarantine_until"] = "Protocol Phase 10"
    manifest["policy"] = policy

    indexed = summary.set_index("partition")
    partitions = OrderedDict()
    for name, start, end in PARTITIONS:
        selector = membership == name
        supervised_selector = selector & (~missing_mask)
        ids = typed.loc[supervised_selector, IDENTIFIER]
        raw_ids = typed.loc[selector, IDENTIFIER]
        part_ts = timestamps[selector]
        sup_ts = timestamps[supervised_selector]
        entry = OrderedDict()
        entry["declared_start"] = start
        entry["declared_end"] = end
        entry["raw_rows"] = int(indexed.loc[name, "raw_rows"])
        entry["missing_target_rows"] = int(
            indexed.loc[name, "missing_target_rows"])
        entry["supervised_rows"] = int(indexed.loc[name, "supervised_rows"])
        entry["missing_target_rate_percent"] = float(
            indexed.loc[name, "missing_target_rate_percent"])
        entry["raw_first_No"] = int(raw_ids.min())
        entry["raw_last_No"] = int(raw_ids.max())
        entry["supervised_first_No"] = int(ids.min())
        entry["supervised_last_No"] = int(ids.max())
        entry["raw_first_timestamp"] = str(part_ts.min())
        entry["raw_last_timestamp"] = str(part_ts.max())
        entry["supervised_first_timestamp"] = str(sup_ts.min())
        entry["supervised_last_timestamp"] = str(sup_ts.max())
        entry["processed_file"] = PROCESSED_FILENAMES[name]
        partitions[name] = entry
    manifest["partitions"] = partitions

    manifest["label_missingness_shift"] = shift

    files = OrderedDict()
    for key, filename in PROCESSED_FILENAMES.items():
        path = os.path.join(processed_dir, filename)
        files[os.path.join("data", "processed", filename)] = OrderedDict([
            ("sha256", sha256_of_file(path)),
            ("rows", int(sum(1 for _ in io.open(path, encoding="utf-8")) - 1)),
        ])
    files[os.path.join("artifacts", "target_missing_exclusions.csv")] = \
        OrderedDict([
            ("sha256", sha256_of_file(exclusions_path)),
            ("rows", int(len(exclusions))),
        ])
    files[os.path.join("results", "preprocessing",
                       "partition_summary.csv")] = OrderedDict([
        ("sha256", sha256_of_file(summary_path)),
        ("rows", int(len(summary))),
    ])
    files[os.path.join("results", "preprocessing",
                       "data_reconciliation.csv")] = OrderedDict([
        ("sha256", sha256_of_file(reconciliation_path)),
        ("rows", int(len(reconciliation))),
    ])
    manifest["files"] = files

    reconcile = OrderedDict()
    reconcile["raw_total_rows"] = EXPECTED_TOTAL_RAW
    reconcile["excluded_target_missing"] = int(missing_mask.sum())
    reconcile["supervised_total_rows"] = int((~missing_mask).sum())
    reconcile["partition_supervised_sum"] = int(sum(
        int(indexed.loc[n, "supervised_rows"]) for n in PARTITION_ORDER))
    reconcile["supervised_plus_excluded_equals_raw"] = bool(
        int((~missing_mask).sum()) + int(missing_mask.sum())
        == EXPECTED_TOTAL_RAW)
    reconcile["partitions_exactly_cover_supervised"] = bool(
        round_trip["partitions_exactly_cover_supervised"])
    reconcile["partition_overlap_rows"] = int(round_trip["partition_overlap"])
    manifest["membership_reconciliation"] = reconcile
    manifest["round_trip_verification"] = round_trip

    sha_after = sha256_of_file(raw_path)
    manifest["raw_dataset_sha256_before"] = sha_before
    manifest["raw_dataset_sha256_after"] = sha_after
    manifest["raw_dataset_unchanged"] = bool(sha_before == sha_after)

    manifest_path = os.path.join(artifacts_dir, "split_manifest.json")
    with io.open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2))
        handle.write("\n")
    log("Wrote artifacts/split_manifest.json")

    if sha_before != sha_after:
        raise IntegrityError(
            "CRITICAL DATA INTEGRITY FAILURE: raw dataset digest changed "
            "during preparation (%s -> %s)" % (sha_before, sha_after))

    log("")
    log("Raw dataset unchanged: %s" % (sha_before == sha_after))
    return manifest
