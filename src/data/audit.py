"""Read-only auditing of the raw AirSense V2 dataset.

AirSense V2, Protocol Phase 0.

This module inspects the official UCI Beijing Multi-Site Air Quality CSVs and
reports what is there. It **repairs nothing**: no imputation, no cleaning, no
type coercion written back, no row dropped. Phase 0 audits; Phase 1 decides
policy.

Target discipline
-----------------
PM2.5 is the forecast target and the 2016-03-01..2017-02-28 partition is a
sealed final test. This module therefore computes **no distributional
statistic of PM2.5 anywhere** - no mean, median, quantile, threshold,
histogram or correlation, in any partition. For PM2.5 it reports only
integrity facts: how many values are present, missing, non-finite, negative or
zero. Covariates (which are inputs, not the sealed target) receive an ordinary
min/max range audit so that impossible sentinel values can be detected.

Standard library only, by design: the V2 environment is not frozen until the
Phase-0 environment review.
"""

import csv
import hashlib
import math
import re
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

MISSING_TOKEN = "NA"

EXPECTED_COLUMNS = [
    "No", "year", "month", "day", "hour",
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station",
]

TARGET = "PM2.5"
INDEX_COLUMNS = ["No"]
TIME_COLUMNS = ["year", "month", "day", "hour"]
POLLUTANT_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEOROLOGY_COLUMNS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
CATEGORICAL_COLUMNS = ["wd", "station"]
NUMERIC_COLUMNS = POLLUTANT_COLUMNS + METEOROLOGY_COLUMNS
NON_NEGATIVE_COLUMNS = POLLUTANT_COLUMNS + ["RAIN", "WSPM"]

CSV_NAME_RE = re.compile(
    r"^PRSA_Data_(?P<station>[A-Za-z]+)_20130301-20170228\.csv$")

EXPECTED_GRID_START = datetime(2013, 3, 1, 0, 0)
EXPECTED_GRID_END = datetime(2017, 2, 28, 23, 0)

CHUNK = 1 << 20


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def station_from_filename(path):
    match = CSV_NAME_RE.match(Path(path).name)
    return match.group("station") if match else None


def _classify(value):
    """Return ('missing'|'int'|'float'|'nonfinite'|'text', parsed_or_None)."""
    if value == MISSING_TOKEN or value == "":
        return "missing", None
    try:
        number = float(value)
    except ValueError:
        return "text", None
    if math.isnan(number) or math.isinf(number):
        return "nonfinite", None
    if re.match(r"^-?\d+$", value):
        return "int", number
    return "float", number


def expected_grid_hours():
    span = EXPECTED_GRID_END - EXPECTED_GRID_START
    return int(span.total_seconds() // 3600) + 1


def audit_station_file(path):
    """Audit one station CSV. Returns an ordered, JSON-serialisable dict."""
    path = Path(path)
    filename_station = station_from_filename(path)

    header = None
    row_count = 0
    station_values = Counter()
    wd_values = Counter()
    kinds = {column: Counter() for column in EXPECTED_COLUMNS}
    missing = Counter()
    missing_by_year = {}
    numeric_extremes = {}
    negative_counts = Counter()
    zero_counts = Counter()
    timestamps = []
    bad_timestamps = 0
    duplicate_rows = Counter()
    malformed_rows = 0
    no_values = []

    with open(str(path), newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for fields in reader:
            row_count += 1
            if len(fields) != len(header):
                malformed_rows += 1
                continue
            row = dict(zip(header, fields))
            duplicate_rows[tuple(fields[1:])] += 1

            station_values[row.get("station", "")] += 1
            wd_values[row.get("wd", "")] += 1

            try:
                stamp = datetime(int(row["year"]), int(row["month"]),
                                 int(row["day"]), int(row["hour"]))
                timestamps.append(stamp)
                year = stamp.year
            except (ValueError, KeyError):
                bad_timestamps += 1
                year = None

            try:
                no_values.append(int(row["No"]))
            except (ValueError, KeyError):
                pass

            for column in EXPECTED_COLUMNS:
                if column not in row:
                    continue
                value = row[column]
                kind, number = _classify(value)
                kinds[column][kind] += 1
                if kind in ("missing", "nonfinite"):
                    if kind == "missing":
                        missing[column] += 1
                        if year is not None:
                            missing_by_year.setdefault(year, Counter())
                            missing_by_year[year][column] += 1
                    continue
                if column in NUMERIC_COLUMNS and number is not None:
                    if column in NON_NEGATIVE_COLUMNS and number < 0:
                        negative_counts[column] += 1
                    if number == 0:
                        zero_counts[column] += 1
                    # Range audit for covariates only. PM2.5 is the sealed
                    # target: no value statistic is recorded for it.
                    if column != TARGET:
                        low, high = numeric_extremes.get(
                            column, (number, number))
                        numeric_extremes[column] = (min(low, number),
                                                    max(high, number))

    observed = sorted(set(timestamps))
    grid_expected = expected_grid_hours()
    expected_set = set()
    cursor = EXPECTED_GRID_START
    while cursor <= EXPECTED_GRID_END:
        expected_set.add(cursor)
        cursor += timedelta(hours=1)
    observed_set = set(timestamps)
    missing_stamps = sorted(expected_set - observed_set)
    outside_grid = sorted(observed_set - expected_set)
    duplicate_stamps = sorted(
        stamp for stamp, count in Counter(timestamps).items() if count > 1)

    per_year_missing = OrderedDict()
    for year in sorted(missing_by_year):
        per_year_missing[str(year)] = OrderedDict(
            (column, missing_by_year[year][column])
            for column in EXPECTED_COLUMNS if missing_by_year[year][column])

    return OrderedDict([
        ("file", path.name),
        ("sha256", sha256_of_file(path)),
        ("size_bytes", path.stat().st_size),
        ("filename_station", filename_station),
        ("station_column_values", sorted(station_values)),
        ("station_identifier_agrees_with_filename",
         sorted(station_values) == [filename_station]),
        ("header", header),
        ("header_matches_expected", header == EXPECTED_COLUMNS),
        ("row_count", row_count),
        ("malformed_rows", malformed_rows),
        ("no_column_is_dense_1_to_n",
         bool(no_values) and no_values == list(range(1, len(no_values) + 1))),
        ("timestamp_min", min(observed).isoformat() if observed else None),
        ("timestamp_max", max(observed).isoformat() if observed else None),
        ("unparseable_timestamps", bad_timestamps),
        ("expected_grid_hours", grid_expected),
        ("observed_distinct_timestamps", len(observed_set)),
        ("missing_timestamps", len(missing_stamps)),
        ("missing_timestamp_examples",
         [s.isoformat() for s in missing_stamps[:5]]),
        ("timestamps_outside_expected_grid", len(outside_grid)),
        ("duplicate_timestamps", len(duplicate_stamps)),
        ("duplicate_timestamp_examples",
         [s.isoformat() for s in duplicate_stamps[:5]]),
        ("fully_duplicated_rows_excluding_index",
         sum(count - 1 for count in duplicate_rows.values() if count > 1)),
        ("wind_direction_categories", sorted(v for v in wd_values
                                             if v != MISSING_TOKEN)),
        ("missing_by_variable", OrderedDict(
            (column, missing[column]) for column in EXPECTED_COLUMNS)),
        ("missing_by_year", per_year_missing),
        ("value_kinds", OrderedDict(
            (column, OrderedDict(sorted(kinds[column].items())))
            for column in EXPECTED_COLUMNS)),
        ("negative_values_in_non_negative_columns",
         OrderedDict(sorted(negative_counts.items()))),
        ("zero_valued_counts", OrderedDict(sorted(zero_counts.items()))),
        ("covariate_ranges", OrderedDict(
            (column, {"min": numeric_extremes[column][0],
                      "max": numeric_extremes[column][1]})
            for column in sorted(numeric_extremes))),
        ("target_value_statistics_computed", False),
    ])


def audit_dataset(paths):
    """Audit every station CSV and aggregate. Repairs nothing."""
    stations = [audit_station_file(path) for path in sorted(paths, key=str)]
    total_rows = sum(s["row_count"] for s in stations)
    headers = {tuple(s["header"]) for s in stations}
    missing_totals = Counter()
    for station in stations:
        for column, count in station["missing_by_variable"].items():
            missing_totals[column] += count

    denominator = float(total_rows) if total_rows else 1.0
    missing_by_variable = OrderedDict()
    for column in EXPECTED_COLUMNS:
        count = missing_totals[column]
        missing_by_variable[column] = OrderedDict([
            ("missing", count),
            ("present", total_rows - count),
            ("missing_rate", round(count / denominator, 8)),
        ])

    return OrderedDict([
        ("station_count", len(stations)),
        ("station_names", sorted(s["filename_station"] for s in stations)),
        ("total_rows", total_rows),
        ("rows_per_station", OrderedDict(
            (s["filename_station"], s["row_count"]) for s in
            sorted(stations, key=lambda x: x["filename_station"]))),
        ("all_headers_identical", len(headers) == 1),
        ("columns", EXPECTED_COLUMNS),
        ("schema_matches_expected",
         all(s["header_matches_expected"] for s in stations)),
        ("station_identifiers_agree_with_filenames",
         all(s["station_identifier_agrees_with_filename"] for s in stations)),
        ("expected_grid_hours_per_station", expected_grid_hours()),
        ("stations_with_missing_timestamps",
         [s["filename_station"] for s in stations if s["missing_timestamps"]]),
        ("stations_with_duplicate_timestamps",
         [s["filename_station"] for s in stations
          if s["duplicate_timestamps"]]),
        ("duplicate_station_timestamp_pairs",
         sum(s["duplicate_timestamps"] for s in stations)),
        ("fully_duplicated_rows",
         sum(s["fully_duplicated_rows_excluding_index"] for s in stations)),
        ("missing_by_variable", missing_by_variable),
        ("target", TARGET),
        ("target_value_statistics_computed", False),
        ("cleaning_applied", False),
        ("imputation_applied", False),
        ("stations", stations),
    ])
