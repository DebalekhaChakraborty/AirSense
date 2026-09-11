"""Canonical causal preprocessing for AirSense V2.

Protocol Phase 2.

Turns the raw hourly station CSVs into the one canonical data layer every
later model consumes. Implements exactly the Phase-1 contract:

* masks are computed **before** any fill, so an imputed value can never
  masquerade as observed;
* causal forward fill with a 6-hour ceiling, then the frozen station-specific
  **training** median;
* no backward fill, no interpolation, no centred statistic;
* forward-fill state is chronological and is **not** reset at partition
  boundaries - the first validation hour may legitimately inherit an
  observation from the last training hours, because that observation is in
  the past;
* every learned constant (median, IQR, std, vocabulary) comes from the
  training partition only.

**Target quarantine.** PM2.5 is the forecast target. This module materialises
PM2.5 *values* only for training timestamps; for validation and test it
materialises presence (mask, gap age) and nothing else. Numeric validation and
test target history reaches models later through the guarded
``src.data.target_history`` interface, never through a persisted array.
"""

import csv
import json
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

MISSING_TOKEN = "NA"
TARGET = "PM2.5"

FULL_START = datetime(2013, 3, 1, 0)
FULL_END = datetime(2017, 2, 28, 23)

PARTITION_BOUNDS = OrderedDict([
    ("train", (datetime(2013, 3, 1, 0), datetime(2015, 2, 28, 23))),
    ("validation", (datetime(2015, 3, 1, 0), datetime(2016, 2, 29, 23))),
    ("test", (datetime(2016, 3, 1, 0), datetime(2017, 2, 28, 23))),
])

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEOROLOGY = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NON_TARGET_NUMERIC = ["PM10", "SO2", "NO2", "CO", "O3"] + METEOROLOGY
ALL_NUMERIC = [TARGET] + NON_TARGET_NUMERIC
GAP_AGE_VARIABLES = POLLUTANTS

WD_VOCABULARY = ["E", "ENE", "ESE", "N", "NE", "NNE", "NNW", "NW",
                 "S", "SE", "SSE", "SSW", "SW", "W", "WNW", "WSW"]
WD_MISSING_CODE = len(WD_VOCABULARY)          # 16

CAUSAL_FFILL_MAX_HOURS = 6
GAP_AGE_CAP_HOURS = 168

STATIONS = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
            "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan",
            "Wanliu", "Wanshouxigong"]


def total_hours():
    return int((FULL_END - FULL_START).total_seconds() // 3600) + 1


def index_of(stamp):
    return int((stamp - FULL_START).total_seconds() // 3600)


def timestamp_at(index):
    return FULL_START + timedelta(hours=int(index))


def partition_of(stamp):
    for name, (start, end) in PARTITION_BOUNDS.items():
        if start <= stamp <= end:
            return name
    return None


def partition_index_bounds(partition):
    start, end = PARTITION_BOUNDS[partition]
    return index_of(start), index_of(end)


def station_code(station):
    return STATIONS.index(station)


def load_station_raw(path):
    """Read one station over the full timeline.

    PM2.5 values are returned for **training timestamps only**; elsewhere the
    value array holds NaN while the presence mask still records observation.
    Non-target variables are returned for the whole timeline: they are model
    inputs, not the sealed target.
    """
    length = total_hours()
    train_end_index = index_of(PARTITION_BOUNDS["train"][1])

    values = {name: np.full(length, np.nan, dtype=np.float64)
              for name in ALL_NUMERIC}
    observed = {name: np.zeros(length, dtype=bool)
                for name in ALL_NUMERIC + ["wd"]}
    wd_raw = np.full(length, -1, dtype=np.int16)

    with open(str(path), newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime(int(row["year"]), int(row["month"]),
                             int(row["day"]), int(row["hour"]))
            if stamp < FULL_START or stamp > FULL_END:
                continue
            position = index_of(stamp)
            for name in ALL_NUMERIC:
                raw = row[name]
                if raw == MISSING_TOKEN or raw == "":
                    continue
                observed[name][position] = True
                if name == TARGET and position > train_end_index:
                    # Target quarantine: presence recorded, value discarded.
                    continue
                values[name][position] = float(raw)
            raw_wd = row["wd"]
            if raw_wd not in (MISSING_TOKEN, ""):
                observed["wd"][position] = True
                wd_raw[position] = WD_VOCABULARY.index(raw_wd)

    return {"values": values, "observed": observed, "wd_raw": wd_raw,
            "length": length}


def causal_forward_fill(values, observed, max_gap_hours, fallback):
    """Carry the last observation forward, at most `max_gap_hours`.

    Returns (filled, source) where `source` records provenance per hour:
        0 = originally observed
        1 = carried forward within the ceiling
        2 = training-median fallback
    Nothing later than the current index is ever consulted.
    """
    length = values.size
    filled = np.empty(length, dtype=np.float64)
    source = np.full(length, 2, dtype=np.int8)
    last_value = np.nan
    last_index = None
    for position in range(length):
        if observed[position] and np.isfinite(values[position]):
            filled[position] = values[position]
            source[position] = 0
            last_value = values[position]
            last_index = position
            continue
        if (last_index is not None
                and (position - last_index) <= max_gap_hours):
            filled[position] = last_value
            source[position] = 1
        else:
            filled[position] = fallback
            source[position] = 2
    return filled, source


def causal_categorical_fill(codes, observed, max_gap_hours, missing_code):
    """Categorical equivalent: carry the category, else an explicit MISSING."""
    length = codes.size
    filled = np.full(length, missing_code, dtype=np.int8)
    source = np.full(length, 2, dtype=np.int8)
    last_code = None
    last_index = None
    for position in range(length):
        if observed[position] and codes[position] >= 0:
            filled[position] = codes[position]
            source[position] = 0
            last_code = int(codes[position])
            last_index = position
            continue
        if (last_index is not None
                and (position - last_index) <= max_gap_hours):
            filled[position] = last_code
            source[position] = 1
        else:
            filled[position] = missing_code
            source[position] = 2
    return filled, source


def gap_age_hours(observed, cap):
    """Hours since the most recent observation at or before each hour.

    Conventions, fixed here and documented in the windowing contract:
        observed now              -> 0
        never observed before now -> cap
        otherwise                 -> min(hours since last observation, cap)
    The value is never negative and never exceeds `cap`.
    """
    length = observed.size
    ages = np.full(length, cap, dtype=np.uint16)
    last_index = None
    for position in range(length):
        if observed[position]:
            ages[position] = 0
            last_index = position
        elif last_index is not None:
            ages[position] = min(position - last_index, cap)
    return ages


def robust_scale(values, statistics, variable):
    """(x - training median) / training IQR, with the RAIN zero-IQR fallback."""
    stat = statistics[variable]
    centre = stat["median"]
    spread = stat["iqr"]
    method = "median_iqr"
    if spread == 0:
        spread = stat["std"]
        method = "median_std_zero_iqr_fallback"
    if spread == 0:
        raise ValueError("zero scale for %s" % variable)
    return (values - centre) / spread, method


def calendar_channels():
    """Deterministic cyclic calendar over the full timeline.

    Station independent, so it is stored once rather than twelve times.
    Columns: hour sin/cos, day-of-year sin/cos, month sin/cos.
    """
    length = total_hours()
    stamps = [timestamp_at(i) for i in range(length)]
    hour = np.array([s.hour for s in stamps], dtype=np.float64)
    doy = np.array([s.timetuple().tm_yday for s in stamps], dtype=np.float64)
    month = np.array([s.month for s in stamps], dtype=np.float64)
    days_in_year = np.array([366.0 if (s.year % 4 == 0 and
                                       (s.year % 100 != 0 or s.year % 400 == 0))
                             else 365.0 for s in stamps])
    channels = np.column_stack([
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * doy / days_in_year),
        np.cos(2 * np.pi * doy / days_in_year),
        np.sin(2 * np.pi * (month - 1) / 12.0),
        np.cos(2 * np.pi * (month - 1) / 12.0),
    ]).astype(np.float32)
    raw = np.column_stack([hour, doy, month]).astype(np.int16)
    return channels, raw


CALENDAR_CHANNEL_NAMES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos",
                          "month_sin", "month_cos"]
CALENDAR_RAW_NAMES = ["hour", "day_of_year", "month"]


def load_training_statistics(config_path):
    """Frozen Phase-1 pooled training statistics."""
    with open(str(config_path), encoding="utf-8") as handle:
        config = json.load(handle)
    return config["scaling_policy"]["training_statistics"]


def station_training_medians(record, statistics_by_station):
    return statistics_by_station


def compute_station_training_medians(raw, station):
    """Station-specific training medians, training partition only."""
    start, end = partition_index_bounds("train")
    medians = OrderedDict()
    for name in ALL_NUMERIC:
        window = raw["values"][name][start:end + 1]
        mask = raw["observed"][name][start:end + 1] & np.isfinite(window)
        sample = np.sort(window[mask])
        if sample.size == 0:
            medians[name] = 0.0
            continue
        position = (sample.size - 1) * 0.5
        lower = int(np.floor(position))
        upper = min(lower + 1, sample.size - 1)
        fraction = position - lower
        medians[name] = float(sample[lower]
                              + fraction * (sample[upper] - sample[lower]))
    return medians
