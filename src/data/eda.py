"""Training-only exploratory analysis for AirSense V2.

Protocol Phase 1.

**Partition discipline is enforced structurally, not by convention.**
``load_partition`` has two modes:

``values``     numeric values are returned. Legal for the TRAINING partition.
``presence``   only an observed/missing boolean array is returned; the numeric
               values are discarded inside the parser and never leave it.
               This is the only mode permitted for the validation and locked
               test partitions, so no code path in Phase 1 can compute a
               validation or test target statistic even by accident.

Deterministic by construction: quantiles use linear interpolation between
order statistics (the "type 7"/`numpy` linear method), moments use the
Fisher-Pearson estimators, and no output depends on wall-clock time, iteration
order or random sampling.
"""

import csv
import math
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

MISSING_TOKEN = "NA"
TARGET = "PM2.5"

NUMERIC_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                   "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
CATEGORICAL_COLUMNS = ["wd"]

PARTITIONS = OrderedDict([
    ("train", (datetime(2013, 3, 1, 0), datetime(2015, 2, 28, 23))),
    ("validation", (datetime(2015, 3, 1, 0), datetime(2016, 2, 29, 23))),
    ("test", (datetime(2016, 3, 1, 0), datetime(2017, 2, 28, 23))),
])

QUANTILE_METHOD = "linear interpolation between order statistics (type 7)"
PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]


def station_paths(raw_dir):
    return sorted(Path(raw_dir).glob("PRSA_Data_*.csv"))


def station_name(path):
    return Path(path).name.split("_")[2]


def hours_between(start, end):
    return int((end - start).total_seconds() // 3600) + 1


def load_partition(path, partition, mode):
    """Load one station CSV restricted to one partition.

    mode="values"   -> numeric arrays (TRAINING ONLY)
    mode="presence" -> observed/missing booleans only; values are discarded
    """
    if mode not in ("values", "presence"):
        raise ValueError("mode must be 'values' or 'presence'")
    if mode == "values" and partition != "train":
        raise ValueError(
            "refusing to load values for partition %r: only the training "
            "partition may expose values in Phase 1" % partition)

    start, end = PARTITIONS[partition]
    length = hours_between(start, end)

    numeric = {column: np.full(length, np.nan) for column in NUMERIC_COLUMNS}
    observed = {column: np.zeros(length, dtype=bool)
                for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS}
    categorical = {column: [None] * length for column in CATEGORICAL_COLUMNS}

    with open(str(path), newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stamp = datetime(int(row["year"]), int(row["month"]),
                             int(row["day"]), int(row["hour"]))
            if stamp < start or stamp > end:
                continue
            index = int((stamp - start).total_seconds() // 3600)
            for column in NUMERIC_COLUMNS:
                raw = row[column]
                if raw == MISSING_TOKEN or raw == "":
                    continue
                observed[column][index] = True
                if mode == "values":
                    numeric[column][index] = float(raw)
            for column in CATEGORICAL_COLUMNS:
                raw = row[column]
                if raw == MISSING_TOKEN or raw == "":
                    continue
                observed[column][index] = True
                if mode == "values":
                    categorical[column][index] = raw

    result = {
        "station": station_name(path),
        "partition": partition,
        "start": start,
        "end": end,
        "length": length,
        "observed": observed,
    }
    if mode == "values":
        result["numeric"] = numeric
        result["categorical"] = categorical
    return result


def timestamps(partition):
    start, end = PARTITIONS[partition]
    return [start + timedelta(hours=i)
            for i in range(hours_between(start, end))]


def quantile(values, percentile):
    """Deterministic type-7 quantile on a finite, sorted-able array."""
    return float(np.quantile(np.asarray(values, dtype=float),
                             percentile / 100.0, method="linear"))


def describe(values):
    """Summary statistics for one finite numeric sample."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return OrderedDict([("count", 0)])
    mean = float(array.mean())
    centred = array - mean
    m2 = float((centred ** 2).mean())
    m3 = float((centred ** 3).mean())
    m4 = float((centred ** 4).mean())
    skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    kurtosis = (m4 / (m2 ** 2) - 3.0) if m2 > 0 else 0.0
    summary = OrderedDict([
        ("count", count),
        ("mean", mean),
        ("std", float(array.std(ddof=1)) if count > 1 else 0.0),
        ("min", float(array.min())),
        ("max", float(array.max())),
    ])
    for percentile in PERCENTILES:
        summary["p%02d" % percentile] = quantile(array, percentile)
    summary["median"] = summary["p50"]
    summary["skewness"] = float(skew)
    summary["excess_kurtosis"] = float(kurtosis)
    return summary


def autocorrelation_at_lag(values, observed, lag):
    """Pearson correlation between t and t+lag on exact hourly positions.

    Operates on the complete hourly grid, so index arithmetic *is* timestamp
    arithmetic. Pairs where either endpoint is unobserved are excluded, never
    filled.
    """
    if lag <= 0 or lag >= values.size:
        return None, 0
    left = values[:-lag]
    right = values[lag:]
    valid = observed[:-lag] & observed[lag:]
    valid &= np.isfinite(left) & np.isfinite(right)
    n = int(valid.sum())
    if n < 3:
        return None, n
    a = left[valid]
    b = right[valid]
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denominator = math.sqrt(float((a_centred ** 2).sum())
                            * float((b_centred ** 2).sum()))
    if denominator == 0.0:
        return None, n
    return float((a_centred * b_centred).sum() / denominator), n


def pearson(a, b, valid):
    n = int(valid.sum())
    if n < 3:
        return None, n
    x = a[valid]
    y = b[valid]
    x_centred = x - x.mean()
    y_centred = y - y.mean()
    denominator = math.sqrt(float((x_centred ** 2).sum())
                            * float((y_centred ** 2).sum()))
    if denominator == 0.0:
        return None, n
    return float((x_centred * y_centred).sum() / denominator), n
