"""Missingness structure analysis for AirSense V2.

Protocol Phase 1.

Gap *duration* decides what causal imputation is defensible, so this module
characterises runs of consecutive missing hours rather than only counting
missing values. It computes nothing from values and repairs nothing: it reads
observed/missing boolean masks, which makes it safe to run on any partition.
"""

from collections import OrderedDict

import numpy as np

RUN_BUCKETS = [("runs_le_3h", 3), ("runs_le_6h", 6), ("runs_le_12h", 12),
               ("runs_le_24h", 24)]


def missing_runs(observed):
    """Lengths of maximal consecutive missing runs on the hourly grid."""
    mask = ~np.asarray(observed, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return (ends - starts).tolist()


def percentile_of_runs(runs, percentile):
    if not runs:
        return 0
    return float(np.percentile(np.asarray(runs, dtype=float), percentile,
                               method="linear"))


def run_summary(observed):
    """Full run-length characterisation of one variable at one station."""
    observed = np.asarray(observed, dtype=bool)
    total = int(observed.size)
    missing = int((~observed).sum())
    runs = missing_runs(observed)
    summary = OrderedDict([
        ("hours", total),
        ("missing", missing),
        ("missing_pct", round(100.0 * missing / total, 6) if total else 0.0),
        ("n_runs", len(runs)),
        ("longest_run_h", max(runs) if runs else 0),
        ("median_run_h", percentile_of_runs(runs, 50)),
        ("p90_run_h", percentile_of_runs(runs, 90)),
        ("isolated_1h_gaps", sum(1 for r in runs if r == 1)),
    ])
    for label, limit in RUN_BUCKETS:
        summary[label] = sum(1 for r in runs if r <= limit)
    summary["runs_gt_24h"] = sum(1 for r in runs if r > 24)
    summary["missing_hours_in_runs_gt_24h"] = sum(r for r in runs if r > 24)
    return summary


def simultaneous_missing(observed_by_station):
    """Per-timestamp count of stations missing a variable.

    Returns (counts_array, histogram) where the histogram maps
    "k stations missing" -> number of timestamps.
    """
    stacked = np.vstack([~np.asarray(observed, dtype=bool)
                         for observed in observed_by_station])
    counts = stacked.sum(axis=0).astype(int)
    n_stations = stacked.shape[0]
    histogram = OrderedDict()
    for k in range(n_stations + 1):
        histogram[k] = int((counts == k).sum())
    return counts, histogram


def group_counts(observed, keys):
    """Missing/total counts grouped by an integer key array (year, hour...)."""
    observed = np.asarray(observed, dtype=bool)
    keys = np.asarray(keys)
    result = OrderedDict()
    for key in sorted(set(keys.tolist())):
        selector = keys == key
        total = int(selector.sum())
        missing = int((~observed[selector]).sum())
        result[key] = OrderedDict([
            ("hours", total),
            ("missing", missing),
            ("missing_pct", round(100.0 * missing / total, 6) if total else 0.0),
        ])
    return result
