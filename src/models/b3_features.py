"""B3 causal feature construction for AirSense V2.

Protocol Phase 3. **Frozen before validation was opened.**

Features are derived from the canonical Phase-2 48-hour window and nothing
else. No model-specific source data is created, no feature consults a
timestamp later than the forecast origin, and no feature was added or removed
after validation results were seen.

Deterministic layout, in this exact order:

1. per numeric variable of the regime: 8 exact lags, then 5 trailing windows
   x {mean, std, min, max};
2. per numeric variable: observed mask at origin, observed fraction over
   6 / 24 / 48 h;
3. per pollutant of the regime: gap age at origin;
4. rain_occurred at origin, then its trailing mean over 3 / 6 / 12 / 24 / 48 h;
5. wind direction one-hot at origin, 16 frozen training categories + MISSING;
6. station one-hot over the frozen 12-station vocabulary;
7. origin and target cyclic calendar channels.

Horizon is not a column: models are trained per horizon, so it is constant
within a matrix. Trailing statistics use population standard deviation
(ddof = 0) and windows that end **at** the origin, never centred.
"""

from collections import OrderedDict

import numpy as np

from src.data.preprocessing import (
    CALENDAR_CHANNEL_NAMES, GAP_AGE_VARIABLES, NON_TARGET_NUMERIC, STATIONS,
    TARGET, WD_MISSING_CODE, WD_VOCABULARY)

LAGS = [0, 1, 2, 3, 6, 12, 24, 47]
SUMMARY_WINDOWS = [3, 6, 12, 24, 48]
SUMMARY_STATS = ["mean", "std", "min", "max"]
MASK_FRACTION_WINDOWS = [6, 24, 48]
CONTEXT_HOURS = 48

REGIME_NUMERIC = OrderedDict([
    ("R0", ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]),
    ("R1", ["PM2.5", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]),
    ("R2", ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
            "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]),
])
REGIME_POLLUTANTS = OrderedDict([
    ("R0", []),
    ("R1", ["PM2.5"]),
    ("R2", ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]),
])


def feature_names(regime):
    """Deterministic, frozen feature order for a regime."""
    names = []
    for variable in REGIME_NUMERIC[regime]:
        for lag in LAGS:
            names.append("%s_lag%d" % (variable, lag))
        for window in SUMMARY_WINDOWS:
            for stat in SUMMARY_STATS:
                names.append("%s_w%d_%s" % (variable, window, stat))
    for variable in REGIME_NUMERIC[regime]:
        names.append("%s_observed_at_origin" % variable)
        for window in MASK_FRACTION_WINDOWS:
            names.append("%s_observed_frac_%dh" % (variable, window))
    for variable in REGIME_POLLUTANTS[regime]:
        names.append("%s_gap_age_at_origin" % variable)
    names.append("rain_occurred_at_origin")
    for window in SUMMARY_WINDOWS:
        names.append("rain_occurred_w%d_mean" % window)
    for category in WD_VOCABULARY + ["MISSING"]:
        names.append("wd_is_%s" % category)
    for station in STATIONS:
        names.append("station_is_%s" % station)
    for channel in CALENDAR_CHANNEL_NAMES:
        names.append("origin_%s" % channel)
    for channel in CALENDAR_CHANNEL_NAMES:
        names.append("target_%s" % channel)
    return names


def trailing_mean(series, window):
    """Mean over [i-window+1, i]. Positions with a partial window are NaN."""
    series = np.asarray(series, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(series)))
    out = np.full(series.size, np.nan)
    out[window - 1:] = (cumulative[window:] - cumulative[:-window]) / window
    return out


def trailing_std(series, window):
    """Population standard deviation over the trailing window (ddof = 0)."""
    series = np.asarray(series, dtype=np.float64)
    mean = trailing_mean(series, window)
    mean_square = trailing_mean(series ** 2, window)
    variance = np.maximum(mean_square - mean ** 2, 0.0)
    return np.sqrt(variance)


def trailing_min_max(series, window):
    series = np.asarray(series, dtype=np.float64)
    view = np.lib.stride_tricks.sliding_window_view(series, window)
    minimum = np.full(series.size, np.nan)
    maximum = np.full(series.size, np.nan)
    minimum[window - 1:] = view.min(axis=1)
    maximum[window - 1:] = view.max(axis=1)
    return minimum, maximum


class StationFeatureSource(object):
    """Precomputed causal channels for one station over the full timeline."""

    def __init__(self, series, calendar, pm25_scaled):
        self.numeric_channels = series.numeric_channels
        self.mask_channels = series.mask_channels
        self.gap_age_channels = series.gap_age_channels
        self.numeric = series.numeric
        self.observed = series.observed
        self.gap_age = series.gap_age
        self.wd_code = series.wd_code
        self.rain_occurred = series.rain_occurred
        self.calendar = calendar
        self.pm25_scaled = pm25_scaled
        self._cache = {}

    def variable_series(self, variable):
        if variable == TARGET:
            return self.pm25_scaled
        return self.numeric[:, self.numeric_channels.index(variable)]

    def mask_series(self, variable):
        return self.observed[:, self.mask_channels.index(variable)].astype(
            np.float64)

    def rolled(self, key, series, window, stat):
        cache_key = (key, window, stat)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if stat == "mean":
            value = trailing_mean(series, window)
        elif stat == "std":
            value = trailing_std(series, window)
        else:
            minimum, maximum = trailing_min_max(series, window)
            self._cache[(key, window, "min")] = minimum
            self._cache[(key, window, "max")] = maximum
            value = minimum if stat == "min" else maximum
        self._cache[cache_key] = value
        return value


def build_matrix(regime, source, station, origin_indices, target_indices):
    """Feature matrix for one station's samples. Rows follow the input order."""
    origin_indices = np.asarray(origin_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if (origin_indices - (CONTEXT_HOURS - 1) < 0).any():
        raise ValueError("a sample lacks a complete 48-hour context")
    columns = []

    for variable in REGIME_NUMERIC[regime]:
        series = source.variable_series(variable)
        for lag in LAGS:
            columns.append(series[origin_indices - lag])
        for window in SUMMARY_WINDOWS:
            for stat in SUMMARY_STATS:
                rolled = source.rolled(variable, series, window, stat)
                columns.append(rolled[origin_indices])

    for variable in REGIME_NUMERIC[regime]:
        mask = source.mask_series(variable)
        columns.append(mask[origin_indices])
        for window in MASK_FRACTION_WINDOWS:
            rolled = source.rolled("mask:" + variable, mask, window, "mean")
            columns.append(rolled[origin_indices])

    for variable in REGIME_POLLUTANTS[regime]:
        ages = source.gap_age[:, source.gap_age_channels.index(variable)]
        columns.append(ages[origin_indices].astype(np.float64))

    rain = source.rain_occurred.astype(np.float64)
    columns.append(rain[origin_indices])
    for window in SUMMARY_WINDOWS:
        columns.append(source.rolled("rain", rain, window, "mean")
                       [origin_indices])

    codes = source.wd_code[origin_indices]
    for index in range(len(WD_VOCABULARY) + 1):
        columns.append((codes == index).astype(np.float64))

    for name in STATIONS:
        columns.append(np.full(origin_indices.size,
                               1.0 if name == station else 0.0))

    for position in range(len(CALENDAR_CHANNEL_NAMES)):
        columns.append(source.calendar[origin_indices, position]
                       .astype(np.float64))
    for position in range(len(CALENDAR_CHANNEL_NAMES)):
        columns.append(source.calendar[target_indices, position]
                       .astype(np.float64))

    matrix = np.column_stack(columns).astype(np.float32)
    expected = len(feature_names(regime))
    if matrix.shape[1] != expected:
        raise ValueError("built %d columns, frozen schema declares %d"
                         % (matrix.shape[1], expected))
    return matrix
