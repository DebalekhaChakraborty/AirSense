"""Neural sequence and static feature construction for AirSense V2.

Protocol Phase 4. **Frozen before any candidate fit begins.**

Channels are derived from the Phase-2 canonical layer and the frozen
information-regime schema. No new source variable is invented, and nothing
later than the forecast origin can enter a window: a sample's dynamic block is
literally the canonical 48 hours ending at the origin.

Two representation decisions are declared here rather than discovered later:

* **wind direction is one-hot**, 16 frozen training categories plus MISSING -
  never an ordinal number, because compass codes carry no numeric order;
* **gap age is divided by its frozen 168-hour cap**, a deterministic rescaling
  into [0, 1] so that a staleness channel does not dominate a network input
  purely by magnitude. The cap and the convention are Phase-2's; only the
  division is new, and it is frozen in
  ``artifacts/neural_feature_schema.json`` before training.

Static conditioning is station one-hot (12), horizon one-hot (4, in the frozen
order 1/6/12/24) and the frozen origin and target cyclic calendar channels.
``day_of_week`` and ``weekend`` are excluded, as in every earlier phase.
"""

from collections import OrderedDict

import numpy as np

from src.data.preprocessing import (
    CALENDAR_CHANNEL_NAMES, GAP_AGE_CAP_HOURS, STATIONS, TARGET,
    WD_MISSING_CODE, WD_VOCABULARY)
from src.models.b3_features import REGIME_NUMERIC, REGIME_POLLUTANTS

CONTEXT_HOURS = 48
HORIZON_VOCABULARY = [1, 6, 12, 24]
DTYPE = "float32"


def dynamic_channel_names(regime):
    """Frozen dynamic sequence channel order for a regime."""
    names = []
    for variable in REGIME_NUMERIC[regime]:
        names.append("value:%s" % variable)
    for variable in REGIME_NUMERIC[regime]:
        names.append("mask:%s" % variable)
    for variable in REGIME_POLLUTANTS[regime]:
        names.append("gap_age_scaled:%s" % variable)
    names.append("rain_occurred")
    for category in WD_VOCABULARY + ["MISSING"]:
        names.append("wd_is_%s" % category)
    return names


def static_channel_names():
    names = ["station_is_%s" % s for s in STATIONS]
    names += ["horizon_is_%dh" % h for h in HORIZON_VOCABULARY]
    names += ["origin_%s" % c for c in CALENDAR_CHANNEL_NAMES]
    names += ["target_%s" % c for c in CALENDAR_CHANNEL_NAMES]
    return names


def feature_schema():
    schema = OrderedDict([
        ("context_hours", CONTEXT_HOURS),
        ("dtype", DTYPE),
        ("horizon_vocabulary", HORIZON_VOCABULARY),
        ("station_vocabulary", list(STATIONS)),
        ("wind_vocabulary", list(WD_VOCABULARY) + ["MISSING"]),
        ("wind_representation", "deterministic one-hot; wd_code is never "
                                "used as an ordinal numeric input"),
        ("gap_age_scaling", "value / %d (the frozen Phase-2 cap)"
                            % GAP_AGE_CAP_HOURS),
        ("static_channels", static_channel_names()),
        ("static_channel_count", len(static_channel_names())),
        ("dynamic_channels", OrderedDict(
            (regime, dynamic_channel_names(regime))
            for regime in ("R0", "R1", "R2"))),
        ("dynamic_channel_counts", OrderedDict(
            (regime, len(dynamic_channel_names(regime)))
            for regime in ("R0", "R1", "R2"))),
        ("day_of_week_included", False),
        ("weekend_included", False),
        ("target_encoding_used", False),
    ])
    return schema


def build_station_dynamic_matrix(regime, series, pm25_scaled):
    """Full-timeline dynamic channel matrix for one station.

    `pm25_scaled` supplies the target-station PM2.5 history: the Phase-2
    train-only array inside training, or the guarded rolling series when
    validation is open. Positions the caller must not read are NaN there, and
    windows are only ever taken at origins the guard permits.
    """
    length = series.numeric.shape[0]
    columns = []
    for variable in REGIME_NUMERIC[regime]:
        if variable == TARGET:
            columns.append(np.asarray(pm25_scaled, dtype=np.float32))
        else:
            columns.append(series.numeric[
                :, series.numeric_channels.index(variable)])
    for variable in REGIME_NUMERIC[regime]:
        columns.append(series.observed[
            :, series.mask_channels.index(variable)].astype(np.float32))
    for variable in REGIME_POLLUTANTS[regime]:
        ages = series.gap_age[:, series.gap_age_channels.index(variable)]
        columns.append(ages.astype(np.float32) / float(GAP_AGE_CAP_HOURS))
    columns.append(series.rain_occurred.astype(np.float32))
    codes = series.wd_code
    for index in range(len(WD_VOCABULARY) + 1):
        columns.append((codes == index).astype(np.float32))
    matrix = np.column_stack(columns).astype(np.float32)
    expected = len(dynamic_channel_names(regime))
    if matrix.shape != (length, expected):
        raise ValueError("built %s, frozen schema declares %d channels"
                         % (matrix.shape, expected))
    return matrix


def build_static_matrix(station_codes, horizons, origin_indices,
                        target_indices, calendar):
    """Static conditioning rows aligned to the sample order given."""
    n = len(station_codes)
    blocks = []
    station_onehot = np.zeros((n, len(STATIONS)), dtype=np.float32)
    station_onehot[np.arange(n), np.asarray(station_codes)] = 1.0
    blocks.append(station_onehot)
    horizon_onehot = np.zeros((n, len(HORIZON_VOCABULARY)), dtype=np.float32)
    lookup = {h: i for i, h in enumerate(HORIZON_VOCABULARY)}
    horizon_onehot[np.arange(n),
                   [lookup[int(h)] for h in horizons]] = 1.0
    blocks.append(horizon_onehot)
    blocks.append(calendar[np.asarray(origin_indices)].astype(np.float32))
    blocks.append(calendar[np.asarray(target_indices)].astype(np.float32))
    matrix = np.hstack(blocks).astype(np.float32)
    if matrix.shape[1] != len(static_channel_names()):
        raise ValueError("static width %d, schema declares %d"
                         % (matrix.shape[1], len(static_channel_names())))
    return matrix


def gather_windows(dynamic_by_station, station_codes, origin_indices):
    """(N, 48, C) windows ending at each origin. Never reads past the origin."""
    origins = np.asarray(origin_indices, dtype=np.int64)
    if (origins - (CONTEXT_HOURS - 1) < 0).any():
        raise ValueError("a sample lacks a complete 48-hour context")
    offsets = np.arange(-(CONTEXT_HOURS - 1), 1, dtype=np.int64)
    positions = origins[:, None] + offsets[None, :]
    return dynamic_by_station[np.asarray(station_codes)[:, None], positions]
