"""On-demand causal window construction for AirSense V2.

Protocol Phase 2.

**Windows are generated, never stored.** Materialising
samples x context x features for every station, horizon and regime would
duplicate the same history millions of times and cost many gigabytes. Instead
each station's series is stored exactly once, samples are stored as compact
metadata, and the 48-hour window is assembled on request.

The canonical sample identifier is

    <partition>|<station>|<YYYYMMDDHH>|h<horizon>

for example ``train|Dongsi|2014090712|h6`` - deterministic, sortable, and
stable across rebuilds. No random identifier is ever issued.

Invariants enforced on every window:

    context_end   == forecast_origin == target_timestamp - horizon
    context_start == forecast_origin - (context_hours - 1)
    len(timestamps) == context_hours, spaced exactly one hour
    max(timestamps) == forecast_origin < target_timestamp

Target quarantine: a numeric target is returned for training samples only.
For validation and test the window carries target *metadata* - timestamp,
partition, observed flag - and no value.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.data.preprocessing import (
    CALENDAR_CHANNEL_NAMES, CALENDAR_RAW_NAMES, GAP_AGE_VARIABLES,
    NON_TARGET_NUMERIC, STATIONS, TARGET, index_of, partition_of,
    station_code, timestamp_at)

CONTEXT_HOURS_PRIMARY = 48
CONTEXT_HOURS_ROBUSTNESS = [24, 72]
HORIZONS = [1, 6, 12, 24]

REGIME_POLLUTANTS = {
    "R0": [],
    "R1": [TARGET],
    "R2": [TARGET, "PM10", "SO2", "NO2", "CO", "O3"],
    "R3": [TARGET, "PM10", "SO2", "NO2", "CO", "O3"],
}
REGIME_CROSS_STATION = {"R0": False, "R1": False, "R2": False, "R3": True}
METEOROLOGY_CHANNELS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


class WindowError(RuntimeError):
    """Raised when a requested window violates a frozen invariant."""


def make_sample_id(partition, station, target_timestamp, horizon):
    return "%s|%s|%s|h%d" % (partition, station,
                             target_timestamp.strftime("%Y%m%d%H"), horizon)


def parse_sample_id(sample_id):
    try:
        partition, station, stamp, horizon = sample_id.split("|")
        target = datetime.strptime(stamp, "%Y%m%d%H")
        return partition, station, target, int(horizon[1:])
    except (ValueError, IndexError):
        raise WindowError("malformed sample id: %r" % sample_id)


def window_bounds(target_timestamp, horizon, context_hours):
    origin = target_timestamp - timedelta(hours=horizon)
    start = origin - timedelta(hours=context_hours - 1)
    return start, origin


class StationSeries(object):
    """Canonical arrays for one station, loaded once and shared."""

    def __init__(self, root, station):
        base = Path(root) / "station_series" / station
        self.station = station
        self.numeric = np.load(str(base / "numeric_scaled.npy"))
        self.observed = np.load(str(base / "observed_mask.npy"))
        self.gap_age = np.load(str(base / "gap_age.npy"))
        self.wd_code = np.load(str(base / "wd_code.npy"))
        self.rain_occurred = np.load(str(base / "rain_occurred.npy"))
        self.fill_source = np.load(str(base / "fill_source.npy"))
        pm25_path = base / "pm25_scaled_train_only.npy"
        self.pm25_train_scaled = (np.load(str(pm25_path))
                                  if pm25_path.is_file() else None)
        with open(str(base / "meta.json"), encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.numeric_channels = self.meta["numeric_channels"]
        self.mask_channels = self.meta["mask_channels"]
        self.gap_age_channels = self.meta["gap_age_channels"]


class ForecastWindowLoader(object):
    """Model-agnostic accessor over the canonical Phase-2 data layer."""

    def __init__(self, root, context_hours=CONTEXT_HOURS_PRIMARY,
                 stations=None, target_history=None):
        self.root = Path(root)
        self.context_hours = context_hours
        self.series = {}
        for station in (stations or STATIONS):
            self.series[station] = StationSeries(self.root, station)
        self.calendar = np.load(str(self.root / "calendar" /
                                    "calendar_cyclic.npy"))
        self.calendar_raw = np.load(str(self.root / "calendar" /
                                        "calendar_raw.npy"))
        self.target_history = target_history or {}

    # -- helpers ------------------------------------------------------------

    def _validate(self, target_timestamp, horizon, start, origin, indices):
        if len(indices) != self.context_hours:
            raise WindowError("context length %d, expected %d"
                              % (len(indices), self.context_hours))
        stamps = [timestamp_at(i) for i in indices]
        for earlier, later in zip(stamps, stamps[1:]):
            if (later - earlier) != timedelta(hours=1):
                raise WindowError("context is not hourly-contiguous")
        if stamps[-1] != origin:
            raise WindowError("window end %s is not the origin %s"
                              % (stamps[-1], origin))
        if stamps[0] != start:
            raise WindowError("window start %s is not origin - %d h"
                              % (stamps[0], self.context_hours - 1))
        if max(stamps) > origin:
            raise WindowError("input timestamp after the forecast origin")
        if not origin < target_timestamp:
            raise WindowError("origin is not before the target")
        if origin + timedelta(hours=horizon) != target_timestamp:
            raise WindowError("origin + horizon does not reach the target")
        return stamps

    # -- public API ---------------------------------------------------------

    def get_window(self, sample_id, regime="R2"):
        if regime not in REGIME_POLLUTANTS:
            raise WindowError("unknown regime %r" % regime)
        partition, station, target_timestamp, horizon = parse_sample_id(
            sample_id)
        if station not in self.series:
            raise WindowError("station %r not loaded" % station)
        start, origin = window_bounds(target_timestamp, horizon,
                                      self.context_hours)
        start_index = index_of(start)
        origin_index = index_of(origin)
        if start_index < 0:
            raise WindowError("context starts before the dataset timeline")
        indices = list(range(start_index, origin_index + 1))
        stamps = self._validate(target_timestamp, horizon, start, origin,
                                indices)

        series = self.series[station]
        window = {
            "sample_id": sample_id,
            "regime": regime,
            "partition": partition,
            "station": station,
            "station_code": station_code(station),
            "context_hours": self.context_hours,
            "timestamps": [s.isoformat() for s in stamps],
            "context_start": start.isoformat(),
            "context_end": origin.isoformat(),
            "forecast_origin": origin.isoformat(),
            "horizon_hours": horizon,
            "target_timestamp": target_timestamp.isoformat(),
        }

        # R0 and upward: meteorology, wind, rain flag, calendar, masks.
        meteorology_columns = [series.numeric_channels.index(name)
                               for name in METEOROLOGY_CHANNELS]
        window["meteorology_channels"] = METEOROLOGY_CHANNELS
        window["meteorology"] = series.numeric[start_index:origin_index + 1,
                                               meteorology_columns]
        window["wd_code"] = series.wd_code[start_index:origin_index + 1]
        window["rain_occurred"] = series.rain_occurred[
            start_index:origin_index + 1]
        mask_columns = [series.mask_channels.index(name)
                        for name in METEOROLOGY_CHANNELS + ["wd"]]
        window["meteorology_masks"] = series.observed[
            start_index:origin_index + 1, mask_columns]
        window["origin_calendar"] = self.calendar[origin_index]
        window["target_calendar"] = self.calendar[index_of(target_timestamp)]
        window["origin_calendar_raw"] = self.calendar_raw[origin_index]
        window["target_calendar_raw"] = self.calendar_raw[
            index_of(target_timestamp)]
        window["calendar_channels"] = CALENDAR_CHANNEL_NAMES
        window["calendar_raw_channels"] = CALENDAR_RAW_NAMES

        # Pollutant history, regime dependent.
        pollutants = REGIME_POLLUTANTS[regime]
        history = {}
        masks = {}
        ages = {}
        for name in pollutants:
            masks[name] = series.observed[
                start_index:origin_index + 1,
                series.mask_channels.index(name)]
            ages[name] = series.gap_age[
                start_index:origin_index + 1,
                series.gap_age_channels.index(name)]
            if name == TARGET:
                history[name] = self._target_history_window(
                    station, partition, origin, start_index, origin_index)
            else:
                history[name] = series.numeric[
                    start_index:origin_index + 1,
                    series.numeric_channels.index(name)]
        window["pollutant_channels"] = pollutants
        window["pollutant_history"] = history
        window["pollutant_masks"] = masks
        window["pollutant_gap_age"] = ages

        if REGIME_CROSS_STATION[regime]:
            neighbours = {}
            for other in sorted(self.series):
                if other == station:
                    continue
                neighbours[other] = {
                    "masks": {
                        name: self.series[other].observed[
                            start_index:origin_index + 1,
                            self.series[other].mask_channels.index(name)]
                        for name in pollutants},
                    "history": {
                        name: (self._target_history_window(
                            other, partition, origin, start_index,
                            origin_index)
                            if name == TARGET else
                            self.series[other].numeric[
                                start_index:origin_index + 1,
                                self.series[other].numeric_channels
                                .index(name)])
                        for name in pollutants},
                }
            window["cross_station"] = neighbours

        window["target_partition"] = partition_of(target_timestamp)
        window["target_value_available"] = (partition == "train")
        if partition == "train":
            window["target_value_ug_m3"] = self._training_target(
                station, target_timestamp)
        return window

    # -- target access ------------------------------------------------------

    def _training_target(self, station, target_timestamp):
        path = self.root / "target_series" / ("%s_train_pm25.npy" % station)
        values = np.load(str(path))
        position = index_of(target_timestamp)
        value = values[position]
        return None if not np.isfinite(value) else float(value)

    def _target_history_window(self, station, partition, origin, start_index,
                               origin_index):
        """Scaled PM2.5 history, from the guarded interface when unsealed.

        Inside training the persisted scaled series is used. Outside it, the
        values are not materialised anywhere in Phase 2, so this returns None
        and the caller sees only the mask and gap age. Phase 3 supplies a
        ``CausalTargetHistory`` unsealed to the end of validation.
        """
        series = self.series[station]
        if partition == "train" and series.pm25_train_scaled is not None:
            return series.pm25_train_scaled[start_index:origin_index + 1]
        accessor = self.target_history.get(station)
        if accessor is None:
            return None
        return accessor.values(origin, self.context_hours)
