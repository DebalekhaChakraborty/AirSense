"""Classical forecasting baselines B0-B2 for AirSense V2.

Protocol Phase 3. Definitions were frozen in Phase 2 and are implemented here
without modification.

Every baseline produces a prediction for **every** canonical eligible sample,
so none of them can score better by declining the hard cases. All PM2.5
history is read through the guarded rolling interface, which cannot return
anything later than the forecast origin.

Predictions are in native ug/m3 and are **never clipped**, even where a value
is implausible - clipping after the fact would alter the metric.
"""

from collections import OrderedDict

import numpy as np

B0_SOURCE_OBSERVED = 0
B0_SOURCE_CARRY = 1
B0_SOURCE_MEDIAN = 2
B0_SOURCE_LABELS = {0: "observed_at_origin", 1: "short_causal_carry",
                    2: "station_training_median_fallback"}

SEASONAL_LAG_HOURS = 24


def b0_causal_persistence(rolling, origin_indices):
    """Latest usable PM2.5 observation available at the forecast origin.

    Resolution order, exactly as frozen: observation at the origin, else a
    causal carry within 6 hours, else the station training median.
    """
    origin_indices = np.asarray(origin_indices, dtype=np.int64)
    predictions = np.empty(origin_indices.size, dtype=np.float64)
    sources = np.empty(origin_indices.size, dtype=np.int8)
    for position, origin in enumerate(origin_indices):
        value = rolling.native_at(int(origin), int(origin))
        if value is None:
            raise RuntimeError("B0 produced no prediction at origin %d"
                               % origin)
        predictions[position] = value
        sources[position] = rolling.fill_source_at(int(origin))
    return predictions, sources


def b1_seasonal_naive(rolling, origin_indices, target_indices,
                      b0_predictions):
    """PM2.5 observed at target - 24 h when causally available, else B0.

    At horizon 24 the seasonal lag coincides with the origin, so B1 reduces to
    B0 by arithmetic. The collapse is measured, not hidden.
    """
    origin_indices = np.asarray(origin_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    predictions = np.array(b0_predictions, dtype=np.float64, copy=True)
    used_direct = np.zeros(origin_indices.size, dtype=bool)
    for position in range(origin_indices.size):
        seasonal = int(target_indices[position]) - SEASONAL_LAG_HOURS
        origin = int(origin_indices[position])
        if seasonal < 0 or seasonal > origin:
            continue
        value = rolling.observed_native_at(seasonal, origin)
        if value is None:
            continue
        predictions[position] = value
        used_direct[position] = True
    return predictions, used_direct


class TrainingClimatology(object):
    """B2: frozen training-only medians with a fixed fallback hierarchy."""

    LEVELS = ["station_month_hour", "station_hour", "station"]

    def __init__(self, rows):
        self.month_hour = {}
        self.hour = {}
        self.station = {}
        for row in rows:
            value = row["median_pm25_ug_m3"]
            if value == "":
                continue
            value = float(value)
            name = row["station"]
            if row["level"] == "station_month_hour":
                self.month_hour[(name, int(row["month"]),
                                 int(row["hour"]))] = value
            elif row["level"] == "station_hour":
                self.hour[(name, int(row["hour"]))] = value
            else:
                self.station[name] = value

    def predict(self, station, month, hour):
        value = self.month_hour.get((station, month, hour))
        if value is not None:
            return value, 0
        value = self.hour.get((station, hour))
        if value is not None:
            return value, 1
        value = self.station.get(station)
        if value is None:
            raise RuntimeError("no training climatology for %s" % station)
        return value, 2


def b2_climatology(climatology, station, months, hours):
    months = np.asarray(months, dtype=np.int64)
    hours = np.asarray(hours, dtype=np.int64)
    predictions = np.empty(months.size, dtype=np.float64)
    levels = np.empty(months.size, dtype=np.int8)
    for position in range(months.size):
        value, level = climatology.predict(station, int(months[position]),
                                           int(hours[position]))
        predictions[position] = value
        levels[position] = level
    return predictions, levels


def source_frequencies(sources, labels):
    counts = OrderedDict()
    for code, label in sorted(labels.items()):
        counts[label] = int((np.asarray(sources) == code).sum())
    return counts
