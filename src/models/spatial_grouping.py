"""Grouped station-network representation for AirSense V2 Phase 6.

Protocol Phase 6, sections 10-14.

Phase 6 asks whether the *other* stations help. A model that answers it must
see every station at one shared forecast origin, so the canonical samples are
regrouped rather than rebuilt:

    group = (target timestamp, horizon)  ->  up to 12 station slots

**The grouping is a computational representation only.** It never changes
which predictions count. Every group carries the canonical sample row of each
observed station target, so a grouped forward pass scatters straight back into
the frozen Phase-2 sample order, and a station slot with no canonical sample
contributes to neither loss nor metric.

Causality, asserted per station rather than assumed:

* a window is the 48 hours **ending at the shared origin**, for all 12
  stations, so no station can contribute an observation later than the origin;
* validation PM2.5 reaches the loader only through the Phase-3 guarded rolling
  series, whose array is forward-filled and never built beyond the unseal
  ceiling, so the locked test period is not on disk here at all;
* ``GuardedNetworkHistory`` keeps that array private and refuses any window
  whose origin lies beyond the ceiling, so the tensor the model sees is never
  the unrestricted validation network.
"""

import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from src.data.preprocessing import (
    PARTITION_BOUNDS, STATIONS, TARGET, index_of, load_training_statistics,
    timestamp_at)
from src.data.rolling_history import RollingPm25Series
from src.data.target_history import TargetAccessError
from src.data.windowing import StationSeries
from src.models.neural_features import (
    CONTEXT_HOURS, build_static_matrix, build_station_dynamic_matrix,
    dynamic_channel_names, static_channel_names)

STATION_COUNT = len(STATIONS)
GROUP_REGIME = "R2"
NO_SAMPLE = -1


def group_id(partition, target_index, horizon):
    """Deterministic group identity: <partition>|<YYYYMMDDHH>|h<horizon>."""
    return "%s|%s|h%d" % (partition,
                          timestamp_at(int(target_index)).strftime("%Y%m%d%H"),
                          int(horizon))


class StationGroups(object):
    """Canonical samples regrouped by shared origin, nothing added or lost."""

    def __init__(self, partition, target_indices, horizons, sample_rows,
                 labels=None):
        self.partition = partition
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.horizons = np.asarray(horizons, dtype=np.int64)
        self.origins = self.target_indices - self.horizons
        self.sample_rows = np.asarray(sample_rows, dtype=np.int64)
        self.target_observed = self.sample_rows != NO_SAMPLE
        self.labels = (None if labels is None
                       else np.asarray(labels, dtype=np.float32))
        self.size = self.target_indices.size

    def identities(self):
        return [group_id(self.partition, t, h)
                for t, h in zip(self.target_indices, self.horizons)]

    def sample_total(self):
        return int(self.target_observed.sum())

    def scatter(self, grouped_predictions, sample_count):
        """Grouped (G, 12) predictions back into canonical sample order."""
        out = np.full(sample_count, np.nan, dtype=np.float64)
        rows = self.sample_rows[self.target_observed]
        out[rows] = np.asarray(grouped_predictions)[self.target_observed]
        if not np.isfinite(out).all():
            raise ValueError("%d canonical samples received no prediction"
                             % int((~np.isfinite(out)).sum()))
        return out


def build_groups(processed, partition, labels_from=None):
    """Regroup the frozen canonical index. Membership is never edited."""
    keys = OrderedDict()
    rows = []
    path = Path(processed) / "sample_index" / ("%s.csv" % partition)
    with open(str(path), encoding="utf-8") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            target = int(row["target_row_index"])
            horizon = int(row["horizon_hours"])
            key = (target, horizon)
            if key not in keys:
                keys[key] = len(keys)
            rows.append((keys[key], STATIONS.index(row["station"]), row_index,
                         target))
    order = sorted(keys, key=lambda k: (k[0], k[1]))
    remap = {keys[k]: i for i, k in enumerate(order)}
    count = len(order)
    sample_rows = np.full((count, STATION_COUNT), NO_SAMPLE, dtype=np.int64)
    for group, station, row_index, _ in rows:
        sample_rows[remap[group], station] = row_index
    target_indices = np.array([k[0] for k in order], dtype=np.int64)
    horizons = np.array([k[1] for k in order], dtype=np.int64)

    labels = None
    if labels_from is not None:
        labels = np.zeros((count, STATION_COUNT), dtype=np.float32)
        for station_index, station in enumerate(STATIONS):
            native = labels_from[station]
            observed = sample_rows[:, station_index] != NO_SAMPLE
            labels[observed, station_index] = native[
                target_indices[observed]].astype(np.float32)
        if not np.isfinite(labels).all():
            raise ValueError("a grouped training label is not finite")
    groups = StationGroups(partition, target_indices, horizons, sample_rows,
                           labels)
    if groups.sample_total() != len(rows):
        raise ValueError("grouping lost %d canonical samples"
                         % (len(rows) - groups.sample_total()))
    return groups


class GuardedNetworkHistory(object):
    """The 12-station dynamic tensor, reachable only one origin at a time.

    The full array stays private. Every request names a forecast origin and
    receives the 48 hours ending there, for all 12 stations at once, after an
    explicit check that the origin is inside the unsealed region.
    """

    def __init__(self, project_root, regime=GROUP_REGIME,
                 unseal_validation=False):
        root = Path(project_root)
        self.regime = regime
        self.processed = root / "data" / "processed" / "phase2"
        self.channels = dynamic_channel_names(regime)
        self.calendar = np.load(str(self.processed / "calendar"
                                    / "calendar_cyclic.npy"))
        self._rolling = OrderedDict()
        statistics = load_training_statistics(
            root / "configs" / "preprocessing.json")
        with open(str(root / "artifacts"
                      / "preprocessing_statistics.json"),
                  encoding="utf-8") as handle:
            station_stats = json.load(handle)["station_training_medians"]
        raw_dir = root / "data" / "raw" / "PRSA_Data_20130301-20170228"
        blocks = []
        self.train_target_native = OrderedDict()
        for station in STATIONS:
            series = StationSeries(self.processed, station)
            self.train_target_native[station] = np.load(str(
                self.processed / "target_series"
                / ("%s_train_pm25.npy" % station)))
            if unseal_validation:
                path = next(raw_dir.glob("PRSA_Data_%s_*.csv" % station))
                rolling = RollingPm25Series(station, path, statistics,
                                            station_stats[station][TARGET])
                self._rolling[station] = rolling
                pm25 = rolling._scaled
            else:
                pm25 = np.load(str(self.processed / "station_series" / station
                                   / "pm25_scaled_train_only.npy"))
            blocks.append(build_station_dynamic_matrix(regime, series, pm25))
        self.__dynamic = np.stack(blocks, axis=0)
        self.unsealed = unseal_validation
        self._ceiling = (index_of(PARTITION_BOUNDS["validation"][1])
                         if unseal_validation
                         else index_of(PARTITION_BOUNDS["train"][1]))
        self._station_codes = np.arange(STATION_COUNT, dtype=np.int64)

    @property
    def dynamic_channel_count(self):
        return len(self.channels)

    def windows(self, origins):
        """(B, 12, 48, C) blocks ending at each origin. Never past it."""
        origins = np.asarray(origins, dtype=np.int64)
        if origins.size and int(origins.max()) > self._ceiling:
            raise TargetAccessError(
                "origin %d is beyond the unsealed ceiling %d"
                % (int(origins.max()), self._ceiling))
        if origins.size and int(origins.min()) - (CONTEXT_HOURS - 1) < 0:
            raise TargetAccessError("a group lacks a complete 48-hour context")
        offsets = np.arange(-(CONTEXT_HOURS - 1), 1, dtype=np.int64)
        positions = origins[:, None] + offsets[None, :]
        # (B, 48) positions -> (B, 12, 48, C); the last position is the origin
        block = self.__dynamic[:, positions, :]
        block = np.transpose(block, (1, 0, 2, 3))
        if block.shape[2] != CONTEXT_HOURS:
            raise ValueError("window depth %d" % block.shape[2])
        if not np.array_equal(positions[:, -1], origins):
            raise ValueError("a window does not end at its origin")
        return np.ascontiguousarray(block, dtype=np.float32)

    def statics(self, origins, targets, horizons):
        """(B, 12, static) conditioning, one row per station slot."""
        origins = np.asarray(origins, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        horizons = np.asarray(horizons, dtype=np.int64)
        count = origins.size
        codes = np.tile(self._station_codes, count)
        rows = build_static_matrix(codes, np.repeat(horizons, STATION_COUNT),
                                   np.repeat(origins, STATION_COUNT),
                                   np.repeat(targets, STATION_COUNT),
                                   self.calendar)
        return np.ascontiguousarray(
            rows.reshape(count, STATION_COUNT, len(static_channel_names())))

    def guarded_pm25_window(self, station, origin):
        """The Phase-3 guarded accessor, for independent cross-checking."""
        if station not in self._rolling:
            raise TargetAccessError("station %s is not unsealed" % station)
        return self._rolling[station].scaled_window(int(origin),
                                                    CONTEXT_HOURS)


class NetworkBundle(object):
    """One partition's groups plus the history they read."""

    def __init__(self, groups, history):
        self.groups = groups
        self.history = history
        self.size = groups.size

    def batch(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        sequence = self.history.windows(self.groups.origins[rows])
        static = self.history.statics(self.groups.origins[rows],
                                      self.groups.target_indices[rows],
                                      self.groups.horizons[rows])
        if not np.isfinite(sequence).all():
            raise ValueError("a station window contains a non-finite value; "
                             "a sealed or unavailable position was reached")
        return sequence, static
