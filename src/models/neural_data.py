"""Sample bundles for AirSense V2 temporal neural baselines.

Protocol Phase 4.

Assembles the canonical Phase-2 samples into the arrays the neural trainers
consume. The Phase-2 sample universe is never rewritten: the internal
core/tuning split is a *training procedure*, applied by filtering the existing
train index on target timestamp, and the external validation index is used
exactly as frozen.

PM2.5 history comes from one of two places and never from anywhere else:
inside training, the Phase-2 train-only scaled array; for external validation,
the Phase-3 guarded rolling series, which cannot return a value later than the
forecast origin and was never unsealed past the end of validation.
"""

import csv
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.preprocessing import (
    PARTITION_BOUNDS, STATIONS, TARGET, index_of, load_training_statistics,
    station_code, timestamp_at)
from src.data.rolling_history import RollingPm25Series
from src.data.windowing import StationSeries
from src.models.neural_features import (
    build_static_matrix, build_station_dynamic_matrix)

INTERNAL_CORE = (datetime(2013, 3, 1, 0), datetime(2014, 2, 28, 23))
INTERNAL_TUNING = (datetime(2014, 3, 1, 0), datetime(2015, 2, 28, 23))


def load_sample_index(processed, partition):
    stations, horizons, targets = [], [], []
    with open(str(Path(processed) / "sample_index" / ("%s.csv" % partition)),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stations.append(row["station"])
            horizons.append(int(row["horizon_hours"]))
            targets.append(int(row["target_row_index"]))
    return (np.array(stations), np.array(horizons, dtype=np.int64),
            np.array(targets, dtype=np.int64))


def internal_mask(target_indices, window):
    low, high = index_of(window[0]), index_of(window[1])
    return (target_indices >= low) & (target_indices <= high)


class NeuralDataSource(object):
    """Loads the canonical layer once and serves regime-specific bundles."""

    def __init__(self, project_root, unseal_validation=False):
        self.root = Path(project_root)
        self.processed = self.root / "data" / "processed" / "phase2"
        self.calendar = np.load(str(self.processed / "calendar"
                                    / "calendar_cyclic.npy"))
        self.series = OrderedDict()
        self.train_pm25_scaled = OrderedDict()
        self.train_target_native = OrderedDict()
        self.rolling_pm25_scaled = OrderedDict()
        statistics = load_training_statistics(
            self.root / "configs" / "preprocessing.json")
        station_stats = json.load(open(str(
            self.root / "artifacts" / "preprocessing_statistics.json"),
            encoding="utf-8"))["station_training_medians"]
        raw_dir = (self.root / "data" / "raw"
                   / "PRSA_Data_20130301-20170228")
        for station in STATIONS:
            self.series[station] = StationSeries(self.processed, station)
            self.train_pm25_scaled[station] = np.load(str(
                self.processed / "station_series" / station
                / "pm25_scaled_train_only.npy"))
            self.train_target_native[station] = np.load(str(
                self.processed / "target_series"
                / ("%s_train_pm25.npy" % station)))
            if unseal_validation:
                path = next(raw_dir.glob("PRSA_Data_%s_*.csv" % station))
                rolling = RollingPm25Series(
                    station, path, statistics,
                    station_stats[station][TARGET])
                self.rolling_pm25_scaled[station] = rolling._scaled

    def dynamic_by_station(self, regime, pm25_source):
        source = (self.train_pm25_scaled if pm25_source == "train"
                  else self.rolling_pm25_scaled)
        blocks = [build_station_dynamic_matrix(regime, self.series[station],
                                               source[station])
                  for station in STATIONS]
        return np.stack(blocks, axis=0)

    def bundle(self, regime, stations, horizons, target_indices,
               pm25_source="train", with_labels=True):
        from src.models.neural_training import SampleBundle
        codes = np.array([station_code(s) for s in stations], dtype=np.int64)
        origins = target_indices - horizons
        static = build_static_matrix(codes, horizons, origins, target_indices,
                                     self.calendar)
        labels = None
        if with_labels:
            labels = np.empty(target_indices.size, dtype=np.float32)
            for station in STATIONS:
                rows = np.flatnonzero(stations == station)
                labels[rows] = self.train_target_native[station][
                    target_indices[rows]]
            if not np.isfinite(labels).all():
                raise ValueError("a training label is missing; the canonical "
                                 "universe promised observed targets")
        return SampleBundle(codes, horizons, origins, target_indices,
                            self.dynamic_by_station(regime, pm25_source),
                            static, labels)
