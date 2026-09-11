"""Locked final-test evaluation layer for AirSense V2.

Protocol Phase 8. **Frozen before the sealed test was numerically opened.**

This module is the only place in the codebase permitted to raise the PM2.5
unseal ceiling into the final-test period, and it does so behind the same two
guards Phase 2 established: an *origin* guard, which refuses any observation
later than the forecast origin of the sample being built, and an *unseal*
guard, which bounds what is materialised off disk at all.

Three separations are enforced structurally rather than by convention:

``prediction before scoring``
    :class:`FinalTestTargetOracle` refuses to hand back a single actual until
    the caller presents predictions it has already produced, and the runner
    writes the prediction lock before the oracle is ever constructed.

``chronological reveal``
    Forecast origins are processed in strictly increasing timestamp order.
    Within a block, every input channel is materialised only up to the block's
    reveal ceiling; positions after it hold NaN, not a value. A feature that
    reached past the ceiling would therefore produce a non-finite column, and
    both model paths reject non-finite input. The leak is made to fail loudly
    instead of being trusted not to happen.

``per-row origin bound``
    The block ceiling bounds exposure to the block; :class:`RevealLedger`
    bounds it to the individual sample, asserting for every index array handed
    to a frozen feature builder that each element is at or before that row's
    own forecast origin.

The decisive evidence is not any of these assertions but the non-influence
test in ``tests/test_phase8_final_test.py``: corrupting every observation
strictly after an origin leaves that origin's prediction bit-identical.
"""

from collections import OrderedDict
import csv
from datetime import datetime

import numpy as np

from src.data.preprocessing import (
    MISSING_TOKEN, PARTITION_BOUNDS, STATIONS, TARGET, index_of, timestamp_at,
    total_hours)
from src.data.target_history import TargetAccessError
from src.models.b3_features import CONTEXT_HOURS
from src.models.neural_features import dynamic_channel_names

TEST_START, TEST_END = PARTITION_BOUNDS["test"]
EXPECTED_TEST_SAMPLES = 411012
HORIZONS = [1, 6, 12, 24]
CONFIRMATORY_MODELS = ["B3_R2", "GRU_R1", "B0"]
SEVERE_THRESHOLD = 244.0

# Sentinel for the categorical channel, which cannot carry NaN. A masked hour
# yields an all-zero one-hot row, which `verify_matrix` detects.
WD_SENTINEL = -99


class FinalTestAccessError(RuntimeError):
    """Raised on any attempt to read final-test data out of protocol order."""


# ---------------------------------------------------------------------------
# Chronological reveal
# ---------------------------------------------------------------------------

class RevealLedger(object):
    """Monotonic reveal ceiling and future-access violation counter.

    The ceiling only ever moves forward. Every index array a frozen feature
    builder is about to receive is checked elementwise against the forecast
    origin of its own row, which is a strictly tighter bound than the ceiling.
    """

    def __init__(self, name="final_test"):
        self.name = name
        self.ceiling_index = -1
        self.blocks_advanced = 0
        self.checks = 0
        self.indices_checked = 0
        self.violations = 0
        self.max_index_seen = -1
        self.max_origin_seen = -1
        self.detail = []

    def advance_to(self, ceiling_index):
        """Move the reveal ceiling forward. Never backwards."""
        ceiling_index = int(ceiling_index)
        if ceiling_index < self.ceiling_index:
            raise FinalTestAccessError(
                "reveal ceiling would move backwards: %d after %d"
                % (ceiling_index, self.ceiling_index))
        self.ceiling_index = ceiling_index
        self.blocks_advanced += 1

    def note(self, label, indices, origins):
        """Assert every index is at or before its row's forecast origin."""
        indices = np.asarray(indices, dtype=np.int64)
        origins = np.asarray(origins, dtype=np.int64)
        self.checks += 1
        self.indices_checked += int(indices.size)
        if indices.size:
            self.max_index_seen = max(self.max_index_seen,
                                      int(indices.max()))
        if origins.size:
            self.max_origin_seen = max(self.max_origin_seen,
                                       int(origins.max()))
        late = indices > origins
        beyond = indices > self.ceiling_index
        offending = int(np.count_nonzero(late | beyond))
        if offending:
            self.violations += offending
            self.detail.append("%s: %d index(es) after the permitted bound"
                               % (label, offending))
        return offending

    def summary(self):
        return OrderedDict([
            ("reveal_ceiling_index", self.ceiling_index),
            ("blocks_advanced", self.blocks_advanced),
            ("index_checks", self.checks),
            ("indices_checked", self.indices_checked),
            ("max_index_read", self.max_index_seen),
            ("max_forecast_origin", self.max_origin_seen),
            ("future_target_access_violations", self.violations),
            ("violation_detail", list(self.detail)),
        ])


class MaskedStationInputs(object):
    """Station channels materialised only up to a reveal ceiling.

    Duck-types :class:`src.data.windowing.StationSeries` so the frozen Phase-3
    and Phase-4 feature builders consume it unchanged. Every position after
    ``ceiling_index`` is NaN (or the wind sentinel), so a feature that reached
    past the ceiling cannot silently succeed.
    """

    def __init__(self, series, pm25_scaled, ceiling_index):
        self.station = series.station
        self.ceiling_index = int(ceiling_index)
        cut = self.ceiling_index + 1
        self.numeric_channels = series.numeric_channels
        self.mask_channels = series.mask_channels
        self.gap_age_channels = series.gap_age_channels

        self.numeric = np.array(series.numeric, dtype=np.float32)
        self.numeric[cut:, :] = np.nan
        # Masks, gap ages and rain are integral upstream; they become float so
        # that an out-of-ceiling read is non-finite rather than plausible.
        self.observed = np.asarray(series.observed, dtype=np.float32).copy()
        self.observed[cut:, :] = np.nan
        self.gap_age = np.asarray(series.gap_age, dtype=np.float32).copy()
        self.gap_age[cut:, :] = np.nan
        self.rain_occurred = np.asarray(series.rain_occurred,
                                        dtype=np.float32).copy()
        self.rain_occurred[cut:] = np.nan
        self.wd_code = np.asarray(series.wd_code, dtype=np.int16).copy()
        self.wd_code[cut:] = WD_SENTINEL

        self.pm25_scaled = np.asarray(pm25_scaled, dtype=np.float32).copy()
        self.pm25_scaled[cut:] = np.nan


def chronological_blocks(origin_indices, block_hours=None):
    """Group sample rows into strictly increasing forecast-origin blocks.

    Blocks follow calendar months of the forecast origin unless an explicit
    ``block_hours`` is given. Returns a list of (ceiling_index, row_indices)
    in strictly increasing ceiling order; every row's origin is at or before
    its own block's ceiling.
    """
    origin_indices = np.asarray(origin_indices, dtype=np.int64)
    if block_hours is None:
        keys = np.array([timestamp_at(o).year * 100 + timestamp_at(o).month
                         for o in origin_indices], dtype=np.int64)
    else:
        keys = origin_indices // int(block_hours)
    blocks = []
    for key in np.unique(keys):
        rows = np.flatnonzero(keys == key)
        blocks.append((int(origin_indices[rows].max()), rows))
    blocks.sort(key=lambda pair: pair[0])
    ceilings = [c for c, _ in blocks]
    if ceilings != sorted(ceilings) or len(set(ceilings)) != len(ceilings):
        raise FinalTestAccessError("blocks are not strictly chronological")
    return blocks


def verify_matrix(matrix, label, wd_slice=None):
    """Reject any non-finite cell, and any all-zero wind one-hot row."""
    matrix = np.asarray(matrix)
    if not np.isfinite(matrix).all():
        raise FinalTestAccessError(
            "%s contains a non-finite value: a position after the reveal "
            "ceiling was read" % label)
    if wd_slice is not None:
        sums = matrix[:, wd_slice].sum(axis=1)
        if not np.allclose(sums, 1.0):
            raise FinalTestAccessError(
                "%s has a wind one-hot row that is not exactly one: a "
                "masked hour was read" % label)
    return True


# ---------------------------------------------------------------------------
# Canonical test sample universe
# ---------------------------------------------------------------------------

def load_test_index(processed):
    """Canonical test sample index in file order. Alignment is by position.

    Reads structural metadata only. The index carries no PM2.5 value; the
    single numeric column it does carry, ``target_row_index``, is a position
    on the timeline, not an observation.
    """
    sample_ids, stations, horizons, targets, origins = [], [], [], [], []
    path = str(processed / "sample_index" / "test.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        forbidden = [c for c in reader.fieldnames
                     if c.strip().lower() in ("pm2.5", "pm25", "target",
                                              "target_value", "value", "y")]
        if forbidden:
            raise FinalTestAccessError(
                "the test index exposes a target value column: %s" % forbidden)
        for row in reader:
            sample_ids.append(row["sample_id"])
            stations.append(row["station"])
            horizons.append(int(row["horizon_hours"]))
            targets.append(int(row["target_row_index"]))
            origins.append(index_of(datetime.strptime(
                row["forecast_origin"], "%Y-%m-%dT%H:%M:%S")))
    out = OrderedDict([
        ("sample_id", np.array(sample_ids)),
        ("station", np.array(stations)),
        ("horizon", np.array(horizons, dtype=np.int64)),
        ("target_index", np.array(targets, dtype=np.int64)),
        ("origin_index", np.array(origins, dtype=np.int64)),
    ])
    if out["target_index"].size != EXPECTED_TEST_SAMPLES:
        raise FinalTestAccessError(
            "test index holds %d samples, the frozen universe declares %d"
            % (out["target_index"].size, EXPECTED_TEST_SAMPLES))
    if not (out["origin_index"] == out["target_index"] - out["horizon"]).all():
        raise FinalTestAccessError("origin is not target minus horizon")
    if (out["origin_index"] >= out["target_index"]).any():
        raise FinalTestAccessError("a forecast origin is not before its target")
    low, high = index_of(TEST_START), index_of(TEST_END)
    if (out["target_index"] < low).any() or (out["target_index"] > high).any():
        raise FinalTestAccessError("a target lies outside the test window")
    if (out["origin_index"] - (CONTEXT_HOURS - 1) < 0).any():
        raise FinalTestAccessError("a sample lacks a complete 48-hour context")
    return out


# ---------------------------------------------------------------------------
# Scoring access - opened only after predictions are frozen
# ---------------------------------------------------------------------------

class FinalTestTargetOracle(object):
    """Final-test actuals, for scoring only, never for feature construction.

    Refuses any timestamp outside the locked test window, and refuses to
    return anything at all unless the caller presents the predictions it has
    already produced. This is the one place in the repository that reads a
    final-test PM2.5 value as evaluation truth.
    """

    def __init__(self, station, csv_path):
        self.station = station
        length = total_hours()
        self._values = np.full(length, np.nan, dtype=np.float64)
        low, high = index_of(TEST_START), index_of(TEST_END)
        with open(str(csv_path), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stamp = datetime(int(row["year"]), int(row["month"]),
                                 int(row["day"]), int(row["hour"]))
                position = index_of(stamp)
                if position < low or position > high:
                    continue        # nothing outside the test window is read
                raw = row[TARGET]
                if raw in (MISSING_TOKEN, ""):
                    continue
                self._values[position] = float(raw)

    def actuals(self, target_indices, predictions):
        """Actuals aligned to already-produced predictions."""
        target_indices = np.asarray(target_indices, dtype=np.int64)
        predictions = np.asarray(predictions)
        if predictions.size != target_indices.size:
            raise FinalTestAccessError(
                "scoring requires one prediction per target; got %d for %d"
                % (predictions.size, target_indices.size))
        low, high = index_of(TEST_START), index_of(TEST_END)
        if (target_indices < low).any() or (target_indices > high).any():
            raise FinalTestAccessError(
                "refusing to score outside the locked test window")
        values = self._values[target_indices]
        if not np.isfinite(values).all():
            raise FinalTestAccessError(
                "the canonical sample universe promised observed targets")
        return values


def assert_confirmatory_only(name):
    """The frozen whitelist. No other model may be evaluated on the test."""
    if name not in CONFIRMATORY_MODELS:
        raise FinalTestAccessError(
            "%s is not in the frozen confirmatory hierarchy %s; no model may "
            "be promoted after test access" % (name, CONFIRMATORY_MODELS))
    return name


def wind_onehot_slice(regime):
    """Column span of the wind one-hot block in a dynamic neural matrix."""
    names = dynamic_channel_names(regime)
    positions = [i for i, n in enumerate(names) if n.startswith("wd_is_")]
    return slice(min(positions), max(positions) + 1)
