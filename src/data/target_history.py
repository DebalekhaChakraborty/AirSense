"""Guarded causal access to PM2.5 target history for AirSense V2.

Protocol Phase 2.

Later forecasting legitimately uses PM2.5 observations that had already
occurred before a forecast origin - including, during validation and the final
test, observations inside those partitions. That is rolling-origin
forecasting, not leakage.

What would be leakage is loading a whole validation or test target vector into
an ordinary feature array. This module is the alternative: a **guarded
interface** that can only ever hand back observations at or before a stated
origin, and only from partitions whose values have been explicitly unsealed.

Two independent guards:

``origin`` guard
    Every accessor takes the forecast origin and returns nothing later than
    it. A direct request for a timestamp after the origin raises.

``unseal`` guard
    ``value_access_until`` is the last timestamp whose *numeric* value this
    instance may return. Values after it are never even read off disk, so
    they cannot leak through a bug elsewhere. In Phase 2 the ceiling is the
    end of training; Phase 3 may raise it to the end of validation when the
    protocol opens validation; only Phase 9 may raise it into the test
    period.

Structural presence - whether an hour was observed - is not a target value
and is available for every partition at any time.
"""

import csv
from datetime import datetime, timedelta

import numpy as np

from src.data.preprocessing import (
    FULL_END, FULL_START, MISSING_TOKEN, PARTITION_BOUNDS, TARGET,
    index_of, timestamp_at, total_hours)

TRAIN_END = PARTITION_BOUNDS["train"][1]
VALIDATION_END = PARTITION_BOUNDS["validation"][1]
TEST_END = PARTITION_BOUNDS["test"][1]


class TargetAccessError(RuntimeError):
    """Raised on any attempt to read a target value that is not permitted."""


class CausalTargetHistory(object):
    """Causal, partition-aware accessor for one station's PM2.5 history."""

    def __init__(self, station, csv_path, value_access_until=TRAIN_END):
        if value_access_until > TEST_END:
            raise TargetAccessError("value_access_until beyond the dataset")
        self.station = station
        self.value_access_until = value_access_until
        self._limit_index = index_of(value_access_until)
        length = total_hours()
        self._values = np.full(length, np.nan, dtype=np.float64)
        self._observed = np.zeros(length, dtype=bool)
        self._load(csv_path)

    # -- construction -------------------------------------------------------

    def _load(self, csv_path):
        with open(str(csv_path), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stamp = datetime(int(row["year"]), int(row["month"]),
                                 int(row["day"]), int(row["hour"]))
                if stamp < FULL_START or stamp > FULL_END:
                    continue
                position = index_of(stamp)
                raw = row[TARGET]
                if raw in (MISSING_TOKEN, ""):
                    continue
                self._observed[position] = True
                if position <= self._limit_index:
                    self._values[position] = float(raw)
                # Beyond the unseal ceiling the value is deliberately dropped
                # here, not merely hidden by an accessor.

    # -- guards -------------------------------------------------------------

    def _check_origin(self, origin):
        if origin < FULL_START or origin > FULL_END:
            raise TargetAccessError("origin outside the dataset timeline")

    def _check_value_access(self, latest_stamp):
        if latest_stamp > self.value_access_until:
            raise TargetAccessError(
                "target values after %s are sealed for this instance "
                "(requested %s, station %s)"
                % (self.value_access_until.isoformat(),
                   latest_stamp.isoformat(), self.station))

    # -- structural accessors (always permitted) ----------------------------

    def observed_mask(self, origin, hours):
        """Observed/missing flags for the `hours` ending at `origin`."""
        self._check_origin(origin)
        end = index_of(origin)
        start = end - hours + 1
        if start < 0:
            raise TargetAccessError("window starts before the dataset")
        return self._observed[start:end + 1].copy()

    def is_observed(self, stamp):
        self._check_origin(stamp)
        return bool(self._observed[index_of(stamp)])

    # -- value accessors (guarded) ------------------------------------------

    def values(self, origin, hours):
        """Target values for the `hours` ending at `origin`, inclusive.

        Never returns anything after `origin`, and refuses entirely if the
        origin lies beyond this instance's unseal ceiling.
        """
        self._check_origin(origin)
        self._check_value_access(origin)
        end = index_of(origin)
        start = end - hours + 1
        if start < 0:
            raise TargetAccessError("window starts before the dataset")
        return self._values[start:end + 1].copy()

    def value_at(self, stamp, origin):
        """One target value, refused if it is after the origin or sealed."""
        self._check_origin(stamp)
        if stamp > origin:
            raise TargetAccessError(
                "refusing future target: %s is after origin %s"
                % (stamp.isoformat(), origin.isoformat()))
        self._check_value_access(stamp)
        value = self._values[index_of(stamp)]
        return None if not np.isfinite(value) else float(value)

    def latest_value_at_or_before(self, origin, max_age_hours):
        """Most recent observation within `max_age_hours` of the origin.

        The primitive behind causal persistence. Returns
        (value, age_hours) or (None, None) when nothing usable exists.
        """
        self._check_origin(origin)
        self._check_value_access(origin)
        end = index_of(origin)
        lowest = max(0, end - max_age_hours)
        for position in range(end, lowest - 1, -1):
            if self._observed[position] and np.isfinite(self._values[position]):
                return float(self._values[position]), end - position
        return None, None

    # -- introspection ------------------------------------------------------

    def sealed_partitions(self):
        return [name for name, (_, end) in PARTITION_BOUNDS.items()
                if end > self.value_access_until]

    def describe(self):
        return {
            "station": self.station,
            "value_access_until": self.value_access_until.isoformat(),
            "sealed_partitions": self.sealed_partitions(),
            "observed_hours_structural": int(self._observed.sum()),
            "values_materialised": int(np.isfinite(self._values).sum()),
        }
