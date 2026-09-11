"""Development scoring access to validation target values.

Protocol Phase 3.

This is the *only* place in the codebase that reads a validation PM2.5 value
as an evaluation truth rather than as causal history, and it is deliberately
separate from every feature path.

Two guarantees:

* it refuses any timestamp outside the validation window, so the locked test
  target cannot be read through it;
* callers must present the predictions they have already produced, which
  enforces the protocol order - predict first, then score.
"""

import csv
from datetime import datetime

import numpy as np

from src.data.preprocessing import (
    MISSING_TOKEN, PARTITION_BOUNDS, TARGET, FULL_START, index_of,
    total_hours)

VALIDATION_START, VALIDATION_END = PARTITION_BOUNDS["validation"]


class ScoringAccessError(RuntimeError):
    """Raised on any attempt to score outside the opened partition."""


class ValidationTargetOracle(object):
    """Validation actuals, for scoring only, never for feature construction."""

    def __init__(self, station, csv_path):
        self.station = station
        length = total_hours()
        self._values = np.full(length, np.nan, dtype=np.float64)
        low = index_of(VALIDATION_START)
        high = index_of(VALIDATION_END)
        with open(str(csv_path), newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stamp = datetime(int(row["year"]), int(row["month"]),
                                 int(row["day"]), int(row["hour"]))
                position = index_of(stamp)
                if position < low or position > high:
                    continue          # test values are never read at all
                raw = row[TARGET]
                if raw in (MISSING_TOKEN, ""):
                    continue
                self._values[position] = float(raw)

    def actuals(self, target_indices, predictions):
        """Actual values aligned to already-produced predictions."""
        target_indices = np.asarray(target_indices, dtype=np.int64)
        if np.asarray(predictions).size != target_indices.size:
            raise ScoringAccessError(
                "scoring requires one prediction per target; got %d for %d"
                % (np.asarray(predictions).size, target_indices.size))
        low = index_of(VALIDATION_START)
        high = index_of(VALIDATION_END)
        if (target_indices < low).any() or (target_indices > high).any():
            raise ScoringAccessError(
                "refusing to score outside the validation window")
        values = self._values[target_indices]
        if not np.isfinite(values).all():
            raise ScoringAccessError(
                "canonical sample universe promised observed targets")
        return values
