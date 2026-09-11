"""Rolling-origin PM2.5 history for AirSense V2 development validation.

Protocol Phase 3.

Phase 2 deliberately never materialised validation or test PM2.5 values. Phase
3 opens **validation only**, for two legitimate purposes: rolling historical
context available strictly before a forecast origin, and development scoring
after a prediction has been produced.

This module provides the causally-filled, training-scaled PM2.5 series a model
may read, behind a guard:

* the series is built by **forward fill only**, so the value at index *i*
  depends on observations at indices <= *i* and on nothing later;
* values are never built beyond ``unseal_until`` - the locked test period is
  not read off disk at all;
* every window request asserts that the window ends exactly at the requested
  forecast origin, so no feature can see past it.

The equivalence between this guarded index lookup and a strictly sequential
per-origin walk through ``CausalTargetHistory`` is asserted in
``tests/test_rolling_history.py``, as the protocol requires.
"""

import numpy as np

from src.data.preprocessing import (
    CAUSAL_FFILL_MAX_HOURS, PARTITION_BOUNDS, TARGET, causal_forward_fill,
    index_of, robust_scale, total_hours)
from src.data.target_history import CausalTargetHistory, TargetAccessError

VALIDATION_END = PARTITION_BOUNDS["validation"][1]


class RollingPm25Series(object):
    """Causally filled, training-scaled PM2.5 history for one station."""

    def __init__(self, station, csv_path, statistics, station_median,
                 unseal_until=VALIDATION_END):
        self.station = station
        self.unseal_until = unseal_until
        self._unseal_index = index_of(unseal_until)
        history = CausalTargetHistory(station, csv_path,
                                      value_access_until=unseal_until)
        length = total_hours()
        raw = np.full(length, np.nan, dtype=np.float64)
        observed = np.zeros(length, dtype=bool)
        # Read through the guarded accessor, never off a bulk array.
        block = history.values(unseal_until, self._unseal_index + 1)
        mask = history.observed_mask(unseal_until, self._unseal_index + 1)
        raw[:self._unseal_index + 1] = block
        observed[:self._unseal_index + 1] = mask
        filled, source = causal_forward_fill(
            raw[:self._unseal_index + 1], observed[:self._unseal_index + 1],
            CAUSAL_FFILL_MAX_HOURS, station_median)
        scaled, _ = robust_scale(filled, statistics, TARGET)
        self._scaled = np.full(length, np.nan, dtype=np.float32)
        self._scaled[:self._unseal_index + 1] = scaled.astype(np.float32)
        self._observed = observed
        self._fill_source = np.full(length, -1, dtype=np.int8)
        self._fill_source[:self._unseal_index + 1] = source
        self._raw_filled = np.full(length, np.nan, dtype=np.float64)
        self._raw_filled[:self._unseal_index + 1] = filled

    # -- guarded accessors --------------------------------------------------

    def _check(self, origin_index, length):
        if origin_index > self._unseal_index:
            raise TargetAccessError(
                "origin index %d is beyond the unseal ceiling %s"
                % (origin_index, self.unseal_until.isoformat()))
        if origin_index - length + 1 < 0:
            raise TargetAccessError("window starts before the dataset")

    def scaled_window(self, origin_index, length):
        """Scaled history for the `length` hours ending at the origin."""
        self._check(origin_index, length)
        return self._scaled[origin_index - length + 1:origin_index + 1]

    def observed_window(self, origin_index, length):
        self._check(origin_index, length)
        return self._observed[origin_index - length + 1:origin_index + 1]

    def scaled_series_upto(self, origin_index):
        """Whole causal series truncated at the origin. Never beyond it."""
        self._check(origin_index, 1)
        return self._scaled[:origin_index + 1]

    def fill_source_at(self, origin_index):
        """0 observed, 1 causal carry, 2 station training-median fallback."""
        self._check(origin_index, 1)
        return int(self._fill_source[origin_index])

    def observed_native_at(self, position, origin_index):
        """Native value only if the hour was *originally observed*."""
        if position > origin_index:
            raise TargetAccessError(
                "refusing index %d after origin %d" % (position, origin_index))
        self._check(origin_index, 1)
        if not self._observed[position]:
            return None
        value = self._raw_filled[position]
        return None if not np.isfinite(value) else float(value)

    def native_at(self, position, origin_index):
        """Causally filled native value, refused if later than the origin."""
        if position > origin_index:
            raise TargetAccessError(
                "refusing index %d after origin %d" % (position, origin_index))
        self._check(origin_index, 1)
        value = self._raw_filled[position]
        return None if not np.isfinite(value) else float(value)
