"""Unit tests for guarded rolling-origin validation history."""

import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (                              # noqa: E402
    CAUSAL_FFILL_MAX_HOURS, PARTITION_BOUNDS, TARGET, causal_forward_fill,
    index_of, load_training_statistics, robust_scale, timestamp_at)
from src.data.rolling_history import RollingPm25Series            # noqa: E402
from src.data.target_history import (                             # noqa: E402
    CausalTargetHistory, TargetAccessError)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
STATION = "Dongsi"
VALIDATION_END = PARTITION_BOUNDS["validation"][1]
TEST_START = PARTITION_BOUNDS["test"][0]
CONTEXT = 48


def build():
    path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % STATION))
    statistics = load_training_statistics(
        PROJECT_ROOT / "configs" / "preprocessing.json")
    medians = json.load(open(str(
        PROJECT_ROOT / "artifacts" / "preprocessing_statistics.json"),
        encoding="utf-8"))["station_training_medians"][STATION]
    return (RollingPm25Series(STATION, path, statistics, medians[TARGET],
                              unseal_until=VALIDATION_END),
            path, statistics, medians[TARGET])


class TestUnsealCeiling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rolling, cls.path, cls.statistics, cls.median = build()

    def test_validation_origins_are_permitted(self):
        origin = index_of(datetime(2015, 6, 1, 12))
        window = self.rolling.scaled_window(origin, CONTEXT)
        self.assertEqual(window.size, CONTEXT)
        self.assertTrue(np.isfinite(window).all())

    def test_test_period_origins_are_refused(self):
        with self.assertRaises(TargetAccessError):
            self.rolling.scaled_window(index_of(TEST_START), CONTEXT)

    def test_test_values_are_never_materialised(self):
        series = self.rolling._scaled
        self.assertTrue(np.isnan(series[index_of(TEST_START):]).all())

    def test_future_index_access_is_refused(self):
        origin = index_of(datetime(2015, 6, 1, 12))
        with self.assertRaises(TargetAccessError):
            self.rolling.native_at(origin + 1, origin)


class TestCausality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rolling, cls.path, cls.statistics, cls.median = build()

    def test_window_ends_exactly_at_the_origin(self):
        origin = index_of(datetime(2015, 9, 9, 9))
        window = self.rolling.scaled_window(origin, CONTEXT)
        whole = self.rolling.scaled_series_upto(origin)
        self.assertEqual(window.size, CONTEXT)
        self.assertTrue(np.array_equal(window, whole[-CONTEXT:],
                                       equal_nan=True))
        self.assertEqual(whole.size, origin + 1)

    def test_guarded_lookup_equals_sequential_per_origin_walk(self):
        """The equivalence the protocol requires to be proven, not assumed.

        A strictly sequential walk that re-derives the causal fill from
        CausalTargetHistory at each origin must agree exactly with the
        precomputed guarded index lookup.
        """
        history = CausalTargetHistory(STATION, self.path,
                                      value_access_until=VALIDATION_END)
        origins = [index_of(datetime(2015, 3, 1, 0)),
                   index_of(datetime(2015, 7, 15, 6)),
                   index_of(datetime(2015, 12, 31, 23)),
                   index_of(datetime(2016, 2, 29, 23))]
        for origin in origins:
            stamp = timestamp_at(origin)
            raw = history.values(stamp, origin + 1)
            observed = history.observed_mask(stamp, origin + 1)
            filled, _ = causal_forward_fill(raw, observed,
                                            CAUSAL_FFILL_MAX_HOURS,
                                            self.median)
            scaled, _ = robust_scale(filled, self.statistics, TARGET)
            sequential = scaled[-CONTEXT:].astype(np.float32)
            guarded = self.rolling.scaled_window(origin, CONTEXT)
            self.assertTrue(np.allclose(sequential, guarded, atol=0, rtol=0),
                            "divergence at origin index %d" % origin)

    def test_training_region_matches_the_phase_2_canonical_array(self):
        canonical = np.load(str(
            PROJECT_ROOT / "data" / "processed" / "phase2" / "station_series"
            / STATION / "pm25_scaled_train_only.npy"))
        train_end = index_of(PARTITION_BOUNDS["train"][1])
        self.assertTrue(np.allclose(canonical[:train_end + 1],
                                    self.rolling._scaled[:train_end + 1],
                                    atol=0, rtol=0))


if __name__ == "__main__":
    unittest.main()
