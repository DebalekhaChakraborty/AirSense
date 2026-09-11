"""Unit tests for the AirSense V2 causal preprocessing primitives."""

import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocessing as pre                        # noqa: E402


class TestCausalForwardFill(unittest.TestCase):

    def test_fills_within_six_hour_ceiling(self):
        values = np.array([10.0] + [np.nan] * 6, dtype=float)
        observed = np.array([True] + [False] * 6)
        filled, source = pre.causal_forward_fill(values, observed, 6, -99.0)
        self.assertTrue(np.all(filled[:7] == 10.0))
        self.assertEqual(source[0], 0)
        self.assertTrue(np.all(source[1:7] == 1))

    def test_falls_back_beyond_the_ceiling(self):
        values = np.array([10.0] + [np.nan] * 8, dtype=float)
        observed = np.array([True] + [False] * 8)
        filled, source = pre.causal_forward_fill(values, observed, 6, -99.0)
        self.assertTrue(np.all(filled[1:7] == 10.0))
        self.assertTrue(np.all(filled[7:] == -99.0))
        self.assertTrue(np.all(source[7:] == 2))

    def test_never_fills_backwards(self):
        values = np.array([np.nan, np.nan, 5.0], dtype=float)
        observed = np.array([False, False, True])
        filled, source = pre.causal_forward_fill(values, observed, 6, -99.0)
        self.assertEqual(filled[0], -99.0)
        self.assertEqual(filled[1], -99.0)
        self.assertEqual(filled[2], 5.0)
        self.assertTrue(np.all(source[:2] == 2))

    def test_no_interpolation_between_observations(self):
        values = np.array([0.0, np.nan, np.nan, 30.0], dtype=float)
        observed = np.array([True, False, False, True])
        filled, _ = pre.causal_forward_fill(values, observed, 6, -99.0)
        self.assertEqual(filled[1], 0.0)
        self.assertEqual(filled[2], 0.0)

    def test_fallback_used_before_any_observation(self):
        values = np.array([np.nan, np.nan], dtype=float)
        observed = np.array([False, False])
        filled, source = pre.causal_forward_fill(values, observed, 6, 7.5)
        self.assertTrue(np.all(filled == 7.5))
        self.assertTrue(np.all(source == 2))


class TestMaskSemantics(unittest.TestCase):

    def test_mask_reflects_raw_observation_not_the_fill(self):
        values = np.array([10.0, np.nan, np.nan], dtype=float)
        observed = np.array([True, False, False])
        filled, source = pre.causal_forward_fill(values, observed, 6, -1.0)
        self.assertTrue(np.all(filled == 10.0))
        self.assertEqual(list(observed.astype(int)), [1, 0, 0])
        self.assertEqual(list(source), [0, 1, 1])


class TestGapAge(unittest.TestCase):

    def test_zero_when_observed(self):
        ages = pre.gap_age_hours(np.array([True, True]), 168)
        self.assertEqual(list(ages), [0, 0])

    def test_counts_hours_since_last_observation(self):
        ages = pre.gap_age_hours(np.array([True, False, False, True]), 168)
        self.assertEqual(list(ages), [0, 1, 2, 0])

    def test_cap_is_respected_and_used_before_first_observation(self):
        observed = np.zeros(200, dtype=bool)
        observed[0] = True
        ages = pre.gap_age_hours(observed, 168)
        self.assertEqual(ages.max(), 168)
        never = pre.gap_age_hours(np.zeros(5, dtype=bool), 168)
        self.assertTrue(np.all(never == 168))

    def test_never_negative(self):
        observed = np.array([False, True, False, False, True])
        ages = pre.gap_age_hours(observed, 168)
        self.assertTrue(np.all(ages >= 0))


class TestScaling(unittest.TestCase):

    def setUp(self):
        self.statistics = pre.load_training_statistics(
            PROJECT_ROOT / "configs" / "preprocessing.json")

    def test_robust_scaling_uses_training_median_and_iqr(self):
        stat = self.statistics["PM2.5"]
        scaled, method = pre.robust_scale(
            np.array([stat["median"]]), self.statistics, "PM2.5")
        self.assertEqual(method, "median_iqr")
        self.assertAlmostEqual(float(scaled[0]), 0.0, places=12)

    def test_rain_zero_iqr_falls_back_to_std(self):
        self.assertEqual(self.statistics["RAIN"]["iqr"], 0)
        scaled, method = pre.robust_scale(
            np.array([self.statistics["RAIN"]["median"]
                      + self.statistics["RAIN"]["std"]]),
            self.statistics, "RAIN")
        self.assertEqual(method, "median_std_zero_iqr_fallback")
        self.assertAlmostEqual(float(scaled[0]), 1.0, places=9)


class TestWindDirection(unittest.TestCase):

    def test_missing_code_is_outside_the_training_vocabulary(self):
        self.assertEqual(len(pre.WD_VOCABULARY), 16)
        self.assertEqual(pre.WD_MISSING_CODE, 16)
        self.assertNotIn(pre.WD_MISSING_CODE,
                         range(len(pre.WD_VOCABULARY)))

    def test_categorical_fill_then_explicit_missing(self):
        codes = np.array([3, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.int16)
        observed = np.array([True] + [False] * 8)
        filled, source = pre.causal_categorical_fill(
            codes, observed, 6, pre.WD_MISSING_CODE)
        self.assertTrue(np.all(filled[1:7] == 3))
        self.assertTrue(np.all(filled[7:] == pre.WD_MISSING_CODE))
        self.assertTrue(np.all(source[7:] == 2))

    def test_no_numeric_order_imposed_on_compass(self):
        self.assertEqual(pre.WD_VOCABULARY, sorted(pre.WD_VOCABULARY))


class TestPartitioning(unittest.TestCase):

    def test_partition_is_decided_by_timestamp(self):
        self.assertEqual(pre.partition_of(datetime(2013, 3, 1, 0)), "train")
        self.assertEqual(pre.partition_of(datetime(2015, 2, 28, 23)), "train")
        self.assertEqual(pre.partition_of(datetime(2015, 3, 1, 0)),
                         "validation")
        self.assertEqual(pre.partition_of(datetime(2016, 2, 29, 23)),
                         "validation")
        self.assertEqual(pre.partition_of(datetime(2016, 3, 1, 0)), "test")
        self.assertEqual(pre.partition_of(datetime(2017, 2, 28, 23)), "test")

    def test_timeline_is_35064_hours(self):
        self.assertEqual(pre.total_hours(), 35064)


if __name__ == "__main__":
    unittest.main()
