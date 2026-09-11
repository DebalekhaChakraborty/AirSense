"""Unit tests for the frozen B3 causal feature contract."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (                              # noqa: E402
    CALENDAR_CHANNEL_NAMES, STATIONS, WD_VOCABULARY)
from src.models import b3_features as bf                          # noqa: E402
from src.models import b3_grid                                    # noqa: E402


class TestLagContract(unittest.TestCase):

    def test_exact_frozen_lag_positions(self):
        self.assertEqual(bf.LAGS, [0, 1, 2, 3, 6, 12, 24, 47])

    def test_lag_47_is_inside_the_48_hour_context(self):
        self.assertEqual(max(bf.LAGS), bf.CONTEXT_HOURS - 1)

    def test_lag_48_is_absent(self):
        self.assertNotIn(48, bf.LAGS)
        for regime in ("R0", "R1", "R2"):
            self.assertFalse(any(name.endswith("_lag48")
                                 for name in bf.feature_names(regime)))

    def test_summary_windows_are_frozen_and_fit_the_context(self):
        self.assertEqual(bf.SUMMARY_WINDOWS, [3, 6, 12, 24, 48])
        self.assertLessEqual(max(bf.SUMMARY_WINDOWS), bf.CONTEXT_HOURS)


class TestTrailingStatistics(unittest.TestCase):

    def setUp(self):
        self.series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_trailing_mean_ends_at_the_index(self):
        got = bf.trailing_mean(self.series, 3)
        self.assertTrue(np.isnan(got[0]) and np.isnan(got[1]))
        self.assertAlmostEqual(got[2], 2.0)
        self.assertAlmostEqual(got[5], 5.0)

    def test_trailing_statistics_never_see_the_future(self):
        extended = np.concatenate([self.series, [1000.0]])
        base = bf.trailing_mean(self.series, 3)
        with_future = bf.trailing_mean(extended, 3)
        self.assertTrue(np.allclose(base[2:], with_future[2:len(base)]))

    def test_trailing_std_is_population_ddof_zero(self):
        got = bf.trailing_std(self.series, 3)
        self.assertAlmostEqual(got[2], float(np.std([1.0, 2.0, 3.0])), 9)

    def test_trailing_min_and_max(self):
        minimum, maximum = bf.trailing_min_max(
            np.array([5.0, 1.0, 9.0, 2.0]), 3)
        self.assertAlmostEqual(minimum[2], 1.0)
        self.assertAlmostEqual(maximum[2], 9.0)
        self.assertAlmostEqual(minimum[3], 1.0)
        self.assertAlmostEqual(maximum[3], 9.0)

    def test_mask_fraction_is_a_trailing_mean_of_the_mask(self):
        mask = np.array([1.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(bf.trailing_mean(mask, 4)[3], 0.75)


class TestVocabularies(unittest.TestCase):

    def test_wind_one_hot_covers_sixteen_categories_plus_missing(self):
        names = bf.feature_names("R0")
        wind = [n for n in names if n.startswith("wd_is_")]
        self.assertEqual(len(wind), 17)
        self.assertIn("wd_is_MISSING", wind)
        for category in WD_VOCABULARY:
            self.assertIn("wd_is_%s" % category, wind)

    def test_station_one_hot_covers_the_frozen_twelve(self):
        names = bf.feature_names("R2")
        stations = [n for n in names if n.startswith("station_is_")]
        self.assertEqual(len(stations), 12)
        for station in STATIONS:
            self.assertIn("station_is_%s" % station, stations)

    def test_no_target_encoding_anywhere(self):
        for regime in ("R0", "R1", "R2"):
            for name in bf.feature_names(regime):
                self.assertNotIn("target_mean", name)
                self.assertNotIn("target_median", name)


class TestCalendarAlignment(unittest.TestCase):

    def test_origin_and_target_calendar_channels_both_present(self):
        names = bf.feature_names("R0")
        for channel in CALENDAR_CHANNEL_NAMES:
            self.assertIn("origin_%s" % channel, names)
            self.assertIn("target_%s" % channel, names)

    def test_day_of_week_and_weekend_are_absent(self):
        for regime in ("R0", "R1", "R2"):
            joined = " ".join(bf.feature_names(regime))
            self.assertNotIn("day_of_week", joined)
            self.assertNotIn("weekend", joined)


class TestRegimeSchema(unittest.TestCase):

    def test_regime_variable_sets_are_nested(self):
        self.assertNotIn("PM2.5", bf.REGIME_NUMERIC["R0"])
        self.assertIn("PM2.5", bf.REGIME_NUMERIC["R1"])
        for variable in bf.REGIME_NUMERIC["R0"]:
            self.assertIn(variable, bf.REGIME_NUMERIC["R1"])
        for variable in bf.REGIME_NUMERIC["R1"]:
            self.assertIn(variable, bf.REGIME_NUMERIC["R2"])
        self.assertIn("CO", bf.REGIME_NUMERIC["R2"])

    def test_feature_counts_are_stable_and_unique(self):
        counts = {r: len(bf.feature_names(r)) for r in ("R0", "R1", "R2")}
        self.assertEqual(counts, {"R0": 207, "R1": 240, "R2": 405})
        for regime in counts:
            names = bf.feature_names(regime)
            self.assertEqual(len(names), len(set(names)))

    def test_gap_age_only_for_pollutants_in_the_regime(self):
        self.assertEqual(bf.REGIME_POLLUTANTS["R0"], [])
        self.assertEqual(bf.REGIME_POLLUTANTS["R1"], ["PM2.5"])
        self.assertEqual(len(bf.REGIME_POLLUTANTS["R2"]), 6)


class TestGridAndSelection(unittest.TestCase):

    def test_grid_has_exactly_eight_candidates_in_frozen_order(self):
        grid = b3_grid.candidates(n_jobs=16)
        self.assertEqual(len(grid), 8)
        self.assertEqual([c["candidate_id"] for c in grid],
                         ["B3_C%02d" % i for i in range(1, 9)])
        self.assertEqual((grid[0]["learning_rate"], grid[0]["num_leaves"],
                          grid[0]["min_child_samples"]), (0.03, 31, 50))
        self.assertEqual((grid[7]["learning_rate"], grid[7]["num_leaves"],
                          grid[7]["min_child_samples"]), (0.07, 63, 200))
        self.assertEqual(grid[0]["parameters"]["min_data_in_leaf"], 50)

    def test_fixed_parameters_are_frozen(self):
        grid = b3_grid.candidates(n_jobs=16)
        for candidate in grid:
            parameters = candidate["parameters"]
            self.assertEqual(candidate["num_boost_round"], 400)
            self.assertEqual(parameters["objective"], "regression_l1")
            self.assertEqual(parameters["bagging_fraction"], 1.0)
            self.assertEqual(parameters["feature_fraction"], 1.0)
            self.assertEqual(parameters["seed"], 42)
            self.assertTrue(parameters["deterministic"])
            self.assertTrue(parameters["force_col_wise"])

    def test_selection_prefers_lower_macro_station_mae(self):
        rows = [
            {"candidate_id": "B3_C01", "num_leaves": 31,
             "min_child_samples": 50, "macro_station_mae": 10.0,
             "macro_station_rmse": 20.0},
            {"candidate_id": "B3_C02", "num_leaves": 63,
             "min_child_samples": 50, "macro_station_mae": 9.5,
             "macro_station_rmse": 21.0},
        ]
        self.assertEqual(b3_grid.select(rows)["candidate_id"], "B3_C02")

    def test_tie_break_order_rmse_then_leaves_then_mcs_then_id(self):
        base = {"macro_station_mae": 10.0, "macro_station_rmse": 20.0}
        rows = [
            dict(base, candidate_id="B3_C05", num_leaves=63,
                 min_child_samples=50),
            dict(base, candidate_id="B3_C03", num_leaves=31,
                 min_child_samples=50),
        ]
        self.assertEqual(b3_grid.select(rows)["candidate_id"], "B3_C03")
        rows = [
            dict(base, candidate_id="B3_C05", num_leaves=31,
                 min_child_samples=50),
            dict(base, candidate_id="B3_C03", num_leaves=31,
                 min_child_samples=200),
        ]
        self.assertEqual(b3_grid.select(rows)["candidate_id"], "B3_C03")
        rows = [
            dict(base, candidate_id="B3_C07", num_leaves=31,
                 min_child_samples=200),
            dict(base, candidate_id="B3_C02", num_leaves=31,
                 min_child_samples=200),
        ]
        self.assertEqual(b3_grid.select(rows)["candidate_id"], "B3_C02")
        rows = [
            dict(candidate_id="B3_C01", num_leaves=31, min_child_samples=50,
                 macro_station_mae=10.0, macro_station_rmse=21.0),
            dict(candidate_id="B3_C02", num_leaves=63, min_child_samples=50,
                 macro_station_mae=10.0, macro_station_rmse=20.0),
        ]
        self.assertEqual(b3_grid.select(rows)["candidate_id"], "B3_C02")


if __name__ == "__main__":
    unittest.main()
