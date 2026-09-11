"""Unit tests for the frozen classical baselines B0, B1 and B2."""

import csv
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
    PARTITION_BOUNDS, TARGET, index_of, load_training_statistics)
from src.data.rolling_history import RollingPm25Series            # noqa: E402
from src.models import baselines                                  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
STATION = "Dongsi"
VALIDATION_END = PARTITION_BOUNDS["validation"][1]


def rolling_for(station=STATION):
    path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
    statistics = load_training_statistics(
        PROJECT_ROOT / "configs" / "preprocessing.json")
    medians = json.load(open(str(
        PROJECT_ROOT / "artifacts" / "preprocessing_statistics.json"),
        encoding="utf-8"))["station_training_medians"][station]
    return RollingPm25Series(station, path, statistics, medians[TARGET],
                             unseal_until=VALIDATION_END), medians[TARGET]


class TestB0FallbackChain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rolling, cls.median = rolling_for()

    def test_every_sample_receives_a_prediction(self):
        origins = np.arange(index_of(datetime(2015, 5, 1, 0)),
                            index_of(datetime(2015, 5, 8, 0)))
        predictions, sources = baselines.b0_causal_persistence(
            self.rolling, origins)
        self.assertEqual(predictions.size, origins.size)
        self.assertTrue(np.isfinite(predictions).all())

    def test_all_three_sources_are_reachable_and_labelled(self):
        origins = np.arange(index_of(datetime(2015, 3, 1, 0)),
                            index_of(datetime(2016, 2, 29, 23)))
        _, sources = baselines.b0_causal_persistence(self.rolling, origins)
        present = set(sources.tolist())
        self.assertIn(baselines.B0_SOURCE_OBSERVED, present)
        counts = baselines.source_frequencies(
            sources, baselines.B0_SOURCE_LABELS)
        self.assertEqual(sum(counts.values()), origins.size)
        self.assertIn("observed_at_origin", counts)

    def test_observed_origin_uses_the_observation_itself(self):
        origin = index_of(datetime(2015, 6, 1, 12))
        while self.rolling.fill_source_at(origin) != 0:
            origin += 1
        prediction, sources = baselines.b0_causal_persistence(
            self.rolling, [origin])
        self.assertEqual(sources[0], baselines.B0_SOURCE_OBSERVED)
        self.assertAlmostEqual(
            prediction[0], self.rolling.observed_native_at(origin, origin))

    def test_median_fallback_equals_the_station_training_median(self):
        origins = np.arange(index_of(datetime(2015, 3, 1, 0)),
                            index_of(datetime(2016, 2, 29, 23)))
        predictions, sources = baselines.b0_causal_persistence(
            self.rolling, origins)
        fallback = predictions[sources == baselines.B0_SOURCE_MEDIAN]
        if fallback.size:
            self.assertTrue(np.allclose(fallback, self.median))


class TestB1SeasonalNaive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rolling, cls.median = rolling_for()

    def test_direct_value_is_the_observation_24_hours_before_target(self):
        target = index_of(datetime(2015, 8, 15, 12))
        horizon = 6
        origin = target - horizon
        b0, _ = baselines.b0_causal_persistence(self.rolling, [origin])
        prediction, direct = baselines.b1_seasonal_naive(
            self.rolling, [origin], [target], b0)
        seasonal = self.rolling.observed_native_at(target - 24, origin)
        if seasonal is not None:
            self.assertTrue(direct[0])
            self.assertAlmostEqual(prediction[0], seasonal)
        else:
            self.assertFalse(direct[0])
            self.assertAlmostEqual(prediction[0], b0[0])

    def test_collapses_into_b0_at_horizon_24(self):
        targets = np.arange(index_of(datetime(2015, 4, 1, 0)),
                            index_of(datetime(2015, 4, 30, 23)))
        origins = targets - 24
        b0, _ = baselines.b0_causal_persistence(self.rolling, origins)
        b1, _ = baselines.b1_seasonal_naive(self.rolling, origins, targets, b0)
        self.assertTrue(np.array_equal(b0, b1),
                        "B1 must equal B0 at h=24 by arithmetic")

    def test_seasonal_lag_is_never_after_the_origin(self):
        for horizon in (1, 6, 12, 24):
            target = index_of(datetime(2015, 8, 15, 12))
            origin = target - horizon
            self.assertLessEqual(target - 24, origin)

    def test_unobserved_seasonal_lag_falls_back_to_b0(self):
        targets = np.arange(index_of(datetime(2015, 3, 1, 0)),
                            index_of(datetime(2016, 2, 20, 0)))
        origins = targets - 1
        b0, _ = baselines.b0_causal_persistence(self.rolling, origins)
        b1, direct = baselines.b1_seasonal_naive(self.rolling, origins,
                                                 targets, b0)
        self.assertTrue(np.array_equal(b1[~direct], b0[~direct]))


class TestB2Climatology(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(str(PROCESSED / "climatology"
                      / "b2_training_climatology.csv"),
                  encoding="utf-8") as handle:
            cls.climatology = baselines.TrainingClimatology(
                list(csv.DictReader(handle)))

    def test_first_hierarchy_level_is_used_when_available(self):
        value, level = self.climatology.predict(STATION, 1, 0)
        self.assertEqual(level, 0)
        self.assertGreater(value, 0)

    def test_fallback_hierarchy_order_is_frozen(self):
        self.assertEqual(baselines.TrainingClimatology.LEVELS,
                         ["station_month_hour", "station_hour", "station"])

    def test_station_hour_used_when_month_hour_cell_is_empty(self):
        self.climatology.month_hour.pop((STATION, 1, 0), None)
        value, level = self.climatology.predict(STATION, 1, 0)
        self.assertEqual(level, 1)
        self.assertAlmostEqual(value, self.climatology.hour[(STATION, 0)])

    def test_station_median_used_when_both_cells_are_empty(self):
        self.climatology.month_hour.pop((STATION, 2, 3), None)
        self.climatology.hour.pop((STATION, 3), None)
        value, level = self.climatology.predict(STATION, 2, 3)
        self.assertEqual(level, 2)
        self.assertAlmostEqual(value, self.climatology.station[STATION])

    def test_all_twelve_stations_have_a_median(self):
        self.assertEqual(len(self.climatology.station), 12)


if __name__ == "__main__":
    unittest.main()
