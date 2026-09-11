"""Unit tests for on-demand causal window construction."""

import csv
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import partition_of                   # noqa: E402
from src.data.windowing import (                                   # noqa: E402
    CONTEXT_HOURS_PRIMARY, HORIZONS, ForecastWindowLoader, WindowError,
    make_sample_id, parse_sample_id, window_bounds)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"


def first_sample(partition, horizon, station=None):
    with open(str(PROCESSED / "sample_index" / ("%s.csv" % partition)),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon_hours"]) != horizon:
                continue
            if station and row["station"] != station:
                continue
            return row
    return None


class TestSampleIdentifier(unittest.TestCase):

    def test_round_trip(self):
        sample_id = make_sample_id("train", "Dongsi",
                                   datetime(2014, 9, 7, 12), 6)
        self.assertEqual(sample_id, "train|Dongsi|2014090712|h6")
        partition, station, target, horizon = parse_sample_id(sample_id)
        self.assertEqual((partition, station, horizon),
                         ("train", "Dongsi", 6))
        self.assertEqual(target, datetime(2014, 9, 7, 12))

    def test_identifier_is_deterministic(self):
        args = ("validation", "Wanliu", datetime(2015, 12, 31, 23), 24)
        self.assertEqual(make_sample_id(*args), make_sample_id(*args))

    def test_malformed_identifier_rejected(self):
        with self.assertRaises(WindowError):
            parse_sample_id("not-a-sample-id")


class TestWindowGeometry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loader = ForecastWindowLoader(PROCESSED)

    def test_context_is_exactly_48_hourly_timestamps(self):
        for partition in ("train", "validation", "test"):
            for horizon in HORIZONS:
                row = first_sample(partition, horizon)
                window = self.loader.get_window(row["sample_id"])
                stamps = [datetime.fromisoformat(s)
                          for s in window["timestamps"]]
                self.assertEqual(len(stamps), CONTEXT_HOURS_PRIMARY)
                for earlier, later in zip(stamps, stamps[1:]):
                    self.assertEqual(later - earlier, timedelta(hours=1))

    def test_origin_plus_horizon_reaches_the_target(self):
        for partition in ("train", "validation", "test"):
            for horizon in HORIZONS:
                row = first_sample(partition, horizon)
                window = self.loader.get_window(row["sample_id"])
                origin = datetime.fromisoformat(window["forecast_origin"])
                target = datetime.fromisoformat(window["target_timestamp"])
                self.assertEqual(origin + timedelta(hours=horizon), target)
                self.assertLess(origin, target)

    def test_no_input_timestamp_after_the_origin(self):
        for partition in ("train", "validation", "test"):
            for horizon in HORIZONS:
                row = first_sample(partition, horizon)
                window = self.loader.get_window(row["sample_id"])
                origin = datetime.fromisoformat(window["forecast_origin"])
                stamps = [datetime.fromisoformat(s)
                          for s in window["timestamps"]]
                self.assertEqual(max(stamps), origin)

    def test_window_start_is_origin_minus_47_hours(self):
        row = first_sample("train", 12)
        window = self.loader.get_window(row["sample_id"])
        origin = datetime.fromisoformat(window["forecast_origin"])
        start = datetime.fromisoformat(window["context_start"])
        self.assertEqual(start, origin - timedelta(hours=47))

    def test_bounds_helper_matches_the_loader(self):
        row = first_sample("validation", 6)
        target = datetime.fromisoformat(row["target_timestamp"])
        start, origin = window_bounds(target, 6, CONTEXT_HOURS_PRIMARY)
        window = self.loader.get_window(row["sample_id"])
        self.assertEqual(start.isoformat(), window["context_start"])
        self.assertEqual(origin.isoformat(), window["forecast_origin"])


class TestPartitionRules(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loader = ForecastWindowLoader(PROCESSED)

    def test_partition_follows_the_target_timestamp(self):
        for partition in ("train", "validation", "test"):
            row = first_sample(partition, 24)
            window = self.loader.get_window(row["sample_id"])
            target = datetime.fromisoformat(window["target_timestamp"])
            self.assertEqual(partition_of(target), partition)
            self.assertEqual(window["target_partition"], partition)

    def test_boundary_context_may_precede_the_partition(self):
        row = first_sample("test", 24)
        window = self.loader.get_window(row["sample_id"])
        start = datetime.fromisoformat(window["context_start"])
        target = datetime.fromisoformat(window["target_timestamp"])
        self.assertEqual(partition_of(target), "test")
        self.assertEqual(partition_of(start), "validation")


class TestTargetQuarantine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loader = ForecastWindowLoader(PROCESSED)

    def test_training_target_value_is_available(self):
        row = first_sample("train", 6)
        window = self.loader.get_window(row["sample_id"])
        self.assertTrue(window["target_value_available"])
        self.assertIsNotNone(window["target_value_ug_m3"])

    def test_validation_target_value_is_hidden(self):
        row = first_sample("validation", 6)
        window = self.loader.get_window(row["sample_id"])
        self.assertFalse(window["target_value_available"])
        self.assertNotIn("target_value_ug_m3", window)

    def test_test_target_value_is_hidden(self):
        row = first_sample("test", 6)
        window = self.loader.get_window(row["sample_id"])
        self.assertFalse(window["target_value_available"])
        self.assertNotIn("target_value_ug_m3", window)

    def test_sealed_pm25_history_is_absent_without_an_accessor(self):
        row = first_sample("test", 1)
        window = self.loader.get_window(row["sample_id"], regime="R1")
        self.assertIsNone(window["pollutant_history"]["PM2.5"])
        self.assertEqual(window["pollutant_masks"]["PM2.5"].size,
                         CONTEXT_HOURS_PRIMARY)


class TestCommonSampleUniverse(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loader = ForecastWindowLoader(PROCESSED)

    def test_every_regime_sees_the_same_sample(self):
        row = first_sample("train", 6)
        geometry = set()
        for regime in ("R0", "R1", "R2", "R3"):
            window = self.loader.get_window(row["sample_id"], regime=regime)
            geometry.add((window["sample_id"], window["context_start"],
                          window["forecast_origin"],
                          window["target_timestamp"],
                          len(window["timestamps"])))
        self.assertEqual(len(geometry), 1)

    def test_regimes_differ_only_in_channels(self):
        row = first_sample("train", 6)
        r0 = self.loader.get_window(row["sample_id"], regime="R0")
        r1 = self.loader.get_window(row["sample_id"], regime="R1")
        r2 = self.loader.get_window(row["sample_id"], regime="R2")
        r3 = self.loader.get_window(row["sample_id"], regime="R3")
        self.assertEqual(r0["pollutant_channels"], [])
        self.assertEqual(r1["pollutant_channels"], ["PM2.5"])
        self.assertEqual(len(r2["pollutant_channels"]), 6)
        self.assertNotIn("cross_station", r2)
        self.assertIn("cross_station", r3)
        self.assertEqual(len(r3["cross_station"]), 11)

    def test_masks_are_binary_and_gap_ages_are_bounded(self):
        row = first_sample("train", 1)
        window = self.loader.get_window(row["sample_id"], regime="R2")
        for name, mask in window["pollutant_masks"].items():
            self.assertTrue(set(mask.tolist()) <= {0, 1}, name)
        for name, ages in window["pollutant_gap_age"].items():
            self.assertTrue(int(ages.min()) >= 0, name)
            self.assertTrue(int(ages.max()) <= 168, name)


if __name__ == "__main__":
    unittest.main()
