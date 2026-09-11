"""Unit tests for the guarded causal target-history interface."""

import sys
import unittest
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PARTITION_BOUNDS                # noqa: E402
from src.data.target_history import (                              # noqa: E402
    CausalTargetHistory, TargetAccessError)

RAW_DIR = (PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228")
STATION = "Dongsi"
TRAIN_END = PARTITION_BOUNDS["train"][1]


def build(value_access_until=TRAIN_END):
    path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % STATION))
    return CausalTargetHistory(STATION, path,
                               value_access_until=value_access_until)


class TestFutureAccessRefused(unittest.TestCase):

    def setUp(self):
        self.history = build()

    def test_value_after_origin_is_refused(self):
        origin = datetime(2014, 6, 10, 6)
        future = datetime(2014, 6, 10, 12)
        with self.assertRaises(TargetAccessError):
            self.history.value_at(future, origin)

    def test_value_at_origin_is_allowed(self):
        origin = datetime(2014, 6, 10, 6)
        self.history.value_at(origin, origin)

    def test_window_never_extends_past_the_origin(self):
        origin = datetime(2014, 6, 10, 6)
        values = self.history.values(origin, 48)
        self.assertEqual(values.size, 48)


class TestSealedPartitions(unittest.TestCase):

    def setUp(self):
        self.history = build()

    def test_validation_values_are_sealed_in_phase_2(self):
        with self.assertRaises(TargetAccessError):
            self.history.values(datetime(2015, 6, 1, 0), 48)

    def test_test_values_are_sealed_in_phase_2(self):
        with self.assertRaises(TargetAccessError):
            self.history.values(datetime(2016, 6, 1, 0), 48)

    def test_sealed_values_are_not_even_loaded(self):
        described = self.history.describe()
        self.assertGreater(described["observed_hours_structural"],
                           described["values_materialised"])
        self.assertEqual(described["sealed_partitions"],
                         ["validation", "test"])

    def test_structural_presence_is_always_available(self):
        self.assertIn(self.history.is_observed(datetime(2016, 6, 1, 0)),
                      (True, False))
        mask = self.history.observed_mask(datetime(2016, 6, 1, 0), 48)
        self.assertEqual(mask.size, 48)

    def test_unsealing_beyond_the_dataset_is_refused(self):
        with self.assertRaises(TargetAccessError):
            build(datetime(2018, 1, 1, 0))


class TestCausalPersistencePrimitive(unittest.TestCase):

    def setUp(self):
        self.history = build()

    def test_latest_observation_is_at_or_before_the_origin(self):
        origin = datetime(2014, 6, 10, 6)
        value, age = self.history.latest_value_at_or_before(origin, 6)
        self.assertIsNotNone(value)
        self.assertGreaterEqual(age, 0)
        self.assertLessEqual(age, 6)

    def test_refuses_sealed_origin(self):
        with self.assertRaises(TargetAccessError):
            self.history.latest_value_at_or_before(datetime(2016, 6, 1, 0), 6)


if __name__ == "__main__":
    unittest.main()
