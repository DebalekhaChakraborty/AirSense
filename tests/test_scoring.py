"""Unit tests for the development scoring oracle.

Phase 12A. These tests were written after the study completed; they add
coverage to existing frozen production code and change none of it.

``src/evaluation/scoring.py`` is the only place in the codebase that reads a
validation PM2.5 value as an evaluation truth rather than as causal history.
It carried no direct tests. Its two documented guarantees are asserted here:

* it refuses any timestamp outside the validation window, so the locked test
  target cannot be read through it;
* callers must present already-produced predictions, enforcing predict-then-
  score ordering.

Most cases use a minimal synthetic station CSV in the canonical 18-column raw
schema, so the assertions are deterministic and do not depend on which hours
happen to be observed in the real archive. One case uses the real archive, to
show the property holds for the oracle the study actually used.
"""

import csv
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (                              # noqa: E402
    MISSING_TOKEN, PARTITION_BOUNDS, TARGET, index_of, total_hours)
from src.evaluation.scoring import (                              # noqa: E402
    ScoringAccessError, ValidationTargetOracle)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
STATION = "Dongsi"

TRAIN_START, TRAIN_END = PARTITION_BOUNDS["train"]
VALIDATION_START, VALIDATION_END = PARTITION_BOUNDS["validation"]
TEST_START, TEST_END = PARTITION_BOUNDS["test"]

COLUMNS = ["No", "year", "month", "day", "hour", "PM2.5", "PM10", "SO2",
           "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM",
           "station"]


def _row(number, stamp, pm25):
    """One canonical raw row; pm25 may be a float or the missing token."""
    return {
        "No": number, "year": stamp.year, "month": stamp.month,
        "day": stamp.day, "hour": stamp.hour,
        "PM2.5": pm25, "PM10": "20", "SO2": "3", "NO2": "10", "CO": "300",
        "O3": "40", "TEMP": "12.0", "PRES": "1010.0", "DEWP": "-1.0",
        "RAIN": "0.0", "wd": "NW", "WSPM": "1.5", "station": STATION,
    }


class SyntheticFixture(unittest.TestCase):
    """A tiny station CSV with known values at known timestamps."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="airsense_scoring_"))
        cls.csv_path = cls.tmp / ("PRSA_Data_%s_synthetic.csv" % STATION)

        # Deliberately spread across all three partitions.
        cls.train_stamp = TRAIN_START + timedelta(hours=5)
        cls.valid_stamps = [VALIDATION_START + timedelta(hours=h)
                            for h in (0, 1, 2, 3)]
        cls.missing_stamp = VALIDATION_START + timedelta(hours=4)
        cls.test_stamp = TEST_START + timedelta(hours=7)

        cls.valid_values = [101.0, 202.0, 303.0, 404.0]
        rows = [_row(1, cls.train_stamp, "11.0")]
        for i, (s, v) in enumerate(zip(cls.valid_stamps, cls.valid_values), 2):
            rows.append(_row(i, s, "%.1f" % v))
        rows.append(_row(6, cls.missing_stamp, MISSING_TOKEN))
        # A finite test-period value that the oracle must refuse to ingest.
        rows.append(_row(7, cls.test_stamp, "999.0"))

        with cls.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        cls.oracle = ValidationTargetOracle(STATION, cls.csv_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(str(cls.tmp), ignore_errors=True)


class TestValidationScoring(SyntheticFixture):

    # -- A ----------------------------------------------------------------
    def test_a_loads_and_scores_valid_validation_targets(self):
        idx = [index_of(s) for s in self.valid_stamps]
        actuals = self.oracle.actuals(idx, np.zeros(len(idx)))
        self.assertEqual(actuals.shape, (len(idx),))
        np.testing.assert_allclose(actuals, np.array(self.valid_values))

    def test_a_backing_array_spans_the_whole_timeline(self):
        self.assertEqual(self.oracle._values.shape, (total_hours(),))

    # -- B ----------------------------------------------------------------
    def test_b_too_few_predictions_raises(self):
        idx = [index_of(s) for s in self.valid_stamps]
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals(idx, np.zeros(len(idx) - 1))

    def test_b_too_many_predictions_raises(self):
        idx = [index_of(s) for s in self.valid_stamps]
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals(idx, np.zeros(len(idx) + 1))

    def test_b_empty_predictions_against_nonempty_targets_raises(self):
        idx = [index_of(self.valid_stamps[0])]
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals(idx, np.zeros(0))

    # -- C ----------------------------------------------------------------
    def test_c_target_before_validation_raises(self):
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals([index_of(self.train_stamp)], np.zeros(1))

    def test_c_one_hour_before_validation_start_raises(self):
        before = index_of(VALIDATION_START) - 1
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals([before], np.zeros(1))

    def test_c_mixed_batch_with_one_early_index_raises(self):
        idx = [index_of(self.valid_stamps[0]), index_of(self.train_stamp)]
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals(idx, np.zeros(2))

    # -- D ----------------------------------------------------------------
    def test_d_target_inside_locked_test_raises(self):
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals([index_of(self.test_stamp)], np.zeros(1))

    def test_d_first_hour_of_locked_test_raises(self):
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals([index_of(TEST_START)], np.zeros(1))

    def test_d_one_hour_after_validation_end_raises(self):
        after = index_of(VALIDATION_END) + 1
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals([after], np.zeros(1))

    def test_d_mixed_batch_with_one_test_index_raises(self):
        idx = [index_of(self.valid_stamps[0]), index_of(TEST_START)]
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals(idx, np.zeros(2))

    # -- E ----------------------------------------------------------------
    def test_e_no_finite_test_period_value_is_materialised(self):
        """The CSV carried a finite test-period value; it must not be read."""
        tail = self.oracle._values[index_of(TEST_START):]
        self.assertTrue(np.isnan(tail).all())

    def test_e_no_finite_pre_validation_value_is_materialised(self):
        head = self.oracle._values[:index_of(VALIDATION_START)]
        self.assertTrue(np.isnan(head).all())

    def test_e_only_validation_hours_are_finite(self):
        finite = np.flatnonzero(np.isfinite(self.oracle._values))
        self.assertTrue(finite.size > 0)
        self.assertGreaterEqual(int(finite.min()), index_of(VALIDATION_START))
        self.assertLessEqual(int(finite.max()), index_of(VALIDATION_END))

    # -- F ----------------------------------------------------------------
    def test_f_structurally_missing_validation_target_raises(self):
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals([index_of(self.missing_stamp)], np.zeros(1))

    def test_f_missing_target_inside_an_otherwise_valid_batch_raises(self):
        idx = [index_of(self.valid_stamps[0]), index_of(self.missing_stamp)]
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals(idx, np.zeros(2))

    def test_f_unobserved_validation_hour_raises(self):
        """An hour absent from the CSV entirely is also refused."""
        absent = index_of(VALIDATION_START + timedelta(hours=500))
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals([absent], np.zeros(1))

    # -- G ----------------------------------------------------------------
    def test_g_row_alignment_is_preserved_in_request_order(self):
        order = [2, 0, 3, 1]
        idx = [index_of(self.valid_stamps[i]) for i in order]
        expected = np.array([self.valid_values[i] for i in order])
        np.testing.assert_allclose(
            self.oracle.actuals(idx, np.zeros(len(idx))), expected)

    def test_g_repeated_index_returns_repeated_value(self):
        first = index_of(self.valid_stamps[0])
        actuals = self.oracle.actuals([first, first, first], np.zeros(3))
        np.testing.assert_allclose(
            actuals, np.repeat(self.valid_values[0], 3))

    def test_g_single_target_returns_length_one(self):
        idx = [index_of(self.valid_stamps[1])]
        actuals = self.oracle.actuals(idx, np.zeros(1))
        self.assertEqual(actuals.shape, (1,))
        self.assertAlmostEqual(float(actuals[0]), self.valid_values[1])


class TestRealArchiveOracle(unittest.TestCase):
    """The same seal property, on the oracle the study actually used."""

    @classmethod
    def setUpClass(cls):
        path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % STATION))
        cls.oracle = ValidationTargetOracle(STATION, path)

    def test_real_oracle_materialises_no_test_period_value(self):
        tail = self.oracle._values[index_of(TEST_START):]
        self.assertTrue(np.isnan(tail).all())

    def test_real_oracle_materialises_no_training_period_value(self):
        head = self.oracle._values[:index_of(VALIDATION_START)]
        self.assertTrue(np.isnan(head).all())

    def test_real_oracle_refuses_a_locked_test_timestamp(self):
        with self.assertRaises(ScoringAccessError):
            self.oracle.actuals([index_of(TEST_START)], np.zeros(1))

    def test_real_oracle_has_observed_validation_targets(self):
        finite = np.flatnonzero(np.isfinite(self.oracle._values))
        self.assertTrue(finite.size > 0)


if __name__ == "__main__":
    unittest.main()
