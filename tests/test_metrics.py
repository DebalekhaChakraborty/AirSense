"""Unit tests for the primary macro station-horizon metric."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import metrics                                # noqa: E402


class TestResidualConvention(unittest.TestCase):

    def test_positive_residual_is_underprediction(self):
        residual = metrics.residuals([100.0], [60.0])
        self.assertGreater(residual[0], 0)

    def test_negative_residual_is_overprediction(self):
        residual = metrics.residuals([60.0], [100.0])
        self.assertLess(residual[0], 0)


class TestMacroMetric(unittest.TestCase):

    def setUp(self):
        # Two stations, two horizons. Deliberately unequal cell sizes so the
        # macro and micro answers differ.
        self.actual = np.array([10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 100.0])
        self.prediction = np.array([12.0, 22.0, 33.0, 1.0, 1.0, 1.0, 90.0])
        self.stations = np.array(["A", "A", "A", "B", "B", "B", "B"])
        self.horizons = np.array([1, 1, 1, 1, 1, 1, 6])

    def test_cells_are_computed_independently(self):
        cells = metrics.cell_metrics(self.actual, self.prediction,
                                     self.stations, self.horizons)
        self.assertEqual(sorted(cells), [("A", 1), ("B", 1), ("B", 6)])
        self.assertAlmostEqual(cells[("A", 1)]["mae"], (2 + 2 + 3) / 3.0)
        self.assertAlmostEqual(cells[("B", 1)]["mae"], 1.0)
        self.assertAlmostEqual(cells[("B", 6)]["mae"], 10.0)

    def test_macro_is_the_equal_weight_mean_of_cells(self):
        cells = metrics.cell_metrics(self.actual, self.prediction,
                                     self.stations, self.horizons)
        macro = metrics.macro_from_cells(cells)
        self.assertAlmostEqual(macro, (7 / 3.0 + 1.0 + 10.0) / 3.0)

    def test_macro_differs_from_micro_when_cells_are_unequal(self):
        cells = metrics.cell_metrics(self.actual, self.prediction,
                                     self.stations, self.horizons)
        micro = metrics.mae(self.actual, self.prediction)
        self.assertNotAlmostEqual(metrics.macro_from_cells(cells), micro)

    def test_macro_station_mae_is_the_selection_criterion_form(self):
        actual = np.array([10.0, 20.0, 0.0, 0.0])
        prediction = np.array([11.0, 21.0, 5.0, 5.0])
        stations = np.array(["A", "A", "B", "B"])
        self.assertAlmostEqual(
            metrics.macro_station_mae(actual, prediction, stations),
            (1.0 + 5.0) / 2.0)


class TestSevereEndpoint(unittest.TestCase):

    def test_threshold_is_the_frozen_244(self):
        self.assertEqual(metrics.SEVERE_THRESHOLD, 244.0)

    def test_severe_selects_strictly_above_the_threshold(self):
        actual = np.array([244.0, 244.1, 300.0])
        prediction = np.array([200.0, 200.0, 200.0])
        stations = np.array(["A", "A", "A"])
        horizons = np.array([1, 1, 1])
        summary = metrics.severe_metrics(actual, prediction, stations,
                                         horizons)
        self.assertEqual(summary["severe_n"], 2)

    def test_underprediction_percentage_uses_the_frozen_convention(self):
        actual = np.array([300.0, 300.0, 300.0, 300.0])
        prediction = np.array([100.0, 100.0, 100.0, 400.0])
        stations = np.array(["A"] * 4)
        horizons = np.array([1] * 4)
        summary = metrics.severe_metrics(actual, prediction, stations,
                                         horizons)
        self.assertAlmostEqual(summary["severe_underprediction_pct"], 75.0)

    def test_empty_severe_cells_are_reported_not_hidden(self):
        actual = np.array([300.0, 10.0])
        prediction = np.array([100.0, 10.0])
        stations = np.array(["A", "B"])
        horizons = np.array([1, 1])
        summary = metrics.severe_metrics(actual, prediction, stations,
                                         horizons)
        self.assertEqual(summary["severe_contributing_cells"], 1)
        self.assertEqual(summary["severe_total_cells"], 2)
        self.assertEqual(summary["severe_empty_cells"], 1)


class TestNegativePredictions(unittest.TestCase):

    def test_negative_predictions_are_counted_not_clipped(self):
        summary = metrics.negative_prediction_summary(
            np.array([-5.0, 1.0, 2.0, 3.0]))
        self.assertEqual(summary["n_negative"], 1)
        self.assertAlmostEqual(summary["pct_negative"], 25.0)
        self.assertAlmostEqual(summary["min_prediction"], -5.0)


if __name__ == "__main__":
    unittest.main()
