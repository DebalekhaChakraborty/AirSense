"""Unit tests for the Phase-4 temporal neural baselines."""

import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (                              # noqa: E402
    PARTITION_BOUNDS, STATIONS, WD_VOCABULARY, index_of)
from src.models import neural_grid                                # noqa: E402
from src.models.gru_forecaster import GRUForecaster               # noqa: E402
from src.models.neural_data import (                              # noqa: E402
    INTERNAL_CORE, INTERNAL_TUNING, internal_mask, load_sample_index)
from src.models.neural_features import (                          # noqa: E402
    CONTEXT_HOURS, HORIZON_VOCABULARY, build_static_matrix,
    dynamic_channel_names, gather_windows, static_channel_names)
from src.models.tcn_forecaster import (                           # noqa: E402
    CONVS_PER_BLOCK, DILATIONS, KERNEL_SIZE, TCNForecaster, receptive_field)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"


class TestGRU(unittest.TestCase):

    def test_output_shape_and_dimensions(self):
        model = GRUForecaster(46, 28, hidden_size=64)
        out = model(torch.randn(7, CONTEXT_HOURS, 46), torch.randn(7, 28))
        self.assertEqual(tuple(out.shape), (7,))

    def test_is_unidirectional_single_layer(self):
        model = GRUForecaster(31, 28, hidden_size=128)
        self.assertFalse(model.gru.bidirectional)
        self.assertEqual(model.gru.num_layers, 1)
        self.assertTrue(model.gru.batch_first)

    def test_output_is_unconstrained(self):
        model = GRUForecaster(28, 28, hidden_size=64)
        with torch.no_grad():
            model.head[-1].bias.fill_(-50.0)
            out = model(torch.zeros(3, CONTEXT_HOURS, 28), torch.zeros(3, 28))
        self.assertTrue(bool((out < 0).all()),
                        "negative predictions must remain visible")


class TestTCNCausality(unittest.TestCase):

    def test_receptive_field_covers_the_context(self):
        self.assertEqual(receptive_field(), 61)
        self.assertGreaterEqual(receptive_field(), CONTEXT_HOURS)

    def test_receptive_field_formula(self):
        expected = 1 + sum((KERNEL_SIZE - 1) * d * CONVS_PER_BLOCK
                           for d in DILATIONS)
        self.assertEqual(receptive_field(), expected)

    def test_no_dependence_on_a_future_timestep(self):
        """Perturbing input after position p must not change output at p."""
        torch.manual_seed(0)
        model = TCNForecaster(8, 4, channels=16)
        model.eval()
        sequence = torch.randn(1, CONTEXT_HOURS, 8)
        static = torch.zeros(1, 4)
        with torch.no_grad():
            blocks = model.blocks(sequence.transpose(1, 2))
            perturbed = sequence.clone()
            perturbed[0, 30:, :] += 1000.0
            blocks_perturbed = model.blocks(perturbed.transpose(1, 2))
        self.assertTrue(torch.equal(blocks[0, :, 29],
                                    blocks_perturbed[0, :, 29]),
                        "a causal network leaked information from the future")

    def test_left_padding_only(self):
        model = TCNForecaster(4, 4, channels=8)
        for block in model.blocks:
            self.assertEqual(block.conv1.conv.padding, (0,))
            self.assertEqual(block.conv2.conv.padding, (0,))
            self.assertGreater(block.conv1.left_padding, 0)

    def test_output_is_unconstrained(self):
        model = TCNForecaster(28, 28, channels=32)
        with torch.no_grad():
            model.head[-1].bias.fill_(-25.0)
            out = model(torch.zeros(2, CONTEXT_HOURS, 28), torch.zeros(2, 28))
        self.assertTrue(bool((out < 0).all()))


class TestFeatureSchema(unittest.TestCase):

    def test_static_dimensions(self):
        names = static_channel_names()
        self.assertEqual(len(names), 28)
        self.assertEqual(len([n for n in names if n.startswith("station_is_")]),
                         12)
        self.assertEqual(len([n for n in names if n.startswith("horizon_is_")]),
                         4)

    def test_horizon_one_hot_in_frozen_order(self):
        self.assertEqual(HORIZON_VOCABULARY, [1, 6, 12, 24])
        static = build_static_matrix([0, 0, 0, 0], [1, 6, 12, 24],
                                     [100, 100, 100, 100],
                                     [101, 106, 112, 124],
                                     np.zeros((200, 6), dtype=np.float32))
        block = static[:, 12:16]
        self.assertTrue(np.array_equal(block, np.eye(4, dtype=np.float32)))

    def test_station_one_hot(self):
        static = build_static_matrix([0, 5, 11], [1, 1, 1], [100, 100, 100],
                                     [101, 101, 101],
                                     np.zeros((200, 6), dtype=np.float32))
        self.assertEqual(static[:, :12].sum(), 3.0)
        self.assertEqual(static[0, 0], 1.0)
        self.assertEqual(static[1, 5], 1.0)
        self.assertEqual(static[2, 11], 1.0)

    def test_wind_is_one_hot_not_ordinal(self):
        for regime in ("R0", "R1", "R2"):
            names = dynamic_channel_names(regime)
            wind = [n for n in names if n.startswith("wd_is_")]
            self.assertEqual(len(wind), len(WD_VOCABULARY) + 1)
            self.assertIn("wd_is_MISSING", wind)
            self.assertNotIn("wd_code", names)

    def test_no_day_of_week_or_weekend(self):
        joined = " ".join(static_channel_names()
                          + dynamic_channel_names("R2"))
        self.assertNotIn("day_of_week", joined)
        self.assertNotIn("weekend", joined)


class TestRegimeRestrictions(unittest.TestCase):

    def test_r0_sees_no_pollutant_history(self):
        names = dynamic_channel_names("R0")
        for pollutant in ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3"):
            self.assertNotIn("value:%s" % pollutant, names)

    def test_r1_sees_pm25_and_no_other_pollutant(self):
        names = dynamic_channel_names("R1")
        self.assertIn("value:PM2.5", names)
        for pollutant in ("PM10", "SO2", "NO2", "CO", "O3"):
            self.assertNotIn("value:%s" % pollutant, names)

    def test_r2_sees_all_frozen_target_station_pollutants(self):
        names = dynamic_channel_names("R2")
        for pollutant in ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3"):
            self.assertIn("value:%s" % pollutant, names)

    def test_channel_counts_are_stable(self):
        self.assertEqual(
            [len(dynamic_channel_names(r)) for r in ("R0", "R1", "R2")],
            [28, 31, 46])


class TestWindowing(unittest.TestCase):

    def test_window_is_48_steps_ending_at_the_origin(self):
        dynamic = np.arange(2 * 200 * 3, dtype=np.float32).reshape(2, 200, 3)
        windows = gather_windows(dynamic, [0, 1], [100, 150])
        self.assertEqual(windows.shape, (2, CONTEXT_HOURS, 3))
        self.assertTrue(np.array_equal(windows[0, -1], dynamic[0, 100]))
        self.assertTrue(np.array_equal(windows[0, 0], dynamic[0, 100 - 47]))

    def test_incomplete_context_is_refused(self):
        dynamic = np.zeros((1, 200, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            gather_windows(dynamic, [0], [10])


class TestInternalSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stations, cls.horizons, cls.targets = load_sample_index(
            PROCESSED, "train")

    def test_core_and_tuning_partition_training_exactly(self):
        core = internal_mask(self.targets, INTERNAL_CORE)
        tuning = internal_mask(self.targets, INTERNAL_TUNING)
        self.assertEqual(int((core & tuning).sum()), 0)
        self.assertEqual(int(core.sum() + tuning.sum()), self.targets.size)

    def test_tuning_is_strictly_after_core(self):
        self.assertLess(INTERNAL_CORE[1], INTERNAL_TUNING[0])

    def test_split_lies_entirely_inside_the_training_partition(self):
        train_start, train_end = PARTITION_BOUNDS["train"]
        self.assertGreaterEqual(INTERNAL_CORE[0], train_start)
        self.assertLessEqual(INTERNAL_TUNING[1], train_end)

    def test_no_external_validation_timestamp_is_involved(self):
        validation_start = PARTITION_BOUNDS["validation"][0]
        self.assertLess(INTERNAL_TUNING[1], validation_start)


class TestGridAndSelection(unittest.TestCase):

    def test_gru_grid_is_exactly_four_frozen_candidates(self):
        grid = neural_grid.candidates("GRU")
        self.assertEqual([c["candidate_id"] for c in grid],
                         ["GRU_C01", "GRU_C02", "GRU_C03", "GRU_C04"])
        self.assertEqual([(c["capacity"], c["learning_rate"]) for c in grid],
                         [(64, 0.0003), (64, 0.001), (128, 0.0003),
                          (128, 0.001)])

    def test_tcn_grid_is_exactly_four_frozen_candidates(self):
        grid = neural_grid.candidates("TCN")
        self.assertEqual([c["candidate_id"] for c in grid],
                         ["TCN_C01", "TCN_C02", "TCN_C03", "TCN_C04"])
        self.assertEqual([(c["capacity"], c["learning_rate"]) for c in grid],
                         [(32, 0.0003), (32, 0.001), (64, 0.0003),
                          (64, 0.001)])

    def test_training_budget_is_fixed_across_candidates(self):
        for architecture in ("GRU", "TCN"):
            for candidate in neural_grid.candidates(architecture):
                self.assertEqual(candidate["epochs"], 8)
                self.assertEqual(candidate["batch_size"], 1024)
                self.assertEqual(candidate["optimizer"], "AdamW")
                self.assertEqual(candidate["loss"], "L1")
                self.assertFalse(candidate["early_stopping"])
                self.assertIsNone(candidate["scheduler"])
                self.assertEqual(candidate["seed"], 42)

    def test_selection_prefers_lower_mean_macro_mae(self):
        rows = [
            {"candidate_id": "GRU_C01", "capacity": 64,
             "learning_rate": 0.0003, "mean_macro_mae": 30.0,
             "mean_macro_rmse": 50.0},
            {"candidate_id": "GRU_C02", "capacity": 64,
             "learning_rate": 0.001, "mean_macro_mae": 29.0,
             "mean_macro_rmse": 51.0},
        ]
        self.assertEqual(neural_grid.select(rows)["candidate_id"], "GRU_C02")

    def test_tie_break_rmse_then_capacity_then_lr_then_id(self):
        base = {"mean_macro_mae": 30.0, "mean_macro_rmse": 50.0}
        rows = [dict(base, candidate_id="GRU_C03", capacity=128,
                     learning_rate=0.0003),
                dict(base, candidate_id="GRU_C01", capacity=64,
                     learning_rate=0.0003)]
        self.assertEqual(neural_grid.select(rows)["candidate_id"], "GRU_C01")
        rows = [dict(base, candidate_id="GRU_C02", capacity=64,
                     learning_rate=0.001),
                dict(base, candidate_id="GRU_C01", capacity=64,
                     learning_rate=0.0003)]
        self.assertEqual(neural_grid.select(rows)["candidate_id"], "GRU_C01")


if __name__ == "__main__":
    unittest.main()
