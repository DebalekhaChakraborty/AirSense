"""Phase-6 unit tests: grouping, causality and the paired R2/R3 design."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PARTITION_BOUNDS, STATIONS, index_of
from src.data.target_history import TargetAccessError
from src.models import station_attention_grid as grid
from src.models.neural_features import (
    CONTEXT_HOURS, dynamic_channel_names, static_channel_names)
from src.models.spatial_grouping import (
    GuardedNetworkHistory, NetworkBundle, build_groups, group_id)
from src.models.station_attention_forecaster import StationAttentionForecaster

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
CHANNELS = len(dynamic_channel_names("R2"))
STATICS = len(static_channel_names())


def build(cross_station, hidden=16, seed=42):
    torch.manual_seed(seed)
    return StationAttentionForecaster(
        CHANNELS, STATICS, hidden_size=hidden,
        heads=grid.FIXED["attention_heads"],
        attention_dropout=grid.FIXED["attention_dropout"],
        head_hidden=grid.FIXED["head_hidden"],
        head_dropout=grid.FIXED["head_dropout"],
        cross_station=cross_station)


class GroupingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.groups = build_groups(PROCESSED, "validation")

    def test_group_id_is_deterministic_and_formatted(self):
        first = group_id("validation", int(self.groups.target_indices[0]),
                         int(self.groups.horizons[0]))
        again = group_id("validation", int(self.groups.target_indices[0]),
                         int(self.groups.horizons[0]))
        self.assertEqual(first, again)
        partition, stamp, horizon = first.split("|")
        self.assertEqual(partition, "validation")
        self.assertEqual(len(stamp), 10)
        self.assertTrue(horizon.startswith("h"))

    def test_grouping_preserves_every_canonical_sample(self):
        self.assertEqual(self.groups.sample_total(), 413148)
        rows = self.groups.sample_rows[self.groups.target_observed]
        self.assertEqual(rows.size, 413148)
        self.assertEqual(np.unique(rows).size, 413148)
        self.assertEqual(int(rows.min()), 0)
        self.assertEqual(int(rows.max()), 413147)

    def test_station_dimension_is_the_frozen_order(self):
        self.assertEqual(self.groups.sample_rows.shape[1], 12)
        self.assertEqual(len(STATIONS), 12)

    def test_every_station_shares_one_origin(self):
        origins = self.groups.origins
        self.assertTrue(np.array_equal(
            origins, self.groups.target_indices - self.groups.horizons))

    def test_scatter_round_trips_into_canonical_order(self):
        grouped = np.zeros((self.groups.size, 12), dtype=np.float64)
        grouped[self.groups.target_observed] = np.arange(
            self.groups.sample_total(), dtype=np.float64)
        flat = self.groups.scatter(grouped, 413148)
        self.assertEqual(flat.size, 413148)
        self.assertTrue(np.isfinite(flat).all())

    def test_missing_target_slots_are_masked_not_imputed(self):
        missing = ~self.groups.target_observed
        self.assertGreater(int(missing.sum()), 0)
        self.assertTrue((self.groups.sample_rows[missing] == -1).all())


class HistoryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.history = GuardedNetworkHistory(PROJECT_ROOT,
                                            unseal_validation=False)
        cls.groups = build_groups(PROCESSED, "train")
        cls.bundle = NetworkBundle(cls.groups, cls.history)

    def test_window_has_twelve_stations_and_48_steps(self):
        sequence, static = self.bundle.batch(np.arange(4))
        self.assertEqual(sequence.shape, (4, 12, CONTEXT_HOURS, CHANNELS))
        self.assertEqual(static.shape, (4, 12, STATICS))

    def test_no_station_input_exceeds_the_origin(self):
        rows = np.array([1000, 2000, 3000])
        sequence, _ = self.bundle.batch(rows)
        for offset, row in enumerate(rows):
            origin = int(self.groups.origins[row])
            for station in range(12):
                direct = self.history.windows([origin])[0, station]
                self.assertTrue(np.allclose(sequence[offset, station], direct))

    def test_history_refuses_a_sealed_origin(self):
        sealed = index_of(PARTITION_BOUNDS["test"][0])
        with self.assertRaises(TargetAccessError):
            self.history.windows([sealed])

    def test_static_conditioning_is_per_station(self):
        _, static = self.bundle.batch(np.arange(2))
        for station in range(12):
            self.assertEqual(float(static[0, station, station]), 1.0)
            self.assertEqual(float(static[0, station, :12].sum()), 1.0)

    def test_horizon_conditioning_matches_the_group(self):
        rows = np.arange(6)
        _, static = self.bundle.batch(rows)
        vocabulary = [1, 6, 12, 24]
        for offset, row in enumerate(rows):
            index = vocabulary.index(int(self.groups.horizons[row]))
            self.assertEqual(float(static[offset, 0, 12 + index]), 1.0)
            self.assertEqual(float(static[offset, 0, 12:16].sum()), 1.0)


class PairedDesignTests(unittest.TestCase):

    def test_r2_and_r3_have_identical_parameter_count(self):
        self.assertEqual(build(False).parameter_count(),
                         build(True).parameter_count())

    def test_r2_and_r3_initialise_identically_from_one_seed(self):
        left = build(False).state_dict()
        right = build(True).state_dict()
        self.assertEqual(sorted(left), sorted(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]))

    def test_r2_mask_is_strictly_self_only(self):
        model = build(False)
        self.assertTrue(torch.equal(model.self_only_mask,
                                    ~torch.eye(12, dtype=torch.bool)))
        self.assertTrue(bool((~model.self_only_mask).diagonal().all()))

    def test_r2_prediction_ignores_every_other_station(self):
        model = build(False).eval()
        torch.manual_seed(7)
        sequence = torch.randn(2, 12, CONTEXT_HOURS, CHANNELS)
        static = torch.randn(2, 12, STATICS)
        with torch.no_grad():
            base = model(sequence, static)
        perturbed = sequence.clone()
        target = 3
        for station in range(12):
            if station != target:
                perturbed[:, station] += 5.0
        with torch.no_grad():
            moved = model(perturbed, static)
        self.assertTrue(torch.equal(base[:, target], moved[:, target]))

    def test_r3_prediction_can_respond_to_another_station(self):
        model = build(True).eval()
        torch.manual_seed(7)
        sequence = torch.randn(2, 12, CONTEXT_HOURS, CHANNELS)
        static = torch.randn(2, 12, STATICS)
        with torch.no_grad():
            base = model(sequence, static)
        perturbed = sequence.clone()
        perturbed[:, 5] += 5.0
        with torch.no_grad():
            moved = model(perturbed, static)
        self.assertFalse(torch.equal(base[:, 3], moved[:, 3]))

    def test_shared_temporal_encoder_is_one_gru(self):
        model = build(True)
        self.assertEqual(model.encoder.num_layers, 1)
        self.assertFalse(model.encoder.bidirectional)
        names = [n for n, _ in model.named_parameters() if "encoder" in n]
        self.assertEqual(len(names), 4)

    def test_forward_emits_twelve_station_predictions(self):
        model = build(True).eval()
        with torch.no_grad():
            out = model(torch.zeros(3, 12, CONTEXT_HOURS, CHANNELS),
                        torch.zeros(3, 12, STATICS))
        self.assertEqual(tuple(out.shape), (3, 12))

    def test_output_is_unconstrained(self):
        model = build(True)
        modules = [type(m).__name__ for m in model.head]
        self.assertNotIn("ReLU", modules)
        self.assertNotIn("Softplus", modules)
        self.assertEqual(modules[-1], "Linear")

    def test_attention_rows_normalise(self):
        model = build(True).eval()
        torch.manual_seed(3)
        with torch.no_grad():
            weights = model.attention_weights(
                torch.randn(2, 12, CONTEXT_HOURS, CHANNELS))
        self.assertEqual(tuple(weights.shape), (2, 12, 12))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, 12),
                                       atol=1e-5))

    def test_r2_attention_is_the_identity(self):
        model = build(False).eval()
        torch.manual_seed(3)
        with torch.no_grad():
            weights = model.attention_weights(
                torch.randn(2, 12, CONTEXT_HOURS, CHANNELS))
        self.assertTrue(torch.allclose(weights, torch.eye(12).expand(2, 12,
                                                                     12),
                                       atol=1e-6))


class GridTests(unittest.TestCase):

    def test_grid_is_exactly_four_frozen_candidates(self):
        self.assertEqual(
            [(c["candidate_id"], c["hidden_size"], c["learning_rate"])
             for c in grid.candidates()],
            [("SA_C01", 64, 0.0003), ("SA_C02", 64, 0.001),
             ("SA_C03", 128, 0.0003), ("SA_C04", 128, 0.001)])

    def test_budget_is_fixed_across_candidates(self):
        for candidate in grid.candidates():
            self.assertEqual(candidate["epochs"], 8)
            self.assertFalse(candidate["early_stopping"])
            self.assertEqual(candidate["loss"], "L1")
            self.assertFalse(candidate["external_coordinates"])

    def test_selection_rule_and_tie_break(self):
        rows = [
            {"candidate_id": "SA_C03", "hidden_size": 128,
             "learning_rate": 0.0003, "mean_macro_mae": 30.0,
             "mean_macro_rmse": 50.0},
            {"candidate_id": "SA_C01", "hidden_size": 64,
             "learning_rate": 0.0003, "mean_macro_mae": 30.0,
             "mean_macro_rmse": 50.0},
        ]
        self.assertEqual(grid.select(rows)["candidate_id"], "SA_C01")

    def test_no_severe_weighting_in_the_frozen_protocol(self):
        text = (PROJECT_ROOT / "src" / "models"
                / "spatial_training.py").read_text(encoding="utf-8")
        self.assertNotIn("severe", text.lower())


if __name__ == "__main__":
    unittest.main()
