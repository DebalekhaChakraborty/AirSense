"""Pre-opening unit tests for the AirSense V2 locked final-test layer.

Protocol Phase 8, section 13. Every fixture here is drawn from training,
internal-tuning or development-validation data. **No test carries a numerical
final-test target**, and the tests are required to pass before the sealed
partition is opened.

The decisive test is :meth:`TestChronologicalReveal.test_no_influence_from_
observations_after_the_origin`, which does not assert an access rule but
demonstrates the consequence: replacing every observation strictly after a
forecast origin leaves that origin's feature row bit-identical.
"""

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
    PARTITION_BOUNDS, STATIONS, TARGET, index_of, load_training_statistics,
    station_code, timestamp_at)
from src.data.rolling_history import RollingPm25Series            # noqa: E402
from src.data.target_history import (                             # noqa: E402
    CausalTargetHistory, TargetAccessError)
from src.data.windowing import StationSeries                      # noqa: E402
from src.evaluation import metrics as M                           # noqa: E402
from src.evaluation.final_test import (                           # noqa: E402
    CONFIRMATORY_MODELS, CONTEXT_HOURS, EXPECTED_TEST_SAMPLES,
    FinalTestAccessError, FinalTestTargetOracle, MaskedStationInputs,
    RevealLedger, SEVERE_THRESHOLD, TEST_END, TEST_START,
    assert_confirmatory_only, chronological_blocks, load_test_index,
    verify_matrix, wind_onehot_slice)
from src.models import baselines                                  # noqa: E402
from src.models.b3_features import (                              # noqa: E402
    StationFeatureSource, build_matrix, feature_names)
from src.models.neural_features import (                          # noqa: E402
    dynamic_channel_names, gather_windows, static_channel_names,
    build_station_dynamic_matrix)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
ARTIFACTS = PROJECT_ROOT / "artifacts"
STATION = "Dongsi"
TRAIN_END = PARTITION_BOUNDS["train"][1]
VALIDATION_START, VALIDATION_END = PARTITION_BOUNDS["validation"]


def raw_path(station):
    return next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))


def development_rolling(station=STATION):
    """A guarded series unsealed no further than development validation."""
    statistics = load_training_statistics(
        PROJECT_ROOT / "configs" / "preprocessing.json")
    with open(str(ARTIFACTS / "preprocessing_statistics.json"),
              encoding="utf-8") as handle:
        medians = json.load(handle)["station_training_medians"]
    return RollingPm25Series(station, raw_path(station), statistics,
                             medians[station][TARGET],
                             unseal_until=VALIDATION_END)


class TestSampleUniverseMetadata(unittest.TestCase):
    """Section 8: the index is structural metadata, not target values."""

    @classmethod
    def setUpClass(cls):
        cls.index = load_test_index(PROCESSED)

    def test_loader_exposes_no_numerical_target(self):
        for key, array in self.index.items():
            self.assertNotIn(key.lower(),
                             ("pm2.5", "pm25", "target", "target_value", "y"))
        # target_row_index is a timeline position, bounded by the timeline.
        self.assertTrue((self.index["target_index"] >= 0).all())
        self.assertTrue((self.index["target_index"]
                         <= index_of(TEST_END)).all())

    def test_canonical_sample_count(self):
        self.assertEqual(self.index["target_index"].size,
                         EXPECTED_TEST_SAMPLES)

    def test_target_timestamp_partition_semantics(self):
        low, high = index_of(TEST_START), index_of(TEST_END)
        self.assertTrue((self.index["target_index"] >= low).all())
        self.assertTrue((self.index["target_index"] <= high).all())

    def test_no_timestamp_after_origin(self):
        self.assertTrue((self.index["origin_index"]
                         < self.index["target_index"]).all())

    def test_origin_is_target_minus_horizon(self):
        self.assertTrue((self.index["origin_index"]
                         == self.index["target_index"]
                         - self.index["horizon"]).all())

    def test_forty_eight_hour_context_available(self):
        self.assertTrue((self.index["origin_index"]
                         - (CONTEXT_HOURS - 1) >= 0).all())

    def test_cross_partition_boundary_history(self):
        """Early test origins legitimately sit before the test start."""
        earliest = int(self.index["origin_index"].min())
        self.assertLess(earliest, index_of(TEST_START))
        self.assertGreaterEqual(earliest - (CONTEXT_HOURS - 1), 0)

    def test_horizons_are_the_frozen_four(self):
        self.assertEqual(sorted(set(self.index["horizon"].tolist())),
                         [1, 6, 12, 24])

    def test_station_vocabulary_is_frozen_twelve(self):
        self.assertEqual(sorted(set(self.index["station"].tolist())),
                         sorted(STATIONS))


class TestConfirmatoryWhitelist(unittest.TestCase):
    """Section 13: only the three frozen models may ever be requested."""

    def test_whitelist_is_exactly_the_frozen_three(self):
        self.assertEqual(CONFIRMATORY_MODELS, ["B3_R2", "GRU_R1", "B0"])

    def test_each_permitted_model_is_accepted(self):
        for name in ("B3_R2", "GRU_R1", "B0"):
            self.assertEqual(assert_confirmatory_only(name), name)

    def test_forbidden_model_cannot_be_requested(self):
        for name in ("TCN_R1", "SA_R3", "SA_R2", "iTransformer_R1",
                     "iTransformer_R2", "B1", "B2", "B3_R0", "B3_R1",
                     "GRU_R0", "GRU_R2", "TCN_R0", "TCN_R2", "Chronos",
                     "TimesFM", "Moirai"):
            with self.assertRaises(FinalTestAccessError):
                assert_confirmatory_only(name)


class TestFeatureSchemas(unittest.TestCase):
    """Sections 22 and 23: the frozen feature contracts are unchanged."""

    def test_b3_r2_feature_schema(self):
        names = feature_names("R2")
        self.assertEqual(len(names), 405)
        with open(str(ARTIFACTS / "information_regime_schema.json"),
                  encoding="utf-8") as handle:
            schema = json.load(handle)
        self.assertIn("R2", json.dumps(schema))
        self.assertEqual(names[0], "PM2.5_lag0")
        self.assertEqual(sum(1 for n in names if n.startswith("station_is_")),
                         12)
        self.assertEqual(sum(1 for n in names if n.startswith("wd_is_")), 17)

    def test_b3_horizon_model_mapping(self):
        with open(str(ARTIFACTS / "phase7_pretest_freeze.json"),
                  encoding="utf-8") as handle:
            freeze = json.load(handle)
        per_horizon = freeze["selected_configuration_and_artifacts"][
            "B3_R2"]["per_horizon_models"]
        self.assertEqual(sorted(per_horizon), ["h1", "h12", "h24", "h6"])
        for key, entry in per_horizon.items():
            horizon = int(key[1:])
            self.assertIn("b3_R2_h%d_" % horizon, entry["model_file"])
            self.assertTrue((PROJECT_ROOT / entry["model_file"]).is_file())

    def test_gru_r1_feature_schema(self):
        with open(str(PROJECT_ROOT
                      / "results/models/neural/GRU_R1_config.json"),
                  encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(config["regime"], "R1")
        self.assertEqual(config["context_hours"], CONTEXT_HOURS)
        self.assertEqual(config["dynamic_channels"],
                         len(dynamic_channel_names("R1")))
        self.assertEqual(config["static_channels"],
                         len(static_channel_names()))
        self.assertEqual(config["training"]["seed"], 42)
        self.assertFalse(config["output_constrained"])

    def test_gru_r1_excludes_co_pollutants(self):
        channels = dynamic_channel_names("R1")
        for pollutant in ("PM10", "SO2", "NO2", "CO", "O3"):
            self.assertNotIn("value:%s" % pollutant, channels)
        self.assertIn("value:PM2.5", channels)


class TestB0FallbackChain(unittest.TestCase):
    """Section 21: observed, else causal carry within 6 h, else the median."""

    @classmethod
    def setUpClass(cls):
        cls.rolling = development_rolling()

    def test_source_labels_are_the_frozen_three(self):
        self.assertEqual(baselines.B0_SOURCE_LABELS, {
            0: "observed_at_origin", 1: "short_causal_carry",
            2: "station_training_median_fallback"})

    def test_predictions_come_only_from_the_declared_chain(self):
        origins = np.arange(index_of(VALIDATION_START),
                            index_of(VALIDATION_START) + 500)
        values, sources = baselines.b0_causal_persistence(self.rolling,
                                                          origins)
        self.assertEqual(values.size, origins.size)
        self.assertTrue(np.isin(sources, [0, 1, 2]).all())
        self.assertTrue(np.isfinite(values).all())

    def test_prediction_equals_the_guarded_causal_value(self):
        origins = np.arange(index_of(VALIDATION_START) + 100,
                            index_of(VALIDATION_START) + 160)
        values, _ = baselines.b0_causal_persistence(self.rolling, origins)
        for position, origin in enumerate(origins):
            self.assertEqual(values[position],
                             self.rolling.native_at(int(origin), int(origin)))

    def test_median_fallback_is_training_derived(self):
        with open(str(ARTIFACTS / "preprocessing_statistics.json"),
                  encoding="utf-8") as handle:
            medians = json.load(handle)["station_training_medians"]
        self.assertIn(STATION, medians)
        self.assertIn(TARGET, medians[STATION])


class TestTargetRevealSemantics(unittest.TestCase):
    """Section 19: a target is inaccessible until it becomes history."""

    @classmethod
    def setUpClass(cls):
        cls.history = CausalTargetHistory(
            STATION, raw_path(STATION), value_access_until=VALIDATION_END)

    def test_target_unavailable_before_it_becomes_historical(self):
        origin = datetime(2015, 6, 1, 12)
        for ahead in (1, 6, 12, 24):
            future = timestamp_at(index_of(origin) + ahead)
            with self.assertRaises(TargetAccessError):
                self.history.value_at(future, origin)

    def test_target_available_once_timestamp_is_at_or_before_origin(self):
        origin = datetime(2015, 6, 1, 12)
        value = self.history.value_at(origin, origin)
        self.assertTrue(value is None or np.isfinite(value))
        earlier = timestamp_at(index_of(origin) - 5)
        value = self.history.value_at(earlier, origin)
        self.assertTrue(value is None or np.isfinite(value))

    def test_rolling_reveal_turns_a_target_into_history(self):
        """The same timestamp is refused at an earlier origin, allowed later."""
        stamp = datetime(2015, 6, 1, 12)
        before = timestamp_at(index_of(stamp) - 1)
        with self.assertRaises(TargetAccessError):
            self.history.value_at(stamp, before)
        self.assertEqual(self.history.value_at(stamp, stamp),
                         self.history.value_at(stamp,
                                               timestamp_at(index_of(stamp)
                                                            + 24)))

    def test_sealed_instance_refuses_a_later_partition(self):
        sealed = CausalTargetHistory(STATION, raw_path(STATION),
                                     value_access_until=TRAIN_END)
        with self.assertRaises(TargetAccessError):
            sealed.value_at(datetime(2015, 6, 1, 12),
                            datetime(2015, 6, 1, 12))

    def test_guarded_series_refuses_an_index_after_the_origin(self):
        rolling = development_rolling()
        origin = index_of(datetime(2015, 6, 1, 12))
        with self.assertRaises(TargetAccessError):
            rolling.native_at(origin + 1, origin)
        self.assertIsNotNone(rolling.native_at(origin, origin))


class TestRevealLedger(unittest.TestCase):
    """Section 20: the ledger counts, and refuses to move backwards."""

    def test_clean_access_records_no_violation(self):
        ledger = RevealLedger()
        ledger.advance_to(1000)
        ledger.note("clean", np.array([990, 995, 1000]),
                    np.array([1000, 1000, 1000]))
        self.assertEqual(ledger.violations, 0)
        self.assertEqual(ledger.summary()[
            "future_target_access_violations"], 0)

    def test_index_after_its_own_origin_is_counted(self):
        ledger = RevealLedger()
        ledger.advance_to(1000)
        offending = ledger.note("dirty", np.array([1001]), np.array([1000]))
        self.assertEqual(offending, 1)
        self.assertEqual(ledger.violations, 1)

    def test_index_after_the_ceiling_is_counted(self):
        ledger = RevealLedger()
        ledger.advance_to(500)
        offending = ledger.note("dirty", np.array([900]), np.array([900]))
        self.assertEqual(offending, 1)

    def test_ceiling_never_moves_backwards(self):
        ledger = RevealLedger()
        ledger.advance_to(1000)
        with self.assertRaises(FinalTestAccessError):
            ledger.advance_to(999)

    def test_blocks_are_strictly_chronological(self):
        index = load_test_index(PROCESSED)
        blocks = chronological_blocks(index["origin_index"])
        ceilings = [c for c, _ in blocks]
        self.assertEqual(ceilings, sorted(ceilings))
        self.assertEqual(len(ceilings), len(set(ceilings)))
        covered = np.concatenate([rows for _, rows in blocks])
        self.assertEqual(covered.size, index["origin_index"].size)
        self.assertEqual(np.unique(covered).size, index["origin_index"].size)
        for ceiling, rows in blocks:
            self.assertTrue((index["origin_index"][rows] <= ceiling).all())


class TestChronologicalReveal(unittest.TestCase):
    """The masking layer is exact below the ceiling and fatal above it."""

    @classmethod
    def setUpClass(cls):
        cls.series = StationSeries(PROCESSED, STATION)
        cls.rolling = development_rolling()
        cls.calendar = np.load(str(PROCESSED / "calendar"
                                   / "calendar_cyclic.npy"))
        cls.origin = index_of(datetime(2015, 9, 1, 0))
        cls.target = cls.origin + 24

    def _matrix(self, ceiling, corrupt=False):
        pm25 = np.array(self.rolling._scaled, dtype=np.float32)
        series = self.series
        if corrupt:
            pm25 = pm25.copy()
            pm25[self.origin + 1:] = 9999.0
            series = _CorruptedSeries(self.series, self.origin)
        masked = MaskedStationInputs(series, pm25, ceiling)
        source = StationFeatureSource(masked, self.calendar,
                                      masked.pm25_scaled)
        return build_matrix("R2", source, STATION,
                            np.array([self.origin]), np.array([self.target]))

    def test_no_influence_from_observations_after_the_origin(self):
        """Corrupting every hour after the origin changes nothing at it."""
        clean = self._matrix(index_of(VALIDATION_END))
        corrupted = self._matrix(index_of(VALIDATION_END), corrupt=True)
        self.assertTrue(np.array_equal(clean, corrupted))

    def test_masking_at_the_origin_is_numerically_exact(self):
        """A ceiling at the origin reproduces the unmasked row exactly."""
        wide = self._matrix(index_of(VALIDATION_END))
        tight = self._matrix(self.origin)
        self.assertTrue(np.array_equal(wide, tight))

    def test_a_row_built_past_the_ceiling_is_rejected(self):
        masked = MaskedStationInputs(self.series, self.rolling._scaled,
                                     self.origin)
        source = StationFeatureSource(masked, self.calendar,
                                      masked.pm25_scaled)
        beyond = np.array([self.origin + 10])
        matrix = build_matrix("R2", source, STATION, beyond, beyond + 24)
        with self.assertRaises(FinalTestAccessError):
            verify_matrix(matrix, "beyond the ceiling")

    def test_neural_window_past_the_ceiling_is_non_finite(self):
        masked = MaskedStationInputs(self.series, self.rolling._scaled,
                                     self.origin)
        dynamic = build_station_dynamic_matrix("R1", masked,
                                               masked.pm25_scaled)[None, :, :]
        clean = gather_windows(dynamic, np.array([0]), np.array([self.origin]))
        self.assertTrue(np.isfinite(clean).all())
        dirty = gather_windows(dynamic, np.array([0]),
                               np.array([self.origin + 5]))
        self.assertFalse(np.isfinite(dirty).all())

    def test_wind_one_hot_is_exactly_one_below_the_ceiling(self):
        masked = MaskedStationInputs(self.series, self.rolling._scaled,
                                     self.origin)
        dynamic = build_station_dynamic_matrix("R1", masked,
                                               masked.pm25_scaled)
        row = dynamic[self.origin, wind_onehot_slice("R1")]
        self.assertAlmostEqual(float(row.sum()), 1.0, places=6)


class _CorruptedSeries(object):
    """A station series whose every channel after `origin` is destroyed."""

    def __init__(self, series, origin):
        cut = origin + 1
        self.station = series.station
        self.numeric_channels = series.numeric_channels
        self.mask_channels = series.mask_channels
        self.gap_age_channels = series.gap_age_channels
        self.numeric = np.array(series.numeric, dtype=np.float32)
        self.numeric[cut:, :] = 9999.0
        self.observed = np.array(series.observed)
        self.observed[cut:, :] = ~self.observed[cut:, :]
        self.gap_age = np.array(series.gap_age)
        self.gap_age[cut:, :] = 123
        self.wd_code = np.array(series.wd_code)
        self.wd_code[cut:] = 3
        self.rain_occurred = np.array(series.rain_occurred)
        self.rain_occurred[cut:] = ~np.asarray(self.rain_occurred[cut:],
                                               dtype=bool)


class TestMetricContract(unittest.TestCase):
    """Sections 30, 31 and 33: endpoints are the frozen ones."""

    def setUp(self):
        rng = np.random.default_rng(7)
        self.stations = np.array(sum([[s] * 40 for s in STATIONS], []))
        self.horizons = np.array([1, 6, 12, 24] * 120)
        self.actual = rng.uniform(0.0, 400.0, self.stations.size)
        self.prediction = self.actual + rng.normal(0.0, 12.0,
                                                   self.stations.size)

    def test_residual_convention_is_actual_minus_prediction(self):
        residual = M.residuals(self.actual, self.prediction)
        self.assertTrue(np.allclose(residual,
                                    self.actual - self.prediction))
        self.assertGreater(float(M.residuals([10.0], [4.0])[0]), 0.0)

    def test_macro_station_horizon_mae_reconciles_from_cells(self):
        cells = M.cell_metrics(self.actual, self.prediction, self.stations,
                               self.horizons)
        self.assertEqual(len(cells), 48)
        macro = M.macro_from_cells(cells, "mae")
        manual = np.mean([c["mae"] for c in cells.values()])
        self.assertAlmostEqual(macro, float(manual), places=12)

    def test_macro_is_not_sample_weighted(self):
        """With unequal cell counts, equal weighting must diverge from micro.

        The synthetic fixture above is perfectly balanced, where the two
        averages coincide by construction; the real test universe is not, so
        the divergence is built here deliberately.
        """
        stations = np.array(["A"] * 100 + ["B"] * 4)
        horizons = np.array([1] * 104)
        actual = np.concatenate([np.full(100, 10.0), np.full(4, 10.0)])
        prediction = np.concatenate([np.full(100, 11.0), np.full(4, 50.0)])
        cells = M.cell_metrics(actual, prediction, stations, horizons)
        self.assertEqual(len(cells), 2)
        macro = M.macro_from_cells(cells, "mae")
        micro = M.mae(actual, prediction)
        self.assertAlmostEqual(macro, (1.0 + 40.0) / 2.0, places=12)
        self.assertAlmostEqual(micro, (100 * 1.0 + 4 * 40.0) / 104.0,
                               places=12)
        self.assertGreater(abs(macro - micro), 1.0)

    def test_severe_threshold_is_exactly_244(self):
        self.assertEqual(SEVERE_THRESHOLD, 244.0)
        self.assertEqual(M.SEVERE_THRESHOLD, 244.0)

    def test_severe_rule_is_strictly_greater_than(self):
        actual = np.array([244.0, 244.0001, 243.9])
        prediction = np.zeros(3)
        stations = np.array(["A", "A", "A"])
        horizons = np.array([1, 1, 1])
        out = M.severe_metrics(actual, prediction, stations, horizons)
        self.assertEqual(out["severe_n"], 1)

    def test_no_output_clipping(self):
        prediction = np.array([-30.0, 5.0])
        summary = M.negative_prediction_summary(prediction)
        self.assertEqual(summary["n_negative"], 1)
        self.assertEqual(summary["min_prediction"], -30.0)


class TestScoringOrder(unittest.TestCase):
    """Sections 17 and 29: predictions must exist before actuals are read."""

    def test_oracle_refuses_a_mismatched_prediction_count(self):
        oracle = FinalTestTargetOracle.__new__(FinalTestTargetOracle)
        oracle.station = STATION
        oracle._values = np.zeros(1, dtype=np.float64)
        with self.assertRaises(FinalTestAccessError):
            oracle.actuals(np.array([index_of(TEST_START)]),
                           np.array([1.0, 2.0]))

    def test_oracle_refuses_outside_the_test_window(self):
        oracle = FinalTestTargetOracle.__new__(FinalTestTargetOracle)
        oracle.station = STATION
        oracle._values = np.zeros(index_of(TEST_END) + 1, dtype=np.float64)
        outside = np.array([index_of(VALIDATION_START)])
        with self.assertRaises(FinalTestAccessError):
            oracle.actuals(outside, np.array([1.0]))

    def test_prediction_alignment_is_positional(self):
        index = load_test_index(PROCESSED)
        self.assertEqual(index["sample_id"].size, EXPECTED_TEST_SAMPLES)
        self.assertEqual(index["sample_id"][0].split("|")[0], "test")
        self.assertEqual(np.unique(index["sample_id"]).size,
                         EXPECTED_TEST_SAMPLES)


if __name__ == "__main__":
    unittest.main()
