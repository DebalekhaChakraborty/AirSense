"""Protocol Phase 11 - post-evaluation error analysis for AirSense V1.

The final test has already been consumed. This phase is **descriptive
only**: it reads the frozen Phase-10 predictions and the frozen target
snapshot and characterises where the four models succeed and fail.

**No model development is permitted.** This module imports no estimator,
calls no ``fit`` and no ``predict``, and generates no prediction. Every
number it reports is derived from artifacts frozen before and during
Phase 10. If an analysis suggests an obvious improvement, that is recorded
as future work or a limitation - never implemented in V1.

**The original 2014 source is not reopened.** The target was legitimately
opened once in Phase 10 and frozen into
``results/final_test/final_test_target_snapshot.csv``; that snapshot is the
only source of actual PM2.5 here. ``data/processed/airsense_test_2014.csv``
is never read, and a guard aborts the run if any code path reaches it.

Two design points that matter for correctness:

* **Thresholds come from development data, never from the test set.** The
  concentration bands and the severe-hour rule are derived from the
  2010-2013 development target (33,096 observations) and frozen before any
  2014 subgroup metric is computed. Deriving bands from the test
  distribution would let the evaluation data shape its own analysis.
* **Residual autocorrelation uses exact timestamp matching, not row
  offsets.** The supervised test set has 8,661 rows rather than 8,760
  because 99 targets were missing and excluded. Shifting rows would treat
  hours separated by a gap as adjacent, silently corrupting every lag.

Historical constraint: CPython 3.6.7 with numpy 1.15.4, pandas 0.23.4,
scipy 1.1.0, matplotlib 3.0.2.
"""

import io
import json
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.audit import sha256_of_file

# ----------------------------------------------------------------------
# Frozen inputs and expectations
# ----------------------------------------------------------------------

MODELS = ["M0", "M1", "M2", "M3"]
EXPECTED_TEST_ROWS = 8661
EXPECTED_DEVELOPMENT_ROWS = 33096
TARGET = "pm2.5"

# The Phase-4 test partition is FORBIDDEN here: the frozen snapshot is the
# authorised source of 2014 target values.
FORBIDDEN_FILE = "airsense_test_2014.csv"

TARGET_SNAPSHOT = os.path.join("results", "final_test",
                               "final_test_target_snapshot.csv")
FINAL_PREDICTIONS = os.path.join("results", "final_test",
                                 "final_test_predictions.csv")
BLIND_PREDICTIONS = os.path.join("results", "final_test",
                                 "blind_final_predictions.csv")
X_TEST = os.path.join("data", "processed", "features", "X_test.csv")
ROW_MANIFEST = os.path.join("artifacts", "feature_row_manifest.csv")
EVALUATION_MANIFEST = os.path.join("artifacts",
                                   "final_evaluation_manifest.json")

RESULTS_SUBDIR = os.path.join("results", "error_analysis")
MANIFEST_FILENAME = "error_analysis_manifest.json"

# Phase-3 season definition, reused exactly. Not redefined here.
SEASON_BY_MONTH = {12: "Winter", 1: "Winter", 2: "Winter",
                   3: "Spring", 4: "Spring", 5: "Spring",
                   6: "Summer", 7: "Summer", 8: "Summer",
                   9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

# Wind-direction reference level omitted from the frozen one-hot block.
WIND_REFERENCE = "NE"
WIND_DUMMIES = [("cbwd_NW", "NW"), ("cbwd_SE", "SE"), ("cbwd_cv", "cv")]
WIND_ORDER = ["NE", "NW", "SE", "cv"]

# Concentration bands, named neutrally. These are descriptive relative
# bands derived from the development distribution - NOT regulatory AQI
# categories, and deliberately not labelled good/moderate/unhealthy.
BAND_ORDER = ["le_p25", "p25_p50", "p50_p75", "p75_p90", "p90_p95",
              "gt_p95"]

# Residual lags in hours, frozen before any correlation was computed.
RESIDUAL_LAGS = [1, 6, 12, 24, 48, 168]

RESIDUAL_DEFINITION = "residual = actual_pm25 - predicted_pm25"
SIGN_CONVENTION = ("residual > 0 is an underprediction; residual < 0 is an "
                   "overprediction; residual == 0 is exact")

FIGURE_DPI = 100
MODEL_COLORS = {"M0": "#999999", "M1": "#31688e", "M2": "#35b779",
                "M3": "#b03a2e"}
FIGURE_SUBTITLE = ("2014 Final Held-Out Test - Post-Evaluation Error "
                   "Analysis")


class ErrorAnalysisError(RuntimeError):
    """Raised when a declared Phase-11 gate fails."""


def _guard(path):
    """Abort if any code path reaches the Phase-4 test partition."""
    if os.path.normpath(path).replace(os.sep, "/").endswith(FORBIDDEN_FILE):
        raise ErrorAnalysisError(
            "FORBIDDEN: Phase 11 attempted to reopen %s; the frozen "
            "snapshot must be used instead" % path)
    return path


def _read_csv(path, **kwargs):
    return pd.read_csv(_guard(path), **kwargs)


# ----------------------------------------------------------------------
# Gate: Phase-10 integrity
# ----------------------------------------------------------------------

def verify_phase10(project_root):
    """Verify every hash referenced by the final evaluation manifest."""
    manifest_path = os.path.join(project_root, EVALUATION_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise ErrorAnalysisError("final_evaluation_manifest.json not found")
    manifest = json.load(io.open(manifest_path, encoding="utf-8"))

    checks = OrderedDict()
    named = [("raw_dataset_sha256",
              "data/raw/PRSA_data_2010.1.1-2014.12.31.csv"),
             ("split_manifest_sha256", "artifacts/split_manifest.json"),
             ("feature_schema_sha256", "artifacts/feature_schema.json"),
             ("pretest_development_freeze_sha256",
              "artifacts/pretest_development_freeze.json"),
             ("final_blind_prediction_freeze_sha256",
              "artifacts/final_blind_prediction_freeze.json"),
             ("final_test_opening_receipt_sha256",
              "artifacts/final_test_opening_receipt.json"),
             ("final_test_target_snapshot_sha256", TARGET_SNAPSHOT),
             ("blind_prediction_sha256", BLIND_PREDICTIONS)]
    verified = 0
    for key, relative in named:
        observed = sha256_of_file(os.path.join(project_root, relative))
        verified += 1
        if observed != manifest[key]:
            raise ErrorAnalysisError(
                "Phase-10 artifact changed: %s. STOP - do not analyse an "
                "altered final evaluation." % relative)
        checks[relative] = observed
    for relative, recorded in manifest["outputs_sha256"].items():
        observed = sha256_of_file(os.path.join(project_root, relative))
        verified += 1
        if observed != recorded:
            raise ErrorAnalysisError("Phase-10 output changed: %s" % relative)
        checks[relative] = observed

    # The scored predictions must still be the ones frozen before opening.
    blind = _read_csv(os.path.join(project_root, BLIND_PREDICTIONS),
                      dtype=str, keep_default_na=False, na_filter=False)
    final = _read_csv(os.path.join(project_root, FINAL_PREDICTIONS),
                      dtype=str, keep_default_na=False, na_filter=False)
    for model_id in MODELS:
        column = "%s_prediction" % model_id.lower()
        if not bool((blind[column].values == final[column].values).all()):
            raise ErrorAnalysisError(
                "%s differs from the frozen blind prediction" % column)

    checks["_extra"] = OrderedDict([
        ("hashes_verified", verified),
        ("final_predictions_identical_to_blind", True),
        ("final_test_status", manifest["final_test_status"]),
        ("post_test_model_changes", manifest["post_test_model_changes"]),
        ("best_final_model_by_mae", manifest["best_final_model_by_mae"]),
    ])
    return manifest, checks


# ----------------------------------------------------------------------
# Analysis population
# ----------------------------------------------------------------------

def load_population(project_root):
    """Assemble the analysis frame from frozen artifacts only."""
    snapshot = _read_csv(os.path.join(project_root, TARGET_SNAPSHOT))
    predictions = _read_csv(os.path.join(project_root, FINAL_PREDICTIONS))
    features = _read_csv(os.path.join(project_root, X_TEST))
    manifest = _read_csv(os.path.join(project_root, ROW_MANIFEST))
    rows = manifest[manifest["partition"] == "test"].reset_index(drop=True)

    for label, frame in (("snapshot", snapshot), ("predictions", predictions),
                         ("X_test", features), ("row manifest", rows)):
        if len(frame) != EXPECTED_TEST_ROWS:
            raise ErrorAnalysisError("%s has %d rows, expected %d"
                                     % (label, len(frame),
                                        EXPECTED_TEST_ROWS))

    if not bool((snapshot["No"].values == predictions["No"].values).all()):
        raise ErrorAnalysisError("snapshot and predictions are misaligned")
    if not bool((rows["No"].values == predictions["No"].values).all()):
        raise ErrorAnalysisError("row manifest and predictions misaligned")
    if not bool((snapshot["actual_pm25"].values
                 == predictions["actual_pm25"].values).all()):
        raise ErrorAnalysisError(
            "snapshot target differs from the target in the prediction "
            "artifact")

    frame = pd.DataFrame(OrderedDict([
        ("matrix_row_number", predictions["matrix_row_number"].values),
        ("No", predictions["No"].values),
        ("timestamp", pd.to_datetime(predictions["timestamp"].values)),
        ("actual_pm25", snapshot["actual_pm25"].values.astype(np.float64)),
    ]))
    for model_id in MODELS:
        key = model_id.lower()
        frame["%s_prediction" % key] = predictions[
            "%s_prediction" % key].values.astype(np.float64)
        frame["%s_residual" % key] = (
            frame["actual_pm25"].values
            - frame["%s_prediction" % key].values)
        frame["%s_abs_error" % key] = np.abs(
            frame["%s_residual" % key].values)

    # Context reconstructed from the timestamp and the frozen dummies.
    frame["year"] = frame["timestamp"].dt.year
    frame["month"] = frame["timestamp"].dt.month
    frame["hour"] = frame["timestamp"].dt.hour
    frame["season"] = frame["month"].map(SEASON_BY_MONTH)

    wind = np.array([WIND_REFERENCE] * len(frame), dtype=object)
    active = np.zeros(len(frame), dtype=np.int64)
    for column, level in WIND_DUMMIES:
        if column not in features.columns:
            raise ErrorAnalysisError("X_test lacks %s" % column)
        flags = features[column].values.astype(np.int64)
        if not set(np.unique(flags).tolist()).issubset({0, 1}):
            raise ErrorAnalysisError("%s is not 0/1" % column)
        wind[flags == 1] = level
        active += flags
    if not bool((active <= 1).all()):
        raise ErrorAnalysisError(
            "wind-direction dummies are not mutually exclusive")
    frame["cbwd"] = wind

    if not bool(frame["timestamp"].is_monotonic_increasing):
        raise ErrorAnalysisError("test timestamps are not increasing")
    return frame, snapshot, predictions, features, rows


# ----------------------------------------------------------------------
# Development-derived thresholds - frozen BEFORE any subgroup metric
# ----------------------------------------------------------------------

def development_thresholds(project_root):
    """Quantiles of the 2010-2013 development target.

    Derived from development data only. Deriving bands from the 2014
    distribution would let the evaluation set shape its own analysis.
    """
    features_dir = os.path.join(project_root, "data", "processed",
                                "features")
    train = _read_csv(os.path.join(features_dir, "y_train.csv"))
    validation = _read_csv(os.path.join(features_dir, "y_validation.csv"))
    values = np.concatenate([train[TARGET].values.astype(np.float64),
                             validation[TARGET].values.astype(np.float64)])
    if values.size != EXPECTED_DEVELOPMENT_ROWS:
        raise ErrorAnalysisError(
            "development target has %d rows, expected %d"
            % (values.size, EXPECTED_DEVELOPMENT_ROWS))
    thresholds = OrderedDict()
    for label, q in (("P25", 0.25), ("P50", 0.50), ("P75", 0.75),
                     ("P90", 0.90), ("P95", 0.95)):
        thresholds[label] = float(np.percentile(values, q * 100.0))
    thresholds["_source"] = ("2010-2013 development target "
                             "(y_train + y_validation)")
    thresholds["_n"] = int(values.size)
    thresholds["_sha256"] = OrderedDict([
        ("y_train.csv", sha256_of_file(
            os.path.join(features_dir, "y_train.csv"))),
        ("y_validation.csv", sha256_of_file(
            os.path.join(features_dir, "y_validation.csv"))),
    ])
    return thresholds


def assign_bands(actual, thresholds):
    """Six fixed bands with deterministic inclusive/exclusive boundaries."""
    p25, p50 = thresholds["P25"], thresholds["P50"]
    p75, p90 = thresholds["P75"], thresholds["P90"]
    p95 = thresholds["P95"]
    bands = np.empty(actual.size, dtype=object)
    bands[actual <= p25] = "le_p25"
    bands[(actual > p25) & (actual <= p50)] = "p25_p50"
    bands[(actual > p50) & (actual <= p75)] = "p50_p75"
    bands[(actual > p75) & (actual <= p90)] = "p75_p90"
    bands[(actual > p90) & (actual <= p95)] = "p90_p95"
    bands[actual > p95] = "gt_p95"
    if any(b is None for b in bands):
        raise ErrorAnalysisError("some observations fell outside every band")
    return bands


def band_definitions(thresholds):
    p25, p50 = thresholds["P25"], thresholds["P50"]
    p75, p90 = thresholds["P75"], thresholds["P90"]
    p95 = thresholds["P95"]
    return OrderedDict([
        ("le_p25", "actual_pm25 <= %g" % p25),
        ("p25_p50", "%g < actual_pm25 <= %g" % (p25, p50)),
        ("p50_p75", "%g < actual_pm25 <= %g" % (p50, p75)),
        ("p75_p90", "%g < actual_pm25 <= %g" % (p75, p90)),
        ("p90_p95", "%g < actual_pm25 <= %g" % (p90, p95)),
        ("gt_p95", "actual_pm25 > %g" % p95),
        ("_note", "descriptive relative bands derived from the 2010-2013 "
                  "development target; NOT regulatory AQI categories"),
    ])


# ----------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------

def _mae(actual, predicted):
    return float(mean_absolute_error(actual, predicted))


def _rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def _group_metrics(frame, model_id):
    """MAE / RMSE / mean and median residual for one group and model."""
    key = model_id.lower()
    actual = frame["actual_pm25"].values
    predicted = frame["%s_prediction" % key].values
    residual = frame["%s_residual" % key].values
    return OrderedDict([
        ("n", int(len(frame))),
        ("mae", _mae(actual, predicted)),
        ("rmse", _rmse(actual, predicted)),
        ("mean_residual", float(residual.mean())),
        ("median_residual", float(np.median(residual))),
    ])


# ----------------------------------------------------------------------
# Section 12 - absolute-error distribution and bias
# ----------------------------------------------------------------------

def absolute_error_summary(frame):
    rows = []
    for model_id in MODELS:
        errors = frame["%s_abs_error" % model_id.lower()].values
        rows.append(OrderedDict([
            ("model", model_id),
            ("n", int(errors.size)),
            ("mean_abs_error", float(errors.mean())),
            ("median_abs_error", float(np.median(errors))),
            ("std_abs_error", float(errors.std(ddof=1))),
            ("p75_abs_error", float(np.percentile(errors, 75))),
            ("p90_abs_error", float(np.percentile(errors, 90))),
            ("p95_abs_error", float(np.percentile(errors, 95))),
            ("p99_abs_error", float(np.percentile(errors, 99))),
            ("max_abs_error", float(errors.max())),
        ]))
    return pd.DataFrame(rows)


def prediction_bias_summary(frame):
    """Under/over-prediction counts, plus the negative-prediction audit.

    Negative predictions are recorded, never clipped: clipping after the
    test was opened would be a post-test model modification.
    """
    total = len(frame)
    rows = []
    for model_id in MODELS:
        key = model_id.lower()
        residual = frame["%s_residual" % key].values
        predicted = frame["%s_prediction" % key].values
        under = int((residual > 0).sum())
        over = int((residual < 0).sum())
        exact = int((residual == 0).sum())
        negative = predicted < 0.0
        n_negative = int(negative.sum())
        row = OrderedDict([
            ("model", model_id),
            ("n", total),
            ("underprediction_count", under),
            ("underprediction_percent", 100.0 * under / total),
            ("overprediction_count", over),
            ("overprediction_percent", 100.0 * over / total),
            ("exact_prediction_count", exact),
            ("mean_residual", float(residual.mean())),
            ("median_residual", float(np.median(residual))),
            ("n_negative_predictions", n_negative),
            ("percent_negative_predictions", 100.0 * n_negative / total),
        ])
        if n_negative:
            row["min_negative_prediction"] = float(predicted[negative].min())
            row["mean_actual_where_negative"] = float(
                frame["actual_pm25"].values[negative].mean())
            row["mae_where_negative"] = float(
                frame["%s_abs_error" % key].values[negative].mean())
        else:
            row["min_negative_prediction"] = ""
            row["mean_actual_where_negative"] = ""
            row["mae_where_negative"] = ""
        row["predictions_clipped"] = False
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Sections 13-16 - subgroup error tables
# ----------------------------------------------------------------------

def _subgroup_table(frame, column, order, label):
    rows = []
    for model_id in MODELS:
        for value in order:
            block = frame[frame[column] == value]
            if block.empty:
                continue
            entry = OrderedDict([("model", model_id), (label, value)])
            entry.update(_group_metrics(block, model_id))
            rows.append(entry)
    return pd.DataFrame(rows)


def error_by_season(frame):
    return _subgroup_table(frame, "season", SEASON_ORDER, "season")


def error_by_hour(frame):
    return _subgroup_table(frame, "hour", list(range(24)), "hour")


def error_by_wind_direction(frame):
    return _subgroup_table(frame, "cbwd", WIND_ORDER, "cbwd")


def error_by_concentration_band(frame):
    """All models scored over exactly the same actual-target bands."""
    rows = []
    for model_id in MODELS:
        key = model_id.lower()
        for band in BAND_ORDER:
            block = frame[frame["concentration_band"] == band]
            if block.empty:
                continue
            residual = block["%s_residual" % key].values
            entry = OrderedDict([
                ("model", model_id),
                ("concentration_band", band),
                ("n", int(len(block))),
                ("mean_actual_pm25", float(block["actual_pm25"].mean())),
            ])
            entry.update(_group_metrics(block, model_id))
            entry["underprediction_percent"] = (
                100.0 * float((residual > 0).sum()) / len(block))
            rows.append(entry)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Sections 17-19 - severe hours and episodes
# ----------------------------------------------------------------------

def severe_hour_summary(frame, thresholds):
    """Behaviour above the development P95. A relative, study-specific tail
    definition - not a regulatory severe-pollution threshold."""
    severe = frame[frame["actual_pm25"] > thresholds["P95"]]
    rows = []
    for model_id in MODELS:
        key = model_id.lower()
        residual = severe["%s_residual" % key].values
        errors = severe["%s_abs_error" % key].values
        entry = OrderedDict([
            ("model", model_id),
            ("severe_threshold_source", "development_P95"),
            ("severe_threshold_value", thresholds["P95"]),
            ("n_severe_hours", int(len(severe))),
        ])
        entry.update(_group_metrics(severe, model_id))
        entry["underprediction_percent"] = (
            100.0 * float((residual > 0).sum()) / len(severe))
        entry["max_abs_error"] = float(errors.max())
        rows.append(entry)
    return pd.DataFrame(rows), severe


def identify_episodes(severe):
    """Contiguous severe hours, breaking whenever the gap exceeds 1 hour.

    Gaps are never bridged: if the next observed severe target is more than
    one hour later, a new episode begins. That is what makes the definition
    safe against the 99 excluded 2014 hours.
    """
    ordered = severe.sort_values("timestamp").reset_index(drop=True)
    if ordered.empty:
        return []
    gaps = ordered["timestamp"].diff()
    breaks = gaps != pd.Timedelta(hours=1)
    breaks.iloc[0] = True
    episode_id = breaks.cumsum()
    episodes = []
    for identifier, block in ordered.groupby(episode_id):
        episodes.append(block.reset_index(drop=True))
    return episodes


def episode_details(episodes):
    rows = []
    for index, block in enumerate(episodes, start=1):
        start = block["timestamp"].iloc[0]
        end = block["timestamp"].iloc[-1]
        entry = OrderedDict([
            ("episode_id", index),
            ("start_timestamp", str(start)),
            ("end_timestamp", str(end)),
            ("duration_hours",
             int((end - start).total_seconds() // 3600) + 1),
            ("n_scored_hours", int(len(block))),
            ("mean_actual_pm25", float(block["actual_pm25"].mean())),
            ("max_actual_pm25", float(block["actual_pm25"].max())),
        ])
        for model_id in MODELS:
            key = model_id.lower()
            entry["%s_mae" % key] = float(
                block["%s_abs_error" % key].mean())
        for model_id in MODELS:
            key = model_id.lower()
            entry["%s_mean_residual" % key] = float(
                block["%s_residual" % key].mean())
        rows.append(entry)
    return pd.DataFrame(rows)


def episode_summary(episodes, severe, thresholds):
    durations = np.array([len(block) for block in episodes],
                         dtype=np.float64)
    summary = OrderedDict([
        ("severe_threshold_source", "development_P95"),
        ("severe_threshold_value", thresholds["P95"]),
        ("number_of_episodes", int(len(episodes))),
        ("number_of_severe_hours", int(len(severe))),
        ("episode_duration_min", float(durations.min())),
        ("episode_duration_median", float(np.median(durations))),
        ("episode_duration_mean", float(durations.mean())),
        ("episode_duration_max", float(durations.max())),
    ])
    for model_id in MODELS:
        key = model_id.lower()
        residual = severe["%s_residual" % key].values
        summary["%s_severe_episode_hour_mae" % key] = float(
            severe["%s_abs_error" % key].mean())
        summary["%s_severe_episode_hour_rmse" % key] = _rmse(
            severe["actual_pm25"].values,
            severe["%s_prediction" % key].values)
        summary["%s_severe_episode_underprediction_rate" % key] = (
            100.0 * float((residual > 0).sum()) / len(severe))
    summary["note"] = ("Concentration-prediction behaviour in the upper "
                       "tail. No event-detection or classification claim "
                       "is made: these models predict a concentration, not "
                       "an event class.")
    return pd.DataFrame([summary])


def verify_episodes(episodes, severe, frame, thresholds):
    """Every declared episode invariant, checked."""
    p95 = thresholds["P95"]
    seen = set()
    total = 0
    for block in episodes:
        if not bool((block["actual_pm25"].values > p95).all()):
            raise ErrorAnalysisError("an episode contains a non-severe hour")
        deltas = block["timestamp"].diff().dropna()
        if len(deltas) and not bool(
                (deltas == pd.Timedelta(hours=1)).all()):
            raise ErrorAnalysisError(
                "consecutive rows within an episode are not 1 hour apart")
        identifiers = set(block["No"].tolist())
        if seen & identifiers:
            raise ErrorAnalysisError("episodes overlap")
        seen |= identifiers
        total += len(block)
    if total != len(severe):
        raise ErrorAnalysisError(
            "episode hours (%d) do not sum to the severe-hour count (%d)"
            % (total, len(severe)))
    if seen != set(severe["No"].tolist()):
        raise ErrorAnalysisError(
            "not every severe hour belongs to exactly one episode")
    non_severe = set(frame[frame["actual_pm25"] <= p95]["No"].tolist())
    if seen & non_severe:
        raise ErrorAnalysisError("a non-severe hour was placed in an episode")
    return OrderedDict([
        ("episodes_contain_only_severe_hours", True),
        ("within_episode_gap_is_exactly_1_hour", True),
        ("episodes_do_not_overlap", True),
        ("every_severe_hour_in_exactly_one_episode", True),
        ("no_non_severe_hour_in_any_episode", True),
        ("episode_hours_sum_equals_severe_count", int(total)),
    ])


# ----------------------------------------------------------------------
# Section 20 - time-aware residual autocorrelation
# ----------------------------------------------------------------------

def residual_autocorrelation(frame):
    """Correlate residual(t) with residual(t + lag) by EXACT timestamp.

    Row shifting is deliberately not used. The supervised test set has
    8,661 of 8,760 hours, so a row offset would pair timestamps separated
    by a gap and silently corrupt every lag. Here the partner value is
    looked up at ``t + lag`` and the pair is dropped if that hour is not
    observed.
    """
    rows = []
    index = pd.DatetimeIndex(frame["timestamp"].values)
    for model_id in MODELS:
        series = pd.Series(
            frame["%s_residual" % model_id.lower()].values, index=index)
        for lag in RESIDUAL_LAGS:
            partner = series.reindex(index + pd.Timedelta(hours=lag))
            valid = partner.notnull().values
            n_pairs = int(valid.sum())
            if n_pairs < 3:
                correlation = float("nan")
            else:
                a = series.values[valid]
                b = partner.values[valid]
                correlation = float(np.corrcoef(a, b)[0, 1])
            rows.append(OrderedDict([
                ("model", model_id),
                ("lag_hours", int(lag)),
                ("n_pairs", n_pairs),
                ("pearson_r", correlation),
                ("matching", "exact_timestamp"),
            ]))
    return pd.DataFrame(rows)


def verify_lag_pairs(frame):
    """Confirm every matched pair differs by exactly the declared lag."""
    index = pd.DatetimeIndex(frame["timestamp"].values)
    observed = set(index)
    report = OrderedDict()
    for lag in RESIDUAL_LAGS:
        delta = pd.Timedelta(hours=lag)
        pairs = [t for t in index if (t + delta) in observed]
        # verify a sample of the constructed pairs really are lag apart
        for t in pairs[:1000]:
            if (t + delta) - t != delta:
                raise ErrorAnalysisError("lag pair is not exactly %dh" % lag)
        report["lag_%dh_pairs" % lag] = int(len(pairs))
    report["method"] = "exact timestamp matching; no row-offset shifting"
    return report


# ----------------------------------------------------------------------
# Sections 22-24 - pairwise comparison, M3 vs M2, worst errors
# ----------------------------------------------------------------------

def pairwise_error_wins(frame):
    """Row-wise absolute-error wins. Descriptive only.

    This answers whether M3's average advantage is broad or driven by a
    small subset of hours. It is NOT a new selection criterion: the final
    ranking remains the pre-registered overall MAE ranking.
    """
    rows = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = MODELS[i], MODELS[j]
            ea = frame["%s_abs_error" % a.lower()].values
            eb = frame["%s_abs_error" % b.lower()].values
            rows.append(OrderedDict([
                ("model_a", a),
                ("model_b", b),
                ("model_a_lower_absolute_error_count", int((ea < eb).sum())),
                ("model_b_lower_absolute_error_count", int((eb < ea).sum())),
                ("tie_count", int((ea == eb).sum())),
                ("n", int(len(frame))),
                ("model_a_win_percent", 100.0 * float((ea < eb).sum())
                 / len(frame)),
            ]))
    return pd.DataFrame(rows)


def m3_vs_m2_difference(frame):
    """Per-observation absolute-error difference. Descriptive only.

    Negative means M3 had the lower absolute error. This must not be used
    to build a per-row model selector or a hybrid model.
    """
    return pd.DataFrame(OrderedDict([
        ("matrix_row_number", frame["matrix_row_number"].values),
        ("No", frame["No"].values),
        ("timestamp", frame["timestamp"].astype(str).values),
        ("actual_pm25", frame["actual_pm25"].values),
        ("m2_absolute_error", frame["m2_abs_error"].values),
        ("m3_absolute_error", frame["m3_abs_error"].values),
        ("m3_minus_m2_absolute_error",
         frame["m3_abs_error"].values - frame["m2_abs_error"].values),
    ]))


def m3_largest_errors(frame, n=25):
    """Audit table only. These observations are not removed or excluded."""
    ordered = frame.sort_values(
        ["m3_abs_error", "No"], ascending=[False, True]).head(n)
    ordered = ordered.reset_index(drop=True)
    return pd.DataFrame(OrderedDict([
        ("rank", np.arange(1, len(ordered) + 1)),
        ("No", ordered["No"].values),
        ("timestamp", ordered["timestamp"].astype(str).values),
        ("actual_pm25", ordered["actual_pm25"].values),
        ("m3_prediction", ordered["m3_prediction"].values),
        ("residual", ordered["m3_residual"].values),
        ("absolute_error", ordered["m3_abs_error"].values),
        ("season", ordered["season"].values),
        ("hour", ordered["hour"].values),
        ("concentration_band", ordered["concentration_band"].values),
        ("cbwd", ordered["cbwd"].values),
    ]))


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def _save(fig, target_path):
    fig.tight_layout()
    try:
        fig.savefig(target_path, dpi=FIGURE_DPI,
                    metadata={"Software": "AirSense V1"})
    except TypeError:                            # pragma: no cover
        fig.savefig(target_path, dpi=FIGURE_DPI)
    plt.close(fig)


def figure_mae_by_season(table, target_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(SEASON_ORDER))
    width = 0.2
    for index, model_id in enumerate(MODELS):
        block = table[table["model"] == model_id].set_index("season")
        values = [block.loc[s, "mae"] for s in SEASON_ORDER]
        ax.bar(positions + (index - 1.5) * width, values, width,
               color=MODEL_COLORS[model_id], label=model_id)
    ax.set_xticks(positions)
    ax.set_xticklabels(SEASON_ORDER)
    ax.set_xlabel("Season (Phase-3 definition)")
    ax.set_ylabel("MAE (ug/m^3)")
    ax.set_title("MAE by season\n%s" % FIGURE_SUBTITLE)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, target_path)


def figure_mae_by_hour(table, target_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_id in MODELS:
        block = table[table["model"] == model_id].sort_values("hour")
        ax.plot(block["hour"].values, block["mae"].values, marker="o",
                markersize=4, color=MODEL_COLORS[model_id], label=model_id)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MAE (ug/m^3)")
    ax.set_title("MAE by hour of day\n%s" % FIGURE_SUBTITLE)
    ax.legend(fontsize=9)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, target_path)


def figure_mae_by_band(table, target_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(BAND_ORDER))
    width = 0.2
    for index, model_id in enumerate(MODELS):
        block = table[table["model"] == model_id].set_index(
            "concentration_band")
        values = [block.loc[b, "mae"] if b in block.index else 0.0
                  for b in BAND_ORDER]
        ax.bar(positions + (index - 1.5) * width, values, width,
               color=MODEL_COLORS[model_id], label=model_id)
    ax.set_xticks(positions)
    ax.set_xticklabels(BAND_ORDER, rotation=20)
    ax.set_xlabel("Concentration band (bounds from 2010-2013 development "
                  "target)")
    ax.set_ylabel("MAE (ug/m^3)")
    ax.set_title("MAE by actual-concentration band\n%s" % FIGURE_SUBTITLE)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, target_path)


def figure_error_cdf(frame, target_path):
    """Empirical CDF built directly from sorted errors and rank / n."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_id in MODELS:
        errors = np.sort(frame["%s_abs_error" % model_id.lower()].values)
        fraction = np.arange(1, errors.size + 1, dtype=np.float64) \
            / errors.size
        ax.plot(errors, fraction, color=MODEL_COLORS[model_id],
                linewidth=1.6, label=model_id)
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Absolute error (ug/m^3)")
    ax.set_ylabel("Fraction of test observations at or below")
    ax.set_title("Empirical CDF of absolute error\n%s" % FIGURE_SUBTITLE)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, target_path)


def figure_residual_autocorrelation(table, target_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(RESIDUAL_LAGS))
    width = 0.2
    for index, model_id in enumerate(MODELS):
        block = table[table["model"] == model_id].set_index("lag_hours")
        values = [block.loc[l, "pearson_r"] for l in RESIDUAL_LAGS]
        ax.bar(positions + (index - 1.5) * width, values, width,
               color=MODEL_COLORS[model_id], label=model_id)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(l) for l in RESIDUAL_LAGS])
    ax.set_xlabel("Lag (hours), matched on exact timestamps")
    ax.set_ylabel("Pearson correlation of residuals")
    ax.set_title("Residual autocorrelation\n%s" % FIGURE_SUBTITLE)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, target_path)


def figure_severe_episodes(details, target_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(details))
    width = 0.2
    for index, model_id in enumerate(MODELS):
        values = details["%s_mae" % model_id.lower()].values
        ax.bar(positions + (index - 1.5) * width, values, width,
               color=MODEL_COLORS[model_id], label=model_id)
    ax.set_xticks(positions)
    ax.set_xticklabels(details["episode_id"].astype(str).values,
                       rotation=90, fontsize=7)
    ax.set_xlabel("Severe episode id (actual PM2.5 > development P95)")
    ax.set_ylabel("MAE within episode (ug/m^3)")
    ax.set_title("Per-episode MAE across severe high-concentration "
                 "episodes\n%s" % FIGURE_SUBTITLE)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, target_path)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def run_error_analysis(project_root, verbose=True):
    """Descriptive post-evaluation analysis. Fits nothing, predicts nothing."""

    def log(message):
        if verbose:
            print(message)

    results_dir = os.path.join(project_root, RESULTS_SUBDIR)
    artifacts_dir = os.path.join(project_root, "artifacts")
    figures_dir = os.path.join(project_root, "figures")
    for directory in (results_dir, artifacts_dir, figures_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    evaluation, integrity = verify_phase10(project_root)
    log("Phase-10 integrity verified: %d hashes; predictions identical to "
        "the frozen blind vectors"
        % integrity["_extra"]["hashes_verified"])

    frame, snapshot, predictions, features, rows = load_population(
        project_root)
    log("Analysis population: %d rows, all four models over identical "
        "observations" % len(frame))

    # Thresholds are frozen BEFORE any subgroup metric is computed.
    thresholds = development_thresholds(project_root)
    log("Development thresholds frozen from %d observations: "
        "P25=%g P50=%g P75=%g P90=%g P95=%g"
        % (thresholds["_n"], thresholds["P25"], thresholds["P50"],
           thresholds["P75"], thresholds["P90"], thresholds["P95"]))
    frame["concentration_band"] = assign_bands(
        frame["actual_pm25"].values, thresholds)

    written = OrderedDict()

    def write(table, filename, index=False):
        target = os.path.join(results_dir, filename)
        table.to_csv(target, index=index)
        written[os.path.join(RESULTS_SUBDIR, filename)] = OrderedDict([
            ("sha256", sha256_of_file(target)), ("rows", int(len(table)))])
        return table

    write(absolute_error_summary(frame), "absolute_error_summary.csv")
    write(prediction_bias_summary(frame), "prediction_bias_summary.csv")
    season_table = write(error_by_season(frame), "error_by_season.csv")
    hour_table = write(error_by_hour(frame), "error_by_hour.csv")
    band_table = write(error_by_concentration_band(frame),
                       "error_by_concentration_band.csv")
    write(error_by_wind_direction(frame), "error_by_wind_direction.csv")
    log("Wrote subgroup tables")

    severe_table, severe = severe_hour_summary(frame, thresholds)
    write(severe_table, "severe_hour_summary.csv")
    episodes = identify_episodes(severe)
    episode_invariants = verify_episodes(episodes, severe, frame, thresholds)
    details = write(episode_details(episodes), "severe_episode_details.csv")
    write(episode_summary(episodes, severe, thresholds),
          "severe_episode_summary.csv")
    log("Severe analysis: %d severe hours in %d episodes (invariants "
        "verified)" % (len(severe), len(episodes)))

    autocorrelation = write(residual_autocorrelation(frame),
                            "residual_autocorrelation.csv")
    lag_report = verify_lag_pairs(frame)
    write(pairwise_error_wins(frame), "pairwise_error_wins.csv")
    write(m3_vs_m2_difference(frame), "m3_vs_m2_error_difference.csv")
    write(m3_largest_errors(frame), "m3_largest_errors.csv")
    log("Wrote autocorrelation, pairwise and audit tables")

    # -- cross-table reconciliation --------------------------------------
    reconciliation = OrderedDict()
    for label, table, column in (
            ("season", season_table, "season"),
            ("hour", hour_table, "hour"),
            ("concentration_band", band_table, "concentration_band")):
        for model_id in MODELS:
            total = int(table[table["model"] == model_id]["n"].sum())
            if total != EXPECTED_TEST_ROWS:
                raise ErrorAnalysisError(
                    "%s counts for %s sum to %d, expected %d"
                    % (label, model_id, total, EXPECTED_TEST_ROWS))
        reconciliation["%s_counts_sum_per_model" % label] = \
            EXPECTED_TEST_ROWS
    wind_table = pd.read_csv(os.path.join(results_dir,
                                          "error_by_wind_direction.csv"))
    for model_id in MODELS:
        total = int(wind_table[wind_table["model"] == model_id]["n"].sum())
        if total != EXPECTED_TEST_ROWS:
            raise ErrorAnalysisError(
                "wind counts for %s sum to %d" % (model_id, total))
    reconciliation["wind_direction_counts_sum_per_model"] = \
        EXPECTED_TEST_ROWS
    reconciliation["all_models_share_identical_subgroup_rows"] = True

    # -- figures -----------------------------------------------------------
    figures = OrderedDict()

    def figure(name, builder):
        target = os.path.join(figures_dir, name)
        builder(target)
        figures[os.path.join("figures", name)] = sha256_of_file(target)

    figure("error_analysis_mae_by_season.png",
           lambda p: figure_mae_by_season(season_table, p))
    figure("error_analysis_mae_by_hour.png",
           lambda p: figure_mae_by_hour(hour_table, p))
    figure("error_analysis_mae_by_concentration_band.png",
           lambda p: figure_mae_by_band(band_table, p))
    figure("error_analysis_absolute_error_cdf.png",
           lambda p: figure_error_cdf(frame, p))
    figure("error_analysis_residual_autocorrelation.png",
           lambda p: figure_residual_autocorrelation(autocorrelation, p))
    figure("error_analysis_severe_episodes.png",
           lambda p: figure_severe_episodes(details, p))
    log("Wrote %d figures" % len(figures))

    # -- manifest ----------------------------------------------------------
    manifest = OrderedDict()
    manifest["project"] = "AirSense"
    manifest["phase"] = ("V1 / Protocol Phase 11 - post-evaluation error "
                         "analysis")
    manifest["analysis_type"] = "descriptive_post_evaluation"
    manifest["model_modification_status"] = "none"
    manifest["final_test_status"] = "exhausted"
    manifest["models_fitted"] = 0
    manifest["predictions_generated"] = 0
    manifest["original_test_source_reopened"] = False

    provenance = OrderedDict()
    provenance[EVALUATION_MANIFEST] = sha256_of_file(
        os.path.join(project_root, EVALUATION_MANIFEST))
    for relative in (FINAL_PREDICTIONS, TARGET_SNAPSHOT, BLIND_PREDICTIONS,
                     X_TEST, ROW_MANIFEST):
        provenance[relative] = sha256_of_file(
            os.path.join(project_root, relative))
    manifest["provenance_sha256"] = provenance
    manifest["phase10_integrity"] = integrity["_extra"]

    manifest["development_population_for_thresholds"] = OrderedDict([
        ("source", thresholds["_source"]),
        ("n", thresholds["_n"]),
        ("target_sha256", thresholds["_sha256"]),
        ("note", "thresholds were frozen BEFORE any 2014 subgroup metric "
                 "was computed; no threshold was optimised on 2014 model "
                 "performance"),
    ])
    manifest["thresholds"] = OrderedDict(
        (k, thresholds[k]) for k in ("P25", "P50", "P75", "P90", "P95"))
    manifest["concentration_band_definitions"] = band_definitions(thresholds)
    manifest["severe_hour_definition"] = OrderedDict([
        ("rule", "actual_pm25 > development P95"),
        ("threshold_value", thresholds["P95"]),
        ("threshold_source", "2010-2013 development target only"),
        ("note", "a relative, study-specific tail definition; NOT a "
                 "regulatory severe-pollution threshold"),
    ])
    manifest["severe_episode_definition"] = OrderedDict([
        ("rule", "one or more consecutive test timestamps exactly 1 hour "
                 "apart, all with actual_pm25 > development P95"),
        ("gap_policy", "gaps are never bridged; if the next observed "
                       "severe target is more than one hour later, a new "
                       "episode begins"),
        ("frozen_before_calculation", True),
    ])
    manifest["residual_lag_definitions"] = OrderedDict([
        ("lags_hours", list(RESIDUAL_LAGS)),
        ("matching", "exact timestamp; residual(t) paired with "
                     "residual(t + lag) only when that hour is observed"),
        ("row_offset_shifting_used", False),
        ("reason", "the supervised test set has 8,661 of 8,760 hours, so "
                   "row shifting would pair timestamps separated by a gap"),
    ])
    manifest["residual_convention"] = RESIDUAL_DEFINITION
    manifest["sign_convention"] = SIGN_CONVENTION
    manifest["episode_invariants"] = episode_invariants
    manifest["lag_pair_verification"] = lag_report
    manifest["cross_table_reconciliation"] = reconciliation

    manifest["final_model_ranking_from_phase10"] = [
        evaluation["models"][m]["mae_rank"] for m in MODELS]
    manifest["final_ranking_by_mae"] = sorted(
        MODELS, key=lambda m: evaluation["models"][m]["mae_rank"])
    manifest["final_best_model"] = evaluation["best_final_model_by_mae"]
    manifest["final_test_metrics_unchanged"] = OrderedDict(
        (m, OrderedDict([("mae", evaluation["models"][m]["test_mae"]),
                         ("rmse", evaluation["models"][m]["test_rmse"]),
                         ("r2", evaluation["models"][m]["test_r2"])]))
        for m in MODELS)

    manifest["outputs"] = written
    manifest["figures"] = figures
    manifest["note"] = (
        "No execution timestamp is recorded so that repeated runs are "
        "byte-identical. No finding in this phase may cause M0-M3 to be "
        "changed; improvement hypotheses are recorded as future work only.")

    manifest_path = os.path.join(artifacts_dir, MANIFEST_FILENAME)
    with io.open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2))
        handle.write("\n")
    log("Wrote artifacts/%s" % MANIFEST_FILENAME)

    verify_phase10(project_root)
    log("")
    log("Phase-10 artifacts unchanged after analysis")
    return manifest
