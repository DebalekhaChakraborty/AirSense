"""M0 naive baseline for AirSense V1.

Protocol Phase 6. The first phase in which predictive metrics are
permitted, and the last before M1 Linear Regression.

M0 is a **training-median constant predictor**:

    c = median(y_train)                     fitted on 2010-2012 only
    prediction(row) = c                     for every 2013 validation row

Why the median and not the mean. The primary decision metric was frozen in
Phase 0 as MAE, and for a constant predictor the sample median is the
minimiser of mean absolute error. The choice therefore follows deductively
from a metric declared long before any result existed; it is not a
selection made after comparing candidates. Exactly one baseline strategy is
implemented here - there is no M0a/M0b, and no mean-versus-median contest.

M0 deliberately uses **no predictors**. The 43-column feature matrices are
never read by this module: a baseline that consulted features would not be
a naive floor. Its only scientific inputs are ``y_train.csv``,
``y_validation.csv`` and the validation rows of the feature row manifest,
which supply traceability metadata only.

**Test-set quarantine.** The 2014 partition is locked until Protocol
Phase 10. This module contains no path to, and no logic for, the test
target, the test feature matrix or 2014 evaluation of any kind. M0 having
no hyperparameters is *not* a licence to score the test early: Phase 10
evaluates M0-M3 together, once, on the same locked data.

Scope: no random operation, no shuffling, no train/test split, no
hyperparameter, no feature use, no target transformation, and no execution
timestamp - so repeated runs are byte-identical.

Historical constraint: CPython 3.6.7 with numpy 1.15.4, pandas 0.23.4 and
scikit-learn 0.20.0. Only ``sklearn.metrics`` is imported; no estimator
class is used anywhere. scikit-learn 0.20's ``mean_squared_error`` has no
``squared=`` keyword (that arrived in 0.22), so RMSE is computed as an
explicit square root.
"""

import io
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

from src.data.audit import sha256_of_file

# ----------------------------------------------------------------------
# Frozen declaration - fixed before any validation result was observed
# ----------------------------------------------------------------------

MODEL_ID = "M0"
MODEL_STRATEGY = "training_median_constant_baseline"
PRIMARY_DECISION_METRIC = "MAE"

FIT_PARTITION = "development_train"
FIT_PERIOD = "2010-01-01 00:00:00 to 2012-12-31 23:00:00"
EVALUATION_PARTITION = "validation"
EVALUATION_PERIOD = "2013-01-01 00:00:00 to 2013-12-31 23:00:00"

EXPECTED_TRAIN_ROWS = 24418
EXPECTED_VALIDATION_ROWS = 8678

TARGET = "pm2.5"

# Declared upstream digests. A mismatch stops the phase rather than
# silently regenerating an earlier phase.
EXPECTED_RAW_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")
EXPECTED_Y_TRAIN_SHA256 = (
    "fbcea1e4611e69cf707776e8925daed511539ee08a9809b54c796754fa7b53fa")
EXPECTED_Y_VALIDATION_SHA256 = (
    "9dc04233f80940865e9ae25e8380b458437fd0dece16b5c9727d72f891c1b0be")

# Metric definitions, recorded explicitly so the reported numbers are
# unambiguous. Only these three are computed: they are the metrics the
# protocol pre-registered.
METRIC_DEFINITIONS = OrderedDict([
    ("MAE", "mean(abs(y_true - y_pred))"),
    ("RMSE", "sqrt(mean((y_true - y_pred) ** 2))"),
    ("R2", "1 - SS_res / SS_tot"),
])

RESULTS_SUBDIR = os.path.join("results", "models", "m0")
METRICS_FILENAME = "m0_validation_metrics.csv"
PREDICTIONS_FILENAME = "m0_validation_predictions.csv"
MANIFEST_FILENAME = "m0_baseline_manifest.json"


class BaselineError(RuntimeError):
    """Raised when a declared M0 gate fails."""


# ----------------------------------------------------------------------
# Input verification
# ----------------------------------------------------------------------

def verify_upstream(project_root):
    """Verify the artifacts M0 depends on against their frozen digests.

    Only the inputs this phase actually consumes are listed. No test path
    appears here, by design.
    """
    checks = OrderedDict()

    raw_path = os.path.join(project_root, "data", "raw",
                            "PRSA_data_2010.1.1-2014.12.31.csv")
    raw_sha = sha256_of_file(raw_path)
    if raw_sha != EXPECTED_RAW_SHA256:
        raise BaselineError(
            "raw dataset SHA-256 changed: expected %s, found %s"
            % (EXPECTED_RAW_SHA256, raw_sha))
    checks["data/raw/PRSA_data_2010.1.1-2014.12.31.csv"] = raw_sha

    features_dir = os.path.join(project_root, "data", "processed", "features")
    for filename, expected in (("y_train.csv", EXPECTED_Y_TRAIN_SHA256),
                               ("y_validation.csv",
                                EXPECTED_Y_VALIDATION_SHA256)):
        path = os.path.join(features_dir, filename)
        observed = sha256_of_file(path)
        if observed != expected:
            raise BaselineError(
                "%s SHA-256 changed: expected %s, found %s"
                % (filename, expected, observed))
        checks[os.path.join("data", "processed", "features", filename)] = \
            observed

    for relative in (os.path.join("artifacts", "split_manifest.json"),
                     os.path.join("artifacts", "feature_schema.json"),
                     os.path.join("artifacts", "feature_row_manifest.csv")):
        path = os.path.join(project_root, relative)
        if not os.path.isfile(path):
            raise BaselineError("%s not found" % relative)
        checks[relative] = sha256_of_file(path)

    return checks


def load_target(path, expected_rows, label):
    """Load a development target and validate its integrity.

    Values are used exactly as written by Phase 5: nothing is clipped,
    transformed, imputed or filtered.
    """
    frame = pd.read_csv(path)
    if list(frame.columns) != [TARGET]:
        raise BaselineError(
            "%s: expected a single column %r, found %s"
            % (label, TARGET, list(frame.columns)))
    if len(frame) != expected_rows:
        raise BaselineError("%s: %d rows, expected %d"
                            % (label, len(frame), expected_rows))

    values = frame[TARGET].values.astype(np.float64)
    if bool(np.isnan(values).any()):
        raise BaselineError("%s: target contains NaN" % label)
    if bool(np.isinf(values).any()):
        raise BaselineError("%s: target contains infinite values" % label)
    return values


def describe_target_integrity(values, label):
    """Integrity facts only - not a predictive result and not a model."""
    return OrderedDict([
        ("partition", label),
        ("count", int(values.size)),
        ("n_missing", 0),
        ("n_infinite", 0),
        ("min", float(values.min())),
        ("max", float(values.max())),
        ("n_zero_valued_retained", int((values == 0).sum())),
    ])


# ----------------------------------------------------------------------
# Fit and predict
# ----------------------------------------------------------------------

def fit_baseline_constant(y_train):
    """The single learned quantity in M0: the training-target median.

    Fitted on the 2010-2012 development-training target alone. The
    validation target is not consulted.
    """
    return float(np.median(y_train))


def predict_constant(constant, n_rows):
    """Every prediction is exactly the baseline constant."""
    return np.full(n_rows, constant, dtype=np.float64)


# ----------------------------------------------------------------------
# Metrics - exactly the three pre-registered by the protocol
# ----------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """MAE, RMSE and R2 using scikit-learn 0.20 APIs.

    ``mean_squared_error`` in 0.20 has no ``squared=`` keyword, so RMSE is
    the explicit square root of MSE. No other score is computed.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return OrderedDict([("mae", mae), ("rmse", rmse), ("r2", r2)])


def verify_prediction_invariants(predictions, constant, y_true):
    """Every declared property of a constant baseline, checked."""
    report = OrderedDict()

    report["n_predictions"] = int(predictions.size)
    if predictions.size != y_true.size:
        raise BaselineError(
            "prediction count %d does not match target count %d"
            % (predictions.size, y_true.size))

    unique = np.unique(predictions)
    report["n_unique_predicted_values"] = int(unique.size)
    if unique.size != 1:
        raise BaselineError(
            "M0 must emit a single constant; found %d distinct values"
            % unique.size)
    if float(unique[0]) != float(constant):
        raise BaselineError(
            "predicted constant %r does not equal the training median %r"
            % (float(unique[0]), float(constant)))
    report["predicted_value_equals_training_median"] = True

    if bool(np.isnan(predictions).any()):
        raise BaselineError("predictions contain NaN")
    if bool(np.isinf(predictions).any()):
        raise BaselineError("predictions contain infinite values")
    if bool(np.isnan(y_true).any()):
        raise BaselineError("actual values contain NaN")
    if bool(np.isinf(y_true).any()):
        raise BaselineError("actual values contain infinite values")
    report["n_missing_predictions"] = 0
    report["n_missing_actuals"] = 0
    report["n_infinite_values"] = 0
    return report


# ----------------------------------------------------------------------
# Row alignment
# ----------------------------------------------------------------------

def load_validation_metadata(project_root, expected_rows):
    """Validation rows of the Phase 5 alignment manifest, in matrix order.

    Supplies ``No`` and ``timestamp`` for traceability only. The manifest
    carries no target, so nothing about the label enters through it.
    """
    path = os.path.join(project_root, "artifacts",
                        "feature_row_manifest.csv")
    manifest = pd.read_csv(path)
    if TARGET in manifest.columns:
        raise BaselineError("row manifest unexpectedly carries the target")

    block = manifest[manifest["partition"] == EVALUATION_PARTITION]
    block = block.reset_index(drop=True)
    if len(block) != expected_rows:
        raise BaselineError(
            "validation manifest has %d rows, expected %d"
            % (len(block), expected_rows))

    expected_order = np.arange(expected_rows, dtype=np.int64)
    if not bool((block["matrix_row_number"].values == expected_order).all()):
        raise BaselineError(
            "validation matrix_row_number is not sequential from 0")
    if not bool((np.diff(block["No"].values) > 0).all()):
        raise BaselineError(
            "validation rows are not in ascending identifier order")
    return block


def verify_row_alignment(metadata, y_validation, project_root):
    """Confirm predictions, targets and manifest describe the same rows.

    Cross-checked against the Phase 4 validation partition, which is the
    ultimate source of both the target and the identifiers.
    """
    source = pd.read_csv(
        os.path.join(project_root, "data", "processed",
                     "airsense_validation_2013.csv"))
    report = OrderedDict()

    if len(source) != len(y_validation) != len(metadata):
        raise BaselineError("row counts disagree across the three sources")
    report["y_validation_rows"] = int(len(y_validation))
    report["manifest_validation_rows"] = int(len(metadata))
    report["phase4_validation_rows"] = int(len(source))

    if not bool((metadata["No"].values == source["No"].values).all()):
        raise BaselineError(
            "manifest identifiers do not match the Phase 4 validation rows")
    report["identifiers_match_phase4"] = True

    if not bool((y_validation
                 == source[TARGET].values.astype(np.float64)).all()):
        raise BaselineError(
            "y_validation values do not match the Phase 4 validation target")
    report["targets_match_phase4"] = True
    report["row_order_preserved"] = True
    return report


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def run_m0_baseline(project_root, verbose=True):
    """Fit M0 on 2010-2012 and evaluate it on 2013. 2014 is untouched."""

    def log(message):
        if verbose:
            print(message)

    results_dir = os.path.join(project_root, RESULTS_SUBDIR)
    artifacts_dir = os.path.join(project_root, "artifacts")
    for directory in (results_dir, artifacts_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    upstream = verify_upstream(project_root)
    log("Upstream artifacts verified against their frozen digests")

    features_dir = os.path.join(project_root, "data", "processed", "features")
    y_train = load_target(os.path.join(features_dir, "y_train.csv"),
                          EXPECTED_TRAIN_ROWS, "y_train")
    y_validation = load_target(
        os.path.join(features_dir, "y_validation.csv"),
        EXPECTED_VALIDATION_ROWS, "y_validation")
    log("Targets loaded: %d training, %d validation"
        % (y_train.size, y_validation.size))

    integrity = [describe_target_integrity(y_train, "development_train"),
                 describe_target_integrity(y_validation, "validation")]

    # -- fit: the only learned quantity in M0 ---------------------------
    constant = fit_baseline_constant(y_train)
    log("M0 baseline constant (median of 2010-2012 target): %r" % constant)

    metadata = load_validation_metadata(project_root,
                                        EXPECTED_VALIDATION_ROWS)
    alignment = verify_row_alignment(metadata, y_validation, project_root)
    log("Row alignment verified against the Phase 4 validation partition")

    # -- predict and evaluate on 2013 ONLY ------------------------------
    predictions = predict_constant(constant, y_validation.size)
    invariants = verify_prediction_invariants(predictions, constant,
                                              y_validation)
    metrics = compute_metrics(y_validation, predictions)
    log("2013 validation metrics computed (MAE / RMSE / R2)")

    # -- prediction audit ------------------------------------------------
    prediction_frame = pd.DataFrame(OrderedDict([
        ("matrix_row_number", metadata["matrix_row_number"].values),
        ("No", metadata["No"].values),
        ("timestamp", metadata["timestamp"].values),
        ("actual_pm25", y_validation),
        ("predicted_pm25", predictions),
    ]))
    predictions_path = os.path.join(results_dir, PREDICTIONS_FILENAME)
    prediction_frame.to_csv(predictions_path, index=False)
    predictions_sha = sha256_of_file(predictions_path)

    # -- metrics table ---------------------------------------------------
    metrics_frame = pd.DataFrame([OrderedDict([
        ("model", MODEL_ID),
        ("strategy", MODEL_STRATEGY),
        ("fit_period", FIT_PERIOD),
        ("evaluation_period", EVALUATION_PERIOD),
        ("n_train", int(y_train.size)),
        ("n_validation", int(y_validation.size)),
        ("baseline_constant", constant),
        ("mae", metrics["mae"]),
        ("rmse", metrics["rmse"]),
        ("r2", metrics["r2"]),
    ])])
    metrics_path = os.path.join(results_dir, METRICS_FILENAME)
    metrics_frame.to_csv(metrics_path, index=False)
    metrics_sha = sha256_of_file(metrics_path)
    log("Wrote %s" % os.path.join(RESULTS_SUBDIR, METRICS_FILENAME))

    # -- manifest --------------------------------------------------------
    manifest = OrderedDict()
    manifest["model_id"] = MODEL_ID
    manifest["model_type"] = "constant_predictor"
    manifest["strategy"] = MODEL_STRATEGY
    manifest["phase"] = "V1 / Protocol Phase 6 - baseline regression (M0)"
    manifest["scientific_rationale"] = (
        "The primary decision metric was frozen in Phase 0 as MAE. For a "
        "constant predictor the sample median minimises mean absolute "
        "error, so the training median follows deductively from an "
        "already-declared metric rather than from any observed result. "
        "Exactly one baseline strategy is implemented; no mean baseline "
        "was computed and no candidate comparison was performed.")
    manifest["primary_decision_metric"] = PRIMARY_DECISION_METRIC
    manifest["strategy_frozen_before_results"] = True
    manifest["hyperparameters"] = None
    manifest["predictors_used"] = []
    manifest["predictors_used_note"] = (
        "M0 deliberately uses none of the 43 prepared features; a baseline "
        "that consulted predictors would not be a naive floor.")

    manifest["fit_partition"] = FIT_PARTITION
    manifest["fit_period"] = FIT_PERIOD
    manifest["evaluation_partition"] = EVALUATION_PARTITION
    manifest["evaluation_period"] = EVALUATION_PERIOD
    manifest["n_train"] = int(y_train.size)
    manifest["n_validation"] = int(y_validation.size)
    manifest["baseline_constant"] = constant

    manifest["metric_definitions"] = METRIC_DEFINITIONS
    manifest["metric_implementation"] = (
        "sklearn.metrics 0.20.0; RMSE computed as sqrt(mean_squared_error) "
        "because the squared= keyword did not exist until 0.22")
    validation_metrics = OrderedDict()
    validation_metrics["evaluation_period_label"] = "2013 validation"
    validation_metrics["result_status"] = (
        "development validation result - NOT final held-out test "
        "performance and NOT a project headline figure")
    validation_metrics["mae"] = metrics["mae"]
    validation_metrics["rmse"] = metrics["rmse"]
    validation_metrics["r2"] = metrics["r2"]
    manifest["validation_metrics"] = validation_metrics

    manifest["target_integrity"] = integrity
    manifest["prediction_invariants"] = invariants
    manifest["row_alignment"] = alignment

    manifest["upstream_sha256"] = upstream
    outputs = OrderedDict()
    outputs[os.path.join(RESULTS_SUBDIR, PREDICTIONS_FILENAME)] = \
        OrderedDict([("sha256", predictions_sha),
                     ("rows", int(len(prediction_frame)))])
    outputs[os.path.join(RESULTS_SUBDIR, METRICS_FILENAME)] = \
        OrderedDict([("sha256", metrics_sha),
                     ("rows", int(len(metrics_frame)))])
    manifest["outputs"] = outputs

    quarantine = OrderedDict()
    quarantine["test_partition"] = "2014"
    quarantine["test_target_quarantine_status"] = "ACTIVE"
    quarantine["test_target_read"] = False
    quarantine["test_features_read"] = False
    quarantine["test_predictions_generated"] = False
    quarantine["test_evaluation_status"] = "not_evaluated"
    quarantine["final_test_status"] = "not_evaluated"
    quarantine["quarantine_lifts_at"] = "Protocol Phase 10"
    quarantine["note"] = (
        "M0 having no hyperparameters is not a licence to score the test "
        "early. Protocol Phase 10 evaluates M0-M3 together, once, on the "
        "same locked 2014 partition.")
    manifest["test_quarantine"] = quarantine

    final_refit = OrderedDict()
    final_refit["executed"] = False
    final_refit["policy"] = (
        "At Protocol Phase 10 the M0 constant is recomputed as the median "
        "of the combined 2010-2013 development target, 2014 predictions "
        "are generated once, and the locked 2014 target is evaluated once. "
        "The constant reported in this phase is a development-validation "
        "figure and is not automatically the final M0 constant.")
    final_refit["combined_2010_2013_target_created"] = False
    manifest["final_refit_policy"] = final_refit

    manifest["note"] = (
        "No execution timestamp is recorded so that repeated runs are "
        "byte-identical.")

    manifest_path = os.path.join(artifacts_dir, MANIFEST_FILENAME)
    with io.open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2))
        handle.write("\n")
    log("Wrote artifacts/%s" % MANIFEST_FILENAME)

    verify_upstream(project_root)
    log("")
    log("Upstream artifacts unchanged after M0")
    return manifest
