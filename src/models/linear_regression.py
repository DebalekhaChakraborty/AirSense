"""M1 ordinary least-squares linear regression for AirSense V1.

Protocol Phase 7. The first feature-based predictive model, fitted after
the M0 naive baseline and before M2 Decision Tree.

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)                 2010-2012 only
    predictions = model.predict(X_validation)   2013 only

M1 consumes the **frozen Phase 5 matrices exactly as produced**. It does not
reconstruct, add, remove, transform, scale or reorder a single feature, and
it verifies the column order against ``artifacts/feature_schema.json``
before fitting.

Deliberate omissions, each fixed before any result was observed:

* **No scaling.** Phase 5 froze "no scaling", and ordinary least squares is
  invariant to it anyway: rescaling a predictor rescales its coefficient
  and leaves the fit and the predictions unchanged. Numerical diagnostics
  may reveal poor conditioning; that is a finding to report, not a licence
  to change the experiment.
* **No regularization.** Ridge, Lasso and ElasticNet are different models
  and are outside the frozen M0-M3 set. The purpose of M1 is specifically
  to evaluate an *ordinary* linear model, including wherever that turns out
  to be a poor choice.
* **No hyperparameter search, no feature ablation.** There is nothing to
  tune in this specification, and fitting variants after seeing results
  would be post-hoc feature selection.

**Test-set quarantine.** The 2014 partition is locked until Protocol
Phase 10. This module has no path to, and no logic for, the test features,
the test target or 2014 evaluation of any kind.

Determinism: ordinary least squares has no random component, and no
execution timestamp is written into any artifact, so repeated runs are
byte-identical.

Historical constraint: CPython 3.6.7 with numpy 1.15.4, pandas 0.23.4,
scikit-learn 0.20.0, matplotlib 3.0.2. scikit-learn 0.20's
``mean_squared_error`` has no ``squared=`` keyword (added in 0.22), so RMSE
is an explicit square root. No dependency is added for diagnostics: matrix
rank, singular values and the condition number come from ``numpy.linalg``.
"""

import io
import json
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")               # no display; must precede pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

from src.data.audit import sha256_of_file

# ----------------------------------------------------------------------
# Frozen declaration
# ----------------------------------------------------------------------

MODEL_ID = "M1"
ESTIMATOR = "sklearn.linear_model.LinearRegression"
FIT_INTERCEPT = True

FIT_PARTITION = "development_train"
FIT_PERIOD = "2010-01-01 00:00:00 to 2012-12-31 23:00:00"
EVALUATION_PARTITION = "validation"
EVALUATION_PERIOD_LABEL = "2013_validation"
EVALUATION_PERIOD = "2013-01-01 00:00:00 to 2013-12-31 23:00:00"

EXPECTED_TRAIN_ROWS = 24418
EXPECTED_VALIDATION_ROWS = 8678
EXPECTED_FEATURE_COUNT = 43

TARGET = "pm2.5"
INTERCEPT_LABEL = "__intercept__"

EXPECTED_RAW_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")

METRIC_DEFINITIONS = OrderedDict([
    ("MAE", "mean(abs(y_true - y_pred))"),
    ("RMSE", "sqrt(mean((y_true - y_pred) ** 2))"),
    ("R2", "1 - SS_res / SS_tot"),
])

# residual = actual - predicted, used with this sign everywhere.
RESIDUAL_DEFINITION = "residual = actual_pm25 - predicted_pm25"

RESULTS_SUBDIR = os.path.join("results", "models", "m1")
COEFFICIENTS_FILENAME = "m1_coefficients.csv"
DIAGNOSTICS_FILENAME = "m1_training_diagnostics.csv"
METRICS_FILENAME = "m1_validation_metrics.csv"
PREDICTIONS_FILENAME = "m1_validation_predictions.csv"
MANIFEST_FILENAME = "m1_linear_regression_manifest.json"

FIGURE_ACTUAL_VS_PREDICTED = "m1_validation_actual_vs_predicted.png"
FIGURE_RESIDUALS_VS_PREDICTED = "m1_validation_residuals_vs_predicted.png"

FIGURE_DPI = 100
PLOT_COLOR = "#31688e"
ACCENT_COLOR = "#b03a2e"

# Feature-type labels for the coefficient artifact, and the reference level
# omitted from each one-hot block.
REFERENCE_LEVELS = OrderedDict([
    ("month", "month_1"), ("hour", "hour_0"), ("cbwd", "cbwd_NE")])
METEO_FEATURES = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]


class LinearModelError(RuntimeError):
    """Raised when a declared M1 gate fails."""


# ----------------------------------------------------------------------
# Input verification
# ----------------------------------------------------------------------

def verify_upstream(project_root):
    """Verify every frozen artifact M1 depends on. No test path appears."""
    checks = OrderedDict()

    raw_path = os.path.join(project_root, "data", "raw",
                            "PRSA_data_2010.1.1-2014.12.31.csv")
    raw_sha = sha256_of_file(raw_path)
    if raw_sha != EXPECTED_RAW_SHA256:
        raise LinearModelError(
            "raw dataset SHA-256 changed: expected %s, found %s"
            % (EXPECTED_RAW_SHA256, raw_sha))
    checks["data/raw/PRSA_data_2010.1.1-2014.12.31.csv"] = raw_sha

    schema_path = os.path.join(project_root, "artifacts",
                               "feature_schema.json")
    schema = json.load(io.open(schema_path, encoding="utf-8"))

    features_dir = os.path.join(project_root, "data", "processed", "features")
    for filename in ("X_train.csv", "X_validation.csv", "y_train.csv",
                     "y_validation.csv"):
        relative = os.path.join("data", "processed", "features", filename)
        path = os.path.join(features_dir, filename)
        recorded = schema["outputs"].get(relative)
        if recorded is None:
            raise LinearModelError("%s is not recorded in feature_schema.json"
                                   % relative)
        observed = sha256_of_file(path)
        if observed != recorded["sha256"]:
            raise LinearModelError(
                "%s SHA-256 changed: schema %s, found %s"
                % (relative, recorded["sha256"], observed))
        checks[relative] = observed

    for relative in (os.path.join("artifacts", "split_manifest.json"),
                     os.path.join("artifacts", "feature_schema.json"),
                     os.path.join("artifacts", "feature_row_manifest.csv"),
                     os.path.join("artifacts", "m0_baseline_manifest.json")):
        path = os.path.join(project_root, relative)
        if not os.path.isfile(path):
            raise LinearModelError("%s not found" % relative)
        checks[relative] = sha256_of_file(path)

    return checks, schema


def load_matrix(path, expected_rows, feature_names, label):
    """Load a frozen feature matrix and verify it exactly."""
    frame = pd.read_csv(path)
    if list(frame.columns) != feature_names:
        raise LinearModelError(
            "%s: column order differs from the frozen feature schema" % label)
    if frame.shape != (expected_rows, EXPECTED_FEATURE_COUNT):
        raise LinearModelError(
            "%s: shape %s, expected (%d, %d)"
            % (label, frame.shape, expected_rows, EXPECTED_FEATURE_COUNT))
    values = frame.values.astype(np.float64)
    if bool(np.isnan(values).any()):
        raise LinearModelError("%s: feature matrix contains NaN" % label)
    if bool(np.isinf(values).any()):
        raise LinearModelError("%s: feature matrix contains infinities"
                               % label)
    return frame, values


def load_target(path, expected_rows, label):
    """Load a development target. Values are used exactly as frozen."""
    frame = pd.read_csv(path)
    if list(frame.columns) != [TARGET]:
        raise LinearModelError(
            "%s: expected a single column %r, found %s"
            % (label, TARGET, list(frame.columns)))
    if len(frame) != expected_rows:
        raise LinearModelError("%s: %d rows, expected %d"
                               % (label, len(frame), expected_rows))
    values = frame[TARGET].values.astype(np.float64)
    if bool(np.isnan(values).any()):
        raise LinearModelError("%s: target contains NaN" % label)
    if bool(np.isinf(values).any()):
        raise LinearModelError("%s: target contains infinities" % label)
    return values


# ----------------------------------------------------------------------
# Training-design diagnostics - TRAINING MATRIX ONLY
# ----------------------------------------------------------------------

def design_diagnostics(X_train, model):
    """Numerical properties of the training design matrix.

    Two design matrices are relevant and both are recorded, because they
    answer different questions and quoting only one would be misleading:

    ``design_with_intercept``
        the 44-column matrix ``[1 | X]`` that ordinary least squares
        actually solves when ``fit_intercept=True``. Its condition number
        is the one that governs the numerical stability of the fitted
        coefficients, and it is reported as the headline figure.

    ``centred_X``
        the 43-column matrix scikit-learn 0.20 internally centres and
        decomposes, whose singular values the estimator exposes as
        ``singular_``. Recorded as an independent cross-check.

    The condition number is defined throughout as

        largest singular value / smallest NON-ZERO singular value

    with "non-zero" meaning above ``max(m, n) * eps * largest``, the
    standard numerical rank tolerance. Computed on training data only.
    """
    n_rows, n_cols = X_train.shape

    def summarise(matrix, name):
        singular = np.linalg.svd(matrix, compute_uv=False)
        largest = float(singular[0])
        smallest = float(singular[-1])
        tolerance = float(max(matrix.shape) * np.finfo(np.float64).eps
                          * largest)
        nonzero = singular[singular > tolerance]
        rank = int(nonzero.size)
        smallest_nonzero = float(nonzero[-1]) if nonzero.size else float("nan")
        condition = (largest / smallest_nonzero
                     if smallest_nonzero and np.isfinite(smallest_nonzero)
                     else float("inf"))
        return OrderedDict([
            ("matrix", name),
            ("n_rows", int(matrix.shape[0])),
            ("n_columns", int(matrix.shape[1])),
            ("matrix_rank", rank),
            ("n_linearly_independent_columns", rank),
            ("rank_deficient", bool(rank < matrix.shape[1])),
            ("largest_singular_value", largest),
            ("smallest_singular_value", smallest),
            ("smallest_nonzero_singular_value", smallest_nonzero),
            ("rank_tolerance", tolerance),
            ("condition_number", float(condition)),
        ])

    intercept_column = np.ones((n_rows, 1), dtype=np.float64)
    with_intercept = np.hstack([intercept_column, X_train])

    report = OrderedDict()
    report["condition_number_definition"] = (
        "largest_singular_value / smallest_nonzero_singular_value, where "
        "non-zero means > max(m, n) * eps * largest_singular_value")
    report["computed_on"] = "development_train (2010-2012) only"
    report["design_with_intercept"] = summarise(
        with_intercept, "design_with_intercept_[1|X]_44_columns")
    report["raw_X_train"] = summarise(X_train, "X_train_43_columns")

    centred = X_train - X_train.mean(axis=0)
    report["centred_X_train"] = summarise(
        centred, "centred_X_train_43_columns_as_sklearn_decomposes")

    sklearn_report = OrderedDict()
    sklearn_report["rank_"] = (int(model.rank_)
                               if hasattr(model, "rank_") else None)
    if hasattr(model, "singular_") and model.singular_ is not None:
        singular = np.asarray(model.singular_, dtype=np.float64)
        sklearn_report["n_singular_values"] = int(singular.size)
        sklearn_report["largest_singular_value"] = float(singular[0])
        sklearn_report["smallest_singular_value"] = float(singular[-1])
    report["sklearn_estimator_attributes"] = sklearn_report
    return report


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------

def fit_m1(X_train, y_train):
    """One predeclared ordinary least-squares fit. Nothing is tuned."""
    model = LinearRegression(fit_intercept=FIT_INTERCEPT)
    model.fit(X_train, y_train)
    return model


def verify_coefficients(model, feature_names):
    """Coefficients must be complete, finite and in the frozen order."""
    coefficients = np.asarray(model.coef_, dtype=np.float64).ravel()
    if coefficients.size != EXPECTED_FEATURE_COUNT:
        raise LinearModelError(
            "%d coefficients, expected %d"
            % (coefficients.size, EXPECTED_FEATURE_COUNT))
    if coefficients.size != len(feature_names):
        raise LinearModelError(
            "coefficient count does not match the feature schema")
    if bool(np.isnan(coefficients).any()):
        raise LinearModelError("fit produced NaN coefficients")
    if bool(np.isinf(coefficients).any()):
        raise LinearModelError("fit produced infinite coefficients")

    intercept = float(model.intercept_)
    if not np.isfinite(intercept):
        raise LinearModelError("fit produced a non-finite intercept")
    return coefficients, intercept


def feature_type(name):
    if name in METEO_FEATURES:
        return "raw_meteorological"
    if name.startswith("month_"):
        return "calendar_month_one_hot"
    if name.startswith("hour_"):
        return "calendar_hour_one_hot"
    if name.startswith("cbwd_"):
        return "wind_direction_one_hot"
    return "unknown"


def reference_note(name):
    if name.startswith("month_"):
        return "coefficient is relative to the omitted reference %s" \
               % REFERENCE_LEVELS["month"]
    if name.startswith("hour_"):
        return "coefficient is relative to the omitted reference %s" \
               % REFERENCE_LEVELS["hour"]
    if name.startswith("cbwd_"):
        return "coefficient is relative to the omitted reference %s" \
               % REFERENCE_LEVELS["cbwd"]
    return "raw predictor in original units; not scaled"


# ----------------------------------------------------------------------
# Metrics and comparison
# ----------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """Exactly the three metrics the protocol pre-registered."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    return OrderedDict([
        ("mae", mae),
        ("rmse", float(np.sqrt(mse))),
        ("r2", float(r2_score(y_true, y_pred))),
    ])


def compare_with_m0(project_root, metrics):
    """Compare against the frozen M0 result, read from its manifest.

    M0's metrics are never hard-coded here; they are read from
    ``artifacts/m0_baseline_manifest.json`` so the comparison cannot drift
    away from the recorded baseline.
    """
    path = os.path.join(project_root, "artifacts",
                        "m0_baseline_manifest.json")
    manifest = json.load(io.open(path, encoding="utf-8"))
    m0 = manifest["validation_metrics"]

    m0_mae = float(m0["mae"])
    m0_rmse = float(m0["rmse"])
    m0_r2 = float(m0["r2"])

    comparison = OrderedDict()
    comparison["m0_source"] = "artifacts/m0_baseline_manifest.json"
    comparison["m0_manifest_sha256"] = sha256_of_file(path)
    comparison["m0_strategy"] = manifest["strategy"]
    comparison["m0_baseline_constant"] = float(
        manifest["baseline_constant"])
    comparison["evaluation_period_label"] = EVALUATION_PERIOD_LABEL
    comparison["m0_mae"] = m0_mae
    comparison["m0_rmse"] = m0_rmse
    comparison["m0_r2"] = m0_r2
    comparison["m1_mae"] = metrics["mae"]
    comparison["m1_rmse"] = metrics["rmse"]
    comparison["m1_r2"] = metrics["r2"]
    comparison["mae_improvement_absolute"] = m0_mae - metrics["mae"]
    comparison["mae_improvement_percent"] = (
        (m0_mae - metrics["mae"]) / m0_mae * 100.0)
    comparison["rmse_improvement_absolute"] = m0_rmse - metrics["rmse"]
    comparison["rmse_improvement_percent"] = (
        (m0_rmse - metrics["rmse"]) / m0_rmse * 100.0)
    comparison["r2_delta"] = metrics["r2"] - m0_r2
    comparison["primary_comparison_metric"] = "MAE"
    comparison["note"] = (
        "Development-validation comparison on 2013 only. The primary "
        "comparison metric remains MAE, as pre-registered; it was not "
        "changed because another metric looked more favourable.")
    return comparison


# ----------------------------------------------------------------------
# Residual diagnostics - descriptive only
# ----------------------------------------------------------------------

def residual_diagnostics(y_true, y_pred):
    """Descriptive residual summary on 2013 validation.

    These are recorded to characterise the fit, not to modify it. Nothing
    here feeds back into M1, and full severe-episode analysis belongs to
    Protocol Phase 11.
    """
    residuals = y_true - y_pred
    correlation = float(np.corrcoef(y_pred, residuals)[0, 1])
    return OrderedDict([
        ("residual_definition", RESIDUAL_DEFINITION),
        ("n", int(residuals.size)),
        ("mean_residual", float(residuals.mean())),
        ("median_residual", float(np.median(residuals))),
        ("residual_std", float(residuals.std(ddof=1))),
        ("min_residual", float(residuals.min())),
        ("max_residual", float(residuals.max())),
        ("pearson_corr_predicted_vs_residual", correlation),
    ])


# ----------------------------------------------------------------------
# Figures - at most two, deterministic
# ----------------------------------------------------------------------

def _save(fig, path):
    """Save without embedding matplotlib's version string in the PNG."""
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=FIGURE_DPI, metadata={"Software": "AirSense V1"})
    except TypeError:                             # pragma: no cover
        fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def figure_actual_vs_predicted(y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(y_pred, y_true, s=6, alpha=0.15, color=PLOT_COLOR,
               edgecolors="none")
    lower = float(min(y_pred.min(), y_true.min()))
    upper = float(max(y_pred.max(), y_true.max()))
    ax.plot([lower, upper], [lower, upper], color=ACCENT_COLOR,
            linestyle="--", linewidth=1.2, label="y = x (perfect prediction)")
    ax.set_xlabel("Predicted PM2.5 (ug/m^3)")
    ax.set_ylabel("Actual PM2.5 (ug/m^3)")
    ax.set_title("M1 Linear Regression - actual vs predicted\n"
                 "2013 DEVELOPMENT VALIDATION (not final test)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, path)


def figure_residuals_vs_predicted(y_true, y_pred, path):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(y_pred, residuals, s=6, alpha=0.15, color=PLOT_COLOR,
               edgecolors="none")
    ax.axhline(0.0, color=ACCENT_COLOR, linestyle="--", linewidth=1.2,
               label="zero residual")
    ax.set_xlabel("Predicted PM2.5 (ug/m^3)")
    ax.set_ylabel("Residual = actual - predicted (ug/m^3)")
    ax.set_title("M1 Linear Regression - residuals vs predicted\n"
                 "2013 DEVELOPMENT VALIDATION (not final test)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, path)


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------

def verify_equation_reconstruction(X_validation, coefficients, intercept,
                                   predictions):
    """Rebuild predictions from the coefficient artifact alone.

    ``intercept + X . coefficients`` must reproduce ``model.predict`` to
    floating-point tolerance. This proves the saved coefficients completely
    represent the fitted linear equation - that nothing was dropped,
    reordered or silently rescaled.
    """
    reconstructed = intercept + X_validation.dot(coefficients)
    difference = np.abs(reconstructed - predictions)
    max_difference = float(difference.max())
    tolerance = 1e-10
    if max_difference > tolerance:
        raise LinearModelError(
            "equation reconstruction differs from model.predict by %.3e, "
            "exceeding %.1e" % (max_difference, tolerance))
    return OrderedDict([
        ("method", "intercept + X_validation.dot(coefficients)"),
        ("max_absolute_difference", max_difference),
        ("tolerance", tolerance),
        ("agrees", True),
    ])


def verify_prediction_invariants(predictions, y_true, m0_constant):
    """Every declared property of the M1 validation predictions."""
    report = OrderedDict()
    report["n_predictions"] = int(predictions.size)
    report["n_actuals"] = int(y_true.size)
    if predictions.size != y_true.size:
        raise LinearModelError("prediction and actual counts differ")
    if predictions.size != EXPECTED_VALIDATION_ROWS:
        raise LinearModelError(
            "%d predictions, expected %d"
            % (predictions.size, EXPECTED_VALIDATION_ROWS))

    if not bool(np.isfinite(predictions).all()):
        raise LinearModelError("predictions contain non-finite values")
    if not bool(np.isfinite(y_true).all()):
        raise LinearModelError("actuals contain non-finite values")
    report["predictions_finite"] = True
    report["actuals_finite"] = True

    n_distinct = int(np.unique(predictions).size)
    report["n_distinct_predictions"] = n_distinct
    if n_distinct < 2:
        raise LinearModelError(
            "M1 produced fewer than two distinct predictions; it is "
            "behaving like a constant baseline")
    report["prediction_std"] = float(predictions.std(ddof=1))

    # M1 must not be numerically identical to the M0 constant.
    identical_to_m0 = bool(np.all(predictions == m0_constant))
    if identical_to_m0:
        raise LinearModelError(
            "M1 predictions are identically the M0 constant")
    report["identical_to_m0_constant"] = False
    report["m0_constant"] = float(m0_constant)
    report["prediction_min"] = float(predictions.min())
    report["prediction_max"] = float(predictions.max())
    return report


def load_validation_metadata(project_root, expected_rows):
    """Validation rows of the Phase 5 alignment manifest, in matrix order."""
    path = os.path.join(project_root, "artifacts",
                        "feature_row_manifest.csv")
    manifest = pd.read_csv(path)
    if TARGET in manifest.columns:
        raise LinearModelError("row manifest unexpectedly carries the target")
    block = manifest[manifest["partition"] == EVALUATION_PARTITION]
    block = block.reset_index(drop=True)
    if len(block) != expected_rows:
        raise LinearModelError(
            "validation manifest has %d rows, expected %d"
            % (len(block), expected_rows))
    if not bool((block["matrix_row_number"].values
                 == np.arange(expected_rows)).all()):
        raise LinearModelError("validation matrix_row_number is not "
                               "sequential from 0")
    return block


def verify_row_alignment(project_root, metadata, y_validation):
    """Confirm predictions, targets and manifest describe the same rows."""
    source = pd.read_csv(
        os.path.join(project_root, "data", "processed",
                     "airsense_validation_2013.csv"))
    if not bool((metadata["No"].values == source["No"].values).all()):
        raise LinearModelError(
            "manifest identifiers do not match the Phase 4 validation rows")
    if not bool((y_validation
                 == source[TARGET].values.astype(np.float64)).all()):
        raise LinearModelError(
            "y_validation does not match the Phase 4 validation target")
    return OrderedDict([
        ("y_validation_rows", int(y_validation.size)),
        ("manifest_validation_rows", int(len(metadata))),
        ("phase4_validation_rows", int(len(source))),
        ("identifiers_match_phase4", True),
        ("targets_match_phase4", True),
        ("row_order_preserved", True),
    ])


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def run_m1_linear_regression(project_root, verbose=True):
    """Fit M1 on 2010-2012 and evaluate it on 2013. 2014 is untouched."""

    def log(message):
        if verbose:
            print(message)

    results_dir = os.path.join(project_root, RESULTS_SUBDIR)
    artifacts_dir = os.path.join(project_root, "artifacts")
    figures_dir = os.path.join(project_root, "figures")
    for directory in (results_dir, artifacts_dir, figures_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    upstream, schema = verify_upstream(project_root)
    feature_names = list(schema["feature_names"])
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise LinearModelError(
            "feature schema declares %d features, expected %d"
            % (len(feature_names), EXPECTED_FEATURE_COUNT))
    log("Upstream artifacts verified; frozen schema has %d features"
        % len(feature_names))

    features_dir = os.path.join(project_root, "data", "processed", "features")
    X_train_frame, X_train = load_matrix(
        os.path.join(features_dir, "X_train.csv"), EXPECTED_TRAIN_ROWS,
        feature_names, "X_train")
    X_validation_frame, X_validation = load_matrix(
        os.path.join(features_dir, "X_validation.csv"),
        EXPECTED_VALIDATION_ROWS, feature_names, "X_validation")
    y_train = load_target(os.path.join(features_dir, "y_train.csv"),
                          EXPECTED_TRAIN_ROWS, "y_train")
    y_validation = load_target(
        os.path.join(features_dir, "y_validation.csv"),
        EXPECTED_VALIDATION_ROWS, "y_validation")
    log("Loaded X_train %s, X_validation %s"
        % (X_train.shape, X_validation.shape))

    # -- fit on 2010-2012 only -------------------------------------------
    model = fit_m1(X_train, y_train)
    coefficients, intercept = verify_coefficients(model, feature_names)
    log("M1 fitted: %d coefficients, intercept %.10f"
        % (coefficients.size, intercept))

    diagnostics = design_diagnostics(X_train, model)
    primary = diagnostics["design_with_intercept"]
    log("Training design: rank %d/%d, condition number %.6g"
        % (primary["matrix_rank"], primary["n_columns"],
           primary["condition_number"]))

    # -- predict and evaluate on 2013 ONLY -------------------------------
    predictions = model.predict(X_validation)
    metrics = compute_metrics(y_validation, predictions)
    comparison = compare_with_m0(project_root, metrics)
    residuals_summary = residual_diagnostics(y_validation, predictions)
    log("2013 validation: MAE %.6f  RMSE %.6f  R2 %.6f"
        % (metrics["mae"], metrics["rmse"], metrics["r2"]))

    equation_check = verify_equation_reconstruction(
        X_validation, coefficients, intercept, predictions)
    invariants = verify_prediction_invariants(
        predictions, y_validation, comparison["m0_baseline_constant"])
    metadata = load_validation_metadata(project_root,
                                        EXPECTED_VALIDATION_ROWS)
    alignment = verify_row_alignment(project_root, metadata, y_validation)
    log("Equation reconstruction and row alignment verified")

    # -- artifacts --------------------------------------------------------
    coefficient_frame = pd.DataFrame(OrderedDict([
        ("feature", feature_names),
        ("coefficient", coefficients),
        ("feature_type", [feature_type(n) for n in feature_names]),
        ("reference_note", [reference_note(n) for n in feature_names]),
    ]))
    coefficients_path = os.path.join(results_dir, COEFFICIENTS_FILENAME)
    coefficient_frame.to_csv(coefficients_path, index=False)
    coefficients_sha = sha256_of_file(coefficients_path)

    diagnostics_frame = pd.DataFrame([OrderedDict([
        ("model", MODEL_ID),
        ("n_train", int(X_train.shape[0])),
        ("n_features", int(X_train.shape[1])),
        ("matrix_rank", primary["matrix_rank"]),
        ("n_design_columns_with_intercept", primary["n_columns"]),
        ("rank_deficient", primary["rank_deficient"]),
        ("largest_singular_value", primary["largest_singular_value"]),
        ("smallest_singular_value", primary["smallest_singular_value"]),
        ("smallest_nonzero_singular_value",
         primary["smallest_nonzero_singular_value"]),
        ("condition_number", primary["condition_number"]),
        ("intercept", intercept),
    ])])
    diagnostics_path = os.path.join(results_dir, DIAGNOSTICS_FILENAME)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    diagnostics_sha = sha256_of_file(diagnostics_path)

    metrics_frame = pd.DataFrame([OrderedDict([
        ("model", MODEL_ID),
        ("fit_period", FIT_PERIOD),
        ("evaluation_period", EVALUATION_PERIOD_LABEL),
        ("n_train", int(X_train.shape[0])),
        ("n_validation", int(X_validation.shape[0])),
        ("n_features", int(X_train.shape[1])),
        ("mae", metrics["mae"]),
        ("rmse", metrics["rmse"]),
        ("r2", metrics["r2"]),
        ("m0_mae", comparison["m0_mae"]),
        ("mae_improvement_absolute", comparison["mae_improvement_absolute"]),
        ("mae_improvement_percent", comparison["mae_improvement_percent"]),
        ("m0_rmse", comparison["m0_rmse"]),
        ("rmse_improvement_absolute",
         comparison["rmse_improvement_absolute"]),
        ("rmse_improvement_percent",
         comparison["rmse_improvement_percent"]),
        ("m0_r2", comparison["m0_r2"]),
        ("r2_delta", comparison["r2_delta"]),
    ])])
    metrics_path = os.path.join(results_dir, METRICS_FILENAME)
    metrics_frame.to_csv(metrics_path, index=False)
    metrics_sha = sha256_of_file(metrics_path)

    prediction_frame = pd.DataFrame(OrderedDict([
        ("matrix_row_number", metadata["matrix_row_number"].values),
        ("No", metadata["No"].values),
        ("timestamp", metadata["timestamp"].values),
        ("actual_pm25", y_validation),
        ("predicted_pm25", predictions),
        ("residual", y_validation - predictions),
    ]))
    predictions_path = os.path.join(results_dir, PREDICTIONS_FILENAME)
    prediction_frame.to_csv(predictions_path, index=False)
    predictions_sha = sha256_of_file(predictions_path)
    log("Wrote coefficients, diagnostics, metrics and predictions")

    actual_path = os.path.join(figures_dir, FIGURE_ACTUAL_VS_PREDICTED)
    figure_actual_vs_predicted(y_validation, predictions, actual_path)
    residual_path = os.path.join(figures_dir, FIGURE_RESIDUALS_VS_PREDICTED)
    figure_residuals_vs_predicted(y_validation, predictions, residual_path)
    log("Wrote 2 validation diagnostic figures")

    # -- manifest ---------------------------------------------------------
    manifest = OrderedDict()
    manifest["model_id"] = MODEL_ID
    manifest["estimator"] = ESTIMATOR
    manifest["sklearn_version"] = __import__("sklearn").__version__
    manifest["fit_intercept"] = FIT_INTERCEPT
    manifest["phase"] = "V1 / Protocol Phase 7 - linear regression (M1)"
    manifest["specification_note"] = (
        "Ordinary least squares on the frozen Phase 5 matrix. No scaling, "
        "no regularization, no hyperparameter search, no feature ablation. "
        "The specification was fixed before any result was observed.")
    manifest["feature_count"] = int(len(feature_names))
    manifest["feature_names"] = feature_names
    manifest["feature_schema_sha256"] = upstream[
        os.path.join("artifacts", "feature_schema.json")]

    manifest["fit_partition"] = FIT_PARTITION
    manifest["fit_period"] = FIT_PERIOD
    manifest["evaluation_partition"] = EVALUATION_PARTITION
    manifest["evaluation_period"] = EVALUATION_PERIOD
    manifest["evaluation_period_label"] = EVALUATION_PERIOD_LABEL
    manifest["n_train"] = int(X_train.shape[0])
    manifest["n_validation"] = int(X_validation.shape[0])

    manifest["upstream_sha256"] = upstream
    manifest["training_design_diagnostics"] = diagnostics
    manifest["intercept"] = intercept

    manifest["metric_definitions"] = METRIC_DEFINITIONS
    manifest["metric_implementation"] = (
        "sklearn.metrics 0.20.0; RMSE computed as sqrt(mean_squared_error) "
        "because the squared= keyword did not exist until 0.22")
    validation_metrics = OrderedDict()
    validation_metrics["evaluation_period_label"] = EVALUATION_PERIOD_LABEL
    validation_metrics["result_status"] = (
        "development validation result - NOT final held-out test "
        "performance and NOT a project headline figure")
    validation_metrics["mae"] = metrics["mae"]
    validation_metrics["rmse"] = metrics["rmse"]
    validation_metrics["r2"] = metrics["r2"]
    manifest["validation_metrics"] = validation_metrics
    manifest["m0_comparison"] = comparison
    manifest["residual_diagnostics"] = residuals_summary
    manifest["equation_reconstruction_check"] = equation_check
    manifest["prediction_invariants"] = invariants
    manifest["row_alignment"] = alignment

    manifest["coefficient_interpretation_caution"] = (
        "Phase 3 found strong pairwise relationships among the "
        "meteorological predictors (DEWP-TEMP +0.825, TEMP-PRES -0.827, "
        "DEWP-PRES -0.778). Individual coefficients are therefore not "
        "stable or individually interpretable, are not comparable across "
        "features because units differ and no scaling was applied, and are "
        "relative to an omitted reference level for every one-hot block. "
        "They describe a conditional association within this fitted linear "
        "specification, not a causal effect.")

    outputs = OrderedDict()
    for relative, digest, rows in (
            (os.path.join(RESULTS_SUBDIR, COEFFICIENTS_FILENAME),
             coefficients_sha, len(coefficient_frame)),
            (os.path.join(RESULTS_SUBDIR, DIAGNOSTICS_FILENAME),
             diagnostics_sha, len(diagnostics_frame)),
            (os.path.join(RESULTS_SUBDIR, METRICS_FILENAME),
             metrics_sha, len(metrics_frame)),
            (os.path.join(RESULTS_SUBDIR, PREDICTIONS_FILENAME),
             predictions_sha, len(prediction_frame))):
        outputs[relative] = OrderedDict([("sha256", digest),
                                         ("rows", int(rows))])
    for relative, path in (
            (os.path.join("figures", FIGURE_ACTUAL_VS_PREDICTED),
             actual_path),
            (os.path.join("figures", FIGURE_RESIDUALS_VS_PREDICTED),
             residual_path)):
        outputs[relative] = OrderedDict([("sha256", sha256_of_file(path))])
    manifest["outputs"] = outputs

    manifest["model_serialization"] = OrderedDict([
        ("serialized", False),
        ("reason", "Protocol Phase 10 refits M1 on the full permitted "
                   "2010-2013 development period, so a serialized "
                   "development-only model would not be the final model. "
                   "The coefficients, intercept, diagnostics, predictions "
                   "and metrics fully and deterministically describe this "
                   "fit."),
    ])

    quarantine = OrderedDict()
    quarantine["test_partition"] = "2014"
    quarantine["test_target_quarantine_status"] = "ACTIVE"
    quarantine["test_features_read"] = False
    quarantine["test_target_read"] = False
    quarantine["test_predictions_generated"] = False
    quarantine["test_evaluation_status"] = "not_evaluated"
    quarantine["final_test_status"] = "not_evaluated"
    quarantine["quarantine_lifts_at"] = "Protocol Phase 10"
    manifest["test_quarantine"] = quarantine

    manifest["final_refit_policy"] = OrderedDict([
        ("executed", False),
        ("policy", "At Protocol Phase 10, after all model-development "
                   "decisions are frozen: combine 2010-2013 development "
                   "data, apply the already-frozen Phase 5 feature policy, "
                   "refit M1 once, generate 2014 predictions, open the "
                   "locked 2014 target and calculate MAE/RMSE/R2 once."),
    ])
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
    log("Upstream artifacts unchanged after M1")
    return manifest
