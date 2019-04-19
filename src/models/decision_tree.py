"""M2 decision tree regression for AirSense V1.

Protocol Phase 8. The first model requiring hyperparameter selection,
fitted after M1 linear regression and before M3 random forest.

M2 uses the **exact same frozen 43-feature matrix as M1**. Nothing about the
representation is altered to suit a tree, so the M1-vs-M2 comparison
reflects model class rather than hidden differences in feature engineering.

Search design, frozen in this module before any tree was fitted:

* ``max_depth`` in {3, 5, 8, 12, None} - spanning a high-bias stump through
  an unrestricted reference tree.
* ``min_samples_leaf`` in {1, 10, 50, 100} - from fully local leaves to
  strong terminal-node smoothing.
* Everything else fixed: ``criterion="mse"`` (the scikit-learn 0.20 name;
  ``"squared_error"`` did not exist until 1.0), ``splitter="best"``,
  ``min_samples_split=2``, ``max_features=None``, ``random_state=42``.

That is exactly 5 x 4 = 20 candidates. **No candidate may be added or
removed after results are seen**, and no neighbouring value may be probed
once a winner emerges - doing either would make the declared grid
meaningless.

Selection is by **lowest 2013 validation MAE**, the metric frozen in
Phase 0, with a tie-break policy also fixed in advance. Training metrics are
computed for every candidate as an overfitting diagnostic only; they play no
part in selection.

An explicit deterministic loop is used rather than ``GridSearchCV``: the
evaluation design is a single declared validation *year*, not
cross-validation folds, and a plain loop makes the candidate order and the
per-candidate diagnostics auditable.

**Test-set quarantine.** The 2014 partition is locked until Protocol
Phase 10. This module has no path to, and no logic for, the test features,
the test target or 2014 evaluation of any kind. In particular, no 2014
information may influence hyperparameter selection.

Historical constraint: CPython 3.6.7 with numpy 1.15.4, pandas 0.23.4,
scikit-learn 0.20.0, matplotlib 3.0.2.
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
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.tree import DecisionTreeRegressor

from src.data.audit import sha256_of_file

# ----------------------------------------------------------------------
# Frozen declaration
# ----------------------------------------------------------------------

MODEL_ID = "M2"
ESTIMATOR = "sklearn.tree.DecisionTreeRegressor"

FIT_PARTITION = "development_train"
FIT_PERIOD = "2010-01-01 00:00:00 to 2012-12-31 23:00:00"
EVALUATION_PARTITION = "validation"
EVALUATION_PERIOD_LABEL = "2013_validation"
EVALUATION_PERIOD = "2013-01-01 00:00:00 to 2013-12-31 23:00:00"

EXPECTED_TRAIN_ROWS = 24418
EXPECTED_VALIDATION_ROWS = 8678
EXPECTED_FEATURE_COUNT = 43

TARGET = "pm2.5"

EXPECTED_RAW_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")

# --- THE COMPLETE DECLARED SEARCH SPACE -------------------------------
# Frozen before any decision tree was fitted. Searched values:
SEARCH_MAX_DEPTH = [3, 5, 8, 12, None]
SEARCH_MIN_SAMPLES_LEAF = [1, 10, 50, 100]
# Held constant and NOT searched:
FIXED_CRITERION = "mse"            # sklearn 0.20 name; not "squared_error"
FIXED_SPLITTER = "best"
FIXED_MIN_SAMPLES_SPLIT = 2
FIXED_MAX_FEATURES = None
FIXED_RANDOM_STATE = 42
EXPECTED_CANDIDATE_COUNT = 20

NOT_SEARCHED = ["criterion", "splitter", "min_samples_split",
                "max_features", "max_leaf_nodes", "min_impurity_decrease"]

SELECTION_METRIC = "validation_mae"
SELECTION_TOLERANCE = 1e-12

TIE_BREAK_POLICY = [
    "1. lowest 2013 validation MAE (ties within 1e-12 treated as equal)",
    "2. lower 2013 validation RMSE",
    "3. simpler tree: smaller configured max_depth, with None treated as "
    "greater than every finite depth",
    "4. if max_depth equal: larger min_samples_leaf",
    "5. deterministic grid order (max_depth outer, min_samples_leaf inner)",
]

METRIC_DEFINITIONS = OrderedDict([
    ("MAE", "mean(abs(y_true - y_pred))"),
    ("RMSE", "sqrt(mean((y_true - y_pred) ** 2))"),
    ("R2", "1 - SS_res / SS_tot"),
])
RESIDUAL_DEFINITION = "residual = actual_pm25 - predicted_pm25"

RESULTS_SUBDIR = os.path.join("results", "models", "m2")
GRID_FILENAME = "m2_grid_results.csv"
METRICS_FILENAME = "m2_validation_metrics.csv"
PREDICTIONS_FILENAME = "m2_validation_predictions.csv"
IMPORTANCES_FILENAME = "m2_feature_importances.csv"
MANIFEST_FILENAME = "m2_decision_tree_manifest.json"

FIGURE_ACTUAL_VS_PREDICTED = "m2_validation_actual_vs_predicted.png"
FIGURE_RESIDUALS_VS_PREDICTED = "m2_validation_residuals_vs_predicted.png"
FIGURE_GRID = "m2_grid_mae_by_complexity.png"

FIGURE_DPI = 100
PLOT_COLOR = "#31688e"
ACCENT_COLOR = "#b03a2e"
SERIES_COLORS = ["#31688e", "#35b779", "#b03a2e", "#6a51a3"]

# Sorting sentinel: unrestricted depth is "more complex" than any finite
# depth, so it sorts last when preferring the simpler tree.
DEPTH_SENTINEL = float("inf")


class DecisionTreeError(RuntimeError):
    """Raised when a declared M2 gate fails."""


def build_grid():
    """The 20 declared candidates, in deterministic order.

    max_depth is the outer loop and min_samples_leaf the inner one; that
    order is the final tie-break and is fixed here.
    """
    candidates = []
    for depth in SEARCH_MAX_DEPTH:
        for leaf in SEARCH_MIN_SAMPLES_LEAF:
            candidates.append(OrderedDict([
                ("candidate_id", len(candidates) + 1),
                ("criterion", FIXED_CRITERION),
                ("splitter", FIXED_SPLITTER),
                ("max_depth", depth),
                ("min_samples_split", FIXED_MIN_SAMPLES_SPLIT),
                ("min_samples_leaf", leaf),
                ("max_features", FIXED_MAX_FEATURES),
                ("random_state", FIXED_RANDOM_STATE),
            ]))
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise DecisionTreeError(
            "grid has %d candidates, expected %d"
            % (len(candidates), EXPECTED_CANDIDATE_COUNT))
    return candidates


# ----------------------------------------------------------------------
# Input verification
# ----------------------------------------------------------------------

def verify_upstream(project_root):
    """Verify every frozen artifact M2 depends on. No test path appears."""
    checks = OrderedDict()

    raw_path = os.path.join(project_root, "data", "raw",
                            "PRSA_data_2010.1.1-2014.12.31.csv")
    raw_sha = sha256_of_file(raw_path)
    if raw_sha != EXPECTED_RAW_SHA256:
        raise DecisionTreeError(
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
        observed = sha256_of_file(os.path.join(features_dir, filename))
        recorded = schema["outputs"].get(relative)
        if recorded is None:
            raise DecisionTreeError("%s not recorded in feature_schema.json"
                                    % relative)
        if observed != recorded["sha256"]:
            raise DecisionTreeError(
                "%s SHA-256 changed: schema %s, found %s"
                % (relative, recorded["sha256"], observed))
        checks[relative] = observed

    for relative in (os.path.join("artifacts", "split_manifest.json"),
                     os.path.join("artifacts", "feature_schema.json"),
                     os.path.join("artifacts", "feature_row_manifest.csv"),
                     os.path.join("artifacts", "m0_baseline_manifest.json"),
                     os.path.join("artifacts",
                                  "m1_linear_regression_manifest.json")):
        path = os.path.join(project_root, relative)
        if not os.path.isfile(path):
            raise DecisionTreeError("%s not found" % relative)
        checks[relative] = sha256_of_file(path)

    return checks, schema


def load_matrix(path, expected_rows, feature_names, label):
    frame = pd.read_csv(path)
    if list(frame.columns) != feature_names:
        raise DecisionTreeError(
            "%s: column order differs from the frozen feature schema" % label)
    if frame.shape != (expected_rows, EXPECTED_FEATURE_COUNT):
        raise DecisionTreeError(
            "%s: shape %s, expected (%d, %d)"
            % (label, frame.shape, expected_rows, EXPECTED_FEATURE_COUNT))
    values = frame.values.astype(np.float64)
    if bool(np.isnan(values).any()) or bool(np.isinf(values).any()):
        raise DecisionTreeError("%s: feature matrix is not finite" % label)
    return values


def load_target(path, expected_rows, label):
    frame = pd.read_csv(path)
    if list(frame.columns) != [TARGET]:
        raise DecisionTreeError(
            "%s: expected a single column %r, found %s"
            % (label, TARGET, list(frame.columns)))
    if len(frame) != expected_rows:
        raise DecisionTreeError("%s: %d rows, expected %d"
                                % (label, len(frame), expected_rows))
    values = frame[TARGET].values.astype(np.float64)
    if bool(np.isnan(values).any()) or bool(np.isinf(values).any()):
        raise DecisionTreeError("%s: target is not finite" % label)
    return values


# ----------------------------------------------------------------------
# Fit, evaluate, diagnose
# ----------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """Exactly the three metrics the protocol pre-registered."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    return (mae, float(np.sqrt(mse)), float(r2_score(y_true, y_pred)))


def tree_structure(model):
    """Structural statistics of a fitted tree.

    ``get_n_leaves`` arrived in scikit-learn 0.22, so leaves are counted
    directly from the underlying tree: a node is a leaf exactly when it has
    no left child.
    """
    tree = model.tree_
    leaf_count = int((tree.children_left == -1).sum())
    return OrderedDict([
        ("observed_tree_depth", int(tree.max_depth)),
        ("node_count", int(tree.node_count)),
        ("leaf_count", leaf_count),
    ])


def fit_candidate(config, X_train, y_train, X_validation, y_validation):
    """Fit one declared candidate and evaluate it. Nothing is selected here."""
    model = DecisionTreeRegressor(
        criterion=config["criterion"],
        splitter=config["splitter"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        random_state=config["random_state"],
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    validation_pred = model.predict(X_validation)
    train_mae, train_rmse, train_r2 = compute_metrics(y_train, train_pred)
    val_mae, val_rmse, val_r2 = compute_metrics(y_validation, validation_pred)

    row = OrderedDict(config)
    row["train_mae"] = train_mae
    row["train_rmse"] = train_rmse
    row["train_r2"] = train_r2
    row["validation_mae"] = val_mae
    row["validation_rmse"] = val_rmse
    row["validation_r2"] = val_r2
    row.update(tree_structure(model))
    return row, model, validation_pred


def selection_key(row):
    """The frozen tie-break ordering, as a sort key.

    Applied to rows whose MAE ties the best within ``SELECTION_TOLERANCE``.
    """
    depth = row["max_depth"]
    depth_rank = DEPTH_SENTINEL if depth is None else float(depth)
    return (
        row["validation_rmse"],        # 2. lower RMSE
        depth_rank,                    # 3. simpler tree: smaller max_depth
        -float(row["min_samples_leaf"]),  # 4. larger min_samples_leaf
        row["candidate_id"],           # 5. deterministic grid order
    )


def select_candidate(rows):
    """Apply the frozen selection rule. Returns the winning row.

    Primary criterion is lowest 2013 validation MAE. Only rows tying the
    best MAE within 1e-12 enter the tie-break.
    """
    if len(rows) != EXPECTED_CANDIDATE_COUNT:
        raise DecisionTreeError(
            "%d candidate results, expected %d"
            % (len(rows), EXPECTED_CANDIDATE_COUNT))
    best_mae = min(row["validation_mae"] for row in rows)
    tied = [row for row in rows
            if abs(row["validation_mae"] - best_mae) <= SELECTION_TOLERANCE]
    tied.sort(key=selection_key)
    return tied[0], len(tied)


# ----------------------------------------------------------------------
# Comparison with the frozen prior models
# ----------------------------------------------------------------------

def load_prior_metrics(project_root):
    """Read M0 and M1 results from their manifests, never hard-coded."""
    prior = OrderedDict()
    for key, filename in (("m0", "m0_baseline_manifest.json"),
                          ("m1", "m1_linear_regression_manifest.json")):
        path = os.path.join(project_root, "artifacts", filename)
        manifest = json.load(io.open(path, encoding="utf-8"))
        metrics = manifest["validation_metrics"]
        prior[key] = OrderedDict([
            ("source", os.path.join("artifacts", filename)),
            ("manifest_sha256", sha256_of_file(path)),
            ("mae", float(metrics["mae"])),
            ("rmse", float(metrics["rmse"])),
            ("r2", float(metrics["r2"])),
        ])
    return prior


def build_comparison(selected, prior):
    """Compare the selected M2 against M0 and M1 on 2013 validation."""
    mae = selected["validation_mae"]
    rmse = selected["validation_rmse"]
    r2 = selected["validation_r2"]

    comparison = OrderedDict()
    comparison["evaluation_period_label"] = EVALUATION_PERIOD_LABEL
    comparison["primary_comparison_metric"] = "MAE"
    comparison["m2_mae"] = mae
    comparison["m2_rmse"] = rmse
    comparison["m2_r2"] = r2
    for key in ("m0", "m1"):
        entry = prior[key]
        block = OrderedDict()
        block["source"] = entry["source"]
        block["manifest_sha256"] = entry["manifest_sha256"]
        block["mae"] = entry["mae"]
        block["rmse"] = entry["rmse"]
        block["r2"] = entry["r2"]
        block["mae_improvement_absolute"] = entry["mae"] - mae
        block["mae_improvement_percent"] = (
            (entry["mae"] - mae) / entry["mae"] * 100.0)
        block["rmse_improvement_absolute"] = entry["rmse"] - rmse
        block["rmse_improvement_percent"] = (
            (entry["rmse"] - rmse) / entry["rmse"] * 100.0)
        block["r2_delta"] = r2 - entry["r2"]
        comparison[key] = block
    comparison["note"] = (
        "Development-validation comparison on 2013 only. The primary "
        "comparison metric remains MAE, as pre-registered.")
    return comparison


# ----------------------------------------------------------------------
# Diagnostics for the selected model
# ----------------------------------------------------------------------

def residual_diagnostics(y_true, y_pred, y_train):
    """Descriptive residual summary. Nothing here modifies M2.

    Also records the prediction-range check: a decision tree predicts leaf
    means of observed training targets, so every prediction should lie
    inside the training target range. Predictions are never clipped.
    """
    residuals = y_true - y_pred
    correlation = float(np.corrcoef(y_pred, residuals)[0, 1])
    train_min = float(y_train.min())
    train_max = float(y_train.max())
    tolerance = 1e-9
    within = bool((y_pred >= train_min - tolerance).all()
                  and (y_pred <= train_max + tolerance).all())
    negative = int((y_pred < 0.0).sum())
    return OrderedDict([
        ("residual_definition", RESIDUAL_DEFINITION),
        ("n", int(residuals.size)),
        ("mean_residual", float(residuals.mean())),
        ("median_residual", float(np.median(residuals))),
        ("residual_std", float(residuals.std(ddof=1))),
        ("min_residual", float(residuals.min())),
        ("max_residual", float(residuals.max())),
        ("pearson_corr_predicted_vs_residual", correlation),
        ("n_negative_predictions", negative),
        ("percent_negative_predictions",
         100.0 * negative / y_pred.size if y_pred.size else 0.0),
        ("prediction_min", float(y_pred.min())),
        ("prediction_max", float(y_pred.max())),
        ("train_target_min", train_min),
        ("train_target_max", train_max),
        ("predictions_within_training_target_range", within),
        ("range_check_tolerance", tolerance),
        ("predictions_clipped", False),
    ])


def verify_prediction_invariants(predictions, y_true):
    report = OrderedDict()
    report["n_predictions"] = int(predictions.size)
    if predictions.size != EXPECTED_VALIDATION_ROWS:
        raise DecisionTreeError(
            "%d predictions, expected %d"
            % (predictions.size, EXPECTED_VALIDATION_ROWS))
    if predictions.size != y_true.size:
        raise DecisionTreeError("prediction and actual counts differ")
    if not bool(np.isfinite(predictions).all()):
        raise DecisionTreeError("predictions contain non-finite values")
    if not bool(np.isfinite(y_true).all()):
        raise DecisionTreeError("actuals contain non-finite values")
    residuals = y_true - predictions
    if not bool(np.isfinite(residuals).all()):
        raise DecisionTreeError("residuals contain non-finite values")
    distinct = int(np.unique(predictions).size)
    if distinct < 2:
        raise DecisionTreeError(
            "M2 produced fewer than two distinct predictions")
    report["predictions_finite"] = True
    report["actuals_finite"] = True
    report["residuals_finite"] = True
    report["n_distinct_predictions"] = distinct
    report["prediction_std"] = float(predictions.std(ddof=1))
    return report


def load_validation_metadata(project_root, expected_rows):
    path = os.path.join(project_root, "artifacts",
                        "feature_row_manifest.csv")
    manifest = pd.read_csv(path)
    if TARGET in manifest.columns:
        raise DecisionTreeError("row manifest unexpectedly carries the target")
    block = manifest[manifest["partition"] == EVALUATION_PARTITION]
    block = block.reset_index(drop=True)
    if len(block) != expected_rows:
        raise DecisionTreeError(
            "validation manifest has %d rows, expected %d"
            % (len(block), expected_rows))
    if not bool((block["matrix_row_number"].values
                 == np.arange(expected_rows)).all()):
        raise DecisionTreeError(
            "validation matrix_row_number is not sequential from 0")
    return block


def verify_row_alignment(project_root, metadata, y_validation):
    source = pd.read_csv(
        os.path.join(project_root, "data", "processed",
                     "airsense_validation_2013.csv"))
    if not bool((metadata["No"].values == source["No"].values).all()):
        raise DecisionTreeError(
            "manifest identifiers do not match the Phase 4 validation rows")
    if not bool((y_validation
                 == source[TARGET].values.astype(np.float64)).all()):
        raise DecisionTreeError(
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
# Figures - at most three, deterministic
# ----------------------------------------------------------------------

def _save(fig, path):
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=FIGURE_DPI, metadata={"Software": "AirSense V1"})
    except TypeError:                            # pragma: no cover
        fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def figure_actual_vs_predicted(y_true, y_pred, label, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(y_pred, y_true, s=6, alpha=0.15, color=PLOT_COLOR,
               edgecolors="none")
    lower = float(min(y_pred.min(), y_true.min()))
    upper = float(max(y_pred.max(), y_true.max()))
    ax.plot([lower, upper], [lower, upper], color=ACCENT_COLOR,
            linestyle="--", linewidth=1.2, label="y = x (perfect prediction)")
    ax.set_xlabel("Predicted PM2.5 (ug/m^3)")
    ax.set_ylabel("Actual PM2.5 (ug/m^3)")
    ax.set_title("M2 Decision Tree - actual vs predicted\n"
                 "%s\n2013 DEVELOPMENT VALIDATION (not final test)" % label)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, path)


def figure_residuals_vs_predicted(y_true, y_pred, label, path):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(y_pred, residuals, s=6, alpha=0.15, color=PLOT_COLOR,
               edgecolors="none")
    ax.axhline(0.0, color=ACCENT_COLOR, linestyle="--", linewidth=1.2,
               label="zero residual")
    ax.set_xlabel("Predicted PM2.5 (ug/m^3)")
    ax.set_ylabel("Residual = actual - predicted (ug/m^3)")
    ax.set_title("M2 Decision Tree - residuals vs predicted\n"
                 "%s\n2013 DEVELOPMENT VALIDATION (not final test)" % label)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, path)


def figure_grid(rows, selected, path):
    """Validation MAE across the declared grid.

    One line per min_samples_leaf, plotted against max_depth. Unrestricted
    depth is drawn at the right-hand end and tick-labelled "None"; it is not
    a numeric depth and the axis is categorical for that reason.
    """
    depth_labels = ["3", "5", "8", "12", "None"]
    positions = np.arange(len(SEARCH_MAX_DEPTH))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for index, leaf in enumerate(SEARCH_MIN_SAMPLES_LEAF):
        series = [row["validation_mae"] for row in rows
                  if row["min_samples_leaf"] == leaf]
        ax.plot(positions, series, marker="o", markersize=5,
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                label="min_samples_leaf = %d" % leaf)

    selected_position = SEARCH_MAX_DEPTH.index(selected["max_depth"])
    ax.scatter([selected_position], [selected["validation_mae"]],
               s=160, facecolors="none", edgecolors=ACCENT_COLOR,
               linewidths=2.0, zorder=5,
               label="selected (candidate %d)" % selected["candidate_id"])

    ax.set_xticks(positions)
    ax.set_xticklabels(depth_labels)
    ax.set_xlabel("max_depth (categorical; None = unrestricted)")
    ax.set_ylabel("2013 validation MAE (ug/m^3)")
    ax.set_title("M2 Decision Tree - validation MAE across the declared "
                 "20-candidate grid\n2013 DEVELOPMENT VALIDATION "
                 "(not final test)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, path)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def _config_label(row):
    depth = "None" if row["max_depth"] is None else str(row["max_depth"])
    return "max_depth=%s, min_samples_leaf=%d" % (depth,
                                                  row["min_samples_leaf"])


def run_m2_decision_tree(project_root, verbose=True):
    """Evaluate the 20 declared candidates, select one, and report."""

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
        raise DecisionTreeError(
            "feature schema declares %d features, expected %d"
            % (len(feature_names), EXPECTED_FEATURE_COUNT))
    log("Upstream verified; frozen schema has %d features"
        % len(feature_names))

    features_dir = os.path.join(project_root, "data", "processed", "features")
    X_train = load_matrix(os.path.join(features_dir, "X_train.csv"),
                          EXPECTED_TRAIN_ROWS, feature_names, "X_train")
    X_validation = load_matrix(
        os.path.join(features_dir, "X_validation.csv"),
        EXPECTED_VALIDATION_ROWS, feature_names, "X_validation")
    y_train = load_target(os.path.join(features_dir, "y_train.csv"),
                          EXPECTED_TRAIN_ROWS, "y_train")
    y_validation = load_target(
        os.path.join(features_dir, "y_validation.csv"),
        EXPECTED_VALIDATION_ROWS, "y_validation")
    log("Loaded X_train %s, X_validation %s"
        % (X_train.shape, X_validation.shape))

    # -- evaluate every declared candidate -------------------------------
    grid = build_grid()
    log("Evaluating %d declared candidates (grid frozen before fitting)"
        % len(grid))
    rows = []
    fitted = {}
    predictions_by_id = {}
    for config in grid:
        row, model, validation_pred = fit_candidate(
            config, X_train, y_train, X_validation, y_validation)
        rows.append(row)
        fitted[config["candidate_id"]] = model
        predictions_by_id[config["candidate_id"]] = validation_pred
        log("  candidate %2d  %-38s  val MAE %10.6f  depth %2d  leaves %6d"
            % (row["candidate_id"], _config_label(row), row["validation_mae"],
               row["observed_tree_depth"], row["leaf_count"]))

    # -- grid invariants --------------------------------------------------
    combinations = set()
    for row in rows:
        key = (row["criterion"], row["splitter"], row["max_depth"],
               row["min_samples_split"], row["min_samples_leaf"],
               row["max_features"], row["random_state"])
        if key in combinations:
            raise DecisionTreeError("duplicate hyperparameter combination")
        combinations.add(key)
        if row["max_depth"] not in SEARCH_MAX_DEPTH:
            raise DecisionTreeError("undeclared max_depth %r"
                                    % (row["max_depth"],))
        if row["min_samples_leaf"] not in SEARCH_MIN_SAMPLES_LEAF:
            raise DecisionTreeError("undeclared min_samples_leaf %r"
                                    % (row["min_samples_leaf"],))
        for name in ("train_mae", "train_rmse", "train_r2",
                     "validation_mae", "validation_rmse", "validation_r2"):
            if not np.isfinite(row[name]):
                raise DecisionTreeError(
                    "candidate %d has non-finite %s"
                    % (row["candidate_id"], name))
    if len(combinations) != EXPECTED_CANDIDATE_COUNT:
        raise DecisionTreeError(
            "%d unique combinations, expected %d"
            % (len(combinations), EXPECTED_CANDIDATE_COUNT))

    # -- selection by the frozen rule ------------------------------------
    selected, n_tied = select_candidate(rows)
    selected_id = selected["candidate_id"]
    log("")
    log("Selected candidate %d: %s  (validation MAE %.6f; %d candidate(s) "
        "tied on MAE)" % (selected_id, _config_label(selected),
                          selected["validation_mae"], n_tied))

    model = fitted[selected_id]
    predictions = predictions_by_id[selected_id]

    prior = load_prior_metrics(project_root)
    comparison = build_comparison(selected, prior)
    residuals_summary = residual_diagnostics(y_validation, predictions,
                                             y_train)
    invariants = verify_prediction_invariants(predictions, y_validation)
    metadata = load_validation_metadata(project_root,
                                        EXPECTED_VALIDATION_ROWS)
    alignment = verify_row_alignment(project_root, metadata, y_validation)

    # -- selected tree structure -----------------------------------------
    tree = model.tree_
    root_feature_index = int(tree.feature[0])
    if root_feature_index < 0:
        root_feature = None            # a single-leaf tree has no split
    else:
        root_feature = feature_names[root_feature_index]
    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    structure = OrderedDict([
        ("configured_max_depth", selected["max_depth"]),
        ("configured_min_samples_leaf", selected["min_samples_leaf"]),
        ("observed_tree_depth", selected["observed_tree_depth"]),
        ("node_count", selected["node_count"]),
        ("leaf_count", selected["leaf_count"]),
        ("root_split_feature", root_feature),
        ("n_nonzero_importances", int((importances > 0.0).sum())),
        ("importance_sum", float(importances.sum())),
    ])
    log("Selected tree: depth %d, %d nodes, %d leaves, root split on %s"
        % (structure["observed_tree_depth"], structure["node_count"],
           structure["leaf_count"], root_feature))

    # -- artifacts ---------------------------------------------------------
    grid_frame = pd.DataFrame([
        OrderedDict(list(row.items())
                    + [("selected", bool(row["candidate_id"] == selected_id))])
        for row in rows])
    grid_path = os.path.join(results_dir, GRID_FILENAME)
    grid_frame.to_csv(grid_path, index=False)
    grid_sha = sha256_of_file(grid_path)
    if int(grid_frame["selected"].sum()) != 1:
        raise DecisionTreeError("exactly one candidate must be selected")

    importance_frame = pd.DataFrame(OrderedDict([
        ("feature", feature_names),
        ("importance", importances),
        ("rank", pd.Series(importances).rank(
            ascending=False, method="min").astype(int).values),
    ]))
    importances_path = os.path.join(results_dir, IMPORTANCES_FILENAME)
    importance_frame.to_csv(importances_path, index=False)
    importances_sha = sha256_of_file(importances_path)

    metrics_frame = pd.DataFrame([OrderedDict([
        ("model", MODEL_ID),
        ("criterion", selected["criterion"]),
        ("splitter", selected["splitter"]),
        ("max_depth", "None" if selected["max_depth"] is None
         else selected["max_depth"]),
        ("min_samples_split", selected["min_samples_split"]),
        ("min_samples_leaf", selected["min_samples_leaf"]),
        ("max_features", "None"),
        ("random_state", selected["random_state"]),
        ("fit_period", FIT_PERIOD),
        ("evaluation_period", EVALUATION_PERIOD_LABEL),
        ("n_train", int(X_train.shape[0])),
        ("n_validation", int(X_validation.shape[0])),
        ("n_features", int(X_train.shape[1])),
        ("train_mae", selected["train_mae"]),
        ("train_rmse", selected["train_rmse"]),
        ("train_r2", selected["train_r2"]),
        ("validation_mae", selected["validation_mae"]),
        ("validation_rmse", selected["validation_rmse"]),
        ("validation_r2", selected["validation_r2"]),
        ("m0_mae", prior["m0"]["mae"]),
        ("m1_mae", prior["m1"]["mae"]),
        ("mae_improvement_vs_m0_absolute",
         comparison["m0"]["mae_improvement_absolute"]),
        ("mae_improvement_vs_m0_percent",
         comparison["m0"]["mae_improvement_percent"]),
        ("mae_improvement_vs_m1_absolute",
         comparison["m1"]["mae_improvement_absolute"]),
        ("mae_improvement_vs_m1_percent",
         comparison["m1"]["mae_improvement_percent"]),
        ("m0_rmse", prior["m0"]["rmse"]),
        ("m1_rmse", prior["m1"]["rmse"]),
        ("rmse_delta_vs_m0", comparison["m0"]["rmse_improvement_absolute"]),
        ("rmse_delta_vs_m1", comparison["m1"]["rmse_improvement_absolute"]),
        ("m0_r2", prior["m0"]["r2"]),
        ("m1_r2", prior["m1"]["r2"]),
        ("r2_delta_vs_m0", comparison["m0"]["r2_delta"]),
        ("r2_delta_vs_m1", comparison["m1"]["r2_delta"]),
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
    log("Wrote grid results, metrics, predictions and feature importances")

    label = _config_label(selected)
    actual_path = os.path.join(figures_dir, FIGURE_ACTUAL_VS_PREDICTED)
    figure_actual_vs_predicted(y_validation, predictions, label, actual_path)
    residual_path = os.path.join(figures_dir, FIGURE_RESIDUALS_VS_PREDICTED)
    figure_residuals_vs_predicted(y_validation, predictions, label,
                                  residual_path)
    grid_fig_path = os.path.join(figures_dir, FIGURE_GRID)
    figure_grid(rows, selected, grid_fig_path)
    log("Wrote 3 figures")

    # -- manifest ----------------------------------------------------------
    manifest = OrderedDict()
    manifest["model_id"] = MODEL_ID
    manifest["estimator"] = ESTIMATOR
    manifest["sklearn_version"] = __import__("sklearn").__version__
    manifest["phase"] = "V1 / Protocol Phase 8 - decision tree (M2)"
    manifest["feature_count"] = int(len(feature_names))
    manifest["feature_names"] = feature_names
    manifest["feature_representation_note"] = (
        "Exactly the frozen Phase 5 43-feature matrix, identical to M1. "
        "Nothing was altered to suit a tree, so the M1-vs-M2 comparison "
        "reflects model class rather than feature engineering.")

    search_space = OrderedDict()
    search_space["searched"] = OrderedDict([
        ("max_depth", ["None" if d is None else d
                       for d in SEARCH_MAX_DEPTH]),
        ("min_samples_leaf", list(SEARCH_MIN_SAMPLES_LEAF)),
    ])
    search_space["fixed"] = OrderedDict([
        ("criterion", FIXED_CRITERION),
        ("splitter", FIXED_SPLITTER),
        ("min_samples_split", FIXED_MIN_SAMPLES_SPLIT),
        ("max_features", "None"),
        ("random_state", FIXED_RANDOM_STATE),
    ])
    search_space["not_searched"] = list(NOT_SEARCHED)
    search_space["candidate_count"] = EXPECTED_CANDIDATE_COUNT
    search_space["candidate_ordering"] = (
        "max_depth outer loop, min_samples_leaf inner loop, both in "
        "declared order")
    search_space["frozen_before_any_result"] = True
    search_space["method"] = (
        "explicit deterministic loop over the declared candidates; "
        "GridSearchCV was deliberately not used because the evaluation "
        "design is a single declared validation year, not "
        "cross-validation folds")
    search_space["criterion_api_note"] = (
        "criterion='mse' is the scikit-learn 0.20 name; 'squared_error' "
        "did not exist until 1.0 and would be a post-cutoff API")
    manifest["search_space"] = search_space

    manifest["selection_metric"] = SELECTION_METRIC
    manifest["selection_metric_note"] = (
        "2013 validation MAE, the decision metric frozen in Phase 0 long "
        "before any M2 result existed")
    manifest["selection_tolerance"] = SELECTION_TOLERANCE
    manifest["tie_break_policy"] = list(TIE_BREAK_POLICY)
    manifest["tie_break_policy_frozen_before_fitting"] = True
    manifest["n_candidates_tied_on_mae"] = int(n_tied)
    manifest["no_test_information_used_in_selection"] = True

    selected_config = OrderedDict([
        ("candidate_id", selected_id),
        ("criterion", selected["criterion"]),
        ("splitter", selected["splitter"]),
        ("max_depth", "None" if selected["max_depth"] is None
         else selected["max_depth"]),
        ("min_samples_split", selected["min_samples_split"]),
        ("min_samples_leaf", selected["min_samples_leaf"]),
        ("max_features", "None"),
        ("random_state", selected["random_state"]),
    ])
    manifest["selected_configuration"] = selected_config
    manifest["selected_tree_structure"] = structure

    manifest["fit_partition"] = FIT_PARTITION
    manifest["fit_period"] = FIT_PERIOD
    manifest["evaluation_partition"] = EVALUATION_PARTITION
    manifest["evaluation_period"] = EVALUATION_PERIOD
    manifest["evaluation_period_label"] = EVALUATION_PERIOD_LABEL
    manifest["n_train"] = int(X_train.shape[0])
    manifest["n_validation"] = int(X_validation.shape[0])
    manifest["upstream_sha256"] = upstream

    manifest["metric_definitions"] = METRIC_DEFINITIONS
    manifest["metric_implementation"] = (
        "sklearn.metrics 0.20.0; RMSE computed as sqrt(mean_squared_error) "
        "because the squared= keyword did not exist until 0.22")
    manifest["selected_training_metrics"] = OrderedDict([
        ("status", "diagnostic only - training metrics played no part in "
                   "selection"),
        ("mae", selected["train_mae"]),
        ("rmse", selected["train_rmse"]),
        ("r2", selected["train_r2"]),
    ])
    validation_metrics = OrderedDict()
    validation_metrics["evaluation_period_label"] = EVALUATION_PERIOD_LABEL
    validation_metrics["result_status"] = (
        "development validation result - NOT final held-out test "
        "performance and NOT a project headline figure")
    validation_metrics["mae"] = selected["validation_mae"]
    validation_metrics["rmse"] = selected["validation_rmse"]
    validation_metrics["r2"] = selected["validation_r2"]
    manifest["validation_metrics"] = validation_metrics
    manifest["model_comparison"] = comparison
    manifest["residual_diagnostics"] = residuals_summary
    manifest["prediction_invariants"] = invariants
    manifest["row_alignment"] = alignment

    manifest["feature_importance_caution"] = (
        "Impurity-based decision-tree feature importance is predictive, "
        "not causal. It favours variables offering many possible split "
        "points, distributes importance unpredictably among correlated "
        "predictors - Phase 3 found |r| of 0.78 to 0.83 among DEWP, TEMP "
        "and PRES - and reflects only this fitted partition. It must not "
        "be read as scientific mechanism, and it was not used to redesign "
        "the Phase 5 feature set. Zero-importance features were retained.")
    manifest["root_split_interpretation_caution"] = (
        "The root split feature is a property of this fitted partition, "
        "not evidence that the variable is a causally dominant driver.")

    outputs = OrderedDict()
    for relative, digest, rows_count in (
            (os.path.join(RESULTS_SUBDIR, GRID_FILENAME), grid_sha,
             len(grid_frame)),
            (os.path.join(RESULTS_SUBDIR, METRICS_FILENAME), metrics_sha,
             len(metrics_frame)),
            (os.path.join(RESULTS_SUBDIR, PREDICTIONS_FILENAME),
             predictions_sha, len(prediction_frame)),
            (os.path.join(RESULTS_SUBDIR, IMPORTANCES_FILENAME),
             importances_sha, len(importance_frame))):
        outputs[relative] = OrderedDict([("sha256", digest),
                                         ("rows", int(rows_count))])
    for relative, path in (
            (os.path.join("figures", FIGURE_ACTUAL_VS_PREDICTED),
             actual_path),
            (os.path.join("figures", FIGURE_RESIDUALS_VS_PREDICTED),
             residual_path),
            (os.path.join("figures", FIGURE_GRID), grid_fig_path)):
        outputs[relative] = OrderedDict([("sha256", sha256_of_file(path))])
    manifest["outputs"] = outputs

    manifest["model_serialization"] = OrderedDict([
        ("serialized", False),
        ("reason", "Protocol Phase 10 refits the selected configuration on "
                   "the full 2010-2013 development period, so a serialized "
                   "development-only tree would not be the final model. The "
                   "configuration, metrics, predictions, importances and "
                   "structural diagnostics describe this fit "
                   "deterministically."),
    ])
    manifest["test_quarantine"] = OrderedDict([
        ("test_partition", "2014"),
        ("test_target_quarantine_status", "ACTIVE"),
        ("test_features_read", False),
        ("test_target_read", False),
        ("test_predictions_generated", False),
        ("test_evaluation_status", "not_evaluated"),
        ("final_test_status", "not_evaluated"),
        ("quarantine_lifts_at", "Protocol Phase 10"),
        ("selection_used_test_information", False),
    ])
    manifest["final_refit_policy"] = OrderedDict([
        ("executed", False),
        ("policy", "At Protocol Phase 10 the selected M2 hyperparameters "
                   "remain frozen: combine 2010-2013 development data, "
                   "refit DecisionTreeRegressor with exactly those "
                   "parameters, generate 2014 predictions once, open the "
                   "2014 target once and calculate MAE/RMSE/R2 once. M2 is "
                   "not retuned after 2014 becomes visible."),
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
    log("Upstream artifacts unchanged after M2")
    return manifest
