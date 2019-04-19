"""M3 random forest regression for AirSense V1.

Protocol Phase 9. The **final development-model phase** before the locked
held-out evaluation.

M3 uses the **exact same frozen 43-feature matrix as M1 and M2**. No
model-specific feature engineering is permitted: the point of the M0-M3
comparison is model class, not different feature sets.

Search design, frozen in this module before any forest was fitted:

* ``max_depth`` in {8, 12, None}
* ``min_samples_leaf`` in {1, 10, 50, 100}
* ``max_features`` in {"auto", 0.5}

3 x 4 x 2 = **24 candidates**, in that nested order.

Fixed for every candidate and deliberately **not** searched:
``n_estimators=100``, ``criterion="mse"`` (the scikit-learn 0.20 name;
``"squared_error"`` did not exist until 1.0), ``bootstrap=True``,
``min_samples_split=2``, ``random_state=42``, ``n_jobs=1``,
``oob_score=False``.

Three of those deserve their reason recorded:

**n_estimators is fixed at 100.** 100 trees was a conventional,
computationally manageable ensemble size entirely available in 2019, and
gives substantially more averaging than scikit-learn 0.20's small default.
Fixing it before results means selection concerns tree structure and
feature subsampling rather than arbitrary ensemble size. 50, 200, 500 and
1000 are not tried.

**One seed only.** ``random_state=42`` for every candidate, with
``n_jobs=1`` so the result does not depend on thread scheduling. The seed
exists to make the historical experiment reproducible, not to be searched;
no best-of-seed result is reported.

**OOB scoring is off and is not used for selection.** The research design
already has an explicit chronological 2013 validation period. Out-of-bag
evaluation would introduce a second selection framework that ignores
temporal ordering, since OOB samples are drawn from across the whole
training period. 2013 validation MAE remains the sole primary criterion.

**Test-set quarantine.** The 2014 partition is locked until Protocol
Phase 10. This module has no path to, and no logic for, the test features,
the test target or 2014 evaluation of any kind, and no 2014 information may
influence hyperparameter selection.

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

from src.data.audit import sha256_of_file

# ----------------------------------------------------------------------
# Frozen declaration
# ----------------------------------------------------------------------

MODEL_ID = "M3"
ESTIMATOR = "sklearn.ensemble.RandomForestRegressor"

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
SEARCH_MAX_DEPTH = [8, 12, None]
SEARCH_MIN_SAMPLES_LEAF = [1, 10, 50, 100]
SEARCH_MAX_FEATURES = ["auto", 0.5]

FIXED_N_ESTIMATORS = 100
FIXED_CRITERION = "mse"
FIXED_BOOTSTRAP = True
FIXED_MIN_SAMPLES_SPLIT = 2
FIXED_RANDOM_STATE = 42
FIXED_N_JOBS = 1
FIXED_OOB_SCORE = False

EXPECTED_CANDIDATE_COUNT = 24

NOT_SEARCHED = ["n_estimators", "criterion", "bootstrap",
                "min_samples_split", "random_state", "n_jobs", "oob_score",
                "max_leaf_nodes", "min_impurity_decrease"]

SELECTION_METRIC = "validation_mae"
SELECTION_TOLERANCE = 1e-12

TIE_BREAK_POLICY = [
    "1. lowest 2013 validation MAE (ties within 1e-12 treated as equal)",
    "2. lower 2013 validation RMSE",
    "3. simpler forest: smaller finite max_depth, with None considered "
    "more complex than every finite value",
    "4. larger min_samples_leaf",
    "5. max_features=0.5 before max_features='auto', because fewer "
    "candidate predictors are exposed at each split",
    "6. deterministic candidate/grid order",
]

METRIC_DEFINITIONS = OrderedDict([
    ("MAE", "mean(abs(y_true - y_pred))"),
    ("RMSE", "sqrt(mean((y_true - y_pred) ** 2))"),
    ("R2", "1 - SS_res / SS_tot"),
])
RESIDUAL_DEFINITION = "residual = actual_pm25 - predicted_pm25"

# Sign convention, identical for every prior-model comparison:
#   mae_improvement_absolute = prior_MAE - M3_MAE   (positive = M3 better)
COMPARISON_SIGN_CONVENTION = (
    "improvement_absolute = prior_metric - m3_metric; positive means M3 "
    "improved on the prior model. improvement_percent = "
    "(prior - m3) / prior * 100. r2_delta = m3_r2 - prior_r2; positive "
    "means M3 explains more variance. The convention is identical for M0, "
    "M1 and M2.")

RESULTS_SUBDIR = os.path.join("results", "models", "m3")
GRID_FILENAME = "m3_grid_results.csv"
METRICS_FILENAME = "m3_validation_metrics.csv"
PREDICTIONS_FILENAME = "m3_validation_predictions.csv"
IMPORTANCES_FILENAME = "m3_feature_importances.csv"
MANIFEST_FILENAME = "m3_random_forest_manifest.json"
FREEZE_FILENAME = "pretest_development_freeze.json"

FIGURE_ACTUAL_VS_PREDICTED = "m3_validation_actual_vs_predicted.png"
FIGURE_RESIDUALS_VS_PREDICTED = "m3_validation_residuals_vs_predicted.png"
FIGURE_GRID = "m3_grid_mae_by_complexity.png"

FIGURE_DPI = 100
PLOT_COLOR = "#31688e"
ACCENT_COLOR = "#b03a2e"
SERIES_COLORS = ["#31688e", "#35b779", "#b03a2e", "#6a51a3"]

DEPTH_SENTINEL = float("inf")


class RandomForestError(RuntimeError):
    """Raised when a declared M3 gate fails."""


def _max_features_label(value):
    return "auto" if value == "auto" else ("%g" % value)


def build_grid():
    """The 24 declared candidates, in deterministic nested order.

    max_depth outer, min_samples_leaf middle, max_features inner. This
    ordering is the final tie-break and is fixed here.
    """
    candidates = []
    for depth in SEARCH_MAX_DEPTH:
        for leaf in SEARCH_MIN_SAMPLES_LEAF:
            for features in SEARCH_MAX_FEATURES:
                candidates.append(OrderedDict([
                    ("candidate_id", len(candidates) + 1),
                    ("n_estimators", FIXED_N_ESTIMATORS),
                    ("criterion", FIXED_CRITERION),
                    ("bootstrap", FIXED_BOOTSTRAP),
                    ("max_depth", depth),
                    ("min_samples_split", FIXED_MIN_SAMPLES_SPLIT),
                    ("min_samples_leaf", leaf),
                    ("max_features", features),
                    ("random_state", FIXED_RANDOM_STATE),
                    ("n_jobs", FIXED_N_JOBS),
                ]))
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise RandomForestError(
            "grid has %d candidates, expected %d"
            % (len(candidates), EXPECTED_CANDIDATE_COUNT))
    return candidates


# ----------------------------------------------------------------------
# Input verification
# ----------------------------------------------------------------------

def verify_upstream(project_root):
    """Verify every frozen artifact M3 depends on. No test path appears."""
    checks = OrderedDict()

    raw_path = os.path.join(project_root, "data", "raw",
                            "PRSA_data_2010.1.1-2014.12.31.csv")
    raw_sha = sha256_of_file(raw_path)
    if raw_sha != EXPECTED_RAW_SHA256:
        raise RandomForestError(
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
            raise RandomForestError("%s not recorded in feature_schema.json"
                                    % relative)
        if observed != recorded["sha256"]:
            raise RandomForestError(
                "%s SHA-256 changed: schema %s, found %s"
                % (relative, recorded["sha256"], observed))
        checks[relative] = observed

    for relative in (os.path.join("artifacts", "split_manifest.json"),
                     os.path.join("artifacts", "feature_schema.json"),
                     os.path.join("artifacts", "feature_row_manifest.csv"),
                     os.path.join("artifacts", "m0_baseline_manifest.json"),
                     os.path.join("artifacts",
                                  "m1_linear_regression_manifest.json"),
                     os.path.join("artifacts",
                                  "m2_decision_tree_manifest.json")):
        path = os.path.join(project_root, relative)
        if not os.path.isfile(path):
            raise RandomForestError("%s not found" % relative)
        checks[relative] = sha256_of_file(path)

    return checks, schema


def load_matrix(path, expected_rows, feature_names, label):
    frame = pd.read_csv(path)
    if list(frame.columns) != feature_names:
        raise RandomForestError(
            "%s: column order differs from the frozen feature schema" % label)
    if frame.shape != (expected_rows, EXPECTED_FEATURE_COUNT):
        raise RandomForestError(
            "%s: shape %s, expected (%d, %d)"
            % (label, frame.shape, expected_rows, EXPECTED_FEATURE_COUNT))
    values = frame.values.astype(np.float64)
    if bool(np.isnan(values).any()) or bool(np.isinf(values).any()):
        raise RandomForestError("%s: feature matrix is not finite" % label)
    return values


def load_target(path, expected_rows, label):
    frame = pd.read_csv(path)
    if list(frame.columns) != [TARGET]:
        raise RandomForestError(
            "%s: expected a single column %r, found %s"
            % (label, TARGET, list(frame.columns)))
    if len(frame) != expected_rows:
        raise RandomForestError("%s: %d rows, expected %d"
                                % (label, len(frame), expected_rows))
    values = frame[TARGET].values.astype(np.float64)
    if bool(np.isnan(values).any()) or bool(np.isinf(values).any()):
        raise RandomForestError("%s: target is not finite" % label)
    return values


# ----------------------------------------------------------------------
# Fit, evaluate, diagnose
# ----------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    return (mae, float(np.sqrt(mse)), float(r2_score(y_true, y_pred)))


def forest_structure(model):
    """Structural statistics across every estimator in the forest.

    ``get_n_leaves`` arrived in scikit-learn 0.22, so leaves are counted
    directly: a node is a leaf exactly when it has no left child.
    """
    depths, leaves, nodes = [], [], []
    for estimator in model.estimators_:
        tree = estimator.tree_
        depths.append(int(tree.max_depth))
        leaves.append(int((tree.children_left == -1).sum()))
        nodes.append(int(tree.node_count))
    depths = np.asarray(depths, dtype=np.float64)
    leaves = np.asarray(leaves, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    return OrderedDict([
        ("n_estimators_fitted", int(len(model.estimators_))),
        ("min_observed_tree_depth", int(depths.min())),
        ("mean_observed_tree_depth", float(depths.mean())),
        ("median_observed_tree_depth", float(np.median(depths))),
        ("max_observed_tree_depth", int(depths.max())),
        ("min_leaf_count", int(leaves.min())),
        ("mean_leaf_count", float(leaves.mean())),
        ("median_leaf_count", float(np.median(leaves))),
        ("max_leaf_count", int(leaves.max())),
        ("min_node_count", int(nodes.min())),
        ("mean_node_count", float(nodes.mean())),
        ("median_node_count", float(np.median(nodes))),
        ("max_node_count", int(nodes.max())),
    ])


def verify_forest_constraints(model, config):
    """Every tree must obey the selected configured constraints."""
    problems = []
    if len(model.estimators_) != config["n_estimators"]:
        problems.append("forest has %d estimators, configured %d"
                        % (len(model.estimators_), config["n_estimators"]))
    depth_limit = config["max_depth"]
    leaf_limit = config["min_samples_leaf"]
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        if depth_limit is not None and int(tree.max_depth) > depth_limit:
            problems.append("tree %d depth %d exceeds max_depth %d"
                            % (index, tree.max_depth, depth_limit))
        leaf_mask = tree.children_left == -1
        smallest_leaf = int(tree.n_node_samples[leaf_mask].min())
        if smallest_leaf < leaf_limit:
            problems.append(
                "tree %d has a leaf with %d samples, below "
                "min_samples_leaf %d" % (index, smallest_leaf, leaf_limit))
        if problems:
            break
    if problems:
        raise RandomForestError("forest constraint violation: %s"
                                % "; ".join(problems))
    return OrderedDict([
        ("n_estimators_verified", int(len(model.estimators_))),
        ("all_trees_respect_max_depth", True),
        ("all_trees_respect_min_samples_leaf", True),
    ])


def fit_candidate(config, X_train, y_train, X_validation, y_validation):
    """Fit one declared candidate and evaluate it. Nothing is selected here."""
    model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        criterion=config["criterion"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        bootstrap=config["bootstrap"],
        oob_score=FIXED_OOB_SCORE,
        random_state=config["random_state"],
        n_jobs=config["n_jobs"],
    )
    model.fit(X_train, y_train)

    train_mae, train_rmse, train_r2 = compute_metrics(
        y_train, model.predict(X_train))
    validation_pred = model.predict(X_validation)
    val_mae, val_rmse, val_r2 = compute_metrics(y_validation,
                                                validation_pred)

    row = OrderedDict(config)
    row["max_features"] = _max_features_label(config["max_features"])
    row["train_mae"] = train_mae
    row["train_rmse"] = train_rmse
    row["train_r2"] = train_r2
    row["validation_mae"] = val_mae
    row["validation_rmse"] = val_rmse
    row["validation_r2"] = val_r2
    row.update(forest_structure(model))
    return row, model, validation_pred


def selection_key(row):
    """The frozen tie-break ordering, as a sort key."""
    depth = row["max_depth"]
    depth_rank = DEPTH_SENTINEL if depth is None else float(depth)
    # max_features=0.5 is preferred over "auto": rank 0 before rank 1.
    features_rank = 1 if str(row["max_features"]) == "auto" else 0
    return (
        row["validation_rmse"],            # 2. lower RMSE
        depth_rank,                        # 3. smaller finite max_depth
        -float(row["min_samples_leaf"]),   # 4. larger min_samples_leaf
        features_rank,                     # 5. 0.5 before "auto"
        row["candidate_id"],               # 6. grid order
    )


def select_candidate(rows):
    """Apply the frozen selection rule. Returns (winning row, n tied)."""
    if len(rows) != EXPECTED_CANDIDATE_COUNT:
        raise RandomForestError(
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

PRIOR_MANIFESTS = OrderedDict([
    ("m0", "m0_baseline_manifest.json"),
    ("m1", "m1_linear_regression_manifest.json"),
    ("m2", "m2_decision_tree_manifest.json"),
])


def load_prior_metrics(project_root):
    """Read M0, M1 and M2 results from their manifests, never hard-coded."""
    prior = OrderedDict()
    for key, filename in PRIOR_MANIFESTS.items():
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
    """Compare selected M3 against M0, M1 and M2 on 2013 validation.

    One sign convention throughout, stated in COMPARISON_SIGN_CONVENTION
    and never varied between models.
    """
    mae = selected["validation_mae"]
    rmse = selected["validation_rmse"]
    r2 = selected["validation_r2"]

    comparison = OrderedDict()
    comparison["evaluation_period_label"] = EVALUATION_PERIOD_LABEL
    comparison["primary_comparison_metric"] = "MAE"
    comparison["sign_convention"] = COMPARISON_SIGN_CONVENTION
    comparison["m3_mae"] = mae
    comparison["m3_rmse"] = rmse
    comparison["m3_r2"] = r2
    for key in PRIOR_MANIFESTS:
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
    """Descriptive residual summary. Nothing here modifies M3.

    A random forest averages tree leaf means of non-negative training
    targets, so predictions should stay inside the training target range.
    That is verified, never enforced: nothing is clipped.
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
        raise RandomForestError(
            "%d predictions, expected %d"
            % (predictions.size, EXPECTED_VALIDATION_ROWS))
    if predictions.size != y_true.size:
        raise RandomForestError("prediction and actual counts differ")
    if not bool(np.isfinite(predictions).all()):
        raise RandomForestError("predictions contain non-finite values")
    if not bool(np.isfinite(y_true).all()):
        raise RandomForestError("actuals contain non-finite values")
    if not bool(np.isfinite(y_true - predictions).all()):
        raise RandomForestError("residuals contain non-finite values")
    distinct = int(np.unique(predictions).size)
    if distinct < 2:
        raise RandomForestError(
            "M3 produced fewer than two distinct predictions")
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
        raise RandomForestError("row manifest unexpectedly carries the target")
    block = manifest[manifest["partition"] == EVALUATION_PARTITION]
    block = block.reset_index(drop=True)
    if len(block) != expected_rows:
        raise RandomForestError(
            "validation manifest has %d rows, expected %d"
            % (len(block), expected_rows))
    if not bool((block["matrix_row_number"].values
                 == np.arange(expected_rows)).all()):
        raise RandomForestError(
            "validation matrix_row_number is not sequential from 0")
    return block


def verify_row_alignment(project_root, metadata, y_validation):
    source = pd.read_csv(
        os.path.join(project_root, "data", "processed",
                     "airsense_validation_2013.csv"))
    if not bool((metadata["No"].values == source["No"].values).all()):
        raise RandomForestError(
            "manifest identifiers do not match the Phase 4 validation rows")
    if not bool((y_validation
                 == source[TARGET].values.astype(np.float64)).all()):
        raise RandomForestError(
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
    ax.set_title("M3 Random Forest - actual vs predicted\n%s\n"
                 "2013 DEVELOPMENT VALIDATION (not final test)" % label)
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
    ax.set_title("M3 Random Forest - residuals vs predicted\n%s\n"
                 "2013 DEVELOPMENT VALIDATION (not final test)" % label)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, path)


def figure_grid(rows, selected, path):
    """Validation MAE across the declared grid, faceted by max_features.

    One panel per max_features setting; within each, one line per
    min_samples_leaf plotted against max_depth. The depth axis is
    categorical because None is not a numeric depth.
    """
    depth_labels = ["8", "12", "None"]
    positions = np.arange(len(SEARCH_MAX_DEPTH))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

    all_mae = [row["validation_mae"] for row in rows]
    margin = 0.05 * (max(all_mae) - min(all_mae))

    for panel, features in enumerate(SEARCH_MAX_FEATURES):
        ax = axes[panel]
        label = _max_features_label(features)
        for index, leaf in enumerate(SEARCH_MIN_SAMPLES_LEAF):
            series = [row["validation_mae"] for row in rows
                      if row["min_samples_leaf"] == leaf
                      and row["max_features"] == label]
            ax.plot(positions, series, marker="o", markersize=5,
                    color=SERIES_COLORS[index % len(SERIES_COLORS)],
                    label="min_samples_leaf = %d" % leaf)
        if str(selected["max_features"]) == label:
            ax.scatter([SEARCH_MAX_DEPTH.index(selected["max_depth"])],
                       [selected["validation_mae"]], s=170,
                       facecolors="none", edgecolors=ACCENT_COLOR,
                       linewidths=2.0, zorder=5,
                       label="selected (candidate %d)"
                             % selected["candidate_id"])
        ax.set_xticks(positions)
        ax.set_xticklabels(depth_labels)
        ax.set_xlabel("max_depth (categorical; None = unrestricted)")
        ax.set_title("max_features = %s" % label)
        ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(min(all_mae) - margin, max(all_mae) + margin)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("2013 validation MAE (ug/m^3)")
    fig.suptitle("M3 Random Forest - validation MAE across the declared "
                 "24-candidate grid (n_estimators=100)\n"
                 "2013 DEVELOPMENT VALIDATION (not final test)", y=1.04,
                 fontsize=12)
    _save(fig, path)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def _config_label(row):
    depth = "None" if row["max_depth"] is None else str(row["max_depth"])
    return ("max_depth=%s, min_samples_leaf=%d, max_features=%s"
            % (depth, row["min_samples_leaf"], row["max_features"]))


def run_m3_random_forest(project_root, verbose=True):
    """Evaluate the 24 declared candidates, select one, and report."""

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
        raise RandomForestError(
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

    grid = build_grid()
    log("Evaluating %d declared candidates, %d trees each "
        "(grid frozen before fitting)"
        % (len(grid), FIXED_N_ESTIMATORS))
    rows = []
    fitted = {}
    predictions_by_id = {}
    for config in grid:
        row, model, validation_pred = fit_candidate(
            config, X_train, y_train, X_validation, y_validation)
        rows.append(row)
        fitted[config["candidate_id"]] = model
        predictions_by_id[config["candidate_id"]] = validation_pred
        log("  candidate %2d  %-58s  val MAE %10.6f  mean depth %5.1f  "
            "mean leaves %8.1f"
            % (row["candidate_id"], _config_label(row),
               row["validation_mae"], row["mean_observed_tree_depth"],
               row["mean_leaf_count"]))

    # -- grid invariants ---------------------------------------------------
    combinations = set()
    for row in rows:
        key = (row["max_depth"], row["min_samples_leaf"],
               str(row["max_features"]))
        if key in combinations:
            raise RandomForestError("duplicate hyperparameter combination")
        combinations.add(key)
        if row["max_depth"] not in SEARCH_MAX_DEPTH:
            raise RandomForestError("undeclared max_depth %r"
                                    % (row["max_depth"],))
        if row["min_samples_leaf"] not in SEARCH_MIN_SAMPLES_LEAF:
            raise RandomForestError("undeclared min_samples_leaf %r"
                                    % (row["min_samples_leaf"],))
        if str(row["max_features"]) not in [_max_features_label(v)
                                            for v in SEARCH_MAX_FEATURES]:
            raise RandomForestError("undeclared max_features %r"
                                    % (row["max_features"],))
        for name, expected in (("n_estimators", FIXED_N_ESTIMATORS),
                               ("criterion", FIXED_CRITERION),
                               ("bootstrap", FIXED_BOOTSTRAP),
                               ("min_samples_split", FIXED_MIN_SAMPLES_SPLIT),
                               ("random_state", FIXED_RANDOM_STATE),
                               ("n_jobs", FIXED_N_JOBS)):
            if row[name] != expected:
                raise RandomForestError(
                    "candidate %d has %s=%r, expected %r"
                    % (row["candidate_id"], name, row[name], expected))
        for name in ("train_mae", "train_rmse", "train_r2",
                     "validation_mae", "validation_rmse", "validation_r2"):
            if not np.isfinite(row[name]):
                raise RandomForestError(
                    "candidate %d has non-finite %s"
                    % (row["candidate_id"], name))
    declared = set()
    for depth in SEARCH_MAX_DEPTH:
        for leaf in SEARCH_MIN_SAMPLES_LEAF:
            for features in SEARCH_MAX_FEATURES:
                declared.add((depth, leaf, _max_features_label(features)))
    if combinations != declared:
        raise RandomForestError(
            "evaluated combinations do not match the declared grid")

    # -- selection by the frozen rule --------------------------------------
    selected, n_tied = select_candidate(rows)
    selected_id = selected["candidate_id"]
    log("")
    log("Selected candidate %d: %s  (validation MAE %.6f; %d tied on MAE)"
        % (selected_id, _config_label(selected),
           selected["validation_mae"], n_tied))

    model = fitted[selected_id]
    predictions = predictions_by_id[selected_id]
    selected_config = grid[selected_id - 1]

    constraints = verify_forest_constraints(model, selected_config)
    structure = forest_structure(model)
    prior = load_prior_metrics(project_root)
    comparison = build_comparison(selected, prior)
    residuals_summary = residual_diagnostics(y_validation, predictions,
                                             y_train)
    invariants = verify_prediction_invariants(predictions, y_validation)
    metadata = load_validation_metadata(project_root,
                                        EXPECTED_VALIDATION_ROWS)
    alignment = verify_row_alignment(project_root, metadata, y_validation)

    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if importances.size != EXPECTED_FEATURE_COUNT:
        raise RandomForestError("importance vector has %d entries, "
                                "expected %d" % (importances.size,
                                                 EXPECTED_FEATURE_COUNT))
    if abs(float(importances.sum()) - 1.0) > 1e-9:
        raise RandomForestError("importances sum to %r, expected ~1.0"
                                % float(importances.sum()))
    log("Forest verified: %d trees, mean depth %.1f, mean leaves %.1f"
        % (structure["n_estimators_fitted"],
           structure["mean_observed_tree_depth"],
           structure["mean_leaf_count"]))

    # -- artifacts ----------------------------------------------------------
    grid_frame = pd.DataFrame([
        OrderedDict(list(row.items())
                    + [("selected", bool(row["candidate_id"] == selected_id))])
        for row in rows])
    grid_path = os.path.join(results_dir, GRID_FILENAME)
    grid_frame.to_csv(grid_path, index=False)
    grid_sha = sha256_of_file(grid_path)
    if int(grid_frame["selected"].sum()) != 1:
        raise RandomForestError("exactly one candidate must be selected")

    importance_frame = pd.DataFrame(OrderedDict([
        ("feature", feature_names),
        ("importance", importances),
        ("rank", pd.Series(importances).rank(
            ascending=False, method="min").astype(int).values),
    ]))
    importances_path = os.path.join(results_dir, IMPORTANCES_FILENAME)
    importance_frame.to_csv(importances_path, index=False)
    importances_sha = sha256_of_file(importances_path)

    metrics_row = OrderedDict([
        ("model", MODEL_ID),
        ("candidate_id", selected_id),
        ("n_estimators", selected["n_estimators"]),
        ("criterion", selected["criterion"]),
        ("bootstrap", selected["bootstrap"]),
        ("max_depth", "None" if selected["max_depth"] is None
         else selected["max_depth"]),
        ("min_samples_split", selected["min_samples_split"]),
        ("min_samples_leaf", selected["min_samples_leaf"]),
        ("max_features", selected["max_features"]),
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
    ])
    for key in PRIOR_MANIFESTS:
        block = comparison[key]
        metrics_row["%s_mae" % key] = block["mae"]
        metrics_row["mae_improvement_vs_%s_absolute" % key] = \
            block["mae_improvement_absolute"]
        metrics_row["mae_improvement_vs_%s_percent" % key] = \
            block["mae_improvement_percent"]
        metrics_row["%s_rmse" % key] = block["rmse"]
        metrics_row["rmse_delta_vs_%s" % key] = \
            block["rmse_improvement_absolute"]
        metrics_row["rmse_improvement_vs_%s_percent" % key] = \
            block["rmse_improvement_percent"]
        metrics_row["%s_r2" % key] = block["r2"]
        metrics_row["r2_delta_vs_%s" % key] = block["r2_delta"]
    metrics_frame = pd.DataFrame([metrics_row])
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

    # -- manifest -----------------------------------------------------------
    manifest = OrderedDict()
    manifest["model_id"] = MODEL_ID
    manifest["estimator"] = ESTIMATOR
    manifest["sklearn_version"] = __import__("sklearn").__version__
    manifest["phase"] = "V1 / Protocol Phase 9 - random forest (M3)"
    manifest["feature_count"] = int(len(feature_names))
    manifest["feature_names"] = feature_names
    manifest["feature_representation_note"] = (
        "Exactly the frozen Phase 5 43-feature matrix, identical to M1 and "
        "M2. No model-specific feature engineering was permitted.")

    search_space = OrderedDict()
    search_space["searched"] = OrderedDict([
        ("max_depth", ["None" if d is None else d
                       for d in SEARCH_MAX_DEPTH]),
        ("min_samples_leaf", list(SEARCH_MIN_SAMPLES_LEAF)),
        ("max_features", [_max_features_label(v)
                          for v in SEARCH_MAX_FEATURES]),
    ])
    search_space["fixed"] = OrderedDict([
        ("n_estimators", FIXED_N_ESTIMATORS),
        ("criterion", FIXED_CRITERION),
        ("bootstrap", FIXED_BOOTSTRAP),
        ("min_samples_split", FIXED_MIN_SAMPLES_SPLIT),
        ("random_state", FIXED_RANDOM_STATE),
        ("n_jobs", FIXED_N_JOBS),
        ("oob_score", FIXED_OOB_SCORE),
    ])
    search_space["not_searched"] = list(NOT_SEARCHED)
    search_space["candidate_count"] = EXPECTED_CANDIDATE_COUNT
    search_space["candidate_ordering"] = (
        "max_depth outer, min_samples_leaf middle, max_features inner; all "
        "in declared order, candidate_id 1..24")
    search_space["frozen_before_any_result"] = True
    search_space["method"] = (
        "explicit deterministic loop; GridSearchCV, RandomizedSearchCV, "
        "KFold, ShuffleSplit and cross_val_score were deliberately not "
        "used because the design contains an explicit chronological "
        "validation year")
    search_space["n_estimators_rationale"] = (
        "100 trees is a conventional, computationally manageable ensemble "
        "size entirely available in 2019, giving substantially more "
        "averaging than sklearn 0.20's small default. Fixed before results "
        "so selection concerns tree structure and feature subsampling, not "
        "ensemble size. 50, 200, 500 and 1000 were not tried.")
    search_space["seed_policy"] = (
        "random_state=42 and n_jobs=1 for every candidate. One seed only; "
        "no best-of-seed reporting.")
    search_space["bootstrap_policy"] = (
        "bootstrap=True for every candidate; bootstrap=False was not "
        "searched. This is the classical random forest formulation.")
    search_space["oob_policy"] = (
        "oob_score=False. Out-of-bag evaluation was not used for selection: "
        "it would introduce a second selection framework that ignores "
        "temporal ordering, since OOB samples are drawn across the whole "
        "training period. 2013 validation MAE is the sole primary "
        "criterion.")
    search_space["criterion_api_note"] = (
        "criterion='mse' is the scikit-learn 0.20 name; 'squared_error' "
        "did not exist until 1.0 and would be a post-cutoff API")
    manifest["search_space"] = search_space

    manifest["selection_metric"] = SELECTION_METRIC
    manifest["selection_metric_note"] = (
        "2013 validation MAE, the decision metric frozen in Phase 0 long "
        "before M3 existed")
    manifest["selection_tolerance"] = SELECTION_TOLERANCE
    manifest["tie_break_policy"] = list(TIE_BREAK_POLICY)
    manifest["tie_break_policy_frozen_before_fitting"] = True
    manifest["n_candidates_tied_on_mae"] = int(n_tied)
    manifest["no_test_information_used_in_selection"] = True

    manifest["selected_configuration"] = OrderedDict([
        ("candidate_id", selected_id),
        ("n_estimators", selected["n_estimators"]),
        ("criterion", selected["criterion"]),
        ("bootstrap", selected["bootstrap"]),
        ("max_depth", "None" if selected["max_depth"] is None
         else selected["max_depth"]),
        ("min_samples_split", selected["min_samples_split"]),
        ("min_samples_leaf", selected["min_samples_leaf"]),
        ("max_features", selected["max_features"]),
        ("random_state", selected["random_state"]),
        ("n_jobs", selected["n_jobs"]),
        ("oob_score", FIXED_OOB_SCORE),
    ])
    manifest["selected_forest_structure"] = structure
    manifest["selected_forest_constraint_verification"] = constraints

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
    manifest["feature_importance_sum"] = float(importances.sum())
    manifest["feature_importance_caution"] = (
        "Random-forest impurity-based importance is predictive, not "
        "causal. It favours continuous and high-cardinality variables, "
        "which offer far more candidate split points than a binary dummy; "
        "it divides importance unpredictably across correlated predictors "
        "- Phase 3 found |r| of 0.78 to 0.83 among DEWP, TEMP and PRES; "
        "and it describes this fitted ensemble rather than an "
        "environmental mechanism. It was not used to perform feature "
        "selection and did not alter the Phase 5 schema.")

    outputs = OrderedDict()
    for relative, digest, count in (
            (os.path.join(RESULTS_SUBDIR, GRID_FILENAME), grid_sha,
             len(grid_frame)),
            (os.path.join(RESULTS_SUBDIR, METRICS_FILENAME), metrics_sha,
             len(metrics_frame)),
            (os.path.join(RESULTS_SUBDIR, PREDICTIONS_FILENAME),
             predictions_sha, len(prediction_frame)),
            (os.path.join(RESULTS_SUBDIR, IMPORTANCES_FILENAME),
             importances_sha, len(importance_frame))):
        outputs[relative] = OrderedDict([("sha256", digest),
                                         ("rows", int(count))])
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
                   "development-only forest would not be the final model."),
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
        ("policy", "At Protocol Phase 10 the selected M3 hyperparameters "
                   "remain frozen, including random_state=42: combine "
                   "2010-2013 development data, refit "
                   "RandomForestRegressor with exactly those parameters, "
                   "generate 2014 predictions once, open the 2014 target "
                   "once, calculate MAE/RMSE/R2 once, and do not retune "
                   "after seeing 2014."),
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
    log("Upstream artifacts unchanged after M3")
    return manifest
