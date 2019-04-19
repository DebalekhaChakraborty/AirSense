"""Stage A of the AirSense V1 final held-out evaluation.

Protocol Phase 10, Stage A: **blind** final refit and prediction generation.

The 2014 target is sealed throughout this module. Stage A refits all four
frozen models on the full permitted 2010-2013 development population,
generates 2014 predictions from **predictors only**, and freezes them.
Only after that freeze may Stage B open the target.

**This module must never load ``data/processed/airsense_test_2014.csv``.**
That file carries the target. Stage A reads only
``data/processed/features/X_test.csv``, which is predictors-only by
construction (Phase 5 wrote it with an explicit ``usecols`` list omitting
``pm2.5``), plus the test rows of the feature row manifest for identifier
and timestamp metadata. A guard in ``_forbidden_path_check`` aborts the run
if the Phase-4 test partition is ever reachable through this code path.

Nothing here is developed, searched or tuned. Every specification was
frozen in ``artifacts/pretest_development_freeze.json`` before this phase
began, and each is re-read from that freeze rather than restated, so a
divergence between the frozen record and this code stops the run.

Determinism: no random operation beyond the frozen ``random_state=42``,
``n_jobs=1`` for the forest, and no execution timestamp in any artifact.

Historical constraint: CPython 3.6.7 with numpy 1.15.4, pandas 0.23.4,
scikit-learn 0.20.0.
"""

import io
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.data.audit import sha256_of_file

# ----------------------------------------------------------------------
# Frozen expectations
# ----------------------------------------------------------------------

EXPECTED_PRETEST_FREEZE_SHA256 = (
    "98568e88cccb6a64a6e52dc03a76b0e3b3e5b7d750ece3d81e3790fb0beb0bf9")
EXPECTED_RAW_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")

EXPECTED_TRAIN_ROWS = 24418
EXPECTED_VALIDATION_ROWS = 8678
EXPECTED_DEVELOPMENT_ROWS = 33096          # 24418 + 8678
EXPECTED_TEST_ROWS = 8661
EXPECTED_FEATURE_COUNT = 43

TARGET = "pm2.5"

FINAL_FIT_PERIOD = "2010_2013"
FINAL_FIT_PERIOD_LONG = "2010-01-01 00:00:00 to 2013-12-31 23:00:00"
TEST_PERIOD = "2014"
TEST_PERIOD_LONG = "2014-01-01 00:00:00 to 2014-12-31 23:00:00"

# The Phase-4 test partition carries the target and is FORBIDDEN in Stage A.
FORBIDDEN_STAGE_A_FILE = os.path.join("data", "processed",
                                      "airsense_test_2014.csv")

RESULTS_SUBDIR = os.path.join("results", "final_test")
REFIT_SUBDIR = os.path.join(RESULTS_SUBDIR, "refit")
BLIND_PREDICTIONS_FILENAME = "blind_final_predictions.csv"
REFIT_MANIFEST_FILENAME = "final_refit_manifest.json"
BLIND_FREEZE_FILENAME = "final_blind_prediction_freeze.json"

M1_COEFFICIENTS_FILENAME = "m1_final_coefficients.csv"
M2_IMPORTANCES_FILENAME = "m2_final_feature_importances.csv"
M3_IMPORTANCES_FILENAME = "m3_final_feature_importances.csv"
REFIT_DIAGNOSTICS_FILENAME = "final_refit_diagnostics.csv"


class FinalRefitError(RuntimeError):
    """Raised when a declared Stage-A gate fails."""


def _forbidden_path_check(path):
    """Abort if a code path ever reaches the Phase-4 test partition."""
    normalised = os.path.normpath(path).replace(os.sep, "/")
    if normalised.endswith("airsense_test_2014.csv"):
        raise FinalRefitError(
            "QUARANTINE BREACH: Stage A attempted to read %s, which carries "
            "the 2014 target" % path)
    return path


def _read_csv(path, **kwargs):
    return pd.read_csv(_forbidden_path_check(path), **kwargs)


# ----------------------------------------------------------------------
# Gate: pre-test freeze
# ----------------------------------------------------------------------

def verify_pretest_freeze(project_root):
    """Verify the pre-test freeze and every hash it references."""
    freeze_path = os.path.join(project_root, "artifacts",
                               "pretest_development_freeze.json")
    if not os.path.isfile(freeze_path):
        raise FinalRefitError("pretest_development_freeze.json not found")
    freeze_sha = sha256_of_file(freeze_path)
    if freeze_sha != EXPECTED_PRETEST_FREEZE_SHA256:
        raise FinalRefitError(
            "pre-test freeze SHA-256 mismatch: expected %s, found %s. The "
            "pre-test state has changed; the planned final evaluation is "
            "invalid until human review."
            % (EXPECTED_PRETEST_FREEZE_SHA256, freeze_sha))

    freeze = json.load(io.open(freeze_path, encoding="utf-8"))
    if freeze.get("final_test_status") != "sealed":
        raise FinalRefitError(
            "freeze reports final_test_status=%r, expected 'sealed'"
            % freeze.get("final_test_status"))

    problems = []
    verified = 0
    for relative, recorded in freeze["provenance_sha256"].items():
        path = os.path.join(project_root, relative)
        verified += 1
        if sha256_of_file(path) != recorded:
            problems.append(relative)
    for relative, recorded in freeze["model_manifest_sha256"].items():
        path = os.path.join(project_root, relative)
        verified += 1
        if sha256_of_file(path) != recorded:
            problems.append(relative)
    if problems:
        raise FinalRefitError(
            "pre-test artifacts changed: %s. STOP - do not open 2014."
            % ", ".join(problems))

    raw_sha = freeze["provenance_sha256"][
        "data/raw/PRSA_data_2010.1.1-2014.12.31.csv"]
    if raw_sha != EXPECTED_RAW_SHA256:
        raise FinalRefitError("raw dataset SHA-256 mismatch")

    return freeze, freeze_sha, verified


# ----------------------------------------------------------------------
# Load the frozen matrices
# ----------------------------------------------------------------------

def load_frozen_inputs(project_root, freeze):
    """Load the Phase-5 matrices. No feature engineering is re-run.

    The feature representation was frozen before the final test, so the
    development matrix is a straight concatenation of the existing
    ``X_train`` and ``X_validation`` files. No ``get_dummies``, no new
    reference category, no 2013-derived level, no scaling.
    """
    features_dir = os.path.join(project_root, "data", "processed", "features")
    schema = json.load(io.open(
        os.path.join(project_root, "artifacts", "feature_schema.json"),
        encoding="utf-8"))
    feature_names = list(schema["feature_names"])
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise FinalRefitError("feature schema declares %d features, "
                              "expected %d" % (len(feature_names),
                                               EXPECTED_FEATURE_COUNT))

    def matrix(filename, expected_rows, label):
        frame = _read_csv(os.path.join(features_dir, filename))
        if list(frame.columns) != feature_names:
            raise FinalRefitError("%s: column order differs from the frozen "
                                  "schema" % label)
        if frame.shape != (expected_rows, EXPECTED_FEATURE_COUNT):
            raise FinalRefitError(
                "%s: shape %s, expected (%d, %d)"
                % (label, frame.shape, expected_rows, EXPECTED_FEATURE_COUNT))
        values = frame.values.astype(np.float64)
        if not bool(np.isfinite(values).all()):
            raise FinalRefitError("%s: matrix is not finite" % label)
        return values

    def target(filename, expected_rows, label):
        frame = _read_csv(os.path.join(features_dir, filename))
        if list(frame.columns) != [TARGET]:
            raise FinalRefitError("%s: unexpected columns %s"
                                  % (label, list(frame.columns)))
        if len(frame) != expected_rows:
            raise FinalRefitError("%s: %d rows, expected %d"
                                  % (label, len(frame), expected_rows))
        values = frame[TARGET].values.astype(np.float64)
        if not bool(np.isfinite(values).all()):
            raise FinalRefitError("%s: target is not finite" % label)
        return values

    X_train = matrix("X_train.csv", EXPECTED_TRAIN_ROWS, "X_train")
    X_validation = matrix("X_validation.csv", EXPECTED_VALIDATION_ROWS,
                          "X_validation")
    X_test = matrix("X_test.csv", EXPECTED_TEST_ROWS, "X_test")
    y_train = target("y_train.csv", EXPECTED_TRAIN_ROWS, "y_train")
    y_validation = target("y_validation.csv", EXPECTED_VALIDATION_ROWS,
                          "y_validation")

    # Chronological concatenation: 2010-2012 then 2013. No shuffling, no
    # further split. 2013 is development data now, not validation.
    X_development = np.vstack([X_train, X_validation])
    y_development = np.concatenate([y_train, y_validation])
    if X_development.shape != (EXPECTED_DEVELOPMENT_ROWS,
                               EXPECTED_FEATURE_COUNT):
        raise FinalRefitError(
            "development matrix shape %s, expected (%d, %d)"
            % (X_development.shape, EXPECTED_DEVELOPMENT_ROWS,
               EXPECTED_FEATURE_COUNT))
    if y_development.size != EXPECTED_DEVELOPMENT_ROWS:
        raise FinalRefitError("development target has %d rows, expected %d"
                              % (y_development.size,
                                 EXPECTED_DEVELOPMENT_ROWS))

    return OrderedDict([
        ("feature_names", feature_names),
        ("X_development", X_development),
        ("y_development", y_development),
        ("X_test", X_test),
    ])


def load_test_metadata(project_root):
    """Test-row identifiers and timestamps from the Phase-5 row manifest.

    The manifest carries no target, so this reveals nothing about 2014
    PM2.5.
    """
    path = os.path.join(project_root, "artifacts",
                        "feature_row_manifest.csv")
    manifest = _read_csv(path)
    if TARGET in manifest.columns:
        raise FinalRefitError("row manifest unexpectedly carries the target")
    block = manifest[manifest["partition"] == "test"].reset_index(drop=True)
    if len(block) != EXPECTED_TEST_ROWS:
        raise FinalRefitError("test manifest has %d rows, expected %d"
                              % (len(block), EXPECTED_TEST_ROWS))
    if not bool((block["matrix_row_number"].values
                 == np.arange(EXPECTED_TEST_ROWS)).all()):
        raise FinalRefitError("test matrix_row_number is not sequential")
    return block


# ----------------------------------------------------------------------
# The four frozen final refits
# ----------------------------------------------------------------------

def frozen_specifications(freeze):
    """Read each model's frozen specification out of the pre-test freeze.

    Specifications are taken from the freeze rather than restated here, so
    that a divergence between the frozen record and this code cannot pass
    silently.
    """
    models = freeze["models"]
    spec = OrderedDict()
    spec["M0"] = OrderedDict([("strategy", models["M0"]["strategy"])])
    spec["M1"] = OrderedDict([
        ("estimator", models["M1"]["estimator"]),
        ("fit_intercept", models["M1"]["specification"]["fit_intercept"]),
    ])
    for model_id in ("M2", "M3"):
        selected = models[model_id]["selected_hyperparameters"]
        spec[model_id] = OrderedDict(selected)
        spec[model_id]["estimator"] = models[model_id]["estimator"]
    return spec


def _expect(spec, key, value, model_id):
    if spec.get(key) != value:
        raise FinalRefitError(
            "%s frozen specification mismatch: %s is %r in the freeze, but "
            "this refit would use %r" % (model_id, key, spec.get(key), value))


def fit_final_m0(y_development):
    """M0: training-median constant, recomputed on 2010-2013.

    The Phase-6 constant (73.0, fitted on 2010-2012) is deliberately NOT
    reused. The pre-registered refit rule recomputes the median over the
    full permitted development target.
    """
    return float(np.median(y_development))


def fit_final_m1(X_development, y_development, spec):
    _expect(spec, "fit_intercept", True, "M1")
    model = LinearRegression(fit_intercept=True)
    model.fit(X_development, y_development)
    return model


def fit_final_m2(X_development, y_development, spec):
    _expect(spec, "criterion", "mse", "M2")
    _expect(spec, "splitter", "best", "M2")
    _expect(spec, "max_depth", 12, "M2")
    _expect(spec, "min_samples_split", 2, "M2")
    _expect(spec, "min_samples_leaf", 10, "M2")
    _expect(spec, "max_features", "None", "M2")
    _expect(spec, "random_state", 42, "M2")
    model = DecisionTreeRegressor(
        criterion="mse", splitter="best", max_depth=12,
        min_samples_split=2, min_samples_leaf=10, max_features=None,
        random_state=42)
    model.fit(X_development, y_development)
    return model


def fit_final_m3(X_development, y_development, spec):
    _expect(spec, "n_estimators", 100, "M3")
    _expect(spec, "criterion", "mse", "M3")
    _expect(spec, "bootstrap", True, "M3")
    _expect(spec, "max_depth", 12, "M3")
    _expect(spec, "min_samples_split", 2, "M3")
    _expect(spec, "min_samples_leaf", 1, "M3")
    _expect(spec, "max_features", "auto", "M3")
    _expect(spec, "random_state", 42, "M3")
    _expect(spec, "n_jobs", 1, "M3")
    model = RandomForestRegressor(
        n_estimators=100, criterion="mse", max_depth=12,
        min_samples_split=2, min_samples_leaf=1, max_features="auto",
        bootstrap=True, oob_score=False, random_state=42, n_jobs=1)
    model.fit(X_development, y_development)
    return model


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def linear_design_diagnostics(X_development):
    """Descriptive numerical properties of the final design matrix."""
    n_rows = X_development.shape[0]
    with_intercept = np.hstack(
        [np.ones((n_rows, 1), dtype=np.float64), X_development])
    singular = np.linalg.svd(with_intercept, compute_uv=False)
    largest = float(singular[0])
    tolerance = float(max(with_intercept.shape)
                      * np.finfo(np.float64).eps * largest)
    nonzero = singular[singular > tolerance]
    smallest_nonzero = float(nonzero[-1])
    return OrderedDict([
        ("matrix", "design_with_intercept_[1|X]_44_columns"),
        ("n_rows", int(n_rows)),
        ("n_columns", int(with_intercept.shape[1])),
        ("matrix_rank", int(nonzero.size)),
        ("rank_deficient", bool(nonzero.size < with_intercept.shape[1])),
        ("largest_singular_value", largest),
        ("smallest_singular_value", float(singular[-1])),
        ("smallest_nonzero_singular_value", smallest_nonzero),
        ("condition_number", largest / smallest_nonzero),
        ("condition_number_definition",
         "largest / smallest non-zero singular value; non-zero means "
         "> max(m, n) * eps * largest"),
        ("status", "descriptive only - M1 is not altered"),
    ])


def tree_structure(model):
    tree = model.tree_
    return OrderedDict([
        ("observed_tree_depth", int(tree.max_depth)),
        ("node_count", int(tree.node_count)),
        ("leaf_count", int((tree.children_left == -1).sum())),
    ])


def forest_structure(model):
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


def prediction_summary(predictions, y_development, label):
    """Blind prediction properties. No target comparison is possible here."""
    development_min = float(y_development.min())
    development_max = float(y_development.max())
    return OrderedDict([
        ("model", label),
        ("n_predictions", int(predictions.size)),
        ("n_distinct_predictions", int(np.unique(predictions).size)),
        ("prediction_min", float(predictions.min())),
        ("prediction_max", float(predictions.max())),
        ("prediction_mean", float(predictions.mean())),
        ("prediction_std", float(predictions.std(ddof=1))),
        ("n_negative_predictions", int((predictions < 0.0).sum())),
        ("all_finite", bool(np.isfinite(predictions).all())),
        ("within_development_target_range",
         bool((predictions >= development_min - 1e-9).all()
              and (predictions <= development_max + 1e-9).all())),
        ("development_target_min", development_min),
        ("development_target_max", development_max),
    ])


# ----------------------------------------------------------------------
# Stage-A orchestration
# ----------------------------------------------------------------------

def run_stage_a(project_root, verbose=True):
    """Blind final refit and prediction generation. 2014 target untouched."""

    def log(message):
        if verbose:
            print(message)

    results_dir = os.path.join(project_root, RESULTS_SUBDIR)
    refit_dir = os.path.join(project_root, REFIT_SUBDIR)
    artifacts_dir = os.path.join(project_root, "artifacts")
    for directory in (results_dir, refit_dir, artifacts_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    freeze, freeze_sha, n_verified = verify_pretest_freeze(project_root)
    log("Pre-test freeze verified: %s" % freeze_sha)
    log("  %d referenced hashes checked; final_test_status=%s"
        % (n_verified, freeze["final_test_status"]))

    spec = frozen_specifications(freeze)
    inputs = load_frozen_inputs(project_root, freeze)
    feature_names = inputs["feature_names"]
    X_development = inputs["X_development"]
    y_development = inputs["y_development"]
    X_test = inputs["X_test"]
    metadata = load_test_metadata(project_root)
    log("Final development population: %s (2010-2012 then 2013, "
        "chronological, not shuffled)" % (X_development.shape,))
    log("Test predictors: %s  (predictors only - target still sealed)"
        % (X_test.shape,))

    # -- M0 ---------------------------------------------------------------
    m0_constant = fit_final_m0(y_development)
    m0_predictions = np.full(EXPECTED_TEST_ROWS, m0_constant,
                             dtype=np.float64)
    log("M0 final constant (median of 33,096 development targets): %r"
        % m0_constant)

    # -- M1 ---------------------------------------------------------------
    m1 = fit_final_m1(X_development, y_development, spec["M1"])
    m1_coefficients = np.asarray(m1.coef_, dtype=np.float64).ravel()
    m1_intercept = float(m1.intercept_)
    if m1_coefficients.size != EXPECTED_FEATURE_COUNT:
        raise FinalRefitError("M1 produced %d coefficients, expected %d"
                              % (m1_coefficients.size,
                                 EXPECTED_FEATURE_COUNT))
    if not (bool(np.isfinite(m1_coefficients).all())
            and np.isfinite(m1_intercept)):
        raise FinalRefitError("M1 produced non-finite parameters")
    m1_predictions = m1.predict(X_test)
    m1_diagnostics = linear_design_diagnostics(X_development)
    log("M1 refitted: intercept %.10f, rank %d/%d, condition %.6g"
        % (m1_intercept, m1_diagnostics["matrix_rank"],
           m1_diagnostics["n_columns"], m1_diagnostics["condition_number"]))

    # -- M2 ---------------------------------------------------------------
    m2 = fit_final_m2(X_development, y_development, spec["M2"])
    m2_predictions = m2.predict(X_test)
    m2_structure = tree_structure(m2)
    m2_importances = np.asarray(m2.feature_importances_, dtype=np.float64)
    log("M2 refitted: depth %d, %d nodes, %d leaves"
        % (m2_structure["observed_tree_depth"], m2_structure["node_count"],
           m2_structure["leaf_count"]))

    # -- M3 ---------------------------------------------------------------
    m3 = fit_final_m3(X_development, y_development, spec["M3"])
    m3_predictions = m3.predict(X_test)
    m3_structure = forest_structure(m3)
    m3_importances = np.asarray(m3.feature_importances_, dtype=np.float64)
    log("M3 refitted: %d trees, mean depth %.1f, mean leaves %.1f"
        % (m3_structure["n_estimators_fitted"],
           m3_structure["mean_observed_tree_depth"],
           m3_structure["mean_leaf_count"]))

    for name, predictions in (("M0", m0_predictions), ("M1", m1_predictions),
                              ("M2", m2_predictions), ("M3", m3_predictions)):
        if predictions.size != EXPECTED_TEST_ROWS:
            raise FinalRefitError("%s produced %d predictions, expected %d"
                                  % (name, predictions.size,
                                     EXPECTED_TEST_ROWS))
        if not bool(np.isfinite(predictions).all()):
            raise FinalRefitError("%s produced non-finite predictions" % name)

    # -- blind prediction artifact - NO actual target ---------------------
    blind = pd.DataFrame(OrderedDict([
        ("matrix_row_number", metadata["matrix_row_number"].values),
        ("No", metadata["No"].values),
        ("timestamp", metadata["timestamp"].values),
        ("m0_prediction", m0_predictions),
        ("m1_prediction", m1_predictions),
        ("m2_prediction", m2_predictions),
        ("m3_prediction", m3_predictions),
    ]))
    for forbidden in ("actual_pm25", TARGET, "residual"):
        if forbidden in blind.columns:
            raise FinalRefitError(
                "blind prediction artifact must not contain %r" % forbidden)
    blind_path = os.path.join(results_dir, BLIND_PREDICTIONS_FILENAME)
    blind.to_csv(blind_path, index=False)
    blind_sha = sha256_of_file(blind_path)
    log("Wrote %s (%d rows, no target, no residuals)"
        % (os.path.join(RESULTS_SUBDIR, BLIND_PREDICTIONS_FILENAME),
           len(blind)))

    # -- refit artifacts ---------------------------------------------------
    m1_frame = pd.DataFrame(OrderedDict([
        ("feature", feature_names),
        ("coefficient", m1_coefficients),
    ]))
    m1_path = os.path.join(refit_dir, M1_COEFFICIENTS_FILENAME)
    m1_frame.to_csv(m1_path, index=False)

    m2_frame = pd.DataFrame(OrderedDict([
        ("feature", feature_names),
        ("importance", m2_importances),
    ]))
    m2_path = os.path.join(refit_dir, M2_IMPORTANCES_FILENAME)
    m2_frame.to_csv(m2_path, index=False)

    m3_frame = pd.DataFrame(OrderedDict([
        ("feature", feature_names),
        ("importance", m3_importances),
    ]))
    m3_path = os.path.join(refit_dir, M3_IMPORTANCES_FILENAME)
    m3_frame.to_csv(m3_path, index=False)

    summaries = [prediction_summary(m0_predictions, y_development, "M0"),
                 prediction_summary(m1_predictions, y_development, "M1"),
                 prediction_summary(m2_predictions, y_development, "M2"),
                 prediction_summary(m3_predictions, y_development, "M3")]
    diagnostics_rows = []
    for summary in summaries:
        row = OrderedDict(summary)
        model_id = row["model"]
        row["final_fit_period"] = FINAL_FIT_PERIOD
        row["n_development"] = EXPECTED_DEVELOPMENT_ROWS
        row["n_features"] = (0 if model_id == "M0"
                             else EXPECTED_FEATURE_COUNT)
        if model_id == "M0":
            row["structure"] = "constant = %r" % m0_constant
        elif model_id == "M1":
            row["structure"] = ("intercept=%.10f; rank %d/%d; condition %.6g"
                                % (m1_intercept,
                                   m1_diagnostics["matrix_rank"],
                                   m1_diagnostics["n_columns"],
                                   m1_diagnostics["condition_number"]))
        elif model_id == "M2":
            row["structure"] = ("depth %d; %d nodes; %d leaves"
                                % (m2_structure["observed_tree_depth"],
                                   m2_structure["node_count"],
                                   m2_structure["leaf_count"]))
        else:
            row["structure"] = ("%d trees; mean depth %.2f; mean leaves %.2f"
                                % (m3_structure["n_estimators_fitted"],
                                   m3_structure["mean_observed_tree_depth"],
                                   m3_structure["mean_leaf_count"]))
        diagnostics_rows.append(row)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    diagnostics_path = os.path.join(refit_dir, REFIT_DIAGNOSTICS_FILENAME)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    log("Wrote final-refit artifacts to %s" % REFIT_SUBDIR)

    # -- refit manifest - NO test number may appear ------------------------
    manifest = OrderedDict()
    manifest["project"] = "AirSense"
    manifest["phase"] = ("V1 / Protocol Phase 10 Stage A - blind final "
                         "refit and prediction")
    manifest["stage"] = "A"
    manifest["pretest_development_freeze_sha256"] = freeze_sha
    manifest["raw_dataset_sha256"] = EXPECTED_RAW_SHA256
    manifest["feature_schema_sha256"] = freeze["provenance_sha256"][
        "artifacts/feature_schema.json"]
    manifest["split_manifest_sha256"] = freeze["provenance_sha256"][
        "artifacts/split_manifest.json"]
    manifest["final_fit_period"] = FINAL_FIT_PERIOD_LONG
    manifest["test_period"] = TEST_PERIOD_LONG
    manifest["n_development"] = EXPECTED_DEVELOPMENT_ROWS
    manifest["n_test_predictors"] = EXPECTED_TEST_ROWS
    manifest["feature_count"] = EXPECTED_FEATURE_COUNT
    manifest["feature_names"] = feature_names
    manifest["feature_engineering_rerun"] = False
    manifest["feature_engineering_note"] = (
        "The Phase-5 matrices were concatenated unchanged. No get_dummies "
        "was re-run, no reference category changed, no 2013-derived level "
        "added, no scaling applied.")

    models = OrderedDict()
    models["M0"] = OrderedDict([
        ("strategy", spec["M0"]["strategy"]),
        ("final_development_median", m0_constant),
        ("predictors_used", []),
        ("note", "recomputed on the combined 2010-2013 development target "
                 "as pre-registered; the Phase-6 constant was not reused"),
    ])
    models["M1"] = OrderedDict([
        ("estimator", spec["M1"]["estimator"]),
        ("fit_intercept", True),
        ("scaling", "none"),
        ("regularization", "none"),
        ("intercept", m1_intercept),
        ("coefficient_artifact",
         os.path.join(REFIT_SUBDIR, M1_COEFFICIENTS_FILENAME)),
        ("coefficient_artifact_sha256", sha256_of_file(m1_path)),
        ("design_diagnostics", m1_diagnostics),
    ])
    models["M2"] = OrderedDict([
        ("estimator", spec["M2"]["estimator"]),
        ("frozen_hyperparameters", OrderedDict([
            ("criterion", "mse"), ("splitter", "best"), ("max_depth", 12),
            ("min_samples_split", 2), ("min_samples_leaf", 10),
            ("max_features", "None"), ("random_state", 42)])),
        ("final_tree_structure", m2_structure),
        ("importance_artifact",
         os.path.join(REFIT_SUBDIR, M2_IMPORTANCES_FILENAME)),
        ("importance_artifact_sha256", sha256_of_file(m2_path)),
        ("importance_sum", float(m2_importances.sum())),
    ])
    models["M3"] = OrderedDict([
        ("estimator", spec["M3"]["estimator"]),
        ("frozen_hyperparameters", OrderedDict([
            ("n_estimators", 100), ("criterion", "mse"),
            ("bootstrap", True), ("max_depth", 12),
            ("min_samples_split", 2), ("min_samples_leaf", 1),
            ("max_features", "auto"), ("random_state", 42),
            ("n_jobs", 1), ("oob_score", False)])),
        ("final_forest_structure", m3_structure),
        ("importance_artifact",
         os.path.join(REFIT_SUBDIR, M3_IMPORTANCES_FILENAME)),
        ("importance_artifact_sha256", sha256_of_file(m3_path)),
        ("importance_sum", float(m3_importances.sum())),
    ])
    manifest["models"] = models
    manifest["blind_prediction_summaries"] = summaries
    manifest["outputs"] = OrderedDict([
        (os.path.join(RESULTS_SUBDIR, BLIND_PREDICTIONS_FILENAME),
         OrderedDict([("sha256", blind_sha), ("rows", int(len(blind)))])),
        (os.path.join(REFIT_SUBDIR, M1_COEFFICIENTS_FILENAME),
         OrderedDict([("sha256", sha256_of_file(m1_path)), ("rows", 43)])),
        (os.path.join(REFIT_SUBDIR, M2_IMPORTANCES_FILENAME),
         OrderedDict([("sha256", sha256_of_file(m2_path)), ("rows", 43)])),
        (os.path.join(REFIT_SUBDIR, M3_IMPORTANCES_FILENAME),
         OrderedDict([("sha256", sha256_of_file(m3_path)), ("rows", 43)])),
        (os.path.join(REFIT_SUBDIR, REFIT_DIAGNOSTICS_FILENAME),
         OrderedDict([("sha256", sha256_of_file(diagnostics_path)),
                      ("rows", int(len(diagnostics_frame)))])),
    ])
    manifest["test_target_status"] = "sealed"
    manifest["test_metrics_status"] = "not_computed"
    manifest["test_target_read"] = False
    manifest["phase4_test_partition_read"] = False
    manifest["hyperparameter_search_performed"] = False
    manifest["note"] = (
        "Stage A contains no test metric of any kind. No execution "
        "timestamp is recorded so that repeated runs are byte-identical.")

    manifest_path = os.path.join(artifacts_dir, REFIT_MANIFEST_FILENAME)
    with io.open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2))
        handle.write("\n")
    log("Wrote artifacts/%s" % REFIT_MANIFEST_FILENAME)

    # -- independent Stage-A model checks ---------------------------------
    checks = OrderedDict()
    if int(np.unique(m0_predictions).size) != 1:
        raise FinalRefitError("M0 predictions are not constant")
    if float(np.unique(m0_predictions)[0]) != m0_constant:
        raise FinalRefitError("M0 constant does not match the development "
                              "median")
    checks["m0_all_predictions_equal_development_median"] = True

    reconstructed = m1_intercept + X_test.dot(m1_coefficients)
    max_difference = float(np.abs(reconstructed - m1_predictions).max())
    if max_difference > 1e-10:
        raise FinalRefitError(
            "M1 equation reconstruction differs by %.3e" % max_difference)
    checks["m1_equation_reconstruction_max_abs_diff"] = max_difference

    development_min = float(y_development.min())
    development_max = float(y_development.max())
    for name, predictions in (("m2", m2_predictions), ("m3", m3_predictions)):
        within = bool((predictions >= development_min - 1e-9).all()
                      and (predictions <= development_max + 1e-9).all())
        if not within:
            raise FinalRefitError(
                "%s predictions fall outside the development target range"
                % name.upper())
        checks["%s_within_development_target_range" % name] = True
    checks["rows_all_models"] = EXPECTED_TEST_ROWS
    checks["metadata_alignment_verified"] = True
    checks["target_used"] = False
    log("Independent Stage-A model checks passed")

    verify_pretest_freeze(project_root)
    log("")
    log("Pre-test artifacts unchanged after Stage A; 2014 target still "
        "SEALED")
    return OrderedDict([("manifest", manifest), ("checks", checks),
                        ("blind_sha256", blind_sha),
                        ("m0_constant", m0_constant)])


def write_blind_freeze(project_root, verbose=True):
    """Freeze the blind predictions and final refit. Call after Stage A."""
    artifacts_dir = os.path.join(project_root, "artifacts")
    freeze = OrderedDict()
    freeze["project"] = "AirSense"
    freeze["freeze_type"] = "final_blind_prediction"
    freeze["created_at_protocol_phase"] = (
        "Protocol Phase 10 Stage A - after blind refit, prediction and "
        "determinism verification, before any test-target access")
    freeze["2014_target_status"] = "sealed"
    freeze["blind_prediction_status"] = "frozen"
    freeze["final_models_status"] = "frozen"
    freeze["statement"] = (
        "Once this artifact exists, no model and no prediction may change. "
        "Stage B may only reveal the 2014 target and calculate scores.")

    referenced = OrderedDict()
    for relative in (
            os.path.join("artifacts", "pretest_development_freeze.json"),
            os.path.join("artifacts", REFIT_MANIFEST_FILENAME),
            os.path.join(RESULTS_SUBDIR, BLIND_PREDICTIONS_FILENAME),
            os.path.join(REFIT_SUBDIR, M1_COEFFICIENTS_FILENAME),
            os.path.join(REFIT_SUBDIR, M2_IMPORTANCES_FILENAME),
            os.path.join(REFIT_SUBDIR, M3_IMPORTANCES_FILENAME),
            os.path.join(REFIT_SUBDIR, REFIT_DIAGNOSTICS_FILENAME)):
        path = os.path.join(project_root, relative)
        if not os.path.isfile(path):
            raise FinalRefitError("%s not found" % relative)
        referenced[relative] = sha256_of_file(path)
    freeze["referenced_sha256"] = referenced
    freeze["note"] = (
        "No execution timestamp is recorded so that repeated runs are "
        "byte-identical.")

    path = os.path.join(artifacts_dir, BLIND_FREEZE_FILENAME)
    with io.open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(freeze, indent=2))
        handle.write("\n")
    if verbose:
        print("Wrote artifacts/%s" % BLIND_FREEZE_FILENAME)
    return freeze, sha256_of_file(path)
