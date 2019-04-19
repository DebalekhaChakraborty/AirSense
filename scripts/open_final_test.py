"""Protocol Phase 10, Stage B: open the locked 2014 target and evaluate.

**This is the only source path in Phase 10 permitted to read the 2014
target.** It deliberately contains no estimator import and no ``fit`` call:
it cannot train, retrain or regenerate a prediction. Its sole job is to
verify the frozen state, reveal the target once, score the already-frozen
blind predictions against it, and freeze the result.

Order of operations, each a gate on the next:

1. verify the pre-test development freeze and every hash it references;
2. verify the Stage-A blind-prediction freeze and every hash it references;
3. verify the frozen blind predictions themselves;
4. **only then** open ``data/processed/airsense_test_2014.csv``;
5. align the revealed target to the frozen predictions by identifier,
   timestamp and row order;
6. compute MAE, RMSE and R2 for M0-M3 against one shared snapshot;
7. freeze the evaluation.

**Single-source-open policy.** If
``artifacts/final_test_opening_receipt.json`` already exists, the Phase-4
test partition is NOT reopened: the run switches to verification/report-only
mode using the frozen snapshot. This prevents repeated exploratory
reopening of 2014.

**The predictions are copied, never recomputed.** Stage B reads the blind
artifact's prediction columns and writes them through unchanged, so the
scored numbers are provably the ones frozen before the target was visible.

Once this script has run, the 2014 target is exhausted. It is no longer an
unseen test set, and no model may honestly claim a fresh evaluation on it.

Usage:
    venv/bin/python scripts/open_final_test.py
"""

import io
import json
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.audit import sha256_of_file  # noqa: E402

EXPECTED_PRETEST_FREEZE_SHA256 = (
    "98568e88cccb6a64a6e52dc03a76b0e3b3e5b7d750ece3d81e3790fb0beb0bf9")
EXPECTED_RAW_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")

EXPECTED_TEST_ROWS = 8661
EXPECTED_DEVELOPMENT_ROWS = 33096
MODELS = ["M0", "M1", "M2", "M3"]

FINAL_FIT_PERIOD = "2010_2013"
TEST_PERIOD = "2014"
TARGET = "pm2.5"

RESULTS_DIR = os.path.join("results", "final_test")
FIGURES_DIR = "figures"

BLIND_PREDICTIONS = os.path.join(RESULTS_DIR, "blind_final_predictions.csv")
TARGET_SNAPSHOT = os.path.join(RESULTS_DIR,
                               "final_test_target_snapshot.csv")
FINAL_PREDICTIONS = os.path.join(RESULTS_DIR, "final_test_predictions.csv")
FINAL_METRICS = os.path.join(RESULTS_DIR, "final_test_metrics.csv")
RESIDUAL_SUMMARY = os.path.join(RESULTS_DIR, "final_residual_summary.csv")
DEV_TO_TEST = os.path.join(RESULTS_DIR,
                           "development_to_test_comparison.csv")
OPENING_RECEIPT = os.path.join("artifacts",
                               "final_test_opening_receipt.json")
EVALUATION_MANIFEST = os.path.join("artifacts",
                                   "final_evaluation_manifest.json")

# The authoritative locked source. Read exactly once, under gate.
TEST_SOURCE = os.path.join("data", "processed", "airsense_test_2014.csv")
TEST_SOURCE_COLUMNS = ["No", "year", "month", "day", "hour", "pm2.5"]

DEV_MANIFESTS = OrderedDict([
    ("M0", "m0_baseline_manifest.json"),
    ("M1", "m1_linear_regression_manifest.json"),
    ("M2", "m2_decision_tree_manifest.json"),
    ("M3", "m3_random_forest_manifest.json"),
])

FIGURE_DPI = 100
PLOT_COLOR = "#31688e"
ACCENT_COLOR = "#b03a2e"
MODEL_COLORS = {"M0": "#999999", "M1": "#31688e", "M2": "#35b779",
                "M3": "#b03a2e"}


class FinalTestError(RuntimeError):
    """Raised when a Stage-B gate fails."""


def path(relative):
    return os.path.join(PROJECT_ROOT, relative)


def verify_freezes():
    """Gates 1-3. Nothing proceeds unless every frozen hash matches."""
    report = OrderedDict()

    pretest = path("artifacts/pretest_development_freeze.json")
    pretest_sha = sha256_of_file(pretest)
    if pretest_sha != EXPECTED_PRETEST_FREEZE_SHA256:
        raise FinalTestError(
            "pre-test freeze SHA-256 mismatch: expected %s, found %s"
            % (EXPECTED_PRETEST_FREEZE_SHA256, pretest_sha))
    freeze = json.load(io.open(pretest, encoding="utf-8"))
    checked = 0
    for relative, recorded in list(freeze["provenance_sha256"].items()) \
            + list(freeze["model_manifest_sha256"].items()):
        checked += 1
        if sha256_of_file(path(relative)) != recorded:
            raise FinalTestError("pre-test artifact changed: %s" % relative)
    report["pretest_freeze_sha256"] = pretest_sha
    report["pretest_hashes_verified"] = checked

    blind_freeze_path = path("artifacts/final_blind_prediction_freeze.json")
    if not os.path.isfile(blind_freeze_path):
        raise FinalTestError(
            "final_blind_prediction_freeze.json not found; Stage A must "
            "complete and freeze before the target may be opened")
    blind_freeze_sha = sha256_of_file(blind_freeze_path)
    blind_freeze = json.load(io.open(blind_freeze_path, encoding="utf-8"))
    if blind_freeze.get("2014_target_status") != "sealed":
        raise FinalTestError("blind freeze does not report a sealed target")
    blind_checked = 0
    for relative, recorded in blind_freeze["referenced_sha256"].items():
        blind_checked += 1
        if sha256_of_file(path(relative)) != recorded:
            raise FinalTestError("blind-frozen artifact changed: %s"
                                 % relative)
    report["blind_freeze_sha256"] = blind_freeze_sha
    report["blind_hashes_verified"] = blind_checked

    raw_sha = sha256_of_file(
        path("data/raw/PRSA_data_2010.1.1-2014.12.31.csv"))
    if raw_sha != EXPECTED_RAW_SHA256:
        raise FinalTestError("raw dataset SHA-256 changed")
    report["raw_sha256"] = raw_sha
    return report, freeze, blind_freeze


def load_blind_predictions(as_text=False):
    """Gate 3: the frozen predictions, carrying no target.

    ``as_text=True`` returns the file's verbatim string representation.
    Stage B writes the prediction columns through from that text view, so
    the scored predictions in the final artifact are byte-identical to the
    ones frozen before the target was visible. Re-serialising the parsed
    floats would shift the last digit by ~1 ULP, because pandas 0.23.4
    does not emit shortest-round-trip floats - a cosmetic difference, but
    one that would weaken the identity guarantee for no benefit.
    """
    if as_text:
        return pd.read_csv(path(BLIND_PREDICTIONS), dtype=str,
                           keep_default_na=False, na_filter=False)
    frame = pd.read_csv(path(BLIND_PREDICTIONS))
    expected = ["matrix_row_number", "No", "timestamp", "m0_prediction",
                "m1_prediction", "m2_prediction", "m3_prediction"]
    if list(frame.columns) != expected:
        raise FinalTestError("blind prediction schema changed: %s"
                             % list(frame.columns))
    if len(frame) != EXPECTED_TEST_ROWS:
        raise FinalTestError("blind predictions have %d rows, expected %d"
                             % (len(frame), EXPECTED_TEST_ROWS))
    for column in expected[3:]:
        values = frame[column].values.astype(np.float64)
        if not bool(np.isfinite(values).all()):
            raise FinalTestError("%s contains non-finite values" % column)
    if not bool((frame["matrix_row_number"].values
                 == np.arange(EXPECTED_TEST_ROWS)).all()):
        raise FinalTestError("blind matrix_row_number is not sequential")
    return frame


def open_test_target():
    """Gate 4: the single authorized reading of the 2014 target.

    Only identifier, calendar and target columns are read. Feature values
    are irrelevant to Stage B and are not loaded, so there is no route by
    which this file could influence a model.
    """
    frame = pd.read_csv(path(TEST_SOURCE), usecols=TEST_SOURCE_COLUMNS)
    frame = frame[TEST_SOURCE_COLUMNS]
    if len(frame) != EXPECTED_TEST_ROWS:
        raise FinalTestError("test source has %d rows, expected %d"
                             % (len(frame), EXPECTED_TEST_ROWS))
    values = frame[TARGET].values.astype(np.float64)
    if bool(np.isnan(values).any()):
        raise FinalTestError("test target contains missing values")
    if not bool(np.isfinite(values).all()):
        raise FinalTestError("test target contains non-finite values")
    timestamps = pd.to_datetime(frame[["year", "month", "day", "hour"]])
    return frame, values, timestamps


def compute_metrics(y_true, y_pred):
    """Exactly the three pre-registered metrics.

    scikit-learn 0.20's mean_squared_error has no ``squared=`` keyword, so
    RMSE is an explicit square root.
    """
    return OrderedDict([
        ("mae", float(mean_absolute_error(y_true, y_pred))),
        ("rmse", float(np.sqrt(mean_squared_error(y_true, y_pred)))),
        ("r2", float(r2_score(y_true, y_pred))),
    ])


def residual_summary(model_id, y_true, y_pred):
    residuals = y_true - y_pred
    return OrderedDict([
        ("model", model_id),
        ("test_period", TEST_PERIOD),
        ("residual_definition", "actual_pm25 - predicted_pm25"),
        ("mean_residual", float(residuals.mean())),
        ("median_residual", float(np.median(residuals))),
        ("residual_std", float(residuals.std(ddof=1))),
        ("min_residual", float(residuals.min())),
        ("max_residual", float(residuals.max())),
        ("n_negative_predictions", int((y_pred < 0.0).sum())),
    ])


def _save(fig, target_path):
    fig.tight_layout()
    try:
        fig.savefig(target_path, dpi=FIGURE_DPI,
                    metadata={"Software": "AirSense V1"})
    except TypeError:                          # pragma: no cover
        fig.savefig(target_path, dpi=FIGURE_DPI)
    plt.close(fig)


def figure_mae_comparison(metrics, best, target_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [metrics[m]["mae"] for m in MODELS]
    positions = np.arange(len(MODELS))
    colors = [MODEL_COLORS[m] for m in MODELS]
    bars = ax.bar(positions, values, color=colors, width=0.6)
    for index, (bar, value) in enumerate(zip(bars, values)):
        marker = "  <- best" if MODELS[index] == best else ""
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.6,
                "%.3f%s" % (value, marker), ha="center", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(MODELS)
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (ug/m^3)")
    ax.set_title("AirSense V1 - 2014 LOCKED HELD-OUT TEST\n"
                 "Primary metric: MAE (lower is better). "
                 "Fitted on 2010-2013.")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, target_path)


def figure_actual_vs_predicted(actual, predictions, metrics, target_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True,
                             sharey=True)
    lower = 0.0
    upper = float(max(actual.max(),
                      max(predictions[m].max() for m in MODELS)))
    for index, model_id in enumerate(MODELS):
        ax = axes.ravel()[index]
        ax.scatter(predictions[model_id], actual, s=5, alpha=0.13,
                   color=MODEL_COLORS[model_id], edgecolors="none")
        ax.plot([lower, upper], [lower, upper], color="black",
                linestyle="--", linewidth=1.0)
        ax.set_title("%s  -  MAE %.3f | RMSE %.3f | R2 %.3f"
                     % (model_id, metrics[model_id]["mae"],
                        metrics[model_id]["rmse"], metrics[model_id]["r2"]),
                     fontsize=10)
        ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Predicted PM2.5 (ug/m^3)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual PM2.5 (ug/m^3)")
    fig.suptitle("AirSense V1 - 2014 LOCKED HELD-OUT TEST\n"
                 "actual vs predicted, dashed line is y = x", y=1.01,
                 fontsize=12)
    _save(fig, target_path)


def figure_rmse_r2(metrics, target_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    positions = np.arange(len(MODELS))
    colors = [MODEL_COLORS[m] for m in MODELS]
    rmse = [metrics[m]["rmse"] for m in MODELS]
    axes[0].bar(positions, rmse, color=colors, width=0.6)
    for index, value in enumerate(rmse):
        axes[0].text(index, value + 0.8, "%.3f" % value, ha="center",
                     fontsize=9)
    axes[0].set_ylabel("RMSE (ug/m^3)")
    axes[0].set_title("RMSE (lower is better)")
    axes[0].set_ylim(0, max(rmse) * 1.18)

    r2 = [metrics[m]["r2"] for m in MODELS]
    axes[1].bar(positions, r2, color=colors, width=0.6)
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    for index, value in enumerate(r2):
        offset = 0.02 if value >= 0 else -0.05
        axes[1].text(index, value + offset, "%.3f" % value, ha="center",
                     fontsize=9)
    axes[1].set_ylabel("R2")
    axes[1].set_title("R2 (higher is better)")

    for ax in axes:
        ax.set_xticks(positions)
        ax.set_xticklabels(MODELS)
        ax.set_xlabel("Model")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("AirSense V1 - 2014 LOCKED HELD-OUT TEST - "
                 "secondary metrics", y=1.04, fontsize=12)
    _save(fig, target_path)


def main():
    print("[AIRSENSE V1 - PROTOCOL PHASE 10, STAGE B]")
    print("Final locked held-out evaluation on 2014")
    print("")
    print("Interpreter : %s" % sys.executable)
    print("Python      : %d.%d.%d" % sys.version_info[:3])
    print("")

    # -- gates 1-3 --------------------------------------------------------
    report, pretest_freeze, blind_freeze = verify_freezes()
    print("Pre-test freeze verified : %s (%d hashes)"
          % (report["pretest_freeze_sha256"],
             report["pretest_hashes_verified"]))
    print("Blind freeze verified    : %s (%d hashes)"
          % (report["blind_freeze_sha256"],
             report["blind_hashes_verified"]))
    blind = load_blind_predictions()
    blind_text = load_blind_predictions(as_text=True)
    if list(blind_text.columns) != list(blind.columns):
        raise FinalTestError("blind text and typed views disagree on schema")
    print("Frozen blind predictions verified: %d rows, no target column"
          % len(blind))

    predictions = OrderedDict()
    for model_id in MODELS:
        predictions[model_id] = blind[
            "%s_prediction" % model_id.lower()].values.astype(np.float64)

    # -- single-source-open policy ---------------------------------------
    receipt_path = path(OPENING_RECEIPT)
    snapshot_path = path(TARGET_SNAPSHOT)
    reopened = False
    if os.path.isfile(receipt_path) and os.path.isfile(snapshot_path):
        print("")
        print("Opening receipt already exists -> VERIFICATION/REPORT-ONLY "
              "mode.")
        print("The Phase-4 test partition will NOT be reopened.")
        snapshot = pd.read_csv(snapshot_path)
        actual = snapshot["actual_pm25"].values.astype(np.float64)
        source_sha = json.load(io.open(receipt_path, encoding="utf-8"))[
            "source_test_partition_sha256"]
    else:
        print("")
        print(">>> ALL PRE-OPENING GATES PASSED. OPENING THE 2014 TARGET "
              "NOW. <<<")
        source_frame, actual, timestamps = open_test_target()
        source_sha = sha256_of_file(path(TEST_SOURCE))
        reopened = True

        # alignment must hold before any metric is computed
        if not bool((source_frame["No"].values == blind["No"].values).all()):
            raise FinalTestError(
                "identifier misalignment between the test source and the "
                "frozen blind predictions")
        if not bool((timestamps.astype(str).values
                     == blind["timestamp"].values).all()):
            raise FinalTestError("timestamp misalignment")
        print("Alignment verified: %d identifiers and timestamps match the "
              "frozen predictions, in order" % len(source_frame))

        snapshot = pd.DataFrame(OrderedDict([
            ("matrix_row_number", blind["matrix_row_number"].values),
            ("No", source_frame["No"].values),
            ("timestamp", timestamps.astype(str).values),
            ("actual_pm25", actual),
        ]))
        snapshot.to_csv(snapshot_path, index=False)
        print("Wrote %s" % TARGET_SNAPSHOT)

    snapshot_sha = sha256_of_file(snapshot_path)

    # -- opening receipt --------------------------------------------------
    if not os.path.isfile(receipt_path):
        receipt = OrderedDict()
        receipt["project"] = "AirSense"
        receipt["phase"] = "V1 / Protocol Phase 10 Stage B"
        receipt["pretest_development_freeze_sha256"] = report[
            "pretest_freeze_sha256"]
        receipt["final_blind_prediction_freeze_sha256"] = report[
            "blind_freeze_sha256"]
        receipt["source_test_partition"] = TEST_SOURCE
        receipt["source_test_partition_sha256"] = source_sha
        receipt["final_test_target_snapshot"] = TARGET_SNAPSHOT
        receipt["final_test_target_snapshot_sha256"] = snapshot_sha
        receipt["test_period"] = TEST_PERIOD
        receipt["rows"] = EXPECTED_TEST_ROWS
        receipt["target_access_status"] = "opened"
        receipt["model_state_at_open"] = "frozen"
        receipt["blind_predictions_state_at_open"] = "frozen"
        receipt["primary_metric"] = "MAE"
        receipt["secondary_metrics"] = ["RMSE", "R2"]
        receipt["statement"] = (
            "No development decision may change after this receipt exists. "
            "The 2014 target is now exhausted and can no longer be treated "
            "as unseen.")
        receipt["note"] = (
            "No timestamp is recorded, by reproducibility policy.")
        with io.open(receipt_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, indent=2))
            handle.write("\n")
        print("Wrote %s" % OPENING_RECEIPT)
    receipt_sha = sha256_of_file(receipt_path)

    # -- metrics against ONE shared snapshot ------------------------------
    metrics = OrderedDict()
    for model_id in MODELS:
        metrics[model_id] = compute_metrics(actual, predictions[model_id])
    order = sorted(MODELS, key=lambda m: metrics[m]["mae"])
    rank = OrderedDict((m, order.index(m) + 1) for m in MODELS)
    best = order[0]

    print("")
    print("2014 LOCKED HELD-OUT TEST RESULTS")
    for model_id in MODELS:
        entry = metrics[model_id]
        marker = "  <- best by MAE" if model_id == best else ""
        print("    %-3s  MAE %10.6f   RMSE %10.6f   R2 %9.6f   rank %d%s"
              % (model_id, entry["mae"], entry["rmse"], entry["r2"],
                 rank[model_id], marker))

    # -- final predictions: copied, never recomputed ----------------------
    final = pd.DataFrame(OrderedDict([
        ("matrix_row_number", blind["matrix_row_number"].values),
        ("No", blind["No"].values),
        ("timestamp", blind["timestamp"].values),
        ("actual_pm25", actual),
    ]))
    for model_id in MODELS:
        key = model_id.lower()
        # Verbatim write-through of the frozen prediction strings, so the
        # final artifact is byte-identical to the blind one. Residuals are
        # computed from the parsed values.
        final["%s_prediction" % key] = blind_text["%s_prediction" % key].values
        final["%s_residual" % key] = actual - predictions[model_id]
    final.to_csv(path(FINAL_PREDICTIONS), index=False)

    # Identity gate: the scored predictions must be exactly the frozen ones.
    written = pd.read_csv(path(FINAL_PREDICTIONS), dtype=str,
                          keep_default_na=False, na_filter=False)
    for model_id in MODELS:
        column = "%s_prediction" % model_id.lower()
        if not bool((written[column].values
                     == blind_text[column].values).all()):
            raise FinalTestError(
                "CRITICAL: %s in the final artifact is not byte-identical "
                "to the frozen blind prediction" % column)

    metrics_frame = pd.DataFrame([OrderedDict([
        ("model", model_id),
        ("final_fit_period", FINAL_FIT_PERIOD),
        ("test_period", TEST_PERIOD),
        ("n_development", EXPECTED_DEVELOPMENT_ROWS),
        ("n_test", EXPECTED_TEST_ROWS),
        ("mae", metrics[model_id]["mae"]),
        ("rmse", metrics[model_id]["rmse"]),
        ("r2", metrics[model_id]["r2"]),
        ("mae_rank", rank[model_id]),
    ]) for model_id in MODELS])
    metrics_frame.to_csv(path(FINAL_METRICS), index=False)

    residuals_frame = pd.DataFrame([
        residual_summary(m, actual, predictions[m]) for m in MODELS])
    residuals_frame.to_csv(path(RESIDUAL_SUMMARY), index=False)

    # -- development-validation vs final-test comparison ------------------
    rows = []
    for model_id in MODELS:
        manifest = json.load(io.open(
            path(os.path.join("artifacts", DEV_MANIFESTS[model_id])),
            encoding="utf-8"))
        dev = manifest["validation_metrics"]
        entry = metrics[model_id]
        rows.append(OrderedDict([
            ("model", model_id),
            ("used_2013_for_hyperparameter_selection",
             model_id in ("M2", "M3")),
            ("validation_2013_mae", float(dev["mae"])),
            ("test_2014_mae", entry["mae"]),
            ("mae_delta_test_minus_validation",
             entry["mae"] - float(dev["mae"])),
            ("validation_2013_rmse", float(dev["rmse"])),
            ("test_2014_rmse", entry["rmse"]),
            ("rmse_delta_test_minus_validation",
             entry["rmse"] - float(dev["rmse"])),
            ("validation_2013_r2", float(dev["r2"])),
            ("test_2014_r2", entry["r2"]),
            ("r2_delta_test_minus_validation",
             entry["r2"] - float(dev["r2"])),
        ]))
    comparison_frame = pd.DataFrame(rows)
    comparison_frame.to_csv(path(DEV_TO_TEST), index=False)
    print("")
    print("Wrote metrics, predictions, residual summary and "
          "development-to-test comparison")

    # -- figures -----------------------------------------------------------
    figures = OrderedDict()
    for name, builder in (
            ("final_test_model_comparison_mae.png",
             lambda p: figure_mae_comparison(metrics, best, p)),
            ("final_test_actual_vs_predicted.png",
             lambda p: figure_actual_vs_predicted(actual, predictions,
                                                  metrics, p)),
            ("final_test_rmse_r2.png",
             lambda p: figure_rmse_r2(metrics, p))):
        target_path = path(os.path.join(FIGURES_DIR, name))
        builder(target_path)
        figures[os.path.join(FIGURES_DIR, name)] = sha256_of_file(
            target_path)
    print("Wrote 3 figures")

    # -- final evaluation manifest ----------------------------------------
    manifest = OrderedDict()
    manifest["project"] = "AirSense"
    manifest["phase"] = ("V1 / Protocol Phase 10 - final locked held-out "
                         "evaluation")
    manifest["raw_dataset_sha256"] = report["raw_sha256"]
    manifest["split_manifest_sha256"] = pretest_freeze[
        "provenance_sha256"]["artifacts/split_manifest.json"]
    manifest["feature_schema_sha256"] = pretest_freeze[
        "provenance_sha256"]["artifacts/feature_schema.json"]
    manifest["pretest_development_freeze_sha256"] = report[
        "pretest_freeze_sha256"]
    manifest["final_blind_prediction_freeze_sha256"] = report[
        "blind_freeze_sha256"]
    manifest["final_test_opening_receipt_sha256"] = receipt_sha
    manifest["final_test_target_snapshot_sha256"] = snapshot_sha
    manifest["blind_prediction_sha256"] = sha256_of_file(
        path(BLIND_PREDICTIONS))
    manifest["final_fit_period"] = FINAL_FIT_PERIOD
    manifest["test_period"] = TEST_PERIOD
    manifest["n_development"] = EXPECTED_DEVELOPMENT_ROWS
    manifest["n_test"] = EXPECTED_TEST_ROWS
    manifest["primary_metric"] = "MAE"
    manifest["secondary_metrics"] = ["RMSE", "R2"]

    refit = json.load(io.open(path("artifacts/final_refit_manifest.json"),
                              encoding="utf-8"))
    model_block = OrderedDict()
    for model_id in MODELS:
        entry = OrderedDict()
        entry["final_specification"] = refit["models"][model_id]
        entry["final_development_population"] = EXPECTED_DEVELOPMENT_ROWS
        entry["test_population"] = EXPECTED_TEST_ROWS
        entry["test_mae"] = metrics[model_id]["mae"]
        entry["test_rmse"] = metrics[model_id]["rmse"]
        entry["test_r2"] = metrics[model_id]["r2"]
        entry["mae_rank"] = rank[model_id]
        model_block[model_id] = entry
    manifest["models"] = model_block
    manifest["best_final_model_by_mae"] = best
    manifest["final_test_status"] = "evaluated"
    manifest["post_test_model_changes"] = "none"
    manifest["test_set_exhaustion_statement"] = (
        "The 2014 target has now been opened. It is no longer available as "
        "an untouched test set. Any model changed after this point must "
        "not claim a fresh evaluation on the same 2014 split.")

    outputs = OrderedDict()
    for relative in (FINAL_METRICS, FINAL_PREDICTIONS, RESIDUAL_SUMMARY,
                     DEV_TO_TEST, TARGET_SNAPSHOT, BLIND_PREDICTIONS):
        outputs[relative] = sha256_of_file(path(relative))
    for relative, digest in figures.items():
        outputs[relative] = digest
    outputs[OPENING_RECEIPT] = receipt_sha
    manifest["outputs_sha256"] = outputs
    manifest["note"] = (
        "No execution timestamp is recorded so that repeated runs are "
        "byte-identical.")

    with io.open(path(EVALUATION_MANIFEST), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2))
        handle.write("\n")
    print("Wrote %s" % EVALUATION_MANIFEST)

    print("")
    print("Best final model by MAE: %s" % best)
    print("Ranking by MAE: %s" % " < ".join(order))
    print("")
    print("The 2014 target is now EXHAUSTED. It cannot honestly be called "
          "unseen again.")
    if reopened:
        print("(This run performed the single authorized target opening.)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FinalTestError as exc:
        sys.stderr.write("\nSTAGE B ABORTED: %s\n" % exc)
        sys.exit(1)
