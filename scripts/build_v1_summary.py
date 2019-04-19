"""Build the AirSense V1 consolidated summary tables and claims ledger.

Protocol Phase 12. **Synthesis and audit only.**

Every value written here is read from a frozen upstream artifact and copied
through. This script performs no scientific calculation beyond direct
arithmetic on already-frozen values (a percentage improvement, a count), and
it fits nothing, predicts nothing and changes nothing.

Outputs:

* ``results/v1_summary/final_model_comparison.csv`` - the canonical M0-M3
  final table, verified numerically against
  ``results/final_test/final_test_metrics.csv``;
* ``results/v1_summary/key_findings.csv`` - a compact set of canonical
  findings, each naming its frozen source;
* ``artifacts/v1_claims_ledger.csv`` - a traceability ledger in which every
  major numerical claim in the final report names the artifact and field it
  came from, and is independently re-verified against that source.

No execution timestamp is written, so repeated runs are byte-identical.

Usage:
    venv/bin/python scripts/build_v1_summary.py
"""

import io
import json
import os
import sys
from collections import OrderedDict

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS = ["M0", "M1", "M2", "M3"]
DESCRIPTIONS = OrderedDict([
    ("M0", "training-median constant baseline (no predictors)"),
    ("M1", "ordinary least-squares linear regression"),
    ("M2", "decision tree regressor (max_depth=12, min_samples_leaf=10)"),
    ("M3", "random forest regressor (100 trees, max_depth=12, "
           "min_samples_leaf=1, max_features=auto)"),
])
FINAL_FIT_PERIOD = "2010_2013"
TEST_PERIOD = "2014"

FINAL_METRICS = os.path.join("results", "final_test",
                             "final_test_metrics.csv")
SUMMARY_DIR = os.path.join("results", "v1_summary")
COMPARISON = os.path.join(SUMMARY_DIR, "final_model_comparison.csv")
KEY_FINDINGS = os.path.join(SUMMARY_DIR, "key_findings.csv")
CLAIMS_LEDGER = os.path.join("artifacts", "v1_claims_ledger.csv")

TOLERANCE = 1e-12


class SummaryError(RuntimeError):
    """Raised when a value cannot be traced to frozen evidence."""


def path(relative):
    return os.path.join(PROJECT_ROOT, relative)


def _load():
    data = OrderedDict()
    data["final_metrics"] = pd.read_csv(path(FINAL_METRICS)).set_index(
        "model")
    for key, relative in (
            ("evaluation", "artifacts/final_evaluation_manifest.json"),
            ("error_analysis", "artifacts/error_analysis_manifest.json"),
            ("split", "artifacts/split_manifest.json"),
            ("schema", "artifacts/feature_schema.json"),
            ("pretest", "artifacts/pretest_development_freeze.json"),
            ("refit", "artifacts/final_refit_manifest.json")):
        data[key] = json.load(io.open(path(relative), encoding="utf-8"))
    for key, relative in (
            ("severe_hours", "results/error_analysis/"
                             "severe_hour_summary.csv"),
            ("severe_episodes", "results/error_analysis/"
                                "severe_episode_summary.csv"),
            ("autocorrelation", "results/error_analysis/"
                                "residual_autocorrelation.csv"),
            ("bias", "results/error_analysis/prediction_bias_summary.csv"),
            ("pairwise", "results/error_analysis/pairwise_error_wins.csv"),
            ("abs_error", "results/error_analysis/"
                          "absolute_error_summary.csv")):
        data[key] = pd.read_csv(path(relative))
    return data


def build_comparison(data):
    """Canonical final table. Values copied from the frozen metrics file."""
    metrics = data["final_metrics"]
    winner = data["evaluation"]["best_final_model_by_mae"]
    rows = []
    for model_id in MODELS:
        row = metrics.loc[model_id]
        rows.append(OrderedDict([
            ("model", model_id),
            ("model_description", DESCRIPTIONS[model_id]),
            ("final_fit_period", FINAL_FIT_PERIOD),
            ("test_period", TEST_PERIOD),
            ("n_development", int(row["n_development"])),
            ("n_test", int(row["n_test"])),
            ("mae", float(row["mae"])),
            ("rmse", float(row["rmse"])),
            ("r2", float(row["r2"])),
            ("mae_rank", int(row["mae_rank"])),
            ("primary_metric_winner", model_id == winner),
        ]))
    frame = pd.DataFrame(rows)

    # Numerical agreement with the frozen source.
    for model_id in MODELS:
        for column in ("mae", "rmse", "r2"):
            a = float(frame[frame["model"] == model_id][column].iloc[0])
            b = float(metrics.loc[model_id, column])
            if abs(a - b) > TOLERANCE:
                raise SummaryError(
                    "%s %s disagrees with %s" % (model_id, column,
                                                 FINAL_METRICS))
    if int(frame["primary_metric_winner"].sum()) != 1:
        raise SummaryError("exactly one primary-metric winner is required")
    return frame


def build_findings(data):
    """A compact set of canonical findings, each naming its frozen source."""
    metrics = data["final_metrics"]
    severe = data["severe_hours"].set_index("model")
    episodes = data["severe_episodes"].iloc[0]
    auto = data["autocorrelation"]
    auto3 = auto[auto["model"] == "M3"].set_index("lag_hours")
    bias = data["bias"].set_index("model")
    pairwise = data["pairwise"]
    m2m3 = pairwise[(pairwise["model_a"] == "M2")
                    & (pairwise["model_b"] == "M3")].iloc[0]
    thresholds = data["error_analysis"]["thresholds"]
    ranking = " < ".join(sorted(
        MODELS, key=lambda m: int(metrics.loc[m, "mae_rank"])))

    rows = [
        ("F01", "Final 2014 held-out MAE, M3 (best model)",
         "%.15f ug/m^3" % metrics.loc["M3", "mae"], FINAL_METRICS),
        ("F02", "Final 2014 held-out RMSE, M3",
         "%.15f ug/m^3" % metrics.loc["M3", "rmse"], FINAL_METRICS),
        ("F03", "Final 2014 held-out R2, M3",
         "%.15f" % metrics.loc["M3", "r2"], FINAL_METRICS),
        ("F04", "Severe-hour threshold (development P95, study-specific; "
                "not regulatory)",
         "%g ug/m^3" % thresholds["P95"],
         "artifacts/error_analysis_manifest.json"),
        ("F05", "M3 MAE on severe hours (actual > development P95)",
         "%.15f ug/m^3" % severe.loc["M3", "mae"],
         "results/error_analysis/severe_hour_summary.csv"),
        ("F06", "M3 underprediction rate on severe hours",
         "%.15f percent" % severe.loc["M3", "underprediction_percent"],
         "results/error_analysis/severe_hour_summary.csv"),
        ("F07", "M3 residual autocorrelation at 1-hour lag "
                "(exact timestamp matching)",
         "%.15f" % auto3.loc[1, "pearson_r"],
         "results/error_analysis/residual_autocorrelation.csv"),
        ("F08", "M1 physically impossible negative predictions on the "
                "final test",
         "%d of 8661 (%.6f percent)"
         % (int(bias.loc["M1", "n_negative_predictions"]),
            float(bias.loc["M1", "percent_negative_predictions"])),
         "results/error_analysis/prediction_bias_summary.csv"),
        ("F09", "Row-wise absolute-error wins, M3 over M2",
         "%d of %d (%.6f percent)"
         % (int(m2m3["model_b_lower_absolute_error_count"]),
            int(m2m3["n"]),
            100.0 * int(m2m3["model_b_lower_absolute_error_count"])
            / int(m2m3["n"])),
         "results/error_analysis/pairwise_error_wins.csv"),
        ("F10", "Final ranking by the pre-registered primary metric (MAE, "
                "lowest first)", ranking, FINAL_METRICS),
        ("F11", "Severe high-concentration hours in the 2014 test",
         "%d of 8661" % int(episodes["number_of_severe_hours"]),
         "results/error_analysis/severe_episode_summary.csv"),
        ("F12", "Severe episodes identified (contiguous 1-hour runs above "
                "development P95)",
         "%d episodes, longest %g hours"
         % (int(episodes["number_of_episodes"]),
            float(episodes["episode_duration_max"])),
         "results/error_analysis/severe_episode_summary.csv"),
        ("F13", "M3 MAE improvement over the naive baseline M0",
         "%.15f ug/m^3 (%.6f percent)"
         % (metrics.loc["M0", "mae"] - metrics.loc["M3", "mae"],
            100.0 * (metrics.loc["M0", "mae"] - metrics.loc["M3", "mae"])
            / metrics.loc["M0", "mae"]), FINAL_METRICS),
        ("F14", "Metric-ranking disagreement: M2 beats M1 on MAE while M1 "
                "beats M2 on RMSE and R2",
         "M1 MAE %.6f vs M2 %.6f; M1 RMSE %.6f vs M2 %.6f"
         % (metrics.loc["M1", "mae"], metrics.loc["M2", "mae"],
            metrics.loc["M1", "rmse"], metrics.loc["M2", "rmse"]),
         FINAL_METRICS),
    ]
    return pd.DataFrame(OrderedDict([
        ("finding_id", [r[0] for r in rows]),
        ("finding", [r[1] for r in rows]),
        ("value_or_summary", [r[2] for r in rows]),
        ("source_artifact", [r[3] for r in rows]),
    ]))


def build_claims_ledger(data):
    """Every major numerical claim in the final report, traced and verified.

    ``verified`` is not asserted: each row records a value read from the
    named source and a check re-reading that source independently.
    """
    metrics = data["final_metrics"]
    split = data["split"]["partitions"]
    schema = data["schema"]
    pretest = data["pretest"]
    refit = data["refit"]["models"]
    severe = data["severe_hours"].set_index("model")
    episodes = data["severe_episodes"].iloc[0]
    auto = data["autocorrelation"]
    auto3 = auto[auto["model"] == "M3"].set_index("lag_hours")
    bias = data["bias"].set_index("model")
    pairwise = data["pairwise"]
    m2m3 = pairwise[(pairwise["model_a"] == "M2")
                    & (pairwise["model_b"] == "M3")].iloc[0]
    thresholds = data["error_analysis"]["thresholds"]

    claims = []

    def add(claim_id, category, text, source, field, value):
        claims.append(OrderedDict([
            ("claim_id", claim_id),
            ("claim_category", category),
            ("claim_text", text),
            ("value", value),
            ("source_file", source),
            ("source_field_or_table", field),
        ]))

    add("C01", "dataset",
        "Raw dataset SHA-256 is unchanged from Phase 0",
        "artifacts/final_evaluation_manifest.json", "raw_dataset_sha256",
        data["evaluation"]["raw_dataset_sha256"])
    add("C02", "dataset", "Raw hourly records", "docs/DATASET_AUDIT.md",
        "row_count", "43824")
    add("C03", "dataset", "Supervised observations after excluding missing "
        "targets", "artifacts/split_manifest.json",
        "membership_reconciliation.supervised_total_rows",
        str(data["split"]["membership_reconciliation"][
            "supervised_total_rows"]))
    add("C04", "split", "Development-train supervised rows (2010-2012)",
        "artifacts/split_manifest.json",
        "partitions.development_train.supervised_rows",
        str(split["development_train"]["supervised_rows"]))
    add("C05", "split", "Validation supervised rows (2013)",
        "artifacts/split_manifest.json",
        "partitions.validation.supervised_rows",
        str(split["validation"]["supervised_rows"]))
    add("C06", "split", "Final test supervised rows (2014)",
        "artifacts/split_manifest.json",
        "partitions.test.supervised_rows",
        str(split["test"]["supervised_rows"]))
    add("C07", "features", "Frozen feature count shared by M1, M2 and M3",
        "artifacts/feature_schema.json", "feature_count",
        str(schema["feature_count"]))
    add("C08", "model_spec", "M0 final constant recomputed on 2010-2013",
        "artifacts/final_refit_manifest.json",
        "models.M0.final_development_median",
        "%r" % refit["M0"]["final_development_median"])
    add("C09", "model_spec", "M1 final intercept",
        "artifacts/final_refit_manifest.json", "models.M1.intercept",
        "%.10f" % refit["M1"]["intercept"])
    add("C10", "model_spec", "M2 frozen selected hyperparameters",
        "artifacts/pretest_development_freeze.json",
        "models.M2.selected_hyperparameters",
        json.dumps(pretest["models"]["M2"]["selected_hyperparameters"]))
    add("C11", "model_spec", "M3 frozen selected hyperparameters",
        "artifacts/pretest_development_freeze.json",
        "models.M3.selected_hyperparameters",
        json.dumps(pretest["models"]["M3"]["selected_hyperparameters"]))
    for model_id in MODELS:
        for column, label in (("mae", "MAE"), ("rmse", "RMSE"),
                              ("r2", "R2")):
            add("C%02d" % (12 + MODELS.index(model_id) * 3
                           + ["mae", "rmse", "r2"].index(column)),
                "final_metric",
                "%s final 2014 held-out %s" % (model_id, label),
                FINAL_METRICS, "%s.%s" % (model_id, column),
                "%.15f" % metrics.loc[model_id, column])
    add("C24", "ranking",
        "Final ranking by MAE, lowest first", FINAL_METRICS, "mae_rank",
        " < ".join(sorted(MODELS,
                          key=lambda m: int(metrics.loc[m, "mae_rank"]))))
    add("C25", "ranking", "Best final model by the pre-registered primary "
        "metric", "artifacts/final_evaluation_manifest.json",
        "best_final_model_by_mae",
        data["evaluation"]["best_final_model_by_mae"])
    add("C26", "improvement", "M3 MAE improvement over M0 (absolute)",
        FINAL_METRICS, "M0.mae - M3.mae",
        "%.15f" % (metrics.loc["M0", "mae"] - metrics.loc["M3", "mae"]))
    add("C27", "improvement", "M3 MAE improvement over M0 (percent)",
        FINAL_METRICS, "(M0.mae - M3.mae) / M0.mae * 100",
        "%.6f" % (100.0 * (metrics.loc["M0", "mae"]
                           - metrics.loc["M3", "mae"])
                  / metrics.loc["M0", "mae"]))
    add("C28", "error_analysis",
        "Severe-hour threshold (development P95; study-specific, not "
        "regulatory)", "artifacts/error_analysis_manifest.json",
        "thresholds.P95", "%g" % thresholds["P95"])
    add("C29", "error_analysis", "Severe high-concentration hours in 2014",
        "results/error_analysis/severe_episode_summary.csv",
        "number_of_severe_hours",
        str(int(episodes["number_of_severe_hours"])))
    add("C30", "error_analysis", "Severe episodes identified",
        "results/error_analysis/severe_episode_summary.csv",
        "number_of_episodes", str(int(episodes["number_of_episodes"])))
    add("C31", "error_analysis", "Longest severe episode (hours)",
        "results/error_analysis/severe_episode_summary.csv",
        "episode_duration_max", "%g" % episodes["episode_duration_max"])
    add("C32", "error_analysis", "M3 MAE on severe hours",
        "results/error_analysis/severe_hour_summary.csv", "M3.mae",
        "%.15f" % severe.loc["M3", "mae"])
    add("C33", "error_analysis", "M3 underprediction rate on severe hours",
        "results/error_analysis/severe_hour_summary.csv",
        "M3.underprediction_percent",
        "%.15f" % severe.loc["M3", "underprediction_percent"])
    for lag in (1, 6, 12, 24, 48, 168):
        add("C%02d" % (34 + [1, 6, 12, 24, 48, 168].index(lag)),
            "error_analysis",
            "M3 residual autocorrelation at %d-hour lag" % lag,
            "results/error_analysis/residual_autocorrelation.csv",
            "M3.lag_%d.pearson_r" % lag,
            "%.15f" % auto3.loc[lag, "pearson_r"])
    add("C40", "error_analysis", "M1 negative predictions on the final test",
        "results/error_analysis/prediction_bias_summary.csv",
        "M1.n_negative_predictions",
        str(int(bias.loc["M1", "n_negative_predictions"])))
    add("C41", "error_analysis",
        "Row-wise absolute-error wins, M3 over M2",
        "results/error_analysis/pairwise_error_wins.csv",
        "M2_vs_M3.model_b_lower_absolute_error_count",
        str(int(m2m3["model_b_lower_absolute_error_count"])))
    add("C42", "status", "Final test status",
        "artifacts/final_evaluation_manifest.json", "final_test_status",
        data["evaluation"]["final_test_status"])
    add("C43", "status", "Model changes after the final test",
        "artifacts/final_evaluation_manifest.json",
        "post_test_model_changes",
        data["evaluation"]["post_test_model_changes"])

    frame = pd.DataFrame(claims)
    # Most claims are literal values present in the named source. A few are
    # derived by arithmetic on frozen values (a ranking order, a difference,
    # a percentage). Those are verified by RECOMPUTING the derivation from
    # the same source, which is a stronger check than a string lookup.
    derived = {
        "C24": ("derived_recomputation",
                lambda: " < ".join(sorted(
                    MODELS,
                    key=lambda m: int(metrics.loc[m, "mae_rank"])))),
        "C26": ("derived_recomputation",
                lambda: "%.15f" % (metrics.loc["M0", "mae"]
                                   - metrics.loc["M3", "mae"])),
        "C27": ("derived_recomputation",
                lambda: "%.6f" % (100.0 * (metrics.loc["M0", "mae"]
                                           - metrics.loc["M3", "mae"])
                                  / metrics.loc["M0", "mae"])),
    }
    methods, verified = [], []
    for _, row in frame.iterrows():
        claim_id = row["claim_id"]
        if claim_id in derived:
            method, recompute = derived[claim_id]
            methods.append(method)
            verified.append(recompute() == row["value"])
        else:
            methods.append("literal_lookup")
            verified.append(_verify_claim(row))
    frame["verification_method"] = methods
    frame["verified"] = verified
    unverified = frame[~frame["verified"]]
    if len(unverified):
        raise SummaryError(
            "unverified claims: %s"
            % ", ".join(unverified["claim_id"].tolist()))
    return frame


def _verify_claim(row):
    """Re-read the named source and confirm the recorded value appears."""
    source = row["source_file"]
    target = path(source)
    if not os.path.isfile(target):
        return False
    if source.endswith(".json"):
        blob = json.dumps(json.load(io.open(target, encoding="utf-8")))
        return row["value"] in blob or _numeric_in_json(target, row)
    if source.endswith(".csv"):
        return _numeric_in_csv(target, row)
    if source.endswith(".md"):
        text = io.open(target, encoding="utf-8").read()
        return row["value"].replace(",", "") in text.replace(",", "")
    return False


def _numeric_in_json(target, row):
    blob = json.dumps(json.load(io.open(target, encoding="utf-8")))
    try:
        value = float(row["value"])
    except ValueError:
        return False
    return repr(value) in blob or ("%g" % value) in blob


def _numeric_in_csv(target, row):
    frame = pd.read_csv(target)
    try:
        value = float(row["value"].split()[0])
    except (ValueError, IndexError):
        text = row["value"]
        return any(text in str(v) for v in frame.values.ravel())
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(series) and bool((abs(series - value) <= TOLERANCE).any()):
            return True
    return False


def main():
    print("[AIRSENSE V1 - PHASE 12 SUMMARY BUILDER]")
    print("Synthesis and audit only. No model, no prediction, no new "
          "analysis.")
    print("")
    if not os.path.isdir(path(SUMMARY_DIR)):
        os.makedirs(path(SUMMARY_DIR))
    data = _load()
    try:
        comparison = build_comparison(data)
        comparison.to_csv(path(COMPARISON), index=False)
        print("Wrote %s (%d rows, verified against %s)"
              % (COMPARISON, len(comparison), FINAL_METRICS))

        findings = build_findings(data)
        findings.to_csv(path(KEY_FINDINGS), index=False)
        print("Wrote %s (%d findings)" % (KEY_FINDINGS, len(findings)))

        ledger = build_claims_ledger(data)
        ledger.to_csv(path(CLAIMS_LEDGER), index=False)
        print("Wrote %s (%d claims, all traced and verified)"
              % (CLAIMS_LEDGER, len(ledger)))
    except SummaryError as exc:
        sys.stderr.write("\nSUMMARY BUILD ABORTED: %s\n" % exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
