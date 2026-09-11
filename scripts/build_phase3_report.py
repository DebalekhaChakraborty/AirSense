"""Development-validation reporting for AirSense V2.

Protocol Phase 3, Stage B reporting. Reads the prediction arrays produced by
``run_phase3_development.py`` and writes the metric tables, diagnostics and
figures.

Everything here is **development validation**, never final test. No prediction
is clipped, no threshold is derived from validation, and the locked test
partition is not opened.

Usage:
    .venv-v2/bin/python scripts/build_phase3_report.py
"""

import csv
import hashlib
import io
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import STATIONS                        # noqa: E402
from src.evaluation import metrics as M                            # noqa: E402
from src.models.baselines import B0_SOURCE_LABELS                   # noqa: E402
from src.visualization.render import (                              # noqa: E402
    BLUE, ORANGE, RED, TEAL, Figure, heatmap)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
OUT = PROJECT_ROOT / "results" / "validation"
FIGURES = PROJECT_ROOT / "figures"
HORIZONS = [1, 6, 12, 24]
MODELS = ["B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2"]
TITLE = "DEVELOPMENT VALIDATION"
SUBTITLE = "2015-03-01 - 2016-02-29  |  NOT FINAL TEST  |  test remains sealed"


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def fmt(value, digits=6):
    return "" if value is None else round(float(value), digits)


def load_index():
    stations, horizons = [], []
    with open(str(PROCESSED / "sample_index" / "validation.csv"),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stations.append(row["station"])
            horizons.append(int(row["horizon_hours"]))
    return np.array(stations), np.array(horizons, dtype=np.int32)


RECORDED = {
    "free_disk_before_kb": 18484028,
    "free_disk_after_kb": None,          # filled by --write-manifest
}

MANIFEST_DOCUMENTS = [
    "docs/PHASE_03_CLASSICAL_BASELINES_RECORD.md",
    "docs/CLASSICAL_BASELINE_RESULTS.md",
    "requirements-v2-core.txt",
    "requirements-v2-core-lock.txt",
]


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def write_manifest(free_after_kb):
    import platform
    artifacts = PROJECT_ROOT / "artifacts"
    selected = []
    with open(str(PROJECT_ROOT / "results" / "models" / "b3"
                  / "b3_selected_models.csv"), encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    summary = json.load(open(str(artifacts / "phase3_validation_summary.json"),
                             encoding="utf-8"))
    freeze = json.load(open(str(artifacts / "phase3_prevalidation_freeze.json"),
                            encoding="utf-8"))
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("phase", "phase_3_classical_development_baselines"),
        ("foundation_manifest_sha256",
         sha256_of_file(artifacts / "v2_foundation_manifest.json")),
        ("phase1_manifest_sha256",
         sha256_of_file(artifacts / "v2_phase1_manifest.json")),
        ("phase2_manifest_sha256",
         sha256_of_file(artifacts / "v2_phase2_manifest.json")),
        ("prevalidation_freeze_sha256",
         sha256_of_file(artifacts / "phase3_prevalidation_freeze.json")),
        ("validation_opening_receipt_sha256",
         sha256_of_file(artifacts
                        / "validation_development_opening_receipt.json")),
        ("runtime", freeze["runtime"]),
        ("b3_grid_sha256", freeze["b3_candidate_grid"]["grid_sha256"]),
        ("selected_candidates", OrderedDict(
            ("%s_h%s" % (row["regime"], row["horizon"]),
             row["selected_candidate"]) for row in selected)),
        ("selected_model_sha256", OrderedDict(
            (row["model_file"], row["model_sha256"]) for row in selected)),
        ("prediction_array_sha256", OrderedDict(
            (path.name, sha256_of_file(path))
            for path in sorted(PRED.glob("*.npy")))),
        ("metric_table_sha256", OrderedDict(
            (path.name, sha256_of_file(path))
            for path in sorted(OUT.glob("*.csv")))),
        ("b3_table_sha256", OrderedDict(
            (path.name, sha256_of_file(path)) for path in
            sorted((PROJECT_ROOT / "results" / "models" / "b3").glob("*.csv")))),
        ("figure_sha256", OrderedDict(
            (path.name, sha256_of_file(path))
            for path in sorted(FIGURES.glob("validation_*.png")))),
        ("document_sha256", OrderedDict(
            (name, sha256_of_file(PROJECT_ROOT / name))
            for name in MANIFEST_DOCUMENTS
            if (PROJECT_ROOT / name).is_file())),
        ("severe_threshold", 244.0),
        ("severe_threshold_source", "training_pooled_p95"),
        ("residual_convention", "residual = actual - prediction"),
        ("validation_status", "development_open"),
        ("final_test_status", "sealed"),
        ("model_runs", 96),
        ("selected_models", len(selected)),
        ("validation_predictions_generated",
         6 * summary["validation_samples"]),
        ("validation_metrics_seen", 1),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("predictions_clipped", False),
        ("neural_frameworks_installed", False),
        ("tsfm_checkpoints_downloaded", False),
        ("v1_legacy_modified", False),
        ("free_disk_before_kb", RECORDED["free_disk_before_kb"]),
        ("free_disk_after_kb", free_after_kb),
        ("note", "Development validation only. No wall-clock timestamp is "
                 "recorded in canonical scientific content; disk figures are "
                 "recorded observations written as literals."),
    ])
    path = artifacts / "v2_phase3_manifest.json"
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print("wrote %s" % path.relative_to(PROJECT_ROOT))
    print("sha256 %s" % sha256_of_file(path))


def main():
    if "--write-manifest" in sys.argv:
        index = sys.argv.index("--write-manifest")
        write_manifest(int(sys.argv[index + 1]))
        return 0
    station, horizon = load_index()
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    predictions = OrderedDict()
    for name in MODELS:
        path = PRED / ("%s_validation.npy" % name)
        predictions[name] = np.load(str(path)).astype(np.float64)
        if predictions[name].size != actual.size:
            raise SystemExit("%s misaligned with the canonical index" % name)

    # ---- station x horizon: the 48-cell basis of the primary metric -----
    cells = OrderedDict()
    rows = []
    for name in MODELS:
        cells[name] = M.cell_metrics(actual, predictions[name], station,
                                     horizon)
        for (st, hz), cell in cells[name].items():
            rows.append([name, st, hz, cell["n"], fmt(cell["mae"]),
                         fmt(cell["rmse"]), fmt(cell["r2"])])
    write_csv(OUT / "metrics_by_station_horizon.csv",
              ["model", "station", "horizon", "n", "MAE", "RMSE", "R2"], rows)

    # ---- primary comparison ---------------------------------------------
    summary = OrderedDict()
    for name in MODELS:
        summary[name] = OrderedDict([
            ("macro_station_horizon_MAE", M.macro_from_cells(cells[name])),
            ("micro_MAE", M.mae(actual, predictions[name])),
            ("macro_RMSE", M.macro_from_cells(cells[name], "rmse")),
            ("micro_RMSE", M.rmse(actual, predictions[name])),
            ("macro_R2", M.macro_from_cells(cells[name], "r2")),
        ])
    ranking = sorted(MODELS,
                     key=lambda m: summary[m]["macro_station_horizon_MAE"])
    write_csv(OUT / "model_comparison.csv",
              ["model", "macro_station_horizon_MAE", "micro_MAE",
               "macro_RMSE", "micro_RMSE", "macro_R2",
               "rank_by_primary_metric"],
              [[name] + [fmt(summary[name][k]) for k in
                         ("macro_station_horizon_MAE", "micro_MAE",
                          "macro_RMSE", "micro_RMSE", "macro_R2")]
               + [ranking.index(name) + 1] for name in MODELS])

    # ---- horizon and station tables --------------------------------------
    horizon_rows = []
    for name in MODELS:
        for hz in HORIZONS:
            selector = horizon == hz
            horizon_rows.append([
                name, hz, int(selector.sum()),
                fmt(M.mae(actual[selector], predictions[name][selector])),
                fmt(M.rmse(actual[selector], predictions[name][selector])),
                fmt(M.r2(actual[selector], predictions[name][selector])),
                fmt(M.macro_station_mae(actual[selector],
                                        predictions[name][selector],
                                        station[selector]))])
    write_csv(OUT / "metrics_by_horizon.csv",
              ["model", "horizon", "n", "micro_MAE", "micro_RMSE", "R2",
               "macro_station_MAE"], horizon_rows)

    station_rows = []
    for name in MODELS:
        for st in STATIONS:
            selector = station == st
            subset = {k: v for k, v in cells[name].items() if k[0] == st}
            station_rows.append([
                name, st, int(selector.sum()),
                fmt(M.macro_from_cells(subset)),
                fmt(M.macro_from_cells(subset, "rmse")),
                fmt(M.macro_from_cells(subset, "r2"))])
    write_csv(OUT / "metrics_by_station.csv",
              ["model", "station", "n", "macro_over_horizon_MAE",
               "macro_over_horizon_RMSE", "macro_over_horizon_R2"],
              station_rows)

    # ---- severe endpoint --------------------------------------------------
    severe_rows = []
    severe = OrderedDict()
    for name in MODELS:
        severe[name] = M.severe_metrics(actual, predictions[name], station,
                                        horizon)
        s = severe[name]
        severe_rows.append([
            name, s["severe_n"], fmt(s["severe_mae"]), fmt(s["severe_rmse"]),
            fmt(s["severe_mean_residual"]),
            fmt(s["severe_underprediction_pct"]),
            fmt(s["severe_macro_station_horizon_mae"]),
            s["severe_contributing_cells"], s["severe_total_cells"],
            s["severe_empty_cells"]])
    write_csv(OUT / "severe_metrics.csv",
              ["model", "severe_n", "severe_MAE", "severe_RMSE",
               "severe_mean_residual", "severe_underprediction_pct",
               "severe_macro_station_horizon_MAE", "contributing_cells",
               "total_cells", "empty_cells"], severe_rows)

    # ---- information regime gains ----------------------------------------
    gain_rows = []

    def gain(scope, label, a, b, selector=None):
        if selector is None:
            selector = np.ones(actual.size, dtype=bool)
        left = M.macro_from_cells(M.cell_metrics(
            actual[selector], predictions[a][selector], station[selector],
            horizon[selector]))
        right = M.macro_from_cells(M.cell_metrics(
            actual[selector], predictions[b][selector], station[selector],
            horizon[selector]))
        gain_rows.append([scope, label, a, b, fmt(left), fmt(right),
                          fmt(left - right),
                          fmt(100.0 * (left - right) / left)])

    severe_mask = actual > M.SEVERE_THRESHOLD
    for a, b, label in (("B3_R0", "B3_R1", "R0_to_R1_pm25_history"),
                        ("B3_R1", "B3_R2", "R1_to_R2_copollutant_history")):
        gain("overall", label, a, b)
        for hz in HORIZONS:
            gain("horizon_%d" % hz, label, a, b, horizon == hz)
        gain("severe", label, a, b, severe_mask)
    write_csv(OUT / "information_regime_gain.csv",
              ["scope", "increment", "from_model", "to_model",
               "from_macro_MAE", "to_macro_MAE", "absolute_MAE_improvement",
               "relative_MAE_improvement_pct"], gain_rows)

    # ---- diagnostics -------------------------------------------------------
    b0_source = np.load(str(PRED / "B0_source_validation.npy"))
    b1_direct = np.load(str(PRED / "B1_direct_validation.npy")).astype(bool)
    b2_level = np.load(str(PRED / "B2_level_validation.npy"))
    diagnostic_rows = []
    for hz in HORIZONS:
        selector = horizon == hz
        for code, label in sorted(B0_SOURCE_LABELS.items()):
            diagnostic_rows.append(["B0", "horizon_%d" % hz, label,
                                    int((b0_source[selector] == code).sum()),
                                    int(selector.sum())])
    for st in STATIONS:
        selector = station == st
        for code, label in sorted(B0_SOURCE_LABELS.items()):
            diagnostic_rows.append(["B0", "station_%s" % st, label,
                                    int((b0_source[selector] == code).sum()),
                                    int(selector.sum())])
    for hz in HORIZONS:
        selector = horizon == hz
        diagnostic_rows.append(["B1", "horizon_%d" % hz, "direct_tau_minus_24",
                                int(b1_direct[selector].sum()),
                                int(selector.sum())])
        diagnostic_rows.append(["B1", "horizon_%d" % hz, "b0_fallback",
                                int((~b1_direct[selector]).sum()),
                                int(selector.sum())])
        equal = int((predictions["B1"][selector]
                     == predictions["B0"][selector]).sum())
        diagnostic_rows.append(["B1", "horizon_%d" % hz,
                                "identical_to_B0_prediction", equal,
                                int(selector.sum())])
    for level, label in enumerate(["station_month_hour", "station_hour",
                                   "station_median"]):
        diagnostic_rows.append(["B2", "all", label,
                                int((b2_level == level).sum()), b2_level.size])
    write_csv(OUT / "baseline_diagnostics.csv",
              ["model", "scope", "category", "count", "total"],
              diagnostic_rows)

    negative_rows = []
    for name in MODELS:
        summary_negative = M.negative_prediction_summary(predictions[name])
        negative_rows.append([name, summary_negative["n_negative"],
                              fmt(summary_negative["pct_negative"]),
                              fmt(summary_negative["min_prediction"]),
                              "false"])
    write_csv(OUT / "negative_predictions.csv",
              ["model", "n_negative", "pct_negative", "min_prediction",
               "clipped"], negative_rows)

    # ---- figures -----------------------------------------------------------
    make_figures(summary, cells, severe, actual, predictions, station,
                 horizon, ranking)

    payload = OrderedDict([
        ("partition", "validation"),
        ("status", "development_open"),
        ("models", MODELS),
        ("summary", OrderedDict((k, OrderedDict((kk, fmt(vv))
                                                for kk, vv in v.items()))
                                for k, v in summary.items())),
        ("ranking_by_primary_metric", ranking),
        ("severe_threshold", M.SEVERE_THRESHOLD),
        ("severe", OrderedDict((k, OrderedDict((kk, fmt(vv) if isinstance(
            vv, float) else vv) for kk, vv in v.items()))
            for k, v in severe.items())),
        ("validation_samples", int(actual.size)),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("predictions_clipped", False),
    ])
    with open(str(PROJECT_ROOT / "artifacts" / "phase3_validation_summary.json"),
              "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("")
    print("PRIMARY DEVELOPMENT RANKING (macro station-horizon MAE)")
    for position, name in enumerate(ranking, 1):
        print("  %d. %-7s %10.6f" % (position, name,
                                     summary[name]["macro_station_horizon_MAE"]))
    return 0


def make_figures(summary, cells, severe, actual, predictions, station,
                 horizon, ranking):
    FIGURES.mkdir(parents=True, exist_ok=True)
    values = [summary[m]["macro_station_horizon_MAE"] for m in MODELS]
    figure = Figure(title="%s: macro station-horizon MAE by model" % TITLE,
                    subtitle=SUBTITLE, xlabel="model",
                    ylabel="MAE (ug/m3)")
    figure.set_xlim(-0.6, len(MODELS) - 0.4)
    figure.set_ylim(0, max(values) * 1.18)
    figure.frame()
    figure.yticks(np.linspace(0, max(values) * 1.15, 6), fmt="%.0f")
    figure.xticks(range(len(MODELS)), MODELS)
    figure.bars(range(len(MODELS)), values, width=0.6)
    for index, value in enumerate(values):
        label = "%.2f" % value
        figure.draw.text((figure.px(index) - 3.4 * len(label),
                          figure.py(value) - 18), label,
                         font=figure.draw.getfont(), fill=(17, 17, 17))
    figure.note("Best by the pre-registered primary metric: %s" % ranking[0])
    figure.save(FIGURES / "validation_mae_by_model.png")

    figure = Figure(title="%s: MAE by forecast horizon" % TITLE,
                    subtitle=SUBTITLE, xlabel="horizon (hours)",
                    ylabel="macro-station MAE (ug/m3)")
    colours = [BLUE, TEAL, ORANGE, RED, (90, 60, 140), (20, 110, 60)]
    series = {}
    top = 0.0
    for name in MODELS:
        series[name] = [M.macro_station_mae(actual[horizon == hz],
                                            predictions[name][horizon == hz],
                                            station[horizon == hz])
                        for hz in HORIZONS]
        top = max(top, max(series[name]))
    figure.set_xlim(-0.4, len(HORIZONS) - 0.6)
    figure.set_ylim(0, top * 1.15)
    figure.frame()
    figure.yticks(np.linspace(0, top * 1.1, 6), fmt="%.0f")
    figure.xticks(range(len(HORIZONS)), ["%d" % h for h in HORIZONS])
    for index, name in enumerate(MODELS):
        figure.line(range(len(HORIZONS)), series[name],
                    colour=colours[index], width=3)
    figure.legend(list(zip(MODELS, colours)))
    figure.save(FIGURES / "validation_mae_by_horizon.png")

    regimes = ["B3_R0", "B3_R1", "B3_R2"]
    values = [summary[m]["macro_station_horizon_MAE"] for m in regimes]
    figure = Figure(title="%s: information-regime gain" % TITLE,
                    subtitle=SUBTITLE + "  |  R0 -> R1 -> R2",
                    xlabel="information regime",
                    ylabel="macro station-horizon MAE (ug/m3)")
    figure.set_xlim(-0.6, len(regimes) - 0.4)
    figure.set_ylim(0, max(values) * 1.2)
    figure.frame()
    figure.yticks(np.linspace(0, max(values) * 1.15, 6), fmt="%.0f")
    figure.xticks(range(len(regimes)),
                  ["R0 met+cal", "R1 +PM2.5", "R2 +co-poll"])
    figure.bars(range(len(regimes)), values, width=0.55)
    for index, value in enumerate(values):
        label = "%.2f" % value
        figure.draw.text((figure.px(index) - 3.4 * len(label),
                          figure.py(value) - 18), label,
                         font=figure.draw.getfont(), fill=(17, 17, 17))
    figure.save(FIGURES / "validation_b3_information_regime_gain.png")

    values = [severe[m]["severe_mae"] for m in MODELS]
    figure = Figure(title="%s: severe-hour MAE (actual > 244.0)" % TITLE,
                    subtitle=SUBTITLE + "  |  threshold frozen from training",
                    xlabel="model", ylabel="severe-hour MAE (ug/m3)")
    figure.set_xlim(-0.6, len(MODELS) - 0.4)
    figure.set_ylim(0, max(values) * 1.18)
    figure.frame()
    figure.yticks(np.linspace(0, max(values) * 1.15, 6), fmt="%.0f")
    figure.xticks(range(len(MODELS)), MODELS)
    figure.bars(range(len(MODELS)), values, width=0.6, colour=RED)
    for index, value in enumerate(values):
        label = "%.1f" % value
        figure.draw.text((figure.px(index) - 3.4 * len(label),
                          figure.py(value) - 18), label,
                         font=figure.draw.getfont(), fill=(17, 17, 17))
    figure.note("severe hours: %d of %d validation samples"
                % (severe[MODELS[0]]["severe_n"], actual.size))
    figure.save(FIGURES / "validation_severe_mae.png")

    matrix = [[cells["B3_R2"][(st, hz)]["mae"] for hz in HORIZONS]
              for st in STATIONS]
    finite = [v for row in matrix for v in row]
    heatmap(FIGURES / "validation_station_horizon_heatmap_B3_R2.png", matrix,
            [s[:9] for s in STATIONS], ["h%d" % h for h in HORIZONS],
            "%s: B3_R2 MAE by station and horizon" % TITLE,
            SUBTITLE + "  |  the 48 cells behind the primary metric",
            vmin=min(finite), vmax=max(finite), fmt="%.1f",
            legend_label="MAE (ug/m3)")

    residual = actual - predictions["B3_R2"]
    limit = float(np.percentile(np.abs(residual), 99))
    edges = np.linspace(-limit, limit, 61)
    counts, _ = np.histogram(residual, bins=edges)
    figure = Figure(title="%s: B3_R2 residual distribution" % TITLE,
                    subtitle=SUBTITLE
                    + "  |  residual = actual - prediction",
                    xlabel="residual (ug/m3), truncated at the 99th pct of |r|",
                    ylabel="samples")
    figure.set_xlim(-limit, limit)
    figure.set_ylim(0, float(counts.max()) * 1.1)
    figure.frame()
    figure.yticks(np.linspace(0, float(counts.max()), 5))
    figure.xticks(np.linspace(-limit, limit, 7),
                  ["%.0f" % v for v in np.linspace(-limit, limit, 7)])
    centres = (edges[:-1] + edges[1:]) / 2.0
    figure.bars(centres, counts.astype(float),
                width=(edges[1] - edges[0]) * 0.9)
    zero = figure.px(0.0)
    figure.draw.line([zero, figure.top, zero, figure.bottom], fill=RED,
                     width=2)
    figure.note("positive residual = under-prediction; mean residual %.3f"
                % float(residual.mean()))
    figure.save(FIGURES / "validation_residual_distribution_B3_R2.png")


if __name__ == "__main__":
    sys.exit(main())
