"""Phase-5 development reporting for the iTransformer track.

Reads the frozen Phase-3 and Phase-4 values and the Phase-5 iTransformer
predictions, then writes the combined development tables, the severe endpoint,
the co-pollutant penalty across architectures and the figures. Earlier-phase
numbers are **reused from frozen artifacts, never recomputed**.

Everything here is development validation. The locked test is not opened.
Chronos-2 was not run: see docs/CHRONOS2_PRETRAINING_AUDIT.md.

Usage:
    python3 scripts/build_phase5_report.py
    python3 scripts/build_phase5_report.py --write-manifest <free_kb>
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
from src.visualization.render import (                             # noqa: E402
    BLUE, ORANGE, RED, TEAL, Figure, heatmap)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
OUT = PROJECT_ROOT / "results" / "validation"
ITRANS_DIR = PROJECT_ROOT / "results" / "models" / "itransformer"
NEURAL_DIR = PROJECT_ROOT / "results" / "models" / "neural"
FIGURES = PROJECT_ROOT / "figures"
ARTIFACTS = PROJECT_ROOT / "artifacts"

HORIZONS = [1, 6, 12, 24]
CLASSICAL = ["B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2"]
NEURAL = ["GRU_R0", "GRU_R1", "GRU_R2", "TCN_R0", "TCN_R1", "TCN_R2"]
ITRANSFORMER = ["iTransformer_R1", "iTransformer_R2"]
ALL_MODELS = CLASSICAL + NEURAL + ITRANSFORMER
PURPLE = (96, 62, 140)
TITLE = "DEVELOPMENT VALIDATION"
SUBTITLE = "2015-03-01 - 2016-02-29  |  NOT FINAL TEST  |  test remains sealed"


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def read_csv(path):
    with open(str(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits=6):
    return "" if value is None else round(float(value), digits)


def family(name):
    if name in CLASSICAL:
        return "classical"
    return "neural" if name in NEURAL else "transformer"


def load_index():
    stations, horizons = [], []
    with open(str(PROCESSED / "sample_index" / "validation.csv"),
              encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stations.append(row["station"])
            horizons.append(int(row["horizon_hours"]))
    return np.array(stations), np.array(horizons, dtype=np.int32)


def write_manifest(free_after_kb):
    freeze = json.load(open(str(
        ARTIFACTS / "phase5_itransformer_selection_freeze.json"),
        encoding="utf-8"))
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("phase", "phase_5_modern_transformer_and_tsfm_evaluation"),
        ("foundation_manifest_sha256",
         sha256_of_file(ARTIFACTS / "v2_foundation_manifest.json")),
        ("phase1_manifest_sha256",
         sha256_of_file(ARTIFACTS / "v2_phase1_manifest.json")),
        ("phase2_manifest_sha256",
         sha256_of_file(ARTIFACTS / "v2_phase2_manifest.json")),
        ("phase3_manifest_sha256",
         sha256_of_file(ARTIFACTS / "v2_phase3_manifest.json")),
        ("phase4_manifest_sha256",
         sha256_of_file(ARTIFACTS / "v2_phase4_manifest.json")),
        ("verification_tooling_registry_sha256",
         sha256_of_file(ARTIFACTS / "verification_tooling_registry.json")),
        ("internal_split_sha256",
         sha256_of_file(ARTIFACTS / "phase4_internal_split.json")),
        ("internal_split_reused_from_phase4", True),
        ("neural_feature_schema_sha256",
         sha256_of_file(ARTIFACTS / "neural_feature_schema.json")),
        ("itransformer_selection_freeze_sha256",
         sha256_of_file(ARTIFACTS
                        / "phase5_itransformer_selection_freeze.json")),
        ("prevalidation_freeze_sha256",
         sha256_of_file(ARTIFACTS / "phase5_prevalidation_freeze.json")),
        ("model_audit_sha256", OrderedDict(
            (name, sha256_of_file(PROJECT_ROOT / "docs" / name)) for name in
            ("PHASE5_MODEL_AUDIT.md", "CHRONOS2_PRETRAINING_AUDIT.md"))),
        ("requirements_sha256", OrderedDict(
            (name, sha256_of_file(PROJECT_ROOT / name)) for name in
            ("requirements-v2-core.txt", "requirements-v2-core-lock.txt",
             "requirements-v2-neural.txt",
             "requirements-v2-neural-lock.txt"))),
        ("runtime", freeze["runtime"]),
        ("selected", freeze["selected"]),
        ("model_source_sha256", OrderedDict(
            (name, sha256_of_file(PROJECT_ROOT / "src" / "models" / name))
            for name in ("itransformer_forecaster.py",
                         "itransformer_grid.py"))),
        ("candidate_table_sha256", OrderedDict(
            (path.name, sha256_of_file(path))
            for path in sorted(ITRANS_DIR.glob("*.csv")))),
        ("model_config_and_weight_sha256", OrderedDict(
            (path.name, sha256_of_file(path)) for path in
            sorted(list(ITRANS_DIR.glob("*_config.json"))
                   + list(ITRANS_DIR.glob("*.safetensors"))))),
        ("prediction_array_sha256", OrderedDict(
            (path.name, sha256_of_file(path)) for path in
            sorted(PRED.glob("iTransformer_*.npy")))),
        ("determinism_prediction_sha256", OrderedDict(
            (path.name, sha256_of_file(path)) for path in
            sorted((ITRANS_DIR / "determinism").glob("*.npy")))),
        ("metric_table_sha256", OrderedDict(
            (path.name, sha256_of_file(path))
            for path in sorted(OUT.glob("phase5_*.csv")))),
        ("figure_sha256", OrderedDict(
            (path.name, sha256_of_file(path))
            for path in sorted(FIGURES.glob("phase5_*.png")))),
        ("document_sha256", OrderedDict(
            (name, sha256_of_file(PROJECT_ROOT / name)) for name in
            ("docs/PHASE_05_MODERN_MODELS_RECORD.md",
             "docs/MODERN_MODEL_RESULTS.md")
            if (PROJECT_ROOT / name).is_file())),
        ("severe_threshold", 244.0),
        ("residual_convention", "residual = actual - prediction"),
        ("context_hours", 48),
        ("itransformer_internal_candidate_fits", 8),
        ("itransformer_full_models", 2),
        ("itransformer_full_determinism_refits", 2),
        ("external_validation_predictions_per_model", 413148),
        ("external_validation_predictions_total", 2 * 413148),
        ("regimes_evaluated", ["R1", "R2"]),
        ("r0_itransformer_trained", False),
        ("r3_implemented", False),
        ("transformer_trained", True),
        ("tsfm_loaded", False),
        ("chronos2_executed", False),
        ("chronos2_checkpoint_downloaded", False),
        ("chronos2_gate", "pretraining-overlap audit classification C"),
        ("timesfm3_excluded", "non-commercial weight licence"),
        ("moirai2_excluded", "cc-by-nc-4.0 weight licence"),
        ("pretrained_weights_used", False),
        ("predictions_clipped", False),
        ("severe_balancing", False),
        ("seeds_used", [42]),
        ("multi_seed_robustness", "reserved for Phase 7"),
        ("test_predictions", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("v1_legacy_modified", False),
        ("free_disk_after_kb", free_after_kb),
        ("note", "Development validation only. No wall-clock timestamp is "
                 "recorded in canonical scientific content."),
    ])
    path = ARTIFACTS / "v2_phase5_manifest.json"
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print("wrote %s" % path.relative_to(PROJECT_ROOT))
    print("sha256 %s" % sha256_of_file(path))


def main():
    if "--write-manifest" in sys.argv:
        write_manifest(int(sys.argv[sys.argv.index("--write-manifest") + 1]))
        return 0

    station, horizon = load_index()
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    predictions = OrderedDict()
    for name in ALL_MODELS:
        values = np.load(str(PRED / ("%s_validation.npy" % name)))
        if values.size != actual.size:
            raise SystemExit("%s misaligned: %d vs %d"
                             % (name, values.size, actual.size))
        predictions[name] = values.astype(np.float64)

    frozen = {r["model"]: r
              for r in read_csv(OUT / "phase4_model_comparison.csv")}
    cells = OrderedDict()
    summary = OrderedDict()
    for name in ALL_MODELS:
        cells[name] = M.cell_metrics(actual, predictions[name], station,
                                     horizon)
        if name in ITRANSFORMER:
            summary[name] = OrderedDict([
                ("macro_station_horizon_MAE", M.macro_from_cells(cells[name])),
                ("micro_MAE", M.mae(actual, predictions[name])),
                ("macro_RMSE", M.macro_from_cells(cells[name], "rmse")),
                ("micro_RMSE", M.rmse(actual, predictions[name])),
                ("macro_R2", M.macro_from_cells(cells[name], "r2")),
                ("source", "phase5"),
            ])
        else:
            row = frozen[name]
            summary[name] = OrderedDict([
                ("macro_station_horizon_MAE",
                 float(row["macro_station_horizon_MAE"])),
                ("micro_MAE", float(row["micro_MAE"])),
                ("macro_RMSE", float(row["macro_RMSE"])),
                ("micro_RMSE", float(row["micro_RMSE"])),
                ("macro_R2", float(row["macro_R2"])),
                ("source", "phase3_frozen" if name in CLASSICAL
                 else "phase4_frozen"),
            ])
    ranking = sorted(ALL_MODELS,
                     key=lambda m: summary[m]["macro_station_horizon_MAE"])
    write_csv(OUT / "phase5_model_comparison.csv",
              ["model", "family", "macro_station_horizon_MAE", "micro_MAE",
               "macro_RMSE", "micro_RMSE", "macro_R2",
               "rank_by_primary_metric", "value_source"],
              [[name, family(name)]
               + [fmt(summary[name][k]) for k in
                  ("macro_station_horizon_MAE", "micro_MAE", "macro_RMSE",
                   "micro_RMSE", "macro_R2")]
               + [ranking.index(name) + 1, summary[name]["source"]]
               for name in ALL_MODELS])

    rows = []
    for name in ITRANSFORMER:
        for (st, hz), cell in cells[name].items():
            rows.append([name, st, hz, cell["n"], fmt(cell["mae"]),
                         fmt(cell["rmse"]), fmt(cell["r2"])])
    write_csv(OUT / "phase5_metrics_by_station_horizon.csv",
              ["model", "station", "horizon", "n", "MAE", "RMSE", "R2"], rows)

    rows = []
    for name in ALL_MODELS:
        for hz in HORIZONS:
            selector = horizon == hz
            rows.append([name, hz, int(selector.sum()),
                         fmt(M.mae(actual[selector],
                                   predictions[name][selector])),
                         fmt(M.rmse(actual[selector],
                                    predictions[name][selector])),
                         fmt(M.r2(actual[selector],
                                  predictions[name][selector])),
                         fmt(M.macro_station_mae(actual[selector],
                                                 predictions[name][selector],
                                                 station[selector]))])
    write_csv(OUT / "phase5_metrics_by_horizon.csv",
              ["model", "horizon", "n", "micro_MAE", "micro_RMSE", "R2",
               "macro_station_MAE"], rows)

    rows = []
    for name in ALL_MODELS:
        for st in STATIONS:
            subset = {k: v for k, v in cells[name].items() if k[0] == st}
            rows.append([name, st, int((station == st).sum()),
                         fmt(M.macro_from_cells(subset)),
                         fmt(M.macro_from_cells(subset, "rmse")),
                         fmt(M.macro_from_cells(subset, "r2"))])
    write_csv(OUT / "phase5_metrics_by_station.csv",
              ["model", "station", "n", "macro_over_horizon_MAE",
               "macro_over_horizon_RMSE", "macro_over_horizon_R2"], rows)

    severe = OrderedDict()
    rows = []
    for name in ALL_MODELS:
        severe[name] = M.severe_metrics(actual, predictions[name], station,
                                        horizon)
        s = severe[name]
        rows.append([name, family(name), s["severe_n"], fmt(s["severe_mae"]),
                     fmt(s["severe_rmse"]), fmt(s["severe_mean_residual"]),
                     fmt(s["severe_underprediction_pct"]),
                     fmt(s["severe_macro_station_horizon_mae"]),
                     s["severe_contributing_cells"], s["severe_total_cells"]])
    write_csv(OUT / "phase5_severe_metrics.csv",
              ["model", "family", "severe_n", "severe_MAE", "severe_RMSE",
               "severe_mean_residual", "severe_underprediction_pct",
               "severe_macro_station_horizon_MAE", "contributing_cells",
               "total_cells"], rows)

    rows = penalty_rows(actual, predictions, station, horizon)
    write_csv(OUT / "phase5_information_regime_gain.csv",
              ["architecture", "scope", "from_model", "to_model",
               "from_macro_MAE", "to_macro_MAE", "absolute_MAE_improvement",
               "relative_MAE_improvement_pct", "value_source"], rows)

    rows = []
    for name in ALL_MODELS:
        s = M.negative_prediction_summary(predictions[name])
        rows.append([name, family(name), s["n_negative"],
                     fmt(s["pct_negative"]), fmt(s["min_prediction"]),
                     "false"])
    write_csv(OUT / "phase5_negative_predictions.csv",
              ["model", "family", "n_negative", "pct_negative",
               "min_prediction", "clipped"], rows)

    make_figures(summary, cells, severe, actual, predictions, station,
                 horizon, ranking)

    payload = OrderedDict([
        ("partition", "validation"),
        ("status", "development_open"),
        ("models", ALL_MODELS),
        ("phase5_models", ITRANSFORMER),
        ("summary", OrderedDict((k, OrderedDict((kk, fmt(vv) if isinstance(
            vv, float) else vv) for kk, vv in v.items()))
            for k, v in summary.items())),
        ("ranking_by_primary_metric", ranking),
        ("severe_threshold", M.SEVERE_THRESHOLD),
        ("severe", OrderedDict((k, OrderedDict((kk, fmt(vv) if isinstance(
            vv, float) else vv) for kk, vv in v.items()))
            for k, v in severe.items())),
        ("validation_samples", int(actual.size)),
        ("chronos2_executed", False),
        ("pretrained_weights_used", False),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("predictions_clipped", False),
    ])
    with open(str(ARTIFACTS / "phase5_validation_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("")
    print("PHASE-5 DEVELOPMENT RANKING (macro station-horizon MAE)")
    for position, name in enumerate(ranking, 1):
        print("  %2d. %-16s %10.6f  %s"
              % (position, name, summary[name]["macro_station_horizon_MAE"],
                 family(name)))
    return 0


def penalty_rows(actual, predictions, station, horizon):
    """R1 -> R2, one row per architecture and scope.

    B3, GRU and TCN values are **copied from the frozen Phase-3 and Phase-4
    tables**, not recomputed. Only the iTransformer rows are new.
    """
    rows = []
    frozen3 = read_csv(OUT / "information_regime_gain.csv")
    for row in frozen3:
        if row["increment"] != "R1_to_R2_copollutant_history":
            continue
        rows.append(["B3", row["scope"], row["from_model"], row["to_model"],
                     fmt(row["from_macro_MAE"]), fmt(row["to_macro_MAE"]),
                     fmt(row["absolute_MAE_improvement"]),
                     fmt(row["relative_MAE_improvement_pct"]),
                     "phase3_frozen"])
    frozen4 = read_csv(OUT / "phase4_information_regime_gain.csv")
    for row in frozen4:
        if not row["from_model"].endswith("_R1"):
            continue
        rows.append([row["architecture"], row["scope"], row["from_model"],
                     row["to_model"], fmt(row["from_macro_MAE"]),
                     fmt(row["to_macro_MAE"]),
                     fmt(row["absolute_MAE_improvement"]),
                     fmt(row["relative_MAE_improvement_pct"]),
                     "phase4_frozen"])

    severe_mask = actual > M.SEVERE_THRESHOLD
    scopes = [("overall", np.ones(actual.size, dtype=bool))]
    scopes += [("horizon_%d" % hz, horizon == hz) for hz in HORIZONS]
    scopes.append(("severe", severe_mask))
    for scope, selector in scopes:
        left = M.macro_from_cells(M.cell_metrics(
            actual[selector], predictions["iTransformer_R1"][selector],
            station[selector], horizon[selector]))
        right = M.macro_from_cells(M.cell_metrics(
            actual[selector], predictions["iTransformer_R2"][selector],
            station[selector], horizon[selector]))
        rows.append(["iTransformer", scope, "iTransformer_R1",
                     "iTransformer_R2", fmt(left), fmt(right),
                     fmt(left - right), fmt(100.0 * (left - right) / left),
                     "phase5"])
    return rows


def make_figures(summary, cells, severe, actual, predictions, station,
                 horizon, ranking):
    FIGURES.mkdir(parents=True, exist_ok=True)
    best_it = min(ITRANSFORMER,
                  key=lambda m: summary[m]["macro_station_horizon_MAE"])
    best_neural = min(NEURAL,
                      key=lambda m: summary[m]["macro_station_horizon_MAE"])
    colour_of = {"classical": BLUE, "neural": TEAL, "transformer": ORANGE}

    values = [summary[m]["macro_station_horizon_MAE"] for m in ALL_MODELS]
    figure = Figure(width=1280,
                    title="%s: macro station-horizon MAE, all models" % TITLE,
                    subtitle=SUBTITLE, xlabel="model", ylabel="MAE (ug/m3)")
    figure.set_xlim(-0.6, len(ALL_MODELS) - 0.4)
    figure.set_ylim(0, max(values) * 1.2)
    figure.frame()
    figure.yticks(np.linspace(0, max(values) * 1.15, 6), fmt="%.0f")
    figure.xticks(range(len(ALL_MODELS)), ALL_MODELS, rotate=True)
    for index, name in enumerate(ALL_MODELS):
        figure.bars([index], [values[index]], width=0.6,
                    colour=colour_of[family(name)])
        label = "%.2f" % values[index]
        figure.draw.text((figure.px(index) - 3.2 * len(label),
                          figure.py(values[index]) - 17), label,
                         font=figure.draw.getfont(), fill=(17, 17, 17))
    figure.legend([("classical (Phase 3, frozen)", BLUE),
                   ("neural (Phase 4, frozen)", TEAL),
                   ("iTransformer (Phase 5)", ORANGE)])
    figure.note("Best by the pre-registered primary metric: %s" % ranking[0])
    figure.save(FIGURES / "phase5_validation_mae_by_model.png")

    figure = Figure(width=1100, title="%s: MAE by forecast horizon" % TITLE,
                    subtitle=SUBTITLE, xlabel="horizon (hours)",
                    ylabel="macro-station MAE (ug/m3)")
    show = ["B0", "B3_R2", best_neural, best_it]
    palette = [BLUE, TEAL, RED, ORANGE]
    top = 0.0
    series = {}
    for name in show:
        series[name] = [M.macro_station_mae(actual[horizon == hz],
                                            predictions[name][horizon == hz],
                                            station[horizon == hz])
                        for hz in HORIZONS]
        top = max(top, max(series[name]))
    figure.set_xlim(-0.4, len(HORIZONS) - 0.6)
    figure.set_ylim(0, top * 1.18)
    figure.frame()
    figure.yticks(np.linspace(0, top * 1.1, 6), fmt="%.0f")
    figure.xticks(range(len(HORIZONS)), ["%d" % h for h in HORIZONS])
    for index, name in enumerate(show):
        figure.line(range(len(HORIZONS)), series[name], colour=palette[index],
                    width=3)
        figure.points(range(len(HORIZONS)), series[name],
                      colour=palette[index], radius=4)
    figure.legend(list(zip(show, palette)))
    figure.save(FIGURES / "phase5_validation_mae_by_horizon.png")

    architectures = ["B3", "GRU", "TCN", "iTransformer"]
    deltas = []
    for architecture in architectures:
        r1 = summary["%s_R1" % architecture]["macro_station_horizon_MAE"]
        r2 = summary["%s_R2" % architecture]["macro_station_horizon_MAE"]
        deltas.append(r1 - r2)
    limit = max(abs(min(deltas)), abs(max(deltas))) * 1.45
    figure = Figure(
        width=1050,
        title="%s: what co-pollutant history does, by architecture" % TITLE,
        subtitle=SUBTITLE + "  |  positive = R2 helps, negative = R2 hurts",
        xlabel="architecture",
        ylabel="macro station-horizon MAE improvement R1 -> R2 (ug/m3)")
    figure.set_xlim(-0.6, len(architectures) - 0.4)
    figure.set_ylim(-limit, limit)
    figure.frame()
    figure.yticks(np.linspace(-limit, limit, 7), fmt="%.2f")
    figure.xticks(range(len(architectures)), architectures)
    for index, value in enumerate(deltas):
        figure.bars([index], [value], width=0.5,
                    colour=TEAL if value >= 0 else RED)
        label = "%+.2f" % value
        figure.draw.text((figure.px(index) - 3.2 * len(label),
                          figure.py(value) + (-17 if value >= 0 else 6)),
                         label, font=figure.draw.getfont(), fill=(17, 17, 17))
    figure.hline(0.0, colour=(17, 17, 17))
    figure.note("Attention across variates is the architectural test of "
                "whether the Phase-4 co-pollutant penalty was a property of "
                "the data or of the model.")
    figure.save(FIGURES / "phase5_copollutant_penalty_by_architecture.png")

    order = ["B0", "B3_R2", best_neural, "GRU_R1"] + ITRANSFORMER
    order = list(OrderedDict((name, None) for name in order))
    values = [severe[m]["severe_mae"] for m in order]
    figure = Figure(width=1150,
                    title="%s: severe-hour MAE (actual > 244.0)" % TITLE,
                    subtitle=SUBTITLE + "  |  threshold frozen from training",
                    xlabel="model", ylabel="severe-hour MAE (ug/m3)")
    figure.set_xlim(-0.6, len(order) - 0.4)
    figure.set_ylim(0, max(values) * 1.2)
    figure.frame()
    figure.yticks(np.linspace(0, max(values) * 1.15, 6), fmt="%.0f")
    figure.xticks(range(len(order)), order, rotate=True)
    for index, name in enumerate(order):
        figure.bars([index], [values[index]], width=0.6,
                    colour=colour_of[family(name)])
        label = "%.1f" % values[index]
        figure.draw.text((figure.px(index) - 3.2 * len(label),
                          figure.py(values[index]) - 17), label,
                         font=figure.draw.getfont(), fill=(17, 17, 17))
    figure.hline(severe["B0"]["severe_mae"], colour=RED)
    figure.note("red line: B0 persistence severe MAE %.2f"
                % severe["B0"]["severe_mae"])
    figure.save(FIGURES / "phase5_severe_mae_comparison.png")

    with open(str(ITRANS_DIR / "training_curves" / "full_refit_curves.json"),
              encoding="utf-8") as handle:
        curves = json.load(handle)
    with open(str(NEURAL_DIR / "training_curves" / "full_refit_curves.json"),
              encoding="utf-8") as handle:
        curves.update({k: v for k, v in json.load(handle).items()
                       if k in (best_neural, "GRU_R1")})
    names = sorted(curves)
    palette = [ORANGE, PURPLE, TEAL, BLUE, RED, (20, 110, 60)]
    figure = Figure(width=1050, title="%s: full-refit training curves" % TITLE,
                    subtitle="training L1 loss per epoch, complete frozen "
                             "training universe, 8 epochs fixed in advance",
                    xlabel="epoch", ylabel="training L1 (ug/m3)")
    length = len(curves[names[0]])
    top = max(max(v) for v in curves.values())
    figure.set_xlim(1, length)
    figure.set_ylim(0, top * 1.1)
    figure.frame()
    figure.yticks(np.linspace(0, top, 6), fmt="%.0f")
    figure.xticks(range(1, length + 1),
                  ["%d" % i for i in range(1, length + 1)])
    for index, name in enumerate(names):
        figure.line(range(1, len(curves[name]) + 1), curves[name],
                    colour=palette[index % len(palette)], width=3)
    figure.legend([(n, palette[i % len(palette)])
                   for i, n in enumerate(names)])
    figure.save(FIGURES / "phase5_training_curves.png")

    residual = actual - predictions[best_it]
    limit = float(np.percentile(np.abs(residual), 99))
    edges = np.linspace(-limit, limit, 61)
    counts, _ = np.histogram(residual, bins=edges)
    figure = Figure(width=1000,
                    title="%s: %s residual distribution" % (TITLE, best_it),
                    subtitle=SUBTITLE + "  |  residual = actual - prediction",
                    xlabel="residual (ug/m3), truncated at the 99th "
                           "percentile of |r|",
                    ylabel="samples")
    figure.set_xlim(-limit, limit)
    figure.set_ylim(0, float(counts.max()) * 1.1)
    figure.frame()
    figure.yticks(np.linspace(0, float(counts.max()), 5))
    figure.xticks(np.linspace(-limit, limit, 7),
                  ["%.0f" % v for v in np.linspace(-limit, limit, 7)])
    centres = (edges[:-1] + edges[1:]) / 2.0
    figure.bars(centres, counts.astype(float),
                width=(edges[1] - edges[0]) * 0.9, colour=ORANGE)
    zero = figure.px(0.0)
    figure.draw.line([zero, figure.top, zero, figure.bottom], fill=RED,
                     width=2)
    figure.note("positive residual = under-prediction; mean residual %.3f"
                % float(residual.mean()))
    figure.save(FIGURES / "phase5_residual_distribution_best_itransformer.png")

    matrix = [[cells[best_it][(st, hz)]["mae"]
               - cells[best_neural][(st, hz)]["mae"] for hz in HORIZONS]
              for st in STATIONS]
    finite = [v for row in matrix for v in row]
    limit = max(abs(min(finite)), abs(max(finite)))
    heatmap(FIGURES / "phase5_station_horizon_heatmap_itransformer.png",
            matrix, [s[:9] for s in STATIONS], ["h%d" % h for h in HORIZONS],
            "%s: %s minus %s MAE" % (TITLE, best_it, best_neural),
            SUBTITLE + "  |  negative favours the iTransformer",
            vmin=-limit, vmax=limit, fmt="%.2f",
            low=(30, 90, 60), high=(150, 40, 40),
            legend_label="MAE difference (ug/m3)")


if __name__ == "__main__":
    sys.exit(main())
