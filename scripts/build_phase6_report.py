"""Phase-6 development reporting for the cross-station experiment.

Reads the frozen earlier-phase values and the Phase-6 station-attention
predictions, then writes the paired H4 tables, the severe endpoint, the
attention diagnostics and the figures. Earlier-phase numbers are **reused from
frozen artifacts, never recomputed**.

Everything here is development validation. The locked test is not opened.
Attention weights are descriptive model diagnostics, never causal influence
estimates, and no geographical claim is made anywhere: no coordinates exist in
this study.

Usage:
    python3 scripts/build_phase6_report.py
    python3 scripts/build_phase6_report.py --write-manifest <free_kb>
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

from src.data.preprocessing import STATIONS                         # noqa
from src.evaluation import metrics as M                             # noqa
from src.visualization.render import (                              # noqa
    BLUE, ORANGE, RED, TEAL, Figure, heatmap)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
OUT = PROJECT_ROOT / "results" / "validation"
SPATIAL = PROJECT_ROOT / "results" / "models" / "spatiotemporal"
EDA = PROJECT_ROOT / "results" / "eda_training"
FIGURES = PROJECT_ROOT / "figures"
ARTIFACTS = PROJECT_ROOT / "artifacts"

HORIZONS = [1, 6, 12, 24]
EARLIER = ["B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2", "GRU_R0", "GRU_R1",
           "GRU_R2", "TCN_R0", "TCN_R1", "TCN_R2", "iTransformer_R1",
           "iTransformer_R2"]
PHASE6 = ["SA_R2", "SA_R3"]
ALL_MODELS = EARLIER + PHASE6
PURPLE = (96, 62, 140)
TITLE = "DEVELOPMENT VALIDATION"
SUBTITLE = "2015-03-01 - 2016-02-29  |  NOT FINAL TEST  |  test remains sealed"
DIAGNOSTIC = "DESCRIPTIVE ATTENTION DIAGNOSTIC"
DIAG_SUB = ("learned from training only  |  no coordinates, no distance, no "
            "validation-derived edge  |  not a causal estimate")


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
    if name in ("B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2"):
        return "classical"
    if name.startswith("GRU") or name.startswith("TCN"):
        return "neural"
    return "transformer" if name.startswith("iTransformer") else "spatial"


def load_index():
    stations, horizons = [], []
    for row in read_csv(PROCESSED / "sample_index" / "validation.csv"):
        stations.append(row["station"])
        horizons.append(int(row["horizon_hours"]))
    return np.array(stations), np.array(horizons, dtype=np.int32)


def spearman(left, right):
    """Deterministic rank association, average ranks for ties."""
    def rank(values):
        values = np.asarray(values, dtype=np.float64)
        order = values.argsort(kind="stable")
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
        unique, inverse, counts = np.unique(values, return_inverse=True,
                                            return_counts=True)
        sums = np.zeros(unique.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        return (sums / counts)[inverse]
    a, b = rank(left), rank(right)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator else float("nan")


def legacy_head():
    """Read-only: the frozen V1 branch tip, so later drift is detectable."""
    import subprocess
    result = subprocess.run(["git", "rev-parse", "legacy"],
                            cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return result.stdout.decode().strip() or None


def write_manifest(free_after_kb):
    freeze = json.load(open(str(
        ARTIFACTS / "phase6_architecture_selection_freeze.json"),
        encoding="utf-8"))
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("phase", "phase_6_cross_station_spatiotemporal_modelling"),
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
        ("phase5_manifest_sha256",
         sha256_of_file(ARTIFACTS / "v2_phase5_manifest.json")),
        ("verification_tooling_registry_sha256",
         sha256_of_file(ARTIFACTS / "verification_tooling_registry.json")),
        ("runtime_benchmark_sha256",
         sha256_of_file(ARTIFACTS / "phase6_runtime_benchmark.json")),
        ("group_schema", OrderedDict([
            ("group_id_format", "<partition>|<YYYYMMDDHH>|h<horizon>"),
            ("grouping_source_sha256", sha256_of_file(
                PROJECT_ROOT / "src" / "models" / "spatial_grouping.py")),
            ("station_order", list(STATIONS)),
            ("station_count", len(STATIONS)),
        ])),
        ("dynamic_feature_schema_sha256",
         sha256_of_file(ARTIFACTS / "neural_feature_schema.json")),
        ("internal_split_sha256",
         sha256_of_file(ARTIFACTS / "phase4_internal_split.json")),
        ("architecture_config_sha256", sha256_of_file(
            PROJECT_ROOT / "src" / "models"
            / "station_attention_forecaster.py")),
        ("architecture_selection_freeze_sha256", sha256_of_file(
            ARTIFACTS / "phase6_architecture_selection_freeze.json")),
        ("external_validation_receipt_sha256", sha256_of_file(
            ARTIFACTS / "phase6_external_validation_receipt.json")),
        ("selected", freeze["selected"]),
        ("runtime", freeze["runtime"]),
        ("candidate_table_sha256", OrderedDict(
            (p.name, sha256_of_file(p)) for p in sorted(SPATIAL.glob("*.csv"))
            if p.name.startswith("internal_"))),
        ("model_config_and_weight_sha256", OrderedDict(
            (p.name, sha256_of_file(p)) for p in
            sorted(list(SPATIAL.glob("*_config.json"))
                   + list(SPATIAL.glob("*.safetensors"))))),
        ("prediction_array_sha256", OrderedDict(
            (p.name, sha256_of_file(p))
            for p in sorted(PRED.glob("SA_R*.npy")))),
        ("determinism_prediction_sha256", OrderedDict(
            (p.name, sha256_of_file(p))
            for p in sorted((SPATIAL / "determinism").glob("*.npy")))),
        ("attention_diagnostic_sha256", OrderedDict(
            (p.name, sha256_of_file(p))
            for p in sorted(SPATIAL.glob("attention_*.csv")))),
        ("metric_table_sha256", OrderedDict(
            (p.name, sha256_of_file(p))
            for p in sorted(OUT.glob("phase6_*.csv")))),
        ("figure_sha256", OrderedDict(
            (p.name, sha256_of_file(p))
            for p in sorted(FIGURES.glob("phase6_*.png")))),
        ("compute_report_sha256",
         sha256_of_file(SPATIAL / "compute_report.json")),
        ("document_sha256", OrderedDict(
            (name, sha256_of_file(PROJECT_ROOT / name)) for name in
            ("docs/PHASE_06_SPATIOTEMPORAL_RECORD.md",
             "docs/SPATIOTEMPORAL_RESULTS.md")
            if (PROJECT_ROOT / name).is_file())),
        ("severe_threshold", 244.0),
        ("residual_convention", "residual = actual - prediction"),
        ("context_hours", 48),
        ("validation_samples_per_model", 413148),
        ("internal_candidate_fits", 8),
        ("full_models", 2),
        ("determinism_refits", 2),
        ("paired_design_identical_capacity", True),
        ("R3_uses_external_coordinates", False),
        ("R3_uses_future_observed_variables", False),
        ("validation_derived_graph", False),
        ("predictions_clipped", False),
        ("severe_balancing", False),
        ("seeds_used", [42]),
        ("multi_seed_robustness", "reserved for Phase 7"),
        ("tsfm_loaded", False),
        ("test_predictions", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("v1_legacy_modified", False),
        ("legacy_head", legacy_head()),
        ("free_disk_after_kb", free_after_kb),
        ("note", "Development validation only. No wall-clock timestamp is "
                 "recorded in canonical scientific content."),
    ])
    path = ARTIFACTS / "v2_phase6_manifest.json"
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
            raise SystemExit("%s misaligned" % name)
        predictions[name] = values.astype(np.float64)

    frozen = {r["model"]: r
              for r in read_csv(OUT / "phase5_model_comparison.csv")}
    cells = OrderedDict()
    summary = OrderedDict()
    for name in ALL_MODELS:
        cells[name] = M.cell_metrics(actual, predictions[name], station,
                                     horizon)
        if name in PHASE6:
            summary[name] = OrderedDict([
                ("macro_station_horizon_MAE", M.macro_from_cells(cells[name])),
                ("micro_MAE", M.mae(actual, predictions[name])),
                ("macro_RMSE", M.macro_from_cells(cells[name], "rmse")),
                ("micro_RMSE", M.rmse(actual, predictions[name])),
                ("macro_R2", M.macro_from_cells(cells[name], "r2")),
                ("source", "phase6"),
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
                ("source", row["value_source"]),
            ])
    ranking = sorted(ALL_MODELS,
                     key=lambda m: summary[m]["macro_station_horizon_MAE"])
    write_csv(OUT / "phase6_model_comparison.csv",
              ["model", "family", "macro_station_horizon_MAE", "micro_MAE",
               "macro_RMSE", "micro_RMSE", "macro_R2",
               "rank_by_primary_metric", "value_source"],
              [[name, family(name)]
               + [fmt(summary[name][k]) for k in
                  ("macro_station_horizon_MAE", "micro_MAE", "macro_RMSE",
                   "micro_RMSE", "macro_R2")]
               + [ranking.index(name) + 1, summary[name]["source"]]
               for name in ALL_MODELS])

    r2 = summary["SA_R2"]["macro_station_horizon_MAE"]
    r3 = summary["SA_R3"]["macro_station_horizon_MAE"]
    severe_mask = actual > M.SEVERE_THRESHOLD
    h4_rows = [["overall", int(actual.size), fmt(r2), fmt(r3), fmt(r2 - r3),
                fmt(100.0 * (r2 - r3) / r2)]]
    for hz in HORIZONS:
        selector = horizon == hz
        left = M.macro_station_mae(actual[selector],
                                   predictions["SA_R2"][selector],
                                   station[selector])
        right = M.macro_station_mae(actual[selector],
                                    predictions["SA_R3"][selector],
                                    station[selector])
        h4_rows.append(["horizon_%d" % hz, int(selector.sum()), fmt(left),
                        fmt(right), fmt(left - right),
                        fmt(100.0 * (left - right) / left)])
    left = M.macro_from_cells(M.cell_metrics(
        actual[severe_mask], predictions["SA_R2"][severe_mask],
        station[severe_mask], horizon[severe_mask]))
    right = M.macro_from_cells(M.cell_metrics(
        actual[severe_mask], predictions["SA_R3"][severe_mask],
        station[severe_mask], horizon[severe_mask]))
    h4_rows.append(["severe", int(severe_mask.sum()), fmt(left), fmt(right),
                    fmt(left - right), fmt(100.0 * (left - right) / left)])
    write_csv(OUT / "phase6_h4_summary.csv",
              ["scope", "n", "SA_R2_macro_MAE", "SA_R3_macro_MAE",
               "absolute_MAE_improvement", "relative_MAE_improvement_pct"],
              h4_rows)

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
    write_csv(OUT / "phase6_metrics_by_horizon.csv",
              ["model", "horizon", "n", "micro_MAE", "micro_RMSE", "R2",
               "macro_station_MAE"], rows)

    rows = []
    improved = 0
    for st in STATIONS:
        left = M.macro_from_cells({k: v for k, v in cells["SA_R2"].items()
                                   if k[0] == st})
        right = M.macro_from_cells({k: v for k, v in cells["SA_R3"].items()
                                    if k[0] == st})
        improved += int(right < left)
        rows.append([st, int((station == st).sum()), fmt(left), fmt(right),
                     fmt(left - right), fmt(100.0 * (left - right) / left),
                     "true" if right < left else "false"])
    write_csv(OUT / "phase6_metrics_by_station.csv",
              ["station", "n", "SA_R2_macro_over_horizon_MAE",
               "SA_R3_macro_over_horizon_MAE", "absolute_MAE_improvement",
               "relative_MAE_improvement_pct", "r3_better"], rows)

    rows = []
    wins = losses = ties = 0
    for st in STATIONS:
        for hz in HORIZONS:
            left = cells["SA_R2"][(st, hz)]["mae"]
            right = cells["SA_R3"][(st, hz)]["mae"]
            if right < left:
                wins += 1
                verdict = "R3_better"
            elif right > left:
                losses += 1
                verdict = "R2_better"
            else:
                ties += 1
                verdict = "equal"
            rows.append([st, hz, cells["SA_R2"][(st, hz)]["n"], fmt(left),
                         fmt(right), fmt(left - right), verdict])
    write_csv(OUT / "phase6_metrics_by_station_horizon.csv",
              ["station", "horizon", "n", "SA_R2_MAE", "SA_R3_MAE",
               "absolute_MAE_improvement", "verdict"], rows)

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
    write_csv(OUT / "phase6_severe_metrics.csv",
              ["model", "family", "severe_n", "severe_MAE", "severe_RMSE",
               "severe_mean_residual", "severe_underprediction_pct",
               "severe_macro_station_horizon_MAE", "contributing_cells",
               "total_cells"], rows)

    rows = []
    for name in ALL_MODELS:
        s = M.negative_prediction_summary(predictions[name])
        rows.append([name, family(name), s["n_negative"],
                     fmt(s["pct_negative"]), fmt(s["min_prediction"]),
                     "false"])
    write_csv(OUT / "phase6_negative_predictions.csv",
              ["model", "family", "n_negative", "pct_negative",
               "min_prediction", "clipped"], rows)

    attention = attention_report()
    make_figures(summary, cells, severe, actual, predictions, station,
                 horizon, ranking, attention, h4_rows, wins, losses, ties)

    payload = OrderedDict([
        ("partition", "validation"),
        ("status", "development_open"),
        ("models", ALL_MODELS),
        ("phase6_models", PHASE6),
        ("summary", OrderedDict((k, OrderedDict(
            (kk, fmt(vv) if isinstance(vv, float) else vv)
            for kk, vv in v.items())) for k, v in summary.items())),
        ("ranking_by_primary_metric", ranking),
        ("h4_primary", OrderedDict([
            ("SA_R2_macro_station_horizon_MAE", fmt(r2)),
            ("SA_R3_macro_station_horizon_MAE", fmt(r3)),
            ("absolute_MAE_improvement", fmt(r2 - r3)),
            ("relative_MAE_improvement_pct", fmt(100.0 * (r2 - r3) / r2)),
        ])),
        ("h4_station_horizon_cells", OrderedDict([
            ("r3_better", wins), ("r2_better", losses), ("equal", ties)])),
        ("h4_stations_improved", improved),
        ("severe_threshold", M.SEVERE_THRESHOLD),
        ("severe", OrderedDict((k, OrderedDict(
            (kk, fmt(vv) if isinstance(vv, float) else vv)
            for kk, vv in v.items())) for k, v in severe.items())),
        ("attention", attention),
        ("validation_samples", int(actual.size)),
        ("external_coordinates_used", False),
        ("validation_derived_graph", False),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
        ("predictions_clipped", False),
    ])
    with open(str(ARTIFACTS / "phase6_validation_summary.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("")
    print("PHASE-6 PRIMARY H4 TEST (macro station-horizon MAE)")
    print("  SA_R2 self-only      %10.6f" % r2)
    print("  SA_R3 cross-station  %10.6f" % r3)
    print("  absolute gain        %+10.6f" % (r2 - r3))
    print("  relative gain        %+9.4f%%" % (100.0 * (r2 - r3) / r2))
    print("  station-horizon cells: R3 better %d, R2 better %d, equal %d"
          % (wins, losses, ties))
    print("  stations improved by R3: %d of 12" % improved)
    print("")
    print("PHASE-6 DEVELOPMENT RANKING")
    for position, name in enumerate(ranking, 1):
        print("  %2d. %-16s %10.6f  %s"
              % (position, name, summary[name]["macro_station_horizon_MAE"],
                 family(name)))
    return 0


def attention_report():
    overall = read_csv(SPATIAL / "attention_matrix_overall.csv")
    matrix = np.array([[float(row[s]) for s in STATIONS] for row in overall])
    self_share = float(np.mean(np.diag(matrix)))
    cross_share = float(np.mean(matrix.sum(axis=1) - np.diag(matrix)))

    rows = [["overall", "", fmt(self_share), fmt(cross_share)]]
    by_horizon = read_csv(SPATIAL / "attention_matrix_by_horizon.csv")
    for hz in HORIZONS:
        block = [r for r in by_horizon if int(r["horizon"]) == hz]
        m = np.array([[float(r[s]) for s in STATIONS] for r in block])
        rows.append(["horizon", hz, fmt(float(np.mean(np.diag(m)))),
                     fmt(float(np.mean(m.sum(axis=1) - np.diag(m))))])
    severe_rows = read_csv(SPATIAL / "attention_matrix_severe.csv")
    severe_matrix = np.array([[float(r[s]) for s in STATIONS]
                              for r in severe_rows])
    rows.append(["severe", "", fmt(float(np.mean(np.diag(severe_matrix)))),
                 fmt(float(np.mean(severe_matrix.sum(axis=1)
                                   - np.diag(severe_matrix))))])
    write_csv(OUT / "phase6_attention_summary.csv",
              ["scope", "horizon", "mean_self_attention",
               "mean_total_cross_station_attention"], rows)

    correlation = {}
    for row in read_csv(EDA / "cross_station_pm25_correlation.csv"):
        correlation[(row["station_a"], row["station_b"])] = float(
            row["pearson_r"])
    pairs = []
    for i, target in enumerate(STATIONS):
        for j, source in enumerate(STATIONS):
            if i == j:
                continue
            pairs.append((target, source, matrix[i, j],
                          correlation[(target, source)]))
    rho = spearman([p[2] for p in pairs], [p[3] for p in pairs])
    write_csv(OUT / "phase6_attention_vs_training_correlation.csv",
              ["target_station", "source_station", "mean_attention_weight",
               "training_pearson_r"],
              [[a, b, fmt(w, 8), fmt(c)] for a, b, w, c in pairs])
    return OrderedDict([
        ("mean_self_attention", fmt(self_share)),
        ("mean_total_cross_station_attention", fmt(cross_share)),
        ("directed_pairs", len(pairs)),
        ("spearman_attention_vs_training_correlation", fmt(rho)),
        ("interpretation", "descriptive only; attention weights are not "
                           "causal influence estimates and no geographical "
                           "relationship is claimed"),
    ])


def make_figures(summary, cells, severe, actual, predictions, station,
                 horizon, ranking, attention, h4_rows, wins, losses, ties):
    FIGURES.mkdir(parents=True, exist_ok=True)
    colour_of = {"classical": BLUE, "neural": TEAL, "transformer": ORANGE,
                 "spatial": PURPLE}

    values = [summary[m]["macro_station_horizon_MAE"] for m in PHASE6]
    figure = Figure(width=880,
                    title="%s: cross-station information, paired test" % TITLE,
                    subtitle=SUBTITLE + "  |  identical capacity, identical "
                                        "seed",
                    xlabel="station attention", ylabel="macro station-horizon "
                                                       "MAE (ug/m3)")
    figure.set_xlim(-0.6, 1.4)
    figure.set_ylim(0, max(values) * 1.25)
    figure.frame()
    figure.yticks(np.linspace(0, max(values) * 1.2, 6), fmt="%.0f")
    figure.xticks([0, 1], ["SA_R2 self-only", "SA_R3 all 12 stations"])
    for index, value in enumerate(values):
        figure.bars([index], [value], width=0.45,
                    colour=TEAL if index == 0 else PURPLE)
        label = "%.3f" % value
        figure.draw.text((figure.px(index) - 3.2 * len(label),
                          figure.py(value) - 17), label,
                         font=figure.draw.getfont(), fill=(17, 17, 17))
    gain = values[0] - values[1]
    figure.note("H4 primary: absolute %+.4f ug/m3, relative %+.3f%% "
                "(positive favours cross-station context)"
                % (gain, 100.0 * gain / values[0]))
    figure.save(FIGURES / "phase6_validation_r2_vs_r3.png")

    values = [summary[m]["macro_station_horizon_MAE"] for m in ALL_MODELS]
    figure = Figure(width=1360,
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
    figure.legend([("classical (frozen)", BLUE), ("neural (frozen)", TEAL),
                   ("iTransformer (frozen)", ORANGE),
                   ("station attention (Phase 6)", PURPLE)])
    figure.note("Best by the pre-registered primary metric: %s" % ranking[0])
    figure.save(FIGURES / "phase6_validation_model_comparison.png")

    figure = Figure(width=1100,
                    title="%s: SA_R2 against SA_R3 by horizon" % TITLE,
                    subtitle=SUBTITLE, xlabel="horizon (hours)",
                    ylabel="macro-station MAE (ug/m3)")
    series = {}
    top = 0.0
    for name in PHASE6:
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
    for name, colour in zip(PHASE6, (TEAL, PURPLE)):
        figure.line(range(len(HORIZONS)), series[name], colour=colour,
                    width=3)
        figure.points(range(len(HORIZONS)), series[name], colour=colour,
                      radius=4)
    figure.legend([("SA_R2 self-only", TEAL), ("SA_R3 cross-station", PURPLE)])
    figure.save(FIGURES / "phase6_validation_r2_r3_by_horizon.png")

    gains = []
    for st in STATIONS:
        left = M.macro_from_cells({k: v for k, v in cells["SA_R2"].items()
                                   if k[0] == st})
        right = M.macro_from_cells({k: v for k, v in cells["SA_R3"].items()
                                    if k[0] == st})
        gains.append(left - right)
    limit = max(abs(min(gains)), abs(max(gains))) * 1.4
    figure = Figure(width=1200,
                    title="%s: cross-station gain by station" % TITLE,
                    subtitle=SUBTITLE + "  |  positive = SA_R3 better",
                    xlabel="station",
                    ylabel="macro-over-horizon MAE improvement (ug/m3)")
    figure.set_xlim(-0.6, len(STATIONS) - 0.4)
    figure.set_ylim(-limit, limit)
    figure.frame()
    figure.yticks(np.linspace(-limit, limit, 7), fmt="%.2f")
    figure.xticks(range(len(STATIONS)), [s[:9] for s in STATIONS],
                  rotate=True)
    for index, value in enumerate(gains):
        figure.bars([index], [value], width=0.55,
                    colour=PURPLE if value >= 0 else RED)
    figure.hline(0.0, colour=(17, 17, 17))
    figure.note("%d of 12 stations improved by cross-station context"
                % sum(1 for g in gains if g > 0))
    figure.save(FIGURES / "phase6_validation_r2_r3_by_station.png")

    matrix = [[cells["SA_R2"][(st, hz)]["mae"]
               - cells["SA_R3"][(st, hz)]["mae"]
               for hz in HORIZONS] for st in STATIONS]
    finite = [v for row in matrix for v in row]
    limit = max(abs(min(finite)), abs(max(finite)))
    heatmap(FIGURES / "phase6_validation_station_horizon_gain_heatmap.png",
            matrix, [s[:9] for s in STATIONS], ["h%d" % h for h in HORIZONS],
            "%s: SA_R2 minus SA_R3 MAE" % TITLE,
            SUBTITLE + "  |  positive favours cross-station context",
            vmin=-limit, vmax=limit, fmt="%.2f", low=(150, 40, 40),
            high=(30, 90, 60), legend_label="MAE improvement (ug/m3)")

    order = ["B0", "B3_R2", "GRU_R1", "iTransformer_R1"] + PHASE6
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
    figure.save(FIGURES / "phase6_validation_severe_mae.png")

    rows = read_csv(SPATIAL / "attention_matrix_overall.csv")
    weights = [[float(r[s]) for s in STATIONS] for r in rows]
    heatmap(FIGURES / "phase6_attention_matrix.png", weights,
            [s[:9] for s in STATIONS], [s[:9] for s in STATIONS],
            "%s: mean station attention" % DIAGNOSTIC, DIAG_SUB,
            vmin=0.0, vmax=float(np.max(weights)), fmt="%.3f",
            low=(247, 251, 255), high=(31, 90, 148), cell=54,
            legend_label="mean attention weight (rows sum to 1)")

    by_horizon = read_csv(SPATIAL / "attention_matrix_by_horizon.csv")
    figure = Figure(width=1000,
                    title="%s: self against cross-station attention" %
                          DIAGNOSTIC,
                    subtitle=DIAG_SUB, xlabel="horizon (hours)",
                    ylabel="mean attention weight")
    selves, crosses = [], []
    for hz in HORIZONS:
        block = [r for r in by_horizon if int(r["horizon"]) == hz]
        m = np.array([[float(r[s]) for s in STATIONS] for r in block])
        selves.append(float(np.mean(np.diag(m))))
        crosses.append(float(np.mean(m.sum(axis=1) - np.diag(m))))
    figure.set_xlim(-0.4, len(HORIZONS) - 0.6)
    figure.set_ylim(0, 1.05)
    figure.frame()
    figure.yticks(np.linspace(0, 1.0, 6), fmt="%.2f")
    figure.xticks(range(len(HORIZONS)), ["%d" % h for h in HORIZONS])
    figure.line(range(len(HORIZONS)), selves, colour=PURPLE, width=3)
    figure.points(range(len(HORIZONS)), selves, colour=PURPLE, radius=4)
    figure.line(range(len(HORIZONS)), crosses, colour=ORANGE, width=3)
    figure.points(range(len(HORIZONS)), crosses, colour=ORANGE, radius=4)
    figure.legend([("attention on own station", PURPLE),
                   ("attention on the other 11", ORANGE)])
    figure.note("Spearman against frozen training-only PM2.5 correlation: "
                "%s (descriptive, not causal)"
                % attention["spearman_attention_vs_training_correlation"])
    figure.save(FIGURES / "phase6_attention_by_horizon.png")


if __name__ == "__main__":
    sys.exit(main())
