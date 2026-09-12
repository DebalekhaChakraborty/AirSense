"""Post-test exploratory figures for AirSense V2.

Protocol Phase 9, section 25. Eleven figures, drawn only after every Phase-9
table exists, so no figure can influence a number. Every title carries
POST-TEST EXPLORATORY.

Axes are not truncated to exaggerate differences: bar panels start at zero.
Units are ug/m3 throughout. Residual panels state the convention; severe
panels state the rule and that 244.0 is the training pooled P95, never a
regulatory threshold.

Pillow lives in the system interpreter, so this runs under python3 and reads
only the frozen Phase-9 tables.

Usage:
    python3 scripts/build_phase9_figures.py
"""

import csv
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.render import (                            # noqa: E402
    BLACK, BLUE, GREY, LIGHT, ORANGE, RED, TEAL, Figure, heatmap)

ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES = PROJECT_ROOT / "figures"
MODELS = ["B3_R2", "GRU_R1", "B0"]
COLOURS = {"B3_R2": BLUE, "GRU_R1": TEAL, "B0": ORANGE}
ROLES = {"B3_R2": "primary confirmatory", "GRU_R1": "secondary confirmatory",
         "B0": "reference benchmark"}
HORIZONS = [1, 6, 12, 24]
LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
LABEL = "POST-TEST EXPLORATORY"
SEVERE_NOTE = ("Severe: actual PM2.5 > 244.0 ug/m3 (training pooled P95, "
               "not a regulatory threshold).")
RESIDUAL_NOTE = "Residual = actual - prediction; positive = under-prediction."

TITLES = OrderedDict()


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def table(name):
    with open(str(ARTIFACTS / ("phase9_%s.csv" % name)), newline="",
              encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ceiling(value, step=None):
    if value <= 0:
        return 1.0
    step = step or 10.0 ** (len(str(int(value))) - 1)
    return step * (int(value / step) + 1)


def save(figure, name, title):
    path = FIGURES / ("phase9_%s.png" % name)
    figure.save(path)
    TITLES[path.name] = title
    print("%-46s %s" % (path.name, sha256_of_file(path)[:16]))
    return path


def grouped_bars(name, title, subtitle, xlabel, ylabel, categories, values,
                 note=None, fmt="%.1f"):
    """values[model] -> list aligned to categories."""
    top = ceiling(max(max(v) for v in values.values()))
    figure = Figure(width=1000, height=600, title=title, subtitle=subtitle,
                    xlabel=xlabel, ylabel=ylabel)
    figure.set_xlim(-0.6, len(categories) - 0.4)
    figure.set_ylim(0.0, top)
    figure.yticks([top * i / 5.0 for i in range(6)], "%g")
    figure.frame()
    width = 0.26
    for position, model in enumerate(MODELS):
        offsets = [i + (position - 1) * width for i in range(len(categories))]
        figure.bars(offsets, values[model], width=width * 0.92,
                    colour=COLOURS[model])
    figure.xticks(range(len(categories)), [str(c) for c in categories])
    figure.legend([("%s - %s" % (m, ROLES[m]), COLOURS[m]) for m in MODELS])
    if note:
        figure.note(note)
    return save(figure, name, title + " | " + subtitle)


def main():
    if not (ARTIFACTS / "phase9_error_complementarity.csv").is_file():
        raise SystemExit("refusing to draw: Phase-9 tables are incomplete")
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1 -- development -> locked test
    gap = table("generalization_gap")
    values = {m: [] for m in MODELS}
    categories = ["macro station-horizon MAE\n(development)",
                  "macro station-horizon MAE\n(locked test)",
                  "severe MAE\n(development)", "severe MAE\n(locked test)"]
    for model in MODELS:
        rows = {r["metric"]: r for r in gap if r["model"] == model}
        values[model] = [
            float(rows["macro_station_horizon_MAE"]["development_value"]),
            float(rows["macro_station_horizon_MAE"]["locked_test_value"]),
            float(rows["severe_MAE"]["development_value"]),
            float(rows["severe_MAE"]["locked_test_value"])]
    grouped_bars(
        "generalization_gap",
        "%s - development validation vs locked final test" % LABEL,
        "Frozen development values against the locked test, same models",
        "metric and partition", "MAE (ug/m3)", categories, values,
        note="GRU_R1 development values are its seed-42 row, the artifact the "
             "locked test evaluated. " + SEVERE_NOTE)

    # 2 -- MAE by horizon
    horizon = table("horizon_error_analysis")
    values = {m: [float(r["MAE"]) for h in HORIZONS
                  for r in horizon
                  if r["model"] == m and int(r["horizon_hours"]) == h]
              for m in MODELS}
    grouped_bars("mae_by_horizon",
                 "%s - MAE by forecast horizon" % LABEL,
                 "All 411,012 locked-test samples, split by horizon",
                 "forecast horizon (hours)", "MAE (ug/m3)", HORIZONS, values)

    # 3 -- station heatmap
    station = table("station_error_analysis")
    stations = sorted(set(r["station"] for r in station))
    matrix = [[float(next(r["MAE"] for r in station
                          if r["model"] == m and r["station"] == s))
               for m in MODELS] for s in stations]
    flat = [v for row in matrix for v in row]
    path = FIGURES / "phase9_station_mae_heatmap.png"
    heatmap(path, matrix, stations, MODELS,
            "%s - MAE by station" % LABEL,
            "Locked final test, all horizons pooled. Units ug/m3.",
            vmin=min(flat), vmax=max(flat))
    TITLES[path.name] = "%s - MAE by station" % LABEL
    print("%-46s %s" % (path.name, sha256_of_file(path)[:16]))

    # 4 -- residual ACF
    acf = table("residual_acf_summary")
    figure = Figure(width=1000, height=600,
                    title="%s - residual autocorrelation, h = 24" % LABEL,
                    subtitle="Mean over 12 stations at frozen lags; "
                             "complete pairs only, no gap bridging",
                    xlabel="lag (hours)", ylabel="autocorrelation")
    figure.set_xlim(-0.5, len(LAGS) - 0.5)
    figure.set_ylim(-0.4, 1.0)
    figure.yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "%.1f")
    figure.frame()
    figure.hline(0.0, colour=GREY)
    for model in MODELS:
        series = []
        for lag in LAGS:
            row = next((r for r in acf if r["model"] == model
                        and int(r["horizon_hours"]) == 24
                        and int(r["lag_hours"]) == lag), None)
            series.append(float(row["mean"]) if row and row["mean"] != "NA"
                          else 0.0)
        figure.line(range(len(LAGS)), series, colour=COLOURS[model], width=3)
        figure.points(range(len(LAGS)), series, colour=COLOURS[model],
                      radius=4)
    figure.xticks(range(len(LAGS)), [str(l) for l in LAGS])
    figure.legend([("%s - %s" % (m, ROLES[m]), COLOURS[m]) for m in MODELS])
    figure.note(RESIDUAL_NOTE + " Lags were frozen before computation.")
    save(figure, "residual_acf",
         "%s - residual autocorrelation, h = 24" % LABEL)

    # 5 -- concentration regimes
    regime = table("concentration_regime_analysis")
    names = ["LOW", "ELEVATED", "SEVERE"]
    values = {m: [float(next(r["MAE"] for r in regime if r["model"] == m
                             and r["regime"] == g)) for g in names]
              for m in MODELS}
    grouped_bars(
        "concentration_regimes",
        "%s - MAE by training-frozen concentration stratum" % LABEL,
        "LOW <= 61.0 < ELEVATED <= 244.0 < SEVERE, both boundaries "
        "training-derived",
        "stratum", "MAE (ug/m3)", names, values,
        note="Boundaries are the training median and training pooled P95. No "
             "test quantile was computed.")

    # 6 -- severe MAE by horizon
    severe = table("severe_tail_analysis")
    values = {m: [float(next(r["MAE"] for r in severe if r["model"] == m
                             and int(r["horizon_hours"]) == h))
                  for h in HORIZONS] for m in MODELS}
    grouped_bars("severe_mae_by_horizon",
                 "%s - severe MAE by forecast horizon" % LABEL,
                 "20,336 severe locked-test samples, 5,084 per horizon",
                 "forecast horizon (hours)", "severe MAE (ug/m3)", HORIZONS,
                 values, note=SEVERE_NOTE)

    # 7 -- severe underprediction
    values = {m: [float(next(r["underprediction_pct"] for r in severe
                             if r["model"] == m
                             and int(r["horizon_hours"]) == h))
                  for h in HORIZONS] for m in MODELS}
    figure = Figure(width=1000, height=600,
                    title="%s - severe under-prediction rate" % LABEL,
                    subtitle="Share of severe hours where actual exceeds the "
                             "prediction, by horizon",
                    xlabel="forecast horizon (hours)",
                    ylabel="under-prediction rate (%)")
    figure.set_xlim(-0.6, len(HORIZONS) - 0.4)
    figure.set_ylim(0.0, 100.0)
    figure.yticks([0, 20, 40, 60, 80, 100], "%g")
    figure.frame()
    for position, model in enumerate(MODELS):
        offsets = [i + (position - 1) * 0.26 for i in range(len(HORIZONS))]
        figure.bars(offsets, values[model], width=0.24,
                    colour=COLOURS[model])
    figure.xticks(range(len(HORIZONS)), [str(h) for h in HORIZONS])
    figure.legend([("%s - %s" % (m, ROLES[m]), COLOURS[m]) for m in MODELS])
    figure.note(RESIDUAL_NOTE + " " + SEVERE_NOTE)
    save(figure, "severe_underprediction",
         "%s - severe under-prediction rate" % LABEL)

    # 8 -- severe event peak error
    events = table("severe_event_model_analysis")
    values = {}
    for model in MODELS:
        series = []
        for horizon in HORIZONS:
            errors = [float(r["abs_error_at_peak"]) for r in events
                      if r["model"] == model
                      and int(r["horizon_hours"]) == horizon
                      and r["abs_error_at_peak"] not in ("NA", "")]
            series.append(sum(errors) / len(errors) if errors else 0.0)
        values[model] = series
    grouped_bars("severe_event_peak_error",
                 "%s - absolute error at the event peak hour" % LABEL,
                 "586 strict severe events; mean over events, by horizon",
                 "forecast horizon (hours)",
                 "mean |error| at peak (ug/m3)", HORIZONS, values,
                 note="Events are consecutive hours with " + SEVERE_NOTE.lower())

    # 9 -- seasonal MAE
    seasonal = table("seasonal_error_analysis")
    values = {}
    for model in MODELS:
        series = []
        for season in SEASONS:
            rows = [r for r in seasonal if r["model"] == model
                    and r["season"] == season]
            total = sum(float(r["MAE"]) * int(r["n"]) for r in rows)
            count = sum(int(r["n"]) for r in rows)
            series.append(total / count if count else 0.0)
        values[model] = series
    grouped_bars("seasonal_mae",
                 "%s - MAE by meteorological season" % LABEL,
                 "Target-timestamp seasons, all horizons pooled",
                 "season", "MAE (ug/m3)", SEASONS, values,
                 note="Season boundaries were frozen, not optimised.")

    # 10 -- hour of day
    hourly = table("hour_of_day_error_analysis")
    figure = Figure(width=1000, height=600,
                    title="%s - MAE by hour of day, h = 24" % LABEL,
                    subtitle="Exact target hour, no adaptive binning",
                    xlabel="target hour (local)", ylabel="MAE (ug/m3)")
    series = {}
    for model in MODELS:
        series[model] = [float(next(r["MAE"] for r in hourly
                                    if r["model"] == model
                                    and int(r["horizon_hours"]) == 24
                                    and int(r["target_hour"]) == hour))
                         for hour in range(24)]
    top = ceiling(max(max(v) for v in series.values()))
    figure.set_xlim(-0.5, 23.5)
    figure.set_ylim(0.0, top)
    figure.yticks([top * i / 5.0 for i in range(6)], "%g")
    figure.frame()
    for model in MODELS:
        figure.line(range(24), series[model], colour=COLOURS[model], width=3)
    figure.xticks(range(0, 24, 3), [str(h) for h in range(0, 24, 3)])
    figure.legend([("%s - %s" % (m, ROLES[m]), COLOURS[m]) for m in MODELS])
    save(figure, "hour_of_day_mae",
         "%s - MAE by hour of day, h = 24" % LABEL)

    # 11 -- error complementarity
    comp = table("error_complementarity")
    pairs = [("B3_R2", "GRU_R1"), ("B3_R2", "B0"), ("GRU_R1", "B0")]
    figure = Figure(width=1000, height=600,
                    title="%s - absolute-error correlation between models"
                          % LABEL,
                    subtitle="Pearson correlation of per-sample |error|; "
                             "higher means the models fail together",
                    xlabel="model pair", ylabel="Pearson r of |error|")
    figure.set_xlim(-0.6, len(pairs) - 0.4)
    figure.set_ylim(0.0, 1.0)
    figure.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "%.1f")
    figure.frame()
    for position, stratum in enumerate(("overall", "severe")):
        heights = []
        for first, second in pairs:
            row = next(r for r in comp if r["stratum"] == stratum
                       and r["model_a"] == first and r["model_b"] == second)
            heights.append(float(row["pearson_abs_error"]))
        offsets = [i + (position - 0.5) * 0.3 for i in range(len(pairs))]
        figure.bars(offsets, heights, width=0.28,
                    colour=BLUE if stratum == "overall" else RED)
    figure.xticks(range(len(pairs)), ["%s\nvs %s" % p for p in pairs])
    figure.legend([("overall (411,012 samples)", BLUE),
                   ("severe only (20,336 samples)", RED)])
    figure.note("No ensemble was constructed. " + SEVERE_NOTE)
    save(figure, "model_error_correlation",
         "%s - absolute-error correlation between models" % LABEL)

    manifest = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase9_figure_manifest"),
        ("classification", "POST_TEST_EXPLORATORY"),
        ("drawn_after_all_tables", True),
        ("analysis_freeze_sha256",
         sha256_of_file(ARTIFACTS / "phase9_analysis_freeze.json")),
        ("figure_titles", TITLES),
        ("figure_sha256", OrderedDict(
            (name, sha256_of_file(FIGURES / name)) for name in TITLES)),
        ("exploratory_figures_only", True),
    ])
    with open(str(ARTIFACTS / "phase9_figure_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("\n%d figures, all labelled %s" % (len(TITLES), LABEL))
    return 0


if __name__ == "__main__":
    sys.exit(main())
