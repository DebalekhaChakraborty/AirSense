"""Training-only EDA for AirSense V2.

Protocol Phase 1.

Reads `data/raw/` and writes deterministic tables to `results/eda_training/`,
figures to `figures/`, and a machine-readable summary to
`artifacts/training_eda.json`.

**Partition discipline.** Target *values* are read for the training partition
only. The validation and locked-test partitions are loaded in `presence` mode,
where the parser discards numeric values and returns observed/missing booleans
only, so no validation or test target statistic can be produced from this
script even by mistake. Counting eligibility is the only thing done with the
later partitions.

Fits nothing, predicts nothing, imputes nothing.

Usage:
    python3 scripts/run_training_eda.py
"""

import csv
import io
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import missingness as miss                        # noqa: E402
from src.data.eda import (                                      # noqa: E402
    CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, PARTITIONS, PERCENTILES,
    QUANTILE_METHOD, TARGET, autocorrelation_at_lag, describe, load_partition,
    pearson, quantile, station_paths, timestamps)
from src.visualization.render import (                          # noqa: E402
    BLUE, ORANGE, RED, TEAL, Figure, heatmap)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
RESULTS = PROJECT_ROOT / "results" / "eda_training"
FIGURES = PROJECT_ROOT / "figures"
SUMMARY_JSON = PROJECT_ROOT / "artifacts" / "training_eda.json"

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
HORIZONS = [1, 6, 12, 24]
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
TITLE_SUFFIX = "TRAINING PERIOD ONLY (2013-03-01 - 2015-02-28)"


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
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, digits)
    return value


def main():
    paths = station_paths(RAW_DIR)
    if len(paths) != 12:
        print("expected 12 station files, found %d" % len(paths),
              file=sys.stderr)
        return 2

    # ---- load -----------------------------------------------------------
    train = OrderedDict()
    for path in paths:
        record = load_partition(path, "train", mode="values")
        train[record["station"]] = record
    stations = sorted(train)

    presence = OrderedDict()
    for partition in ("validation", "test"):
        presence[partition] = OrderedDict()
        for path in paths:
            record = load_partition(path, partition, mode="presence")
            presence[partition][record["station"]] = record

    stamps = timestamps("train")
    years = np.array([s.year for s in stamps])
    months = np.array([s.month for s in stamps])
    hours = np.array([s.hour for s in stamps])
    n_hours = len(stamps)

    # ---- pooled and per-station target summary --------------------------
    pooled_values = np.concatenate(
        [train[s]["numeric"][TARGET][train[s]["observed"][TARGET]]
         for s in stations])
    pooled_summary = describe(pooled_values)
    pooled_total = n_hours * len(stations)
    pooled_missing = pooled_total - int(pooled_summary["count"])

    summary_fields = (["count", "mean", "std", "min"]
                      + ["p%02d" % p for p in PERCENTILES]
                      + ["max", "median", "skewness", "excess_kurtosis"])

    write_csv(RESULTS / "pm25_summary.csv",
              ["scope", "hours", "observed", "missing", "missing_pct"]
              + summary_fields,
              [["pooled_training", pooled_total, pooled_summary["count"],
                pooled_missing,
                round(100.0 * pooled_missing / pooled_total, 6)]
               + [fmt(pooled_summary[f]) for f in summary_fields]])

    by_station_summary = OrderedDict()
    rows = []
    for station in stations:
        observed = train[station]["observed"][TARGET]
        values = train[station]["numeric"][TARGET][observed]
        station_summary = describe(values)
        by_station_summary[station] = station_summary
        missing = n_hours - int(station_summary["count"])
        rows.append([station, n_hours, station_summary["count"], missing,
                     round(100.0 * missing / n_hours, 6)]
                    + [fmt(station_summary[f]) for f in summary_fields])
    write_csv(RESULTS / "pm25_summary_by_station.csv",
              ["station", "hours", "observed", "missing", "missing_pct"]
              + summary_fields, rows)

    # ---- severe thresholds ----------------------------------------------
    pooled_p95 = quantile(pooled_values, 95)
    threshold_rows = [["pooled_training", "ALL", int(pooled_values.size),
                       round(pooled_p95, 6), "p95", QUANTILE_METHOD]]
    station_thresholds = OrderedDict()
    for station in stations:
        observed = train[station]["observed"][TARGET]
        values = train[station]["numeric"][TARGET][observed]
        value = quantile(values, 95)
        station_thresholds[station] = value
        threshold_rows.append(["station_training", station, int(values.size),
                               round(value, 6), "p95", QUANTILE_METHOD])
    write_csv(RESULTS / "severe_thresholds.csv",
              ["scope", "station", "n_observed", "p95_ug_m3", "quantile",
               "method"], threshold_rows)

    # ---- missing-run structure ------------------------------------------
    variables = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    run_rows = []
    run_summaries = OrderedDict()
    for variable in variables:
        for station in stations:
            summary = miss.run_summary(train[station]["observed"][variable])
            run_summaries["%s|%s" % (variable, station)] = summary
            run_rows.append([variable, station]
                            + [fmt(summary[k]) for k in summary])
        pooled_runs = []
        pooled_missing_count = 0
        for station in stations:
            observed = train[station]["observed"][variable]
            pooled_runs.extend(miss.missing_runs(observed))
            pooled_missing_count += int((~observed).sum())
        pooled = OrderedDict([
            ("hours", n_hours * len(stations)),
            ("missing", pooled_missing_count),
            ("missing_pct", round(100.0 * pooled_missing_count
                                  / (n_hours * len(stations)), 6)),
            ("n_runs", len(pooled_runs)),
            ("longest_run_h", max(pooled_runs) if pooled_runs else 0),
            ("median_run_h", miss.percentile_of_runs(pooled_runs, 50)),
            ("p90_run_h", miss.percentile_of_runs(pooled_runs, 90)),
            ("isolated_1h_gaps", sum(1 for r in pooled_runs if r == 1)),
        ])
        for label, limit in miss.RUN_BUCKETS:
            pooled[label] = sum(1 for r in pooled_runs if r <= limit)
        pooled["runs_gt_24h"] = sum(1 for r in pooled_runs if r > 24)
        pooled["missing_hours_in_runs_gt_24h"] = sum(
            r for r in pooled_runs if r > 24)
        run_summaries["%s|POOLED" % variable] = pooled
        run_rows.append([variable, "POOLED"] + [fmt(pooled[k])
                                                for k in pooled])
    sample_keys = list(run_summaries[list(run_summaries)[0]].keys())
    write_csv(RESULTS / "missing_run_summary.csv",
              ["variable", "station"] + sample_keys, run_rows)

    # ---- target missingness in time --------------------------------------
    temporal_rows = []
    for dimension, keys in (("year", years), ("month", months),
                            ("hour", hours)):
        pooled_observed = np.concatenate(
            [train[s]["observed"][TARGET] for s in stations])
        pooled_keys = np.concatenate([keys for _ in stations])
        for key, entry in miss.group_counts(pooled_observed,
                                            pooled_keys).items():
            temporal_rows.append([dimension, key, "ALL", entry["hours"],
                                  entry["missing"], entry["missing_pct"]])
    for station in stations:
        for key, entry in miss.group_counts(
                train[station]["observed"][TARGET], years).items():
            temporal_rows.append(["year", key, station, entry["hours"],
                                  entry["missing"], entry["missing_pct"]])
    write_csv(RESULTS / "target_missingness_temporal.csv",
              ["dimension", "key", "station", "hours", "missing",
               "missing_pct"], temporal_rows)

    # ---- simultaneous cross-station missingness ---------------------------
    simultaneous_rows = []
    simultaneous = OrderedDict()
    for variable in POLLUTANTS:
        _, histogram = miss.simultaneous_missing(
            [train[s]["observed"][variable] for s in stations])
        simultaneous[variable] = histogram
        for k, count in histogram.items():
            simultaneous_rows.append([variable, k, count,
                                      round(100.0 * count / n_hours, 6)])
    write_csv(RESULTS / "simultaneous_station_missingness.csv",
              ["variable", "stations_missing", "timestamps", "pct_of_hours"],
              simultaneous_rows)

    # ---- wind direction ---------------------------------------------------
    categories = sorted({value for station in stations
                         for value in train[station]["categorical"]["wd"]
                         if value is not None})
    wd_rows = []
    for station in stations:
        column = train[station]["categorical"]["wd"]
        counts = OrderedDict((c, 0) for c in categories)
        for value in column:
            if value is not None:
                counts[value] += 1
        missing_count = sum(1 for v in column if v is None)
        for category in categories:
            wd_rows.append([station, category, counts[category],
                            round(100.0 * counts[category] / n_hours, 6)])
        wd_rows.append([station, "MISSING", missing_count,
                        round(100.0 * missing_count / n_hours, 6)])
    write_csv(RESULTS / "wind_direction_category_counts.csv",
              ["station", "category", "count", "pct_of_hours"], wd_rows)

    # ---- autocorrelation --------------------------------------------------
    autocorr_rows = []
    autocorr = OrderedDict()
    for lag in LAGS:
        pooled_numerator = []
        for station in stations:
            values = train[station]["numeric"][TARGET]
            observed = train[station]["observed"][TARGET]
            rho, n = autocorrelation_at_lag(values, observed, lag)
            autocorr_rows.append(["station", station, lag, fmt(rho), n])
            if rho is not None:
                pooled_numerator.append((rho, n))
        left = np.concatenate([train[s]["numeric"][TARGET][:-lag]
                               for s in stations])
        right = np.concatenate([train[s]["numeric"][TARGET][lag:]
                                for s in stations])
        valid = np.concatenate([train[s]["observed"][TARGET][:-lag]
                                & train[s]["observed"][TARGET][lag:]
                                for s in stations])
        rho, n = pearson(left, right, valid)
        autocorr[lag] = {"pooled_pairs_r": rho, "pairs": n}
        autocorr_rows.append(["pooled_pairs", "ALL", lag, fmt(rho), n])
    write_csv(RESULTS / "pm25_autocorrelation.csv",
              ["scope", "station", "lag_hours", "pearson_r", "valid_pairs"],
              autocorr_rows)

    # ---- cross-station correlation ----------------------------------------
    matrix = [[None] * len(stations) for _ in stations]
    cross_rows = []
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            valid = (train[a]["observed"][TARGET]
                     & train[b]["observed"][TARGET])
            rho, n = pearson(train[a]["numeric"][TARGET],
                             train[b]["numeric"][TARGET], valid)
            matrix[i][j] = rho
            cross_rows.append([a, b, fmt(rho), n])
    write_csv(RESULTS / "cross_station_pm25_correlation.csv",
              ["station_a", "station_b", "pearson_r", "valid_pairs"],
              cross_rows)

    # ---- eligibility accounting (counts only) -----------------------------
    dataset_start_index = {"train": 0, "validation": 17520, "test": 26304}
    eligibility_rows = []
    eligibility = OrderedDict()
    for partition in ("train", "validation", "test"):
        length = (n_hours if partition == "train"
                  else presence[partition][stations[0]]["length"])
        offset = dataset_start_index[partition]
        for horizon in HORIZONS:
            observed_total = 0
            missing_target = 0
            no_origin = 0
            for station in stations:
                observed = (train[station]["observed"][TARGET]
                            if partition == "train"
                            else presence[partition][station]
                            ["observed"][TARGET])
                for index in range(length):
                    if offset + index - horizon < 0:
                        no_origin += 1
                        continue
                    if observed[index]:
                        observed_total += 1
                    else:
                        missing_target += 1
            candidates = length * len(stations)
            eligibility_rows.append([partition, horizon, candidates,
                                     no_origin, missing_target,
                                     observed_total])
            eligibility["%s|%d" % (partition, horizon)] = {
                "candidates": candidates, "origin_before_dataset": no_origin,
                "target_missing": missing_target, "eligible": observed_total}
    write_csv(RESULTS / "target_eligibility_by_partition_horizon.csv",
              ["partition", "horizon_h", "candidate_targets",
               "excluded_origin_before_dataset", "excluded_target_missing",
               "eligible_targets"], eligibility_rows)

    # ---- figures ----------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)
    render_figures(stations, train, pooled_values, pooled_summary,
                   by_station_summary, pooled_p95, station_thresholds,
                   months, hours, run_summaries, autocorr, matrix,
                   variables, n_hours)

    # ---- machine-readable summary -----------------------------------------
    payload = OrderedDict([
        ("partition", "train"),
        ("training_start", PARTITIONS["train"][0].isoformat()),
        ("training_end", PARTITIONS["train"][1].isoformat()),
        ("stations", stations),
        ("hours_per_station", n_hours),
        ("training_rows", pooled_total),
        ("target_observed", int(pooled_summary["count"])),
        ("target_missing", pooled_missing),
        ("quantile_method", QUANTILE_METHOD),
        ("pooled_pm25_summary", OrderedDict(
            (k, fmt(v)) for k, v in pooled_summary.items())),
        ("pooled_training_p95", round(pooled_p95, 6)),
        ("station_training_p95", OrderedDict(
            (s, round(v, 6)) for s, v in station_thresholds.items())),
        ("autocorrelation_pooled_pairs", OrderedDict(
            (str(lag), OrderedDict([("r", fmt(autocorr[lag]["pooled_pairs_r"])),
                                    ("pairs", autocorr[lag]["pairs"])]))
            for lag in LAGS)),
        ("wind_direction_categories", categories),
        ("eligibility_counts", eligibility),
        ("validation_target_analysis", False),
        ("test_target_analysis", False),
        ("model_runs", 0),
        ("predictions_generated", 0),
    ])
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(SUMMARY_JSON), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("")
    print("training rows        : %d (%d stations x %d hours)"
          % (pooled_total, len(stations), n_hours))
    print("PM2.5 observed       : %d" % pooled_summary["count"])
    print("PM2.5 missing        : %d (%.4f%%)"
          % (pooled_missing, 100.0 * pooled_missing / pooled_total))
    print("pooled training P95  : %.6f ug/m3" % pooled_p95)
    print("lag-1 autocorrelation: %.6f" % autocorr[1]["pooled_pairs_r"])
    print("")
    for path in sorted(RESULTS.glob("*.csv")) + sorted(FIGURES.glob("*.png")):
        print("  %s" % path.relative_to(PROJECT_ROOT))
    return 0


def render_figures(stations, train, pooled_values, pooled_summary,
                   by_station, pooled_p95, station_p95, months, hours,
                   run_summaries, autocorr, matrix, variables, n_hours):
    short = [s[:9] for s in stations]

    # 1. distribution
    upper = pooled_summary["p99"]
    edges = np.linspace(0.0, upper, 61)
    counts, _ = np.histogram(pooled_values, bins=edges)
    figure = Figure(title="Training PM2.5 distribution",
                    subtitle=TITLE_SUFFIX + "  |  observed hours only",
                    xlabel="PM2.5 (ug/m3), truncated at P99",
                    ylabel="hours")
    figure.set_xlim(0, upper)
    figure.set_ylim(0, float(counts.max()) * 1.08)
    figure.frame()
    figure.yticks(np.linspace(0, float(counts.max()), 5))
    figure.xticks(np.linspace(0, upper, 7),
                  ["%d" % v for v in np.linspace(0, upper, 7)])
    centres = (edges[:-1] + edges[1:]) / 2.0
    figure.bars(centres, counts.astype(float),
                width=(edges[1] - edges[0]) * 0.9)
    figure.hline(0, colour=(255, 255, 255))
    for value, colour, label in ((pooled_summary["median"], ORANGE, "median"),
                                 (pooled_p95, RED, "P95 severe threshold")):
        x = figure.px(value)
        figure.draw.line([x, figure.top, x, figure.bottom], fill=colour,
                         width=2)
    figure.legend([("median %.1f" % pooled_summary["median"], ORANGE),
                   ("P95 threshold %.1f" % pooled_p95, RED)])
    figure.note("Right tail beyond P99 (%.1f) not shown; max is %.1f"
                % (pooled_summary["p99"], pooled_summary["max"]))
    figure.save(FIGURES / "training_pm25_distribution.png")

    # 2. by station
    figure = Figure(title="Training PM2.5 by station",
                    subtitle=TITLE_SUFFIX,
                    xlabel="station", ylabel="PM2.5 (ug/m3)")
    top = max(by_station[s]["p95"] for s in stations) * 1.12
    figure.set_xlim(-0.6, len(stations) - 0.4)
    figure.set_ylim(0, top)
    figure.frame()
    figure.yticks(np.linspace(0, top, 6), fmt="%.0f")
    figure.xticks(range(len(stations)), short, rotate=True)
    figure.bars(range(len(stations)),
                [by_station[s]["median"] for s in stations], width=0.62)
    figure.points(range(len(stations)), [by_station[s]["p75"] for s in
                                         stations], colour=TEAL, radius=4)
    figure.points(range(len(stations)), [station_p95[s] for s in stations],
                  colour=RED, radius=4)
    figure.hline(pooled_p95, colour=RED)
    figure.legend([("median (bar)", BLUE), ("P75", TEAL),
                   ("station P95", RED),
                   ("pooled P95 %.1f" % pooled_p95, RED)])
    figure.save(FIGURES / "training_pm25_by_station.png")

    # 3. monthly
    pooled_target = np.concatenate([train[s]["numeric"]["PM2.5"]
                                    for s in stations])
    pooled_observed = np.concatenate([train[s]["observed"]["PM2.5"]
                                      for s in stations])
    pooled_months = np.concatenate([months for _ in stations])
    pooled_hours = np.concatenate([hours for _ in stations])

    def by_key(keys, key_values):
        medians, means = [], []
        for key in key_values:
            selector = (keys == key) & pooled_observed
            sample = pooled_target[selector]
            sample = sample[np.isfinite(sample)]
            medians.append(float(np.median(sample)) if sample.size else 0.0)
            means.append(float(sample.mean()) if sample.size else 0.0)
        return medians, means

    month_values = list(range(1, 13))
    month_medians, month_means = by_key(pooled_months, month_values)
    figure = Figure(title="Training PM2.5 monthly pattern",
                    subtitle=TITLE_SUFFIX + "  |  pooled across 12 stations",
                    xlabel="calendar month", ylabel="PM2.5 (ug/m3)")
    figure.set_xlim(0.4, 12.6)
    figure.set_ylim(0, max(month_means) * 1.15)
    figure.frame()
    figure.yticks(np.linspace(0, max(month_means) * 1.1, 6), fmt="%.0f")
    figure.xticks(month_values, ["%d" % m for m in month_values])
    figure.bars(month_values, month_medians, width=0.66)
    figure.line(month_values, month_means, colour=ORANGE, width=3)
    figure.legend([("median (bar)", BLUE), ("mean (line)", ORANGE)])
    figure.save(FIGURES / "training_monthly_pm25_pattern.png")

    # 4. hourly
    hour_values = list(range(24))
    hour_medians, hour_means = by_key(pooled_hours, hour_values)
    figure = Figure(title="Training PM2.5 hour-of-day pattern",
                    subtitle=TITLE_SUFFIX + "  |  pooled across 12 stations",
                    xlabel="hour of day", ylabel="PM2.5 (ug/m3)")
    figure.set_xlim(-0.6, 23.6)
    figure.set_ylim(0, max(hour_means) * 1.15)
    figure.frame()
    figure.yticks(np.linspace(0, max(hour_means) * 1.1, 6), fmt="%.0f")
    figure.xticks(hour_values, ["%d" % h for h in hour_values])
    figure.bars(hour_values, hour_medians, width=0.66)
    figure.line(hour_values, hour_means, colour=ORANGE, width=3)
    figure.legend([("median (bar)", BLUE), ("mean (line)", ORANGE)])
    figure.save(FIGURES / "training_hourly_pm25_pattern.png")

    # 5. missingness heatmap
    grid = [[run_summaries["%s|%s" % (variable, station)]["missing_pct"]
             for station in stations] for variable in variables]
    heatmap(FIGURES / "training_missingness_by_variable_station.png", grid,
            variables, short,
            "Training missingness by variable and station",
            TITLE_SUFFIX + "  |  percent of hours missing",
            vmin=0.0, vmax=max(max(row) for row in grid), fmt="%.1f",
            legend_label="% of training hours missing")

    # 6. run lengths
    pooled_runs = []
    for station in stations:
        pooled_runs.extend(miss.missing_runs(train[station]["observed"]
                                             ["PM2.5"]))
    buckets = [("1", lambda r: r == 1), ("2-3", lambda r: 2 <= r <= 3),
               ("4-6", lambda r: 4 <= r <= 6), ("7-12", lambda r: 7 <= r <= 12),
               ("13-24", lambda r: 13 <= r <= 24),
               ("25-48", lambda r: 25 <= r <= 48),
               (">48", lambda r: r > 48)]
    counts = [sum(1 for r in pooled_runs if test(r)) for _, test in buckets]
    figure = Figure(title="Training PM2.5 missing-run lengths",
                    subtitle=TITLE_SUFFIX + "  |  pooled across 12 stations",
                    xlabel="consecutive missing hours",
                    ylabel="number of gaps")
    figure.set_xlim(-0.6, len(buckets) - 0.4)
    figure.set_ylim(0, max(counts) * 1.14)
    figure.frame()
    figure.yticks(np.linspace(0, max(counts), 6), fmt="%.0f")
    figure.xticks(range(len(buckets)), [b for b, _ in buckets])
    figure.bars(range(len(buckets)), [float(c) for c in counts], width=0.62)
    for index, count in enumerate(counts):
        label = "%d" % count
        figure.draw.text((figure.px(index) - 3.4 * len(label),
                          figure.py(count) - 18), label,
                         font=figure.draw.getfont(), fill=(17, 17, 17))
    figure.note("Total gaps %d, longest %d h. Short gaps dominate."
                % (len(pooled_runs), max(pooled_runs) if pooled_runs else 0))
    figure.save(FIGURES / "training_pm25_missing_run_lengths.png")

    # 7. autocorrelation
    lags = sorted(autocorr)
    values = [autocorr[lag]["pooled_pairs_r"] for lag in lags]
    figure = Figure(title="Training PM2.5 autocorrelation at exact lags",
                    subtitle=TITLE_SUFFIX
                    + "  |  observed pairs only, no imputation",
                    xlabel="lag (hours)", ylabel="Pearson r")
    figure.set_xlim(-0.6, len(lags) - 0.4)
    figure.set_ylim(0, 1.0)
    figure.frame()
    figure.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fmt="%.1f")
    figure.xticks(range(len(lags)), ["%d" % lag for lag in lags])
    figure.bars(range(len(lags)), values, width=0.6)
    figure.line(range(len(lags)), values, colour=ORANGE, width=2)
    for index, value in enumerate(values):
        label = "%.3f" % value
        figure.draw.text((figure.px(index) - 3.4 * len(label),
                          figure.py(value) - 18), label,
                         font=figure.draw.getfont(), fill=(17, 17, 17))
    figure.save(FIGURES / "training_pm25_autocorrelation.png")

    # 8. cross-station correlation
    finite = [v for row in matrix for v in row if v is not None]
    heatmap(FIGURES / "training_cross_station_correlation.png", matrix,
            short, short, "Training cross-station PM2.5 correlation",
            TITLE_SUFFIX + "  |  synchronised timestamps, observed pairs",
            vmin=min(finite), vmax=1.0, fmt="%.2f",
            legend_label="Pearson r")


if __name__ == "__main__":
    sys.exit(main())
