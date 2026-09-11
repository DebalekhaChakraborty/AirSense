"""Minimal confirmatory figure set for the AirSense V2 locked final test.

Protocol Phase 8, section 42. Three figures, no exploratory plots, every
title carrying LOCKED FINAL TEST. Run **after** the primary-results lock, so
no figure can influence a metric.

Pillow lives in the system interpreter, not in .venv-v2, so this runs under
python3 and reads only the frozen metric tables - never a model, never a
prediction array, never a target.

Usage:
    python3 scripts/build_phase8_figures.py
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
    BLUE, GREY, ORANGE, TEAL, Figure)

ARTIFACTS = PROJECT_ROOT / "artifacts"
TABLES = ARTIFACTS / "phase8_final_test_tables"
FIGURES = PROJECT_ROOT / "figures"
MODELS = ["B3_R2", "GRU_R1", "B0"]
COLOURS = {"B3_R2": BLUE, "GRU_R1": TEAL, "B0": ORANGE}
ROLES = {"B3_R2": "primary confirmatory",
         "GRU_R1": "secondary confirmatory (severe tail)",
         "B0": "reference benchmark"}


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with open(str(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def nice_ceiling(value):
    step = 10.0 ** (len(str(int(value))) - 2) * 5 if value > 10 else 1.0
    return step * (int(value / step) + 1)


def primary_figure(comparison):
    values = [float(row["macro_station_horizon_MAE"]) for row in comparison]
    top = nice_ceiling(max(values))
    figure = Figure(
        width=880, height=560,
        title="LOCKED FINAL TEST - primary endpoint",
        subtitle="Macro station-horizon MAE, 48 equally weighted cells, "
                 "2016-03-01 to 2017-02-28",
        xlabel="model (frozen confirmatory role)",
        ylabel="MAE (ug/m3)")
    figure.set_xlim(-0.6, len(comparison) - 0.4)
    figure.set_ylim(0.0, top)
    figure.yticks([top * i / 5.0 for i in range(6)], "%.0f")
    figure.frame()
    for position, row in enumerate(comparison):
        figure.bars([position], [float(row["macro_station_horizon_MAE"])],
                    width=0.55, colour=COLOURS[row["model"]])
    figure.xticks(range(len(comparison)), [r["model"] for r in comparison])
    for position, row in enumerate(comparison):
        value = float(row["macro_station_horizon_MAE"])
        figure.draw.text((figure.px(position) - 22, figure.py(value) - 18),
                         "%.3f" % value, fill=(17, 17, 17))
    figure.note("B3_R2 is the primary confirmatory model; B0 is a reference "
                "benchmark, not a selected learned model.")
    path = FIGURES / "phase8_final_test_primary_mae.png"
    figure.save(path)
    return path


def severe_figure(severe):
    values = [float(row["severe_MAE"]) for row in severe]
    top = nice_ceiling(max(values))
    n = severe[0]["severe_n"]
    figure = Figure(
        width=880, height=560,
        title="LOCKED FINAL TEST - secondary endpoint (severe tail)",
        subtitle="Severe MAE where actual PM2.5 > 244.0 ug/m3 "
                 "(n = %s test hours)" % n,
        xlabel="model (frozen confirmatory role)",
        ylabel="severe MAE (ug/m3)")
    figure.set_xlim(-0.6, len(severe) - 0.4)
    figure.set_ylim(0.0, top)
    figure.yticks([top * i / 5.0 for i in range(6)], "%.0f")
    figure.frame()
    for position, row in enumerate(severe):
        figure.bars([position], [float(row["severe_MAE"])], width=0.55,
                    colour=COLOURS[row["model"]])
    figure.xticks(range(len(severe)), [r["model"] for r in severe])
    for position, row in enumerate(severe):
        value = float(row["severe_MAE"])
        figure.draw.text((figure.px(position) - 26, figure.py(value) - 18),
                         "%.3f" % value, fill=(17, 17, 17))
    figure.note("GRU_R1 (seed 42) is the prespecified secondary confirmatory "
                "model on this endpoint. Threshold frozen at 244.0.")
    path = FIGURES / "phase8_final_test_severe_mae.png"
    figure.save(path)
    return path


def horizon_figure(by_horizon):
    series = OrderedDict()
    for row in by_horizon:
        series.setdefault(row["model"], []).append(
            (int(row["horizon_hours"]), float(row["macro_station_MAE"])))
    top = nice_ceiling(max(v for rows in series.values() for _, v in rows))
    figure = Figure(
        width=880, height=560,
        title="LOCKED FINAL TEST - macro station MAE by horizon",
        subtitle="Equal-weight mean over 12 stations at each forecast horizon",
        xlabel="forecast horizon (hours)",
        ylabel="MAE (ug/m3)")
    figure.set_xlim(0, 25)
    figure.set_ylim(0.0, top)
    figure.yticks([top * i / 5.0 for i in range(6)], "%.0f")
    figure.frame()
    for name in MODELS:
        rows = sorted(series[name])
        figure.line([h for h, _ in rows], [v for _, v in rows],
                    colour=COLOURS[name], width=3)
        figure.points([h for h, _ in rows], [v for _, v in rows],
                      colour=COLOURS[name], radius=4)
    figure.xticks([1, 6, 12, 24], ["1", "6", "12", "24"])
    figure.legend([("%s - %s" % (name, ROLES[name]), COLOURS[name])
                   for name in MODELS])
    figure.note("Descriptive breakdown of the frozen endpoints; it does not "
                "alter any model's confirmatory role.")
    path = FIGURES / "phase8_final_test_mae_by_horizon.png"
    figure.save(path)
    return path


def main():
    lock = json.load(open(str(ARTIFACTS / "phase8_primary_results_lock.json"),
                          encoding="utf-8"))
    if lock["final_test_status"] != "evaluated":
        raise SystemExit("refusing to draw: the results lock is not final")
    FIGURES.mkdir(parents=True, exist_ok=True)

    comparison = read_csv(TABLES / "final_model_comparison.csv")
    order = {name: position for position, name in enumerate(MODELS)}
    comparison.sort(key=lambda row: order[row["model"]])
    severe = read_csv(TABLES / "severe_metrics.csv")
    severe.sort(key=lambda row: order[row["model"]])
    by_horizon = read_csv(TABLES / "metrics_by_horizon.csv")

    produced = OrderedDict()
    for path in (primary_figure(comparison), severe_figure(severe),
                 horizon_figure(by_horizon)):
        produced[path.name] = sha256_of_file(path)
        print("%-44s %s" % (path.name, produced[path.name][:16]))

    with open(str(ARTIFACTS / "phase8_figure_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(OrderedDict([
            ("study", "AirSense V2"),
            ("artifact", "phase8_figure_manifest"),
            ("drawn_after_primary_results_lock", True),
            ("primary_results_lock_sha256",
             sha256_of_file(ARTIFACTS / "phase8_primary_results_lock.json")),
            ("exploratory_figures", 0),
            ("figure_sha256", produced),
        ]), handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
