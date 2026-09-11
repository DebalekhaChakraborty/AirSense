"""Phase-7 analysis-only geographic robustness diagnostics.

Coordinates never enter a model, a graph, a feature tensor, or a selection
fit. This script combines the frozen Phase-6 SA_R3 attention matrix and
training-only PM2.5 correlations with the externally sourced Phase-7 station
coordinates. All relationships are descriptive and non-causal.
"""

import csv
import hashlib
import io
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import STATIONS  # noqa

ARTIFACTS = PROJECT_ROOT / "artifacts"
SPATIAL = PROJECT_ROOT / "results" / "models" / "spatiotemporal"
EDA = PROJECT_ROOT / "results" / "eda_training"
OUT = PROJECT_ROOT / "results" / "models" / "robustness"
COORDINATES = ARTIFACTS / "phase7_station_coordinates.json"


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(str(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def ranks(values):
    """Average ranks for ties, matching the frozen Phase-6 implementation."""
    values = np.asarray(values, dtype=np.float64)
    order = values.argsort(kind="stable")
    ranked = np.empty(values.size, dtype=np.float64)
    ranked[order] = np.arange(1, values.size + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    sums = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, ranked)
    return (sums / counts)[inverse]


def spearman(left, right):
    left_rank = ranks(left)
    right_rank = ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(
        np.sqrt((left_rank * left_rank).sum() * (right_rank * right_rank).sum())
    )
    return float((left_rank * right_rank).sum() / denominator)


def haversine_km(first, second):
    radius_km = 6371.0088
    lat1, lon1 = map(math.radians, (first["latitude"], first["longitude"]))
    lat2, lon2 = map(math.radians, (second["latitude"], second["longitude"]))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(value))


def main():
    freeze = ARTIFACTS / "phase7_candidate_freeze.json"
    if not freeze.is_file():
        raise SystemExit("the Phase-7 candidate freeze must exist first")

    coordinate_payload = load_json(COORDINATES)
    coordinates = {row["station"]: row for row in coordinate_payload["stations"]}
    if list(coordinates) != list(STATIONS):
        raise SystemExit("coordinate station order differs from frozen station order")
    if any(row["source_station"] != name for name, row in coordinates.items()):
        raise SystemExit("coordinate mapping must use exact station names")
    if coordinate_payload["model_input"] or coordinate_payload["graph_constructed"]:
        raise SystemExit("coordinates must remain analysis-only")

    attention_rows = read_csv(SPATIAL / "attention_matrix_overall.csv")
    if [row["target_station"] for row in attention_rows] != list(STATIONS):
        raise SystemExit("attention matrix station order differs from frozen order")
    attention = {
        (row["target_station"], source): float(row[source])
        for row in attention_rows
        for source in STATIONS
    }
    pm25 = {
        (row["station_a"], row["station_b"]): float(row["pearson_r"])
        for row in read_csv(EDA / "cross_station_pm25_correlation.csv")
    }

    distances = {
        (target, source): haversine_km(coordinates[target], coordinates[source])
        for target in STATIONS
        for source in STATIONS
    }
    write_csv(
        OUT / "station_distance_matrix_km.csv",
        ["target_station"] + list(STATIONS),
        [
            [target]
            + [round(distances[(target, source)], 6) for source in STATIONS]
            for target in STATIONS
        ],
    )

    pair_rows = []
    nearest_rows = []
    pair_values = []
    for target in STATIONS:
        sources = [source for source in STATIONS if source != target]
        nearest = min(sources, key=lambda source: distances[(target, source)])
        highest = max(sources, key=lambda source: attention[(target, source)])
        attention_order = sorted(
            sources, key=lambda source: attention[(target, source)], reverse=True
        )
        nearest_rows.append(
            [
                target,
                nearest,
                round(distances[(target, nearest)], 6),
                round(attention[(target, nearest)], 8),
                attention_order.index(nearest) + 1,
                highest,
                nearest == highest,
            ]
        )
        for source in sources:
            values = (
                distances[(target, source)],
                attention[(target, source)],
                pm25[(target, source)],
            )
            pair_values.append(values)
            pair_rows.append(
                [
                    target,
                    source,
                    round(values[0], 6),
                    round(values[1], 8),
                    round(values[2], 6),
                    source == nearest,
                    source == highest,
                ]
            )

    write_csv(
        OUT / "spatial_attention_pairs.csv",
        [
            "target_station",
            "source_station",
            "distance_km",
            "mean_attention_weight",
            "training_pm25_pearson_r",
            "nearest_geographic_station",
            "highest_attention_station",
        ],
        pair_rows,
    )
    write_csv(
        OUT / "nearest_station_attention.csv",
        [
            "target_station",
            "nearest_source_station",
            "distance_km",
            "mean_attention_weight",
            "attention_rank_among_other_stations",
            "highest_attention_source_station",
            "nearest_is_highest_attention",
        ],
        nearest_rows,
    )

    distances_vector = np.array([row[0] for row in pair_values])
    attention_vector = np.array([row[1] for row in pair_values])
    correlation_vector = np.array([row[2] for row in pair_values])
    quartiles = np.quantile(distances_vector, [0.25, 0.5, 0.75])
    bins = np.digitize(distances_vector, quartiles, right=True)
    distance_bins = []
    for index, label in enumerate(("nearest_quartile", "q2", "q3", "farthest_quartile")):
        selected = bins == index
        distance_bins.append(
            OrderedDict(
                [
                    ("label", label),
                    ("n_directed_pairs", int(selected.sum())),
                    ("mean_distance_km", round(float(distances_vector[selected].mean()), 6)),
                    ("mean_attention_weight", round(float(attention_vector[selected].mean()), 8)),
                ]
            )
        )

    slope, intercept = np.polyfit(distances_vector, attention_vector, 1)
    nearest_attention = np.array([float(row[3]) for row in nearest_rows])
    summary = OrderedDict(
        [
            ("artifact", "phase7_spatial_analysis"),
            ("analysis_only", True),
            ("coordinates_used_as_model_input", False),
            ("graph_constructed", False),
            ("causal_claim", False),
            ("directed_off_diagonal_pairs", len(pair_values)),
            ("spearman_distance_vs_attention", round(spearman(distances_vector, attention_vector), 6)),
            ("spearman_distance_vs_training_pm25_correlation", round(spearman(distances_vector, correlation_vector), 6)),
            ("spearman_attention_vs_training_pm25_correlation", round(spearman(attention_vector, correlation_vector), 6)),
            ("linear_attention_per_km_slope", round(float(slope), 10)),
            ("linear_attention_intercept", round(float(intercept), 8)),
            ("mean_nearest_station_attention", round(float(nearest_attention.mean()), 8)),
            ("median_nearest_station_attention", round(float(np.median(nearest_attention)), 8)),
            ("nearest_is_highest_attention_count", sum(bool(row[6]) for row in nearest_rows)),
            ("station_count", len(STATIONS)),
            ("distance_quartiles", distance_bins),
            ("coordinate_artifact_sha256", sha256_of_file(COORDINATES)),
            ("attention_matrix_sha256", sha256_of_file(SPATIAL / "attention_matrix_overall.csv")),
            ("training_correlation_sha256", sha256_of_file(EDA / "cross_station_pm25_correlation.csv")),
            ("candidate_freeze_sha256", sha256_of_file(freeze)),
            ("interpretation", "Descriptive robustness evidence only. Attention weights are not causal influence estimates, and geographic association does not establish mechanism."),
        ]
    )
    path = ARTIFACTS / "phase7_spatial_analysis.json"
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
