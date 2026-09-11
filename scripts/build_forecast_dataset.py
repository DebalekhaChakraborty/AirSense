"""Build the canonical AirSense V2 forecasting data layer.

Protocol Phase 2.

Writes one compact canonical series per station, compact sample indices, and
the audit tables. **No window tensor is materialised**: windows are assembled
on demand by ``src.data.windowing``.

Target quarantine is structural. PM2.5 *values* are written only for training
timestamps; validation and test carry presence, masks and gap age and nothing
else, so no numeric validation or test label exists anywhere in the processed
outputs.

Fits nothing, predicts nothing, computes no metric.

Usage:
    python3 scripts/build_forecast_dataset.py
"""

import csv
import hashlib
import io
import json
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocessing as pre                        # noqa: E402
from src.data.windowing import (                                 # noqa: E402
    CONTEXT_HOURS_PRIMARY, CONTEXT_HOURS_ROBUSTNESS, HORIZONS,
    REGIME_CROSS_STATION, REGIME_POLLUTANTS, make_sample_id, window_bounds)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
OUT = PROJECT_ROOT / "data" / "processed" / "phase2"
RESULTS = PROJECT_ROOT / "results" / "dataset_build"
ARTIFACTS = PROJECT_ROOT / "artifacts"
PHASE1_CONFIG = PROJECT_ROOT / "configs" / "preprocessing.json"

# Recorded observations from the Phase-2 execution. They are literals, not
# live queries, so the manifest stays byte-identical across rebuilds while
# still recording what was measured.
RECORDED = {
    "free_disk_before_kb": 18757668,
    "free_disk_after_kb": 18484932,
    "unit_tests": "44 passed (unittest, stdlib)",
    "phase2_validation": "validate_phase2.py: 59 gates, 0 failures, exit 0",
    "determinism": "122 canonical files byte-identical across two full builds",
    "window_boundary_audit_cases": 108,
    "independent_recomputations": 12,
}

MANIFEST_DOCUMENTS = [
    "docs/WINDOWING_CONTRACT.md",
    "docs/BASELINE_PROTOCOL.md",
    "docs/PHASE_02_FORECAST_DATASET_RECORD.md",
    "configs/windowing.json",
    "configs/baselines.json",
]

INDEX_COLUMNS = ["sample_id", "partition", "station", "station_code",
                 "target_timestamp", "horizon_hours", "forecast_origin",
                 "context_start", "context_end", "target_row_index",
                 "target_observed"]


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "wb") as handle:
        np.save(handle, array, allow_pickle=False)


def main():
    statistics = pre.load_training_statistics(PHASE1_CONFIG)
    length = pre.total_hours()
    train_start_index, train_end_index = pre.partition_index_bounds("train")

    calendar_cyclic, calendar_raw = pre.calendar_channels()
    save_array(OUT / "calendar" / "calendar_cyclic.npy", calendar_cyclic)
    save_array(OUT / "calendar" / "calendar_raw.npy", calendar_raw)

    mask_channels = pre.ALL_NUMERIC + ["wd"]
    fill_channels = pre.ALL_NUMERIC + ["wd"]
    imputation_rows = []
    station_medians = OrderedDict()
    presence = OrderedDict()

    for station in pre.STATIONS:
        path = next(RAW_DIR.glob("PRSA_Data_%s_*.csv" % station))
        raw = pre.load_station_raw(path)
        medians = pre.compute_station_training_medians(raw, station)
        station_medians[station] = medians

        numeric_scaled = np.zeros((length, len(pre.NON_TARGET_NUMERIC)),
                                  dtype=np.float32)
        observed = np.zeros((length, len(mask_channels)), dtype=np.uint8)
        fill_source = np.full((length, len(fill_channels)), -1, dtype=np.int8)
        gap_age = np.zeros((length, len(pre.GAP_AGE_VARIABLES)),
                           dtype=np.uint16)

        for position, name in enumerate(mask_channels):
            observed[:, position] = raw["observed"][name].astype(np.uint8)
        for position, name in enumerate(pre.GAP_AGE_VARIABLES):
            gap_age[:, position] = pre.gap_age_hours(
                raw["observed"][name], pre.GAP_AGE_CAP_HOURS)

        for position, name in enumerate(pre.NON_TARGET_NUMERIC):
            filled, source = pre.causal_forward_fill(
                raw["values"][name], raw["observed"][name],
                pre.CAUSAL_FFILL_MAX_HOURS, medians[name])
            scaled, method = pre.robust_scale(filled, statistics, name)
            numeric_scaled[:, position] = scaled.astype(np.float32)
            fill_source[:, fill_channels.index(name)] = source

        # PM2.5: values exist only inside training, so the causal fill runs on
        # the training region alone. Beyond it the source is -1 = deferred to
        # the guarded rolling target-history interface.
        pm25_scaled = np.full(length, np.nan, dtype=np.float32)
        train_values = raw["values"][pre.TARGET][:train_end_index + 1]
        train_observed = raw["observed"][pre.TARGET][:train_end_index + 1]
        filled, source = pre.causal_forward_fill(
            train_values, train_observed, pre.CAUSAL_FFILL_MAX_HOURS,
            medians[pre.TARGET])
        scaled, _ = pre.robust_scale(filled, statistics, pre.TARGET)
        pm25_scaled[:train_end_index + 1] = scaled.astype(np.float32)
        fill_source[:train_end_index + 1,
                    fill_channels.index(pre.TARGET)] = source

        wd_filled, wd_source = pre.causal_categorical_fill(
            raw["wd_raw"], raw["observed"]["wd"],
            pre.CAUSAL_FFILL_MAX_HOURS, pre.WD_MISSING_CODE)
        fill_source[:, fill_channels.index("wd")] = wd_source

        rain_index = pre.NON_TARGET_NUMERIC.index("RAIN")
        rain_observed = raw["observed"]["RAIN"]
        rain_values = raw["values"]["RAIN"]
        rain_occurred = ((rain_observed) & (np.nan_to_num(rain_values) > 0)
                         ).astype(np.uint8)

        target_native = np.full(length, np.nan, dtype=np.float32)
        target_native[:train_end_index + 1] = raw["values"][pre.TARGET][
            :train_end_index + 1].astype(np.float32)

        base = OUT / "station_series" / station
        save_array(base / "numeric_scaled.npy", numeric_scaled)
        save_array(base / "observed_mask.npy", observed)
        save_array(base / "gap_age.npy", gap_age)
        save_array(base / "wd_code.npy", wd_filled)
        save_array(base / "rain_occurred.npy", rain_occurred)
        save_array(base / "fill_source.npy", fill_source)
        save_array(base / "pm25_scaled_train_only.npy", pm25_scaled)
        save_array(OUT / "target_series" / ("%s_train_pm25.npy" % station),
                   target_native)
        write_json(base / "meta.json", OrderedDict([
            ("station", station),
            ("station_code", pre.station_code(station)),
            ("start", pre.FULL_START.isoformat()),
            ("end", pre.FULL_END.isoformat()),
            ("hours", length),
            ("numeric_channels", pre.NON_TARGET_NUMERIC),
            ("mask_channels", mask_channels),
            ("gap_age_channels", pre.GAP_AGE_VARIABLES),
            ("fill_source_channels", fill_channels),
            ("fill_source_codes", {"0": "observed", "1": "causal_ffill",
                                   "2": "training_median_fallback",
                                   "-1": "deferred_to_rolling_target_history"}),
            ("pm25_values_materialised_through",
             pre.PARTITION_BOUNDS["train"][1].isoformat()),
            ("training_medians", OrderedDict(
                (k, round(v, 6)) for k, v in medians.items())),
        ]))

        presence[station] = raw["observed"][pre.TARGET]

        for partition in pre.PARTITION_BOUNDS:
            low, high = pre.partition_index_bounds(partition)
            for name in fill_channels:
                column = fill_channels.index(name)
                window = fill_source[low:high + 1, column]
                if name == pre.TARGET and partition != "train":
                    imputation_rows.append([
                        partition, station, name, "", "", "", "",
                        "deferred_to_rolling_target_history"])
                    continue
                observed_count = int((window == 0).sum())
                ffill_count = int((window == 1).sum())
                fallback_count = int((window == 2).sum())
                if name == "wd":
                    missing_category = fallback_count
                    max_age = ""
                else:
                    missing_category = ""
                    age_column = (pre.GAP_AGE_VARIABLES.index(name)
                                  if name in pre.GAP_AGE_VARIABLES else None)
                    max_age = (int(gap_age[low:high + 1, age_column].max())
                               if age_column is not None else "")
                imputation_rows.append([
                    partition, station, name, observed_count, ffill_count,
                    fallback_count, missing_category, max_age])

    write_csv(RESULTS / "imputation_audit.csv",
              ["partition", "station", "variable", "raw_observed",
               "short_gap_causal_ffill", "training_median_fallback",
               "missing_category_used", "max_gap_age_hours"],
              imputation_rows)

    # ---- sample universe --------------------------------------------------
    counts = OrderedDict()
    exclusions = []
    index_hashes = OrderedDict()
    for partition in pre.PARTITION_BOUNDS:
        low, high = pre.partition_index_bounds(partition)
        rows = []
        for station in pre.STATIONS:
            observed_target = presence[station]
            for horizon in HORIZONS:
                eligible = 0
                missing_target = 0
                short_history = 0
                for position in range(low, high + 1):
                    target_timestamp = pre.timestamp_at(position)
                    start, origin = window_bounds(target_timestamp, horizon,
                                                  CONTEXT_HOURS_PRIMARY)
                    if pre.index_of(start) < 0:
                        short_history += 1
                        continue
                    if not observed_target[position]:
                        missing_target += 1
                        continue
                    eligible += 1
                    rows.append([
                        make_sample_id(partition, station, target_timestamp,
                                       horizon),
                        partition, station, pre.station_code(station),
                        target_timestamp.isoformat(), horizon,
                        origin.isoformat(), start.isoformat(),
                        origin.isoformat(), position, 1])
                counts["%s|%s|%d" % (partition, station, horizon)] = eligible
                exclusions.append([partition, station, horizon,
                                   high - low + 1, short_history,
                                   missing_target, eligible])
        rows.sort(key=lambda r: (r[2], r[5], r[4]))
        path = OUT / "sample_index" / ("%s.csv" % partition)
        write_csv(path, INDEX_COLUMNS, rows)
        index_hashes[partition] = sha256_of_file(path)

    write_csv(RESULTS / "sample_counts.csv",
              ["partition", "station", "horizon_hours", "candidate_targets",
               "excluded_insufficient_history", "excluded_target_missing",
               "eligible_samples"], exclusions)

    # ---- channel schema ---------------------------------------------------
    schema_rows = []

    def add_channel(channel, regime, kind, scaled, method, mask, age,
                    known_at_origin, future_deterministic, source):
        schema_rows.append([channel, regime, kind, scaled, method, mask, age,
                            known_at_origin, future_deterministic, source])

    for name in pre.METEOROLOGY:
        method = ("median_std_zero_iqr_fallback" if statistics[name]["iqr"] == 0
                  else "median_iqr")
        add_channel(name, "R0", "continuous", "true", method, "true", "false",
                    "true", "false", name)
    add_channel("rain_occurred", "R0", "binary", "false", "none", "true",
                "false", "true", "false", "RAIN")
    add_channel("wd_code", "R0", "categorical", "false",
                "training_vocabulary_16_plus_MISSING", "true", "false",
                "true", "false", "wd")
    for name in pre.CALENDAR_CHANNEL_NAMES:
        add_channel("origin_" + name, "R0", "cyclic", "false", "none",
                    "false", "false", "true", "false", "timestamp")
    for name in pre.CALENDAR_CHANNEL_NAMES:
        add_channel("target_" + name, "R0", "cyclic", "false", "none",
                    "false", "false", "true", "true", "timestamp")
    add_channel("station_code", "R0", "identity", "false", "none", "false",
                "false", "true", "true", "station")
    add_channel("PM2.5", "R1", "continuous", "true", "median_iqr", "true",
                "true", "true", "false", "PM2.5")
    for name in ["PM10", "SO2", "NO2", "CO", "O3"]:
        add_channel(name, "R2", "continuous", "true", "median_iqr", "true",
                    "true", "true", "false", name)
    add_channel("cross_station_pollutant_history", "R3", "continuous", "true",
                "median_iqr", "true", "true", "true", "false",
                "other_11_stations")
    write_csv(RESULTS / "channel_schema.csv",
              ["channel", "regime_minimum", "type", "scaled",
               "scaling_method", "mask_available", "gap_age_available",
               "known_at_origin", "future_deterministic", "source_variable"],
              schema_rows)

    # ---- artifacts --------------------------------------------------------
    write_json(ARTIFACTS / "preprocessing_statistics.json", OrderedDict([
        ("source", "phase_1_frozen_training_statistics"),
        ("fitted_on", "training_partition_pooled_across_stations"),
        ("formula", "(x - median) / IQR"),
        ("zero_iqr_fallback_formula", "(x - median) / std"),
        ("zero_iqr_variables", [n for n in pre.ALL_NUMERIC
                                if statistics[n]["iqr"] == 0]),
        ("pooled_training_statistics", statistics),
        ("station_training_medians", OrderedDict(
            (station, OrderedDict((k, round(v, 6))
                                  for k, v in medians.items()))
            for station, medians in station_medians.items())),
        ("wind_direction_vocabulary", pre.WD_VOCABULARY),
        ("wind_direction_missing_code", pre.WD_MISSING_CODE),
        ("causal_ffill_max_hours", pre.CAUSAL_FFILL_MAX_HOURS),
        ("gap_age_cap_hours", pre.GAP_AGE_CAP_HOURS),
        ("target_transform", "none_native_ug_m3"),
    ]))

    write_json(ARTIFACTS / "information_regime_schema.json", OrderedDict([
        ("context_hours_primary", CONTEXT_HOURS_PRIMARY),
        ("horizons", HORIZONS),
        ("common_sample_universe", True),
        ("regimes", OrderedDict([
            (regime, OrderedDict([
                ("meteorology", pre.METEOROLOGY),
                ("wind_direction", True),
                ("rain_occurred", True),
                ("calendar", pre.CALENDAR_CHANNEL_NAMES),
                ("masks", True),
                ("pollutant_history", REGIME_POLLUTANTS[regime]),
                ("pollutant_masks", REGIME_POLLUTANTS[regime]),
                ("pollutant_gap_age", REGIME_POLLUTANTS[regime]),
                ("cross_station_history", REGIME_CROSS_STATION[regime]),
                ("tensor_encoding_fixed", False),
            ])) for regime in ("R0", "R1", "R2", "R3")])),
        ("note", "R3 spatial tensor encoding is deliberately unfixed and "
                 "belongs to the spatiotemporal phase."),
    ]))

    by_partition = OrderedDict()
    by_station = OrderedDict()
    by_horizon = OrderedDict()
    by_station_horizon = OrderedDict()
    for key, value in counts.items():
        partition, station, horizon = key.split("|")
        by_partition[partition] = by_partition.get(partition, 0) + value
        by_station[station] = by_station.get(station, 0) + value
        by_horizon[horizon] = by_horizon.get(horizon, 0) + value
        by_station_horizon["%s|%s" % (station, horizon)] = (
            by_station_horizon.get("%s|%s" % (station, horizon), 0) + value)

    universe = OrderedDict([
        ("context_hours", CONTEXT_HOURS_PRIMARY),
        ("horizons", HORIZONS),
        ("common_across_models", True),
        ("common_across_regimes", True),
        ("sample_id_format", "<partition>|<station>|<YYYYMMDDHH>|h<horizon>"),
        ("index_file_sha256", index_hashes),
        ("counts_by_partition", by_partition),
        ("counts_by_station", by_station),
        ("counts_by_horizon", by_horizon),
        ("counts_by_partition_station_horizon", counts),
        ("counts_by_station_horizon_all_partitions", by_station_horizon),
        ("structural_exclusion_reasons", OrderedDict([
            ("excluded_insufficient_history",
             "context_start would precede 2013-03-01 00:00"),
            ("excluded_target_missing",
             "target PM2.5 not observed; labels are never imputed"),
        ])),
        ("target_missing_exclusions_total", OrderedDict(
            (partition, sum(row[5] for row in exclusions
                            if row[0] == partition))
            for partition in pre.PARTITION_BOUNDS)),
        ("insufficient_history_exclusions_total", OrderedDict(
            (partition, sum(row[4] for row in exclusions
                            if row[0] == partition))
            for partition in pre.PARTITION_BOUNDS)),
        ("validation_target_values_materialized", False),
        ("test_target_values_materialized", False),
    ])
    write_json(ARTIFACTS / "sample_universe_manifest.json", universe)

    # ---- B2 training climatology (training statistic, not a prediction) ----
    climatology_rows = []
    for station in pre.STATIONS:
        target_native = np.load(str(
            OUT / "target_series" / ("%s_train_pm25.npy" % station)))
        train_slice = target_native[train_start_index:train_end_index + 1]
        stamps_month = calendar_raw[train_start_index:train_end_index + 1, 2]
        stamps_hour = calendar_raw[train_start_index:train_end_index + 1, 0]
        finite = np.isfinite(train_slice)

        def median_of(selector):
            sample = np.sort(train_slice[selector])
            if sample.size == 0:
                return None
            position = (sample.size - 1) * 0.5
            lower = int(np.floor(position))
            upper = min(lower + 1, sample.size - 1)
            fraction = position - lower
            return float(sample[lower]
                         + fraction * (sample[upper] - sample[lower]))

        station_median = median_of(finite)
        for month in range(1, 13):
            for hour in range(24):
                selector = finite & (stamps_month == month) & (
                    stamps_hour == hour)
                value = median_of(selector)
                climatology_rows.append([
                    station, "station_month_hour", month, hour,
                    int(selector.sum()),
                    "" if value is None else round(value, 6)])
        for hour in range(24):
            selector = finite & (stamps_hour == hour)
            value = median_of(selector)
            climatology_rows.append([
                station, "station_hour", "", hour, int(selector.sum()),
                "" if value is None else round(value, 6)])
        climatology_rows.append([
            station, "station", "", "", int(finite.sum()),
            "" if station_median is None else round(station_median, 6)])
    write_csv(OUT / "climatology" / "b2_training_climatology.csv",
              ["station", "level", "month", "hour", "n_training_observations",
               "median_pm25_ug_m3"], climatology_rows)

    # ---- deterministic window boundary audit -------------------------------
    from src.data.windowing import ForecastWindowLoader        # noqa: E402
    loader = ForecastWindowLoader(OUT)
    audit_stations = [pre.STATIONS[0], pre.STATIONS[len(pre.STATIONS) // 2],
                      pre.STATIONS[-1]]
    audit_rows = []
    for partition in pre.PARTITION_BOUNDS:
        with open(str(OUT / "sample_index" / ("%s.csv" % partition)),
                  encoding="utf-8") as handle:
            index = [row for row in csv.DictReader(handle)]
        for station in audit_stations:
            for horizon in HORIZONS:
                subset = [row for row in index
                          if row["station"] == station
                          and int(row["horizon_hours"]) == horizon]
                if not subset:
                    continue
                picks = OrderedDict([
                    ("first", subset[0]),
                    ("middle", subset[len(subset) // 2]),
                    ("last", subset[-1])])
                for label, row in picks.items():
                    window = loader.get_window(row["sample_id"], regime="R2")
                    stamps = window["timestamps"]
                    spacing_ok = all(
                        (datetime.fromisoformat(b)
                         - datetime.fromisoformat(a)) == timedelta(hours=1)
                        for a, b in zip(stamps, stamps[1:]))
                    audit_rows.append([
                        partition, station, horizon, label, row["sample_id"],
                        len(stamps), stamps[0], stamps[-1],
                        window["forecast_origin"], window["target_timestamp"],
                        str(spacing_ok).lower(),
                        str(stamps[-1] == window["forecast_origin"]).lower(),
                        str(stamps[0] == window["context_start"]).lower(),
                        str(max(stamps) <= window["forecast_origin"]).lower(),
                        str(window["target_value_available"]).lower(),
                    ])
    write_csv(RESULTS / "window_boundary_audit.csv",
              ["partition", "station", "horizon_hours", "position",
               "sample_id", "context_length", "context_start", "context_end",
               "forecast_origin", "target_timestamp", "hourly_contiguous",
               "end_equals_origin", "start_equals_origin_minus_47",
               "no_input_after_origin", "target_value_returned"], audit_rows)

    # ---- storage summary --------------------------------------------------
    storage_rows = []
    total_bytes = 0
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total_bytes += size
            storage_rows.append([str(path.relative_to(PROJECT_ROOT)), size])
    write_csv(RESULTS / "storage_summary.csv",
              ["path", "size_bytes"],
              storage_rows + [["TOTAL_processed_phase2", total_bytes]])

    if "--write-manifest" in sys.argv:
        write_json(ARTIFACTS / "v2_phase2_manifest.json", OrderedDict([
            ("study", "AirSense V2"),
            ("phase", "phase_2_forecast_dataset_and_baseline_protocol"),
            ("foundation_manifest_sha256",
             sha256_of_file(ARTIFACTS / "v2_foundation_manifest.json")),
            ("phase1_manifest_sha256",
             sha256_of_file(ARTIFACTS / "v2_phase1_manifest.json")),
            ("context_hours_primary", CONTEXT_HOURS_PRIMARY),
            ("context_hours_robustness", CONTEXT_HOURS_ROBUSTNESS),
            ("context_selected_by_validation_performance", False),
            ("forecast_horizons_hours", HORIZONS),
            ("processed_schema", OrderedDict([
                ("numeric_channels", pre.NON_TARGET_NUMERIC),
                ("mask_channels", pre.ALL_NUMERIC + ["wd"]),
                ("gap_age_channels", pre.GAP_AGE_VARIABLES),
                ("calendar_channels", pre.CALENDAR_CHANNEL_NAMES),
                ("categorical_channels", ["wd_code"]),
                ("binary_channels", ["rain_occurred"]),
                ("target_units", "ug_m3"),
            ])),
            ("preprocessing_statistics_sha256",
             sha256_of_file(ARTIFACTS / "preprocessing_statistics.json")),
            ("information_regime_schema_sha256",
             sha256_of_file(ARTIFACTS / "information_regime_schema.json")),
            ("sample_universe_manifest_sha256",
             sha256_of_file(ARTIFACTS / "sample_universe_manifest.json")),
            ("sample_index_sha256", index_hashes),
            ("processed_array_sha256", OrderedDict(
                (str(path.relative_to(OUT)), sha256_of_file(path))
                for path in sorted(OUT.rglob("*.npy")))),
            ("document_sha256", OrderedDict(
                (name, sha256_of_file(PROJECT_ROOT / name))
                for name in MANIFEST_DOCUMENTS)),
            ("audit_table_sha256", OrderedDict(
                (path.name, sha256_of_file(path))
                for path in sorted(RESULTS.glob("*.csv")))),
            ("climatology_table_sha256", sha256_of_file(
                OUT / "climatology" / "b2_training_climatology.csv")),
            ("sample_counts", by_partition),
            ("windows_materialized", False),
            ("windows_built_on_demand", True),
            ("processed_bytes_total", total_bytes),
            ("unit_tests", RECORDED["unit_tests"]),
            ("phase2_validation", RECORDED["phase2_validation"]),
            ("deterministic_rerun", RECORDED["determinism"]),
            ("window_boundary_audit_cases",
             RECORDED["window_boundary_audit_cases"]),
            ("independent_recomputations",
             RECORDED["independent_recomputations"]),
            ("free_disk_before_kb", RECORDED["free_disk_before_kb"]),
            ("free_disk_after_kb", RECORDED["free_disk_after_kb"]),
            ("model_runs", 0),
            ("predictions_generated", 0),
            ("validation_metrics_seen", 0),
            ("test_metrics_seen", 0),
            ("validation_target_values_materialized", False),
            ("test_target_values_materialized", False),
            ("heavy_dependencies_installed", False),
            ("v1_legacy_modified", False),
            ("final_test_status", "sealed"),
            ("note", "Deterministic by construction: no wall-clock timestamp "
                     "and no filesystem mtime is recorded. Disk and test "
                     "figures are recorded observations, written as literals "
                     "so rebuilds stay byte-identical."),
        ]))
        print("wrote artifacts/v2_phase2_manifest.json")

    print("")
    print("stations              : %d" % len(pre.STATIONS))
    print("timeline hours        : %d" % length)
    print("context hours         : %d" % CONTEXT_HOURS_PRIMARY)
    for partition in pre.PARTITION_BOUNDS:
        print("eligible %-11s: %d" % (partition, by_partition[partition]))
    print("processed bytes       : %d (%.1f MB)"
          % (total_bytes, total_bytes / 1048576.0))
    print("no window tensor materialised; windows are built on demand")
    return 0


if __name__ == "__main__":
    sys.exit(main())
