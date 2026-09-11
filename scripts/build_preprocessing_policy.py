"""Freeze the AirSense V2 Phase-1 preprocessing contract.

Protocol Phase 1.

Writes:
    results/eda_training/training_scaling_statistics.csv
    configs/preprocessing.json
    artifacts/v2_phase1_manifest.json      (with --write-manifest)

Every statistic here is computed from the **training partition only**. The
validation and locked-test partitions are never opened by this script. No
model is fitted; scaling statistics are descriptive constants, not learned
parameters of a predictor.

Usage:
    python3 scripts/build_preprocessing_policy.py
    python3 scripts/build_preprocessing_policy.py --write-manifest
"""

import argparse
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

from src.data.eda import (                                      # noqa: E402
    NUMERIC_COLUMNS, PARTITIONS, QUANTILE_METHOD, TARGET, load_partition,
    quantile, station_paths)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PRSA_Data_20130301-20170228"
RESULTS = PROJECT_ROOT / "results" / "eda_training"
SCALING_CSV = RESULTS / "training_scaling_statistics.csv"
CONFIG = PROJECT_ROOT / "configs" / "preprocessing.json"
MANIFEST = PROJECT_ROOT / "artifacts" / "v2_phase1_manifest.json"
FOUNDATION = PROJECT_ROOT / "artifacts" / "v2_foundation_manifest.json"
TRAINING_EDA = PROJECT_ROOT / "artifacts" / "training_eda.json"
FIGURES = PROJECT_ROOT / "figures"

# Frozen from the Phase-1 gap audit: 91.6% of PM2.5 gaps are <= 6 h, the P90
# gap is 5 h, and PM2.5 autocorrelation is still 0.752 at lag 6 but only 0.368
# at lag 24. Carrying an observation forward beyond 6 h would assert a value
# the data no longer supports.
MAX_FORWARD_FILL_H = 6

CONTEXT_LENGTH_CANDIDATES = [24, 48, 72]
CONTEXT_LENGTH_OPTIONAL = [168]

EDA_TABLES = [
    "pm25_summary.csv",
    "pm25_summary_by_station.csv",
    "missing_run_summary.csv",
    "target_missingness_temporal.csv",
    "simultaneous_station_missingness.csv",
    "wind_direction_category_counts.csv",
    "pm25_autocorrelation.csv",
    "cross_station_pm25_correlation.csv",
    "severe_thresholds.csv",
    "target_eligibility_by_partition_horizon.csv",
    "training_scaling_statistics.csv",
]

EDA_FIGURES = [
    "training_pm25_distribution.png",
    "training_pm25_by_station.png",
    "training_monthly_pm25_pattern.png",
    "training_hourly_pm25_pattern.png",
    "training_missingness_by_variable_station.png",
    "training_pm25_missing_run_lengths.png",
    "training_pm25_autocorrelation.png",
    "training_cross_station_correlation.png",
]

POLICY_DOC = "docs/PREPROCESSING_POLICY.md"
EDA_DOC = "docs/TRAINING_EDA.md"


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def training_scaling_statistics():
    """Robust location/scale per numeric variable, training partition only."""
    pooled = {column: [] for column in NUMERIC_COLUMNS}
    per_station = OrderedDict()
    for path in station_paths(RAW_DIR):
        record = load_partition(path, "train", mode="values")
        station = record["station"]
        per_station[station] = {}
        for column in NUMERIC_COLUMNS:
            values = record["numeric"][column][record["observed"][column]]
            values = values[np.isfinite(values)]
            pooled[column].append(values)
            per_station[station][column] = values
    rows = []
    statistics = OrderedDict()
    for column in NUMERIC_COLUMNS:
        values = np.concatenate(pooled[column])
        p25 = quantile(values, 25)
        p50 = quantile(values, 50)
        p75 = quantile(values, 75)
        iqr = p75 - p25
        statistics[column] = OrderedDict([
            ("n_observed", int(values.size)),
            ("median", round(p50, 6)),
            ("p25", round(p25, 6)),
            ("p75", round(p75, 6)),
            ("iqr", round(iqr, 6)),
            ("mean", round(float(values.mean()), 6)),
            ("std", round(float(values.std(ddof=1)), 6)),
        ])
        rows.append(["pooled_training", "ALL", column]
                    + [statistics[column][k] for k in
                       ("n_observed", "median", "p25", "p75", "iqr", "mean",
                        "std")])
    for station, columns in per_station.items():
        for column in NUMERIC_COLUMNS:
            values = columns[column]
            if values.size == 0:
                continue
            p25 = quantile(values, 25)
            p50 = quantile(values, 50)
            p75 = quantile(values, 75)
            rows.append(["station_training", station, column, int(values.size),
                         round(p50, 6), round(p25, 6), round(p75, 6),
                         round(p75 - p25, 6),
                         round(float(values.mean()), 6),
                         round(float(values.std(ddof=1)), 6)])
    write_csv(SCALING_CSV,
              ["scope", "station", "variable", "n_observed", "median", "p25",
               "p75", "iqr", "mean", "std"], rows)
    return statistics


def build_config(statistics, eda):
    return OrderedDict([
        ("study", "AirSense V2"),
        ("phase", "phase_1_preprocessing_contract"),
        ("fitted_on", "training_partition_only"),
        ("training_window", OrderedDict([
            ("start", PARTITIONS["train"][0].isoformat()),
            ("end", PARTITIONS["train"][1].isoformat())])),
        ("target_missing_policy", OrderedDict([
            ("rule", "never_impute_target_labels"),
            ("eligible_only_if_target_observed", True),
            ("excluded_counts_recorded_in",
             "results/eda_training/target_eligibility_by_partition_horizon.csv"),
            ("applies_to", ["train", "validation", "test"]),
        ])),
        ("numeric_missing_policy", OrderedDict([
            ("method", "causal_forward_fill_then_training_median_fallback"),
            ("max_forward_fill_hours", MAX_FORWARD_FILL_H),
            ("direction", "backward_in_time_only_from_origin"),
            ("backward_fill_permitted", False),
            ("interpolation_permitted", False),
            ("centred_statistics_permitted", False),
            ("fallback", "station_specific_training_median"),
            ("fallback_fitted_on", "train"),
            ("applies_to", NUMERIC_COLUMNS),
            ("justification",
             "Median PM2.5 gap is 1 h and the P90 gap is 5 h; 1019 of 1112 "
             "training PM2.5 gaps are <= 6 h. Autocorrelation is 0.752 at "
             "lag 6 but 0.368 at lag 24, so a carried value older than 6 h "
             "asserts information the data does not support."),
        ])),
        ("categorical_missing_policy", OrderedDict([
            ("variables", ["wd"]),
            ("method", "causal_forward_fill_then_explicit_missing_category"),
            ("max_forward_fill_hours", MAX_FORWARD_FILL_H),
            ("numeric_interpolation_permitted", False),
            ("vocabulary_fitted_on", "train"),
            ("vocabulary", eda["wind_direction_categories"]),
            ("reserved_unknown_category", "UNKNOWN"),
            ("explicit_missing_category", "MISSING"),
        ])),
        ("missing_mask_policy", OrderedDict([
            ("emit_observed_mask", True),
            ("mask_variables", NUMERIC_COLUMNS + ["wd"]),
            ("encoding", "1_observed_0_imputed_or_absent"),
            ("mask_precedes_imputation", True),
            ("note", "Masks record the state of the raw value before any "
                     "causal fill, so a model can distinguish a measurement "
                     "from a carried value."),
        ])),
        ("time_since_observed_policy", OrderedDict([
            ("emit", True),
            ("variables", ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]),
            ("units", "hours"),
            ("cap_hours", 168),
            ("value_at_origin_if_never_observed", "cap"),
            ("justification",
             "40% of missing training PM2.5 hours and 60% of missing CO "
             "hours fall in runs longer than 24 h, so gap age varies by "
             "orders of magnitude and must be visible to the model."),
        ])),
        ("scaling_policy", OrderedDict([
            ("scheme", "global_training_robust_scaling"),
            ("formula", "(x - median) / IQR"),
            ("zero_iqr_fallback_formula", "(x - median) / std"),
            ("zero_iqr_variables", [name for name, stat in statistics.items()
                                    if stat["iqr"] == 0]),
            ("zero_iqr_note",
             "RAIN is zero-inflated in training: median, P25 and P75 are all "
             "0.0, so the IQR denominator is zero and the robust formula is "
             "undefined. Such variables use the training standard deviation "
             "instead. A companion binary rain-occurred feature is a Phase-2 "
             "option, not assumed here."),
            ("fitted_on", "training_partition_pooled_across_stations"),
            ("station_specific_scaling", False),
            ("station_specific_rejected_because",
             "Station level differences are the signal the macro "
             "station-horizon metric is designed to compare fairly, and "
             "per-station centring would erase them and make "
             "leave-one-station-out transfer ill-defined."),
            ("target_transform", "none_in_native_ug_m3"),
            ("target_transform_note",
             "MAE is the primary metric and is reported in native units; a "
             "log transform is a Phase-2 option to be decided explicitly, "
             "not assumed here."),
            ("applied_to_validation_and_test", "using_training_statistics_only"),
            ("training_statistics", statistics),
            ("quantile_method", QUANTILE_METHOD),
        ])),
        ("calendar_features", OrderedDict([
            ("frozen", ["origin_hour", "origin_day_of_year", "origin_month",
                        "target_hour", "target_day_of_year", "target_month"]),
            ("cyclic_encoding", ["sin_cos_hour", "sin_cos_day_of_year"]),
            ("target_time_features_permitted", True),
            ("target_time_features_permitted_because",
             "calendar values at t+h are deterministic and knowable at the "
             "origin without observing anything"),
            ("optional_low_priority", ["day_of_week", "weekend"]),
            ("optional_because",
             "Training PM2.5 autocorrelation at a 168 h lag is 0.008, which "
             "gives no support for a weekly cycle; retained only as an "
             "explicit Phase-2 ablation."),
            ("future_observed_weather_permitted", False),
        ])),
        ("station_identity_policy", OrderedDict([
            ("encoding", "learned_embedding_or_one_hot_over_12_training_stations"),
            ("fitted_on", "train"),
            ("must_not_encode_partition_or_future_information", True),
            ("track_b_note",
             "One-hot station identity cannot transfer to an unseen station; "
             "Track B leave-one-station-out will need identity-free or "
             "attribute-based station representation."),
        ])),
        ("sequence_eligibility_rules", OrderedDict([
            ("sample_key", ["station", "target_timestamp", "horizon"]),
            ("rules", [
                "target_timestamp lies inside the assigned partition",
                "target PM2.5 at target_timestamp is observed (never imputed)",
                "forecast_origin = target_timestamp - horizon exists in the "
                "dataset timeline",
                "every input timestamp is <= forecast_origin",
                "the required context window [origin - context + 1, origin] "
                "lies inside the dataset timeline",
                "missing inputs inside the window are handled by the causal "
                "policy above, never by future information",
            ]),
            ("boundary_context_rule",
             "context may extend into the preceding partition when its "
             "timestamps precede the forecast origin; such samples are kept, "
             "not discarded"),
            ("boundary_context_constraint",
             "model parameters and all preprocessing statistics remain "
             "fitted on the training partition only"),
            ("context_dependent_eligibility_finalised_in", "phase_2"),
        ])),
        ("primary_severe_threshold_definition", OrderedDict([
            ("rule", "pooled_training_pm25_p95"),
            ("comparison", "actual_pm25 > threshold"),
            ("derived_from", "training_partition_only"),
            ("quantile_method", QUANTILE_METHOD),
            ("not_a_regulatory_aqi_category", True),
        ])),
        ("primary_severe_threshold_value", eda["pooled_training_p95"]),
        ("station_specific_thresholds", eda["station_training_p95"]),
        ("context_length_candidates", CONTEXT_LENGTH_CANDIDATES),
        ("context_length_optional_control", CONTEXT_LENGTH_OPTIONAL),
        ("context_length_justification",
         "Training PM2.5 autocorrelation falls from 0.965 at 1 h to 0.368 at "
         "24 h, 0.098 at 48 h and -0.004 at 72 h, with 0.008 at 168 h. Usable "
         "own-history memory is effectively exhausted by 48-72 h and there is "
         "no weekly structure, so 24/48/72 h brackets the informative range; "
         "168 h is retained only as an optional long-context control."),
        ("context_length_selected_by_model_performance", False),
        ("model_hyperparameters", None),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    if not TRAINING_EDA.is_file():
        print("run scripts/run_training_eda.py first", file=sys.stderr)
        return 2
    with open(str(TRAINING_EDA), encoding="utf-8") as handle:
        eda = json.load(handle)

    statistics = training_scaling_statistics()
    config = build_config(statistics, eda)
    with open(str(CONFIG), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print("wrote %s" % CONFIG.relative_to(PROJECT_ROOT))
    print("wrote %s" % SCALING_CSV.relative_to(PROJECT_ROOT))

    if args.write_manifest:
        missing = [name for name in (POLICY_DOC, EDA_DOC)
                   if not (PROJECT_ROOT / name).is_file()]
        if missing:
            print("cannot write manifest, missing: %s" % missing,
                  file=sys.stderr)
            return 2
        manifest = OrderedDict([
            ("study", "AirSense V2"),
            ("phase", "phase_1_training_eda_and_preprocessing_contract"),
            ("foundation_manifest_sha256", sha256_of_file(FOUNDATION)),
            ("training_start", eda["training_start"]),
            ("training_end", eda["training_end"]),
            ("training_rows", eda["training_rows"]),
            ("training_stations", len(eda["stations"])),
            ("training_hours_per_station", eda["hours_per_station"]),
            ("target_observed", eda["target_observed"]),
            ("target_missing", eda["target_missing"]),
            ("quantile_method", eda["quantile_method"]),
            ("primary_severe_threshold_definition",
             "pooled_training_pm25_p95"),
            ("primary_severe_threshold_value", eda["pooled_training_p95"]),
            ("station_specific_thresholds", eda["station_training_p95"]),
            ("max_forward_fill_hours", MAX_FORWARD_FILL_H),
            ("context_length_candidates", CONTEXT_LENGTH_CANDIDATES),
            ("context_length_optional_control", CONTEXT_LENGTH_OPTIONAL),
            ("preprocessing_policy_document", POLICY_DOC),
            ("preprocessing_policy_sha256",
             sha256_of_file(PROJECT_ROOT / POLICY_DOC)),
            ("training_eda_document", EDA_DOC),
            ("training_eda_document_sha256",
             sha256_of_file(PROJECT_ROOT / EDA_DOC)),
            ("preprocessing_config_sha256", sha256_of_file(CONFIG)),
            ("training_eda_summary_sha256", sha256_of_file(TRAINING_EDA)),
            ("eda_table_sha256", OrderedDict(
                (name, sha256_of_file(RESULTS / name))
                for name in EDA_TABLES)),
            ("eda_figure_sha256", OrderedDict(
                (name, sha256_of_file(FIGURES / name))
                for name in EDA_FIGURES)),
            ("validation_target_analysis", False),
            ("test_target_analysis", False),
            ("future_aware_imputation_permitted", False),
            ("model_runs", 0),
            ("predictions_generated", 0),
            ("validation_metrics_seen", 0),
            ("test_metrics_seen", 0),
            ("final_test_status", "sealed"),
            ("heavy_dependencies_installed", False),
            ("v1_legacy_modified", False),
            ("note", "Deterministic by construction: no wall-clock "
                     "timestamp and no filesystem mtime is recorded."),
        ])
        with open(str(MANIFEST), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print("wrote %s" % MANIFEST.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
