"""Phase-7 pre-analysis candidate freeze.

Protocol Phase 7, section 1. **Written before any robustness experiment runs.**

Freezes the final candidate set so it cannot be edited after robustness results
are seen. Five candidates, each the strongest representative of its family on
development validation, with its exact artifact hashes, regime, architecture,
parameter count, training configuration and previously recorded validation
result.

Deliberately excluded: iTransformer_R1/R2, SA_R2 and every unsuccessful
variant. SA_R2 is excluded because it is the *control arm* of the Phase-6
paired experiment, not a candidate forecaster.

This artifact contains no new performance value: every number in it is copied
from a frozen earlier-phase artifact.

Usage:
    python3 scripts/build_phase7_candidate_freeze.py
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

ARTIFACTS = PROJECT_ROOT / "artifacts"
OUT = PROJECT_ROOT / "results" / "validation"
PRED = OUT / "predictions"
FREEZE = ARTIFACTS / "phase7_candidate_freeze.json"

SEEDS = [42, 43, 44, 45, 46]
STOCHASTIC = ["GRU_R1", "TCN_R1", "SA_R3"]
DETERMINISTIC = ["B0", "B3_R2"]
CANDIDATES = DETERMINISTIC + STOCHASTIC
EXCLUDED = OrderedDict([
    ("iTransformer_R1", "Phase-5 inverted Transformer; sixth of sixteen on "
                        "the primary metric and beaten by every candidate "
                        "here except B0"),
    ("iTransformer_R2", "co-pollutant variant, worse than iTransformer_R1"),
    ("SA_R2", "the self-only control arm of the Phase-6 paired experiment, "
              "not a candidate forecaster"),
    ("GRU_R2 / TCN_R2 / *_R0 / B1 / B2 / B3_R0 / B3_R1",
     "dominated within their own family on development validation"),
])


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def read_csv(path):
    with open(str(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def main():
    comparison = {r["model"]: r
                  for r in read_csv(OUT / "phase6_model_comparison.csv")}
    severe = {r["model"]: r
              for r in read_csv(OUT / "phase6_severe_metrics.csv")}
    negatives = {r["model"]: r
                 for r in read_csv(OUT / "phase6_negative_predictions.csv")}

    neural = {r["model"]: r
              for r in read_csv(PROJECT_ROOT / "results" / "models" / "neural"
                                / "neural_full_models.csv")}
    spatial = {r["model"]: r
               for r in read_csv(PROJECT_ROOT / "results" / "models"
                                 / "spatiotemporal"
                                 / "spatiotemporal_full_models.csv")}
    phase4_freeze = load_json(ARTIFACTS
                              / "phase4_architecture_selection_freeze.json")
    phase6_freeze = load_json(ARTIFACTS
                              / "phase6_architecture_selection_freeze.json")
    b3_selected = read_csv(PROJECT_ROOT / "results" / "models" / "b3"
                           / "b3_selected_models.csv")

    entries = OrderedDict()
    for name in CANDIDATES:
        entry = OrderedDict([
            ("model", name),
            ("stochastic_training", name in STOCHASTIC),
            ("prediction_file",
             "results/validation/predictions/%s_validation.npy" % name),
            ("prediction_sha256",
             sha256_of_file(PRED / ("%s_validation.npy" % name))),
            ("development_validation", OrderedDict([
                ("macro_station_horizon_MAE",
                 float(comparison[name]["macro_station_horizon_MAE"])),
                ("micro_MAE", float(comparison[name]["micro_MAE"])),
                ("macro_RMSE", float(comparison[name]["macro_RMSE"])),
                ("macro_R2", float(comparison[name]["macro_R2"])),
                ("severe_MAE", float(severe[name]["severe_MAE"])),
                ("severe_underprediction_pct",
                 float(severe[name]["severe_underprediction_pct"])),
                ("negative_predictions", int(negatives[name]["n_negative"])),
                ("rank_of_sixteen",
                 int(comparison[name]["rank_by_primary_metric"])),
                ("value_source", comparison[name]["value_source"]),
            ])),
        ])
        if name == "B0":
            entry.update(OrderedDict([
                ("family", "classical"),
                ("regime", "none - causal persistence"),
                ("architecture", "repeat the last observed PM2.5 at or "
                                 "before the forecast origin"),
                ("parameter_count", 0),
                ("training_configuration", "none; deterministic rule"),
            ]))
        elif name == "B3_R2":
            entry.update(OrderedDict([
                ("family", "classical"),
                ("regime", "R2 - meteorology + PM2.5 + co-pollutant history"),
                ("architecture", "LightGBM gradient boosting on engineered "
                                 "lag and rolling features, one model per "
                                 "horizon"),
                ("parameter_count", None),
                ("per_horizon_models", OrderedDict(
                    ("h%s" % r["horizon"], OrderedDict([
                        ("candidate_id", r["selected_candidate"]),
                        ("model_file", r["model_file"]),
                        ("model_sha256", r["model_sha256"]),
                    ])) for r in b3_selected if r["regime"] == "R2")),
                ("training_configuration", "frozen Phase-3 B3 grid and "
                                           "selection rule"),
            ]))
        elif name in ("GRU_R1", "TCN_R1"):
            architecture = name.split("_")[0]
            config = load_json(PROJECT_ROOT / "results" / "models" / "neural"
                               / ("%s_config.json" % name))
            entry.update(OrderedDict([
                ("family", "neural"),
                ("regime", "R1 - meteorology + PM2.5 history"),
                ("architecture", "single-layer unidirectional GRU"
                                 if architecture == "GRU"
                                 else "causal dilated temporal convnet, "
                                      "4 residual blocks"),
                ("parameter_count", int(neural[name]["parameter_count"])),
                ("weight_sha256", neural[name]["weight_sha256"]),
                ("config_sha256", neural[name]["config_sha256"]),
                ("capacity", config["capacity"]),
                ("learning_rate", config["learning_rate"]),
                ("selected_candidate",
                 phase4_freeze["selected"][architecture]["candidate_id"]),
                ("training_configuration", config["training"]),
            ]))
        else:
            config = load_json(PROJECT_ROOT / "results" / "models"
                               / "spatiotemporal" / ("%s_config.json" % name))
            entry.update(OrderedDict([
                ("family", "spatiotemporal"),
                ("regime", "R3 - R2 at every station plus synchronized "
                           "cross-station history"),
                ("architecture", "AirSense Station-Attention Forecaster: "
                                 "shared GRU encoder, multi-head attention "
                                 "across the 12 stations, residual fusion"),
                ("parameter_count", int(spatial[name]["parameter_count"])),
                ("weight_sha256", spatial[name]["weight_sha256"]),
                ("config_sha256", spatial[name]["config_sha256"]),
                ("hidden_size", config["hidden_size"]),
                ("learning_rate", config["learning_rate"]),
                ("selected_candidate",
                 phase6_freeze["selected"]["candidate_id"]),
                ("station_order", config["station_order"]),
                ("external_coordinates_used", False),
                ("training_configuration", config["training"]),
            ]))
        entries[name] = entry

    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase7_candidate_freeze"),
        ("stage", "A_before_any_robustness_experiment"),
        ("purpose", "Freeze the final candidate set before robustness "
                    "results exist, so the set cannot be edited after seeing "
                    "them."),
        ("candidates", entries),
        ("excluded_with_reason", EXCLUDED),
        ("robustness_seeds", SEEDS),
        ("stochastic_models", STOCHASTIC),
        ("deterministic_models", DETERMINISTIC),
        ("selection_rule", OrderedDict([
            ("primary", "development validation macro station-horizon MAE"),
            ("secondary", ["severe MAE", "macro RMSE", "macro R2"]),
            ("must_consider", ["mean across seeds", "seed variance",
                               "severe-tail behaviour"]),
            ("forbidden", "selecting on a single best seed"),
        ])),
        ("frozen_inputs_sha256", OrderedDict([
            ("phase6_model_comparison",
             sha256_of_file(OUT / "phase6_model_comparison.csv")),
            ("phase6_severe_metrics",
             sha256_of_file(OUT / "phase6_severe_metrics.csv")),
            ("phase4_architecture_selection_freeze", sha256_of_file(
                ARTIFACTS / "phase4_architecture_selection_freeze.json")),
            ("phase6_architecture_selection_freeze", sha256_of_file(
                ARTIFACTS / "phase6_architecture_selection_freeze.json")),
            ("verification_tooling_registry", sha256_of_file(
                ARTIFACTS / "verification_tooling_registry.json")),
        ])),
        ("upstream_manifest_sha256", OrderedDict(
            (name, sha256_of_file(ARTIFACTS / name)) for name in
            ("v2_foundation_manifest.json", "v2_phase1_manifest.json",
             "v2_phase2_manifest.json", "v2_phase3_manifest.json",
             "v2_phase4_manifest.json", "v2_phase5_manifest.json",
             "v2_phase6_manifest.json"))),
        ("severe_threshold", 244.0),
        ("context_hours", 48),
        ("candidate_set_may_change_after_results", False),
        ("architecture_search_permitted", False),
        ("feature_changes_permitted", False),
        ("test_predictions_generated", 0),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
    ])
    with open(str(FREEZE), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print("wrote %s" % FREEZE.relative_to(PROJECT_ROOT))
    print("sha256 %s" % sha256_of_file(FREEZE))
    for name, entry in entries.items():
        print("  %-8s %-15s macro MAE %10.6f  severe %8.2f  rank %2d"
              % (name, entry["family"],
                 entry["development_validation"]["macro_station_horizon_MAE"],
                 entry["development_validation"]["severe_MAE"],
                 entry["development_validation"]["rank_of_sixteen"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
