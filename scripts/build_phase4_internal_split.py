"""Freeze the Phase-4 internal split and the neural feature schema.

Protocol Phase 4, before any candidate fit.

The internal core/tuning split lives entirely inside the frozen Phase-2
training partition. It is a training procedure, not a new sample universe:
`train.csv`, `validation.csv`, `test.csv` and
`sample_universe_manifest.json` are untouched.

Usage:
    .venv-v2/bin/python scripts/build_phase4_internal_split.py
"""

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import PARTITION_BOUNDS, STATIONS   # noqa: E402
from src.models.neural_data import (                            # noqa: E402
    INTERNAL_CORE, INTERNAL_TUNING, internal_mask, load_sample_index)
from src.models.neural_features import feature_schema           # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
ARTIFACTS = PROJECT_ROOT / "artifacts"
HORIZONS = [1, 6, 12, 24]


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def write_json(path, payload):
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    stations, horizons, targets = load_sample_index(PROCESSED, "train")
    core = internal_mask(targets, INTERNAL_CORE)
    tuning = internal_mask(targets, INTERNAL_TUNING)
    if int((core & tuning).sum()) != 0:
        raise SystemExit("internal split overlaps")
    if int(core.sum() + tuning.sum()) != targets.size:
        raise SystemExit("internal split does not partition training")

    counts = OrderedDict()
    for label, mask in (("core", core), ("tuning", tuning)):
        by_station_horizon = OrderedDict()
        for station in STATIONS:
            for horizon in HORIZONS:
                selector = mask & (stations == station) & (horizons == horizon)
                by_station_horizon["%s|%d" % (station, horizon)] = int(
                    selector.sum())
        counts[label] = OrderedDict([
            ("total", int(mask.sum())),
            ("by_horizon", OrderedDict(
                (str(h), int((mask & (horizons == h)).sum()))
                for h in HORIZONS)),
            ("by_station", OrderedDict(
                (s, int((mask & (stations == s)).sum())) for s in STATIONS)),
            ("by_station_horizon", by_station_horizon),
        ])

    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase4_internal_split"),
        ("purpose",
         "Chronological model-selection split used only to choose neural "
         "architecture capacity and learning rate. Phase 3 already spent the "
         "development validation partition on classical model selection; "
         "tuning every later architecture on that same period would compound "
         "selection optimism. Hyperparameters are therefore chosen inside "
         "training, and the selected configuration is refitted on the "
         "complete frozen training universe before external validation is "
         "touched at all."),
        ("partition_rule", "target_timestamp"),
        ("boundary_context_rule",
         "context for an early internal-tuning sample may extend back into "
         "the training-core period; both lie inside the frozen training "
         "partition, so no external data is involved"),
        ("chosen_by", "calendar logic, not model performance"),
        ("internal_core", OrderedDict([
            ("start", INTERNAL_CORE[0].isoformat()),
            ("end", INTERNAL_CORE[1].isoformat()),
            ("annual_cycle", "complete March-to-February"),
        ])),
        ("internal_tuning", OrderedDict([
            ("start", INTERNAL_TUNING[0].isoformat()),
            ("end", INTERNAL_TUNING[1].isoformat()),
            ("annual_cycle", "complete March-to-February"),
            ("strictly_after_core", True),
        ])),
        ("regime_independence",
         "counts are identical for R0, R1 and R2: the regime changes which "
         "channels a model may read, never which samples exist"),
        ("regimes", ["R0", "R1", "R2"]),
        ("counts", counts),
        ("phase2_sample_universe_rewritten", False),
        ("phase2_sample_universe_manifest_sha256",
         sha256_of_file(ARTIFACTS / "sample_universe_manifest.json")),
        ("train_index_sha256",
         sha256_of_file(PROCESSED / "sample_index" / "train.csv")),
        ("external_validation_used_for_selection", False),
        ("final_test_status", "sealed"),
    ])
    write_json(ARTIFACTS / "phase4_internal_split.json", payload)
    print("wrote artifacts/phase4_internal_split.json  %s"
          % sha256_of_file(ARTIFACTS / "phase4_internal_split.json")[:16])

    schema = feature_schema()
    schema["frozen_before"] = "any internal candidate fit"
    schema["source_of_truth"] = [
        "artifacts/information_regime_schema.json",
        "results/dataset_build/channel_schema.csv",
    ]
    write_json(ARTIFACTS / "neural_feature_schema.json", schema)
    print("wrote artifacts/neural_feature_schema.json  %s"
          % sha256_of_file(ARTIFACTS / "neural_feature_schema.json")[:16])
    print("core %d | tuning %d | dynamic R0/R1/R2 %s | static %d"
          % (counts["core"]["total"], counts["tuning"]["total"],
             list(schema["dynamic_channel_counts"].values()),
             schema["static_channel_count"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
