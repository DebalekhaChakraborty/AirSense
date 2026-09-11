"""Validate the AirSense V2 Phase-6 cross-station spatiotemporal modelling.

Read-only. Trains nothing, predicts nothing new, repairs nothing. Exits 0 only
if every gate passes.

Phase 6 is a **paired information experiment**: one architecture, two station
attention masks. Several gates below exist to prove the pairing was honest -
identical parameter counts, identical seeded initialisation, a self-only mask
that provably cannot see another station, and a cross-station mask that
provably can.

Usage:
    .venv-v2/bin/python scripts/validate_phase6.py
"""

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (                                 # noqa
    PARTITION_BOUNDS, STATIONS, index_of)
from src.data.target_history import TargetAccessError                # noqa
from src.evaluation import metrics as M                              # noqa
from src.models import station_attention_grid as grid                # noqa
from src.models.neural_data import INTERNAL_CORE, INTERNAL_TUNING    # noqa
from src.models.neural_features import (                             # noqa
    CONTEXT_HOURS, dynamic_channel_names, static_channel_names)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
PRED = PROJECT_ROOT / "results" / "validation" / "predictions"
OUT = PROJECT_ROOT / "results" / "validation"
SPATIAL = PROJECT_ROOT / "results" / "models" / "spatiotemporal"
ARTIFACTS = PROJECT_ROOT / "artifacts"

MODELS = ["SA_R2", "SA_R3"]
EARLIER = ["B0", "B1", "B2", "B3_R0", "B3_R1", "B3_R2", "GRU_R0", "GRU_R1",
           "GRU_R2", "TCN_R0", "TCN_R1", "TCN_R2", "iTransformer_R1",
           "iTransformer_R2"]
HORIZONS = [1, 6, 12, 24]
EXPECTED_SAMPLES = 413148
# Scanning for the *word* "coordinate" is unsound: Phase-6 source mentions it
# only to declare its absence ("external_coordinates": False). The gate must
# detect coordinate *data* instead - a numeric latitude/longitude binding, or
# a geospatial library - which a negative declaration can never trip.
COORDINATE_ASSIGNMENT = re.compile(
    r"\b(lat|latitude|lon|lng|longitude|easting|northing)\b\s*[=:]\s*"
    r"[-+]?\d", re.IGNORECASE)
GEO_IMPORTS = re.compile(
    r"^\s*(?:import|from)\s+(geopy|shapely|pyproj|haversine|geopandas|"
    r"cartopy|osmnx)\b", re.MULTILINE)
TSFM_PACKAGES = ("chronos", "chronos_forecasting", "timesfm", "uni2ts",
                 "transformers", "gluonts")


class Report(object):
    def __init__(self):
        self.rows = []

    def add(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        return ok

    def render(self):
        width = 70
        for label, ok, detail in self.rows:
            status = "PASS" if ok else "FAIL"
            dots = "." * max(3, width - len(label) - len(status) - 2)
            print("%s %s %s" % (label, dots, status))
            if detail and not ok:
                print("      %s" % detail)

    def failed(self):
        return [row for row in self.rows if not row[1]]


def sha256_of_file(path):
    return hashlib.sha256(open(str(path), "rb").read()).hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(str(path), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_index():
    stations, horizons, targets = [], [], []
    for row in read_csv(PROCESSED / "sample_index" / "validation.csv"):
        stations.append(row["station"])
        horizons.append(int(row["horizon_hours"]))
        targets.append(int(row["target_row_index"]))
    return (np.array(stations), np.array(horizons, dtype=np.int32),
            np.array(targets, dtype=np.int64))


def upstream(report):
    for name, interpreter in (("validate_foundation.py", "python3"),
                              ("validate_phase1.py", "python3"),
                              ("validate_phase2.py", "python3"),
                              ("validate_phase3.py", sys.executable),
                              ("validate_phase4.py", sys.executable),
                              ("validate_phase5.py", sys.executable)):
        result = subprocess.run(
            [interpreter, str(PROJECT_ROOT / "scripts" / name)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report.add("Upstream %s passes" % name, result.returncode == 0,
                   "exit %d" % result.returncode)

    registry = load_json(ARTIFACTS / "verification_tooling_registry.json")
    drift = [name for name, entry in registry["validators"].items()
             if sha256_of_file(PROJECT_ROOT / name) != entry["sha256"]]
    report.add("Verification tooling matches the registry", not drift,
               "changed: %s" % drift)

    manifest = load_json(ARTIFACTS / "v2_phase6_manifest.json")
    for key, path in (("foundation_manifest_sha256",
                       "artifacts/v2_foundation_manifest.json"),
                      ("phase1_manifest_sha256",
                       "artifacts/v2_phase1_manifest.json"),
                      ("phase2_manifest_sha256",
                       "artifacts/v2_phase2_manifest.json"),
                      ("phase3_manifest_sha256",
                       "artifacts/v2_phase3_manifest.json"),
                      ("phase4_manifest_sha256",
                       "artifacts/v2_phase4_manifest.json"),
                      ("phase5_manifest_sha256",
                       "artifacts/v2_phase5_manifest.json")):
        report.add("Upstream manifest unchanged: %s" % Path(path).name,
                   manifest[key] == sha256_of_file(PROJECT_ROOT / path))

    stale = []
    for phase, groups in (
            ("v2_phase5_manifest.json",
             (("prediction_array_sha256", "results/validation/predictions/"),
              ("metric_table_sha256", "results/validation/"),
              ("figure_sha256", "figures/"),
              ("candidate_table_sha256", "results/models/itransformer/"),
              ("model_config_and_weight_sha256",
               "results/models/itransformer/"))),
            ("v2_phase4_manifest.json",
             (("prediction_array_sha256", "results/validation/predictions/"),
              ("metric_table_sha256", "results/validation/"),
              ("figure_sha256", "figures/"),
              ("candidate_table_sha256", "results/models/neural/"),
              ("model_config_and_weight_sha256",
               "results/models/neural/")))):
        payload = load_json(ARTIFACTS / phase)
        for group, base in groups:
            for name, digest in payload[group].items():
                path = PROJECT_ROOT / (base + name)
                if not path.is_file() or sha256_of_file(path) != digest:
                    stale.append("%s:%s" % (phase, name))
    report.add("Phase-4 and Phase-5 evidence unchanged", not stale,
               "drifted: %s" % stale[:5])


def contract(report):
    schema = load_json(ARTIFACTS / "neural_feature_schema.json")
    report.add("Context length is unchanged at 48",
               schema["context_hours"] == CONTEXT_HOURS == 48)
    report.add("Severe threshold is unchanged at 244.0",
               M.SEVERE_THRESHOLD == 244.0)
    manifest2 = load_json(ARTIFACTS / "v2_phase2_manifest.json")
    report.add("Validation sample index unchanged since Phase 2",
               manifest2["sample_index_sha256"]["validation"]
               == sha256_of_file(PROCESSED / "sample_index"
                                 / "validation.csv"))
    report.add("Training sample index unchanged since Phase 2",
               manifest2["sample_index_sha256"]["train"]
               == sha256_of_file(PROCESSED / "sample_index" / "train.csv"))
    universe = load_json(ARTIFACTS / "sample_universe_manifest.json")
    report.add("Phase-2 sample universe was not rewritten",
               universe["validation_target_values_materialized"] is False
               and universe["test_target_values_materialized"] is False)

    freeze = load_json(ARTIFACTS
                       / "phase6_architecture_selection_freeze.json")
    configs = {name: load_json(SPATIAL / ("%s_config.json" % name))
               for name in MODELS}
    report.add("Station order is exactly the frozen 12",
               all(c["station_order"] == list(STATIONS) for c in
                   configs.values())
               and len(STATIONS) == 12
               and load_json(ARTIFACTS / "v2_phase6_manifest.json")
               ["group_schema"]["station_order"] == list(STATIONS))
    report.add("Dynamic channels are the frozen R2 schema",
               all(c["dynamic_channels"] == len(dynamic_channel_names("R2"))
                   for c in configs.values()))
    report.add("Static channels are the frozen conditioning",
               all(c["static_channels"] == len(static_channel_names())
                   for c in configs.values())
               and freeze["static_schema_channels"] == static_channel_names())

    split = load_json(ARTIFACTS / "phase4_internal_split.json")
    core_start = datetime.fromisoformat(split["internal_core"]["start"])
    tune_end = datetime.fromisoformat(split["internal_tuning"]["end"])
    report.add("Internal selection reused the Phase-4 training-only split",
               INTERNAL_CORE[0] == core_start
               and INTERNAL_TUNING[1] == tune_end
               and freeze["internal_split_reused_from_phase4"] is True)
    report.add("Internal selection lies entirely inside training",
               core_start >= PARTITION_BOUNDS["train"][0]
               and tune_end <= PARTITION_BOUNDS["train"][1]
               and tune_end < PARTITION_BOUNDS["validation"][0])
    report.add("External validation did not choose hyperparameters",
               freeze["external_validation_used_for_selection"] is False
               and freeze["external_phase6_validation_metrics_seen"] == 0
               and freeze["test_metrics_seen"] == 0)


def grid_and_pairing(report):
    candidates = grid.candidates()
    report.add("The grid is exactly the four frozen candidates",
               [(c["candidate_id"], c["hidden_size"], c["learning_rate"])
                for c in candidates]
               == [("SA_C01", 64, 0.0003), ("SA_C02", 64, 0.001),
                   ("SA_C03", 128, 0.0003), ("SA_C04", 128, 0.001)])
    report.add("Training budget fixed at 8 epochs, no early stopping",
               all(c["epochs"] == 8 and c["early_stopping"] is False
                   and c["loss"] == "L1" and c["optimizer"] == "AdamW"
                   for c in candidates))
    report.add("Only the two station-attention modes were evaluated",
               grid.REGIMES == ["R2", "R3"])

    metrics = read_csv(SPATIAL / "internal_candidate_metrics.csv")
    counts = Counter(row["candidate_id"] for row in metrics)
    report.add("8 internal candidate fits, 2 modes per candidate",
               len(metrics) == 8 and set(counts.values()) == {2})
    report.add("Both modes hold identical capacity at every candidate",
               all(len({int(r["parameter_count"]) for r in metrics
                        if r["candidate_id"] == cid}) == 1 for cid in counts))

    selection = read_csv(SPATIAL / "internal_candidate_selection.csv")
    pool = [{"candidate_id": r["candidate_id"],
             "hidden_size": int(r["hidden_size"]),
             "learning_rate": float(r["learning_rate"]),
             "mean_macro_mae": float(r["mean_macro_station_horizon_MAE"]),
             "mean_macro_rmse": float(r["mean_macro_RMSE"])}
            for r in selection]
    recomputed = grid.select(pool)["candidate_id"]
    marked = [r["candidate_id"] for r in selection if r["selected"] == "true"]
    report.add("Selected candidate matches the frozen selection rule",
               marked == [recomputed],
               "marked %s, rule gives %s" % (marked, recomputed))
    report.add("Selection score is the equal-weight mean of both modes",
               not [c for c in counts
                    if abs(float(np.mean(
                        [float(r["internal_macro_station_horizon_MAE"])
                         for r in metrics if r["candidate_id"] == c]))
                        - float([r["mean_macro_station_horizon_MAE"]
                                 for r in selection
                                 if r["candidate_id"] == c][0])) > 1e-6])

    freeze = load_json(ARTIFACTS
                       / "phase6_architecture_selection_freeze.json")
    configs = {name: load_json(SPATIAL / ("%s_config.json" % name))
               for name in MODELS}
    report.add("The same selected configuration is used for both modes",
               freeze["same_configuration_used_for_both_modes"] is True
               and len({c["hidden_size"] for c in configs.values()}) == 1
               and len({c["learning_rate"] for c in configs.values()}) == 1
               and len({c["candidate_id"] for c in configs.values()}) == 1)
    report.add("SA_R2 and SA_R3 hold identical parameter counts",
               len({c["parameter_count"] for c in configs.values()}) == 1,
               "counts %s" % {c["parameter_count"] for c in configs.values()})
    report.add("Full models trained on the complete training universe from a "
               "clean initialisation",
               all(c["trained_on"]
                   == "complete frozen Phase-2 training universe"
                   and c["initialised_from"].startswith("clean seeded")
                   for c in configs.values()))
    report.add("Grouped batch size was frozen before candidate fitting",
               freeze["grouped_batch_size"]
               == load_json(ARTIFACTS / "phase6_runtime_benchmark.json")
               ["frozen_grouped_batch_size"])
    benchmark = load_json(ARTIFACTS / "phase6_runtime_benchmark.json")
    report.add("Runtime benchmark read no external validation data",
               benchmark["external_validation_read"] is False
               and benchmark["projection"]["within_runtime_gate"] is True)


def masks_and_causality(report):
    """The paired design is asserted numerically, not taken on trust."""
    from src.models.spatial_grouping import (
        GuardedNetworkHistory, NetworkBundle, build_groups)
    from src.models.station_attention_forecaster import (
        StationAttentionForecaster)

    freeze = load_json(ARTIFACTS
                       / "phase6_architecture_selection_freeze.json")
    hidden = int(freeze["selected"]["hidden_size"])
    channels = len(dynamic_channel_names("R2"))
    statics = len(static_channel_names())

    def build(cross_station, seed=42):
        torch.manual_seed(seed)
        return StationAttentionForecaster(
            channels, statics, hidden_size=hidden,
            heads=grid.FIXED["attention_heads"],
            attention_dropout=grid.FIXED["attention_dropout"],
            head_hidden=grid.FIXED["head_hidden"],
            head_dropout=grid.FIXED["head_dropout"],
            cross_station=cross_station)

    self_only = build(False).eval()
    cross = build(True).eval()
    report.add("SA_R2 mask is strictly self-only",
               bool(torch.equal(self_only.self_only_mask,
                                ~torch.eye(12, dtype=torch.bool))))
    report.add("SA_R3 applies no station mask at all",
               cross.cross_station is True and self_only.cross_station
               is False)
    report.add("Both modes initialise identically from seed 42",
               all(torch.equal(a, b) for a, b in
                   zip(self_only.state_dict().values(),
                       cross.state_dict().values())))
    report.add("Both modes hold the same live parameter count",
               self_only.parameter_count() == cross.parameter_count())

    torch.manual_seed(11)
    sequence = torch.randn(2, 12, CONTEXT_HOURS, channels)
    static = torch.randn(2, 12, statics)
    with torch.no_grad():
        base_self = self_only(sequence, static)
        base_cross = cross(sequence, static)
    perturbed = sequence.clone()
    for station in range(12):
        if station != 3:
            perturbed[:, station] += 7.0
    with torch.no_grad():
        moved_self = self_only(perturbed, static)
        moved_cross = cross(perturbed, static)
    report.add("SA_R2 cannot use any other station's history",
               bool(torch.equal(base_self[:, 3], moved_self[:, 3])))
    report.add("SA_R3 does respond to other stations' history",
               not bool(torch.equal(base_cross[:, 3], moved_cross[:, 3])))
    with torch.no_grad():
        weights = self_only.attention_weights(sequence)
    report.add("SA_R2 attention is exactly the identity",
               bool(torch.allclose(weights, torch.eye(12).expand(2, 12, 12),
                                   atol=1e-6)))

    history = GuardedNetworkHistory(PROJECT_ROOT, unseal_validation=True)
    groups = build_groups(PROCESSED, "validation")
    bundle = NetworkBundle(groups, history)
    rows = np.array([0, 5000, 20000])
    sequence, _ = bundle.batch(rows)
    names = dynamic_channel_names("R2")
    pm25 = names.index("value:PM2.5")
    problems = []
    for offset, row in enumerate(rows):
        origin = int(groups.origins[row])
        for index, station in enumerate(STATIONS):
            guarded = history.guarded_pm25_window(station, origin)
            if not np.allclose(sequence[offset, index, :, pm25], guarded,
                               equal_nan=True):
                problems.append("%s at origin %d" % (station, origin))
    report.add("Every station's PM2.5 window equals the guarded accessor "
               "ending at the shared origin", not problems,
               "; ".join(problems[:3]))
    sealed = False
    try:
        history.windows([index_of(PARTITION_BOUNDS["test"][0])])
    except TargetAccessError:
        sealed = True
    report.add("The network history refuses a sealed test origin", sealed)


def determinism(report):
    directory = SPATIAL / "determinism"
    problems = []
    for name in MODELS:
        repeat = directory / ("%s_validation.npy" % name)
        primary = PRED / ("%s_validation.npy" % name)
        if not repeat.is_file():
            problems.append("%s: no determinism refit" % name)
        elif sha256_of_file(repeat) != sha256_of_file(primary):
            deviation = float(np.abs(
                np.load(str(primary)).astype(np.float64)
                - np.load(str(repeat)).astype(np.float64)).max())
            problems.append("%s: predictions differ, max deviation %.10f"
                            % (name, deviation))
    report.add("Determinism refit reproduces every prediction array byte for "
               "byte", not problems, "; ".join(problems))

    problems = []
    for name in MODELS:
        repeat = directory / ("%s.safetensors" % name)
        primary = SPATIAL / ("%s.safetensors" % name)
        if not repeat.is_file():
            problems.append("%s: no determinism weight file" % name)
        elif sha256_of_file(repeat) != sha256_of_file(primary):
            problems.append("%s: weights differ" % name)
    report.add("Determinism refit reproduces every weight file byte for byte",
               not problems, "; ".join(problems))

    manifest = load_json(ARTIFACTS / "v2_phase6_manifest.json")
    report.add("Manifest records two full models and two determinism refits",
               manifest["full_models"] == 2
               and manifest["determinism_refits"] == 2
               and manifest["internal_candidate_fits"] == 8
               and manifest["seeds_used"] == [42])


def coordinates_and_prohibitions(report):
    offenders = []
    sources = ("src/models/spatial_grouping.py",
                     "src/models/station_attention_forecaster.py",
                     "src/models/station_attention_grid.py",
                     "src/models/spatial_training.py",
                     "scripts/run_phase6_benchmark.py",
                     "scripts/run_phase6_internal_selection.py",
                     "scripts/run_phase6_full_refit.py",
               "scripts/build_phase6_report.py")
    for relative in sources:
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        for match in COORDINATE_ASSIGNMENT.finditer(text):
            offenders.append("%s:%s" % (relative, match.group(0).strip()))
        if GEO_IMPORTS.search(text):
            offenders.append("%s: geospatial import" % relative)
    report.add("No coordinate value or geospatial library enters Phase-6 "
               "code (%d files)" % len(sources),
               not offenders, "found: %s" % offenders[:5])

    data_files = [str(p.relative_to(PROJECT_ROOT))
                  for p in sorted((PROJECT_ROOT / "data").rglob("*"))
                  if p.is_file()
                  and any(w in p.name.lower() for w in
                          ("coord", "latlon", "geo", "position"))]
    report.add("No station coordinate data file exists", not data_files,
               "found: %s" % data_files[:5])
    manifest = load_json(ARTIFACTS / "v2_phase6_manifest.json")
    report.add("Manifest declares no coordinates and no derived graph",
               manifest["R3_uses_external_coordinates"] is False
               and manifest["validation_derived_graph"] is False
               and manifest["R3_uses_future_observed_variables"] is False)
    configs = {name: load_json(SPATIAL / ("%s_config.json" % name))
               for name in MODELS}
    report.add("Model configs declare no coordinates and no pretrained "
               "weights",
               all(c["external_coordinates_used"] is False
                   and c["validation_derived_graph"] is False
                   and c["pretrained_weights_used"] is False
                   for c in configs.values()))
    report.add("Outputs are declared unconstrained",
               all(c["output_constrained"] is False
                   for c in configs.values()))
    negatives = read_csv(OUT / "phase6_negative_predictions.csv")
    report.add("No prediction was clipped",
               not [r for r in negatives if r["clipped"] != "false"])

    for name in TSFM_PACKAGES:
        try:
            __import__(name)
            report.add("%s is absent" % name, False, "installed")
        except ImportError:
            report.add("%s is absent" % name, True)

    forbidden = ("graph_wavenet", "gwnet", "dcrnn", "stgcn", "astgcn")
    hits = [str(p.relative_to(PROJECT_ROOT))
            for p in sorted((PROJECT_ROOT / "src").rglob("*.py"))
            if any(word in p.name.lower() for word in forbidden)]
    report.add("No Graph WaveNet, DCRNN, STGCN or ASTGCN implementation",
               not hits, "found: %s" % hits)

    test_artifacts = [str(p.relative_to(PROJECT_ROOT))
                      for p in sorted(PRED.glob("*test*"))
                      + sorted(SPATIAL.rglob("*test*"))]
    report.add("No test prediction artifact exists", not test_artifacts,
               "found: %s" % test_artifacts)
    summary = load_json(ARTIFACTS / "phase6_validation_summary.json")
    receipt = load_json(ARTIFACTS
                        / "phase6_external_validation_receipt.json")
    report.add("Summary and receipt record a sealed test",
               summary["final_test_status"] == "sealed"
               and summary["test_predictions_generated"] == 0
               and summary["test_metrics_seen"] == 0
               and receipt["final_test_status"] == "sealed"
               and receipt["external_phase6_metrics_seen"] == 0)
    report.add("Receipt pins the selection freeze that preceded it",
               receipt["phase6_architecture_selection_freeze_sha256"]
               == sha256_of_file(
                   ARTIFACTS / "phase6_architecture_selection_freeze.json"))

    result = subprocess.run(["git", "rev-parse", "legacy"],
                            cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    live = result.stdout.decode().strip()
    report.add("V1 legacy branch is unchanged",
               live == manifest.get("legacy_head") and bool(live),
               "live %s recorded %s" % (live[:12],
                                        str(manifest.get("legacy_head"))[:12]))
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                            cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    report.add("Nothing is staged; Phase-6 changes remain unstaged",
               not staged.stdout.decode().strip())


def coverage_and_metrics(report, station, horizon):
    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    predictions = {}
    problems = []
    for name in MODELS:
        path = PRED / ("%s_validation.npy" % name)
        if not path.is_file():
            problems.append("%s missing" % name)
            continue
        values = np.load(str(path)).astype(np.float64)
        predictions[name] = values
        if values.size != EXPECTED_SAMPLES:
            problems.append("%s has %d predictions" % (name, values.size))
        elif not np.isfinite(values).all():
            problems.append("%s has non-finite predictions" % name)
    report.add("SA_R2 predicts exactly 413,148 canonical samples",
               "SA_R2" in predictions
               and predictions["SA_R2"].size == EXPECTED_SAMPLES)
    report.add("SA_R3 predicts exactly 413,148 canonical samples",
               "SA_R3" in predictions
               and predictions["SA_R3"].size == EXPECTED_SAMPLES)
    report.add("No validation sample was dropped by grouping", not problems,
               "; ".join(problems[:3]))
    report.add("Both modes cover exactly the same samples",
               station.size == EXPECTED_SAMPLES
               and predictions["SA_R2"].size == predictions["SA_R3"].size)

    comparison = {r["model"]: r
                  for r in read_csv(OUT / "phase6_model_comparison.csv")}
    cells = read_csv(OUT / "phase6_metrics_by_station_horizon.csv")
    problems = []
    for name in MODELS:
        column = "SA_R2_MAE" if name == "SA_R2" else "SA_R3_MAE"
        values = [float(r[column]) for r in cells]
        if len(values) != 48:
            problems.append("%s has %d cells" % (name, len(values)))
            continue
        if abs(float(np.mean(values))
               - float(comparison[name]["macro_station_horizon_MAE"])) > 1e-6:
            problems.append("%s macro mismatch" % name)
    report.add("Primary H4 metric reconciles from the 48 cells", not problems,
               "; ".join(problems[:3]))

    h4 = {r["scope"]: r for r in read_csv(OUT / "phase6_h4_summary.csv")}
    left = float(comparison["SA_R2"]["macro_station_horizon_MAE"])
    right = float(comparison["SA_R3"]["macro_station_horizon_MAE"])
    # Both sides are published at six decimals, so a difference of rounded
    # values and a rounded difference may legitimately differ by one unit in
    # the last place. The exact reconciliation is done from the prediction
    # arrays in independent().
    report.add("H4 primary gain reconciles with the comparison table",
               abs((left - right)
                   - float(h4["overall"]["absolute_MAE_improvement"]))
               <= 2e-6)
    report.add("Earlier-phase values are reused, not recomputed",
               all(comparison[n]["value_source"] != "phase6"
                   for n in EARLIER)
               and all(comparison[n]["value_source"] == "phase6"
                       for n in MODELS))
    frozen5 = {r["model"]: r
               for r in read_csv(OUT / "phase5_model_comparison.csv")}
    drift = [n for n in EARLIER
             if abs(float(frozen5[n]["macro_station_horizon_MAE"])
                    - float(comparison[n]["macro_station_horizon_MAE"]))
             > 1e-12]
    report.add("Frozen earlier-phase values are byte-faithful", not drift,
               "drifted: %s" % drift)

    severe = {r["model"]: r
              for r in read_csv(OUT / "phase6_severe_metrics.csv")}
    mask = actual > 244.0
    problems = []
    for name in MODELS:
        recomputed = float(np.abs(actual[mask]
                                  - predictions[name][mask]).mean())
        if abs(recomputed - float(severe[name]["severe_MAE"])) > 1e-6:
            problems.append(name)
    report.add("Severe metrics reconcile at the frozen 244.0 threshold",
               not problems, "mismatched: %s" % problems)
    report.add("Severe threshold was not re-derived from validation",
               all(int(severe[n]["severe_n"]) == int(severe["B0"]["severe_n"])
                   for n in MODELS))

    overall = read_csv(SPATIAL / "attention_matrix_overall.csv")
    matrix = np.array([[float(row[s]) for s in STATIONS] for row in overall])
    report.add("Attention rows normalise to one",
               matrix.shape == (12, 12)
               and bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-4)),
               "row sums %s" % np.round(matrix.sum(axis=1), 5)[:4])
    report.add("Attention diagnostics cover all 12 x 12 directed pairs",
               len(read_csv(OUT
                            / "phase6_attention_vs_training_correlation.csv"))
               == 132)


def independent(report, station, horizon, target):
    """Seventeen recomputations through a path separate from the reporting."""
    from safetensors.torch import load_file

    from src.models.spatial_grouping import (
        GuardedNetworkHistory, NetworkBundle, build_groups, group_id)
    from src.models.station_attention_forecaster import (
        StationAttentionForecaster)

    freeze = load_json(ARTIFACTS
                       / "phase6_architecture_selection_freeze.json")
    hidden = int(freeze["selected"]["hidden_size"])
    channels = len(dynamic_channel_names("R2"))
    statics = len(static_channel_names())

    train_groups = build_groups(PROCESSED, "train")
    report.add("Independent: one grouped training example reconciles",
               train_groups.sample_total() == 821184
               and train_groups.origins[0]
               == train_groups.target_indices[0] - train_groups.horizons[0])
    groups = build_groups(PROCESSED, "validation")
    report.add("Independent: one grouped validation example reconciles",
               groups.sample_total() == EXPECTED_SAMPLES
               and group_id("validation", int(groups.target_indices[0]),
                            int(groups.horizons[0])).startswith("validation|"))

    history = GuardedNetworkHistory(PROJECT_ROOT, unseal_validation=True)
    bundle = NetworkBundle(groups, history)
    row = 12345
    origin = int(groups.origins[row])
    sequence, static = bundle.batch(np.array([row]))
    names = dynamic_channel_names("R2")
    pm25 = names.index("value:PM2.5")
    ends = []
    for index, name in enumerate(STATIONS):
        guarded = history.guarded_pm25_window(name, origin)
        ends.append(np.allclose(sequence[0, index, :, pm25], guarded,
                                equal_nan=True))
    report.add("Independent: all 12 windows end at the same shared origin",
               all(ends) and sequence.shape[2] == CONTEXT_HOURS)
    beyond = False
    try:
        history.windows([index_of(PARTITION_BOUNDS["test"][0])])
    except TargetAccessError:
        beyond = True
    report.add("Independent: no source station input may exceed the origin",
               beyond)

    reloaded = {}
    for name in MODELS:
        config = load_json(SPATIAL / ("%s_config.json" % name))
        model = StationAttentionForecaster(
            config["dynamic_channels"], config["static_channels"],
            hidden_size=config["hidden_size"],
            heads=config["training"]["attention_heads"],
            attention_dropout=config["training"]["attention_dropout"],
            head_hidden=config["training"]["head_hidden"],
            head_dropout=config["training"]["head_dropout"],
            cross_station=(name == "SA_R3"))
        model.load_state_dict(load_file(str(SPATIAL
                                            / ("%s.safetensors" % name))))
        model.eval()
        reloaded[name] = model
        with torch.no_grad():
            grouped = model(torch.from_numpy(sequence),
                            torch.from_numpy(static))[0].numpy()
        slot = int(np.flatnonzero(groups.target_observed[row])[0])
        stored = float(np.load(str(PRED / ("%s_validation.npy" % name)))[
            groups.sample_rows[row, slot]])
        report.add("Independent: reloaded %s reproduces a stored prediction"
                   % name, abs(float(grouped[slot]) - stored) < 1e-3,
                   "recomputed %.6f vs stored %.6f"
                   % (float(grouped[slot]), stored))

    perturbed = sequence.copy()
    keep = int(np.flatnonzero(groups.target_observed[row])[0])
    for index in range(12):
        if index != keep:
            perturbed[0, index] += 9.0
    with torch.no_grad():
        base = reloaded["SA_R2"](torch.from_numpy(sequence),
                                 torch.from_numpy(static))[0, keep]
        moved = reloaded["SA_R2"](torch.from_numpy(perturbed),
                                  torch.from_numpy(static))[0, keep]
    report.add("Independent: SA_R2 is invariant to another station's history",
               bool(torch.equal(base, moved)))
    with torch.no_grad():
        base3 = reloaded["SA_R3"](torch.from_numpy(sequence),
                                  torch.from_numpy(static))[0, keep]
        moved3 = reloaded["SA_R3"](torch.from_numpy(perturbed),
                                   torch.from_numpy(static))[0, keep]
    report.add("Independent: SA_R3 responds to another station's history",
               not bool(torch.equal(base3, moved3)))

    actual = np.load(str(PRED / "validation_actual.npy")).astype(np.float64)
    predictions = {n: np.load(str(PRED / ("%s_validation.npy" % n)))
                   .astype(np.float64) for n in MODELS}
    flat = groups.scatter(
        np.zeros((groups.size, 12)) + 1.0, EXPECTED_SAMPLES)
    report.add("Independent: grouped scatter covers every canonical sample",
               flat.size == EXPECTED_SAMPLES and bool(np.isfinite(flat).all()))

    probe = "Dongsi"
    cells = read_csv(OUT / "phase6_metrics_by_station_horizon.csv")
    selector = (station == probe) & (horizon == 6)
    for name, column in (("SA_R2", "SA_R2_MAE"), ("SA_R3", "SA_R3_MAE")):
        cell = [r for r in cells if r["station"] == probe
                and r["horizon"] == "6"][0]
        report.add("Independent: %s %s h6 cell MAE" % (name, probe),
                   abs(float(np.abs(actual[selector]
                                    - predictions[name][selector]).mean())
                       - float(cell[column])) < 1e-6)

    comparison = {r["model"]: r
                  for r in read_csv(OUT / "phase6_model_comparison.csv")}

    def macro(name):
        values = []
        for st in STATIONS:
            for hz in HORIZONS:
                m = (station == st) & (horizon == hz)
                values.append(float(np.abs(actual[m]
                                           - predictions[name][m]).mean()))
        return float(np.mean(values))

    left, right = macro("SA_R2"), macro("SA_R3")
    for name, value in (("SA_R2", left), ("SA_R3", right)):
        report.add("Independent: %s overall macro station-horizon MAE" % name,
                   abs(value - float(comparison[name]
                                     ["macro_station_horizon_MAE"])) < 1e-6,
                   "recomputed %.6f" % value)
    h4 = {r["scope"]: r for r in read_csv(OUT / "phase6_h4_summary.csv")}
    report.add("Independent: primary H4 relative gain",
               abs(100.0 * (left - right) / left
                   - float(h4["overall"]["relative_MAE_improvement_pct"]))
               < 1e-6,
               "recomputed %+.6f%%" % (100.0 * (left - right) / left))

    severe_rows = {r["model"]: r
                   for r in read_csv(OUT / "phase6_severe_metrics.csv")}
    mask = actual > 244.0
    recomputed = float(np.abs(actual[mask]
                              - predictions["SA_R3"][mask]).mean())
    report.add("Independent: SA_R3 severe MAE at the frozen threshold",
               abs(recomputed - float(severe_rows["SA_R3"]["severe_MAE"]))
               < 1e-6, "recomputed %.6f over %d severe hours"
               % (recomputed, int(mask.sum())))

    overall = read_csv(SPATIAL / "attention_matrix_overall.csv")
    matrix = np.array([[float(r[s]) for s in STATIONS] for r in overall])
    report.add("Independent: attention rows sum to one",
               bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-4)))
    report.add("Independent: both arrays hold 413,148 predictions",
               all(v.size == EXPECTED_SAMPLES for v in predictions.values()))
    report.add("Independent: no artifact holds a test-period target",
               not list(PRED.glob("*test*"))
               and bool(np.isnan(history.guarded_pm25_window(
                   "Dongsi", index_of(PARTITION_BOUNDS["validation"][1]))
                   ).sum() == 0))


def main():
    print("")
    print("[AIRSENSE V2 PHASE-6 VALIDATION]")
    print("")
    print("Read-only. Development validation only; the final test is sealed.")
    print("")
    report = Report()
    station, horizon, target = load_index()
    upstream(report)
    contract(report)
    grid_and_pairing(report)
    masks_and_causality(report)
    coordinates_and_prohibitions(report)
    determinism(report)
    coverage_and_metrics(report, station, horizon)
    independent(report, station, horizon, target)

    print("=" * 78)
    report.render()
    print("=" * 78)
    failures = report.failed()
    print("")
    if failures:
        print("V2 PHASE-6 VALIDATION: FAIL (%d gate(s))" % len(failures))
        return 1
    print("V2 PHASE-6 VALIDATION: PASS")
    print("Development validation only. The 2016-17 test remains sealed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
