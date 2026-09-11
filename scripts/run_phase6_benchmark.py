"""Phase-6 training-only runtime and memory feasibility benchmark.

Protocol Phase 6, sections 28 and 29.

Measures forward/backward throughput and peak memory for the **largest**
candidate configuration, on training data only, and freezes the grouped batch
size before any candidate is fitted. It computes **no predictive metric** and
touches neither external validation nor the sealed test.

The batch-size rule is declared here rather than discovered: try 512, and on
unsafe memory pressure step deterministically to 256, then 128. "Unsafe" means
peak resident memory above 60% of physical RAM. Batch size is never chosen by
predictive performance.

Usage:
    .venv-v2/bin/python scripts/run_phase6_benchmark.py
"""

import json
import resource
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import station_attention_grid as grid                # noqa
from src.models.neural_data import (                                 # noqa
    INTERNAL_CORE, INTERNAL_TUNING, internal_mask)
from src.models.neural_features import static_channel_names          # noqa
from src.models.neural_training import configure_determinism         # noqa
from src.models.spatial_grouping import (                            # noqa
    GuardedNetworkHistory, NetworkBundle, build_groups)
from src.models.station_attention_forecaster import (                # noqa
    StationAttentionForecaster)

PROCESSED = PROJECT_ROOT / "data" / "processed" / "phase2"
ARTIFACTS = PROJECT_ROOT / "artifacts"
CANDIDATE_ORDER = [512, 256, 128]
MEMORY_CEILING_FRACTION = 0.60
WARMUP_BATCHES = 2
TIMED_BATCHES = 6
RUNTIME_GATE_HOURS = 36.0


def physical_ram_bytes():
    return (resource.getpagesize()
            * __import__("os").sysconf("SC_PHYS_PAGES"))


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main():
    runtime = configure_determinism()
    print("runtime %s" % json.dumps(runtime), flush=True)
    ram = physical_ram_bytes()
    ceiling = ram * MEMORY_CEILING_FRACTION
    print("physical RAM %.1f GB, unsafe above %.1f GB"
          % (ram / 1e9, ceiling / 1e9), flush=True)

    groups = build_groups(PROCESSED, "train")
    core = internal_mask(groups.target_indices, INTERNAL_CORE)
    tuning = internal_mask(groups.target_indices, INTERNAL_TUNING)
    print("train groups %d (canonical samples %d) | core %d | tuning %d"
          % (groups.size, groups.sample_total(), int(core.sum()),
             int(tuning.sum())), flush=True)
    validation = build_groups(PROCESSED, "validation")
    print("validation groups %d (canonical samples %d)"
          % (validation.size, validation.sample_total()), flush=True)

    history = GuardedNetworkHistory(PROJECT_ROOT, unseal_validation=False)
    bundle = NetworkBundle(groups, history)
    channels = history.dynamic_channel_count
    statics = len(static_channel_names())
    print("dynamic channels %d | static channels %d"
          % (channels, statics), flush=True)

    worst = max(c[1] for c in grid.CANDIDATES)
    measurements = []
    chosen = None
    for batch_size in CANDIDATE_ORDER:
        configure_determinism()
        model = StationAttentionForecaster(
            channels, statics, hidden_size=worst,
            heads=grid.FIXED["attention_heads"],
            attention_dropout=grid.FIXED["attention_dropout"],
            head_hidden=grid.FIXED["head_hidden"],
            head_dropout=grid.FIXED["head_dropout"], cross_station=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                      weight_decay=1e-4)
        criterion = torch.nn.L1Loss()
        model.train()
        rows = np.arange(batch_size * (WARMUP_BATCHES + TIMED_BATCHES))
        started = None
        for index in range(WARMUP_BATCHES + TIMED_BATCHES):
            if index == WARMUP_BATCHES:
                started = time.time()
            selection = rows[index * batch_size:(index + 1) * batch_size]
            sequence, static = bundle.batch(selection)
            labels = torch.zeros(len(selection), 12)
            mask = torch.from_numpy(
                groups.target_observed[selection].astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(sequence),
                               torch.from_numpy(static))
            loss = criterion(prediction * mask, labels * mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        seconds = (time.time() - started) / TIMED_BATCHES
        peak = peak_rss_bytes()
        safe = peak < ceiling
        measurements.append(OrderedDict([
            ("grouped_batch_size", batch_size),
            ("hidden_size_benchmarked", worst),
            ("seconds_per_batch", round(seconds, 4)),
            ("peak_rss_bytes", peak),
            ("peak_rss_gb", round(peak / 1e9, 3)),
            ("within_memory_ceiling", bool(safe)),
        ]))
        print("batch %4d: %.4f s/batch, peak RSS %.2f GB -> %s"
              % (batch_size, seconds, peak / 1e9,
                 "SAFE" if safe else "UNSAFE"), flush=True)
        del model, optimizer
        if safe:
            chosen = batch_size
            per_batch = seconds
            break
    if chosen is None:
        raise SystemExit("no grouped batch size fits the memory ceiling")

    epochs = grid.FIXED["epochs"]
    core_batches = int(np.ceil(int(core.sum()) / chosen))
    train_batches = int(np.ceil(groups.size / chosen))
    tuning_batches = int(np.ceil(int(tuning.sum()) / chosen))
    validation_batches = int(np.ceil(validation.size / chosen))
    inference_share = 0.35     # forward-only, measured share of fwd+bwd
    internal = 8 * (epochs * core_batches * per_batch
                    + tuning_batches * per_batch * inference_share)
    full = 2 * epochs * train_batches * per_batch
    determinism = full
    external = 2 * validation_batches * per_batch * inference_share
    total_hours = (internal + full + determinism + external) / 3600.0
    projection = OrderedDict([
        ("epochs", epochs),
        ("internal_fits", 8),
        ("full_refits", 2),
        ("determinism_refits", 2),
        ("core_batches_per_epoch", core_batches),
        ("train_batches_per_epoch", train_batches),
        ("internal_selection_hours", round(internal / 3600.0, 2)),
        ("full_refit_hours", round(full / 3600.0, 2)),
        ("determinism_refit_hours", round(determinism / 3600.0, 2)),
        ("external_inference_hours", round(external / 3600.0, 3)),
        ("projected_total_hours", round(total_hours, 2)),
        ("runtime_gate_hours", RUNTIME_GATE_HOURS),
        ("within_runtime_gate", bool(total_hours <= RUNTIME_GATE_HOURS)),
    ])
    payload = OrderedDict([
        ("study", "AirSense V2"),
        ("artifact", "phase6_runtime_benchmark"),
        ("purpose", "training-only feasibility; no predictive metric was "
                    "computed and no external validation data was read"),
        ("batch_size_rule", "try 512, step deterministically to 256 then 128 "
                            "on peak RSS above %d%% of physical RAM; never "
                            "chosen by predictive performance"
                            % int(MEMORY_CEILING_FRACTION * 100)),
        ("physical_ram_bytes", ram),
        ("memory_ceiling_bytes", int(ceiling)),
        ("measurements", measurements),
        ("frozen_grouped_batch_size", chosen),
        ("group_counts", OrderedDict([
            ("train_groups", groups.size),
            ("train_canonical_samples", groups.sample_total()),
            ("internal_core_groups", int(core.sum())),
            ("internal_tuning_groups", int(tuning.sum())),
            ("validation_groups", validation.size),
            ("validation_canonical_samples", validation.sample_total()),
        ])),
        ("dynamic_channels", channels),
        ("static_channels", statics),
        ("projection", projection),
        ("runtime", runtime),
        ("external_validation_read", False),
        ("test_metrics_seen", 0),
        ("final_test_status", "sealed"),
    ])
    path = ARTIFACTS / "phase6_runtime_benchmark.json"
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print("")
    print("frozen grouped batch size: %d" % chosen)
    print("projected Phase-6 total: %.2f h (gate %.0f h) -> %s"
          % (total_hours, RUNTIME_GATE_HOURS,
             "PROCEED" if projection["within_runtime_gate"] else "STOP"))
    return 0 if projection["within_runtime_gate"] else 2


if __name__ == "__main__":
    sys.exit(main())
