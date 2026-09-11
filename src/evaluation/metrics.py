"""Development-validation metrics for AirSense V2.

Protocol Phase 3.

The primary metric is **macro station-horizon MAE**: MAE computed
independently in each (station x horizon) cell, then averaged with equal
weight over cells. Equal weighting is deliberate - after missingness the cells
hold different row counts, and a micro average would let the most complete
stations set the score.

Residual convention, frozen: ``residual = actual - prediction``, so a
**positive residual is an under-prediction**.
"""

from collections import OrderedDict

import numpy as np

SEVERE_THRESHOLD = 244.0


def residuals(actual, prediction):
    return np.asarray(actual, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64)


def mae(actual, prediction):
    return float(np.abs(residuals(actual, prediction)).mean())


def rmse(actual, prediction):
    return float(np.sqrt((residuals(actual, prediction) ** 2).mean()))


def r2(actual, prediction):
    actual = np.asarray(actual, dtype=np.float64)
    total = ((actual - actual.mean()) ** 2).sum()
    if total == 0:
        return None
    return float(1.0 - (residuals(actual, prediction) ** 2).sum() / total)


def cell_metrics(actual, prediction, stations, horizons):
    """Per (station, horizon) cell metrics. Returns an ordered dict."""
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    stations = np.asarray(stations)
    horizons = np.asarray(horizons)
    cells = OrderedDict()
    for station in sorted(set(stations.tolist())):
        for horizon in sorted(set(horizons.tolist())):
            selector = (stations == station) & (horizons == horizon)
            n = int(selector.sum())
            if n == 0:
                continue
            cells[(station, int(horizon))] = OrderedDict([
                ("n", n),
                ("mae", mae(actual[selector], prediction[selector])),
                ("rmse", rmse(actual[selector], prediction[selector])),
                ("r2", r2(actual[selector], prediction[selector])),
            ])
    return cells


def macro_from_cells(cells, key="mae"):
    """Equal-weight mean over cells - the primary metric's definition."""
    values = [cell[key] for cell in cells.values() if cell[key] is not None]
    return float(np.mean(values)) if values else None


def macro_station_mae(actual, prediction, stations):
    """Horizon-specific macro-station MAE - the B3 selection criterion."""
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    stations = np.asarray(stations)
    values = []
    for station in sorted(set(stations.tolist())):
        selector = stations == station
        if selector.any():
            values.append(mae(actual[selector], prediction[selector]))
    return float(np.mean(values)) if values else None


def macro_station_rmse(actual, prediction, stations):
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    stations = np.asarray(stations)
    values = []
    for station in sorted(set(stations.tolist())):
        selector = stations == station
        if selector.any():
            values.append(rmse(actual[selector], prediction[selector]))
    return float(np.mean(values)) if values else None


def severe_metrics(actual, prediction, stations, horizons,
                   threshold=SEVERE_THRESHOLD):
    """Predeclared upper-tail endpoint. Threshold is frozen, never derived."""
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    severe = actual > threshold
    n = int(severe.sum())
    if n == 0:
        return OrderedDict([("severe_n", 0)])
    residual = residuals(actual[severe], prediction[severe])
    cells = cell_metrics(actual[severe], prediction[severe],
                         np.asarray(stations)[severe],
                         np.asarray(horizons)[severe])
    total_cells = len(set(zip(np.asarray(stations).tolist(),
                              np.asarray(horizons).tolist())))
    return OrderedDict([
        ("severe_n", n),
        ("severe_mae", float(np.abs(residual).mean())),
        ("severe_rmse", float(np.sqrt((residual ** 2).mean()))),
        ("severe_mean_residual", float(residual.mean())),
        ("severe_underprediction_pct",
         float(100.0 * (residual > 0).sum() / n)),
        ("severe_macro_station_horizon_mae", macro_from_cells(cells)),
        ("severe_contributing_cells", len(cells)),
        ("severe_total_cells", total_cells),
        ("severe_empty_cells", total_cells - len(cells)),
    ])


def negative_prediction_summary(prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    negative = prediction < 0
    return OrderedDict([
        ("n_negative", int(negative.sum())),
        ("pct_negative", float(100.0 * negative.sum() / prediction.size)),
        ("min_prediction", float(prediction.min())),
    ])
