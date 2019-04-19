"""Exploratory data analysis for AirSense V1.

Phase 3 of the research protocol: characterise the UCI Beijing PM2.5 data
set before any cleaning, feature preparation or modelling.

Scope boundaries, enforced by construction:

* The raw CSV is opened read-only and is never written, renamed or
  re-saved. Rows with a missing target are filtered IN MEMORY for
  individual statistics only; no cleaned dataset is produced.
* No train/test split, no encoding, no scaling, no imputation, no lag or
  rolling features, and no model of any kind.
* Every artifact is deterministic. No random sampling is used anywhere,
  including for visualisation - meteorological relationships are shown as
  binned summaries over all available observations rather than as a
  subsampled scatter, so no seed or sample-size caveat is needed.
* No execution timestamp is written into any scientific artifact, so
  repeated runs produce byte-identical output.

Historical constraint: CPython 3.6.7 with pandas 0.23.4, numpy 1.15.4,
scipy 1.1.0, matplotlib 3.0.2, seaborn 0.9.0. Only APIs available in those
releases are used. In particular this module avoids seaborn's estimator
plots (``barplot``, ``pointplot``), whose confidence intervals are computed
by bootstrap resampling and would make figures non-deterministic; plain
matplotlib is used instead, and seaborn only for the correlation heatmap.
"""

import io
import json
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")           # no display; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RAW_FILENAME = "PRSA_data_2010.1.1-2014.12.31.csv"
EXPECTED_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")

TARGET = "pm2.5"

REQUIRED_COLUMNS = [
    "No", "year", "month", "day", "hour", "pm2.5",
    "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir",
]

# Numerical meteorological predictors, in the order used throughout.
METEO_COLUMNS = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]

# Variables entering the correlation analysis.
CORRELATION_COLUMNS = [TARGET] + METEO_COLUMNS

# Units, taken from the UCI variable documentation. Nothing is invented:
# columns whose units are not documented are labelled without units.
UNITS = {
    "pm2.5": "ug/m^3",
    "DEWP": "degrees C",
    "TEMP": "degrees C",
    "PRES": "hPa",
    "Iws": "m/s",
    "Is": "hours",
    "Ir": "hours",
}

# Meteorological relationships shown as binned summaries.
RELATIONSHIP_COLUMNS = ["TEMP", "DEWP", "PRES", "Iws"]
N_BINS = 20

# Four-season mapping used throughout this phase. Documented here and in
# docs/EDA_ANALYSIS.md. Meteorological seasons, not astronomical ones.
SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Written with a fixed precision so repeated runs are byte-identical and
# the files stay readable.
FLOAT_FORMAT = "%.6f"

FIGURE_DPI = 100
PLOT_COLOR = "#31688e"
ACCENT_COLOR = "#b03a2e"


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

def project_paths(project_root):
    paths = OrderedDict()
    paths["root"] = project_root
    paths["raw_csv"] = os.path.join(project_root, "data", "raw", RAW_FILENAME)
    paths["results"] = os.path.join(project_root, "results", "eda")
    paths["figures"] = os.path.join(project_root, "figures")
    paths["artifacts"] = os.path.join(project_root, "artifacts")
    return paths


def ensure_output_dirs(paths):
    """Create output directories only. Never touches data/raw/."""
    for key in ("results", "figures", "artifacts"):
        if not os.path.isdir(paths[key]):
            os.makedirs(paths[key])


# ----------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------

def load_raw(raw_csv_path):
    """Load the raw CSV read-only and validate its schema.

    ``NA`` is the missing-value token used by the original UCI
    distribution. Nothing is dropped or filled here.
    """
    frame = pd.read_csv(raw_csv_path, na_values=["NA"], keep_default_na=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError("raw CSV is missing required columns: %s"
                         % ", ".join(missing))
    unexpected = [c for c in frame.columns if c not in REQUIRED_COLUMNS]
    if unexpected:
        raise ValueError("raw CSV has unexpected columns: %s"
                         % ", ".join(unexpected))
    return frame


def add_datetime(frame):
    """Return a copy carrying an in-memory ``datetime`` column.

    The enriched frame is never persisted. pandas 0.23 assembles a
    datetime from a frame whose columns are named year/month/day/hour.
    """
    enriched = frame.copy()
    enriched["datetime"] = pd.to_datetime(
        enriched[["year", "month", "day", "hour"]])
    return enriched


def add_season(frame):
    """Return a copy carrying an in-memory ``season`` column."""
    enriched = frame.copy()
    enriched["season"] = enriched["month"].map(SEASON_BY_MONTH)
    return enriched


# ----------------------------------------------------------------------
# Section 6 - dataset overview
# ----------------------------------------------------------------------

def dataset_profile(frame, enriched):
    """One row per column: dtype, non-null count, missing count and rate."""
    rows = []
    total = len(frame)
    for column in frame.columns:
        series = frame[column]
        n_missing = int(series.isnull().sum())
        rows.append(OrderedDict([
            ("column", column),
            ("dtype", str(series.dtype)),
            ("non_null_count", int(total - n_missing)),
            ("missing_count", n_missing),
            ("missing_percent", 100.0 * n_missing / total if total else 0.0),
            ("n_unique", int(series.nunique(dropna=True))),
        ]))
    return pd.DataFrame(rows)


def missing_values_table(frame):
    """Missing count and percentage per column, most-missing first."""
    total = len(frame)
    counts = frame.isnull().sum()
    table = pd.DataFrame(OrderedDict([
        ("column", counts.index),
        ("missing_count", counts.values.astype(int)),
        ("missing_percent", 100.0 * counts.values / total if total else 0.0),
        ("total_rows", total),
    ]))
    return table.sort_values(
        ["missing_count", "column"], ascending=[False, True]).reset_index(
        drop=True)


def observations_per_year(frame):
    grouped = frame.groupby("year").size()
    return pd.DataFrame(OrderedDict([
        ("year", grouped.index.astype(int)),
        ("observations", grouped.values.astype(int)),
    ]))


# ----------------------------------------------------------------------
# Section 7 - PM2.5 descriptive statistics
# ----------------------------------------------------------------------

def pm25_descriptive_statistics(frame):
    """Descriptive statistics over non-missing PM2.5 observations.

    High values are reported as high-concentration observations. They are
    not labelled outliers: severe pollution episodes in Beijing over
    2010-2014 are genuine measurements, and nothing in this data provides
    evidence that they are erroneous.
    """
    series = frame[TARGET]
    present = series.dropna()

    stats_rows = [
        ("count_non_missing", float(present.count())),
        ("count_missing", float(series.isnull().sum())),
        ("mean", float(present.mean())),
        ("median", float(present.median())),
        ("std", float(present.std())),
        ("min", float(present.min())),
        ("max", float(present.max())),
        ("percentile_25", float(present.quantile(0.25))),
        ("percentile_75", float(present.quantile(0.75))),
        ("iqr", float(present.quantile(0.75) - present.quantile(0.25))),
        ("percentile_90", float(present.quantile(0.90))),
        ("percentile_95", float(present.quantile(0.95))),
        ("percentile_99", float(present.quantile(0.99))),
        ("skewness", float(present.skew())),
        ("kurtosis_excess", float(present.kurtosis())),
    ]
    return pd.DataFrame(OrderedDict([
        ("statistic", [name for name, _ in stats_rows]),
        ("value", [value for _, value in stats_rows]),
        ("units", [UNITS[TARGET] if name not in
                   ("count_non_missing", "count_missing",
                    "skewness", "kurtosis_excess") else ""
                   for name, _ in stats_rows]),
    ]))


# ----------------------------------------------------------------------
# Sections 9 and 10 - temporal and seasonal grouping
# ----------------------------------------------------------------------

def _group_summary(frame, key, label):
    """count / mean / median / std of PM2.5 within each group of ``key``."""
    grouped = frame.groupby(key)[TARGET]
    table = pd.DataFrame(OrderedDict([
        (label, grouped.count().index),
        ("count_non_missing", grouped.count().values.astype(int)),
        ("mean", grouped.mean().values),
        ("median", grouped.median().values),
        ("std", grouped.std().values),
    ]))
    return table


def pm25_by_year(frame):
    return _group_summary(frame, "year", "year")


def pm25_by_month(frame):
    table = _group_summary(frame, "month", "month")
    table.insert(1, "month_name",
                 [MONTH_ABBR[int(m) - 1] for m in table["month"]])
    return table


def pm25_by_hour(frame):
    return _group_summary(frame, "hour", "hour")


def pm25_by_season(frame_with_season):
    table = _group_summary(frame_with_season, "season", "season")
    order = pd.Categorical(table["season"], categories=SEASON_ORDER,
                           ordered=True)
    table = table.assign(_order=order).sort_values("_order")
    return table.drop(columns=["_order"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Section 11 - meteorological variables
# ----------------------------------------------------------------------

def numerical_feature_summary(frame):
    """count / missing / mean / median / std / min / quartiles / max."""
    rows = []
    total = len(frame)
    for column in [TARGET] + METEO_COLUMNS:
        series = frame[column]
        present = series.dropna()
        rows.append(OrderedDict([
            ("variable", column),
            ("units", UNITS.get(column, "")),
            ("count_non_missing", int(present.count())),
            ("count_missing", int(total - present.count())),
            ("mean", float(present.mean())),
            ("median", float(present.median())),
            ("std", float(present.std())),
            ("min", float(present.min())),
            ("percentile_25", float(present.quantile(0.25))),
            ("percentile_75", float(present.quantile(0.75))),
            ("max", float(present.max())),
        ]))
    return pd.DataFrame(rows)


def binned_relationship(frame, column, n_bins=N_BINS):
    """Mean PM2.5 within equal-width bins of ``column``.

    Used instead of a 40k-point scatter, which would be an unreadable
    block of ink. Every available observation contributes; nothing is
    sampled, so the result is deterministic and is a genuine summary
    rather than a view of a subset.
    """
    subset = frame[[column, TARGET]].dropna()
    if subset.empty:
        return pd.DataFrame(
            columns=["bin_center", "count", "mean_pm25", "median_pm25"])
    edges = np.linspace(subset[column].min(), subset[column].max(),
                        n_bins + 1)
    # include_lowest so the minimum value is not dropped
    binned = pd.cut(subset[column], bins=edges, include_lowest=True)
    grouped = subset.groupby(binned)[TARGET]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return pd.DataFrame(OrderedDict([
        ("bin_center", centers),
        ("count", grouped.count().values.astype(int)),
        ("mean_pm25", grouped.mean().values),
        ("median_pm25", grouped.median().values),
    ]))


# ----------------------------------------------------------------------
# Section 12 - wind direction
# ----------------------------------------------------------------------

def pm25_by_wind_direction(frame):
    """Category frequencies and PM2.5 summaries per wind direction.

    ``cbwd`` is left as-is. No ordinal or one-hot encoding is applied:
    encoding is a preprocessing decision, not an EDA one.
    """
    total = len(frame)
    counts = frame["cbwd"].value_counts(dropna=False)
    grouped = frame.groupby("cbwd")[TARGET]
    rows = []
    for category in sorted(counts.index.tolist()):
        n_rows = int(counts[category])
        series = grouped.get_group(category).dropna()
        rows.append(OrderedDict([
            ("cbwd", category),
            ("row_count", n_rows),
            ("row_percent", 100.0 * n_rows / total if total else 0.0),
            ("pm25_count_non_missing", int(series.count())),
            ("pm25_mean", float(series.mean())),
            ("pm25_median", float(series.median())),
            ("pm25_std", float(series.std())),
        ]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Section 13 - correlation
# ----------------------------------------------------------------------

def pearson_correlation(frame):
    """Pearson correlation matrix using pairwise available observations.

    pandas ``DataFrame.corr`` excludes missing values pairwise by default,
    which is what is wanted here: PM2.5 is the only column with gaps, so
    every meteorological pair uses all 43,824 rows while PM2.5 pairs use
    the 41,757 rows where the target is present.
    """
    return frame[CORRELATION_COLUMNS].corr(method="pearson")


def pm25_spearman_correlations(frame):
    """Spearman rank correlation of PM2.5 against each meteorological
    variable, as a robustness check against the skew of PM2.5 and the
    non-linearity visible in the binned relationships.

    Declared before the values were inspected. Reported descriptively;
    p-values are included because scipy returns them, but no significance
    threshold is applied and no selection decision is taken from them.
    """
    rows = []
    for column in METEO_COLUMNS:
        subset = frame[[TARGET, column]].dropna()
        rho, p_value = stats.spearmanr(subset[TARGET], subset[column])
        pearson_r = subset[TARGET].corr(subset[column], method="pearson")
        rows.append(OrderedDict([
            ("variable", column),
            ("units", UNITS.get(column, "")),
            ("n_pairs", int(len(subset))),
            ("spearman_rho", float(rho)),
            ("spearman_p_value", float(p_value)),
            ("pearson_r", float(pearson_r)),
        ]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Section 14 - missingness pattern
# ----------------------------------------------------------------------

def missingness_by(frame, key, label):
    """Missing-PM2.5 count and rate within each group of ``key``."""
    is_missing = frame[TARGET].isnull()
    total = frame.groupby(key).size()
    missing = is_missing.groupby(frame[key]).sum()
    table = pd.DataFrame(OrderedDict([
        (label, total.index),
        ("total_rows", total.values.astype(int)),
        ("missing_pm25", missing.values.astype(int)),
        ("missing_rate_percent", 100.0 * missing.values / total.values),
    ]))
    return table


# ----------------------------------------------------------------------
# Section 15 - time-series integrity
# ----------------------------------------------------------------------

def time_series_integrity(enriched):
    """Characterise the hourly timeline. No split boundary is defined."""
    timestamps = enriched["datetime"]
    first = timestamps.min()
    last = timestamps.max()

    expected = int((last - first).total_seconds() // 3600) + 1
    actual = int(len(timestamps))
    unique = int(timestamps.nunique())
    duplicates = actual - unique

    full_index = pd.date_range(start=first, end=last, freq="H")
    observed = pd.DatetimeIndex(timestamps.unique())
    absent = full_index.difference(observed)

    ordered = timestamps.sort_values()
    deltas = ordered.diff().dropna()
    if len(deltas):
        largest_gap_hours = float(deltas.max().total_seconds() / 3600.0)
    else:
        largest_gap_hours = 0.0

    monotonic = bool(timestamps.is_monotonic_increasing)
    hourly = bool(len(deltas) > 0 and
                  deltas.max().total_seconds() == 3600.0 and
                  deltas.min().total_seconds() == 3600.0)

    rows = [
        ("first_timestamp", str(first)),
        ("last_timestamp", str(last)),
        ("expected_hourly_timestamps", expected),
        ("actual_rows", actual),
        ("unique_timestamps", unique),
        ("duplicate_timestamps", duplicates),
        ("missing_timestamps", int(len(absent))),
        ("largest_gap_hours", largest_gap_hours),
        ("monotonic_increasing_in_file_order", monotonic),
        ("strictly_hourly", hourly),
        ("timestamps_valid", bool(timestamps.notnull().all())),
    ]
    return pd.DataFrame(OrderedDict([
        ("property", [name for name, _ in rows]),
        ("value", [value for _, value in rows]),
    ]))


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def _save(fig, path):
    """Save a figure without embedding a creation timestamp.

    matplotlib 3.0 writes a ``Software`` tEXt chunk containing its version
    into PNG output; the metadata argument lets that be pinned so repeated
    runs are byte-identical.
    """
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=FIGURE_DPI, metadata={"Software": "AirSense V1"})
    except TypeError:                         # pragma: no cover
        fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def figure_distribution(frame, path):
    present = frame[TARGET].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(present.values, bins=60, color=PLOT_COLOR,
                 edgecolor="white", linewidth=0.4)
    axes[0].set_title("PM2.5 distribution (observed values)")
    axes[0].set_xlabel("PM2.5 (%s)" % UNITS[TARGET])
    axes[0].set_ylabel("Number of hourly observations")
    axes[0].axvline(present.mean(), color=ACCENT_COLOR, linestyle="--",
                    linewidth=1.2,
                    label="mean = %.1f" % present.mean())
    axes[0].axvline(present.median(), color="black", linestyle=":",
                    linewidth=1.2,
                    label="median = %.1f" % present.median())
    axes[0].legend(loc="upper right", fontsize=9)

    # Display transform only. The target variable itself is NOT transformed;
    # no modelling transformation has been approved.
    positive = present[present > 0]
    axes[1].hist(np.log10(positive.values), bins=60, color=PLOT_COLOR,
                 edgecolor="white", linewidth=0.4)
    axes[1].set_title("Same data, log10 display scale\n"
                      "(visualisation only - target is not transformed)")
    axes[1].set_xlabel("log10 PM2.5 (%s)" % UNITS[TARGET])
    axes[1].set_ylabel("Number of hourly observations")

    fig.suptitle("AirSense V1 EDA - PM2.5 distribution, Beijing 2010-2014",
                 y=1.02, fontsize=12)
    _save(fig, path)


def figure_by_group(table, key, value_columns, title, xlabel, path,
                    tick_labels=None, rotate=0):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(table))
    width = 0.38
    ax.bar(x - width / 2.0, table[value_columns[0]].values, width,
           color=PLOT_COLOR, label="mean")
    ax.bar(x + width / 2.0, table[value_columns[1]].values, width,
           color=ACCENT_COLOR, label="median")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels if tick_labels is not None
                       else table[key].astype(str).tolist(), rotation=rotate)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PM2.5 (%s)" % UNITS[TARGET])
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _save(fig, path)


def figure_wind_direction(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(table))
    labels = table["cbwd"].astype(str).tolist()

    axes[0].bar(x, table["row_percent"].values, color=PLOT_COLOR)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Combined wind direction (cbwd)")
    axes[0].set_ylabel("Share of observations (%)")
    axes[0].set_title("Frequency of wind-direction categories")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    axes[0].set_axisbelow(True)

    width = 0.38
    axes[1].bar(x - width / 2.0, table["pm25_mean"].values, width,
                color=PLOT_COLOR, label="mean")
    axes[1].bar(x + width / 2.0, table["pm25_median"].values, width,
                color=ACCENT_COLOR, label="median")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Combined wind direction (cbwd)")
    axes[1].set_ylabel("PM2.5 (%s)" % UNITS[TARGET])
    axes[1].set_title("PM2.5 by wind-direction category")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    axes[1].set_axisbelow(True)

    fig.suptitle("AirSense V1 EDA - wind direction", y=1.02, fontsize=12)
    _save(fig, path)


def figure_correlation_heatmap(correlation, path):
    fig, ax = plt.subplots(figsize=(7, 5.8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1.0, vmax=1.0, center=0.0, square=True,
                linewidths=0.5, cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title("Pearson correlation (pairwise complete observations)")
    _save(fig, path)


def figure_meteorology(relationships, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    flat = axes.ravel()
    for index, column in enumerate(RELATIONSHIP_COLUMNS):
        table = relationships[column]
        ax = flat[index]
        ax.plot(table["bin_center"].values, table["mean_pm25"].values,
                marker="o", markersize=4, color=PLOT_COLOR, label="mean")
        ax.plot(table["bin_center"].values, table["median_pm25"].values,
                marker="s", markersize=4, linestyle="--",
                color=ACCENT_COLOR, label="median")
        unit = UNITS.get(column, "")
        ax.set_xlabel("%s (%s)" % (column, unit) if unit else column)
        ax.set_ylabel("PM2.5 (%s)" % UNITS[TARGET])
        ax.set_title("PM2.5 against %s (%d equal-width bins)"
                     % (column, N_BINS))
        ax.grid(linestyle=":", linewidth=0.6, alpha=0.7)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("AirSense V1 EDA - binned meteorological relationships\n"
                 "(all available observations; no sampling)",
                 y=1.01, fontsize=12)
    _save(fig, path)


def figure_missingness(by_year, by_month, by_hour, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].bar(np.arange(len(by_year)),
                by_year["missing_rate_percent"].values, color=PLOT_COLOR)
    axes[0].set_xticks(np.arange(len(by_year)))
    axes[0].set_xticklabels(by_year["year"].astype(int).astype(str).tolist())
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Missing PM2.5 (% of hours)")
    axes[0].set_title("Missing rate by year")

    axes[1].bar(np.arange(len(by_month)),
                by_month["missing_rate_percent"].values, color=PLOT_COLOR)
    axes[1].set_xticks(np.arange(len(by_month)))
    axes[1].set_xticklabels(
        [MONTH_ABBR[int(m) - 1] for m in by_month["month"]], rotation=45)
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Missing PM2.5 (% of hours)")
    axes[1].set_title("Missing rate by month")

    axes[2].bar(np.arange(len(by_hour)),
                by_hour["missing_rate_percent"].values, color=PLOT_COLOR)
    axes[2].set_xticks(np.arange(0, len(by_hour), 2))
    axes[2].set_xticklabels(
        by_hour["hour"].astype(int).astype(str).tolist()[::2])
    axes[2].set_xlabel("Hour of day")
    axes[2].set_ylabel("Missing PM2.5 (% of hours)")
    axes[2].set_title("Missing rate by hour")

    for ax in axes:
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle("AirSense V1 EDA - distribution of missing PM2.5 "
                 "observations", y=1.03, fontsize=12)
    _save(fig, path)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def _write_csv(table, directory, filename, index=False):
    path = os.path.join(directory, filename)
    table.to_csv(path, index=index, float_format=FLOAT_FORMAT)
    return filename


def run_analysis(project_root, verbose=True):
    """Run the full EDA and write every artifact. Returns the summary dict.

    The raw CSV is read once and never written. Every output goes to
    results/eda/, figures/ or artifacts/.
    """
    from src.data.audit import sha256_of_file

    paths = project_paths(project_root)
    ensure_output_dirs(paths)

    def log(message):
        if verbose:
            print(message)

    raw_path = paths["raw_csv"]
    sha_before = sha256_of_file(raw_path)
    if sha_before != EXPECTED_SHA256:
        raise RuntimeError(
            "raw dataset SHA-256 mismatch: expected %s, found %s. "
            "Refusing to run EDA against an unexpected dataset."
            % (EXPECTED_SHA256, sha_before))

    log("Loading %s (read-only)" % RAW_FILENAME)
    frame = load_raw(raw_path)
    enriched = add_datetime(frame)
    enriched = add_season(enriched)

    results_dir = paths["results"]
    figures_dir = paths["figures"]
    written_results = []
    written_figures = []

    # -- Section 6: overview --------------------------------------------
    log("Dataset overview")
    profile = dataset_profile(frame, enriched)
    written_results.append(_write_csv(profile, results_dir,
                                      "dataset_profile.csv"))
    missing_table = missing_values_table(frame)
    written_results.append(_write_csv(missing_table, results_dir,
                                      "missing_values.csv"))
    per_year = observations_per_year(frame)

    # -- Section 7: PM2.5 descriptives ----------------------------------
    log("PM2.5 descriptive statistics")
    descriptives = pm25_descriptive_statistics(frame)
    written_results.append(_write_csv(descriptives, results_dir,
                                      "pm25_descriptive_statistics.csv"))
    descriptive_map = OrderedDict(
        zip(descriptives["statistic"], descriptives["value"]))

    # -- Sections 9 and 10: temporal and seasonal -----------------------
    log("Temporal analysis (year / month / hour) and seasonal analysis")
    by_year = pm25_by_year(frame)
    by_month = pm25_by_month(frame)
    by_hour = pm25_by_hour(frame)
    by_season = pm25_by_season(enriched)
    written_results.append(_write_csv(by_year, results_dir,
                                      "pm25_by_year.csv"))
    written_results.append(_write_csv(by_month, results_dir,
                                      "pm25_by_month.csv"))
    written_results.append(_write_csv(by_hour, results_dir,
                                      "pm25_by_hour.csv"))
    written_results.append(_write_csv(by_season, results_dir,
                                      "pm25_by_season.csv"))

    # -- Section 11: meteorological variables ---------------------------
    log("Meteorological feature summary and binned relationships")
    numerical = numerical_feature_summary(frame)
    written_results.append(_write_csv(numerical, results_dir,
                                      "numerical_feature_summary.csv"))
    relationships = OrderedDict()
    for column in RELATIONSHIP_COLUMNS:
        relationships[column] = binned_relationship(frame, column)

    # -- Section 12: wind direction -------------------------------------
    log("Wind-direction analysis")
    wind = pm25_by_wind_direction(frame)
    written_results.append(_write_csv(wind, results_dir,
                                      "pm25_by_wind_direction.csv"))

    # -- Section 13: correlation ----------------------------------------
    log("Correlation analysis (Pearson and Spearman)")
    correlation = pearson_correlation(frame)
    correlation_path = os.path.join(results_dir, "pearson_correlation.csv")
    correlation.to_csv(correlation_path, float_format=FLOAT_FORMAT)
    written_results.append("pearson_correlation.csv")
    spearman = pm25_spearman_correlations(frame)
    written_results.append(_write_csv(spearman, results_dir,
                                      "pm25_spearman_correlations.csv"))

    # -- Section 14: missingness ----------------------------------------
    log("Missing-PM2.5 pattern analysis")
    miss_year = missingness_by(frame, "year", "year")
    miss_month = missingness_by(frame, "month", "month")
    miss_hour = missingness_by(frame, "hour", "hour")
    written_results.append(_write_csv(miss_year, results_dir,
                                      "pm25_missingness_by_year.csv"))
    written_results.append(_write_csv(miss_month, results_dir,
                                      "pm25_missingness_by_month.csv"))
    written_results.append(_write_csv(miss_hour, results_dir,
                                      "pm25_missingness_by_hour.csv"))

    # -- Section 15: time-series integrity ------------------------------
    log("Time-series integrity")
    integrity = time_series_integrity(enriched)
    written_results.append(_write_csv(integrity, results_dir,
                                      "time_series_integrity.csv"))
    integrity_map = OrderedDict(
        zip(integrity["property"], integrity["value"]))

    # -- Figures ---------------------------------------------------------
    log("Generating figures")
    figure_distribution(frame, os.path.join(
        figures_dir, "eda_pm25_distribution.png"))
    written_figures.append("eda_pm25_distribution.png")

    figure_by_group(by_year, "year", ["mean", "median"],
                    "PM2.5 by year, Beijing 2010-2014", "Year",
                    os.path.join(figures_dir, "eda_pm25_yearly.png"))
    written_figures.append("eda_pm25_yearly.png")

    figure_by_group(by_month, "month", ["mean", "median"],
                    "PM2.5 by calendar month (all years pooled)", "Month",
                    os.path.join(figures_dir, "eda_pm25_monthly.png"),
                    tick_labels=by_month["month_name"].tolist())
    written_figures.append("eda_pm25_monthly.png")

    figure_by_group(by_hour, "hour", ["mean", "median"],
                    "PM2.5 by hour of day (all days pooled)",
                    "Hour of day",
                    os.path.join(figures_dir, "eda_pm25_hourly.png"))
    written_figures.append("eda_pm25_hourly.png")

    figure_by_group(by_season, "season", ["mean", "median"],
                    "PM2.5 by meteorological season", "Season",
                    os.path.join(figures_dir, "eda_pm25_seasonal.png"))
    written_figures.append("eda_pm25_seasonal.png")

    figure_wind_direction(wind, os.path.join(
        figures_dir, "eda_pm25_wind_direction.png"))
    written_figures.append("eda_pm25_wind_direction.png")

    figure_correlation_heatmap(correlation, os.path.join(
        figures_dir, "eda_correlation_heatmap.png"))
    written_figures.append("eda_correlation_heatmap.png")

    figure_meteorology(relationships, os.path.join(
        figures_dir, "eda_pm25_meteorology.png"))
    written_figures.append("eda_pm25_meteorology.png")

    figure_missingness(miss_year, miss_month, miss_hour, os.path.join(
        figures_dir, "eda_pm25_missingness.png"))
    written_figures.append("eda_pm25_missingness.png")

    # -- Section 19: machine-readable summary ----------------------------
    log("Writing artifacts/eda_summary.json")
    summary = OrderedDict()
    summary["project"] = "AirSense"
    summary["phase"] = "V1 / Phase 3 - exploratory data analysis"
    summary["historical_cutoff"] = "2019-04-26"
    summary["note"] = (
        "Descriptive characterisation only. No cleaning, imputation, "
        "encoding, splitting, feature engineering or modelling was "
        "performed. No execution timestamp is recorded here so that "
        "repeated runs are byte-identical.")

    dataset = OrderedDict()
    dataset["file"] = os.path.join("data", "raw", RAW_FILENAME)
    dataset["sha256"] = sha_before
    dataset["row_count"] = int(len(frame))
    dataset["column_count"] = int(len(frame.columns))
    dataset["columns"] = list(frame.columns)
    dataset["first_timestamp"] = str(integrity_map["first_timestamp"])
    dataset["last_timestamp"] = str(integrity_map["last_timestamp"])
    dataset["n_years"] = int(frame["year"].nunique())
    dataset["observations_per_year"] = OrderedDict(
        (str(int(r.year)), int(r.observations))
        for r in per_year.itertuples(index=False))
    summary["dataset"] = dataset

    quality = OrderedDict()
    quality["missing_pm25_count"] = int(frame[TARGET].isnull().sum())
    quality["missing_pm25_percent"] = float(
        100.0 * frame[TARGET].isnull().sum() / len(frame))
    quality["columns_with_missing_values"] = [
        str(c) for c in frame.columns if int(frame[c].isnull().sum()) > 0]
    quality["duplicate_rows_all_columns"] = int(frame.duplicated().sum())
    quality["duplicate_rows_excluding_index"] = int(
        frame.drop(columns=["No"]).duplicated().sum())
    summary["data_quality"] = quality

    summary["pm25_descriptive_statistics"] = OrderedDict(
        (str(k), float(v)) for k, v in descriptive_map.items())

    summary["temporal_integrity"] = OrderedDict(
        (str(k), v if not isinstance(v, (np.integer, np.floating))
         else v.item()) for k, v in integrity_map.items())

    summary["pearson_correlation_with_pm25"] = OrderedDict(
        (str(c), float(correlation.loc[TARGET, c])) for c in METEO_COLUMNS)
    summary["spearman_correlation_with_pm25"] = OrderedDict(
        (str(r.variable), float(r.spearman_rho))
        for r in spearman.itertuples(index=False))

    summary["wind_direction_categories"] = OrderedDict(
        (str(r.cbwd), OrderedDict([
            ("row_count", int(r.row_count)),
            ("row_percent", float(r.row_percent)),
            ("pm25_mean", float(r.pm25_mean)),
            ("pm25_median", float(r.pm25_median)),
        ])) for r in wind.itertuples(index=False))

    summary["season_mapping"] = OrderedDict(
        (str(month), SEASON_BY_MONTH[month]) for month in range(1, 13))

    summary["generated_result_files"] = sorted(written_results)
    summary["generated_figure_files"] = sorted(written_figures)

    sha_after = sha256_of_file(raw_path)
    summary["raw_dataset_sha256_before_analysis"] = sha_before
    summary["raw_dataset_sha256_after_analysis"] = sha_after
    summary["raw_dataset_unchanged"] = bool(sha_before == sha_after)

    summary_path = os.path.join(paths["artifacts"], "eda_summary.json")
    with io.open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2))
        handle.write("\n")

    if sha_before != sha_after:
        raise RuntimeError(
            "CRITICAL DATA INTEGRITY FAILURE: raw dataset digest changed "
            "during analysis (%s -> %s)" % (sha_before, sha_after))

    log("")
    log("Wrote %d result tables to results/eda/" % len(written_results))
    log("Wrote %d figures to figures/" % len(written_figures))
    log("Raw dataset unchanged: %s" % (sha_before == sha_after))
    return summary
