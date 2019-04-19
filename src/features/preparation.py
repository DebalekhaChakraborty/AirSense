"""Feature preparation for AirSense V1.

Protocol Phase 5. Runs after the Phase 4 cleaning and split freeze, and
before M0-M3 are fitted.

One shared feature representation is built for M1 (Linear Regression),
M2 (Decision Tree) and M3 (Random Forest). This is deliberate: giving every
model the same inputs means the later comparison reflects model class
rather than differences in feature engineering.

Every decision below was fixed before any predictive performance existed.

Design rules enforced by construction:

* **Training-only vocabulary.** Categorical levels are established from the
  2010-2012 development-training predictors and then frozen. Validation and
  test are *applied* against that frozen schema, never unioned into it, so
  future data cannot influence the representation. An unseen level raises
  rather than silently adding a column.
* **Test-target quarantine.** ``data/processed/airsense_test_2014.csv`` is
  read with an explicit ``usecols`` list that omits ``pm2.5``, so the test
  target is never materialised into a DataFrame in this phase. An assertion
  enforces it. No ``y_test`` artifact is produced.
* **No scaling and no transformation.** Meteorological predictors are used
  in their original units. The target is written unmodified.
* **No target-derived features.** The X transformation never reads
  ``pm2.5``; only the dedicated ``y_train`` / ``y_validation`` extraction
  touches the development target.
* **Determinism.** No random operation and no execution timestamp, so
  repeated runs produce byte-identical output.

Historical constraint: CPython 3.6.7 with pandas 0.23.4 and numpy 1.15.4.
No scikit-learn estimator is imported; pandas and numpy are sufficient.
"""

import io
import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.data.audit import sha256_of_file

# ----------------------------------------------------------------------
# Frozen upstream inputs
# ----------------------------------------------------------------------

RAW_SHA256 = (
    "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c")

SOURCE_FILES = OrderedDict([
    ("development_train", "airsense_train_2010_2012.csv"),
    ("validation", "airsense_validation_2013.csv"),
    ("test", "airsense_test_2014.csv"),
])

EXPECTED_SOURCE_ROWS = OrderedDict([
    ("development_train", 24418), ("validation", 8678), ("test", 8661)])

TARGET = "pm2.5"
IDENTIFIER = "No"

# Columns read from the TEST file. pm2.5 is deliberately absent: the 2014
# target is quarantined until Protocol Phase 10 and must never enter this
# phase's memory, let alone its outputs.
TEST_USECOLS = ["No", "year", "month", "day", "hour",
                "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]

# ----------------------------------------------------------------------
# THE FROZEN FEATURE SCHEMA
# ----------------------------------------------------------------------

# Six raw meteorological predictors, used exactly as distributed: no
# standardisation, normalisation, log/sqrt transform, clipping,
# winsorizing, binning or polynomial expansion.
METEO_FEATURES = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]

# Source fields deliberately excluded from the feature matrix, with the
# reason each was excluded. None of these reasons is a model score.
EXCLUDED_SOURCE_FIELDS = OrderedDict([
    ("No", "observation identifier and traceability field, not a physical "
           "predictor; retained only in alignment manifests"),
    ("year", "the evaluation periods are disjoint later years, so a numeric "
             "year would ask the model to extrapolate a calendar counter "
             "into values it never saw; Phase 3 found no monotonic annual "
             "PM2.5 trend that would justify that extrapolation"),
    ("day", "day-of-month has no predeclared physical interpretation in the "
            "V1 hypothesis, is not naturally ordinal across month "
            "boundaries, and Phase 3 identified no defensible day-of-month "
            "relationship"),
    ("pm2.5", "the prediction target; never a predictor"),
])

# Categorical fields, their reference level, and the column prefix. The
# reference level is fixed in advance and is NOT chosen by target
# statistics or predictive advantage.
CATEGORICAL_SPECS = [
    ("month", "month", 1),
    ("hour", "hour", 0),
    ("cbwd", "cbwd", "NE"),
]

# Vocabularies expected from the development-training partition. Verified,
# never assumed: a mismatch stops the phase.
EXPECTED_VOCABULARIES = OrderedDict([
    ("month", list(range(1, 13))),
    ("hour", list(range(0, 24))),
    ("cbwd", ["NE", "NW", "SE", "cv"]),
])

EXPECTED_FEATURE_COUNT = 43

# Names that would indicate target leakage if they appeared in the schema.
LEAKAGE_TOKENS = ["pm2.5", "pm25", "target", "label", "aqi", "pollution_band"]

OUTPUT_FILES = OrderedDict([
    ("X_train", "X_train.csv"),
    ("X_validation", "X_validation.csv"),
    ("X_test", "X_test.csv"),
    ("y_train", "y_train.csv"),
    ("y_validation", "y_validation.csv"),
])


class FeatureError(RuntimeError):
    """Raised when a declared feature-preparation gate fails."""


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_development_partition(path):
    """Load a development partition (train or validation) in full.

    The development target is permitted here: it becomes y_train /
    y_validation. It is never used to construct X.
    """
    return pd.read_csv(path)


def load_test_predictors(path):
    """Load the TEST partition WITHOUT its target.

    ``usecols`` omits ``pm2.5`` so the 2014 target is never read from disk
    into this phase. The assertion afterwards is belt-and-braces: if the
    column list is ever edited carelessly, the phase stops rather than
    quietly quarantine-breaking.
    """
    if TARGET in TEST_USECOLS:
        raise FeatureError(
            "TEST_USECOLS must not contain the target column")
    frame = pd.read_csv(path, usecols=TEST_USECOLS)
    # Column order from usecols follows the file, so reindex explicitly.
    frame = frame[TEST_USECOLS]
    if TARGET in frame.columns:
        raise FeatureError(
            "QUARANTINE BREACH: the test target was materialised")
    return frame


def verify_upstream(project_root):
    """Verify raw and Phase-4 inputs against the frozen split manifest."""
    manifest_path = os.path.join(project_root, "artifacts",
                                 "split_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FeatureError("artifacts/split_manifest.json not found")
    manifest = json.load(io.open(manifest_path, encoding="utf-8"))

    raw_path = os.path.join(project_root, "data", "raw",
                            "PRSA_data_2010.1.1-2014.12.31.csv")
    raw_sha = sha256_of_file(raw_path)
    if raw_sha != RAW_SHA256:
        raise FeatureError(
            "raw dataset SHA-256 changed: expected %s, found %s"
            % (RAW_SHA256, raw_sha))

    checked = OrderedDict()
    for partition, filename in SOURCE_FILES.items():
        rel = os.path.join("data", "processed", filename)
        path = os.path.join(project_root, rel)
        recorded = manifest["files"].get(rel)
        if recorded is None:
            raise FeatureError("%s is not recorded in the split manifest"
                               % rel)
        observed = sha256_of_file(path)
        if observed != recorded["sha256"]:
            raise FeatureError(
                "%s SHA-256 changed: manifest %s, found %s"
                % (rel, recorded["sha256"], observed))
        rows = int(recorded["rows"])
        if rows != EXPECTED_SOURCE_ROWS[partition]:
            raise FeatureError(
                "%s row count %d, expected %d"
                % (rel, rows, EXPECTED_SOURCE_ROWS[partition]))
        checked[rel] = OrderedDict([("sha256", observed), ("rows", rows)])

    return manifest, raw_sha, checked, sha256_of_file(manifest_path)


# ----------------------------------------------------------------------
# Vocabulary - established from TRAINING ONLY
# ----------------------------------------------------------------------

def learn_vocabularies(train):
    """Establish categorical vocabularies from training predictors only.

    Validation and test are never consulted. Levels are sorted so the
    resulting column order is deterministic.
    """
    vocabularies = OrderedDict()
    for column, _prefix, _reference in CATEGORICAL_SPECS:
        levels = sorted(train[column].dropna().unique().tolist())
        vocabularies[column] = levels
    return vocabularies


def verify_vocabularies(vocabularies):
    """Stop if the training vocabulary is not what the phase declared."""
    problems = []
    for column, expected in EXPECTED_VOCABULARIES.items():
        observed = vocabularies.get(column)
        if observed != expected:
            problems.append("%s: expected %s, observed %s"
                            % (column, expected, observed))
    for column, _prefix, reference in CATEGORICAL_SPECS:
        if reference not in vocabularies.get(column, []):
            problems.append("%s: declared reference level %r is absent from "
                            "the training vocabulary"
                            % (column, reference))
    if problems:
        raise FeatureError(
            "training categorical vocabulary differs from the declared "
            "schema; stopping before changing it: " + "; ".join(problems))


def build_feature_names(vocabularies):
    """The single deterministic column order used by every partition."""
    names = list(METEO_FEATURES)
    for column, prefix, reference in CATEGORICAL_SPECS:
        for level in vocabularies[column]:
            if level == reference:
                continue          # reference level omitted, see §13
            names.append("%s_%s" % (prefix, level))
    return names


# ----------------------------------------------------------------------
# Transformation - applies the FROZEN schema, never re-learns it
# ----------------------------------------------------------------------

def transform(frame, vocabularies, feature_names, partition):
    """Build the feature matrix for one partition under the frozen schema.

    Dummy columns are constructed by explicit comparison against the frozen
    training vocabulary rather than by calling ``get_dummies`` on the
    partition. Calling ``get_dummies`` per partition and reconciling columns
    afterwards would let the partition's own contents shape the
    representation, which is exactly what the training-only rule forbids.
    """
    unseen_report = OrderedDict()
    columns = OrderedDict()

    for name in METEO_FEATURES:
        if name not in frame.columns:
            raise FeatureError("%s: predictor %s absent" % (partition, name))
        columns[name] = frame[name].values

    for column, prefix, reference in CATEGORICAL_SPECS:
        series = frame[column]
        observed = set(series.dropna().unique().tolist())
        unseen = sorted(observed - set(vocabularies[column]),
                        key=lambda value: str(value))
        unseen_report[column] = unseen
        if unseen:
            raise FeatureError(
                "%s: %s contains level(s) %s not present in the training "
                "vocabulary %s. Stopping rather than adding a feature "
                "column." % (partition, column, unseen,
                             vocabularies[column]))
        for level in vocabularies[column]:
            if level == reference:
                continue
            name = "%s_%s" % (prefix, level)
            columns[name] = (series == level).values.astype(np.int64)

    matrix = pd.DataFrame(columns, index=frame.index)
    matrix = matrix[feature_names]         # enforce the frozen order

    if list(matrix.columns) != feature_names:
        raise FeatureError("%s: column order does not match the frozen "
                           "schema" % partition)
    return matrix, unseen_report


def validate_matrix(matrix, feature_names, partition):
    """Every X column must be numeric, finite and complete."""
    if list(matrix.columns) != feature_names:
        raise FeatureError("%s: schema mismatch" % partition)
    if matrix.shape[1] != EXPECTED_FEATURE_COUNT:
        raise FeatureError(
            "%s: %d features, expected %d"
            % (partition, matrix.shape[1], EXPECTED_FEATURE_COUNT))

    object_columns = [c for c in matrix.columns
                      if matrix[c].dtype == np.object_]
    if object_columns:
        raise FeatureError("%s: non-numeric columns %s"
                           % (partition, object_columns))

    n_missing = int(matrix.isnull().sum().sum())
    if n_missing:
        raise FeatureError("%s: %d missing feature cells"
                           % (partition, n_missing))

    numeric = matrix.select_dtypes(include=[np.number])
    if numeric.shape[1] != matrix.shape[1]:
        raise FeatureError("%s: some columns are not numeric" % partition)
    if bool(np.isinf(numeric.values.astype(np.float64)).any()):
        raise FeatureError("%s: infinite values present" % partition)

    # Dummy columns must be strictly 0/1.
    for name in feature_names:
        if name in METEO_FEATURES:
            continue
        unique = set(np.unique(matrix[name].values).tolist())
        if not unique.issubset({0, 1}):
            raise FeatureError("%s: dummy column %s holds values %s"
                               % (partition, name, sorted(unique)))
    return OrderedDict([
        ("missing_feature_cells", 0),
        ("non_numeric_feature_columns", 0),
        ("infinite_values", 0),
    ])


def check_no_leakage(feature_names):
    """No feature name may reference the target or a target derivative."""
    offending = []
    for name in feature_names:
        lowered = name.lower()
        for token in LEAKAGE_TOKENS:
            if token in lowered:
                offending.append(name)
                break
    if offending:
        raise FeatureError(
            "target leakage detected in feature names: %s" % offending)


# ----------------------------------------------------------------------
# Round-trip verification against the Phase-4 source
# ----------------------------------------------------------------------

def verify_round_trip(source, matrix, vocabularies, partition):
    """Verify the matrix reproduces its source, row by row.

    Checks:

    * the six meteorological features match the source values exactly;
    * for each categorical, the dummy pattern maps back to exactly the
      original level, with an all-zero pattern meaning the reference level;
    * row order is preserved.
    """
    report = OrderedDict()

    for name in METEO_FEATURES:
        if not bool((matrix[name].values == source[name].values).all()):
            raise FeatureError(
                "%s: feature %s does not match the source values"
                % (partition, name))
    report["meteorological_features_match_source"] = True

    for column, prefix, reference in CATEGORICAL_SPECS:
        levels = [lv for lv in vocabularies[column] if lv != reference]
        names = ["%s_%s" % (prefix, lv) for lv in levels]
        block = matrix[names].values

        row_sums = block.sum(axis=1)
        if not bool(((row_sums == 0) | (row_sums == 1)).all()):
            raise FeatureError(
                "%s: %s dummies are not mutually exclusive" % (partition,
                                                               column))

        # Reconstruct the original level from the dummy pattern.
        reconstructed = np.empty(len(matrix), dtype=object)
        reconstructed[:] = reference                 # all-zero -> reference
        for index, level in enumerate(levels):
            reconstructed[block[:, index] == 1] = level

        original = source[column].values
        if not bool((reconstructed == original).all()):
            mismatches = int((reconstructed != original).sum())
            raise FeatureError(
                "%s: %s dummy pattern does not map back to the original "
                "category for %d rows" % (partition, column, mismatches))
        report["%s_dummies_map_back_exactly" % column] = True

    if not bool((matrix.index.values == source.index.values).all()):
        raise FeatureError("%s: row order was not preserved" % partition)
    report["row_order_preserved"] = True
    return report


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def _timestamps(frame):
    return pd.to_datetime(frame[["year", "month", "day", "hour"]])


def _write_csv(frame, path, index=False):
    frame.to_csv(path, index=index)
    return sha256_of_file(path)


def run_feature_preparation(project_root, verbose=True):
    """Execute Phase 5 end to end and write every artifact."""

    def log(message):
        if verbose:
            print(message)

    processed_dir = os.path.join(project_root, "data", "processed")
    features_dir = os.path.join(processed_dir, "features")
    artifacts_dir = os.path.join(project_root, "artifacts")
    results_dir = os.path.join(project_root, "results", "features")
    for directory in (features_dir, artifacts_dir, results_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    # -- upstream gate ---------------------------------------------------
    manifest, raw_sha, source_hashes, manifest_sha = verify_upstream(
        project_root)
    log("Upstream inputs verified against artifacts/split_manifest.json")

    # -- load ------------------------------------------------------------
    train = load_development_partition(
        os.path.join(processed_dir, SOURCE_FILES["development_train"]))
    validation = load_development_partition(
        os.path.join(processed_dir, SOURCE_FILES["validation"]))
    test = load_test_predictors(
        os.path.join(processed_dir, SOURCE_FILES["test"]))

    quarantine = OrderedDict()
    quarantine["test_usecols"] = list(TEST_USECOLS)
    quarantine["target_column_in_test_usecols"] = bool(TARGET in TEST_USECOLS)
    quarantine["target_column_in_test_dataframe"] = bool(
        TARGET in test.columns)
    quarantine["assertion"] = "'pm2.5' not in test_dataframe.columns"
    quarantine["assertion_holds"] = bool(TARGET not in test.columns)
    quarantine["y_test_artifact_created"] = False
    quarantine["status"] = "ACTIVE - 2014 target not read in Phase 5"
    if TARGET in test.columns:
        raise FeatureError("QUARANTINE BREACH: test target was loaded")
    log("Test-target quarantine verified: 'pm2.5' not in test columns")

    for partition, frame in (("development_train", train),
                             ("validation", validation), ("test", test)):
        expected = EXPECTED_SOURCE_ROWS[partition]
        if len(frame) != expected:
            raise FeatureError("%s: %d rows, expected %d"
                               % (partition, len(frame), expected))

    # -- predictor completeness -----------------------------------------
    predictor_columns = METEO_FEATURES + [c for c, _, _ in CATEGORICAL_SPECS]
    for partition, frame in (("development_train", train),
                             ("validation", validation), ("test", test)):
        n_missing = int(frame[predictor_columns].isnull().sum().sum())
        if n_missing:
            raise FeatureError(
                "%s: %d missing predictor values; no imputer will be "
                "introduced" % (partition, n_missing))
    log("All selected predictors complete in every partition")

    # -- vocabulary from TRAINING ONLY -----------------------------------
    vocabularies = learn_vocabularies(train)
    verify_vocabularies(vocabularies)
    feature_names = build_feature_names(vocabularies)
    check_no_leakage(feature_names)
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise FeatureError(
            "final schema has %d features, expected %d"
            % (len(feature_names), EXPECTED_FEATURE_COUNT))
    log("Training vocabulary frozen; schema has %d features"
        % len(feature_names))

    # -- transform -------------------------------------------------------
    matrices = OrderedDict()
    unseen = OrderedDict()
    validations = OrderedDict()
    round_trips = OrderedDict()
    sources = OrderedDict([("development_train", train),
                           ("validation", validation), ("test", test)])
    for partition, frame in sources.items():
        matrix, unseen_report = transform(frame, vocabularies, feature_names,
                                          partition)
        validations[partition] = validate_matrix(matrix, feature_names,
                                                 partition)
        round_trips[partition] = verify_round_trip(frame, matrix,
                                                   vocabularies, partition)
        matrices[partition] = matrix
        unseen[partition] = unseen_report
    log("All three partitions transformed under the frozen schema")

    # -- cross-partition schema validation -------------------------------
    orders = [list(m.columns) for m in matrices.values()]
    if not all(order == orders[0] for order in orders):
        raise FeatureError("feature schemas differ between partitions")
    log("Cross-partition schema identical across train / validation / test")

    # -- targets (development only) --------------------------------------
    def development_target(frame, partition):
        series = frame[TARGET]
        if bool(series.isnull().any()):
            raise FeatureError("%s: target contains missing values"
                               % partition)
        values = series.values
        integral = bool(np.all(values == np.floor(values)))
        # Lossless representation choice only; the values are unchanged.
        out = series.astype(np.int64) if integral else series
        return pd.DataFrame(OrderedDict([(TARGET, out)]), index=frame.index)

    y_train = development_target(train, "development_train")
    y_validation = development_target(validation, "validation")

    # -- write outputs ---------------------------------------------------
    output_hashes = OrderedDict()
    output_hashes[OUTPUT_FILES["X_train"]] = _write_csv(
        matrices["development_train"],
        os.path.join(features_dir, OUTPUT_FILES["X_train"]))
    output_hashes[OUTPUT_FILES["X_validation"]] = _write_csv(
        matrices["validation"],
        os.path.join(features_dir, OUTPUT_FILES["X_validation"]))
    output_hashes[OUTPUT_FILES["X_test"]] = _write_csv(
        matrices["test"], os.path.join(features_dir, OUTPUT_FILES["X_test"]))
    output_hashes[OUTPUT_FILES["y_train"]] = _write_csv(
        y_train, os.path.join(features_dir, OUTPUT_FILES["y_train"]))
    output_hashes[OUTPUT_FILES["y_validation"]] = _write_csv(
        y_validation,
        os.path.join(features_dir, OUTPUT_FILES["y_validation"]))
    log("Wrote 5 feature artifacts to data/processed/features/")

    # A y_test artifact must never exist.
    forbidden = os.path.join(features_dir, "y_test.csv")
    if os.path.exists(forbidden):
        raise FeatureError(
            "a y_test artifact exists; the 2014 target is quarantined")

    # -- row alignment manifest ------------------------------------------
    manifest_rows = []
    for partition in ("development_train", "validation", "test"):
        frame = sources[partition]
        stamps = _timestamps(frame)
        block = pd.DataFrame(OrderedDict([
            ("partition", partition),
            ("matrix_row_number", np.arange(len(frame), dtype=np.int64)),
            ("No", frame[IDENTIFIER].values),
            ("timestamp", stamps.astype(str).values),
        ]))
        manifest_rows.append(block)
    row_manifest = pd.concat(manifest_rows, ignore_index=True)
    if TARGET in row_manifest.columns:
        raise FeatureError("row manifest must not carry the target")
    row_manifest_path = os.path.join(artifacts_dir,
                                     "feature_row_manifest.csv")
    row_manifest_sha = _write_csv(row_manifest, row_manifest_path)
    if len(row_manifest) != sum(EXPECTED_SOURCE_ROWS.values()):
        raise FeatureError(
            "row manifest has %d rows, expected %d"
            % (len(row_manifest), sum(EXPECTED_SOURCE_ROWS.values())))
    log("Wrote artifacts/feature_row_manifest.csv (%d rows)"
        % len(row_manifest))

    # Alignment: manifest row N must name the observation at matrix row N.
    for partition in ("development_train", "validation", "test"):
        block = row_manifest[row_manifest["partition"] == partition]
        source_ids = sources[partition][IDENTIFIER].values
        if not bool((block["No"].values == source_ids).all()):
            raise FeatureError("%s: row manifest is misaligned" % partition)
        if not bool((block["matrix_row_number"].values
                     == np.arange(len(source_ids))).all()):
            raise FeatureError("%s: matrix row numbering is not sequential"
                               % partition)

    # -- decision table --------------------------------------------------
    decisions = build_decision_table()
    decisions_path = os.path.join(results_dir, "feature_decisions.csv")
    decisions_sha = _write_csv(decisions, decisions_path)

    # -- matrix summary --------------------------------------------------
    summary_rows = []
    for partition in ("development_train", "validation", "test"):
        frame = sources[partition]
        matrix = matrices[partition]
        stamps = _timestamps(frame)
        checks = validations[partition]
        unseen_levels = sorted(
            "%s=%s" % (col, lv)
            for col, levels in unseen[partition].items() for lv in levels)
        if partition == "test":
            target_count = ""          # quarantined - never counted
        else:
            target_count = int(frame[TARGET].notnull().sum())
        summary_rows.append(OrderedDict([
            ("partition", partition),
            ("row_count", int(len(matrix))),
            ("feature_count", int(matrix.shape[1])),
            ("missing_feature_cells", int(checks["missing_feature_cells"])),
            ("non_numeric_feature_cells",
             int(checks["non_numeric_feature_columns"])),
            ("infinite_feature_cells", int(checks["infinite_values"])),
            ("first_No", int(frame[IDENTIFIER].iloc[0])),
            ("last_No", int(frame[IDENTIFIER].iloc[-1])),
            ("first_timestamp", str(stamps.iloc[0])),
            ("last_timestamp", str(stamps.iloc[-1])),
            ("unseen_categorical_levels",
             ";".join(unseen_levels) if unseen_levels else "none"),
            ("feature_schema_match", "yes"),
            ("target_row_count", target_count),
        ]))
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "feature_matrix_summary.csv")
    summary_sha = _write_csv(summary, summary_path)
    log("Wrote results/features/ audit tables")

    # -- feature schema manifest -----------------------------------------
    schema = OrderedDict()
    schema["project"] = "AirSense"
    schema["phase"] = "V1 / Protocol Phase 5 - feature preparation"
    schema["feature_policy_source"] = "docs/FEATURE_POLICY.md (Phase 0 "\
        "declaration); resolved in docs/FEATURE_PREPARATION.md"
    schema["historical_cutoff"] = "2019-04-26"
    schema["note"] = (
        "All feature decisions were fixed before any predictive model "
        "performance existed. No execution timestamp is recorded here so "
        "that repeated runs are byte-identical.")

    provenance = OrderedDict()
    provenance["raw_dataset_sha256"] = raw_sha
    provenance["split_manifest_sha256"] = manifest_sha
    provenance["source_files"] = source_hashes
    schema["provenance"] = provenance

    periods = OrderedDict()
    for name, key in (("development_train", "development_train"),
                      ("validation", "validation"), ("test", "test")):
        entry = manifest["partitions"][key]
        periods[name] = OrderedDict([
            ("start", entry["declared_start"]),
            ("end", entry["declared_end"]),
            ("supervised_rows", entry["supervised_rows"]),
        ])
    schema["periods"] = periods

    schema["test_target_quarantine"] = quarantine

    schema["raw_included_features"] = list(METEO_FEATURES)
    schema["excluded_source_fields"] = EXCLUDED_SOURCE_FIELDS
    schema["categorical_fields"] = [c for c, _, _ in CATEGORICAL_SPECS]
    schema["training_vocabulary"] = OrderedDict(
        (column, [str(v) for v in levels])
        for column, levels in vocabularies.items())
    schema["vocabulary_source"] = (
        "development_train (2010-2012) only; validation and test are "
        "compatibility checks, never unioned into the vocabulary")
    schema["reference_categories"] = OrderedDict(
        (column, str(reference))
        for column, _prefix, reference in CATEGORICAL_SPECS)
    schema["reference_category_rationale"] = (
        "one level per categorical is omitted so the dummy block is not "
        "exactly linearly dependent with the intercept of an unregularized "
        "LinearRegression; the references are declared in advance and are "
        "not selected for predictive advantage")

    schema["feature_names"] = list(feature_names)
    schema["feature_count"] = int(len(feature_names))
    schema["shared_by_models"] = ["M1", "M2", "M3"]
    schema["scaling_policy"] = (
        "none - no StandardScaler, MinMaxScaler, RobustScaler, manual "
        "z-scoring or normalization; predictors keep their original units")
    schema["target_transformation_policy"] = (
        "none - pm2.5 is written unmodified; no log, standardisation, "
        "normalisation, clipping or binning")
    schema["engineered_feature_policy"] = OrderedDict([
        ("season", "not created - redundant with the month one-hot block"),
        ("weekend", "not created - the Phase 0 admission condition (a "
                    "weekday/weekend PM2.5 difference established by the "
                    "pre-modelling EDA) was not met, because Phase 3 did "
                    "not perform that comparison"),
        ("cyclical_sin_cos", "not created - month and hour are represented "
                             "categorically, which lets a linear model fit "
                             "non-monotone structure without imposing a "
                             "sinusoidal shape"),
        ("lag_and_rolling", "not created - a lagged target would change the "
                            "task definition to forecasting"),
        ("interactions_and_polynomials", "not created"),
        ("target_encoding", "prohibited"),
    ])

    outputs = OrderedDict()
    for filename, digest in output_hashes.items():
        path = os.path.join(features_dir, filename)
        rows = int(sum(1 for _ in io.open(path, encoding="utf-8")) - 1)
        with io.open(path, encoding="utf-8") as handle:
            columns = len(handle.readline().rstrip("\n").split(","))
        outputs[os.path.join("data", "processed", "features", filename)] = \
            OrderedDict([("sha256", digest), ("rows", rows),
                         ("columns", columns)])
    outputs[os.path.join("artifacts", "feature_row_manifest.csv")] = \
        OrderedDict([("sha256", row_manifest_sha),
                     ("rows", int(len(row_manifest)))])
    outputs[os.path.join("results", "features",
                         "feature_decisions.csv")] = OrderedDict(
        [("sha256", decisions_sha), ("rows", int(len(decisions)))])
    outputs[os.path.join("results", "features",
                         "feature_matrix_summary.csv")] = OrderedDict(
        [("sha256", summary_sha), ("rows", int(len(summary)))])
    schema["outputs"] = outputs

    schema["round_trip_verification"] = round_trips
    schema["cross_partition_schema_identical"] = True

    schema_path = os.path.join(artifacts_dir, "feature_schema.json")
    with io.open(schema_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(schema, indent=2))
        handle.write("\n")
    log("Wrote artifacts/feature_schema.json")

    # -- final upstream recheck ------------------------------------------
    verify_upstream(project_root)
    log("")
    log("Upstream inputs unchanged after feature preparation")
    return schema


def build_decision_table():
    """One row per source field, recording its Phase 5 disposition."""
    rows = [
        ("No", "identifier", "exclude", "not in feature matrix",
         "observation identifier and traceability field, not a physical "
         "predictor; kept only in alignment manifests"),
        ("year", "calendar", "exclude", "not in feature matrix",
         "evaluation periods are disjoint later years, so a numeric year "
         "would require extrapolating a calendar counter into unseen "
         "values; Phase 3 found no monotonic annual trend"),
        ("day", "calendar", "exclude", "not in feature matrix",
         "no predeclared physical interpretation, not naturally ordinal "
         "across month boundaries, and no Phase 3 evidence for it"),
        ("month", "calendar", "include", "one-hot, reference month_1",
         "non-monotone seasonal structure observed in Phase 3; categorical "
         "encoding lets a linear model represent it without assuming a "
         "shape"),
        ("hour", "calendar", "include", "one-hot, reference hour_0",
         "clear diurnal cycle observed in Phase 3; avoids imposing a false "
         "ordering 23 > 22 > ... > 0"),
        ("DEWP", "meteorological", "include", "raw continuous",
         "direct meteorological measurement; strongest positive rank "
         "association with PM2.5 in Phase 3"),
        ("TEMP", "meteorological", "include", "raw continuous",
         "direct meteorological measurement; Phase 3 found its Pearson and "
         "Spearman coefficients disagree in sign, indicating non-monotone "
         "structure the tree models may represent differently"),
        ("PRES", "meteorological", "include", "raw continuous",
         "direct meteorological measurement"),
        ("cbwd", "categorical", "include",
         "one-hot, reference cbwd_NE",
         "largest categorical separation of the target observed in Phase 3; "
         "vocabulary learned from training only"),
        ("Iws", "meteorological", "include", "raw cumulative",
         "strongest single association with PM2.5 in Phase 3; skew is a "
         "documented property of a cumulative quantity, not evidence of "
         "error, so it is used as distributed"),
        ("Is", "meteorological", "include", "raw sparse cumulative snow",
         "zero in most hours, but no evidence establishes the values as "
         "erroneous; kept raw for a simple period-authentic baseline"),
        ("Ir", "meteorological", "include", "raw sparse cumulative rain",
         "zero in most hours; Phase 3 rank association was not "
         "distinguishable from zero, but features are not selected on "
         "correlation strength at this phase"),
        ("season", "engineered", "exclude", "not created",
         "deterministic function of month; the month one-hot block already "
         "carries a more granular form of the same calendar information"),
        ("weekend", "engineered", "exclude", "not created",
         "the Phase 0 policy admitted it only if the pre-modelling EDA "
         "established a weekday/weekend PM2.5 difference; Phase 3 did not "
         "perform that comparison, so the predeclared condition was not "
         "met. This records adherence to the frozen policy, not evidence "
         "that weekends have no effect"),
        ("sin/cos month and hour", "engineered", "exclude", "not created",
         "month and hour are represented categorically; adding a cyclical "
         "form as well would duplicate the same information under an "
         "imposed sinusoidal shape"),
        ("pm2.5 lags and rolling means", "engineered", "exclude",
         "not created",
         "a lagged target changes the task definition to forecasting and "
         "would need its own baseline and experiment"),
        ("pm2.5", "target", "exclude from X", "y_train / y_validation only",
         "the prediction target; never a predictor. The 2014 target is "
         "quarantined and is not read in this phase"),
    ]
    return pd.DataFrame(OrderedDict([
        ("source_field", [r[0] for r in rows]),
        ("feature_class", [r[1] for r in rows]),
        ("decision", [r[2] for r in rows]),
        ("representation", [r[3] for r in rows]),
        ("reason", [r[4] for r in rows]),
    ]))
