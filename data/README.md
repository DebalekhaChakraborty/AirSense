# Data — AirSense V1

## Dataset

**Beijing PM2.5 Data Set**, UCI Machine Learning Repository (dataset ID 381).

| Field | Value |
|---|---|
| Dataset name | Beijing PM2.5 |
| Repository | UCI Machine Learning Repository |
| Dataset ID | 381 |
| DOI | `10.24432/C5JS49` |
| Creator / donor | Song Xi Chen, Guanghua School of Management & Center for Statistical Science, Peking University |
| Donated to UCI | 2017-01-18 |
| Observation period | 2010-01-01 through 2014-12-31 |
| Temporal resolution | hourly |
| Instances (rows) | 43,824 |
| Columns | 13 (1 ID + 11 features + 1 target) |
| Missing values | yes, encoded as the literal string `NA` |
| Subject area | Climate and environment |
| Associated task | Regression |

**Measurement sources.** PM2.5 concentrations were recorded at the **US
Embassy in Beijing**. The meteorological variables were recorded at
**Beijing Capital International Airport**. The two series are joined on the
hourly timestamp.

**Introductory paper.** Liang, X., Zou, T., Guo, B., Li, S., Zhang, H.,
Zhang, S., Huang, H., & Chen, S. X. (2015). *Assessing Beijing's PM2.5
pollution: severity, weather impact, APEC and winter heating.* Proceedings
of the Royal Society A, 471(2182).

**Citation.** Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5JS49

---

## Historical availability

The dataset was donated to UCI on **2017-01-18**, roughly two years and three
months before the AirSense V1 cutoff of **2019-04-26**. It was therefore
genuinely available to a Data Science trainee working in early 2019, and its
selection is period-consistent.

The introductory paper was published in **2015**, also well before the
cutoff.

---

## Source and retrieval

| Field | Value |
|---|---|
| Retrieved from | `https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv` |
| Landing page consulted | `https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data` |
| HTTP status | 200 |
| **Retrieval date (this reconstruction)** | **2026-09-05** |
| Dataset publication date | 2017-01-18 (donation to UCI) |

> **Publication date is not retrieval date.** The dataset was published in
> 2017. The copy in this repository was downloaded on **2026-09-05** as part
> of a present-day reconstruction. No claim is made that this file was
> downloaded, held, or used in 2019.

**Original form, not a derivative.** The file was fetched directly from the
UCI legacy `machine-learning-databases` path, which serves the original
distributed CSV. No Kaggle mirror, no pre-cleaned derivative and no
re-published variant was used. The retrieved file's header and column order
match the schema documented on the UCI variables table exactly.

Two retrieval notes, recorded as observed:

- The legacy *directory index* `.../machine-learning-databases/00381/`
  returns `NOT FOUND`, but the direct file URL under that path serves the
  original CSV with HTTP 200. Only the direct file URL is used.
- UCI now offers a modern convenience API (`pip install ucimlrepo`,
  `fetch_ucirepo(id=381)`). That package postdates the V1 cutoff and is
  **deliberately not used**. Retrieval is a plain HTTP fetch of the original
  file, which is what a 2019 workflow would have done.

### Re-acquisition

If `data/raw/` is empty, re-acquire with:

```sh
curl -L -o data/raw/PRSA_data_2010.1.1-2014.12.31.csv \
  https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv
```

Then verify integrity against the recorded digest:

```sh
sha256sum data/raw/PRSA_data_2010.1.1-2014.12.31.csv
# expected: 4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c
python scripts/preflight.py
```

If the digest differs, **stop**. A changed digest means the upstream file is
not the one this project was audited against, and it must be investigated
rather than accepted.

---

## License and usage

The UCI Machine Learning Repository currently presents this dataset under a
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license,
permitting sharing and adaptation for any purpose provided appropriate credit
is given.

*Honest qualification:* the CC BY 4.0 label is what the UCI site displays
**today (as read on 2026-09-05)**. In 2019 UCI presented a general citation
policy rather than per-dataset CC BY labelling. The current label is recorded
here because it is what can be verified now; it is not asserted as the 2019
license text. Attribution is given via the citation above in either case.

This project uses the data for non-commercial research and portfolio
purposes with attribution.

---

## Variables

| Column | Role | Type | Description | Units |
|---|---|---|---|---|
| `No` | ID | integer | Row number as distributed | — |
| `year` | feature | integer | Year of observation | — |
| `month` | feature | integer | Month of observation | 1–12 |
| `day` | feature | integer | Day of month | 1–31 |
| `hour` | feature | integer | Hour of day | 0–23 |
| `pm2.5` | **target** | integer | PM2.5 concentration | µg/m³ |
| `DEWP` | feature | integer | Dew point | °C |
| `TEMP` | feature | real | Temperature | °C |
| `PRES` | feature | real | Pressure | hPa |
| `cbwd` | feature | categorical | Combined wind direction | — |
| `Iws` | feature | real | Cumulated wind speed | m/s |
| `Is` | feature | integer | Cumulated hours of snow | hours |
| `Ir` | feature | integer | Cumulated hours of rain | hours |

Target variable: **`pm2.5`**.

`Iws`, `Is` and `Ir` are *cumulative* quantities that reset between spells.
They are not instantaneous readings, and that distinction is carried into
[`../docs/FEATURE_POLICY.md`](../docs/FEATURE_POLICY.md).

---

## Missing-value behaviour

Missing values are encoded as the literal string `NA` in the raw file.

As audited (see [`../docs/DATASET_AUDIT.md`](../docs/DATASET_AUDIT.md)):

- **`pm2.5` is the only column with missing values: 2,067 of 43,824 rows
  (4.7166%).**
- All other twelve columns are fully populated.
- 41,757 rows (95.2834%) carry a usable target value.

The 43,824 rows correspond exactly to the 1,826 days from 2010-01-01 to
2014-12-31 at 24 hourly slots per day, so **no hourly timestamp is absent**.
Missingness is confined to the target value within otherwise present rows.

**No missing values have been removed or imputed.** Handling of missing
`pm2.5` is a Phase 4 decision and is deliberately deferred so that the
choice is made and documented explicitly rather than absorbed silently
during loading.

---

## Directory layout and immutability

```
data/
├── README.md      this file
├── raw/           original, immutable inputs
└── processed/     derived outputs (currently empty)
```

**`data/raw/` is never modified in place.** The raw CSV is stored read-only
(mode `444`) and its SHA-256 is recorded in
[`../artifacts/data_audit.json`](../artifacts/data_audit.json). Every
cleaning or transformation step must read from `raw/` and write to
`processed/`, leaving the original byte-identical and re-verifiable.

`data/processed/` is empty at the end of Phase 0. Nothing has been cleaned,
filtered, imputed or engineered.
