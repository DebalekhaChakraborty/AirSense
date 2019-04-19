# AirSense V1 Exploratory Data Analysis

Phase 3 of the [V1 research protocol](V1_RESEARCH_PROTOCOL.md). Descriptive
characterisation of the UCI Beijing PM2.5 data set, carried out before any
cleaning, feature preparation or modelling.

Every figure quoted here is produced by
[`src/analysis/eda.py`](../src/analysis/eda.py) and stored in
`results/eda/`. Nothing in this document is asserted without a
corresponding generated table.

**No model has been trained and no predictive metric has been calculated.**

---

## Dataset

| Property | Value |
|---|---|
| Source | UCI Beijing PM2.5, `PRSA_data_2010.1.1-2014.12.31.csv` |
| SHA-256 | `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c` |
| Rows | 43,824 |
| Columns | 13 (1 index + 11 predictors + 1 target) |
| Period | 2010-01-01 00:00 to 2014-12-31 23:00 |
| Resolution | hourly |
| Target | `pm2.5`, µg/m³ |

Observations per year: 8,760 (2010), 8,760 (2011), 8,784 (2012), 8,760
(2013), 8,760 (2014) — 2012 being a leap year accounts for the extra 24.

Column types as inferred by pandas: `No`, `year`, `month`, `day`, `hour`,
`DEWP`, `Is`, `Ir` are integer; `pm2.5`, `TEMP`, `PRES`, `Iws` are float;
`cbwd` is categorical (object) with 4 levels.

*Table:* `results/eda/dataset_profile.csv`

---

## Data Quality

**Missing values occur in one column only.** `pm2.5` has 2,067 missing
observations (4.7166% of rows). All eleven other columns and the index
column are complete.

| Column | Missing | % |
|---|---:|---:|
| `pm2.5` | 2,067 | 4.7166 |
| all others | 0 | 0.0000 |

**Duplicates: none.** Zero duplicate rows whether or not the `No` index
column is included in the comparison.

**A note on `pm2.5` minimum.** The observed minimum is 0 µg/m³. Whether
these are genuine near-zero readings or a reporting floor cannot be
determined from this file alone, and no evidence here shows them to be
erroneous. They are therefore retained and flagged for Phase 4 rather than
removed.

*Tables:* `results/eda/missing_values.csv`,
`results/eda/dataset_profile.csv`

---

## PM2.5 Distribution

| Statistic | Value (µg/m³) |
|---|---:|
| Count (non-missing) | 41,757 |
| Missing | 2,067 |
| Mean | 98.61 |
| Median | 72.00 |
| Std. deviation | 92.05 |
| Minimum | 0 |
| 25th percentile | 29.00 |
| 75th percentile | 137.00 |
| Interquartile range | 108.00 |
| 90th percentile | 221.00 |
| 95th percentile | 284.00 |
| 99th percentile | 420.00 |
| Maximum | 994 |
| Skewness | 1.80 |
| Excess kurtosis | 4.77 |

**PM2.5 exhibits a strongly right-skewed distribution.** The mean (98.61)
sits well above the median (72.00), skewness is 1.80, and excess kurtosis
is 4.77. The upper tail is long: the 99th percentile is 420 µg/m³ while the
maximum reaches 994 µg/m³, roughly 9.7 standard deviations above the mean.

These high values are described as **high-concentration observations**, not
outliers. Severe pollution episodes in Beijing over 2010–2014 are
well-documented, and nothing in this dataset indicates that the extreme
readings are measurement errors. Removing them would discard exactly the
events a pollution model most needs to represent.

*Table:* `results/eda/pm25_descriptive_statistics.csv`
*Figure:* `figures/eda_pm25_distribution.png` — the right-hand panel uses a
log10 **display** scale; the target variable itself is not transformed.

---

## Temporal Patterns

### By year

| Year | n | Mean | Median |
|---|---:|---:|---:|
| 2010 | 8,091 | 104.05 | 79.0 |
| 2011 | 8,032 | 99.07 | 71.0 |
| 2012 | 8,295 | 90.55 | 69.0 |
| 2013 | 8,678 | 101.71 | 71.5 |
| 2014 | 8,661 | 97.73 | 72.0 |

Annual means vary within a band of roughly 91–104 µg/m³ with **no monotonic
trend** across the five years. The lowest annual mean is 2012 and the
highest is 2010; 2013 rises again. This is an observed pattern in five
annual aggregates, which is far too short a series to support any statement
about a longer-term trajectory.

### By hour of day

A clear and regular diurnal pattern is present. Mean PM2.5 is lowest in
mid-afternoon — hour 15 at 85.53 µg/m³ (median 58) — and highest overnight,
peaking at hour 01 with 113.70 µg/m³ (median 81). The mean varies by about
33% between the daily minimum and maximum, and the median by more (58 to
81). The curve is smooth: concentrations fall through the morning, bottom
out mid-afternoon, then climb through the evening.

### By calendar month

| Month | Mean | Median | | Month | Mean | Median |
|---|---:|---:|---|---|---:|---:|
| Jan | 115.06 | 66.0 | | Jul | 94.33 | 79.0 |
| Feb | 125.74 | 87.0 | | Aug | 80.00 | 70.0 |
| Mar | 97.76 | 70.0 | | Sep | 85.21 | 69.0 |
| Apr | 83.71 | 69.0 | | Oct | 120.40 | 80.0 |
| May | 80.11 | 69.0 | | Nov | 105.76 | 70.0 |
| Jun | 96.51 | 85.0 | | Dec | 98.20 | 54.0 |

The highest monthly means are February (125.74) and October (120.40); the
lowest are August (80.00) and May (80.11). Mean and median diverge sharply
in some months — December has a mean of 98.20 against a median of only
54.0, indicating that its average is driven by a minority of severe hours
rather than by generally elevated conditions.

*Tables:* `results/eda/pm25_by_year.csv`, `pm25_by_month.csv`,
`pm25_by_hour.csv`
*Figures:* `figures/eda_pm25_yearly.png`, `eda_pm25_monthly.png`,
`eda_pm25_hourly.png`

---

## Seasonal Patterns

Seasons are mapped from calendar month as follows. This mapping is
meteorological, not astronomical, and is fixed for the whole project:

| Season | Months |
|---|---|
| Winter | December, January, February |
| Spring | March, April, May |
| Summer | June, July, August |
| Autumn | September, October, November |

| Season | n | Mean | Median | Std |
|---|---:|---:|---:|---:|
| Winter | 10,385 | 112.78 | 70.0 | 119.20 |
| Spring | 10,570 | 87.21 | 69.0 | 73.45 |
| Summer | 10,389 | 90.44 | 77.0 | 64.17 |
| Autumn | 10,413 | 104.22 | 73.0 | 98.98 |

**The winter months show a higher mean concentration than the summer months
in this dataset** — 112.78 against 90.44 µg/m³.

**But the mean and the median disagree, and the disagreement is the more
informative result.** Winter has the *highest* mean yet a median of 70.0,
which is the *lowest* of the four seasons alongside spring. Summer has a
lower mean but the *highest* median at 77.0. Winter's standard deviation
(119.20) is nearly double summer's (64.17).

The observed pattern is therefore not "winter is uniformly more polluted".
It is that winter contains both the cleanest and by far the most extreme
hours, while summer is more consistently moderate. A seasonal summary
reported on means alone would misrepresent this.

No causal mechanism is claimed. Seasonal variation here is an association
observed in five years of data from one location; heating, meteorology and
emissions are not separately measured in this dataset.

*Table:* `results/eda/pm25_by_season.csv`
*Figure:* `figures/eda_pm25_seasonal.png`

---

## Meteorological Relationships

| Variable | Units | Mean | Median | Std | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| `DEWP` dew point | °C | 1.82 | 2.0 | 14.43 | −40 | 28 |
| `TEMP` temperature | °C | 12.45 | 14.0 | 12.20 | −19 | 42 |
| `PRES` pressure | hPa | 1016.45 | 1016.0 | 10.27 | 991 | 1046 |
| `Iws` cumulated wind speed | m/s | 23.89 | 5.37 | 50.01 | 0.45 | 585.60 |
| `Is` cumulated snow hours | hours | 0.05 | 0.0 | 0.76 | 0 | 27 |
| `Ir` cumulated rain hours | hours | 0.19 | 0.0 | 1.42 | 0 | 36 |

Three properties matter for later phases:

- **`Iws` is extremely skewed.** Its median is 5.37 m/s but its mean is
  23.89 and its maximum 585.60. This is expected: it is a *cumulative*
  quantity that resets between spells, not an instantaneous reading.
- **`Is` and `Ir` are zero most of the time.** Both have a median and a
  75th percentile of 0. They are effectively sparse event counters.
- All six are **fully populated** — no missing values.

Relationships with PM2.5 are shown as binned summaries over all available
observations rather than as a 40,000-point scatter, which would obscure
rather than reveal structure. No sampling is used, so no seed or sample-size
caveat applies and the figure is deterministic.

*Table:* `results/eda/numerical_feature_summary.csv`
*Figure:* `figures/eda_pm25_meteorology.png`

---

## Wind Direction

`cbwd` has four levels. It is reported as distributed; **no encoding has
been applied**, since encoding is a preprocessing decision.

| `cbwd` | Rows | % of rows | PM2.5 n | PM2.5 mean | PM2.5 median |
|---|---:|---:|---:|---:|---:|
| `NE` | 4,997 | 11.40 | 4,756 | 90.18 | 55.0 |
| `NW` | 14,150 | 32.29 | 13,484 | 70.13 | 31.0 |
| `SE` | 15,290 | 34.89 | 14,573 | 110.82 | 91.0 |
| `cv` | 9,387 | 21.42 | 8,944 | 126.15 | 98.0 |

This is the **largest categorical separation observed anywhere in this
analysis**. Median PM2.5 under `NW` conditions is 31 µg/m³ against 98 µg/m³
under `cv` — a factor of more than three. The mean ordering is
`NW` < `NE` < `SE` < `cv`, and the median ordering is identical.

The `cv` category is not expanded in the UCI variable documentation, so no
interpretation of the abbreviation is asserted here. It is noted only that
the category associated with the highest concentrations is also the one
whose label does not name a compass direction.

*Table:* `results/eda/pm25_by_wind_direction.csv`
*Figure:* `figures/eda_pm25_wind_direction.png`

---

## Correlation Analysis

Pearson correlation is computed over pairwise-complete observations.
Spearman rank correlation is reported as a robustness check, declared in
advance because PM2.5 is strongly skewed and its relationships with
meteorology are visibly non-linear in the binned figures.

### With PM2.5 (n = 41,757 pairs)

| Variable | Pearson r | Spearman ρ |
|---|---:|---:|
| `Iws` cumulated wind speed | −0.248 | **−0.360** |
| `DEWP` dew point | 0.171 | **0.300** |
| `PRES` pressure | −0.047 | −0.142 |
| `TEMP` temperature | −0.091 | 0.011 |
| `Ir` cumulated rain hours | −0.051 | −0.002 |
| `Is` cumulated snow hours | 0.019 | 0.047 |

**Wind speed is negatively correlated with PM2.5 in the observed data**, and
it is the strongest single association measured here on both coefficients
(Pearson −0.248, Spearman −0.360). Dew point is the strongest positive
association (Spearman 0.300).

Three cautions, each of which matters more than the ranking itself:

1. **`TEMP` changes sign between the two coefficients** — Pearson −0.091,
   Spearman +0.011. A linear coefficient and a rank coefficient disagreeing
   in sign is a signature of a non-monotonic relationship, and it means the
   Pearson value for temperature should not be read as "temperature is
   mildly protective". Neither coefficient summarises this relationship
   well.
2. **Spearman exceeds Pearson in magnitude for every variable except
   `TEMP` and `Ir`**, which indicates monotonic but non-linear structure
   that a purely linear model will not capture.
3. **`Ir` has a Spearman ρ of −0.002 with p = 0.63.** This is the one
   variable whose rank association is not distinguishable from zero in this
   sample. No conclusion is drawn from it either way.

All coefficients are small in absolute terms. The largest is −0.360. On its
own this indicates that meteorology alone accounts for a limited share of
the variation in hourly PM2.5 — which is unsurprising, since emissions,
traffic and industrial activity are not measured in this dataset.

### Among the predictors

| Pair | Pearson r |
|---|---:|
| `DEWP` – `TEMP` | 0.825 |
| `TEMP` – `PRES` | −0.827 |
| `DEWP` – `PRES` | −0.778 |

**The three primary meteorological predictors are strongly correlated with
one another.** This is a property of the atmosphere rather than a defect in
the data, but it has a direct consequence for Phase 7: individual
coefficients of a linear regression fitted on all three will not be stable
or individually interpretable.

**Correlation is not causation, and no feature has been selected or removed
on the basis of these values.** Feature decisions belong to Phase 5 and are
governed by [`FEATURE_POLICY.md`](FEATURE_POLICY.md).

*Tables:* `results/eda/pearson_correlation.csv`,
`pm25_spearman_correlations.csv`
*Figure:* `figures/eda_correlation_heatmap.png`

---

## Missingness Analysis

2,067 PM2.5 observations are missing (4.7166%). The question is whether they
are randomly distributed in time.

### By year

| Year | Rows | Missing | Missing rate |
|---|---:|---:|---:|
| 2010 | 8,760 | 669 | 7.64% |
| 2011 | 8,760 | 728 | 8.31% |
| 2012 | 8,784 | 489 | 5.57% |
| 2013 | 8,760 | 82 | **0.94%** |
| 2014 | 8,760 | 99 | **1.13%** |

**Missingness is not uniform across years.** The rate falls by roughly a
factor of seven between 2011 (8.31%) and 2013 (0.94%). 2010–2012 account for
1,886 of the 2,067 missing values — 91.2% of all missing targets fall in the
first three years.

### By month and by hour

By calendar month the rate ranges from 0.33% (February) to 10.30%
(August) — a thirty-fold spread.

By hour of day the rate is nearly flat, ranging only from 4.11% (hour 18) to
5.42% (hour 11). **Missingness is essentially independent of time of day
while being strongly dependent on year and month.**

### Consequence for the chronological evaluation

This is the most consequential data-quality finding of the phase. The
protocol requires an earlier period for training and a later period for
testing. Because the missing-target rate is roughly seven times higher in
2010–2012 than in 2013–2014, **any chronological split will produce a
training period and a test period with materially different amounts of
missing data.** Whatever handling rule Phase 4 adopts will therefore affect
the two periods unequally, and the effect must be quantified rather than
assumed away.

**No missing value has been imputed, and no row has been removed from any
stored dataset.** Handling is a Phase 4 decision.

*Tables:* `results/eda/pm25_missingness_by_year.csv`,
`pm25_missingness_by_month.csv`, `pm25_missingness_by_hour.csv`
*Figure:* `figures/eda_pm25_missingness.png`

---

## Time-Series Integrity

| Property | Value |
|---|---|
| First timestamp | 2010-01-01 00:00:00 |
| Last timestamp | 2014-12-31 23:00:00 |
| Expected hourly timestamps | 43,824 |
| Actual rows | 43,824 |
| Unique timestamps | 43,824 |
| Duplicate timestamps | 0 |
| Missing timestamps | 0 |
| Largest gap | 1.0 hour |
| Monotonic increasing in file order | True |
| Strictly hourly | True |
| All timestamps valid | True |

**The hourly timeline is complete.** Expected and actual counts agree
exactly, every timestamp is unique, no hour is absent, the largest interval
between consecutive observations is exactly one hour, and the rows are
already in chronological order in the file.

This is a favourable result for the chronological evaluation the protocol
requires: a split can be placed at any calendar boundary without creating a
gap, and no reindexing or resampling is needed.

Note the distinction from the missingness section: **no hourly row is
absent; some present rows have an absent target value.** These are different
problems and Phase 4 only faces the second.

**No train/validation/test boundary has been defined.** This section is data
characterisation only.

*Table:* `results/eda/time_series_integrity.csv`

---

## Key Observations

Each of the following is supported directly by a generated table.

1. **PM2.5 is strongly right-skewed** — mean 98.61 against median 72.00,
   skewness 1.80, maximum 994 µg/m³.
2. **The hourly timeline is complete** — 43,824 of 43,824 expected
   timestamps, no gap, no duplicate.
3. **Missing values affect the target only** — 2,067 (4.7166%); all
   predictors are fully populated.
4. **Missingness is strongly year-dependent** — 8.31% in 2011 against 0.94%
   in 2013; 91.2% of missing targets fall in 2010–2012.
5. **Missingness is essentially independent of hour of day** — 4.11% to
   5.42% across all 24 hours.
6. **A clear diurnal pattern is present** — mean lowest at hour 15 (85.53),
   highest at hour 01 (113.70).
7. **Winter shows the highest seasonal mean but not the highest median** —
   mean 112.78 with median 70.0, against summer's mean 90.44 with median
   77.0. Winter's variability is roughly double summer's.
8. **Wind speed is negatively correlated with PM2.5** — Pearson −0.248,
   Spearman −0.360, the strongest single association measured.
9. **Wind direction separates the target more than any other variable
   examined** — median 31 µg/m³ under `NW` against 98 µg/m³ under `cv`.
10. **Temperature's Pearson and Spearman coefficients disagree in sign**,
    indicating a non-monotonic relationship that neither coefficient
    summarises well.
11. **The meteorological predictors are strongly intercorrelated** —
    |r| between 0.78 and 0.83 among `DEWP`, `TEMP` and `PRES`.
12. **No annual trend is evident** across the five years, whose means lie
    within a 91–104 µg/m³ band with no monotonic direction.

---

## Limitations

- **One location.** A single monitoring site in one city. Nothing here
  generalises to other cities or to other sites within Beijing.
- **Fixed window.** 2010–2014 only. Five annual aggregates cannot establish
  a trend.
- **Meteorology is not the whole system.** Emissions, traffic, industrial
  activity and regional transport are unmeasured. The small correlation
  magnitudes are consistent with that omission, and it bounds what any V1
  model can achieve.
- **Cumulative variables.** `Iws`, `Is` and `Ir` accumulate within a spell
  and reset between spells. They are not instantaneous readings, and their
  distributions reflect that.
- **Two measurement sources.** PM2.5 comes from the US Embassy site and the
  meteorology from Beijing Capital International Airport, which are not
  co-located. The pairing is by hour, not by place.
- **Correlation only.** Pearson and Spearman coefficients describe
  association. No causal claim is made anywhere in this document.
- **The `pm2.5` minimum of 0** has not been established as either genuine or
  artefactual.

---

## Implications for Preprocessing

Decisions this phase hands to Phase 4 and Phase 5. None has been taken here.

**For Phase 4 (cleaning):**

1. **Missing-target handling must account for the year dependence.** A rule
   that drops the 2,067 rows removes 91.2% of them from 2010–2012, which
   thins the training period of a chronological split far more than the test
   period. Whichever rule is chosen, its unequal effect must be quantified
   and reported.
2. **High-concentration observations should be retained** unless positive
   evidence of measurement error emerges. They are the events the project
   exists to predict.
3. **The zero `pm2.5` readings need a decision**, recorded with its reason.
4. **No reindexing or resampling is required** — the hourly timeline is
   already complete and ordered.

**For Phase 5 (features):**

5. **`cbwd` requires encoding**, with categories fixed from the training
   period only. Four levels are present: `NE`, `NW`, `SE`, `cv`.
6. **Predictor collinearity affects linear-model interpretation.** With
   |r| ≈ 0.78–0.83 among `DEWP`, `TEMP` and `PRES`, the M1 coefficients will
   not be individually interpretable, and the write-up should say so rather
   than reading them as effect sizes.
7. **`Iws`, `Is` and `Ir` need explicit treatment.** `Iws` spans 0.45 to
   585.60 with a median of 5.37; `Is` and `Ir` are zero in more than 75% of
   hours.
8. **The diurnal and seasonal patterns justify carrying `hour` and `month`
   forward**, with the cyclical-encoding question decided in Phase 5 per
   [`FEATURE_POLICY.md`](FEATURE_POLICY.md).
9. **Temperature's non-monotonic relationship** suggests the tree-based
   models M2 and M3 may represent it differently from the linear model M1.
   This is recorded as an expectation to test, **not** as a prediction of
   which model will perform better.

**Unchanged from the protocol:** the primary evaluation remains a
chronological split, no lag feature is created, and the test set is used
once.
