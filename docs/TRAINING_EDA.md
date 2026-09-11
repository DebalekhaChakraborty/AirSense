# AirSense V2 — Training-Only EDA

**Protocol Phase 1. Training partition only: 2013-03-01 00:00 → 2015-02-28 23:00.**

Every target statistic on this page comes from the **training** partition.
Validation target values were not mined for any decision, and the locked test
target was never opened — the loader physically refuses to return values for
those partitions (`src/data/eda.py`, `load_partition(mode="values")` raises
outside training). Language here is descriptive and predictive; no
environmental cause is claimed.

Tables: `results/eda_training/`. Figures: `figures/`, each titled
**TRAINING PERIOD ONLY**.

---

## 1. Scale of the training partition

| | |
|---|---|
| Stations | 12 |
| Hours per station | 17,520 (730 days) |
| Rows | **210,240** |
| PM2.5 observed | **205,989** |
| PM2.5 missing | **4,251 (2.022%)** |

## 2. PM2.5 distribution (pooled training)

| Statistic | Value (µg/m³) |
|---|---:|
| mean | 83.996758 |
| std | 79.045565 |
| min | 2.0 |
| P01 | 3.0 |
| P05 | 6.0 |
| P25 | 23.0 |
| **median (P50)** | **61.0** |
| P75 | 119.0 |
| P90 | 191.0 |
| **P95** | **244.0** |
| P99 | 351.0 |
| max | 844.0 |
| skewness | 1.644480 |
| excess kurtosis | 3.640253 |

Strongly right-skewed and heavy-tailed: the mean sits 38% above the median, and
the maximum is 3.5× the P95. The top 5% of hours span 244 → 844 µg/m³, so the
severe tail is not a narrow band near the threshold but a long reach above it —
which is precisely where V1's random forest failed, and why V2 declares tail
reliability as a first-class endpoint rather than a footnote.

Quantile method: linear interpolation between order statistics (type 7),
applied identically everywhere.

## 3. Station differences

| Station | Observed | Mean | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Wanliu | 17,399 | 90.9 | 68.0 | 200.0 | 258.0 | 589.0 |
| Nongzhanguan | 17,281 | 89.0 | 65.0 | 200.0 | 259.0 | 844.0 |
| Dongsi | 17,260 | 88.7 | 66.0 | 203.0 | 254.0 | 737.0 |
| Wanshouxigong | 17,263 | 88.1 | 67.0 | 197.0 | 253.9 | 704.0 |
| Aotizhongxin | 16,990 | 87.0 | 66.0 | 191.0 | 247.0 | 665.0 |
| Gucheng | 17,200 | 86.7 | 65.0 | 191.0 | 243.0 | 770.0 |
| Guanyuan | 17,198 | 86.2 | 65.0 | 189.0 | 240.2 | 603.0 |
| Tiantan | 17,134 | 86.2 | 65.0 | 194.0 | 241.0 | 541.0 |
| Shunyi | 17,049 | 82.4 | 60.0 | 189.0 | 238.0 | 641.0 |
| Changping | 17,182 | 77.5 | 53.0 | 184.0 | 238.0 | 581.0 |
| Huairou | 16,882 | 73.9 | 52.0 | 172.9 | 221.0 | 762.0 |
| Dingling | 17,151 | 71.0 | 45.0 | 172.0 | 234.0 | 548.0 |

Median ranges 45.0 → 68.0, a 1.5× spread; every station is skewed the same way
(skewness 1.53–1.82). The ordering is stable across mean, median and P90, so
station level is a persistent property of the network in this period rather
than an artefact of a few episodes.

**This matters for two decisions.** It supports station identity as a feature,
and it argues against per-station scaling — normalising each station by its own
centre would erase exactly the level differences the macro station-horizon
metric exists to compare fairly.

## 4. Temporal patterns

**Month** (pooled median → mean): February is worst (92.0 → 120.2), December
lowest by median (36.0) yet with a mean of 68.4 — a 1.9× mean-to-median ratio
that says December is mostly clean air punctuated by severe episodes. August is
lowest by mean (61.9). October shows the same episodic signature (median 73.0,
mean 105.2).

**Hour of day**: a shallow diurnal cycle — median 71.0 at 23:00 and 69.0 at
00:00 against 51.0 at 15:00. The afternoon minimum is consistent with a deeper
mixing layer, but no causal claim is made. The amplitude is small relative to
the synoptic variation, so hour-of-day alone is a weak predictor; it earns its
place as a cheap deterministic feature, not as a driver.

Figures: `training_monthly_pm25_pattern.png`, `training_hourly_pm25_pattern.png`.

## 5. Missingness structure

Pooled training missingness and gap structure, per variable:

| Variable | Missing | % | Gaps | Longest | Median | P90 | 1-h gaps | Gaps ≤6 h | Gaps >24 h | Hours in >24 h runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PM2.5 | 4,251 | 2.02 | 1,112 | 343 | 1 | 5 | 675 | 1,019 | 23 | 1,702 |
| PM10 | 3,224 | 1.53 | 938 | 343 | 1 | 5 | 641 | 860 | 15 | 1,116 |
| SO2 | 5,407 | 2.57 | 1,359 | 457 | 1 | 6 | 969 | 1,235 | 31 | 2,520 |
| NO2 | 7,395 | 3.52 | 1,890 | 900 | 1 | 5 | 1,329 | 1,733 | 30 | 3,269 |
| CO | 16,391 | 7.80 | 3,049 | 1,517 | 1 | 6 | 2,125 | 2,786 | 57 | 9,823 |
| O3 | 8,194 | 3.90 | 2,329 | 429 | 1 | 7 | 1,607 | 2,089 | 34 | 2,683 |
| TEMP | 176 | 0.08 | 158 | 7 | 1 | 1 | 146 | 157 | 0 | 0 |
| PRES | 177 | 0.08 | 159 | 7 | 1 | 1 | 147 | 158 | 0 | 0 |
| DEWP | 181 | 0.09 | 161 | 7 | 1 | 1 | 148 | 160 | 0 | 0 |
| RAIN | 171 | 0.08 | 159 | 3 | 1 | 1 | 148 | 159 | 0 | 0 |
| WSPM | 177 | 0.08 | 163 | 3 | 1 | 1 | 150 | 163 | 0 | 0 |
| wd | 257 | 0.12 | 215 | 13 | 1 | 2 | 192 | 213 | 0 | 0 |

**The single most important finding of Phase 1 is that missingness is
bimodal.** The median gap in every variable is one hour, and 91.6% of PM2.5
gaps are six hours or shorter — but 23 PM2.5 gaps exceed 24 hours and hold
1,702 of the 4,251 missing hours, i.e. **40% of missing PM2.5 hours sit in
long outages**. CO is worse: 60% of its missing hours are in runs over 24 h,
with a longest run of 1,517 h (63 days).

A single imputation rule cannot serve both regimes. Short gaps are trivially
bridged by carrying the last observation forward; long outages are station
downtime, where any carried value would be fiction. This is the evidence behind
the two-tier policy in [`PREPROCESSING_POLICY.md`](PREPROCESSING_POLICY.md).

**Meteorology is effectively complete**: under 0.13% missing, longest gap 7 h
(TEMP/PRES/DEWP), 13 h (wd), 3 h (RAIN/WSPM), and **zero** runs over 24 h. Any
sane causal fill covers it entirely.

Per-station PM2.5 outages are concentrated: Aotizhongxin (343 h), Huairou
(208 h) and Shunyi (197 h) carry the long ones, while Wanliu's longest gap is
13 h. Dropping long-gap windows would therefore remove data unevenly by station
— a reason not to drop them.

**When missingness happens**: by year 2013 1.40%, 2014 2.72%, 2015 (Jan–Feb)
0.94%; by month worst in December (4.04%), July (3.34%) and May (3.32%), best
in March (0.45%); by hour a mild midday bump (12:00 = 3.09%) against night
(06:00 = 1.58%). The pattern is real but modest. No cause is inferred.

Figures: `training_missingness_by_variable_station.png`,
`training_pm25_missing_run_lengths.png`.

## 6. Cross-station simultaneity

Per training timestamp, how many of the 12 stations lack PM2.5:

| Stations missing | Timestamps | % of hours |
|---:|---:|---:|
| 0 | 14,224 | 81.19 |
| 1 | 2,751 | 15.70 |
| 2 | 434 | 2.48 |
| 3 | 53 | 0.30 |
| 4 | 22 | 0.13 |
| 5–11 | 10 | 0.06 |
| **12 (all)** | **26** | **0.15** |

Missingness is overwhelmingly **local**: in 15.70% of hours exactly one station
is down while eleven are reporting. That is the empirical basis for R3 helping
with input coverage — when a station's own history has a hole, its neighbours
usually do not. The exception is real but tiny: 26 hours (0.15%) where the
whole network is dark, which no cross-station scheme can rescue.

## 7. Temporal memory — the evidence for context length

Pooled training PM2.5 autocorrelation at exact timestamp lags, computed on
observed pairs only with no imputation:

| Lag (h) | Pearson r | Valid pairs |
|---:|---:|---:|
| 1 | **0.965086** | 204,865 |
| 2 | 0.918880 | 204,466 |
| 3 | 0.873972 | 204,378 |
| 6 | 0.751537 | 203,980 |
| 12 | 0.566832 | 203,512 |
| 24 | 0.368044 | 202,844 |
| 48 | 0.097818 | 202,161 |
| 72 | −0.004124 | 201,647 |
| 168 | 0.008177 | 200,145 |

Three consequences, and they drive most of Phase 1:

1. **PM2.5 history is by far the strongest single predictor available.** At
   r = 0.965 the one-hour lag alone explains ~93% of variance in a linear
   sense. V1's exclusion of lagged PM2.5 was not a small omission, and H1 has
   strong prior support at short horizons. It also means **persistence will be
   a hard baseline at h = 1** — any V2 model that cannot beat it there has
   shown nothing.
2. **Memory is exhausted between 48 and 72 hours.** r falls to 0.098 at 48 h
   and to −0.004 at 72 h. Context beyond ~72 h buys essentially nothing from
   the target's own history.
3. **There is no weekly cycle** (r = 0.008 at 168 h). Day-of-week and weekend
   features have no support in this evidence and are demoted to optional.

The difficulty of the task also rises sharply with horizon: r drops from 0.965
(h = 1) to 0.368 (h = 24), so a flat metric pooled across horizons would hide
very different problems. This is why horizons are reported separately.

Figure: `training_pm25_autocorrelation.png`.

## 8. Cross-station association

Pairwise PM2.5 correlation on synchronised training timestamps, observed pairs
only, 132 off-diagonal pairs, minimum 16,390 valid pairs per cell:

- mean r = **0.881**
- maximum r = **0.966** (Tiantan ↔ Dongsi)
- minimum r = **0.785** (Dingling ↔ Wanshouxigong)

Every pair of stations in the network is strongly associated; even the least
related pair correlates at 0.785. R3 therefore has a plausible signal basis —
and equally, a caution: at r ≈ 0.88 the stations are largely telling the same
story, so cross-station information may prove substantially **redundant** with
a station's own recent history. H4 is a real question, not a foregone
conclusion. No causal or advective interpretation is made here; this is
association on synchronous timestamps only.

Figure: `training_cross_station_correlation.png`.

## 9. Wind direction

All **16** compass categories appear at all 12 stations within training, and a
structural check of the Phase-0 audit confirms the same 16-category set across
the whole dataset period at every station — so a training-fitted vocabulary
covers validation and test with no unseen category. The rarest category at any
one station still has 289 observations. `wd` is missing in only 0.12% of
training hours, with a longest gap of 13 h.

Table: `wind_direction_category_counts.csv`.

## 10. Spatial metadata

**The official UCI package contains no station coordinates.** The schema is
`No, year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP,
RAIN, wd, WSPM, station` — station is a name, with no latitude, longitude or
elevation, and the archive ships no metadata file. No coordinates were invented
and none were scraped. Track B graph modelling would need an authoritative
external source; that acquisition is deferred to the spatiotemporal phase.

Until then, cross-station structure must be learned from the correlation
evidence above rather than from geometry.

## 11. Implications carried into the preprocessing contract

- Two-tier causal missingness handling, with a 6-hour forward-fill ceiling
  justified by both the gap distribution and the autocorrelation decay.
- Masks and gap-age features are mandatory, because 40% of missing PM2.5 hours
  are in long outages that a fill cannot honestly represent.
- Global, not per-station, scaling — station level differences are signal.
- Robust (median/IQR) scaling given skewness 1.64 and excess kurtosis 3.64;
  RAIN needs a special case because its training IQR is zero.
- Context-length shortlist 24 / 48 / 72 h, with 168 h only as an optional
  control.
- Persistence is a serious baseline at short horizons and must be reported
  honestly.
