# AirSense V2 — Dataset Provenance

**Protocol Phase 0. Acquisition and audit record.**

Machine-readable companions: `artifacts/raw_dataset_manifest.json`,
`artifacts/dataset_audit.json`.

---

## 1. Source

| | |
|---|---|
| Data set | Beijing Multi-Site Air-Quality Data |
| UCI repository ID | 501 |
| DOI | `10.24432/C5RK5G` |
| Landing page | https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data |
| Resolved download | https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip |
| License | CC BY 4.0 as presented by UCI |
| Acquired by | `scripts/acquire_dataset.py` |

Acquired from the official UCI source only. No Kaggle mirror, no GitHub mirror,
no pre-cleaned derivative, no third-party parquet, no tutorial copy.

## 2. Archive identity

| Artefact | Bytes | SHA-256 |
|---|---:|---|
| `beijing+multi+site+air+quality+data.zip` (outer) | 8,192,212 | `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4` |
| `PRSA2017_Data_20130301-20170228.zip` (inner) | 7,959,991 | `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8` |
| 12 station CSVs, aggregate | 32,508,303 | `9fbecc200af90f1010574a503badb8057360889b0d5c888f34519210e2e298e8` |

Per-file digests are in `artifacts/raw_dataset_manifest.json`. The aggregate is
the SHA-256 of the sorted `member:sha256` lines, so it changes if any source
file changes.

## 3. Archive structure — and a contamination finding

The official archive is **nested**, and it carries members that are not part of
this data set:

```
beijing+multi+site+air+quality+data.zip
├── PRSA2017_Data_20130301-20170228.zip     the data set
│   └── PRSA_Data_20130301-20170228/
│       └── PRSA_Data_<Station>_20130301-20170228.csv   x 12
├── 2A2478DC-8517-4490-9FA7-36F9A7A542BE.JPG            page illustration
├── data.csv                                            NOT air quality
└── test.csv                                            NOT air quality
```

**`data.csv` and `test.csv` inside the official UCI archive are unrelated
financial time series.** Both carry the header
`Date,Open,High,Low,Close,Adj Close,Volume` with 2018 dates and 503 rows —
stock OHLCV data, not Beijing air quality. They appear to be an upload
accident in the official distribution.

Consequences, recorded deliberately:

- Neither file is extracted into `data/raw/`; neither is used by V2.
- Both are hashed in `artifacts/raw_dataset_manifest.json` under
  `auxiliary_members_not_part_of_dataset` so the finding is evidence, not
  folklore.
- **The file named `test.csv` has nothing to do with the V2 locked final
  test.** Any future reader who greps for `test.csv` must not confuse them.
- `scripts/acquire_dataset.py` refuses any archive whose structure differs from
  the one verified here, and refuses any station-shaped CSV found outside the
  inner archive.

## 4. Raw immutability

- Raw material lives under `data/raw/` and is set read-only (mode 0444) at
  extraction.
- The acquisition script never re-downloads over an existing archive and never
  overwrites an existing raw file.
- Re-running `scripts/acquire_dataset.py --verify-only` re-hashes what is on
  disk without writing anything.
- **No cleaning, no imputation, no type coercion, no row removal has been
  applied.** `NA` tokens remain exactly as distributed.
- Derived data will live under `data/processed/` and will never overwrite raw.

## 5. Verified dataset facts

Every figure below was computed from the downloaded files by
`scripts/audit_dataset.py`, not taken from documentation.

| Property | Documented by UCI | Verified |
|---|---|---|
| Rows | 420,768 | **420,768** ✓ |
| Stations | 12 | **12** ✓ |
| Period | 2013-03-01 → 2017-02-28 | **2013-03-01 00:00 → 2017-02-28 23:00** ✓ |
| Missing marker | `NA` | `NA` ✓ |

**Stations** (12, identical spelling in filename and `station` column):
Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou,
Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong.

**Schema** — identical across all 12 files, in this order:

```
No, year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3,
TEMP, PRES, DEWP, RAIN, wd, WSPM, station
```

Loaded as text by the CSV reader; `No` and the four time columns parse as
integers, the eleven pollutant/meteorology columns as floats, `wd` and
`station` are categorical. `wd` has 16 compass categories (E, ENE, ESE, N, NE,
NNE, NNW, NW, S, SE, SSE, SSW, SW, W, WNW, WSW).

**Timestamp grid — perfect.** Each station holds exactly 35,064 rows, one per
hour of the 1,461-day span, with:

- 0 missing timestamps,
- 0 duplicate `(station, timestamp)` pairs,
- 0 timestamps outside the expected grid,
- 0 fully duplicated rows,
- 0 malformed rows,
- `No` dense and 1-based in every file.

12 × 35,064 = 420,768. The calendar is complete; **all missingness is value
missingness, never row absence.** This materially simplifies causal window
construction: the hourly index needs no reindexing.

## 6. Missingness

Aggregate over 420,768 rows:

| Variable | Missing | Rate |
|---|---:|---:|
| CO | 20,701 | 4.920% |
| O3 | 13,277 | 3.155% |
| NO2 | 12,116 | 2.879% |
| SO2 | 9,021 | 2.144% |
| **PM2.5 (target)** | **8,739** | **2.077%** |
| PM10 | 6,449 | 1.533% |
| wd | 1,822 | 0.433% |
| DEWP | 403 | 0.096% |
| TEMP | 398 | 0.095% |
| PRES | 393 | 0.093% |
| RAIN | 390 | 0.093% |
| WSPM | 318 | 0.076% |
| No, year, month, day, hour, station | 0 | 0% |

Meteorology is nearly complete (< 0.1%); pollutants carry the missingness, CO
worst at 4.9%. Full breakdowns:
`results/dataset_audit/missingness_by_variable.csv`,
`results/dataset_audit/missingness_by_station.csv`.

### PM2.5 missingness by station and year

| Station | 2013 | 2014 | 2015 | 2016 | 2017 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Aotizhongxin | 11 | 505 | 216 | 177 | 16 | 925 |
| Changping | 33 | 281 | 332 | 114 | 14 | 774 |
| Dingling | 134 | 221 | 109 | 306 | 9 | 779 |
| Dongsi | 143 | 97 | 127 | 353 | 30 | 750 |
| Guanyuan | 102 | 204 | 117 | 140 | 53 | 616 |
| Gucheng | 136 | 179 | 166 | 144 | 21 | 646 |
| Huairou | 292 | 330 | 159 | 156 | 16 | 953 |
| Nongzhanguan | 39 | 189 | 254 | 121 | 25 | 628 |
| Shunyi | 262 | 191 | 124 | 321 | 15 | 913 |
| Tiantan | 17 | 361 | 154 | 126 | 19 | 677 |
| Wanliu | 21 | 93 | 147 | 103 | 18 | 382 |
| Wanshouxigong | 41 | 209 | 222 | 196 | 28 | 696 |
| **Total** | **1,231** | **2,860** | **2,127** | **2,257** | **264** | **8,739** |

2013 covers 10 months and 2017 covers 2 months, so their totals are not
comparable to full years. Station-level target missingness ranges from 1.09%
(Wanliu, 382) to 2.72% (Huairou, 953) — a 2.5× spread that the macro
station-horizon metric is designed to keep from silently reweighting the
result. These are **counts of absence, not values**: no PM2.5 value statistic
was computed. Source:
`results/dataset_audit/target_missingness_by_station_year.csv`.

## 7. Value integrity

- **No negative values** in any non-negative column (the six pollutants, RAIN,
  WSPM) at any station.
- **No non-finite values** (`inf`, `nan` literals) anywhere.
- No sentinel codes such as −999 were found.

Covariate ranges across all stations, used to detect impossible values:

| Variable | Min | Max |
|---|---:|---:|
| PM10 | 2.0 | 999.0 |
| SO2 | 0.2856 | 500.0 |
| NO2 | 1.0265 | 290.0 |
| CO | 100.0 | 10000.0 |
| O3 | 0.2142 | 1071.0 |
| TEMP (°C) | −19.9 | 41.6 |
| PRES (hPa) | 982.4 | 1042.8 |
| DEWP (°C) | −43.4 | 29.1 |
| RAIN (mm) | 0.0 | 72.5 |
| WSPM (m/s) | 0.0 | 13.2 |

All are physically plausible for Beijing. PM10 topping out at 999.0 is worth a
Phase-1 look as a possible instrument ceiling rather than a true maximum.

**No range, minimum, maximum or any other value statistic was computed for
PM2.5, in any partition.** The target is reported by presence and absence
only — see §Target discipline in the protocol.

## 8. Reproducibility

`scripts/audit_dataset.py` was run twice; all six machine-readable outputs were
**byte-identical** across runs. Artefacts are sorted, carry no wall-clock
timestamp and no filesystem mtime, and hash file content rather than metadata.
