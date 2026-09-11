# AirSense V2 — Phase 7 Coordinate Acquisition Audit

**Protocol Phase 7, §4. Outcome: coordinates ACQUIRED from a licensed,
citable source, verified against it, and used for descriptive analysis only.**

Audit date: 2026-09-10. Independently re-verified against the source on the
same date.

---

## 1. Why acquisition was needed

The official UCI package contains no station coordinates. Its schema is
`No, year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP,
RAIN, wd, WSPM, station` — station is a name, with no latitude, longitude or
elevation, and the archive ships no metadata file. Phase 1 recorded this and
deferred acquisition to the spatiotemporal work; Phase 6 deliberately required
none and answered the *relational* question without geometry.

## 2. Source

| Field | Value |
|---|---|
| Citation | Wu, L., Xie, J. & Kang, K. *Changing weekend effects of air pollutants in Beijing under 2020 COVID-19 lockdown controls.* npj Urban Sustainability **2**, 23 (2022) |
| DOI | [10.1038/s42949-022-00070-0](https://doi.org/10.1038/s42949-022-00070-0) |
| Open-access copy | [PMC9510312](https://pmc.ncbi.nlm.nih.gov/articles/PMC9510312/) |
| Table used | Table 1, *Descriptions of 32 stations in Beijing* |
| Underlying monitoring authority | Beijing Environmental Protection Monitoring Center |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Retrieval date | 2026-09-10 |

The licence permits use, sharing and adaptation with attribution, which this
document provides.

## 3. Mapping rule

**Exact, case-sensitive match between the 12 frozen AirSense station names and
the source table's station names. No alias inference of any kind.** All 12
matched directly; none required a judgement call such as
`Changpinzhen` → `Changping`. Had any station required an inferred alias, the
acquisition would have been abandoned.

| AirSense station | Latitude | Longitude |
|---|---:|---:|
| Aotizhongxin | 39.982 | 116.397 |
| Changping | 40.217 | 116.230 |
| Dingling | 40.292 | 116.220 |
| Dongsi | 39.929 | 116.417 |
| Guanyuan | 39.929 | 116.339 |
| Gucheng | 39.914 | 116.184 |
| Huairou | 40.328 | 116.628 |
| Nongzhanguan | 39.937 | 116.461 |
| Shunyi | 40.127 | 116.655 |
| Tiantan | 39.886 | 116.407 |
| Wanliu | 39.987 | 116.287 |
| Wanshouxigong | 39.878 | 116.352 |

Decimal degrees exactly as reported; **no transformation, rounding or
interpolation was applied**.

## 4. Independent verification

The stored artifact was re-checked against the source article after the
analysis was run. **All 12 latitude/longitude pairs matched the published
Table 1 exactly**, and the licence statement was confirmed as CC BY 4.0.

A second, internal check supports the mapping without reference to the source:
the Spearman correlation between inter-station distance and the frozen
Phase-1 training-only pairwise PM2.5 correlation is **−0.938**. Nearer stations
are far more correlated. Mismatched or fabricated coordinates would not
reproduce that structure, so the geometry is consistent with the data the
study already had.

## 5. Constraints on use

Recorded in `artifacts/phase7_station_coordinates.json` and enforced by how
the analysis was run:

- `model_input: false` — **no coordinate ever entered a model**;
- `graph_constructed: false` — no adjacency, no neighbour selection, no edge;
- `causal_interpretation_permitted: false`;
- coordinates arrived **after** every Phase-6 model was trained and frozen, so
  they cannot have influenced any architecture, weight or prediction.

## 6. Limitations, stated

1. The source table describes the network as used for **2018–2020**, after
   AirSense's 2013–2017 observation period. It is not proof that no station
   relocated between the two.
2. Coordinates are reported to **three decimal places**, which bounds distance
   precision to roughly ±100 m — immaterial for inter-station distances of
   6–48 km, but recorded.
3. A single secondary source is used. It is licensed, citable and matches
   exactly, but it is not the monitoring authority's own register.

## 7. Correction notice

An earlier draft of this document asserted that no usable coordinate source
existed and that §4 could not be performed. **That assertion was wrong.** It
was written after the acquisition had already been completed and verified, and
reflected a failed second search rather than the state of the evidence. The
acquisition described above stands, is verified, and is the record. This notice
is kept rather than deleted so the error remains legible.
