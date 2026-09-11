# AirSense V2 — Chronos-2 Pretraining-Overlap Audit

**Protocol Phase 5, §12. Conducted BEFORE any checkpoint download or
inference.**

> ## CLASSIFICATION: **C — DIRECT OR MATERIAL OVERLAP FOUND**
>
> **Chronos-2 was therefore NOT downloaded and NOT run.** Under the phase's own
> rule, classification C requires completing the iTransformer track and
> stopping for human review. No Chronos weights were fetched, no Chronos
> inference was performed, and no Chronos metric exists.

---

## 1. What was audited

Whether Amazon's Chronos-2 pretraining corpus contains data overlapping the
AirSense V2 study: air-quality series, Beijing series, PRSA, the UCI Beijing
Multi-Site data set (DOI `10.24432/C5RK5G`), the twelve AirSense station names,
or 2013–2017 Beijing station observations.

Sources consulted, all authoritative:

| Source | What it gave |
|---|---|
| Chronos-2 paper, arXiv 2510.15821 (HTML full text, Appendix A, Table 6) | the enumerated pretraining corpus |
| Monash Time Series Forecasting Archive entry for KDD Cup 2018 | series count and series lengths |
| KDD Cup 2018 competition / archive records | cities, stations, date range, pollutants |
| `huggingface.co/amazon/chronos-2` model card | licence, size, architecture |

## 2. The finding

Chronos-2's paper lists 20 named real datasets in its pretraining corpus
(Table 6, Appendix A):

> Electricity · **KDD Cup (2018)** · M4 (Daily, Hourly, Monthly, Weekly) ·
> Mexico City Bikes · Pedestrian Counts · Solar · Taxi · Uber TLC · USHCN ·
> Weatherbench · Wiki · Wind Farms · Temperature-Rain · London Smart Meters ·
> Alibaba Cluster Trace (2018) · Azure VM Traces (2017) · Borg Cluster Data
> (2011) · LargeST (2017) · Q-Traffic · Buildings 900K

At first reading none of these looks like an air-quality corpus. **KDD Cup
(2018) is one.** The KDD Cup 2018 challenge — "KDD Cup of Fresh Air" — was an
air-quality forecasting competition, and the standard forecasting dataset
derived from it is:

| Property | Value |
|---|---|
| Cities | **Beijing (35 stations)** and London (24 stations) |
| Period | **2017-01-01 → 2018-03-31** |
| Frequency | hourly |
| Measurements | **PM2.5**, PM10, NO2, CO, O3, SO2 |
| Series | 282 (270 in the gap-filled variant) |

The Monash archive's own figures corroborate the period exactly: it records a
maximum series length of **10,920**, and 2017-01-01 00:00 → 2018-03-31 23:00
is **precisely 10,920 hours**.

### 2.1 Station overlap

The KDD Cup 2018 Beijing station set includes stations named
**Aotizhongxin** and **Dongsi** among others — these are AirSense stations, not
merely nearby ones. The Beijing municipal network sampled by KDD Cup 2018
covers essentially all twelve AirSense sites (Aotizhongxin, Changping,
Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan,
Wanliu, Wanshouxigong).

### 2.2 Temporal overlap with the sealed test

| Window | Period |
|---|---|
| KDD Cup 2018 series | 2017-01-01 → 2018-03-31 |
| AirSense **locked final test** | 2016-03-01 → 2017-02-28 |
| **Overlap** | **2017-01-01 → 2017-02-28** = **1,416 hours per station** |
| AirSense development validation | 2015-03-01 → 2016-02-29 — **no overlap** |

So the overlap is with the **sealed final test period**, not with the
development validation period.

## 3. Why this is classification C and not B

Classification B is for related-domain or related-geography data where no
direct series overlap is established. This is not that. Four properties align
simultaneously:

- **same city** — Beijing;
- **same monitoring stations**, by name;
- **same target variable** — hourly PM2.5;
- **overlapping calendar window** — 1,416 hours inside AirSense's locked test.

That is a direct series overlap on the exact quantity V2 forecasts, in the
exact period V2 has sealed for its single final evaluation. The protocol's
instruction not to infer absence of overlap merely because dataset names differ
cuts the other way here: the names *match* once the KDD Cup 2018 dataset's
actual content is looked up rather than assumed from its title.

## 4. Honest statement of what remains uncertain

Two things this audit could **not** establish from official sources:

1. **Which variant** of "KDD Cup (2018)" Chronos-2 ingested — the Beijing +
   London forecasting dataset, a London-only subset, or another packaging. The
   paper names the corpus but does not enumerate its series.
2. **Whether the 2017-01 → 2017-02 window specifically** survived any internal
   filtering. The paper states the corpus was checked against GIFT-Eval *test*
   portions and that it "does include partial overlap with the training
   portions of some GIFT-Eval datasets" — no comparable statement exists for
   the UCI Beijing Multi-Site data, which is not a Chronos evaluation
   benchmark and would not have been screened against.

Under that residual uncertainty the conservative classification is still C.
Treating an unresolved question in the model's favour is exactly the error the
gate exists to prevent, and the specificity of the station-name and
calendar-window match makes B untenable.

## 5. Consequence, per §12

- **Chronos-2 was not downloaded.** No checkpoint, no config, no weights.
- **Chronos-2 was not run.** `Chronos2_R1` and `Chronos2_R2` do not exist; no
  prediction array, no metric, no compute report.
- The iTransformer track was completed as instructed.
- **Stopped for human review.**

No Chronos result may be described as a clean or contamination-free zero-shot
benchmark on this study, because on the evidence above it would not be one.

## 6. What a human review might consider

Recorded as options, not recommendations, and none of them taken here:

1. **Drop Chronos-2 from V2 entirely** and record the reason. The cleanest
   outcome, and it costs a comparison rather than a result.
2. **Evaluate on validation only, permanently barred from the final test.**
   The validation period genuinely does not overlap, so a development-only
   Chronos number is defensible — but it could never advance to Phase 9, which
   makes it a comparison that leads nowhere.
3. **Seek an older Chronos checkpoint** whose corpus provably excludes KDD Cup
   2018, and audit that corpus the same way.
4. **Re-examine the sealed test boundary.** V2's test window was frozen in
   Phase 0 on calendar grounds, long before any foundation model was
   considered. Moving it now to dodge a contaminated corpus would be
   performance-motivated boundary selection and should be refused.

Option 4 is noted only to be argued against.
