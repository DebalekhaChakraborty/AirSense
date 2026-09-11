# AirSense V2 — Phase 7 Record

**Robustness, final model selection and pre-test freeze.**
**Executed on branch `master` from `09942928…`.**

Development research only. The locked 2016-03-01 → 2017-02-28 test remains
**sealed**: no test target was read, no test prediction generated, no test
metric computed. `legacy` was not touched. No architecture search, no feature
change, no context change, no threshold change.

Results feed [`artifacts/phase7_pretest_freeze.json`](../artifacts/phase7_pretest_freeze.json).

---

## 1. Gates

All seven upstream validators passed before any Phase-7 artifact existed —
Phase-0 24 gates, Phase-1 31, Phase-2 58, Phase-3 47, Phase-4 74, Phase-5 95,
Phase-6 97, zero failures. **231 scientific artifacts were fingerprinted and
230 verified unchanged**, the single difference being the tooling registry
carrying the approved Phase-6 ownership correction.

## 2. Frozen candidate set

`artifacts/phase7_candidate_freeze.json` was written **before any robustness
experiment ran**, so the set could not be edited once results existed.

| Candidate | Family | Regime | Development macro MAE |
|---|---|---|---:|
| B3_R2 | classical | R2 | 30.111722 |
| TCN_R1 | neural | R1 | 30.688629 |
| GRU_R1 | neural | R1 | 30.912193 |
| SA_R3 | spatiotemporal | R3 | 31.532702 |
| B0 | classical | none | 33.127699 |

Excluded with reasons recorded: iTransformer_R1/R2, and every dominated
variant. **SA_R2 was excluded as a candidate** but carried through the seed
sweep as the *paired control*, because §3A asks whether the h = 24 reversal
survives reseeding and that question only exists in pairs.

## 3. Multi-seed robustness

Seeds 42–46. Seed 42 was **not retrained**: its frozen Phase-4 and Phase-6
prediction arrays were re-scored through the Phase-7 code path, and reproduced
the published values exactly. 16 refits, ~4.3 h.

| Model | Mean | SD | Range |
|---|---:|---:|---:|
| B3_R2 | 30.1117 | deterministic | — |
| TCN_R1 | 30.5437 | 0.3387 | 0.80 |
| GRU_R1 | 30.7591 | 0.1233 | 0.29 |
| SA_R3 | 31.5452 | 0.4247 | 1.05 |
| B0 | 33.1277 | deterministic | — |

**Two conclusions from earlier phases do not survive contact with reseeding.**

1. **TCN_R1's severe-tail weakness was largely one unlucky seed.** Phase 4
   published 152.53 and called it clearly the worst learned model in the tail.
   Across seeds it ranges 123.1–152.5 with a mean of **137.8** — indistinguishable
   from B3_R2's 136.6. The published figure was its worst of five.
2. **Rank order among the learned models is inside seed noise.** TCN_R1 and
   GRU_R1 differ by 0.22 in mean while TCN_R1 alone spans 0.80. The Phase-6
   gap between TCN_R1 and SA_R3 (0.84) is about one seed-width of TCN_R1.

**Severe-tail measurements are far less seed-stable than overall ones.** Spread
across seeds: GRU_R1 7.4, TCN_R1 29.4, SA_R3 35.2, SA_R2 34.8, against overall
spreads near 1. Any severe-tail claim from a single seed should be treated as
provisional — including several made in Phases 4 and 6.

## 4. H4 under reseeding — the paired contrast

| Seed | SA_R2 | SA_R3 | Gain | Severe gain |
|---:|---:|---:|---:|---:|
| 42 | 32.2398 | 31.5327 | +0.707 (+2.19%) | +21.43 (+15.02%) |
| 43 | 33.2130 | 31.2211 | +1.992 (+6.00%) | +30.26 (+18.46%) |
| 44 | 32.3441 | 31.2803 | +1.064 (+3.29%) | +5.12 (+3.42%) |
| 45 | 31.8509 | 31.4186 | +0.432 (+1.36%) | +12.52 (+8.91%) |
| 46 | 32.9450 | 32.2733 | +0.672 (+2.04%) | +18.76 (+10.71%) |

**Cross-station context wins on 5 of 5 seeds overall (mean +0.973, sd 0.61)
and on 5 of 5 in the severe tail (mean +17.6, sd 9.5, mean +11.3%).**
Under-prediction is lower for SA_R3 on every seed. H4 is the most robust
positive finding in the study, and Phase 6's published +0.707 was the
second-smallest of the five — it *understated* the effect.

## 5. The h = 24 reversal — §3A

Phase 6 reported cross-station context as harmful at h = 24 (−1.31%). **It does
not survive reseeding.**

| Seed | 42 | 43 | 44 | 45 | 46 |
|---|---:|---:|---:|---:|---:|
| R3 gain at h = 24 | −1.31% | +0.88% | +1.48% | −0.15% | +0.98% |

Mean **+0.38%**, sd 1.13, sign positive on 3 of 5. The effect is
indistinguishable from zero, and the Phase-6 finding was a seed artifact of the
published run. The honest statement is that **cross-station information neither
reliably helps nor reliably hurts at 24 hours**, in contrast to h = 6 where it
helps consistently.

This is exactly what the phase was for, and it corrects a specific published
claim rather than confirming it.

## 6. Spatial robustness — §4

Coordinates were acquired from a licensed, citable source and **verified
against it**: Wu, Xie & Kang (2022), npj Urban Sustainability 2:23,
doi:10.1038/s42949-022-00070-0, Table 1, CC BY 4.0. All 12 stations matched by
exact case-sensitive name with **no alias inference**, and all 12
latitude/longitude pairs match the published table exactly. Full provenance and
limitations: [`PHASE7_COORDINATE_ACQUISITION_AUDIT.md`](PHASE7_COORDINATE_ACQUISITION_AUDIT.md).

Coordinates were used for **analysis only**: never a model input, never a
graph, never a selection criterion, and acquired after every model was frozen.

| Diagnostic | Value |
|---|---:|
| Spearman(distance, attention) | **+0.136** |
| Spearman(distance, training PM2.5 correlation) | **−0.938** |
| Spearman(attention, training PM2.5 correlation) | −0.137 |
| Nearest station is the most-attended source | **1 of 12** |
| Mean attention by distance quartile (near→far) | 0.080, 0.076, 0.087, 0.088 |

**There is no proximity effect.** Attention very slightly *increases* with
distance. The −0.938 distance/correlation relationship confirms the coordinates
are real and correctly mapped — nearer stations really are more correlated —
which makes the absence of a distance effect in attention a finding rather than
a data error. SA_R3's advantage does not come from attending to neighbours.

## 7. Leave-one-station-out — §5

Twelve folds, each withholding one station's ~68,000 training targets from the
loss while leaving its history visible to the shared encoder, then evaluating
on exactly that station. Seed 42 throughout, ~2.5 h.

**Mean relative penalty +0.134%. Median +0.129%. Five of twelve stations are
*better* when held out.** Mean absolute penalty +0.015 µg/m³; range −2.61%
(Wanliu) to +3.94%.

Withholding every training target for a station costs essentially nothing.
**SA_R3 learned transferable regional structure, not station identity.** The
severe tail costs slightly more, +3.4 µg/m³ mean with 3 of 12 better, but the
overall result is unambiguous.

**Required interpretation.** This experiment measures *held-out
target-station supervision transfer with causal historical observations from
the held-out station available*. It must **not** be described as "completely
unseen-station forecasting": only the station's training targets were
withheld, while its historical observation stream remained available to the
shared encoder. A fully unseen station — one whose observations never reach
the model at all — is Phase-8-or-later work and was not tested here. This
wording is frozen in the pre-test addendum.

## 8. Final selection

The rule was frozen before any robustness result existed: primary is
development macro station-horizon MAE, with severe MAE, RMSE and R² as
secondary evidence that does not override the primary endpoint, and selection
must consider mean performance, seed variance and severe-tail behaviour, never
a single best seed.

Applied to seed means, **B3_R2 is selected**: 30.1117, the lowest, and
deterministic so it carries no seed risk at all.

**The tension in that choice is recorded rather than hidden.** B3_R2 is fourth
of five on the severe endpoint (136.6). Persistence beats it by 26% there
(101.3) and GRU_R1 by 15% (115.8). The severe tail is the failure that
motivated V2, and the selected model is not good at it. The frozen rule makes
the primary metric decisive, and changing that rule after seeing the results is
precisely what the phase forbids — so the selection stands as the rule dictates,
and whether to carry a second candidate into the test is a decision for human
review, not a post-hoc edit here.

## 8a. Pre-test confirmatory hierarchy (addendum)

After the pre-test freeze was written, a human-approved protocol addendum
fixed the confirmatory hierarchy for the sealed test.
`artifacts/phase7_pretest_freeze.json` was **not rewritten**; the addendum is a
separate artifact referencing it by digest.

| Rank | Model | Role | Prespecified endpoint |
|---:|---|---|---|
| 1 | **B3_R2** | primary confirmatory model | macro station-horizon MAE |
| 2 | **GRU_R1** | secondary confirmatory severe-tail model | severe MAE at the frozen 244.0 µg/m³ threshold |
| 3 | B0 | reference persistence baseline, not a selected learned model | — |

GRU_R1 is added because severe-pollution forecasting was a predeclared
secondary objective throughout V2, and five-seed robustness shows it
**15.24% better than B3_R2 in the severe tail** while costing **2.15%**
overall. It does **not** replace B3_R2 as the primary selected model.

TCN_R1, SA_R3, both iTransformers, SA_R2, B1, B2 and the remaining B3 regimes
are excluded from the confirmatory set and may appear only in historical
development tables. **No model may be promoted after test access.**

Multiple-comparison discipline is frozen: B3_R2's overall result is the primary
confirmatory result, GRU_R1's severe result is a prespecified secondary
confirmatory result, B0 is a benchmark, and every other station, horizon or
severe diagnostic is descriptive unless frozen before the test is opened.

## 9. Two corrections recorded

**A validator ownership generalisation.** Phase-7 artifacts tripped
earlier-phase ownership scans for the third consecutive phase. On approval, the
hand-maintained prefix lists in the Phase-0/1/2 validators were replaced with a
structural rule — later-phase-owned means a phase number greater than the
validator's own. Verified path by path over all 336 repository paths: **zero
paths lost exclusion**, every validator still scans its own phase, gate counts
unchanged, all seven pass. No further approvals are needed through Phase 11.

**A reporting error in this phase.** An earlier draft of the coordinate audit
asserted that no usable source existed and that §4 could not be performed. That
was wrong: the acquisition had already been completed and verified, and the
draft reflected a failed second search rather than the evidence. The audit
document carries a correction notice rather than a silent rewrite.

**A recovered crash.** The multi-seed runner computed all 20 rows and then died
writing its output, because it derived metric column names by scoring a
ten-sample slice containing no severe hour. All 16 prediction arrays were
already on disk; the defect was fixed, a `--from-saved` rescoring path added,
and every recovered value reproduces the original run's log exactly. No model
was retrained.

## 10. Limitations

**Five seeds is a small sample.** Standard deviations over five fits are
themselves uncertain. The h = 24 conclusion — an effect indistinguishable from
zero — is the safest reading of five points, not a measurement of zero.

**Leave-one-station-out is target transfer only**, as described in §7.

**Coordinates describe the 2018–2020 network**, after the 2013–2017
observation period, and are reported to three decimals. Neither materially
affects a 6–48 km distance analysis, but neither is proof that no station moved.

**Development only.** One validation year, one city, one network. Nothing here
is held-out performance, and no hypothesis is confirmed.

## 11. Pre-test discipline

No model was refit on full development data. The test was not opened, no test
target accessed, no test prediction generated, no test metric computed. The
pre-test freeze records the selection and stops there.

## 12. Git

No write operation: no `git add`, `git commit`, `git push`, reset, rebase,
merge, tag, branch creation or history rewrite. All Phase-7 changes were left
unstaged. Inspection was read-only.
