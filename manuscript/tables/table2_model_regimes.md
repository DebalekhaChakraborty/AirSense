# Table 2 — Model families and information regimes

**Evidence class:** C5 (methodological / governance)
**Source artifact:** `artifacts/phase10_information_regime_ledger.json`
**Source SHA-256:** `6da508c559ba01446237af1d5fdc7c4cb094b4d446693c9ea33616345177bdb0`
**Manuscript section:** 3. Data and protocol

| Regime | Available information |
|---|---|
| R0 | meteorology, wind direction, rain occurrence, calendar and masks; no pollutant history |
| R1 | R0 plus target-station PM2.5 history, its mask and its gap age |
| R2 | R1 plus co-pollutant history (PM10, SO2, NO2, CO, O3) with masks and gap ages |
| R3 | R2 plus cross-station history |

**Causality rules, identical in every regime**

| Rule | Status |
|---|---|
| Future observed meteorology | PROHIBITED |
| Future observed pollutants | PROHIBITED |
| Deterministic future calendar | ALLOWED |
| Historical observations | only timestamps <= forecast origin |
| Rolling PM2.5 reveal | permitted once a timestamp becomes past relative to the forecast origin |

**Caption (draft).** Nested information regimes evaluated over a common sample
universe, so that a difference between regimes is a difference in information
rather than in data.

**Footnotes and qualifiers.**

- Regimes are definitional, not results. This table asserts what each regime may
  observe, never how well it performs.
- The deterministic future calendar is permitted because it is knowable at the
  forecast origin without observing anything.
- Enforcement is structural. Across
  1,233,036 index checks during final-test
  prediction generation there were
  **0** future-target access
  violations.
