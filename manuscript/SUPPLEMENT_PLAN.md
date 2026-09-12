# AirSense V2 — Main Paper vs Supplement Plan

A governing rule first, because it is the one most easily violated in practice:

> **No result is moved to the supplement because it is unfavourable.** The
> severe-tail negative result, the under-prediction rates, the detection-recall
> collapse and the unclipped negative predictions all stay in the main paper.
> The supplement carries detail, not disappointment.

---

## Main paper

### Tables

| Table | Content | Class | Why main |
|---|---|---|---|
| Table 1 | Dataset and chronological split | C5 | The sealed design is the study's central methodological claim. |
| Table 2 | Information regimes and causality rules | C5 | Regime definitions are needed to read every later result. |
| Table 3 | Development-validation comparison | C2 | Establishes what motivated the frozen roles. |
| Table 4 | Multi-seed robustness | C2 | Carries H4 evidence and the seed-stability caveat that bounds every severe claim. |
| Table 5 | Locked final-test results | C1 | The only confirmatory table. |
| Table 6 | Post-test severe and error diagnostics | C3 | Carries the negative result in detail; must not be relegated. |

### Figures

| Figure | Content | Class |
|---|---|---|
| Figure 9 | Locked final-test primary endpoint | C1 |
| Figure 10 | Locked final-test secondary endpoint | C1 |
| Figure 2 | Error growth with forecast horizon | C3 |
| Figure 4 | Severe-tail error by horizon | C3 |

Four figures, deliberately. The two confirmatory figures carry the result; the
two exploratory figures carry the horizon structure that explains it. Everything
else is detail.

---

## Supplement

### Figures

| Figure | Content | Class | Why supplement |
|---|---|---|---|
| Figure 1 | Development vs locked test | C3 | Useful context; the numbers are already in Tables 3 and 5. |
| Figure 3 | Station-level error structure | C3 | Descriptive; ordering is associated with site character, not model behaviour. |
| Figure 5 | Severe under-prediction rate by horizon | C3 | The headline value is already stated in the main text. |
| Figure 6 | Residual autocorrelation at frozen lags | C3 | Detail behind a claim fully stated in the main text. |
| Figure 7 | Error by concentration stratum | C3 | Supports the regression-to-the-mean description. |
| Figure 8 | Error at severe-event peaks | C3 | Event-level detail behind the main-text summary. |

### Tables and material

- Station-level metric breakdowns for all twelve stations and four horizons.
- Hour-of-day error tables. The diurnal effect is small next to the horizon and
  concentration effects and does not earn main-text space.
- Full residual autocorrelation tables at every frozen lag.
- All twelve leave-one-station-out folds with per-station penalties.
- Coordinate-distance diagnostics, including distance quartiles and the
  nearest-station attention counts.
- Negative-prediction detail by horizon and station.
- Complete artifact SHA-256 inventory and per-phase manifests.
- Validator inventory, gate counts and the append-only tooling-registry chain.
- Foundation-model eligibility audit in full.
- The full feature schema and hyperparameter grids.

---

## Placement decisions worth defending

1. **Table 6 stays in the main paper.** It is the study's least flattering
   table. Moving it would leave the main text asserting a limitation the reader
   cannot inspect.
2. **Figure 10 stays in the main paper.** It is the figure on which the
   reference benchmark beats both learned models. It belongs next to Figure 9,
   not behind it.
3. **Governance mechanics go to the supplement.** Validators, registries and
   manifests are how the study is verifiable, not what it found. They are
   summarised in the reproducibility statement and detailed in
   `manuscript/SUPPLEMENTARY_METHODS.md`.
4. **Station-level detail goes to the supplement.** With near-identical ordering
   across models it is descriptive of the network, not of the methods.
