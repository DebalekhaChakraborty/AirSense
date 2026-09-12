# AirSense V2 — Manuscript Language Audit

Every prohibited or risky phrase from the publication claim guide was searched
in `manuscript/AirSense_V2_Manuscript.md` with **word-boundary** matching.

> Word boundaries are not a detail here. A naive substring search for `solved`
> matches `unresolved`, which is the exact opposite of the banned claim, and
> `proves` matches `improves`. A substring-based audit would report failures
> where the manuscript is correct, and would train the reader to ignore it.

---

## Terms searched and found absent

The following appear **nowhere** in the manuscript:

`state-of-the-art` · `SOTA` · `novel` · `novelty` · `breakthrough` · `proves` ·
`proved` · `proven` · `solve` · `solves` · `solved` · `eliminate` ·
`eliminates` · `causes` · `drives` · `spatial reasoning` ·
`pollution propagation` · `operationally ready` · `early-warning ready` ·
`all hypotheses confirmed` · `foundation models failed` · `outperforms V1` ·
`unseen-station` · `best model` · `guarantee` · `statistically significant`

Terms deliberately avoided in drafting rather than removed afterwards:
`clearly`, `obviously`, and every construction attributing a mechanism with
`because of`. Causal attribution uses **associated with** throughout, per the
claim guide's terminology table.

---

## Terms found, with a decision for each

### 1. `winner` — 1 occurrence — **KEPT**

> "…the roles frozen before opening are the roles reported, and no overall
> **winner** label is assigned."

- **Context:** Section 9.3, locked final-test results.
- **Decision:** Keep. The frozen model-role ledger records
  `winner_label_assigned: false` with an explicit rationale, and the manuscript
  is obliged to disclose that. The prohibition is on *asserting* a winner, not
  on stating that none was named.
- **Replacement required:** none. Removing the sentence would suppress a
  governance disclosure.
- **Enforcement:** the Phase-11 validator treats `winner` as
  negation-required rather than absolutely banned: it fails on any sentence
  using the term without a negation marker.

### 2. `p-value` and `significance` — 1 sentence — **KEPT**

> "None was predeclared before the test was opened, so no interval, no
> **p-value** and no **significance** test appears anywhere in this study."

- **Context:** Section 5, evaluation metrics.
- **Decision:** Keep. This is the manuscript's explicit statement that no
  inferential procedure is reported, which is a required disclosure rather than
  a prohibited claim.
- **Replacement required:** none.

### 3. `confidence interval` — 1 occurrence — **KEPT**

> "No formal inferential **confidence intervals** were predeclared for the
> locked test, so none is reported…"

- **Context:** Section 12, limitation 8.
- **Decision:** Keep, for the same reason. The limitations ledger requires this
  disclosure verbatim in substance.
- **Replacement required:** none.

### 4. `robustly` — 1 occurrence — **KEPT**

> "H4 is **robustly** supported on development and robustness evidence, and was
> not independently confirmed on the locked final test…"

- **Context:** Section 8.2.
- **Decision:** Keep. This is the frozen hypothesis ledger's own status string
  (`ROBUSTLY SUPPORTED ON DEVELOPMENT / ROBUSTNESS EVIDENCE, BUT NOT
  INDEPENDENTLY CONFIRMED ON THE LOCKED TEST`), and the qualifier travels in the
  same sentence as required. Paraphrasing a frozen status string would be a
  silent restatement of hypothesis status.
- **Replacement required:** none.

---

## Structural claim checks

| Check | Result |
|---|---|
| Severe rule written as strictly `> 244.0` | Pass — no `>=` or `≥` form appears |
| Severe threshold described as training pooled P95 | Pass |
| Severe threshold disclaimed as non-regulatory | Pass |
| Residual convention stated as actual − prediction | Pass |
| Positive residual identified as under-prediction | Pass |
| LOSO uses the required long-form description | Pass |
| Cross-station attention never framed as transport or propagation | Pass |
| Attention-distance diagnostic reported (+0.136289) | Pass |
| No development-only model appears in the locked-test section | Pass |
| No V1 sentence carries a percentage or metric benchmark | Pass |
| No foundation-model performance is described | Pass |
| Chronos-2 framed as overlap risk, not demonstrated contamination | Pass |
| Both headline halves appear together in abstract and conclusion | Pass |
| Every C3 finding labelled exploratory | Pass |

---

## Audit conclusion

Four terms from the risk list survive in the manuscript. **All four are
negated disclaimers or frozen ledger wording, and all four are required
disclosures.** No prohibited assertion was found, and no replacement was
necessary.
