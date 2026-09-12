# AirSense V2 — Manuscript Readiness Review

An assessment of the Phase-11 draft against submission standards. This is not a
summary of what went well. Where the draft is not ready, it says so.

**Ratings:** READY · MINOR REVISION NEEDED · MAJOR REVISION NEEDED

---

## Summary

| Category | Rating |
|---|---|
| Scientific coherence | READY |
| Methodological completeness | READY |
| Claim discipline | READY |
| Negative-result visibility | READY |
| Reproducibility | READY |
| Limitations coverage | READY |
| Figure and table sufficiency | MINOR REVISION NEEDED |
| Discussion depth | MINOR REVISION NEEDED |
| Citation completeness | **MAJOR REVISION NEEDED** |
| Overall journal-readiness | **MAJOR REVISION NEEDED** |

**The manuscript cannot be submitted in its current state.** One category blocks
submission outright, and it is not a scientific problem.

---

## Blocking issue

### Citation completeness — MAJOR REVISION NEEDED

Thirteen `REF-NEEDED` placeholders remain, covering the entire Related Work
section and the one epidemiological claim in the introduction. No journal will
accept a manuscript in this state.

This is a deliberate outcome, not an oversight: fabricating a bibliography would
have been faster and is prohibited. But the consequence must be stated plainly —
**Section 2 currently has no verified citations at all**, and a reader cannot
evaluate how the work is positioned.

Two placeholders carry more weight than the rest:

- **The PM2.5 health claim (1.1).** The only epidemiological statement in the
  manuscript. If it cannot be supported by a verified source, the sentence
  should be reduced to a non-quantitative motivation rather than cited loosely.
- **The iTransformer reference (2.2).** The manuscript describes an
  *iTransformer-style* in-repository architecture. Without the reference, the
  style attribution is unanchored, and a reviewer may reasonably ask what
  exactly was implemented.

**Required before submission:** verify all thirteen against publisher records,
recording authors, title, venue, year and DOI plus the verification source for
each. Snippet-level verification is not acceptable.

---

## Non-blocking weaknesses

### Figure and table sufficiency — MINOR REVISION NEEDED

- Figures are **existing frozen development-era PNGs**, selected unmodified.
  They are scientifically correct but were not drawn to any journal's style
  guide. Font sizes, panel dimensions and colour choices have not been checked
  against a target venue, and no venue has been chosen.
- There is **no schematic figure** of the study design. For a manuscript whose
  central methodological contribution is a sealed, staged evaluation protocol,
  a design diagram would carry real weight. Creating one is a presentation
  change, not new science, but it is out of scope for this phase.
- Table 3 has fourteen rows, eleven of which are development-only models. A
  reviewer skimming could mistake it for a results table. The caption says
  otherwise; the visual design does not yet reinforce it.

### Discussion depth — MINOR REVISION NEEDED

- The discussion explains the severe-tail failure well but stays descriptive. It
  does not engage with the atmospheric-science literature on episode formation,
  because that literature is not yet cited. Filling the citation gaps will
  likely require rewriting parts of the discussion rather than merely inserting
  references.
- The comparison between this study's information-regime findings and published
  results elsewhere cannot be written at all until Related Work exists.

---

## Categories rated READY, with the reasoning

**Scientific coherence.** The research question, hypotheses, design, results and
conclusion form one argument. The headline carries both halves — the
improvement over persistence and the unresolved severe tail — in the abstract,
the results and the conclusion.

**Methodological completeness.** Preprocessing, eligibility, regimes,
architectures, hyperparameter grids, selection rules and determinism controls
are all specified, with the dense detail moved to supplementary methods. A
reader could reimplement the study.

**Claim discipline.** Every quantitative statement carries an evidence class.
No C3 finding is presented as C1. All 235 distinct numeric values in the
manuscript trace to a hash-pinned artifact, verified twice through independent
code paths.

**Negative-result visibility.** The severe-tail result appears in the abstract,
in Table 5, in its own results subsection, in the discussion, in the limitations
and in the conclusion. It is not softened anywhere, and Table 6 — the least
flattering table — is placed in the main paper rather than the supplement.

**Reproducibility.** Git anchors, raw-data digests, environment pins, seeds, the
test-opening chronology and audit counts are all stated. The processed data
layer is regenerable from tracked raw data.

**Limitations coverage.** Seventeen limitations, stated in full rather than
compressed into a paragraph, including several that bound the central claims
directly.

---

## Anticipated reviewer objections

Worth preparing for, though none is a defect in the current draft:

1. **"The confirmatory design is thin."** Three models and two endpoints. A
   reviewer may argue the sealed test was under-used, since no paired regime
   contrast was evaluated there, leaving H1–H4 development-supported only. The
   manuscript states this, but a reviewer may still see it as a design
   limitation rather than a disclosure. It is both.
2. **"Where is the uncertainty quantification?"** None was predeclared, so none
   is reported. This is defensible and is stated, but it will be raised.
3. **"Single city."** Generalisation is untested and is listed as the second
   limitation.
4. **"Is a negative tail result publishable?"** Venue selection matters here.
   The study's contribution is as much the evaluation protocol as the accuracy
   number, and venues differ in how they weigh that.
5. **"Why was no foundation model run?"** Fully documented as a governance
   outcome, but reviewers may push for a contaminated-but-reported baseline.
   Running one would break the study's own preregistered policy.

---

## Required before submission

1. **Resolve all thirteen citation placeholders.** Blocking.
2. Choose a target venue and convert to its template.
3. Replace title, author, affiliation, email, ORCID, competing-interests,
   funding and author-contribution placeholders.
4. Approve or replace the working title.
5. Decide whether to add a study-design schematic.
6. Re-render figures to the chosen venue's style, under a new freeze.

Items 2–6 are presentation decisions reserved for human review. Item 1 is
substantive and blocks submission.
