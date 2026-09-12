#!/usr/bin/env python3
"""AirSense V2 - Phase 11 independent verification.

A SEPARATE code path from scripts/validate_phase11.py. It shares no helper, no
constant table and no parsing routine with that validator: values are re-derived
here by reading the frozen artifacts directly and re-parsing the manuscript with
independently written extraction.

The point is redundancy, not coverage. If both agree, a single parsing bug is
unlikely to be the reason the manuscript looks correct.

Read-only. Exits non-zero on any disagreement.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript" / "AirSense_V2_Manuscript.md"

checks: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, bool(ok), detail))


def text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def jload(rel: str):
    return json.loads(text(rel))


def csv_rows(rel: str) -> list[dict]:
    """Deliberately hand-rolled, quote-aware CSV parsing."""
    raw = text(rel).splitlines()
    def split(line: str) -> list[str]:
        out, cur, q = [], "", False
        for ch in line:
            if ch == '"':
                q = not q
            elif ch == "," and not q:
                out.append(cur); cur = ""
            else:
                cur += ch
        out.append(cur)
        return out
    head = split(raw[0])
    return [dict(zip(head, split(r))) for r in raw[1:] if r.strip()]


def section_of(md: str, heading_prefix: str) -> str:
    """Independently written section slicer, driven by a heading prefix."""
    lines = md.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("## ") and ln[3:].strip().startswith(heading_prefix):
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return "\n".join(lines[start:end])


def main() -> int:
    md = MS.read_text(encoding="utf-8")
    lock = jload("artifacts/phase8_primary_results_lock.json")
    metrics = jload("artifacts/phase8_final_test_metrics.json")
    concl = jload("artifacts/phase10_canonical_conclusion.json")
    roles = jload("artifacts/phase10_model_role_ledger.json")
    hyp = jload("artifacts/phase10_hypothesis_ledger.json")["hypotheses"]
    ds = jload("artifacts/phase10_dataset_ledger.json")
    t5 = csv_rows("artifacts/phase10_table_locked_test.csv")

    abstract = section_of(md, "Abstract")

    # ---- 1-5: abstract headline values, re-derived and re-rounded -----------
    def r2(x: float) -> str:
        return f"{x:.2f}"

    check("1. Abstract primary result",
          r2(lock["primary_value"]) in abstract, r2(lock["primary_value"]))
    check("2. Abstract reference macro value",
          r2(lock["reference_primary_value"]) in abstract,
          r2(lock["reference_primary_value"]))

    improvement = 100.0 * (lock["reference_primary_value"] - lock["primary_value"]) \
        / lock["reference_primary_value"]
    check("3. Abstract relative improvement (independently recomputed)",
          r2(improvement) in abstract, f"recomputed {improvement!r}")
    check("3b. Recomputed improvement matches the frozen conclusion",
          abs(improvement - concl["b3r2_relative_improvement_vs_B0_pct"]) < 1e-12,
          f"{improvement!r} vs {concl['b3r2_relative_improvement_vs_B0_pct']!r}")

    check("4. Abstract secondary severe value",
          r2(lock["secondary_value"]) in abstract, r2(lock["secondary_value"]))
    check("5. Abstract reference severe value",
          r2(lock["reference_severe_value"]) in abstract,
          r2(lock["reference_severe_value"]))

    # ---- 6: severe n -------------------------------------------------------
    n = lock["secondary_severe_n"]
    check("6. Severe n appears with thousands separator",
          f"{n:,}" in md, f"{n:,}")
    check("6b. Severe n agrees across three frozen artifacts",
          n == metrics["severe_sample_count"] == concl["canonical_locked_results"]["severe_n"])

    # ---- 7-11: hypothesis wording fidelity ---------------------------------
    for hid in ("H1", "H2", "H3", "H4", "H5"):
        body = hyp[hid]
        check(f"{6 + int(hid[1])}. {hid} is not claimed as locked-test retested",
              body["locked_test_retested"] is False)
    h_sec = section_of(md, "1. Introduction")
    for hid in ("H1", "H2", "H3", "H4", "H5"):
        check(f"{hid} is stated in the manuscript introduction",
              re.search(rf"\*\*{hid}\.\*\*", h_sec) is not None)
    check("H3 is reported as model-dependent, not general",
          re.search(r"model-dependent", md, re.I) is not None)
    check("H4 is reported as not independently confirmed on the locked test",
          re.search(r"H4[^.]{0,200}not\s+\n?independently confirmed", md, re.I)
          or re.search(r"not\s+independently confirmed on the locked final test", md, re.I))
    check("H5 is not claimed as confirmed",
          not re.search(r"H5[^.]{0,80}\bconfirmed\b", md, re.I))

    # ---- 12: severe threshold definition -----------------------------------
    thr = ds["severe_threshold"]
    check("12. Severe threshold stated as the training pooled P95",
          re.search(r"training pooled P95", md) is not None)
    check("12b. Severe threshold value matches the dataset ledger",
          f"{thr}" in md and thr == lock["severe_threshold"] == 244.0)
    check("12c. Severe threshold disclaimed as non-regulatory",
          re.search(r"not\b[^.]{0,120}(regulatory|air-quality-index)", md, re.I) is not None)
    check("12d. Severe rule stated strictly greater-than",
          re.search(r">\s*244\.0", md) is not None
          and not re.search(r"(>=|≥)\s*244", md))

    # ---- 13: locked-test dates ---------------------------------------------
    lo, hi = ds["partitions"]["locked_test"].split(" to ")
    check("13. Locked-test start date appears", lo[:10] in md, lo[:10])
    check("13b. Locked-test end date appears", hi[:10] in md, hi[:10])
    for part in ("train", "validation"):
        a, b = ds["partitions"][part].split(" to ")
        check(f"13c. {part} window appears ({a[:10]})", a[:10] in md)

    # ---- 14: model-role hierarchy ------------------------------------------
    role_words = {
        "B3_R2": "primary confirmatory model",
        "GRU_R1": "secondary confirmatory severe-tail model",
        "B0": "reference benchmark",
    }
    for model, phrase in role_words.items():
        check(f"14. {model} is described as {phrase}",
              phrase.lower() in md.lower())
    check("14b. Role ledger still matches the manuscript hierarchy",
          roles["roles"]["B3_R2"].lower().startswith("primary")
          and roles["roles"]["GRU_R1"].lower().startswith("secondary")
          and roles["roles"]["B0"].lower().startswith("reference"))
    check("14c. No winner label is asserted",
          roles["winner_label_assigned"] is False
          and not re.search(r"\bis the winner\b|\boverall winner is\b", md, re.I))

    # ---- 15: no SA claim on the locked test --------------------------------
    locked_sec = section_of(md, "9. Locked final-test results")
    dev_only = set(roles["development_only_models"])
    leaked = sorted(m for m in dev_only if m in locked_sec)
    check("15. No development-only model appears in the locked-test section",
          not leaked, f"leaked={leaked}")
    sa_claims = [s for s in re.split(r"(?<=[.])\s+", re.sub(r"\s+", " ", md))
                 if re.search(r"\bSA_R[23]\b", s)
                 and re.search(r"locked (final )?test", s, re.I)
                 and not re.search(r"\bnot\b|\bexcluded\b|\bnever\b", s, re.I)]
    check("15b. No sentence claims a cross-station model was evaluated on the locked test",
          not sa_claims, f"hits={[s[:100] for s in sa_claims[:2]]}")

    # ---- 16: no foundation-model performance claim -------------------------
    gov = jload("artifacts/phase10_foundation_model_governance.json")
    check("16. Governance ledger still records zero executions",
          gov["foundation_models_executed"] == 0
          and gov["foundation_model_predictions_generated"] == 0)
    fm = [s for s in re.split(r"(?<=[.])\s+", re.sub(r"\s+", " ", md))
          if re.search(r"Chronos|TimesFM|Moirai|foundation model", s, re.I)]
    perf = [s for s in fm
            if re.search(r"\b(MAE|RMSE|accuracy|error of|scored|achieved|performed)\b", s)
            and not re.search(r"\bnot\b|\bnever\b|\bno\b|\bzero\b", s, re.I)]
    check("16b. No foundation-model performance is reported",
          not perf, f"hits={[s[:100] for s in perf[:2]]}")
    check("16c. Chronos-2 exclusion framed as overlap risk, not demonstrated contamination",
          re.search(r"did not demonstrate", md, re.I) is not None)

    # ---- 17: no direct V1/V2 metric comparison -----------------------------
    v1v2 = jload("artifacts/phase10_v1_v2_structural_ledger.json")
    check("17. Structural ledger still forbids direct metric comparison",
          v1v2["direct_metric_comparison_valid"] is False
          and v1v2["percentage_improvements_computed"] == 0)
    v1_sents = [s for s in re.split(r"(?<=[.])\s+", re.sub(r"\s+", " ", md))
                if re.search(r"\bV1\b", s)]
    v1_num = [s for s in v1_sents if re.search(r"\d+(\.\d+)?\s*%", s)]
    check("17b. No V1 sentence carries a percentage", not v1_num,
          f"hits={[s[:100] for s in v1_num[:2]]}")
    flat = re.sub(r"\s+", " ", md)
    check("17c. Manuscript states the V1/V2 comparison is not numerical",
          re.search(r"No numerical benchmark between them is valid", flat, re.I) is not None)

    # ---- 18: every Table 5 value, re-parsed --------------------------------
    numeric_cols = ["macro_station_horizon_MAE", "severe_MAE_gt_244", "micro_MAE",
                    "macro_station_horizon_RMSE", "severe_mean_residual",
                    "severe_underprediction_pct", "n_negative"]
    mismatches = []
    for row in t5:
        model = row["model"]
        src = metrics["models"][model]
        pairs = [
            ("macro_station_horizon_MAE", src["macro_station_horizon_MAE"]),
            ("severe_MAE_gt_244", src["severe_mae"]),
            ("micro_MAE", src["micro_MAE"]),
            ("macro_station_horizon_RMSE", src["macro_station_horizon_RMSE"]),
            ("severe_mean_residual", src["severe_mean_residual"]),
            ("severe_underprediction_pct", src["severe_underprediction_pct"]),
            ("n_negative", src["n_negative"]),
        ]
        for col, expected in pairs:
            if abs(float(row[col]) - float(expected)) > 1e-9:
                mismatches.append(f"{model}.{col}")
    check("18. Table 5 agrees with the frozen Phase-8 metrics artifact",
          not mismatches, f"mismatch={mismatches[:5]}")

    ms_table5 = [s for s in md.splitlines() if s.startswith("| B3_R2 |") or
                 s.startswith("| GRU_R1 |") or s.startswith("| B0 |")]
    check("18b. Manuscript Table 5 rows render all three frozen roles",
          len(ms_table5) >= 3, f"rows={len(ms_table5)}")
    for model in ("B3_R2", "GRU_R1", "B0"):
        v = metrics["models"][model]["macro_station_horizon_MAE"]
        check(f"18c. Manuscript renders {model} macro MAE as {v:.2f}", f"{v:.2f}" in md)

    # ---- 19: numeric traceability, independently re-extracted --------------
    trace = csv_rows("manuscript/MANUSCRIPT_CLAIM_TRACEABILITY.csv")
    declared = {row["claim_value"].strip() for row in trace}
    # Independent extraction: strip fences, headings, list markers and
    # digit-bearing identifiers, then take numeric literals.
    body, fence, refs = [], False, False
    for ln in md.splitlines():
        st = ln.strip()
        if st.startswith("```"):
            fence = not fence; continue
        if fence:
            continue
        if st.startswith("## "):
            refs = st[3:].strip() == "References"; continue
        if st.startswith("#") or refs:
            continue
        if re.fullmatch(r"\|[\s:|-]*\|", st):
            continue
        ln = re.sub(r"^\s*\d+\.\s+", " ", ln)
        ln = re.sub(r"\[REF-NEEDED:[^\]]*\]", " ", ln)
        for pat in (r"\b(?:Table|Figure|Section|Stage|Appendix)\s+\d+",
                    r"\b[A-Za-z]+_R\d\b", r"\bB[0-3]\b", r"\bR[0-3]\b",
                    r"\bC[1-5]\b", r"\bH[1-5]\b", r"\bPM2\.5\b", r"\bPM10\b",
                    r"\bSO2\b", r"\bNO2\b", r"\bO3\b", r"\bCC BY 4\.0\b",
                    r"\bChronos-2\b", r"\bTimesFM-3\.0\b",
                    r"\bMoirai-2\.0-R-small\b", r"\bAirSense V[12]\b", r"\bV[12]\b",
                    r"\bSHA-256\b", r"\bx86-64\b", r"\bUTF-8\b",
                    r"10\.\d{4,9}/[^\s)\],;]+", r"arXiv:\d+\.\d+"):
            ln = re.sub(pat, " ", ln)
        body.append(ln)
    found = set()
    for tok in re.findall(r"\d{4}-\d{2}-\d{2}|\d[\d,]*(?:\.\d+)?%?", "\n".join(body)):
        t = tok.rstrip("%").replace(",", "")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
            found.add(t); continue
        if "." in t:
            t = t.rstrip("0").rstrip(".")
        else:
            t = str(int(t))
        found.add(t)
    undeclared = sorted(found - declared)
    check("19. Every manuscript numeric claim is declared in the traceability matrix",
          not undeclared, f"undeclared={undeclared[:10]}")
    unverified = [r["claim_id"] for r in trace if r["verification_status"] != "VERIFIED"]
    check("19b. Every traceability row is marked VERIFIED", not unverified,
          f"n={len(unverified)}")
    bad_cls = [r["claim_id"] for r in trace
               if r["evidence_class"] not in {"C1", "C2", "C3", "C4", "C5"}]
    check("19c. Every traceability row carries a valid evidence class", not bad_cls)

    # ---- 20: citation placeholders visible, none presented as verified -----
    ph = re.findall(r"\[REF-NEEDED:[^\]]*\]", md)
    check("20. Citation placeholders are present and visible", len(ph) > 0, f"n={len(ph)}")
    refs_md = text("manuscript/REFERENCES_VERIFIED.md")
    verified_block = refs_md.split("## NEEDS VERIFICATION")[0]
    check("20b. No placeholder appears in the VERIFIED block",
          "[REF-NEEDED" not in verified_block)
    def dois(t: str) -> set[str]:
        # Trailing sentence punctuation is not part of a DOI.
        return {d.rstrip(".,;:*)") for d in re.findall(r"10\.\d{4,9}/[^\s)\],;*]+", t)}
    dois_ms, dois_refs = dois(md), dois(refs_md)
    check("20c. Every manuscript DOI is listed in the reference file",
          dois_ms <= dois_refs, f"missing={sorted(dois_ms - dois_refs)}")
    check("20d. Reference file separates verified from unverified",
          "## VERIFIED" in refs_md and "## NEEDS VERIFICATION" in refs_md)

    # ---- report ------------------------------------------------------------
    print("AirSense V2 - Phase 11 independent verification")
    print("Separate parsing path. Read-only.\n")
    failed = 0
    for label, ok, detail in checks:
        pad = max(3, 78 - len(label))
        print(f"{label} {'.' * pad} {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed += 1
            if detail:
                print(f"    -> {detail}")
    print(f"\n{'=' * 72}")
    print(f"Independent verification: {len(checks) - failed}/{len(checks)} checks passed, "
          f"{failed} failed")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
