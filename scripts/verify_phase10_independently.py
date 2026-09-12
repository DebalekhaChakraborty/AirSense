"""Independent verification of the AirSense V2 final synthesis.

Protocol Phase 10, section 35. A separate parsing path from
``scripts/build_phase10_synthesis.py``: values are re-read from the frozen
Phase-7/8/9 sources rather than from the Phase-10 ledgers, and compared against
what the ledgers assert.

No new science. No model, no prediction, no new metric.

Usage:
    .venv-v2/bin/python scripts/verify_phase10_independently.py
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
DOCS = PROJECT_ROOT / "docs"
RESULTS = PROJECT_ROOT / "results"
CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append((label, bool(ok), detail))
    print("%s %s %s" % (label, "." * max(3, 66 - len(label) - 6),
                        "PASS" if ok else "FAIL"))
    if detail and not ok:
        print("      %s" % detail)


def sha256_of_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(str(path), encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with open(str(path), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    # Sources of truth, read independently of the Phase-10 ledgers.
    results = load_json(ARTIFACTS / "phase8_primary_results_lock.json")
    locked_table = {r["model"]: r for r in
                    read_csv(ARTIFACTS / "phase10_table_locked_test.csv")}
    conclusion = load_json(ARTIFACTS / "phase10_canonical_conclusion.json")
    hypotheses = load_json(ARTIFACTS
                           / "phase10_hypothesis_ledger.json")["hypotheses"]
    matrix = read_csv(ARTIFACTS / "phase10_claim_evidence_matrix.csv")
    dev = {r["model"]: r for r in read_csv(
        RESULTS / "validation" / "phase6_model_comparison.csv")}
    seeds = read_csv(RESULTS / "models" / "robustness"
                     / "multiseed_metrics.csv")
    h4 = {r["scope"]: r for r in read_csv(
        RESULTS / "validation" / "phase6_h4_summary.csv")}
    phase9 = load_json(ARTIFACTS / "phase9_post_test_analysis_summary.json")

    # 1-5: the locked values, straight from the Phase-8 lock
    for label, key, column in (
            ("1  Primary locked value", "primary_value",
             "macro_station_horizon_MAE"),
            ("3  B0 macro value", "reference_primary_value",
             "macro_station_horizon_MAE")):
        model = "B3_R2" if "Primary" in label else "B0"
        check("%s matches the locked-test table" % label,
              abs(float(locked_table[model][column])
                  - results[key]) < 1e-12,
              "%s vs %s" % (locked_table[model][column], results[key]))
    check("2  Secondary locked value matches the locked-test table",
          abs(float(locked_table["GRU_R1"]["severe_MAE_gt_244"])
              - results["secondary_value"]) < 1e-12)
    check("4  B0 severe value matches the locked-test table",
          abs(float(locked_table["B0"]["severe_MAE_gt_244"])
              - results["reference_severe_value"]) < 1e-12)
    check("5  Severe n matches the locked-test table",
          int(locked_table["B3_R2"]["severe_n"])
          == results["secondary_severe_n"] == 20336)

    # 6: relative improvement, recomputed here
    expected = 100.0 * (results["reference_primary_value"]
                        - results["primary_value"]) \
        / results["reference_primary_value"]
    check("6  B3_R2 relative improvement over B0 recomputes",
          abs(conclusion["b3r2_relative_improvement_vs_B0_pct"]
              - expected) < 1e-12,
          "%.12f vs %.12f" % (conclusion[
              "b3r2_relative_improvement_vs_B0_pct"], expected))
    check("   Improvement rounds to the expected 11.487%",
          abs(round(expected, 3) - 11.487) < 1e-9, "%.6f" % expected)

    # 7-8: H1 and H2 evidence sources
    h1 = hypotheses["H1"]["evidence"]
    check("7  H1 evidence matches the frozen development comparison",
          abs(h1["B3_R0_macro_MAE"]
              - float(dev["B3_R0"]["macro_station_horizon_MAE"])) < 1e-9
          and abs(h1["B3_R1_macro_MAE"]
                  - float(dev["B3_R1"]["macro_station_horizon_MAE"])) < 1e-9)
    check("8  H2 is development-supported and not retested",
          hypotheses["H2"]["evidence_class"] == "C2"
          and hypotheses["H2"]["locked_test_retested"] is False)

    # 9: H3 mixed evidence - only B3 improved
    h3 = hypotheses["H3"]["evidence"]
    recomputed = {
        "B3": float(dev["B3_R1"]["macro_station_horizon_MAE"])
        - float(dev["B3_R2"]["macro_station_horizon_MAE"]),
        "GRU": float(dev["GRU_R1"]["macro_station_horizon_MAE"])
        - float(dev["GRU_R2"]["macro_station_horizon_MAE"]),
        "TCN": float(dev["TCN_R1"]["macro_station_horizon_MAE"])
        - float(dev["TCN_R2"]["macro_station_horizon_MAE"]),
    }
    check("9  H3 evidence shows gain only for the boosted family",
          recomputed["B3"] > 0 and recomputed["GRU"] < 0
          and recomputed["TCN"] < 0
          and abs(h3["B3_R1_to_B3_R2"] - recomputed["B3"]) < 1e-6
          and "WEAK" in hypotheses["H3"]["status"].upper(),
          "B3 %+.4f GRU %+.4f TCN %+.4f" % (recomputed["B3"],
                                            recomputed["GRU"],
                                            recomputed["TCN"]))

    # 10: H4 paired-seed evidence
    sa_r2 = {r["seed"]: float(r["macro_station_horizon_MAE"])
             for r in seeds if r["model"] == "SA_R2"}
    sa_r3 = {r["seed"]: float(r["macro_station_horizon_MAE"])
             for r in seeds if r["model"] == "SA_R3"}
    better = sum(1 for s in sa_r3 if s in sa_r2 and sa_r3[s] < sa_r2[s])
    check("10 H4 rests on SA_R3 better than SA_R2 in 5/5 paired seeds",
          better == 5 and len(sa_r3) == 5
          and abs(hypotheses["H4"]["evidence"]["relative_improvement_pct"]
                  - float(h4["overall"][
                      "relative_MAE_improvement_pct"])) < 1e-9,
          "%d/5 seeds better" % better)
    check("   H4 records the unreliable h=24 direction",
          hypotheses["H4"]["evidence"][
              "h24_direction_reliable_across_seeds"] is False)

    # 11: H5 locked-test limitation
    check("11 H5 records persistence as the stronger severe benchmark",
          hypotheses["H5"]["evidence"]["locked_B0_severe_MAE"]
          < hypotheses["H5"]["evidence"]["locked_GRU_R1_severe_MAE"]
          and "NOT SUPPORTED AS SUPERIORITY OVER PERSISTENCE"
          in hypotheses["H5"]["status"].upper())

    # 12-14: Phase-9 findings carried faithfully
    posttest = {(r["subject"], r["diagnostic"]): r["value"] for r in
                read_csv(ARTIFACTS
                         / "phase10_table_posttest_diagnostics.csv")}
    acf = {(r["model"], r["lag_hours"]): r["mean"] for r in read_csv(
        ARTIFACTS / "phase9_residual_acf_summary.csv")
        if r["horizon_hours"] == "24"}
    check("12 Phase-9 lag-1 residual finding is carried exactly",
          all(posttest[(m, "residual_acf_lag1_h24")] == acf[(m, "1")]
              for m in ("B3_R2", "GRU_R1", "B0")))
    severe9 = {(r["model"], r["horizon_hours"]): r["MAE"] for r in read_csv(
        ARTIFACTS / "phase9_severe_tail_analysis.csv")}
    check("13 Phase-9 severe h=24 finding is carried exactly",
          all(posttest[(m, "severe_MAE_h24")] == severe9[(m, "24")]
              for m in ("B3_R2", "GRU_R1", "B0")))
    events = read_csv(ARTIFACTS / "phase9_severe_events.csv")
    check("14 Phase-9 severe-event count is carried exactly",
          int(posttest[("all_models", "severe_event_count")]) == len(events)
          == 586)

    # 15: V1 comparison prohibition
    v1 = load_json(ARTIFACTS / "phase10_v1_v2_structural_ledger.json")
    check("15 V1/V2 ledger forbids direct metric comparison",
          v1["direct_metric_comparison_valid"] is False
          and v1["percentage_improvements_computed"] == 0
          and v1["comparison_type"] == "QUALITATIVE / STRUCTURAL ONLY")

    # 16: LOSO wording
    loso = next(r for r in matrix if r["claim_id"] == "loso")
    check("16 LOSO wording keeps the target-transfer caveat",
          "held-out target-station supervision transfer"
          in loso["permitted_wording"].lower()
          and "unseen" in loso["prohibited_wording"].lower())

    # 17: foundation models
    governance = load_json(ARTIFACTS
                           / "phase10_foundation_model_governance.json")
    check("17 Foundation-model execution count is zero",
          governance["foundation_models_executed"] == 0
          and all(m["executed"] is False
                  for m in governance["models"].values()))

    # 18: every claim traces to an artifact
    untraceable = [r["claim_id"] for r in matrix
                   if not any((PROJECT_ROOT / p.strip()).exists()
                              for p in r["artifact_source"].split(";"))]
    check("18 Every claim maps to an artifact that exists", not untraceable,
          "untraceable: %s" % untraceable)
    index = load_json(ARTIFACTS / "phase10_master_evidence_index.json")
    bad = [k for k, v in index["evidence"].items()
           if sha256_of_file(PROJECT_ROOT / v["artifact"]) != v["sha256"]]
    check("   Master evidence index hashes reconcile", not bad,
          "unreconciled: %s" % bad)
    check("   Phase-9 summary is cited as exploratory, not confirmatory",
          phase9["classification"] == "POST_TEST_EXPLORATORY"
          and all(r["evidence_class"] != "C1" for r in matrix
                  if "phase9" in r["artifact_source"].lower()))

    failed = [row for row in CHECKS if not row[1]]
    print("")
    if failed:
        print("PHASE 10 INDEPENDENT VERIFICATION FAILED: %d check(s)"
              % len(failed))
        return 1
    print("PHASE 10 INDEPENDENT VERIFICATION PASSED: %d checks agree"
          % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
