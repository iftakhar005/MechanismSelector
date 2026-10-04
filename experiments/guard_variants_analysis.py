"""Guard-variant analysis [DIAGNOSTIC].

Compares each guard variant against the original selector, and against
AlwaysRebuild and NeverAdapt from the frozen grid. Blocks, Wilcoxon and Holm
come from `src/analysis.py` unchanged, so the statistic is the one Table IV
uses: the median over the 12 (family x seed) blocks of a stream, with XGBoost
and Gaussian NB contributing one block each because they are deterministic.

Accuracy here is prequential *balanced* accuracy for every stream, which is what
the guard prompt asked for; Table IV uses plain accuracy on the binary streams.

    python experiments/guard_variants_analysis.py --variants C A B D
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analysis as an  # noqa: E402

VARIANT_DIRS = {"A": "guard_A", "B": "guard_B", "C": "guard_C", "D": "guard_D"}
VARIANT_NAMES = {
    "A": "A keep the window", "B": "B smaller minimum (25/50)",
    "C": "C skip when unsure", "D": "D larger minimum (100/200)",
}
STREAMS = ("elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental")
# Our own thresholds, declared before the runs; not standard values.
CUT_TRAINING = 0.25      # >= 25% less median training work
CUT_STREAMS = 3          # on at least 3 of 5 streams
MAX_ACC_LOSS = 1.0       # and no more than 1 point of balanced accuracy lost anywhere


def balanced_of(row: dict) -> float:
    return 100.0 * float(row["mean_prequential_balanced_accuracy"])


an.METRICS["balanced"] = balanced_of
an.HIGHER_IS_BETTER["balanced"] = True


def counts_of(row: dict) -> dict:
    return {k: int(float(row["n_" + k])) for k in ("skip", "nudge", "rebuild", "degraded")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["C"], choices=list(VARIANT_DIRS))
    ap.add_argument("--out", default="results/analysis/guard_variants.json")
    args = ap.parse_args()

    base_rows = an.load_grid(ROOT / "results/grid")
    report = {"label": "[DIAGNOSTIC] guard variants",
              "statistic": "median over the 12 (family x seed) blocks of a stream",
              "accuracy": "mean prequential balanced accuracy",
              "thresholds": {"training_cut": CUT_TRAINING, "streams": CUT_STREAMS,
                             "max_accuracy_loss_pp": MAX_ACC_LOSS,
                             "note": "our own choices, not standard values"},
              "variants": {}}

    for v in args.variants:
        path = ROOT / "results" / VARIANT_DIRS[v] / "grid.csv"
        if not path.exists():
            print(f"variant {v}: not run yet ({path})")
            continue
        var_rows = []
        for r in an.load_grid(path.parent):
            r = dict(r)
            r["policy_key"] = f"guard_{v}"
            var_rows.append(r)
        rows = base_rows + var_rows
        key = f"guard_{v}"
        policies = ("mechanism_selector", key, "always_rebuild", "never_adapt")

        entry = {"name": VARIANT_NAMES[v], "n_runs": len(var_rows), "streams": {}}
        wins = 0
        worst_acc = 0.0
        for stream in STREAMS:
            cells = [c for c in an.cells_of(rows) if c[0] == stream]
            blk = {m: an.build_blocks(rows, m, "n60", policies, cells)
                   for m in ("cost", "balanced")}
            med = {m: {p: float(np.median(blk[m].values[:, blk[m].policies.index(p)]))
                       for p in policies} for m in ("cost", "balanced")}

            sel_cost, var_cost = med["cost"]["mechanism_selector"], med["cost"][key]
            cut = (sel_cost - var_cost) / sel_cost if sel_cost else 0.0
            acc_delta = med["balanced"][key] - med["balanced"]["mechanism_selector"]
            wins += cut >= CUT_TRAINING
            worst_acc = min(worst_acc, acc_delta)

            tally = {"skip": 0, "nudge": 0, "rebuild": 0, "degraded": 0}
            for r in var_rows:
                if r["dataset"] == stream:
                    for k2, n2 in counts_of(r).items():
                        tally[k2] += n2
            base_tally = {"skip": 0, "nudge": 0, "rebuild": 0, "degraded": 0}
            for r in base_rows:
                if r["dataset"] == stream and r["policy_key"] == "mechanism_selector":
                    for k2, n2 in counts_of(r).items():
                        base_tally[k2] += n2

            tests = an.paired_comparisons(blk, key, ["mechanism_selector"])
            entry["streams"][stream] = {
                "n_blocks": blk["cost"].n,
                "median_training": med["cost"], "median_balanced": med["balanced"],
                "training_cut_vs_selector": cut, "balanced_delta_vs_selector_pp": acc_delta,
                "decisions_variant": tally, "decisions_selector": base_tally,
                "beats_always_rebuild": {
                    "balanced": med["balanced"][key] >= med["balanced"]["always_rebuild"],
                    "training": med["cost"][key] <= med["cost"]["always_rebuild"]},
                "beats_never_adapt": {
                    "balanced": med["balanced"][key] >= med["balanced"]["never_adapt"],
                    "training": med["cost"][key] <= med["cost"]["never_adapt"]},
                "wilcoxon_vs_selector": tests,
            }

        helps = wins >= CUT_STREAMS and worst_acc >= -MAX_ACC_LOSS
        near = (not helps) and (wins >= CUT_STREAMS - 1) and worst_acc >= -(MAX_ACC_LOSS + 0.5)
        entry["verdict"] = "helps" if helps else ("near" if near else "does not help")
        entry["streams_cutting_training"] = wins
        entry["worst_balanced_delta_pp"] = worst_acc
        # Holm across every test in this variant's family (5 streams x 2 metrics),
        # not within a stream: the family is the variant, so this is the honest scope.
        all_tests = [test for s in STREAMS for test in entry["streams"][s]["wilcoxon_vs_selector"]]
        for test, adj in zip(all_tests, an.holm([test["p"] for test in all_tests])):
            test["p_holm"] = adj
        entry["holm_family"] = "5 streams x 2 metrics = 10 tests per variant"
        report["variants"][v] = entry

        print(f"\n================ variant {v}: {VARIANT_NAMES[v]}   -> {entry['verdict'].upper()}")
        print(f"{'stream':22}{'sel train':>12}{'var train':>12}{'cut':>8}"
              f"{'sel bal':>9}{'var bal':>9}{'delta':>8}{'guard':>7}{'S/N/R':>18}")
        for stream in STREAMS:
            s = entry["streams"][stream]
            d = s["decisions_variant"]
            mix = f"{d['skip']}/{d['nudge']}/{d['rebuild']}"
            print(f"{stream:22}{s['median_training']['mechanism_selector']:12,.0f}"
                  f"{s['median_training'][f'guard_{v}']:12,.0f}"
                  f"{s['training_cut_vs_selector']:8.1%}"
                  f"{s['median_balanced']['mechanism_selector']:9.2f}"
                  f"{s['median_balanced'][f'guard_{v}']:9.2f}"
                  f"{s['balanced_delta_vs_selector_pp']:+8.2f}{d['degraded']:7}"
                  f"{mix:>18}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
