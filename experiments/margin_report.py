"""Early warning: is the selector actually beating FixedSchedule?

Reads `results/grid/grid.csv` -- complete or partial -- and, for every
(dataset, model) cell with both policies present, compares MechanismSelector
against FixedSchedule(k=3) on cost and accuracy, seed by seed.

    python experiments/margin_report.py

**Cost** is total adaptation work units as a percentage of AlwaysRebuild's for
the same (dataset, model, seed). **Accuracy** is mean prequential accuracy in the
scorer each stream uses: plain accuracy for binary streams, balanced accuracy
for multi-class ones.

Each cell is labelled:

- `SELECTOR DOMINATES` -- cheaper and at least as accurate
- `FIXEDSCHED DOMINATES` -- cheaper and at least as accurate
- `TRADE-OFF` -- one is cheaper, the other more accurate
- `THIN` is appended when the cost gap is under 5 points of AlwaysRebuild and the
  accuracy gap under 1 point -- the margin that would not survive a reviewer.

It also reports model growth for the selector against AlwaysNudge, the
inference-cost side of the comparison.

XGBoost and GaussianNB produce identical results for every seed (neither model
uses randomness at these settings), so for those cells n seeds is one result
repeated, not independent replicates.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results" / "grid" / "grid.csv"
THIN_COST_PP, THIN_ACC_PP = 5.0, 1.0
DETERMINISTIC_MODELS = {"xgb", "gnb"}


def accuracy_of(row: dict) -> float:
    key = "mean_prequential_accuracy" if row["scorer"] == "accuracy" else "mean_prequential_balanced_accuracy"
    return float(row[key])


def main(path: Path = GRID) -> int:
    if not path.exists():
        print(f"no grid results yet at {path}")
        return 1
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_run = {(r["dataset"], r["model"], r["policy_key"], int(r["seed"])): r for r in rows}
    cells = sorted({(r["dataset"], r["model"]) for r in rows})

    print(f"{len(rows)} runs in {path.name}\n")
    header = (f"{'dataset':<20}{'model':<6}{'seeds':>6}{'Sel %':>8}{'FS %':>8}{'cost gap':>10}"
              f"{'Sel acc':>9}{'FS acc':>9}{'acc gap':>9}{'Sel grow':>10}{'Nudge grow':>11}  verdict")
    print(header)
    print("-" * len(header))

    cost_gaps, verdicts = [], defaultdict(int)
    for dataset, model in cells:
        sel_pct, fs_pct, sel_acc, fs_acc, sel_grow, nudge_grow = [], [], [], [], [], []
        for seed in sorted({s for (d, m, _, s) in by_run if (d, m) == (dataset, model)}):
            sel = by_run.get((dataset, model, "mechanism_selector", seed))
            fs = by_run.get((dataset, model, "fixed_schedule", seed))
            reb = by_run.get((dataset, model, "always_rebuild", seed))
            if not (sel and fs and reb) or float(reb["total_work_units"]) == 0:
                continue
            base = float(reb["total_work_units"])
            sel_pct.append(100 * float(sel["total_work_units"]) / base)
            fs_pct.append(100 * float(fs["total_work_units"]) / base)
            sel_acc.append(100 * accuracy_of(sel))
            fs_acc.append(100 * accuracy_of(fs))
            sel_grow.append(float(sel["inference_growth"]))
            nudge = by_run.get((dataset, model, "always_nudge", seed))
            if nudge:
                nudge_grow.append(float(nudge["inference_growth"]))

        if not sel_pct:
            continue
        cost_gap = mean(fs_pct) - mean(sel_pct)        # positive: selector cheaper
        acc_gap = mean(sel_acc) - mean(fs_acc)         # positive: selector more accurate
        cost_gaps.append(cost_gap)

        if cost_gap >= 0 and acc_gap >= 0:
            verdict = "SELECTOR DOMINATES"
        elif cost_gap <= 0 and acc_gap <= 0:
            verdict = "FIXEDSCHED DOMINATES"
        else:
            verdict = "TRADE-OFF"
        if abs(cost_gap) < THIN_COST_PP and abs(acc_gap) < THIN_ACC_PP:
            verdict += " (THIN)"
        verdicts[verdict] += 1

        seeds_label = f"{len(sel_pct)}" + ("*" if model in DETERMINISTIC_MODELS and len(sel_pct) > 1 else "")
        print(f"{dataset:<20}{model:<6}{seeds_label:>6}{mean(sel_pct):>7.1f}%{mean(fs_pct):>7.1f}%"
              f"{cost_gap:>+9.1f}p{mean(sel_acc):>8.2f}%{mean(fs_acc):>8.2f}%{acc_gap:>+8.2f}p"
              f"{mean(sel_grow):>9.2f}x{(mean(nudge_grow) if nudge_grow else float('nan')):>10.2f}x  {verdict}")

    if not cost_gaps:
        print("no cell yet has selector, FixedSchedule and AlwaysRebuild together")
        return 1

    print(f"\ncost gap = FixedSchedule % minus Selector % of AlwaysRebuild (positive: selector cheaper)")
    print(f"acc gap  = Selector minus FixedSchedule accuracy, percentage points")
    print("* deterministic model: every seed is the same result, not an independent replicate\n")
    print(f"cells compared: {len(cost_gaps)}   selector cheaper in {sum(g > 0 for g in cost_gaps)}, "
          f"FixedSchedule cheaper in {sum(g < 0 for g in cost_gaps)}   median cost gap {median(cost_gaps):+.1f}p")
    for verdict, count in sorted(verdicts.items()):
        print(f"  {verdict:<32}{count}")
    thin = sum(n for v, n in verdicts.items() if "THIN" in v)
    if thin * 2 >= len(cost_gaps):
        print(f"\nWARNING: {thin} of {len(cost_gaps)} cells are within the thin margin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else GRID))
