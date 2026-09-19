"""Does drift-triggered adaptation pay for itself?

    python experiments/detector_premise.py

Three analyses over the finished grid, written to
`results/analysis/detector_premise.json`, and a fourth once the direction-logged
runs in `results/grid_direction/` exist:

1. **NeverAdapt against every adaptive policy.** NeverAdapt costs nothing, so it
   dominates a policy wherever it is at least as accurate (and the policy
   spent anything). Reported per cell with accuracy differences, and tested with
   paired Wilcoxon at n = 60 and n = 20, Holm-corrected, effect size the
   matched-pairs rank-biserial r.
2. **Heterogeneity.** A non-significant Friedman test pooled over streams is not
   evidence of no effect: large gains on some streams and large losses on others
   can cancel in ranks. The same comparison is broken down per stream.
3. **Does alarm quality predict whether adaptation pays?** Share of NeverAdapt
   alarms fired on a falling error rate (from `alarm_direction.py`) against the
   accuracy AlwaysRebuild gains over NeverAdapt, across the 20 cells and across
   the 5 streams. Cells share streams, so the cell-level test is not 20
   independent points; the stream-level one has only 5. Both are descriptive.
4. **Where adaptive policies spend their cost.** From the direction-logged runs:
   the share of each policy's adaptations, and of its work units, triggered by
   alarms that fired on falling error. Also verifies those runs reproduce the
   main grid's outcomes exactly.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analysis as an  # noqa: E402

warnings.filterwarnings("ignore")
GRID = ROOT / "results" / "grid"
DIRECTION = ROOT / "results" / "grid_direction"
OUT = ROOT / "results" / "analysis"
ADAPTIVE = ("always_rebuild", "fixed_schedule", "always_nudge", "mechanism_selector")
DATASETS = ("elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental")
#: "Robust" dominance: NeverAdapt ahead by at least this many points and, for the
#: stochastic models, in at least this many of five seeds. Several strict wins are
#: SGD cells near chance accuracy with margins under a point.
ROBUST_MARGIN_PP = 1.0
ROBUST_SEEDS = 4


def cell_accuracy(rows, dataset, model, policy):
    runs = an.index_runs(rows)
    seeds = an.analysis_seeds(rows, model)
    accs = [an.accuracy_of(runs[(dataset, model, policy, s)]) for s in seeds]
    costs = [an.cost_of(runs[(dataset, model, policy, s)]) for s in seeds]
    return accs, costs


def never_adapt_dominance(rows):
    out = {}
    for policy in ADAPTIVE:
        cells = []
        for dataset, model in an.cells_of(rows):
            never, _ = cell_accuracy(rows, dataset, model, "never_adapt")
            pol, costs = cell_accuracy(rows, dataset, model, policy)
            diffs = [n - p for n, p in zip(never, pol)]
            mean_diff = float(np.mean(never) - np.mean(pol))
            spent = float(np.mean(costs)) > 0
            dominates = mean_diff >= 0 and (spent or mean_diff > 0)
            cells.append({
                "cell": f"{dataset}/{model}", "never_minus_policy_pp": mean_diff,
                "seed_min_pp": min(diffs) if len(diffs) > 1 else None,
                "seed_max_pp": max(diffs) if len(diffs) > 1 else None,
                "never_better_in_seeds": sum(d > 0 for d in diffs) if len(diffs) > 1 else None,
                "n_seeds": len(diffs), "policy_spent_cost": spent, "never_adapt_dominates": dominates,
            })
        robust = [c["cell"] for c in cells if c["never_adapt_dominates"] and c["never_minus_policy_pp"] >= ROBUST_MARGIN_PP
                  and (c["n_seeds"] == 1 or (c["never_better_in_seeds"] or 0) >= ROBUST_SEEDS)]
        out[policy] = {
            "n_cells_dominated": sum(c["never_adapt_dominates"] for c in cells),
            "cells_dominated": [c["cell"] for c in cells if c["never_adapt_dominates"]],
            "n_cells_dominated_robust": len(robust),
            "cells_dominated_robust": robust,
            "cells": cells,
        }
    return out


def paired_tests(rows):
    """NeverAdapt vs each adaptive policy on accuracy, Holm across the four, per level."""
    out = {}
    for level in ("n60", "n20"):
        blocks = an.build_blocks(rows, "accuracy", level)
        never = blocks.values[:, blocks.policies.index("never_adapt")]
        results = []
        for policy in ADAPTIVE:
            pol = blocks.values[:, blocks.policies.index(policy)]
            d = never - pol
            nonzero = int(np.count_nonzero(d))
            results.append({
                "policy": an.LABELS[policy], "n": blocks.n, "median_never_minus_policy_pp": float(np.median(d)),
                "never_better_in": int(np.sum(d > 0)), "never_worse_in": int(np.sum(d < 0)),
                "p": float(stats.wilcoxon(never, pol).pvalue) if nonzero else 1.0,
                "rank_biserial": an.rank_biserial_paired(never, pol),
            })
        for r, adj in zip(results, an.holm([r["p"] for r in results])):
            r["p_holm"] = adj
        out[level] = results
    return out


def per_stream(rows):
    """Is the pooled null a real null, or cancelling effects?"""
    out = {}
    for dataset in DATASETS:
        cells = [c for c in an.cells_of(rows) if c[0] == dataset]
        blocks = an.build_blocks(rows, "accuracy", "n60", cells=cells)
        fr = an.friedman(blocks)
        never = blocks.values[:, blocks.policies.index("never_adapt")]
        best_adaptive = blocks.values[:, [blocks.policies.index(p) for p in ADAPTIVE]].max(axis=1)
        rebuild = blocks.values[:, blocks.policies.index("always_rebuild")]
        out[dataset] = {
            "n_blocks": blocks.n, "friedman_p": fr["p"],
            "median_rebuild_minus_never_pp": float(np.median(rebuild - never)),
            "median_best_adaptive_minus_never_pp": float(np.median(best_adaptive - never)),
            "mean_ranks": {an.LABELS[p]: r for p, r in an.mean_ranks(blocks).items()},
        }
    return out


def alarm_quality_vs_payoff(rows):
    direction = json.loads((GRID / "alarm_direction.json").read_text())["cells"]
    points = []
    for dataset, model in an.cells_of(rows):
        d = direction[f"{dataset}/{model}"]
        if not d["alarms"]:
            continue
        never, _ = cell_accuracy(rows, dataset, model, "never_adapt")
        rebuild, _ = cell_accuracy(rows, dataset, model, "always_rebuild")
        points.append({"dataset": dataset, "model": model, "alarms": d["alarms"],
                       "share_falling": d["share_on_improvement"],
                       "rebuild_gain_over_never_pp": float(np.mean(rebuild) - np.mean(never))})
    cell_rho = stats.spearmanr([p["share_falling"] for p in points], [p["rebuild_gain_over_never_pp"] for p in points])
    by_stream = defaultdict(lambda: {"alarms": 0, "falling": 0, "gains": []})
    for p in points:
        s = by_stream[p["dataset"]]
        s["alarms"] += p["alarms"]
        s["falling"] += p["share_falling"] * p["alarms"]
        s["gains"].append(p["rebuild_gain_over_never_pp"])
    streams = [{"dataset": k, "share_falling": v["falling"] / v["alarms"], "mean_rebuild_gain_pp": float(np.mean(v["gains"]))}
               for k, v in by_stream.items()]
    stream_rho = stats.spearmanr([s["share_falling"] for s in streams], [s["mean_rebuild_gain_pp"] for s in streams])
    return {
        "cells": points,
        "cell_level": {"n": len(points), "spearman_rho": float(cell_rho.statistic), "p": float(cell_rho.pvalue)},
        "streams": streams,
        "stream_level": {"n": len(streams), "spearman_rho": float(stream_rho.statistic), "p": float(stream_rho.pvalue)},
    }


def adaptive_cost_on_falling_error(rows):
    path = DIRECTION / "grid.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        drows = list(csv.DictReader(f))

    main = an.index_runs(rows)
    compare = [c for c in rows[0] if c in drows[0] and c not in an.VOLATILE_FIELDS | {"total_energy_kwh"}]
    mismatches = []
    for r in drows:
        m = main[(r["dataset"], r["model"], r["policy_key"], int(r["seed"]))]
        diff = [c for c in compare if m[c] != r[c]]
        if diff:
            mismatches.append({"run": f"{r['dataset']}/{r['model']}/{r['policy_key']}/{r['seed']}", "fields": diff[:5]})

    per_policy = defaultdict(lambda: {"adaptations": 0, "falling": 0, "work": 0.0, "work_falling": 0.0, "directed": 0})
    per_cell = []
    for r in drows:
        events = an.load_events(DIRECTION, r["dataset"], r["model"], r["policy_key"], [int(r["seed"])])
        directed = [e for e in events if e["alarm_direction"]]
        acc = per_policy[r["policy_key"]]
        acc["adaptations"] += len(events)
        acc["directed"] += len(directed)
        acc["falling"] += int(r["n_adaptations_on_falling_error"])
        acc["work"] += float(r["total_work_units"])
        acc["work_falling"] += float(r["work_units_on_falling_error"])
        per_cell.append({
            "dataset": r["dataset"], "model": r["model"], "policy": r["policy_key"], "seed": int(r["seed"]),
            "adaptations": len(events), "on_falling_error": int(r["n_adaptations_on_falling_error"]),
            "work_units": float(r["total_work_units"]), "work_units_on_falling_error": float(r["work_units_on_falling_error"]),
        })
    summary = {an.LABELS[p]: {
        "adaptations": v["adaptations"], "with_direction": v["directed"],
        "share_on_falling_error": v["falling"] / v["directed"] if v["directed"] else None,
        "share_of_work_units_on_falling_error": v["work_falling"] / v["work"] if v["work"] else None,
    } for p, v in per_policy.items()}
    return {"runs": len(drows), "outcome_mismatches_vs_main_grid": mismatches, "by_policy": summary, "per_run": per_cell}


def main() -> int:
    rows = an.load_grid(GRID)
    an.verify_determinism(rows)
    report = {
        "never_adapt_dominance": never_adapt_dominance(rows),
        "never_adapt_paired_tests": paired_tests(rows),
        "per_stream": per_stream(rows),
        "alarm_quality_vs_payoff": alarm_quality_vs_payoff(rows),
        "adaptive_cost_on_falling_error": adaptive_cost_on_falling_error(rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "detector_premise.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== 1. NeverAdapt (zero cost) dominates the adaptive policy in ... ===")
    for policy, v in report["never_adapt_dominance"].items():
        print(f"  {an.LABELS[policy]:<18} {v['n_cells_dominated']:>2} of 20 cells (robust: {v['n_cells_dominated_robust']})")
    print("\n  NeverAdapt accuracy minus policy accuracy (pp), per cell:")
    header = "  " + f"{'cell':<26}" + "".join(f"{an.LABELS[p][:13]:>15}" for p in ADAPTIVE)
    print(header)
    cells = [c["cell"] for c in report["never_adapt_dominance"]["always_rebuild"]["cells"]]
    for i, cell in enumerate(cells):
        line = f"  {cell:<26}"
        for p in ADAPTIVE:
            c = report["never_adapt_dominance"][p]["cells"][i]
            mark = "*" if c["never_adapt_dominates"] else " "
            line += f"{c['never_minus_policy_pp']:>+13.1f}{mark} "
        print(line)
    print("  * NeverAdapt dominates (at least as accurate, at zero cost)")

    print("\n  paired Wilcoxon, NeverAdapt vs policy, accuracy:")
    for level, res in report["never_adapt_paired_tests"].items():
        for r in res:
            print(f"    {level}  vs {r['policy']:<18} median {r['median_never_minus_policy_pp']:+6.2f} pp  "
                  f"Never better/worse {r['never_better_in']:>2}/{r['never_worse_in']:<2}  Holm p={r['p_holm']:.3g}  r={r['rank_biserial']:+.2f}")

    print("\n=== 2. per stream (n60 blocks): is the pooled null a real null? ===")
    for ds, v in report["per_stream"].items():
        print(f"  {ds:<21} blocks {v['n_blocks']:>2}  Friedman p={v['friedman_p']:.2g}  "
              f"median Rebuild-Never {v['median_rebuild_minus_never_pp']:+6.1f} pp  best adaptive-Never {v['median_best_adaptive_minus_never_pp']:+6.1f} pp")

    aq = report["alarm_quality_vs_payoff"]
    print("\n=== 3. share of alarms on falling error vs AlwaysRebuild's accuracy gain over NeverAdapt ===")
    print(f"  cells   n={aq['cell_level']['n']}  Spearman rho={aq['cell_level']['spearman_rho']:+.2f}  p={aq['cell_level']['p']:.3g}")
    print(f"  streams n={aq['stream_level']['n']}   Spearman rho={aq['stream_level']['spearman_rho']:+.2f}  p={aq['stream_level']['p']:.3g}")
    for s in sorted(aq["streams"], key=lambda s: s["share_falling"]):
        print(f"    {s['dataset']:<21} falling {100 * s['share_falling']:4.0f}%   mean rebuild gain {s['mean_rebuild_gain_pp']:+6.1f} pp")

    ad = report["adaptive_cost_on_falling_error"]
    if ad is None:
        print("\n=== 4. direction-logged runs not available yet ===")
    else:
        print(f"\n=== 4. adaptive policies: cost triggered by falling-error alarms ({ad['runs']} runs) ===")
        print(f"  outcome mismatches vs main grid: {len(ad['outcome_mismatches_vs_main_grid'])}")
        for pol, v in ad["by_policy"].items():
            print(f"  {pol:<18} adaptations {v['adaptations']:>5}  on falling error {100 * v['share_on_falling_error']:4.0f}%  "
                  f"work units on falling error {100 * v['share_of_work_units_on_falling_error']:4.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
