"""Phase 7: statistics over the finished grid.

    python experiments/run_analysis.py

Writes `results/analysis/phase7.json` and CSV tables, and prints a summary.
Figures are drawn separately by `experiments/make_figures.py` from these outputs.

Everything reported twice where the plan requires it:

- **level** `n60` (primary: RF/SGD per seed, XGBoost/GaussianNB collapsed) and
  `n20` (sensitivity: one observation per dataset-model cell);
- **scope** `all` cells and `placeholder_free` cells -- those where no
  adaptation by any policy, and no initial training, used a class placeholder.
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
OUT = ROOT / "results" / "analysis"
FAMILIES = ("xgb", "rf", "gnb", "sgd")


def statistics_for(rows, cells, scope):
    report = {}
    for level in ("n60", "n20"):
        blocks = {m: an.build_blocks(rows, m, level, cells=cells) for m in ("accuracy", "cost")}
        per_metric = {}
        for metric, b in blocks.items():
            fr = an.friedman(b)
            ranks = an.mean_ranks(b)
            per_metric[metric] = {
                "friedman": fr,
                "mean_ranks": {an.LABELS[p]: r for p, r in ranks.items()},
                "critical_difference": an.nemenyi_cd(len(b.policies), b.n),
                "nemenyi_p": an.nemenyi(b).round(6).to_dict(),
            }
        comparisons = an.paired_comparisons(blocks, "mechanism_selector", ["always_rebuild", "fixed_schedule"])
        report[level] = {"n": blocks["accuracy"].n, "metrics": per_metric, "selector_comparisons": comparisons}
    return report


def event_split(rows):
    """Per family: AlwaysNudge nudge effects and selector nudge success, on all
    windows and on windows whose training part contained every stream class.

    The split is decided by the *window*, never by whether a placeholder was
    injected. Filtering on injection is outcome-dependent: for the selector the
    injecting operation is usually the rebuild that follows a failed nudge, so
    dropping injected events would drop mostly failures and inflate the success
    rate. A window's class coverage is fixed before any policy acts, and it is
    exactly what triggers placeholders (in the 80% training part a nudge fits;
    the full window contains that part).
    """
    import datasets as ds

    streams = {}
    out = {}
    for family in FAMILIES:
        seeds = an.analysis_seeds(rows, family)
        tagged = {"always_nudge": [], "mechanism_selector": []}
        for dataset, model in an.cells_of(rows):
            if model != family:
                continue
            if dataset not in streams:
                streams[dataset] = ds.load_stream(dataset)[1]
            y = streams[dataset]
            classes = set(np.unique(y))
            for policy in tagged:
                for e in an.load_events(GRID, dataset, model, policy, seeds):
                    end = int(e["adapt_row"]) + 1
                    start = end - int(e["buffer_rows"])
                    train_end = start + int((end - start) * 0.8)
                    e["complete_window"] = set(np.unique(y[start:train_end])) == classes
                    tagged[policy].append(e)

        def summarise(events):
            gains = [100 * (float(e["acc_nudged"]) - float(e["acc_before"])) for e in events if e["acc_nudged"]]
            change = [100 * float(e["nudge_prediction_change"]) for e in events if e["nudge_prediction_change"]]
            return {
                "n": len(gains),
                "median_gain_pp": float(np.median(gains)) if gains else None,
                "share_accuracy_unchanged": float(np.mean([g == 0 for g in gains])) if gains else None,
                "median_prediction_change_pct": float(np.median(change)) if change else None,
                "share_no_prediction_changed": float(np.mean([c == 0 for c in change])) if change else None,
            }

        def success(events):
            attempts = [e for e in events if e["acc_nudged"] and e["degraded_to_rebuild"] == "False"]
            wins = sum(e["action"] == "NUDGE" for e in attempts)
            return {"attempts": len(attempts), "successes": wins,
                    "rate": wins / len(attempts) if attempts else None}

        complete = lambda evs: [e for e in evs if e["complete_window"]]  # noqa: E731
        nudge, selector = tagged["always_nudge"], tagged["mechanism_selector"]
        out[family] = {
            "share_of_always_nudge_windows_complete": float(np.mean([e["complete_window"] for e in nudge])) if nudge else None,
            "always_nudge_all": summarise(nudge), "always_nudge_complete_windows": summarise(complete(nudge)),
            "selector_success_all": success(selector), "selector_success_complete_windows": success(complete(selector)),
        }
    return out


def decay(rows):
    """Nudge-decay diagnostic: AlwaysNudge (accumulates) vs FixedSchedule (resets) at matched positions."""
    manifest = {m["name"]: m["n_rows"] for m in json.loads((ROOT / "results" / "dataset_manifest.json").read_text())}
    runs = an.index_runs(rows)
    per_family = {}
    for family in FAMILIES:
        trends = {"gain": [], "prediction_change": []}
        pooled = {"always_nudge": [], "fixed_schedule": []}
        detail = []
        for dataset, model in an.cells_of(rows):
            if model != family:
                continue
            for seed in an.analysis_seeds(rows, family):
                n_stream = int(runs[(dataset, model, "always_nudge", seed)]["n_stream_rows"])
                start = manifest[dataset] - n_stream
                an_pts = an.nudge_points(an.load_events(GRID, dataset, model, "always_nudge", [seed]), start, n_stream)
                fs_pts = an.nudge_points(an.load_events(GRID, dataset, model, "fixed_schedule", [seed]), start, n_stream)
                pooled["always_nudge"] += an_pts
                pooled["fixed_schedule"] += fs_pts
                row = {"dataset": dataset, "seed": seed, "n_always_nudge": len(an_pts), "n_fixed_schedule": len(fs_pts)}
                for field in trends:
                    t = an.matched_position_trend(an_pts, fs_pts, field)
                    row[f"trend_{field}"] = t
                    if t is not None:
                        trends[field].append(t)
                detail.append(row)

        summary = {"runs": detail}
        for field, values in trends.items():
            nonzero = [v for v in values if v != 0]
            summary[field] = {
                "runs_with_trend": len(values),
                "median_trend": float(np.median(values)) if values else None,
                "share_negative": float(np.mean([v < 0 for v in values])) if values else None,
                "wilcoxon_p": float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 6 else None,
            }
        for policy, pts in pooled.items():
            if len(pts) >= 10:
                rho = stats.spearmanr([p["position"] for p in pts], [p["gain"] for p in pts])
                summary[f"{policy}_gain_vs_position"] = {"n": len(pts), "rho": float(rho.statistic), "p": float(rho.pvalue)}
        per_family[family] = summary
    return per_family


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = an.load_grid(GRID)
    groups = an.verify_determinism(rows)
    print(f"{len(rows)} runs; determinism verified for XGBoost and GaussianNB across {groups} groups\n")

    exposure = an.placeholder_exposure(rows, GRID)
    all_cells = an.cells_of(rows)
    clean_cells = [c for c in all_cells if not exposure[c]["affected"]]
    print("placeholder exposure (share of adaptations, all policies):")
    for c in all_cells:
        e = exposure[c]
        print(f"  {c[0] + '/' + c[1]:<26}{100 * e['share']:5.1f}%  of {e['adaptations']:>5}"
              f"{'  +initial training' if e['initial_training_affected'] else ''}{'' if e['affected'] else '   (clean)'}")
    print(f"placeholder-free cells: {len(clean_cells)} of {len(all_cells)}\n")

    report = {
        "determinism_groups_verified": groups,
        "placeholder_exposure": {f"{d}/{m}": v for (d, m), v in exposure.items()},
        "placeholder_free_cells": [f"{d}/{m}" for d, m in clean_cells],
        "statistics": {"all": statistics_for(rows, all_cells, "all")},
    }
    if len(clean_cells) >= 3:
        report["statistics"]["placeholder_free"] = statistics_for(rows, clean_cells, "placeholder_free")

    for scope, levels in report["statistics"].items():
        for level, res in levels.items():
            print(f"=== {scope} / {level} (n={res['n']}) ===")
            for metric, m in res["metrics"].items():
                ranks = ", ".join(f"{k} {v:.2f}" for k, v in sorted(m["mean_ranks"].items(), key=lambda kv: kv[1]))
                print(f"  {metric:<9} Friedman chi2={m['friedman']['statistic']:.1f} p={m['friedman']['p']:.2g}"
                      f"   CD={m['critical_difference']:.2f}   ranks: {ranks}")
            for c in res["selector_comparisons"]:
                print(f"  selector vs {an.LABELS[c['against']]:<14} {c['metric']:<9} median diff {c['median_difference']:+12.2f}"
                      f"  better/worse {c['target_better_in']:>2}/{c['target_worse_in']:<2}"
                      f"  p={c['p']:.2g} Holm={c['p_holm']:.2g}  Cliff={c['cliffs_delta']:+.2f}  r_rb={c['rank_biserial']:+.2f}")
            print()

    pareto = [an.cell_pareto(rows, d, m) for d, m in all_cells]
    report["pareto"] = pareto
    on = sum(c["selector_on_frontier"] for c in pareto)
    print(f"=== Pareto: selector on the cost-accuracy frontier in {on} of {len(pareto)} cells ===")
    verdicts = defaultdict(list)
    for c in pareto:
        verdicts[c["selector_vs_fixed_schedule"]].append(f"{c['dataset']}/{c['model']}")
        share = c["selector_frontier_share_across_seeds"]
        print(f"  {c['dataset'] + '/' + c['model']:<26} frontier: {', '.join(an.LABELS[p] for p in c['frontier']):<60}"
              f" selector {'ON ' if c['selector_on_frontier'] else 'off'}"
              f"{f' ({100 * share:.0f}% of seeds)' if share is not None else ''}"
              f"{'  dominated by ' + ', '.join(an.LABELS[p] for p in c['selector_dominated_by']) if c['selector_dominated_by'] else ''}")
    print("\n  selector vs FixedSchedule:")
    for v, cells in sorted(verdicts.items()):
        print(f"    {len(cells):>2}  {v}")

    report["event_split"] = event_split(rows)
    report["decay"] = decay(rows)

    (OUT / "phase7.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pareto_rows = []
    for c in pareto:
        for p, v in c["policies"].items():
            pareto_rows.append({"dataset": c["dataset"], "model": c["model"], "policy": an.LABELS[p],
                                "on_frontier": p in c["frontier"], **v})
    write_csv(OUT / "pareto_points.csv", pareto_rows,
              ["dataset", "model", "policy", "cost_pct_of_rebuild", "cost_pct_sd", "accuracy", "accuracy_sd", "on_frontier"])
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
