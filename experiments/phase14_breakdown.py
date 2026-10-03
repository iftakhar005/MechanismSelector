"""Phase 14 breakdown — per family, per dataset, from committed results only.

DIAGNOSTIC. Runs no experiment: it reads `results/grid/`, `results/grid_wait200/`
and `results/grid_wait500/` and re-cuts the numbers that the verdict was taken
on. The verdict itself is unchanged and is not recomputed here.

Produces:
1. per (dataset, family): training ratio and accuracy delta against the selector,
   so it is visible which families drive the 3.2x training on insects_abrupt and
   the +9.40 pp accuracy;
2. per dataset: Wilcoxon signed-rank against the selector on training ops and on
   balanced accuracy, Holm-corrected across the five datasets within each metric,
   with the effective n stated;
3. alarm counts per dataset, both W.

Output: results/analysis/phase14_breakdown.json
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC = ("xgb", "gnb")
FAMILIES = ("xgb", "rf", "sgd", "gnb")


def _inference_module():
    spec = importlib.util.spec_from_file_location(
        "phase9_inference_cost", ROOT / "experiments" / "phase9_inference_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INF = _inference_module()


def load(grid_dir: Path, policy: str):
    out = {}
    for r in csv.DictReader((grid_dir / "grid.csv").open()):
        if r["policy_key"] != policy:
            continue
        if r["model"] in DETERMINISTIC and int(r["seed"]) != 0:
            continue
        traj = INF.size_trajectory(grid_dir / "events", r["dataset"], r["model"],
                                   r["policy_key"], int(r["seed"]))
        infer, _ = INF.inference_ops(r, traj)
        out[(r["dataset"], r["model"], int(r["seed"]))] = {
            "training": float(r["total_work_units"]),
            "inference": infer,
            "acc_balanced": float(r["mean_prequential_balanced_accuracy"]) * 100,
            "acc_plain": float(r["mean_prequential_accuracy"]) * 100,
            "n_alarms": int(float(r["n_alarms"])),
            "n_adaptations": int(float(r["n_adaptations"])),
            "n_degraded": int(float(r["n_degraded"])),
            "cancelled": int(float(r.get("n_waits_cancelled") or 0)),
            "proceeded": int(float(r.get("n_waits_proceeded") or 0)),
            "ignored": int(float(r.get("n_alarms_ignored_during_wait") or 0)),
            "truncated": int(float(r.get("n_waits_truncated") or 0)),
        }
    return out


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, running)
    return out


def main() -> None:
    sel = load(ROOT / "results/grid", "mechanism_selector")
    waits = {w: load(ROOT / f"results/grid_wait{w}", "mechanism_selector") for w in (200, 500)}
    datasets = sorted({k[0] for k in sel})

    # ---- 1. per family
    per_family = {}
    for w, wait in waits.items():
        rows = {}
        for dataset in datasets:
            for fam in FAMILIES:
                keys = [k for k in sel if k[0] == dataset and k[1] == fam and k in wait]
                if not keys:
                    continue
                s_t = np.array([sel[k]["training"] for k in keys])
                x_t = np.array([wait[k]["training"] for k in keys])
                s_a = np.array([sel[k]["acc_balanced"] for k in keys])
                x_a = np.array([wait[k]["acc_balanced"] for k in keys])
                rows[f"{dataset}/{fam}"] = {
                    "n_seeds": len(keys),
                    "selector_training": float(s_t.mean()), "wait_training": float(x_t.mean()),
                    "training_ratio": float(x_t.mean() / s_t.mean()) if s_t.mean() else None,
                    "training_delta": float(x_t.mean() - s_t.mean()),
                    "selector_acc_balanced": float(s_a.mean()), "wait_acc_balanced": float(x_a.mean()),
                    "accuracy_delta_pp": float(x_a.mean() - s_a.mean()),
                }
        per_family[w] = rows

    # ---- 2. per-dataset Wilcoxon, Holm across the five datasets within each metric
    tests = {}
    for w, wait in waits.items():
        per_metric = {}
        for metric, better_is_low in (("training", True), ("acc_balanced", False)):
            rows, ps = [], []
            for dataset in datasets:
                keys = [k for k in sel if k[0] == dataset and k in wait]
                a = np.array([sel[k][metric] for k in keys])
                b = np.array([wait[k][metric] for k in keys])
                diff = b - a
                nz = diff[diff != 0]
                p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 6 else None
                rows.append({"dataset": dataset, "n_blocks": len(keys), "n_nonzero": int(len(nz)),
                             "median_difference": float(np.median(diff)),
                             "selector_mean": float(a.mean()), "wait_mean": float(b.mean()),
                             "p": p})
                if p is not None:
                    ps.append(p)
            adj = iter(holm(ps))
            for r in rows:
                r["p_holm"] = next(adj) if r["p"] is not None else None
                r["direction"] = ("wait lower" if r["median_difference"] < 0 else "wait higher")
            per_metric[metric] = rows
        tests[w] = per_metric

    # ---- 3. alarm counts
    alarms = {}
    for w, wait in waits.items():
        per_dataset = {}
        for dataset in datasets:
            keys = [k for k in wait if k[0] == dataset]
            per_dataset[dataset] = {
                "alarms_total": sum(wait[k]["n_alarms"] for k in keys),
                "cancelled": sum(wait[k]["cancelled"] for k in keys),
                "proceeded": sum(wait[k]["proceeded"] for k in keys),
                "ignored_during_wait": sum(wait[k]["ignored"] for k in keys),
                "truncated": sum(wait[k]["truncated"] for k in keys),
                "guard_rebuilds_wait": sum(wait[k]["n_degraded"] for k in keys),
                "guard_rebuilds_selector": sum(sel[k]["n_degraded"] for k in keys if k in sel),
            }
        alarms[w] = per_dataset

    payload = {"label": "diagnostic (re-cut of committed results; the verdict is unchanged)",
               "effective_n_note": "blocks are (family, seed) within a dataset; xgb and gnb are "
                                   "deterministic and contribute seed 0 only, so n = 12 per dataset "
                                   "and 60 overall",
               "per_family": per_family, "per_dataset_tests": tests, "alarms": alarms}
    (ROOT / "results/analysis/phase14_breakdown.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    for w in (200, 500):
        print(f"\n===== W = {w}: per family (training ratio, balanced accuracy delta)")
        print(f"{'cell':26}{'sel train':>13}{'wait train':>13}{'ratio':>8}{'sel acc':>9}{'wait acc':>10}{'delta':>8}")
        for cell, v in per_family[w].items():
            print(f"{cell:26}{v['selector_training']:13,.0f}{v['wait_training']:13,.0f}"
                  f"{v['training_ratio']:8.2f}{v['selector_acc_balanced']:9.1f}"
                  f"{v['wait_acc_balanced']:10.1f}{v['accuracy_delta_pp']:+8.2f}")

    for w in (200, 500):
        print(f"\n===== W = {w}: per-dataset Wilcoxon (n = 12 blocks each), Holm across 5 datasets")
        for metric, rows in tests[w].items():
            print(f"  {metric}")
            for r in rows:
                p = "n/a" if r["p"] is None else f"{r['p']:.4g}"
                ph = "n/a" if r["p_holm"] is None else f"{r['p_holm']:.4g}"
                print(f"    {r['dataset']:22} n={r['n_blocks']:3} nonzero={r['n_nonzero']:3} "
                      f"median diff {r['median_difference']:+13,.2f}  p={p:>10}  Holm={ph:>10}")

    print("\n===== alarm counts")
    for w in (200, 500):
        for dataset, a in alarms[w].items():
            print(f"  W={w} {dataset:22} total {a['alarms_total']:5} | cancelled {a['cancelled']:5}"
                  f" | proceeded {a['proceeded']:5} | ignored {a['ignored_during_wait']:5}"
                  f" | truncated {a['truncated']:3}")
    print(f"\nwrote {ROOT / 'results/analysis/phase14_breakdown.json'}")


if __name__ == "__main__":
    main()
