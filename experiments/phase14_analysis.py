"""Phase 14 analysis — WaitAndCheck(W) against the selector and the baselines.

Reads the frozen grid (`results/grid/`) for NeverAdapt, AlwaysRebuild,
FixedSchedule and MechanismSelector, and the new `results/grid_wait200/` and
`results/grid_wait500/` for WaitAndCheck. Nothing in `results/grid/` is written.

Accuracy conventions, stated per column:
- `acc_balanced` - whole-stream prequential, balanced on multi-class streams,
  which is the metric the floor uses.
- `acc_plain` - whole-stream prequential, plain 0/1.

Inference ops use the Phase 9.5 method: units per row interpolated linearly in
model size between the two measured endpoints and integrated over the logged
size trajectory. Exact at the endpoints, an estimate between them.

Output: results/analysis/phase14_waitandcheck.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC = ("xgb", "gnb")
BASELINES = ("never_adapt", "always_rebuild", "fixed_schedule", "mechanism_selector")
# Souza et al. 2020, Table 2 (arXiv:2005.00113, p. 37) - INSECTS Abrupt (balanced)
CHANGE_POINTS = (14352, 19500, 33240, 38682, 39510)


def _inference_module():
    spec = importlib.util.spec_from_file_location(
        "phase9_inference_cost", ROOT / "experiments" / "phase9_inference_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INF = _inference_module()


def load_runs(grid_dir: Path, policies=None):
    runs = []
    for r in csv.DictReader((grid_dir / "grid.csv").open()):
        if policies and r["policy_key"] not in policies:
            continue
        if r["model"] in DETERMINISTIC and int(r["seed"]) != 0:
            continue
        traj = INF.size_trajectory(grid_dir / "events", r["dataset"], r["model"],
                                   r["policy_key"], int(r["seed"]))
        infer, _ = INF.inference_ops(r, traj)
        runs.append({
            "dataset": r["dataset"], "model": r["model"], "seed": int(r["seed"]),
            "policy": r["policy_key"],
            "training": float(r["total_work_units"]), "inference": infer,
            "combined": float(r["total_work_units"]) + infer,
            "acc_balanced": float(r["mean_prequential_balanced_accuracy"]) * 100,
            "acc_plain": float(r["mean_prequential_accuracy"]) * 100,
            "n_alarms": int(float(r["n_alarms"])), "n_adaptations": int(float(r["n_adaptations"])),
            "n_degraded": int(float(r["n_degraded"])),
            "n_cancelled": int(float(r.get("n_waits_cancelled") or 0)),
            "n_proceeded": int(float(r.get("n_waits_proceeded") or 0)),
            "n_ignored": int(float(r.get("n_alarms_ignored_during_wait") or 0)),
            "n_truncated": int(float(r.get("n_waits_truncated") or 0)),
        })
    return runs


def by_cell(runs):
    out = defaultdict(list)
    for r in runs:
        out[(r["dataset"], r["model"], r["policy"])].append(r)
    return out


def paired(a_runs, b_runs, field):
    """Pair runs on (dataset, model, seed); returns the two aligned vectors."""
    a = {(r["dataset"], r["model"], r["seed"]): r[field] for r in a_runs}
    b = {(r["dataset"], r["model"], r["seed"]): r[field] for r in b_runs}
    keys = sorted(set(a) & set(b))
    return [a[k] for k in keys], [b[k] for k in keys], keys


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, running)
    return out


def cancelled_near_change_points(grid_dir: Path, tol=1000):
    near = total = 0
    for f in sorted((grid_dir / "events").glob("insects_abrupt__*__mechanism_selector__*.csv")):
        _, model, _, seed = f.stem.split("__")
        if model in DETERMINISTIC and int(seed) != 0:
            continue
        for e in csv.DictReader(f.open()):
            if e["action"] != "CANCELLED":
                continue
            total += 1
            row = int(float(e["alarm_row"]))
            near += any(cp <= row <= cp + tol for cp in CHANGE_POINTS)
    return {"cancelled": total, "within_1000_rows_of_a_change_point": near,
            "share": (near / total) if total else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/analysis/phase14_waitandcheck.json")
    args = ap.parse_args()

    base = load_runs(ROOT / "results/grid", BASELINES)
    waits = {w: load_runs(ROOT / f"results/grid_wait{w}") for w in (200, 500)}
    sel = [r for r in base if r["policy"] == "mechanism_selector"]

    cells = by_cell(base)
    for w, runs in waits.items():
        for r in runs:
            r["policy"] = f"waitandcheck_{w}"
        cells.update(by_cell(runs))

    per_cell = {}
    for (dataset, model, policy), runs in sorted(cells.items()):
        per_cell.setdefault(f"{dataset}/{model}", {})[policy] = {
            "n_runs": len(runs),
            "training": st.mean(r["training"] for r in runs),
            "inference": st.mean(r["inference"] for r in runs),
            "combined": st.mean(r["combined"] for r in runs),
            "acc_balanced": st.mean(r["acc_balanced"] for r in runs),
            "acc_plain": st.mean(r["acc_plain"] for r in runs),
            "n_alarms": st.mean(r["n_alarms"] for r in runs),
            "n_adaptations": st.mean(r["n_adaptations"] for r in runs),
            "n_degraded": st.mean(r["n_degraded"] for r in runs),
        }

    # --- verdict rules, W = 200, per dataset, median over family x seed cells
    verdict = {}
    for w in (200, 500):
        per_dataset = {}
        for dataset in sorted({r["dataset"] for r in sel}):
            s = [r for r in sel if r["dataset"] == dataset]
            x = [r for r in waits[w] if r["dataset"] == dataset]
            s_t, x_t, _ = paired(s, x, "training")
            s_a, x_a, _ = paired(s, x, "acc_balanced")
            ratios = [(xi / si) if si else float("inf") for si, xi in zip(s_t, x_t)]
            per_dataset[dataset] = {
                "n_cells": len(s_t),
                "training_per_adaptation_selector": sum(s_t) / max(1, sum(r["n_adaptations"] for r in s)),
                "training_per_proceed_wait": sum(x_t) / max(1, sum(r["n_adaptations"] for r in x)),
                "adaptations_selector": sum(r["n_adaptations"] for r in s),
                "adaptations_wait": sum(r["n_adaptations"] for r in x),
                "neveradapt_acc_balanced": float(np.mean([r["acc_balanced"] for r in base
                                                          if r["policy"] == "never_adapt" and r["dataset"] == dataset])),
                "neveradapt_training": float(np.mean([r["training"] for r in base
                                                      if r["policy"] == "never_adapt" and r["dataset"] == dataset])),
                "wait_training_mean": float(np.mean([r["training"] for r in x])),
                "median_training_ratio": float(np.median(ratios)),
                "rule1_training_le_0.75": bool(np.median(ratios) <= 0.75),
                "selector_acc_balanced": float(np.mean(s_a)),
                "wait_acc_balanced": float(np.mean(x_a)),
                "accuracy_delta_pp": float(np.mean(x_a) - np.mean(s_a)),
                "rule2_acc_within_1pp": bool(np.mean(x_a) - np.mean(s_a) >= -1.0),
            }
        verdict[w] = per_dataset

    # --- Wilcoxon + Holm against the selector
    tests = {}
    for w in (200, 500):
        res = []
        for field in ("training", "combined", "acc_balanced"):
            a, b, keys = paired(sel, waits[w], field)
            diff = np.array(b) - np.array(a)
            nz = diff[diff != 0]
            p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 6 else None
            res.append({"metric": field, "n_blocks": len(keys), "n_nonzero": int(len(nz)),
                        "median_difference": float(np.median(diff)), "p": p})
        ps = [r["p"] for r in res if r["p"] is not None]
        adj = holm(ps)
        it = iter(adj)
        for r in res:
            r["p_holm"] = next(it) if r["p"] is not None else None
        tests[w] = res

    # --- alarm bookkeeping per dataset
    alarms = {}
    for w in (200, 500):
        per_dataset = {}
        for dataset in sorted({r["dataset"] for r in waits[w]}):
            rs = [r for r in waits[w] if r["dataset"] == dataset]
            ss = [r for r in sel if r["dataset"] == dataset]
            per_dataset[dataset] = {
                "alarms_total": sum(r["n_alarms"] for r in rs),
                "cancelled": sum(r["n_cancelled"] for r in rs),
                "proceeded": sum(r["n_proceeded"] for r in rs),
                "ignored_during_wait": sum(r["n_ignored"] for r in rs),
                "truncated": sum(r["n_truncated"] for r in rs),
                "guard_rebuilds_wait": sum(r["n_degraded"] for r in rs),
                "guard_rebuilds_selector": sum(r["n_degraded"] for r in ss),
                "adaptations_wait": sum(r["n_adaptations"] for r in rs),
                "adaptations_selector": sum(r["n_adaptations"] for r in ss),
            }
        alarms[w] = per_dataset

    # --- capture fraction against the Phase 10 oracle, training ops, lambda = 1e3
    capture = {}
    oracle = json.loads((ROOT / "results/analysis/phase10_per_stream.json").read_text(encoding="utf-8"))
    for dataset in ("elec2", "covtype"):
        orc = oracle["per_stream"][dataset]["by_lambda"]["1000.0"]["oracle_O1"]["train"]
        s = sum(r["training"] for r in sel if r["dataset"] == dataset
                and (r["model"] not in DETERMINISTIC or r["seed"] == 0))
        for w in (200, 500):
            x = sum(r["training"] for r in waits[w] if r["dataset"] == dataset)
            capture[f"{dataset}_W{w}"] = {
                "selector_training": s, "wait_training": x, "oracle_training": orc,
                "capture_fraction": (s - x) / (s - orc) if (s - orc) else None}

    payload = {
        "label": "declared follow-up (Phase 14, WaitAndCheck)",
        "accuracy_definitions": {
            "acc_balanced": "whole-stream prequential, balanced on multi-class (the floor's metric)",
            "acc_plain": "whole-stream prequential, plain 0/1",
        },
        "thresholds_are_our_own": True,
        "per_cell": per_cell, "verdict_inputs": verdict, "wilcoxon_holm": tests,
        "alarms": alarms, "capture_fraction": capture,
        "missed_real_drift_W200": cancelled_near_change_points(ROOT / "results/grid_wait200"),
        "missed_real_drift_W500": cancelled_near_change_points(ROOT / "results/grid_wait500"),
    }
    (ROOT / args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("per cell (families separate), mean over seeds: training / inference / combined / acc_balanced")
    for cell, pols in payload["per_cell"].items():
        print(f"  {cell}")
        for pol in ("never_adapt", "always_rebuild", "fixed_schedule", "mechanism_selector",
                    "waitandcheck_200", "waitandcheck_500"):
            v = pols.get(pol)
            if v:
                print(f"    {pol:20}{v['training']:14,.0f}{v['inference']:16,.0f}"
                      f"{v['combined']:16,.0f}{v['acc_balanced']:8.1f}")

    print("")
    print("Wilcoxon vs the selector (paired on dataset x family x seed), Holm-corrected")
    for w in (200, 500):
        for r in tests[w]:
            print(f"  W={w} {r['metric']:14} n={r['n_blocks']:3} nonzero={r['n_nonzero']:3} "
                  f"median diff {r['median_difference']:+14,.1f} p={r['p']} Holm={r['p_holm']}")

    print("")
    print("guard-forced rebuilds and training per adaptation")
    for w in (200, 500):
        for d, a in alarms[w].items():
            v = verdict[w][d]
            print(f"  W={w} {d:22} guard {a['guard_rebuilds_selector']:5.0f} -> {a['guard_rebuilds_wait']:5.0f}"
                  f" | adaptations {a['adaptations_selector']:5.0f} -> {a['adaptations_wait']:5.0f}"
                  f" | train/adapt {v['training_per_adaptation_selector']:12,.0f} ->"
                  f" {v['training_per_proceed_wait']:12,.0f}")

    print("")
    print("vs NeverAdapt (balanced accuracy, training ops)")
    for w in (200, 500):
        for d, v in verdict[w].items():
            beats_acc = v["wait_acc_balanced"] >= v["neveradapt_acc_balanced"]
            cheaper = v["wait_training_mean"] <= v["neveradapt_training"]
            print(f"  W={w} {d:22} acc {v['wait_acc_balanced']:5.1f} vs {v['neveradapt_acc_balanced']:5.1f}"
                  f" {'BEATS' if beats_acc else 'loses':>6} | train {v['wait_training_mean']:12,.0f}"
                  f" vs {v['neveradapt_training']:12,.0f} {'cheaper' if cheaper else 'DEARER':>7}")
    for w in (200, 500):
        print(f"\n===== W = {w}: per dataset (median training ratio, accuracy in balanced pp)")
        for dataset, v in verdict[w].items():
            print(f"  {dataset:22} ratio {v['median_training_ratio']:.3f} "
                  f"{'PASS' if v['rule1_training_le_0.75'] else 'fail':>5} | "
                  f"acc {v['selector_acc_balanced']:5.1f} -> {v['wait_acc_balanced']:5.1f} "
                  f"({v['accuracy_delta_pp']:+.2f} pp) "
                  f"{'PASS' if v['rule2_acc_within_1pp'] else 'FAIL':>5}")
        print("  alarms:", {d: {k: v for k, v in a.items() if k in
                                ('alarms_total', 'cancelled', 'proceeded', 'ignored_during_wait',
                                 'truncated')} for d, a in alarms[w].items()})
    print("\ncapture fraction (training ops, lambda 1e3):",
          {k: (None if v["capture_fraction"] is None else round(v["capture_fraction"], 3))
           for k, v in capture.items()})
    print("missed real drift (W=200):", payload["missed_real_drift_W200"])
    print(f"\nwrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
