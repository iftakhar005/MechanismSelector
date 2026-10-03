"""ConfirmMD3 vs WaitAndCheck vs the selector — comparison only, no verdict.

DIAGNOSTIC. ConfirmMD3 is WaitAndCheck with one change: the cancel rule. Instead
of cancelling when the post-alarm accuracy is at or above `reference - 2 points`,
it cancels unless the drop exceeds Theta standard errors of the reference
estimate:

    cancel  <=>  reference - acc_W  <=  Theta * sigma,
    sigma = sqrt(p (1 - p) / W),  p = reference accuracy,  Theta = 2.

This approximates MD3 (Sethi & Kantardzic 2017), which uses
`Acc_Ref - Acc_Labeled > Theta * sigma_Acc` with sigma estimated from training
data. Two approximations are ours and are stated: sigma is computed from the
reference estimate on W rows rather than from a training-set distribution, and
for multi-class streams the accuracy is balanced accuracy, for which the binomial
standard error is only approximate.

No verdict is attached: the pre-declared rules of Phase 14 were written for
WaitAndCheck and are not re-used here.

Output: results/analysis/md3_compare.json
"""

from __future__ import annotations

import csv
import importlib.util
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC = ("xgb", "gnb")


def _inf_module():
    spec = importlib.util.spec_from_file_location(
        "phase9_inference_cost", ROOT / "experiments" / "phase9_inference_cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INF = _inf_module()


def load(grid_dir: Path):
    out = {}
    for r in csv.DictReader((grid_dir / "grid.csv").open()):
        if r["model"] in DETERMINISTIC and int(r["seed"]) != 0:
            continue
        traj = INF.size_trajectory(grid_dir / "events", r["dataset"], r["model"],
                                   r["policy_key"], int(r["seed"]))
        infer, _ = INF.inference_ops(r, traj)
        out[(r["dataset"], r["model"], int(r["seed"]))] = {
            "training": float(r["total_work_units"]), "inference": infer,
            "acc_balanced": float(r["mean_prequential_balanced_accuracy"]) * 100,
            "n_alarms": int(float(r["n_alarms"])),
            "cancelled": int(float(r.get("n_waits_cancelled") or 0)),
            "proceeded": int(float(r.get("n_waits_proceeded") or 0)),
            "n_degraded": int(float(r["n_degraded"])),
        }
    return out


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, run)
    return out


def main() -> None:
    sel = load(ROOT / "results/grid")
    sel = {k: v for k, v in sel.items() if True}
    sel_only = {k: v for k, v in load(ROOT / "results/grid").items()}
    selector = {k: v for k, v in sel_only.items()}
    # keep only the selector rows from the frozen grid
    selector = {}
    for r in csv.DictReader((ROOT / "results/grid/grid.csv").open()):
        if r["policy_key"] != "mechanism_selector":
            continue
        if r["model"] in DETERMINISTIC and int(r["seed"]) != 0:
            continue
        traj = INF.size_trajectory(ROOT / "results/grid/events", r["dataset"], r["model"],
                                   r["policy_key"], int(r["seed"]))
        infer, _ = INF.inference_ops(r, traj)
        selector[(r["dataset"], r["model"], int(r["seed"]))] = {
            "training": float(r["total_work_units"]), "inference": infer,
            "acc_balanced": float(r["mean_prequential_balanced_accuracy"]) * 100,
            "n_alarms": int(float(r["n_alarms"])), "n_degraded": int(float(r["n_degraded"])),
            "cancelled": 0, "proceeded": 0}

    arms = {"waitandcheck_200": load(ROOT / "results/grid_wait200"),
            "waitandcheck_500": load(ROOT / "results/grid_wait500"),
            "confirmmd3_200": load(ROOT / "results/grid_md3_200"),
            "confirmmd3_500": load(ROOT / "results/grid_md3_500")}

    datasets = sorted({k[0] for k in selector})
    per_dataset = {}
    for ds_ in datasets:
        keys = [k for k in selector if k[0] == ds_]
        row = {"selector": {
            "training": st.mean(selector[k]["training"] for k in keys),
            "acc_balanced": st.mean(selector[k]["acc_balanced"] for k in keys),
            "alarms": sum(selector[k]["n_alarms"] for k in keys),
            "cancelled": 0,
            "guard_rebuilds": sum(selector[k]["n_degraded"] for k in keys)}}
        for name, arm in arms.items():
            kk = [k for k in keys if k in arm]
            a = np.array([selector[k]["training"] for k in kk])
            b = np.array([arm[k]["training"] for k in kk])
            acc_a = np.array([selector[k]["acc_balanced"] for k in kk])
            acc_b = np.array([arm[k]["acc_balanced"] for k in kk])
            row[name] = {
                "training": float(b.mean()),
                "training_ratio": float(b.mean() / a.mean()) if a.mean() else None,
                "acc_balanced": float(acc_b.mean()),
                "accuracy_delta_pp": float(acc_b.mean() - acc_a.mean()),
                "alarms": sum(arm[k]["n_alarms"] for k in kk),
                "cancelled": sum(arm[k]["cancelled"] for k in kk),
                "proceeded": sum(arm[k]["proceeded"] for k in kk),
                "guard_rebuilds": sum(arm[k]["n_degraded"] for k in kk),
            }
        per_dataset[ds_] = row

    # ConfirmMD3 vs WaitAndCheck at the same W, paired, Holm across the five streams
    tests = {}
    for w in (200, 500):
        rows, ps = [], []
        for ds_ in datasets:
            keys = [k for k in arms[f"confirmmd3_{w}"] if k[0] == ds_
                    and k in arms[f"waitandcheck_{w}"]]
            for metric in ("training", "acc_balanced"):
                pass
            a = np.array([arms[f"waitandcheck_{w}"][k]["training"] for k in keys])
            b = np.array([arms[f"confirmmd3_{w}"][k]["training"] for k in keys])
            d = b - a
            nz = d[d != 0]
            p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 6 else None
            rows.append({"dataset": ds_, "metric": "training", "n": len(keys),
                         "median_difference": float(np.median(d)), "p": p})
            if p is not None:
                ps.append(p)
        adj = iter(holm(ps))
        for r in rows:
            r["p_holm"] = next(adj) if r["p"] is not None else None
        tests[w] = rows

    payload = {
        "label": "diagnostic (comparison only, no verdict)",
        "rule": "cancel if reference - acc_W <= Theta*sigma, Theta=2, sigma=sqrt(p(1-p)/W)",
        "approximation": "sigma from the reference estimate over W rows rather than from a "
                         "training-set distribution; balanced accuracy is not a binomial "
                         "proportion, so sigma is approximate on multi-class streams",
        "per_dataset": per_dataset, "md3_vs_wait_training": tests,
    }
    (ROOT / "results/analysis/md3_compare.json").write_text(json.dumps(payload, indent=2),
                                                            encoding="utf-8")

    print(f"{'dataset':22}{'arm':20}{'training':>14}{'ratio':>8}{'acc':>8}{'Δacc':>8}"
          f"{'cancelled':>11}{'guard':>8}")
    for ds_, row in per_dataset.items():
        s = row["selector"]
        print(f"{ds_:22}{'selector':20}{s['training']:14,.0f}{1.0:8.2f}{s['acc_balanced']:8.1f}"
              f"{0.0:8.2f}{0:11}{s['guard_rebuilds']:8}")
        for name in ("waitandcheck_200", "confirmmd3_200", "waitandcheck_500", "confirmmd3_500"):
            v = row[name]
            print(f"{'':22}{name:20}{v['training']:14,.0f}{v['training_ratio']:8.2f}"
                  f"{v['acc_balanced']:8.1f}{v['accuracy_delta_pp']:+8.2f}"
                  f"{v['cancelled']:11}{v['guard_rebuilds']:8}")
    print("\nConfirmMD3 vs WaitAndCheck, training ops (paired, Holm across 5 streams)")
    for w, rows in tests.items():
        for r in rows:
            p = "n/a" if r["p"] is None else f"{r['p']:.4g}"
            ph = "n/a" if r["p_holm"] is None else f"{r['p_holm']:.4g}"
            print(f"  W={w} {r['dataset']:22} median diff {r['median_difference']:+14,.0f}"
                  f"  p={p:>10}  Holm={ph:>10}")
    print("\nwrote results/analysis/md3_compare.json")


if __name__ == "__main__":
    main()
