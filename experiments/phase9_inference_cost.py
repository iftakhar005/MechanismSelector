"""Phase 9.5 DIAGNOSTIC — total cost when inference is counted, not just training.

DIAGNOSTIC of existing results. Reads committed logs only; runs no model and
writes only new files.

Training work units count what an adaptation costs once. They say nothing about
what the resulting model costs on every later prediction, and an XGBoost booster
nudged repeatedly grows from 100 to 365-790 rounds, so that omission is not
small.

## How inference operations are estimated

`grid.csv` records, per run, the model size and the measured inference units per
row (tree node visits for xgb/rf, parameter reads for sgd/gnb) for the **initial**
and the **final** model. The event log records `model_size_after` at every
adaptation, so the size trajectory across the stream is known exactly.

Between the two measured points the units-per-row are interpolated linearly in
model size:

    units(size) = initial_units + (final_units - initial_units)
                  * (size - initial_size) / (final_size - initial_size)

and held constant where size does not change. Total inference operations are then
the sum over stream rows of units(size at that row).

**This is an estimate, and it is labelled as one.** It is exact at the two
measured endpoints and exact for sgd/gnb (whose size never changes, so the
interpolation is a constant). For rf the size is constant but tree depth varies
with the training window, so the endpoints differ and the interpolation is flat
in size -- for those runs the mean of the two measured values is used and the
spread between them is reported as `rf_endpoint_spread`. Only a replay that
re-measures the footprint after every adaptation would remove the assumption.

Training and inference are reported separately and combined. They are **not the
same unit**: a training work unit is one row-pass, an inference unit is one node
visit or parameter read. The combined figure is reported because the instruction
asks for it, with that caveat attached.

Output: results/analysis/phase9_inference_cost.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETERMINISTIC = ("xgb", "gnb")
POLICIES = ("never_adapt", "always_rebuild", "always_nudge", "fixed_schedule", "mechanism_selector")


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def size_trajectory(events_dir: Path, dataset: str, model: str, policy: str, seed: int):
    f = events_dir / f"{dataset}__{model}__{policy}__{seed}.csv"
    if not f.exists():
        return []
    out = []
    for e in csv.DictReader(f.open()):
        row, size = e.get("adapt_row") or e.get("alarm_row"), e.get("model_size_after")
        if row in (None, "") or size in (None, ""):
            continue
        out.append((int(float(row)), float(size)))
    return sorted(out)


def inference_ops(run: dict, traj) -> tuple[float, float]:
    """Total inference operations over the stream, and the endpoint spread."""
    n_rows = num(run["n_stream_rows"])
    s0, s1 = num(run["initial_model_size"]), num(run["final_model_size"])
    u0, u1 = num(run["initial_inference_units"]), num(run["final_inference_units"])
    start_row = num(run["n_init_train_rows"])

    if s1 == s0:                      # size never changed: use the mean of the two measurements
        return float(np.mean([u0, u1]) * n_rows), abs(u1 - u0)

    def units(size: float) -> float:
        return u0 + (u1 - u0) * (size - s0) / (s1 - s0)

    total, prev_row, prev_size = 0.0, start_row, s0
    for row, size in traj:
        total += units(prev_size) * max(0.0, row - prev_row)
        prev_row, prev_size = row, size
    total += units(prev_size) * max(0.0, (start_row + n_rows) - prev_row)
    return float(total), abs(u1 - u0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="results/grid")
    ap.add_argument("--out", default="results/analysis/phase9_inference_cost.json")
    args = ap.parse_args()

    grid_dir = ROOT / args.grid
    runs = list(csv.DictReader((grid_dir / "grid.csv").open()))
    per_run, by_policy = [], defaultdict(lambda: {"train": 0.0, "infer": 0.0, "n": 0})
    by_cell_policy = defaultdict(dict)

    for r in runs:
        if r["model"] in DETERMINISTIC and int(r["seed"]) != 0:
            continue
        traj = size_trajectory(grid_dir / "events", r["dataset"], r["model"],
                               r["policy_key"], int(r["seed"]))
        infer, spread = inference_ops(r, traj)
        train = num(r["total_work_units"])
        per_run.append({
            "dataset": r["dataset"], "model": r["model"], "policy": r["policy_key"],
            "seed": int(r["seed"]), "training_work_units": train,
            "inference_units_total": infer, "combined": train + infer,
            "inference_share": infer / (train + infer) if (train + infer) else None,
            "endpoint_spread_units_per_row": spread,
            "n_adaptations": int(float(r["n_adaptations"])),
        })
        b = by_policy[r["policy_key"]]
        b["train"] += train
        b["infer"] += infer
        b["n"] += 1
        by_cell_policy[f"{r['dataset']}/{r['model']}"][r["policy_key"]] = train + infer

    pooled = {p: {"training": v["train"], "inference": v["infer"],
                  "combined": v["train"] + v["infer"], "runs": v["n"],
                  "inference_share": v["infer"] / (v["train"] + v["infer"])}
              for p, v in by_policy.items()}

    # does counting inference change which policy is cheapest in a cell?
    flips = []
    for cell, costs in by_cell_policy.items():
        train_only = {p: sum(x["training_work_units"] for x in per_run
                             if f"{x['dataset']}/{x['model']}" == cell and x["policy"] == p)
                      for p in costs}
        cheapest_train = min(train_only, key=train_only.get)
        cheapest_comb = min(costs, key=costs.get)
        if cheapest_train != cheapest_comb:
            flips.append({"cell": cell, "cheapest_training_only": cheapest_train,
                          "cheapest_combined": cheapest_comb})

    payload = {
        "label": "diagnostic",
        "question": "does counting per-prediction inference change the cost picture?",
        "method": "units-per-row interpolated linearly in model size between the two measured "
                  "endpoints in grid.csv, integrated over the logged size trajectory",
        "caveat": "training work units (row-passes) and inference units (node visits / parameter "
                  "reads) are different quantities; the combined figure is reported as instructed, "
                  "not as a claim that they are commensurable",
        "pooled_by_policy": pooled,
        "cells_where_cheapest_policy_changes": flips,
        "per_run": per_run,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'policy':20}{'training':>16}{'inference':>16}{'infer share':>13}")
    for p in POLICIES:
        if p not in pooled:
            continue
        v = pooled[p]
        print(f"{p:20}{v['training']:16,.0f}{v['inference']:16,.0f}{v['inference_share']:12.1%}")
    print(f"\ncells where the cheapest policy changes once inference is counted: {len(flips)}")
    for f in flips:
        print(f"  {f['cell']:24} {f['cheapest_training_only']} -> {f['cheapest_combined']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
