"""Phase 10 — oracle upper bound by full-information replay.

DECLARED FOLLOW-UP measurement on the existing design. It changes no committed
result: the trajectory is the MechanismSelector run exactly as the grid ran it,
and the extra actions are evaluated from saved state and then discarded.

## Why this is exact rather than estimated

We are in simulation, so at every alarm each action can actually be executed
from the same snapshot. No off-policy estimation is needed.

## Per alarm

From the snapshot (current model, window, reference accuracy, stream position)
four branches are executed:

- **SKIP** - keep the model, no training.
- **NUDGE** - nudge a copy on the older 80% of the window.
- **REBUILD** - fresh model on the full window.
- **DEFER** - keep the model and re-decide once the buffer reaches the minimum.
  In a per-alarm frame DEFER pays no training and carries the current model's
  errors, so it coincides with SKIP; its value shows up only where the guard
  would otherwise force a rebuild, which is exactly what O2 is meant to expose.

Each branch is charged:

    Ops(a) = training work units
           + evaluation ops (inference units x holdout rows, per scoring the action performs)
           + inference ops over the horizon (inference units x H)

and scored by its errors over the next H rows. Inference units come from
`footprint.measure_footprint` on a probe taken from the window, so a shallower
rebuilt model is correctly cheaper to query than a deep initial one -- the effect
Phase 9.5 found.

Horizons: H = rows until the next alarm on the real trajectory (capped at
`--max-horizon`), plus fixed H = 500 and 1,000 as sensitivity.

    J(a) = Ops(a) + lambda * Errors(a)

reported over a grid of lambda rather than at one chosen value.

## Outputs

- **O1** best of SKIP/NUDGE/REBUILD, guard kept.
- **O2** best of all four, guard removed.
- **G1** = cost(selector) - cost(O1); **G2** = cost(selector) - cost(O2).
- cost(NeverAdapt) - cost(O1): does any adaptation beat doing nothing in hindsight?

Declared subset (agreed): ADWIN only, elec2 and covtype, which carry most of the
alarm volume; insects cells have 5-11 alarms each, too few to estimate headroom.

Output: results/analysis/phase10_oracle.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from footprint import measure_footprint  # noqa: E402
from mechanisms import GRID_MODELS, NudgeMechanism, RebuildMechanism, anchor_missing_classes, make_spec  # noqa: E402
from runner import RunConfig, default_detector, scorer_for  # noqa: E402
from selector import MechanismSelector, NUDGE, REBUILD, SKIP, temporal_split  # noqa: E402

CFG = RunConfig(track_energy=False)
DETERMINISTIC = ("xgb", "gnb")
ACTIONS = ("SKIP", "NUDGE", "REBUILD", "DEFER")
LAMBDA_GRID = [0.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
PROBE_ROWS = 200


def branch_cost(model, X_win, y_win, spec, seed, action, probe):
    """Execute one action from the snapshot; return (model, train_ops, n_scorings)."""
    if action in ("SKIP", "DEFER"):
        return model, 0, 1                     # the policy still scores the incoming model
    X_train, y_train, _, _ = temporal_split(X_win, y_win, CFG.holdout_frac)
    if action == "NUDGE":
        cand, cost = NudgeMechanism(spec, seed=seed).apply(copy.deepcopy(model), X_train, y_train, None)
        return cand, int(cost.work_units), 2   # incoming + nudged
    fresh, cost = RebuildMechanism(spec, seed=seed).apply(None, X_win, y_win, None)
    return fresh, int(cost.work_units), 2      # incoming + rebuilt


def run_cell(dataset: str, model_name: str, seed: int, max_horizon: int,
             limit_alarms: int | None) -> dict:
    X, y = ds.load_stream(dataset, cap=ds.COVTYPE_DEFAULT_CAP if dataset == "covtype" else None)
    classes = np.unique(y)
    scorer, _ = scorer_for(classes)
    spec = make_spec(model_name, classes)

    n = len(X)
    n_init = int(n * CFG.init_frac)
    n_init_train = int(n_init * (1 - CFG.init_holdout_frac))
    model = spec.factory(seed)
    X_fit, y_fit, kwargs = anchor_missing_classes(X[:n_init_train], y[:n_init_train], classes)
    model.fit(X_fit, y_fit, **kwargs)
    reference = float(scorer(y[n_init_train:n_init], model.predict(X[n_init_train:n_init])))

    policy = MechanismSelector(NudgeMechanism(spec, seed=seed), RebuildMechanism(spec, seed=seed),
                               holdout_frac=CFG.holdout_frac, floor_drop=CFG.floor_drop,
                               scorer=scorer, seed=seed)
    detector = default_detector(CFG)
    buffer: deque = deque(maxlen=CFG.buffer_size)

    records, pending = [], None      # `pending` holds the previous alarm awaiting its H_next
    buf_start, pending_since = n_init, None
    ref_start, ref_ready = None, True
    preds = np.empty(n - n_init, dtype=y.dtype)
    row, t_branch = n_init, 0.0

    while row < n:
        end = min(row + CFG.chunk_size, n)
        chunk = model.predict(X[row:end])
        adapted_at = None
        for off, r in enumerate(range(row, end)):
            preds[r - n_init] = chunk[off]
            buffer.append(r)
            if not ref_ready and r - ref_start + 1 == CFG.n_ref:
                reference = float(scorer(y[ref_start:r + 1], preds[ref_start - n_init:r + 1 - n_init]))
                ref_ready = True
            detector.update(int(chunk[off] != y[r]))
            if detector.drift_detected and pending_since is None:
                pending_since = r
            if pending_since is None:
                continue
            window_start = max(buf_start, r + 1 - CFG.buffer_size)
            if r + 1 - window_start < CFG.min_buffer:
                continue

            # ---------------- snapshot and branch
            t0 = time.perf_counter()
            X_win, y_win = X[window_start:r + 1], y[window_start:r + 1]
            _, _, X_hold, _ = temporal_split(X_win, y_win, CFG.holdout_frac)
            probe = X_win[:min(PROBE_ROWS, len(X_win))]
            lo, hi = r + 1, min(r + 1 + max_horizon, n)

            branches = {}
            for action in ACTIONS:
                cand, train_ops, n_scorings = branch_cost(model, X_win, y_win, spec, seed, action, probe)
                units = measure_footprint(cand, probe, timing_repeats=1).inference_units
                err = (cand.predict(X[lo:hi]) != y[lo:hi]).astype(np.int8) if hi > lo else np.zeros(0, np.int8)
                branches[action] = {
                    "train_ops": train_ops,
                    "eval_ops": float(units * len(X_hold) * n_scorings),
                    "infer_units_per_row": float(units),
                    "cum_err": np.cumsum(err),
                }
            t_branch += time.perf_counter() - t0

            result = policy.adapt(model, X_win, y_win, reference)   # the real trajectory
            rec = {
                "alarm_row": int(r), "window_rows": int(len(X_win)),
                "reference": float(reference),
                "acc_before": float(result.acc_before),
                "deficit": float(reference - result.acc_before),
                "rows_since_previous_alarm": int(r - pending["alarm_row"]) if pending else None,
                "policy_action": result.action,
                "degraded_to_rebuild": bool(result.degraded_to_rebuild),
                "policy_work_units": int(result.cost.work_units),
                "horizon_available": int(hi - lo),
                "branches": branches,
            }
            if pending is not None:              # fill the previous alarm's H_next
                pending["h_next"] = int(min(r - pending["alarm_row"], max_horizon))
                records.append(pending)
            pending = rec

            model = result.model
            buffer.clear()
            buf_start = r + 1
            pending_since = None
            ref_start, ref_ready = r + 1, False
            adapted_at = r
            if limit_alarms is not None and len(records) >= limit_alarms:
                pending["h_next"] = int(min(max_horizon, hi - lo))
                records.append(pending)
                return finish(records, dataset, model_name, seed, t_branch, truncated=True)
            break
        row = adapted_at + 1 if adapted_at is not None else end

    if pending is not None:
        pending["h_next"] = int(min(max_horizon, pending["horizon_available"]))
        records.append(pending)
    return finish(records, dataset, model_name, seed, t_branch, truncated=False)


def errors_at(branch: dict, h: int) -> int:
    cum = branch["cum_err"]
    if len(cum) == 0 or h <= 0:
        return 0
    return int(cum[min(h, len(cum)) - 1])


def finish(records, dataset, model_name, seed, t_branch, truncated) -> dict:
    """Collapse the per-alarm branch data into J-costs over the lambda grid."""
    out = {"dataset": dataset, "model": model_name, "seed": seed,
           "n_alarms": len(records), "branch_seconds": t_branch,
           "truncated": truncated, "horizons": {}, "per_alarm": []}

    for rec in records:
        row = {k: rec[k] for k in ("alarm_row", "window_rows", "reference", "acc_before",
                                   "deficit", "rows_since_previous_alarm", "policy_action",
                                   "degraded_to_rebuild", "policy_work_units", "h_next",
                                   "horizon_available")}
        row["h"] = {}
        row["branch_ops"] = {a: {"train": b["train_ops"], "eval": b["eval_ops"],
                                 "infer_per_row": b["infer_units_per_row"]}
                             for a, b in rec["branches"].items()}
        for hname in ("next", "500", "1000"):
            h = rec["h_next"] if hname == "next" else min(int(hname), rec["horizon_available"])
            row["h"][hname] = {a: [b["train_ops"] + b["eval_ops"] + b["infer_units_per_row"] * h,
                                   errors_at(b, h), h]
                               for a, b in rec["branches"].items()}
        out["per_alarm"].append(row)

    for hname in ("next", "500", "1000"):
        totals = {a: {"ops": 0.0, "errors": 0} for a in ACTIONS}
        per_lambda = {str(lam): {"policy": 0.0, "O1": 0.0, "O2": 0.0, "never": 0.0,
                                 "o1_counts": {a: 0 for a in ACTIONS},
                                 "o2_counts": {a: 0 for a in ACTIONS}} for lam in LAMBDA_GRID}
        for rec in records:
            h = rec["h_next"] if hname == "next" else min(int(hname), rec["horizon_available"])
            costs = {}
            for a, b in rec["branches"].items():
                ops = b["train_ops"] + b["eval_ops"] + b["infer_units_per_row"] * h
                errs = errors_at(b, h)
                costs[a] = (ops, errs)
                totals[a]["ops"] += ops
                totals[a]["errors"] += errs
            for lam in LAMBDA_GRID:
                j = {a: ops + lam * errs for a, (ops, errs) in costs.items()}
                o1 = min(("SKIP", "NUDGE", "REBUILD"), key=lambda a: j[a])
                o2 = min(ACTIONS, key=lambda a: j[a])
                slot = per_lambda[str(lam)]
                slot["policy"] += j[rec["policy_action"]]
                slot["O1"] += j[o1]
                slot["O2"] += j[o2]
                slot["never"] += j["SKIP"]
                slot["o1_counts"][o1] += 1
                slot["o2_counts"][o2] += 1
        for lam, slot in per_lambda.items():
            slot["G1"] = slot["policy"] - slot["O1"]
            slot["G2"] = slot["policy"] - slot["O2"]
            slot["G1_share_of_policy"] = slot["G1"] / slot["policy"] if slot["policy"] else None
            slot["G2_share_of_policy"] = slot["G2"] / slot["policy"] if slot["policy"] else None
            slot["never_minus_O1"] = slot["never"] - slot["O1"]
        out["horizons"][hname] = {"totals": {a: totals[a] for a in ACTIONS}, "by_lambda": per_lambda}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=["elec2", "covtype"])
    ap.add_argument("--models", nargs="*", default=list(GRID_MODELS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--max-horizon", type=int, default=1000)
    ap.add_argument("--limit-alarms", type=int, default=None,
                    help="stop each cell after N alarms (timing mode)")
    ap.add_argument("--out", default="results/analysis/phase10_oracle.json")
    args = ap.parse_args()

    cells = []
    for dataset in args.datasets:
        for model_name in args.models:
            for seed in ([0] if model_name in DETERMINISTIC else args.seeds):
                t0 = time.perf_counter()
                c = run_cell(dataset, model_name, seed, args.max_horizon, args.limit_alarms)
                c["wall_seconds"] = time.perf_counter() - t0
                per = c["branch_seconds"] / c["n_alarms"] if c["n_alarms"] else float("nan")
                print(f"{dataset:10} {model_name:4} seed {seed}: {c['n_alarms']:4} alarms | "
                      f"{c['wall_seconds']:7.1f}s total | {per:6.3f}s/alarm branching", flush=True)
                cells.append(c)

    for c in cells:                      # numpy arrays are not JSON-serialisable
        for h in c["horizons"].values():
            pass
    payload = {
        "label": "declared follow-up (oracle upper bound)",
        "subset": {"detector": "adwin", "datasets": args.datasets, "models": args.models,
                   "seeds": args.seeds, "limit_alarms": args.limit_alarms},
        "lambda_grid": LAMBDA_GRID,
        "max_horizon": args.max_horizon,
        "probe_rows": PROBE_ROWS,
        "cells": [{k: v for k, v in c.items()} for c in cells],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
