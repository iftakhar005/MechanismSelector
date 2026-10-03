"""DIAGNOSTIC — does a kept XGBoost nudge pay for itself before the next adaptation?

A nudge saves training ops now and costs inference ops on every later prediction,
because the booster grows. This measures both sides per kept nudge, against the
model a rebuild would have produced at that same alarm (the oracle's
counterfactual, fitted from the same snapshot and then discarded).

Random Forest is the size-neutral control: its nudge adds k trees and retires k,
so the model does not grow and the extra inference cost should be about zero.

Per kept nudge under the MechanismSelector trajectory:

- `training_saved` = counterfactual rebuild's training ops - the nudge's
- `extra_ops_per_prediction` = nudged model's inference units per row
  - counterfactual rebuild's
- `predictions_to_next_adaptation` = rows until the selector next adapts (or
  stream end)
- `net` = training_saved - extra_ops_per_prediction x predictions
- `break_even_predictions` = training_saved / extra_ops_per_prediction

The trajectory is the selector's real one; nothing here changes a committed
result. Declared as a diagnostic.

Output: results/analysis/nudge_growth_diagnostic.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from footprint import measure_footprint  # noqa: E402
from mechanisms import NudgeMechanism, RebuildMechanism, anchor_missing_classes, make_spec  # noqa: E402
from runner import RunConfig, default_detector, scorer_for  # noqa: E402
from selector import MechanismSelector, NUDGE  # noqa: E402

CFG = RunConfig(track_energy=False)
PROBE_ROWS = 200
DETERMINISTIC = ("xgb", "gnb")


def run_cell(dataset: str, model_name: str, seed: int) -> dict:
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
    preds = np.empty(n - n_init, dtype=y.dtype)

    events, pending_event = [], None
    buf_start, pending_since = n_init, None
    ref_start, ref_ready = None, True
    row = n_init

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

            X_win, y_win = X[window_start:r + 1], y[window_start:r + 1]
            result = policy.adapt(model, X_win, y_win, reference)

            if pending_event is not None:                      # close the previous kept nudge
                pending_event["predictions_to_next_adaptation"] = int(r - pending_event["alarm_row"])
                events.append(pending_event)
                pending_event = None

            if result.action == NUDGE:
                probe = X_win[:min(PROBE_ROWS, len(X_win))]
                fresh, reb_cost = RebuildMechanism(spec, seed=seed).apply(None, X_win, y_win, None)
                u_before = measure_footprint(model, probe, timing_repeats=1).inference_units
                u_nudge = measure_footprint(result.model, probe, timing_repeats=1).inference_units
                u_rebuild = measure_footprint(fresh, probe, timing_repeats=1).inference_units
                saved = int(reb_cost.work_units) - int(result.cost.work_units)
                extra = float(u_nudge - u_rebuild)
                pending_event = {
                    "alarm_row": int(r), "window_rows": int(len(X_win)),
                    "nudge_train_ops": int(result.cost.work_units),
                    "counterfactual_rebuild_train_ops": int(reb_cost.work_units),
                    "training_saved": saved,
                    "incoming_units_per_prediction": float(u_before),
                    "growth_ops_per_prediction": float(u_nudge - u_before),
                    "window_era_ops_per_prediction": float(u_before - u_rebuild),
                    "nudged_units_per_prediction": float(u_nudge),
                    "rebuild_units_per_prediction": float(u_rebuild),
                    "extra_ops_per_prediction": extra,
                    "break_even_predictions": (saved / extra) if extra > 0 else None,
                }

            model = result.model
            buffer.clear()
            buf_start = r + 1
            pending_since = None
            ref_start, ref_ready = r + 1, False
            adapted_at = r
            break
        row = adapted_at + 1 if adapted_at is not None else end

    if pending_event is not None:
        pending_event["predictions_to_next_adaptation"] = int(n - pending_event["alarm_row"])
        events.append(pending_event)

    for e in events:
        e["net"] = e["training_saved"] - e["extra_ops_per_prediction"] * e["predictions_to_next_adaptation"]
    return {"dataset": dataset, "model": model_name, "seed": seed,
            "n_kept_nudges": len(events), "events": events}


def summarise(cells):
    out = {}
    for key in sorted({(c["dataset"], c["model"]) for c in cells}):
        evs = [e for c in cells if (c["dataset"], c["model"]) == key for e in c["events"]]
        if not evs:
            out[f"{key[0]}/{key[1]}"] = {"n_kept_nudges": 0}
            continue
        be = [e["break_even_predictions"] for e in evs if e["break_even_predictions"] is not None]
        out[f"{key[0]}/{key[1]}"] = {
            "n_kept_nudges": len(evs),
            "median_training_saved": float(np.median([e["training_saved"] for e in evs])),
            "median_extra_ops_per_prediction": float(np.median([e["extra_ops_per_prediction"] for e in evs])),
            "median_growth_ops_per_prediction": float(np.median([e["growth_ops_per_prediction"] for e in evs])),
            "median_window_era_ops_per_prediction": float(np.median([e["window_era_ops_per_prediction"] for e in evs])),
            "median_predictions_to_next_adaptation": float(np.median(
                [e["predictions_to_next_adaptation"] for e in evs])),
            "median_net": float(np.median([e["net"] for e in evs])),
            "share_net_positive": float(np.mean([e["net"] > 0 for e in evs])),
            "median_break_even_predictions": float(np.median(be)) if be else None,
            "share_break_even_reached": float(np.mean(
                [(e["break_even_predictions"] is not None
                  and e["predictions_to_next_adaptation"] > e["break_even_predictions"]) for e in evs])),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(ds.STREAMS))
    ap.add_argument("--models", nargs="*", default=["xgb", "rf"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/analysis/nudge_growth_diagnostic.json")
    args = ap.parse_args()

    cells = []
    for dataset in args.datasets:
        for model_name in args.models:
            for seed in ([0] if model_name in DETERMINISTIC else args.seeds):
                c = run_cell(dataset, model_name, seed)
                print(f"{dataset:20} {model_name:4} seed {seed}: {c['n_kept_nudges']:4} kept nudges",
                      flush=True)
                cells.append(c)

    payload = {"label": "diagnostic",
               "question": "does a kept nudge pay for itself before the next adaptation?",
               "counterfactual": "a rebuild fitted from the same snapshot and discarded",
               "probe_rows": PROBE_ROWS,
               "summary": summarise(cells), "cells": cells}
    out = ROOT / args.out
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'cell':24}{'n':>5}{'train saved':>14}{'extra/pred':>12}{'preds':>9}"
          f"{'net':>14}{'break-even':>12}{'reached':>9}")
    for k, v in payload["summary"].items():
        if not v["n_kept_nudges"]:
            print(f"{k:24}{0:>5}")
            continue
        be = v["median_break_even_predictions"]
        print(f"{k:24}{v['n_kept_nudges']:5}{v['median_training_saved']:14,.0f}"
              f"{v['median_extra_ops_per_prediction']:12.1f}"
              f"{v['median_predictions_to_next_adaptation']:9.0f}{v['median_net']:14,.0f}"
              f"{(be if be is not None else float('nan')):12,.0f}{v['share_break_even_reached']:9.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
