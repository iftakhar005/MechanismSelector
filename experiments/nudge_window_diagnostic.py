"""Diagnostic: does the nudge fail because of the mechanism, or because of the data it sees?

The selector fits a nudge on `X_train` -- the older 80% of the window -- so the
newest 20% can judge it out-of-sample. A kept nudge therefore never trains on
those newest rows (see the defect noted in `selector.py`). That raises a
question the main grid cannot answer: how much of "cheap repair does not work"
is the mechanism being weak, and how much is the mechanism being starved of the
rows that describe the drift?

This is a SEPARATE experiment. It does not touch, re-run or re-interpret the
grid in `results/grid/`, and it changes nothing in `src/`.

## Design

One prequential run per (stream, model), following AlwaysNudge: nudge at every
served alarm. At each alarm, from the SAME incoming model and the SAME window,
two candidates are fitted:

    nudge_80   fit on the older 80% of the window   (what the selector does)
    nudge_100  fit on the whole window              (the counterfactual)

Both are scored on the next `n_ref` = 200 rows of the stream, which neither has
trained on, so the comparison is paired and out-of-sample for both. The
incoming model is scored on the same rows as the reference point.

The run then CONTINUES WITH `nudge_80`, exactly as the real policy would, so
the trajectory, the alarms and the windows are the ones the 80% rule actually
produces. `nudge_100` is measured and discarded; it never influences the run.

Floor accounting uses the runner's rule: `reference_accuracy` is measured on
the 200 rows after an adaptation, carried forward if the next alarm arrives
first, and `floor = reference - floor_drop`. "Cleared the floor" is then asked
of both candidates on the same out-of-sample rows -- a fair question to ask of
both, and not the in-window holdout test the selector actually applies.

Output: results/analysis/nudge_window_diagnostic.json
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
from mechanisms import GRID_MODELS, NudgeMechanism, anchor_missing_classes, make_spec  # noqa: E402
from runner import default_detector, scorer_for, RunConfig  # noqa: E402
from selector import temporal_split  # noqa: E402

CFG = RunConfig(track_energy=False)
CHUNK = 2000


def initial_model(spec, X, y, n_train, seed):
    model = spec.factory(seed)
    X_fit, y_fit, kwargs = anchor_missing_classes(X[:n_train], y[:n_train], spec.classes)
    model.fit(X_fit, y_fit, **kwargs)
    return model


def score_rows(model, X, y, lo, hi, scorer):
    """Accuracy of `model` on rows [lo, hi) -- rows it has not trained on."""
    if hi <= lo:
        return None
    return float(scorer(y[lo:hi], model.predict(X[lo:hi])))


def run_cell(dataset: str, model_name: str, seed: int) -> dict:
    X, y = ds.load_stream(dataset, cap=ds.COVTYPE_DEFAULT_CAP if dataset == "covtype" else None)
    classes = np.unique(y)
    scorer, scorer_name = scorer_for(classes)
    spec = make_spec(model_name, classes)
    nudge = NudgeMechanism(spec, seed=seed)

    n = len(X)
    n_init = int(n * CFG.init_frac)
    n_init_train = int(n_init * (1 - CFG.init_holdout_frac))
    model = initial_model(spec, X, y, n_init_train, seed)
    reference = float(scorer(y[n_init_train:n_init], model.predict(X[n_init_train:n_init])))

    detector = default_detector(CFG)
    buffer: deque = deque(maxlen=CFG.buffer_size)
    preds = np.empty(n - n_init, dtype=y.dtype)

    events: list[dict] = []
    ref_start, ref_ready = None, True
    row = n_init
    while row < n:
        end = min(row + CHUNK, n)
        chunk = model.predict(X[row:end])
        alarm_at = None
        for i, r in enumerate(range(row, end)):
            preds[r - n_init] = chunk[i]
            buffer.append(r)
            detector.update(int(chunk[i] != y[r]))
            if not ref_ready and r - ref_start + 1 == CFG.n_ref:
                reference = float(scorer(y[ref_start:r + 1], preds[ref_start - n_init:r + 1 - n_init]))
                ref_ready = True
            if detector.drift_detected and len(buffer) >= CFG.min_buffer:
                alarm_at = r
                break
        if alarm_at is None:
            row = end
            continue

        rows = np.fromiter(buffer, dtype=int)
        X_win, y_win = X[rows], y[rows]
        X_train, y_train, _, _ = temporal_split(X_win, y_win, CFG.holdout_frac)
        lo, hi = alarm_at + 1, min(alarm_at + 1 + CFG.n_ref, n)

        cand80, cost80 = nudge.apply(copy.deepcopy(model), X_train, y_train, None)
        cand100, cost100 = nudge.apply(copy.deepcopy(model), X_win, y_win, None)

        acc_in = score_rows(model, X, y, lo, hi, scorer)
        acc80 = score_rows(cand80, X, y, lo, hi, scorer)
        acc100 = score_rows(cand100, X, y, lo, hi, scorer)
        if acc_in is not None:
            pred_in = model.predict(X[lo:hi])
            floor = reference - CFG.floor_drop
            events.append({
                "alarm_row": int(alarm_at),
                "window_rows": int(len(rows)),
                "train_rows": int(len(X_train)),
                "n_ref_rows": int(hi - lo),
                "reference": reference,
                "floor": floor,
                "acc_incoming": acc_in,
                "acc_nudge80": acc80,
                "acc_nudge100": acc100,
                "gain80_pp": (acc80 - acc_in) * 100,
                "gain100_pp": (acc100 - acc_in) * 100,
                "delta_pp": (acc100 - acc80) * 100,
                "cleared80": bool(acc80 >= floor),
                "cleared100": bool(acc100 >= floor),
                "unchanged80": bool(np.array_equal(cand80.predict(X[lo:hi]), pred_in)),
                "unchanged100": bool(np.array_equal(cand100.predict(X[lo:hi]), pred_in)),
                "work_units80": int(cost80.work_units),
                "work_units100": int(cost100.work_units),
            })

        model = cand80          # the run continues exactly as the 80% rule would
        buffer.clear()
        ref_start, ref_ready = alarm_at + 1, False
        row = alarm_at + 1

    return {
        "dataset": dataset, "model": model_name, "seed": seed, "scorer": scorer_name,
        "n_alarms_measured": len(events), "events": events,
    }


def summarise(cells: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = {}
    for cell in cells:
        by_model.setdefault(cell["model"], []).extend(cell["events"])

    def block(events: list[dict]) -> dict:
        if not events:
            return {"n": 0}
        d = np.array([e["delta_pp"] for e in events])
        g80 = np.array([e["gain80_pp"] for e in events])
        g100 = np.array([e["gain100_pp"] for e in events])
        out = {
            "n": len(events),
            "median_gain80_pp": float(np.median(g80)),
            "median_gain100_pp": float(np.median(g100)),
            "median_delta_pp": float(np.median(d)),
            "mean_delta_pp": float(np.mean(d)),
            "share_full_window_better": float(np.mean(d > 0)),
            "share_identical": float(np.mean(d == 0)),
            "cleared80": float(np.mean([e["cleared80"] for e in events])),
            "cleared100": float(np.mean([e["cleared100"] for e in events])),
            "unchanged80": float(np.mean([e["unchanged80"] for e in events])),
            "unchanged100": float(np.mean([e["unchanged100"] for e in events])),
            "median_work_ratio": float(np.median(
                [e["work_units100"] / e["work_units80"] for e in events if e["work_units80"]]
            )) if any(e["work_units80"] for e in events) else None,
        }
        nonzero = d[d != 0]
        if len(nonzero) >= 6:
            from scipy import stats
            out["wilcoxon_p"] = float(stats.wilcoxon(nonzero).pvalue)
            ranks = stats.rankdata(np.abs(nonzero))
            pos = ranks[nonzero > 0].sum()
            out["rank_biserial"] = float(2 * pos / ranks.sum() - 1)
        return out

    return {
        "by_model": {m: block(ev) for m, ev in by_model.items()},
        "by_cell": {f"{c['dataset']}/{c['model']}": block(c["events"]) for c in cells},
        "overall": block([e for ev in by_model.values() for e in ev]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(ds.STREAMS))
    ap.add_argument("--models", nargs="*", default=list(GRID_MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/analysis/nudge_window_diagnostic.json")
    args = ap.parse_args()

    cells = []
    for dataset in args.datasets:
        for model_name in args.models:
            cell = run_cell(dataset, model_name, args.seed)
            print(f"{dataset}/{model_name}: {cell['n_alarms_measured']} alarms measured", flush=True)
            cells.append(cell)

    out = Path(ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": "paired per alarm: nudge on older 80% vs nudge on full window; "
                  "both scored on the next 200 rows; run continues with the 80% nudge",
        "config": {"init_frac": CFG.init_frac, "buffer_size": CFG.buffer_size,
                   "min_buffer": CFG.min_buffer, "n_ref": CFG.n_ref,
                   "holdout_frac": CFG.holdout_frac, "floor_drop": CFG.floor_drop,
                   "detector": CFG.detector, "adwin_delta": CFG.adwin_delta,
                   "seed": args.seed},
        "summary": summarise(cells),
        "cells": cells,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
