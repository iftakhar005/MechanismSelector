"""Does ADWIN fire on improvements? Measured from the detector, not the reference.

The event log's `accuracy_deficit` compares accuracy at alarm time against a
200-row reference, and that reference is itself noisy (interquartile ranges of
about +-20 points), so a negative deficit is not proof the alarm fired on an
improvement.

`NeverAdapt` allows a direct test. Its model never changes, so the exact error
stream ADWIN saw can be replayed: fit the initial model exactly as the runner
does, predict the stream, and feed the errors to a fresh ADWIN. At each alarm,
ADWIN drops the older part of its window. Comparing its error estimate just
before the update with the estimate over the retained, recent part gives the
direction of the change it detected:

- error rate went **up** -> the model got worse -> a degradation alarm
- error rate went **down** -> the model got better -> an improvement alarm

Correctness check: the replay must reproduce every NeverAdapt run's alarm count
in the grid exactly, or its directions mean nothing.

    python experiments/alarm_direction.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from river.drift import ADWIN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from mechanisms import GRID_MODELS, anchor_missing_classes, make_spec  # noqa: E402
from runner import RunConfig  # noqa: E402

warnings.filterwarnings("ignore")
DETERMINISTIC_MODELS = {"xgb", "gnb"}


def replay(X, y, model_name, seed, config: RunConfig):
    n = len(X)
    classes = np.unique(y)
    n_init = int(n * config.init_frac)
    n_init_train = int(n_init * (1 - config.init_holdout_frac))
    X0, y0, kwargs = anchor_missing_classes(X[:n_init_train], y[:n_init_train], classes)
    model = make_spec(model_name, classes).factory(seed).fit(X0, y0, **kwargs)
    errors = (model.predict(X[n_init:]) != y[n_init:]).astype(int)

    detector = ADWIN(delta=config.adwin_delta)
    alarms = []
    for step, err in enumerate(errors):
        before = detector.estimation
        detector.update(int(err))
        if detector.drift_detected:
            alarms.append({"row": n_init + step, "error_before": before, "error_after": detector.estimation})
    return alarms


def main() -> int:
    config = RunConfig(track_energy=False)
    with (ROOT / "results" / "grid" / "grid.csv").open(newline="", encoding="utf-8") as f:
        grid_alarms = {(r["dataset"], r["model"], int(r["seed"])): int(r["n_alarms"])
                       for r in csv.DictReader(f) if r["policy_key"] == "never_adapt"}

    report, mismatches = {}, []
    print(f"{'cell':<26}{'alarms':>8}{'error went up':>15}{'error went down':>17}   median error change at alarm (pp)")
    for dataset in ds.STREAMS:
        X, y = ds.load_stream(dataset)
        for model in GRID_MODELS:
            seeds = [0] if model in DETERMINISTIC_MODELS else range(5)
            changes = []
            for seed in seeds:
                alarms = replay(X, y, model, seed, config)
                expected = grid_alarms[(dataset, model, seed)]
                if len(alarms) != expected:
                    mismatches.append(f"{dataset}/{model}/seed{seed}: replay {len(alarms)} vs grid {expected}")
                changes += [100 * (a["error_after"] - a["error_before"]) for a in alarms]

            up = sum(c > 0 for c in changes)
            down = sum(c < 0 for c in changes)
            report[f"{dataset}/{model}"] = {
                "alarms": len(changes), "error_up": up, "error_down": down,
                "share_on_improvement": down / len(changes) if changes else None,
                "median_error_change_pp": float(np.median(changes)) if changes else None,
                "seeds": list(seeds),
            }
            share = f"{100 * down / len(changes):.0f}%" if changes else "--"
            med = f"{np.median(changes):+.1f}" if changes else "--"
            print(f"{dataset + '/' + model:<26}{len(changes):>8}{up:>15}{down:>12} ({share:>4}){med:>20}")

    total = sum(v["alarms"] for v in report.values())
    down = sum(v["error_down"] for v in report.values())
    print(f"\nall cells: {total} alarms, {down} ({100 * down / total:.0f}%) fired on a fall in error rate")
    out = ROOT / "results" / "grid" / "alarm_direction.json"
    out.write_text(json.dumps({"cells": report, "replay_mismatches": mismatches}, indent=2), encoding="utf-8")

    if mismatches:
        print(f"\nREPLAY INVALID: {len(mismatches)} alarm counts differ from the grid")
        for m in mismatches[:10]:
            print("  ", m)
        return 1
    print("replay check: every NeverAdapt alarm count reproduced exactly")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
