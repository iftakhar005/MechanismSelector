"""Inference-cost audit — what is counted per prediction, and is it plausible?

DIAGNOSTIC. Writes only new files; the pre-registered grid is untouched.

## What one "inference unit" is, per family

Taken from `src/footprint.py`, not from assumption:

| family | unit | how it is counted |
|---|---|---|
| XGBoost | tree node visits | the leaf each probe row reaches, plus that leaf's depth, summed over every boosting round; exact per row |
| RandomForest | tree node visits | `decision_path` length per row summed over trees — every split comparison plus one leaf per tree; exact per row |
| SGDClassifier | parameter reads | every coefficient and intercept is read once per prediction: `n_classes x n_features + n_classes` |
| GaussianNB | parameter reads | every mean and variance and prior: `2 x n_classes x n_features + n_classes` |

Units are comparable **within** a family, never across families: a node visit and
a parameter read are different amounts of work.

## What this script measures, per dataset x family

- **initial** - the model the run starts from, trained on the real initial slice.
- **rebuilt** - a model of the same family trained on a 1,000-row window late in
  the stream, i.e. what a REBUILD actually produces.
- **control** - the same family trained on 1,000 rows taken from the *initial*
  slice. This separates "rebuilds are cheaper because the window is small" from
  "rebuilds are cheaper because the data is later". Candidate B requires it.

For each: ops per prediction on a common probe, tree count, total node count,
mean path length per tree, training rows, and accuracy on the 1,000 rows
following its own training slice.

Output: results/analysis/inference_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from footprint import measure_footprint  # noqa: E402
from mechanisms import GRID_MODELS, anchor_missing_classes, make_spec  # noqa: E402
from runner import RunConfig, scorer_for  # noqa: E402

CFG = RunConfig(track_energy=False)
PROBE_ROWS = 1000
REBUILD_WINDOW = 1000


def fit(spec, X, y, lo, hi, seed):
    model = spec.factory(seed)
    X_fit, y_fit, kwargs = anchor_missing_classes(X[lo:hi], y[lo:hi], spec.classes)
    model.fit(X_fit, y_fit, **kwargs)
    return model


def describe(model, probe, X, y, eval_lo, eval_hi, scorer, rows_trained) -> dict:
    fp = measure_footprint(model, probe, timing_repeats=1)
    n_trees = fp.size if fp.size_unit in ("rounds", "trees") else None
    return {
        "rows_trained": int(rows_trained),
        "ops_per_prediction": float(fp.inference_units),
        "unit": fp.inference_unit,
        "size": int(fp.size),
        "size_unit": fp.size_unit,
        "n_nodes": None if fp.n_nodes is None else int(fp.n_nodes),
        "mean_path_per_tree": (float(fp.inference_units) / n_trees) if n_trees else None,
        "accuracy_next_1000": (float(scorer(y[eval_lo:eval_hi], model.predict(X[eval_lo:eval_hi])))
                               if eval_hi > eval_lo else None),
    }


def run_cell(dataset: str, model_name: str, seed: int = 0) -> dict:
    X, y = ds.load_stream(dataset, cap=ds.COVTYPE_DEFAULT_CAP if dataset == "covtype" else None)
    classes = np.unique(y)
    scorer, scorer_name = scorer_for(classes)
    spec = make_spec(model_name, classes)

    n = len(X)
    n_init = int(n * CFG.init_frac)
    n_init_train = int(n_init * (1 - CFG.init_holdout_frac))
    probe = X[n_init:n_init + PROBE_ROWS]                       # one probe for all three models

    initial = fit(spec, X, y, 0, n_init_train, seed)
    reb_lo = n - REBUILD_WINDOW - PROBE_ROWS                      # a late-stream rebuild window
    rebuilt = fit(spec, X, y, reb_lo, reb_lo + REBUILD_WINDOW, seed)
    ctrl_lo = n_init_train - REBUILD_WINDOW                       # same era as the initial model
    control = fit(spec, X, y, ctrl_lo, n_init_train, seed)

    out = {
        "dataset": dataset, "model": model_name, "seed": seed, "scorer": scorer_name,
        "initial": describe(initial, probe, X, y, n_init_train, n_init, scorer, n_init_train),
        "rebuilt": describe(rebuilt, probe, X, y, reb_lo + REBUILD_WINDOW, n, scorer, REBUILD_WINDOW),
        "control_initial_era_1000_rows": describe(control, probe, X, y, n_init_train, n_init,
                                                  scorer, REBUILD_WINDOW),
    }
    out["ratio_rebuilt_over_initial"] = (out["rebuilt"]["ops_per_prediction"]
                                         / out["initial"]["ops_per_prediction"])
    out["ratio_control_over_initial"] = (out["control_initial_era_1000_rows"]["ops_per_prediction"]
                                         / out["initial"]["ops_per_prediction"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(ds.STREAMS))
    ap.add_argument("--models", nargs="*", default=list(GRID_MODELS))
    ap.add_argument("--out", default="results/analysis/inference_audit.json")
    args = ap.parse_args()

    cells = []
    for dataset in args.datasets:
        for model_name in args.models:
            c = run_cell(dataset, model_name)
            cells.append(c)
            print(f"{dataset:20} {model_name:4} | initial {c['initial']['ops_per_prediction']:8.0f} "
                  f"({c['initial']['rows_trained']:6} rows) | rebuilt "
                  f"{c['rebuilt']['ops_per_prediction']:8.0f} | control "
                  f"{c['control_initial_era_1000_rows']['ops_per_prediction']:8.0f} | "
                  f"rebuilt/initial {c['ratio_rebuilt_over_initial']:.2f} | "
                  f"control/initial {c['ratio_control_over_initial']:.2f}", flush=True)

    payload = {
        "label": "diagnostic",
        "unit_definitions": {
            "xgb": "tree node visits: leaf reached per boosting round plus its depth, summed over rounds",
            "rf": "tree node visits: decision_path length per row summed over trees",
            "sgd": "parameter reads: n_classes x n_features + n_classes",
            "gnb": "parameter reads: 2 x n_classes x n_features + n_classes",
            "comparability": "within a family only; a node visit and a parameter read differ",
        },
        "probe_rows": PROBE_ROWS,
        "rebuild_window": REBUILD_WINDOW,
        "cells": cells,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
