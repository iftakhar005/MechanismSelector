"""Controlled test: are small-window models cheaper to serve because of fewer rows, or class mix?

DIAGNOSTIC for the inference audit. Phase 9.5 and experiments/inference_audit.py find that
tree models rebuilt on <=1,000-row windows visit fewer nodes per prediction than the initial
model. On the real streams two explanations are confounded: the window has fewer rows, and a
drifting window has a different (often narrower) class mix. This separates them on stationary
synthetic data, where nothing drifts:

A. Size only: train on n in {500, 1000, 2000, 4000, 8000} i.i.d. rows of one concept.
B. Class mix only: n = 1000 rows of a 7-class concept, with all 7 classes present vs only the
   two most frequent classes present (rows of the other classes removed and refilled from the
   two).

Inference units are the study's own measure (footprint.measure_footprint), on a fixed 1,000-row
probe from the same concept. Models are the grid's own factories (mechanisms.make_spec).

Output: results/analysis/window_size_control.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footprint import measure_footprint  # noqa: E402
from mechanisms import anchor_missing_classes, make_spec  # noqa: E402

SIZES = (500, 1000, 2000, 4000, 8000)
SEEDS = range(5)
MODELS = ("xgb", "rf")


def draw(gen, n):
    names, X, y = None, [], []
    for x, label in gen.take(n):
        names = names or list(x)
        X.append([float(x[k]) for k in names])
        y.append(int(label))
    return np.asarray(X), np.asarray(y, dtype=np.int64)


def units(model_name, classes, X, y, probe, seed):
    spec = make_spec(model_name, classes)
    Xa, ya, kw = anchor_missing_classes(X, y, classes)
    model = spec.factory(seed).fit(Xa, ya, **kw)
    return measure_footprint(model, probe, timing_repeats=1).inference_units


def main() -> None:
    warnings.filterwarnings("ignore")
    from river.datasets import synth

    size_rows = []
    for stream in ("sea", "agrawal"):
        for seed in SEEDS:
            gen = (synth.SEA(variant=0, noise=0.1, seed=seed) if stream == "sea"
                   else synth.Agrawal(classification_function=0, perturbation=0.05, seed=seed))
            X, y = draw(gen, max(SIZES) + 1000)
            probe, classes = X[-1000:], np.unique(y)
            for m in MODELS:
                for n in SIZES:
                    size_rows.append({"stream": stream, "seed": seed, "model": m, "n_train": n,
                                      "units": units(m, classes, X[:n], y[:n], probe, seed)})

    mix_rows = []
    for seed in SEEDS:
        X, y = draw(synth.RandomRBF(seed_model=seed, seed_sample=seed, n_classes=7, n_features=10), 12_000)
        probe, classes = X[-1000:], np.arange(7)
        top2 = np.argsort(np.bincount(y[:-1000], minlength=7))[-2:]
        keep = np.flatnonzero(np.isin(y[:-1000], top2))[:1000]
        for m in MODELS:
            mix_rows.append({"seed": seed, "model": m,
                             "units_all_classes": units(m, classes, X[:1000], y[:1000], probe, seed),
                             "units_two_classes": units(m, classes, X[keep], y[keep], probe, seed)})

    summary = {}
    for m in MODELS:
        for stream in ("sea", "agrawal"):
            per_n = {n: float(np.mean([r["units"] for r in size_rows
                                       if r["model"] == m and r["stream"] == stream and r["n_train"] == n]))
                     for n in SIZES}
            summary[f"size/{m}/{stream}"] = {**per_n, "ratio_1000_vs_8000": per_n[1000] / per_n[8000]}
        a = [r["units_all_classes"] for r in mix_rows if r["model"] == m]
        b = [r["units_two_classes"] for r in mix_rows if r["model"] == m]
        summary[f"mix/{m}/rbf7"] = {"all_classes": float(np.mean(a)), "two_classes": float(np.mean(b)),
                                    "ratio_two_vs_all": float(np.mean(b) / np.mean(a))}

    out = ROOT / "results/analysis/window_size_control.json"
    out.write_text(json.dumps({"label": "diagnostic", "summary": summary,
                               "size_runs": size_rows, "mix_runs": mix_rows}, indent=2), encoding="utf-8")
    for k, v in summary.items():
        print(k, {kk: round(vv, 3) for kk, vv in v.items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
