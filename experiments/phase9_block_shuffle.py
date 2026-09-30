"""Phase 9.4 DIAGNOSTIC — is label autocorrelation manufacturing the "changes"?

DIAGNOSTIC of existing results. NeverAdapt only, so the model never changes and
the replay is exact and cheap. Writes only new files.

Elec2 and Covertype are known to carry strong short-range label dependence: the
next label is very often the previous one. A detector watching a 0/1 error
stream can read that dependence as a change in error rate even when the
underlying concept is unchanged.

The test: shuffle rows **within consecutive blocks of `block_size`**, which
destroys the local label structure while leaving the overall order -- and so any
real drift -- in place. If alarm counts collapse, the alarms were being driven by
local dependence rather than by drift.

Only the streamed part is shuffled; the initial training slice is left intact, so
the model under test is exactly the model from the original run and the only
thing that changes is the error stream the detector sees.

Reported per cell: alarms before and after, and the lag-1 autocorrelation of the
error stream before and after (the quantity the shuffle is meant to remove).

Output: results/analysis/phase9_block_shuffle.json
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
from mechanisms import GRID_MODELS, anchor_missing_classes, make_spec  # noqa: E402
from runner import RunConfig, default_detector  # noqa: E402

DETERMINISTIC = ("xgb", "gnb")


def lag1(e: np.ndarray) -> float:
    a, b = e[:-1].astype(float), e[1:].astype(float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def block_shuffle(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Permutation of range(n) that only moves rows within blocks of `block`."""
    idx = np.arange(n)
    for start in range(0, n, block):
        end = min(start + block, n)
        rng.shuffle(idx[start:end])
    return idx


def count_alarms(errors: np.ndarray, detector_name: str) -> int:
    cfg = RunConfig(track_energy=False, detector=detector_name)
    detector = default_detector(cfg)
    n = 0
    for e in errors:
        detector.update(int(e))
        if detector.drift_detected:
            n += 1
    return n


def run_cell(dataset: str, model_name: str, seed: int, block: int, detectors: list[str]) -> dict:
    cfg = RunConfig(track_energy=False)
    X, y = ds.load_stream(dataset, cap=ds.COVTYPE_DEFAULT_CAP if dataset == "covtype" else None)
    classes = np.unique(y)
    spec = make_spec(model_name, classes)

    n = len(X)
    n_init = int(n * cfg.init_frac)
    n_init_train = int(n_init * (1 - cfg.init_holdout_frac))

    model = spec.factory(seed)
    X_fit, y_fit, kwargs = anchor_missing_classes(X[:n_init_train], y[:n_init_train], classes)
    model.fit(X_fit, y_fit, **kwargs)

    preds = model.predict(X[n_init:])
    errors = (preds != y[n_init:]).astype(np.int8)

    rng = np.random.default_rng(1000 + seed)
    perm = block_shuffle(len(errors), block, rng)
    shuffled = errors[perm]

    out = {
        "dataset": dataset, "model": model_name, "seed": seed, "block_size": block,
        "stream_rows": int(len(errors)),
        "error_rate": float(errors.mean()),
        "lag1_original": lag1(errors),
        "lag1_shuffled": lag1(shuffled),
        "alarms": {},
    }
    for det in detectors:
        before = count_alarms(errors, det)
        after = count_alarms(shuffled, det)
        out["alarms"][det] = {
            "original": before, "block_shuffled": after,
            "retained": (after / before) if before else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=["elec2", "covtype"])
    ap.add_argument("--models", nargs="*", default=list(GRID_MODELS))
    ap.add_argument("--detectors", nargs="*", default=["adwin", "ddm"])
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--out", default="results/analysis/phase9_block_shuffle.json")
    args = ap.parse_args()

    cells = []
    for dataset in args.datasets:
        for model_name in args.models:
            for seed in ([0] if model_name in DETERMINISTIC else args.seeds):
                c = run_cell(dataset, model_name, seed, args.block, args.detectors)
                msg = " · ".join(f"{d}: {c['alarms'][d]['original']} -> {c['alarms'][d]['block_shuffled']}"
                                 for d in args.detectors)
                print(f"{dataset:10} {model_name:4} seed {seed} | rho1 {c['lag1_original']:+.3f} -> "
                      f"{c['lag1_shuffled']:+.3f} | {msg}", flush=True)
                cells.append(c)

    pooled = {}
    for det in args.detectors:
        b = sum(c["alarms"][det]["original"] for c in cells)
        a = sum(c["alarms"][det]["block_shuffled"] for c in cells)
        pooled[det] = {"original": b, "block_shuffled": a, "retained": (a / b) if b else None}

    payload = {
        "label": "diagnostic",
        "question": "do the detectors' alarms survive destroying local label dependence?",
        "design": f"rows after the initial training slice permuted within blocks of {args.block}; "
                  "initial model and overall order unchanged",
        "pooled": pooled,
        "per_cell": cells,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
