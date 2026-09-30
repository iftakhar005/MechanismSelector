"""Phase 9.2 DIAGNOSTIC — what is DDM actually comparing against?

DIAGNOSTIC of existing results. NeverAdapt + DDM only, so the model never
changes and the replay is exact and cheap. Writes only new files.

## What the river 0.25.0 source says (quoted in the report, not paraphrased)

`river/drift/binary/ddm.py`:

- `update()` opens with `if self.drift_detected: self._reset()` (lines 122-124),
  so the reset happens on the update AFTER a drift is signalled, not at the
  signal itself.
- `_reset()` clears `_p = stats.Mean()`, `_p_min = None`, `_s_min = None`,
  `_ps_min = inf` (lines 109-120).
- `_p` is a **cumulative mean over every sample since the last reset**, not a
  sliding window. `s_i = sqrt(p_i * (1 - p_i) / n)` with `n = self._p.n`
  (lines 129-132).
- Statistics are only evaluated once `n > warm_start` (= 30) (line 134), and
  there is no other lockout: DDM can re-arm 31 rows after a reset.
- `p_min`/`s_min` track the running minimum of `p_i + s_i` (lines 135-138), and
  drift fires when `p_i + s_i > p_min + 3 * s_min` (lines 145-146).

So the reference DDM compares against is a running minimum of a cumulative
mean, taken since its own last reset -- not the reference accuracy our policy
uses, and not a recent window.

## What this script measures

At every alarm it logs `p_i`, `s_i`, `p_min`, `s_min`, rows since reset
(`_p.n`), and our own reference accuracy, then compares the distribution of
rows-since-reset against a stationary simulation: an i.i.d. Bernoulli stream
with the same error rate and the same length, run through a fresh DDM. Any
alarm in that simulation is false by construction.

Output: results/analysis/phase9_ddm_internals.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from mechanisms import GRID_MODELS, anchor_missing_classes, make_spec  # noqa: E402
from runner import RunConfig, default_detector, scorer_for  # noqa: E402

CFG = RunConfig(track_energy=False, detector="ddm")
DETERMINISTIC = ("xgb", "gnb")
BINS = [0, 50, 100, 200, 400, 800, 1600, 3200, np.inf]


def snapshot(detector) -> dict:
    """DDM's internal state, read at the alarm (the reset happens next update)."""
    return {
        "p_i": float(detector._p.get()),
        "n_since_reset": int(detector._p.n),
        "p_min": None if detector._p_min is None else float(detector._p_min),
        "s_min": None if detector._s_min is None else float(detector._s_min),
    }


def stationary_null(p_err: float, n_rows: int, seed: int, reps: int = 3) -> dict:
    """Run i.i.d. Bernoulli(p_err) through a fresh DDM. Every alarm is false."""
    rng = np.random.default_rng(seed)
    counts, gaps = [], []
    for _ in range(reps):
        detector = default_detector(CFG)
        stream = rng.random(n_rows) < p_err
        n_alarms = 0
        for x in stream:
            detector.update(int(x))
            if detector.drift_detected:
                n_alarms += 1
                gaps.append(int(detector._p.n))
        counts.append(n_alarms)
    return {
        "alarms_mean": float(np.mean(counts)),
        "alarms_per_10k": float(np.mean(counts) / n_rows * 10000),
        "median_rows_since_reset": float(np.median(gaps)) if gaps else None,
        "hist_rows_since_reset": np.histogram(gaps, bins=BINS)[0].tolist() if gaps else None,
        "reps": reps,
    }


def replay_cell(dataset: str, model_name: str, seed: int) -> dict:
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

    preds = model.predict(X[n_init:])
    errors = (preds != y[n_init:]).astype(np.int8)
    reference = float(scorer(y[n_init_train:n_init], model.predict(X[n_init_train:n_init])))

    detector = default_detector(CFG)
    buffer: deque = deque(maxlen=CFG.buffer_size)
    alarms, served = [], []
    buf_start, pending_since = n_init, None
    ref_start, ref_ready = None, True

    for row in range(n_init, n):
        i = row - n_init
        buffer.append(row)
        if not ref_ready and row - ref_start + 1 == CFG.n_ref:
            reference = float(scorer(y[ref_start:row + 1], preds[ref_start - n_init:row + 1 - n_init]))
            ref_ready = True

        detector.update(int(errors[i]))
        if detector.drift_detected:
            snap = snapshot(detector)
            snap.update({"row": int(row), "reference_accuracy": reference,
                         "reference_error": 1.0 - reference})
            alarms.append(snap)
            if pending_since is None:
                pending_since = row

        if pending_since is None:
            continue
        window_start = max(buf_start, row + 1 - CFG.buffer_size)
        if row + 1 - window_start < CFG.min_buffer:
            continue

        rows = np.arange(window_start, row + 1)
        hold = rows[int(len(rows) * (1 - CFG.holdout_frac)):]
        served.append({"row": int(row),
                       "acc_hold": float(scorer(y[hold], preds[hold - n_init])),
                       "reference_accuracy": reference})
        buf_start = row + 1
        pending_since = None
        ref_start, ref_ready = row + 1, False

    gaps = [a["n_since_reset"] for a in alarms]
    p_err = float(errors.mean())
    below = [a for a in alarms if a["p_min"] is not None and a["p_min"] < a["reference_error"]]
    return {
        "dataset": dataset, "model": model_name, "seed": seed,
        "n_alarms": len(alarms), "n_served": len(served),
        "stream_error_rate": p_err,
        "median_rows_since_reset": float(np.median(gaps)) if gaps else None,
        "hist_rows_since_reset": np.histogram(gaps, bins=BINS)[0].tolist() if gaps else None,
        "share_alarms_within_100_rows_of_reset": float(np.mean(np.array(gaps) <= 100)) if gaps else None,
        "p_min_below_reference_error": {
            "share": len(below) / len(alarms) if alarms else None,
            "median_gap_pp": float(np.median([100 * (a["reference_error"] - a["p_min"]) for a in below]))
            if below else None,
        },
        "median_p_i_minus_p_min_pp": float(np.median(
            [100 * (a["p_i"] - a["p_min"]) for a in alarms if a["p_min"] is not None])) if alarms else None,
        "null": stationary_null(p_err, n - n_init, seed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(ds.STREAMS))
    ap.add_argument("--models", nargs="*", default=list(GRID_MODELS))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--out", default="results/analysis/phase9_ddm_internals.json")
    args = ap.parse_args()

    cells = []
    for dataset in args.datasets:
        for model_name in args.models:
            for seed in ([0] if model_name in DETERMINISTIC else args.seeds):
                c = replay_cell(dataset, model_name, seed)
                pct = c['p_min_below_reference_error']['share']
                print(f"{dataset:20} {model_name:4} seed {seed}: {c['n_alarms']:5} alarms "
                      f"(null {c['null']['alarms_mean']:.0f}) | median rows since reset "
                      f"{c['median_rows_since_reset']} (null {c['null']['median_rows_since_reset']}) "
                      f"| p_min below our reference error in "
                      f"{'n/a' if pct is None else format(pct, '.0%')}", flush=True)
                cells.append(c)

    tot_obs = sum(c["n_alarms"] for c in cells)
    tot_null = sum(c["null"]["alarms_mean"] for c in cells)
    payload = {
        "label": "diagnostic",
        "question": "what does DDM compare against, and how many of its alarms survive a stationary null?",
        "source_facts": {
            "file": "river/drift/binary/ddm.py (river 0.25.0)",
            "reset_is_lazy": "update() line 122-124: `if self.drift_detected: self._reset()`",
            "state_cleared": "_reset() lines 109-120: _p (cumulative Mean), _p_min, _s_min, _ps_min",
            "cumulative_not_windowed": "p_i = self._p.get() over all samples since reset (line 129)",
            "warm_start": "statistics evaluated only when n > 30 (line 134); no other lockout",
            "rule": "drift when p_i + s_i > p_min + 3 * s_min (lines 145-146)",
        },
        "bins_rows_since_reset": [0, 50, 100, 200, 400, 800, 1600, 3200, "inf"],
        "pooled": {"observed_alarms": tot_obs, "stationary_null_alarms": tot_null,
                   "ratio_observed_to_null": tot_obs / tot_null if tot_null else None},
        "per_cell": cells,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
