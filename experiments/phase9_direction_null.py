"""Phase 9.6 DIAGNOSTIC — what does the reference-free direction measure do on noise?

DIAGNOSTIC of existing results. Writes only new files.

ADWIN is two-sided, so on a stationary stream its alarms should split roughly
evenly between "error fell" and "error rose". This script measures that split
instead of assuming it, and then asks whether the observed falling-error share
is carried by alarms that follow a real change point or by the ones that do not.

Two parts:

1. **Null direction split.** For each cell, an i.i.d. Bernoulli stream at that
   cell's own error rate and length is run through a fresh ADWIN. Every alarm is
   false by construction. Direction comes from `runner.alarm_direction`, read
   from ADWIN's own estimate before and after the update -- the same
   reference-free measure the grid used.

2. **Observed split by true/false alarm.** NeverAdapt is replayed on
   insects_abrupt (the only stream with published change points, verified in 9.3
   against Souza et al. 2020 Table 2). Each alarm is classified as a true
   detection if it lands within `tol` rows after a change point, and the
   falling-error share is reported for true and false alarms separately, at
   tolerances 500, 1,000 and 2,000.

Output: results/analysis/phase9_direction_null.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from mechanisms import GRID_MODELS, anchor_missing_classes, make_spec  # noqa: E402
from runner import RunConfig, alarm_direction, default_detector  # noqa: E402

# Souza et al. 2020, Table 2 (arXiv:2005.00113, p. 37) -- INSECTS Abrupt (balanced).
CHANGE_POINTS = (14352, 19500, 33240, 38682, 39510)
TOLERANCES = (500, 1000, 2000)
DETERMINISTIC = ("xgb", "gnb")
CFG = RunConfig(track_energy=False)


def error_stream(dataset: str, model_name: str, seed: int) -> tuple[np.ndarray, int]:
    """NeverAdapt error stream: the model never changes, so one predict pass is exact."""
    X, y = ds.load_stream(dataset, cap=ds.COVTYPE_DEFAULT_CAP if dataset == "covtype" else None)
    classes = np.unique(y)
    spec = make_spec(model_name, classes)
    n = len(X)
    n_init = int(n * CFG.init_frac)
    n_init_train = int(n_init * (1 - CFG.init_holdout_frac))
    model = spec.factory(seed)
    X_fit, y_fit, kwargs = anchor_missing_classes(X[:n_init_train], y[:n_init_train], classes)
    model.fit(X_fit, y_fit, **kwargs)
    preds = model.predict(X[n_init:])
    return (preds != y[n_init:]).astype(np.int8), n_init


def directions(stream: np.ndarray) -> list[tuple[int, str | None]]:
    """Run ADWIN over a 0/1 stream; return (index, direction) for every alarm."""
    detector = default_detector(CFG)
    out = []
    for i, e in enumerate(stream):
        before = getattr(detector, "estimation", None)
        detector.update(int(e))
        if detector.drift_detected:
            out.append((i, alarm_direction(before, getattr(detector, "estimation", None))))
    return out


def fmt(v, spec=".3f"):
    return "  n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def ar1_bernoulli(n: int, p: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Two-state Markov chain with marginal P(error)=p and lag-1 autocorrelation rho.

    A stationary null WITH dependence: no concept change anywhere, but the error
    stream is as autocorrelated as the observed one. i.i.d. nulls barely make
    ADWIN fire, so they cannot answer what direction its spurious alarms take.
    """
    rho = float(np.clip(rho, 0.0, 0.95))
    p11 = p + (1 - p) * rho
    p01 = p * (1 - rho)
    out = np.empty(n, dtype=np.int8)
    out[0] = rng.random() < p
    u = rng.random(n)
    for t in range(1, n):
        out[t] = u[t] < (p11 if out[t - 1] else p01)
    return out


def lag1(e: np.ndarray) -> float:
    a, b = e[:-1].astype(float), e[1:].astype(float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def share_falling(dirs) -> tuple[float | None, int]:
    counts = Counter(d for _, d in dirs if d is not None)
    total = sum(counts.values())
    return ((counts["error_down"] / total) if total else None, total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(ds.STREAMS))
    ap.add_argument("--models", nargs="*", default=list(GRID_MODELS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/analysis/phase9_direction_null.json")
    args = ap.parse_args()

    # ---------------- part 1: the null split
    null_cells, obs_cells = [], []
    for dataset in args.datasets:
        for model_name in args.models:
            errors, _ = error_stream(dataset, model_name, 0)
            p_err = float(errors.mean())
            obs = directions(errors)
            o_share, o_n = share_falling(obs)
            obs_cells.append({"dataset": dataset, "model": model_name,
                              "alarms": o_n, "share_falling": o_share, "error_rate": p_err})

            rng = np.random.default_rng(7)
            rho = lag1(errors)
            nulls = {}
            for kind in ("iid", "ar1"):
                reps = []
                for _ in range(args.reps):
                    sim = ((rng.random(len(errors)) < p_err).astype(np.int8) if kind == "iid"
                           else ar1_bernoulli(len(errors), p_err, rho, rng))
                    sh, n = share_falling(directions(sim))
                    reps.append({"alarms": n, "share_falling": sh})
                got = [r["share_falling"] for r in reps if r["share_falling"] is not None]
                nulls[kind] = {"alarms_mean": float(np.mean([r["alarms"] for r in reps])),
                               "share_falling_mean": float(np.mean(got)) if got else None,
                               "reps": reps}
            null_cells.append({"dataset": dataset, "model": model_name, "error_rate": p_err,
                               "lag1_observed": rho, "nulls": nulls})
            print(f"{dataset:20} {model_name:4} | observed {o_n:5} alarms, falling {fmt(o_share)}"
                  f" | iid null {nulls['iid']['alarms_mean']:6.1f} alarms, falling "
                  f"{fmt(nulls['iid']['share_falling_mean'])}"
                  f" | AR(1) null {nulls['ar1']['alarms_mean']:6.1f} alarms, falling "
                  f"{fmt(nulls['ar1']['share_falling_mean'])}", flush=True)

    # ---------------- part 2: observed split by true/false alarm on insects_abrupt
    split = {}
    for tol in TOLERANCES:
        table = Counter()
        for model_name in args.models:
            for seed in ([0] if model_name in DETERMINISTIC else [0, 1, 2, 3, 4]):
                errors, n_init = error_stream("insects_abrupt", model_name, seed)
                for i, d in directions(errors):
                    if d is None:
                        continue
                    row = i + n_init
                    kind = "true" if any(cp <= row <= cp + tol for cp in CHANGE_POINTS) else "false"
                    table[(kind, d)] += 1
        def share(kind):
            n = sum(v for (k, _), v in table.items() if k == kind)
            return (table[(kind, "error_down")] / n if n else None), n
        s_true, n_true = share("true")
        s_false, n_false = share("false")
        split[str(tol)] = {
            "true_alarms": n_true, "share_falling_given_true": s_true,
            "false_alarms": n_false, "share_falling_given_false": s_false,
            "share_of_falling_alarms_that_are_false": (
                table[("false", "error_down")] /
                (table[("false", "error_down")] + table[("true", "error_down")])
                if (table[("false", "error_down")] + table[("true", "error_down")]) else None),
            "cross_tab": {f"{k}_{d}": v for (k, d), v in sorted(table.items())},
        }
        print(f"insects_abrupt tol={tol}: falling | true {fmt(s_true)} (n={n_true}) "
              f"vs false {fmt(s_false)} (n={n_false})", flush=True)

    def null_pool(kind):
        vals = [(c["nulls"][kind]["share_falling_mean"], c["nulls"][kind]["alarms_mean"])
                for c in null_cells if c["nulls"][kind]["share_falling_mean"] is not None]
        if not vals:
            return None
        return float(np.average([v for v, _ in vals], weights=[w for _, w in vals]))
    payload = {
        "label": "diagnostic",
        "question": "is the falling-error share what a population of spurious two-sided alarms produces?",
        "direction_measure": "runner.alarm_direction, from ADWIN's own estimate before/after the "
                             "update -- reference-free",
        "null_iid_pooled_share_falling": null_pool("iid"),
        "null_ar1_pooled_share_falling": null_pool("ar1"),
        "null_iid_total_alarms": float(sum(c["nulls"]["iid"]["alarms_mean"] for c in null_cells)),
        "null_ar1_total_alarms": float(sum(c["nulls"]["ar1"]["alarms_mean"] for c in null_cells)),
        "observed_pooled_share_falling": float(np.average(
            [c["share_falling"] for c in obs_cells if c["share_falling"] is not None],
            weights=[c["alarms"] for c in obs_cells if c["share_falling"] is not None])),
        "insects_abrupt_true_false_split": split,
        "null_per_cell": null_cells,
        "observed_per_cell": obs_cells,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nnull falling share (pooled over cells with alarms): "
          f"{payload['null_pooled_share_falling']}")
    print(f"observed falling share (alarm-weighted): {payload['observed_pooled_share_falling']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
