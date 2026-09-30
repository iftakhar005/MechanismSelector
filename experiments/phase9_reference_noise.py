"""Phase 9.1 DIAGNOSTIC — is "alarm fired above reference" just estimation noise?

DIAGNOSTIC of existing results. It reruns NeverAdapt (the model never changes,
so a replay is exact and cheap), writes only new files, and touches nothing in
`results/grid*/`.

## The worry

Our reference accuracy is measured on `n_ref` = 200 rows. At ~85% accuracy that
estimate has a standard error near 2.5 points, while the floor tolerance is 2
points. Under *no change at all*, a large share of alarms could read
"at or above reference" purely by chance, because two noisy estimates are being
compared with a raw `>=`.

## What this script does, per alarm, without any new training

1. **Rescore against larger references.** The logged reference uses the 200 rows
   after the previous adaptation. Where the stream allows it, the same slice is
   extended to 500 and 1,000 rows and the comparison is redone.
2. **Significance instead of raw comparison.** A one-sided two-proportion z-test
   asks whether the alarm-time accuracy is *significantly* above the reference
   at alpha = 0.05, rather than merely numerically above.
3. **Autocorrelation correction.** Prequential errors are not independent, so
   each window's effective sample size is n_eff = n / (1 + 2*rho1), with rho1 the
   lag-1 autocorrelation of the 0/1 error stream in that window (clipped at 0).
4. **A null model.** Under H0 the two windows are draws from one Bernoulli(p)
   with p the cell's observed accuracy. P(alarm reads at/above reference) is then
   computed exactly from the binomial convolution, per alarm, using that alarm's
   real window sizes. Pooling those gives the share of "above reference" alarms
   expected when nothing has changed.

## Caveat recorded with the results

Multi-class streams score with balanced accuracy, which is not a binomial
proportion, so the null is computed against plain accuracy. Both the balanced
(as logged) and plain observed rates are reported so the comparison is like for
like.

Output: results/analysis/phase9_reference_noise.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from mechanisms import GRID_MODELS, anchor_missing_classes, make_spec  # noqa: E402
from runner import RunConfig, default_detector, scorer_for  # noqa: E402

CFG = RunConfig(track_energy=False)
REF_SIZES = (200, 500, 1000)
ALPHA = 0.05
DETERMINISTIC = ("xgb", "gnb")


# --------------------------------------------------------------------------- statistics
def lag1_autocorr(e: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 0/1 error stream; 0.0 when undefined."""
    if len(e) < 3:
        return 0.0
    a, b = e[:-1].astype(float), e[1:].astype(float)
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -0.99, 0.99))


def n_eff(n: int, rho: float) -> float:
    """Effective sample size under lag-1 dependence. Positive rho only."""
    return n / (1.0 + 2.0 * max(rho, 0.0))


def z_above(acc_h: float, n_h: float, acc_r: float, n_r: float) -> float | None:
    """One-sided z for H1: holdout accuracy > reference accuracy."""
    if n_h < 2 or n_r < 2:
        return None
    p = (acc_h * n_h + acc_r * n_r) / (n_h + n_r)
    se = np.sqrt(p * (1 - p) * (1 / n_h + 1 / n_r))
    if se == 0:
        return None
    return float((acc_h - acc_r) / se)


@lru_cache(maxsize=None)
def null_share_above(n_h: int, n_r: int, p_bucket: int) -> float:
    """P(holdout accuracy >= reference accuracy) under one common Bernoulli(p).

    Exact: sum over the reference count of P(ref = y) * P(holdout >= y * n_h/n_r).
    `p_bucket` is p rounded to 3 decimals so the cache is effective.
    """
    p = p_bucket / 1000.0
    ys = np.arange(n_r + 1)
    p_ref = stats.binom.pmf(ys, n_r, p)
    need = np.ceil(ys * n_h / n_r - 1e-12)
    p_hold_ge = stats.binom.sf(need - 1, n_h, p)   # P(X >= need)
    return float(np.dot(p_ref, p_hold_ge))


# --------------------------------------------------------------------------- replay
def replay_cell(dataset: str, model_name: str, seed: int, detector_name: str) -> dict:
    """NeverAdapt replay: the model never changes, so one predict pass is exact."""
    cfg = RunConfig(track_energy=False, detector=detector_name)
    X, y = ds.load_stream(dataset, cap=ds.COVTYPE_DEFAULT_CAP if dataset == "covtype" else None)
    classes = np.unique(y)
    scorer, scorer_name = scorer_for(classes)
    spec = make_spec(model_name, classes)

    n = len(X)
    n_init = int(n * cfg.init_frac)
    n_init_train = int(n_init * (1 - cfg.init_holdout_frac))

    model = spec.factory(seed)
    X_fit, y_fit, kwargs = anchor_missing_classes(X[:n_init_train], y[:n_init_train], classes)
    model.fit(X_fit, y_fit, **kwargs)

    preds = model.predict(X[n_init:])                       # model never changes
    errors = (preds != y[n_init:]).astype(np.int8)
    reference = float(scorer(y[n_init_train:n_init], model.predict(X[n_init_train:n_init])))

    detector = default_detector(cfg)
    buffer: deque = deque(maxlen=cfg.buffer_size)
    events, n_alarms = [], 0
    buf_start, pending_since = n_init, None
    ref_start, ref_ready = None, True

    for row in range(n_init, n):
        i = row - n_init
        buffer.append(row)
        if not ref_ready and row - ref_start + 1 == cfg.n_ref:
            reference = float(scorer(y[ref_start:row + 1], preds[ref_start - n_init:row + 1 - n_init]))
            ref_ready = True

        detector.update(int(errors[i]))
        if detector.drift_detected:
            n_alarms += 1
            if pending_since is None:
                pending_since = row

        if pending_since is None:
            continue

        window_start = max(buf_start, row + 1 - cfg.buffer_size)
        if row + 1 - window_start < cfg.min_buffer:
            continue

        # --- serve the alarm exactly where the runner would
        rows = np.arange(window_start, row + 1)
        k = int(len(rows) * (1 - cfg.holdout_frac))
        hold = rows[k:]
        hi = hold - n_init
        acc_hold_scorer = float(scorer(y[hold], preds[hi]))
        acc_hold_plain = float(1.0 - errors[hi].mean())

        event = {
            "alarm_row": int(row),
            "n_holdout": int(len(hold)),
            "reference_carried": not ref_ready,
            "acc_hold_scorer": acc_hold_scorer,
            "acc_hold_plain": acc_hold_plain,
            "reference_scorer_200": reference,
            "rho_holdout": lag1_autocorr(errors[hi]),
        }

        # references over 200 / 500 / 1000 rows from the same slice start
        slice_start = ref_start if ref_start is not None else n_init_train
        for size in REF_SIZES:
            lo = slice_start
            hi_r = lo + size
            if ref_start is None or hi_r > row + 1:          # not enough rows before this alarm
                event[f"ref_plain_{size}"] = None
                event[f"ref_rho_{size}"] = None
                continue
            seg = errors[lo - n_init:hi_r - n_init]
            event[f"ref_plain_{size}"] = float(1.0 - seg.mean())
            event[f"ref_rho_{size}"] = lag1_autocorr(seg)
        events.append(event)

        buf_start = row + 1
        pending_since = None
        ref_start, ref_ready = row + 1, False

    return {
        "dataset": dataset, "model": model_name, "seed": seed, "detector": detector_name,
        "scorer": scorer_name, "n_alarms": n_alarms, "n_served": len(events),
        "cell_accuracy_plain": float(1.0 - errors.mean()),
        "events": events,
    }


# --------------------------------------------------------------------------- scoring
def score_events(cell: dict) -> dict:
    """Observed vs null 'at or above reference' rates, four ways."""
    p_cell = cell["cell_accuracy_plain"]
    out = {k: [0, 0] for k in ("logged_scorer", "plain_200", "plain_500", "plain_1000",
                               "sig_200", "sig_200_neff", "null_200")}
    null_probs = []

    for e in cell["events"]:
        # 1. as logged: the runner's scorer, 200-row reference
        out["logged_scorer"][1] += 1
        out["logged_scorer"][0] += int(e["acc_hold_scorer"] >= e["reference_scorer_200"])

        for size in REF_SIZES:
            ref = e[f"ref_plain_{size}"]
            if ref is None:
                continue
            key = f"plain_{size}"
            out[key][1] += 1
            out[key][0] += int(e["acc_hold_plain"] >= ref)

        ref200, rho_r = e["ref_plain_200"], e["ref_rho_200"]
        if ref200 is None:
            continue
        n_h, n_r = e["n_holdout"], 200

        # 2. significance rather than raw comparison
        z = z_above(e["acc_hold_plain"], n_h, ref200, n_r)
        out["sig_200"][1] += 1
        out["sig_200"][0] += int(z is not None and z > stats.norm.ppf(1 - ALPHA))

        # 3. the same test on effective sample sizes
        z_e = z_above(e["acc_hold_plain"], n_eff(n_h, e["rho_holdout"]),
                      ref200, n_eff(n_r, rho_r or 0.0))
        out["sig_200_neff"][1] += 1
        out["sig_200_neff"][0] += int(z_e is not None and z_e > stats.norm.ppf(1 - ALPHA))

        # 4. the null: what this alarm would show if nothing had changed
        null_probs.append(null_share_above(int(n_h), n_r, int(round(p_cell * 1000))))

    res = {k: {"above": v[0], "total": v[1], "rate": (v[0] / v[1] if v[1] else None)}
           for k, v in out.items() if k != "null_200"}
    res["null_200"] = {"expected_rate": float(np.mean(null_probs)) if null_probs else None,
                       "n": len(null_probs)}
    return res


def pool(cells: list[dict]) -> dict:
    keys = ("logged_scorer", "plain_200", "plain_500", "plain_1000", "sig_200", "sig_200_neff")
    tot = {k: [0, 0] for k in keys}
    null_w, null_n = 0.0, 0
    for c in cells:
        s = c["scored"]
        for k in keys:
            tot[k][0] += s[k]["above"]
            tot[k][1] += s[k]["total"]
        if s["null_200"]["expected_rate"] is not None:
            null_w += s["null_200"]["expected_rate"] * s["null_200"]["n"]
            null_n += s["null_200"]["n"]
    out = {k: {"above": v[0], "total": v[1], "rate": (v[0] / v[1] if v[1] else None)}
           for k, v in tot.items()}
    out["null_200"] = {"expected_rate": (null_w / null_n if null_n else None), "n": null_n}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(ds.STREAMS))
    ap.add_argument("--models", nargs="*", default=list(GRID_MODELS))
    ap.add_argument("--detectors", nargs="*", default=["adwin", "ddm"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results/analysis/phase9_reference_noise.json")
    args = ap.parse_args()

    cells = []
    for detector in args.detectors:
        for dataset in args.datasets:
            for model_name in args.models:
                seeds = [0] if model_name in DETERMINISTIC else args.seeds
                for seed in seeds:
                    cell = replay_cell(dataset, model_name, seed, detector)
                    cell["scored"] = score_events(cell)
                    s = cell["scored"]
                    fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
                    print(f"{detector:5} {dataset:20} {model_name:4} seed {seed}: "
                          f"{cell['n_alarms']:5} alarms | logged {fmt(s['logged_scorer']['rate'])}"
                          f" | sig {fmt(s['sig_200']['rate'])}"
                          f" | null {fmt(s['null_200']['expected_rate'])}", flush=True)
                    cells.append(cell)

    by_detector = {d: pool([c for c in cells if c["detector"] == d]) for d in args.detectors}
    payload = {
        "label": "diagnostic",
        "question": "is 'alarm above reference' explained by estimation noise in a 200-row reference?",
        "config": {"n_ref": CFG.n_ref, "buffer_size": CFG.buffer_size, "min_buffer": CFG.min_buffer,
                   "holdout_frac": CFG.holdout_frac, "alpha": ALPHA, "ref_sizes": list(REF_SIZES)},
        "caveat": "the null is computed on plain accuracy; multi-class cells score with balanced "
                  "accuracy, so 'logged_scorer' and 'plain_200' are both reported",
        "pooled_by_detector": by_detector,
        "per_cell": [{k: v for k, v in c.items() if k != "events"} for c in cells],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
