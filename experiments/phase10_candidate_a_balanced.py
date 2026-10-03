"""Candidate A in balanced accuracy as well as plain.

Phase 10 quoted the DEFER-perfect accuracy gain (+3.30 covtype, +2.72 elec2) as
plain accuracy over post-alarm rows, pooled over families. Multi-class streams
are scored with balanced accuracy everywhere else, so this recomputes the same
comparison in both metrics.

At every alarm on the selector's real trajectory it collects, over the horizon
(rows until the next alarm, capped at 1,000):

- the predictions of the model the selector actually used, and
- the predictions of the incoming model, which is what DEFER keeps.

"DEFER-perfect" defers wherever deferring costs no extra errors over that
horizon, exactly as in the Phase 10 report.

Output: results/analysis/phase10_candidate_a_balanced.json
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from mechanisms import NudgeMechanism, RebuildMechanism, anchor_missing_classes, make_spec  # noqa: E402
from runner import RunConfig, default_detector, scorer_for  # noqa: E402
from selector import MechanismSelector  # noqa: E402

CFG = RunConfig(track_energy=False)
MAX_H = 1000
DETERMINISTIC = ("xgb", "gnb")


def run_cell(dataset, model_name, seed):
    X, y = ds.load_stream(dataset, cap=ds.COVTYPE_DEFAULT_CAP if dataset == "covtype" else None)
    classes = np.unique(y)
    scorer, _ = scorer_for(classes)
    spec = make_spec(model_name, classes)
    n = len(X)
    n_init = int(n * CFG.init_frac)
    n_init_train = int(n_init * (1 - CFG.init_holdout_frac))

    model = spec.factory(seed)
    Xf, yf, kw = anchor_missing_classes(X[:n_init_train], y[:n_init_train], classes)
    model.fit(Xf, yf, **kw)
    reference = float(scorer(y[n_init_train:n_init], model.predict(X[n_init_train:n_init])))

    policy = MechanismSelector(NudgeMechanism(spec, seed=seed), RebuildMechanism(spec, seed=seed),
                               holdout_frac=CFG.holdout_frac, floor_drop=CFG.floor_drop,
                               scorer=scorer, seed=seed)
    detector = default_detector(CFG)
    preds = np.empty(n - n_init, dtype=y.dtype)
    buffer: deque = deque(maxlen=CFG.buffer_size)
    buf_start, pending_since = n_init, None
    ref_start, ref_ready = None, True
    pend = None
    segments = []
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
            incoming = model
            result = policy.adapt(model, X_win, y_win, reference)
            lo, hi = r + 1, min(r + 1 + MAX_H, n)

            if pend is not None:                       # close the previous alarm at its real horizon
                h = min(r - pend["row"], MAX_H, len(pend["sel"]))
                segments.append((np.asarray(pend["true"][:h]), np.asarray(pend["sel"][:h]),
                                 np.asarray(pend["def"][:h])))
            pend = {"row": r, "true": y[lo:hi],
                    "sel": result.model.predict(X[lo:hi]) if hi > lo else np.empty(0, y.dtype),
                    "def": incoming.predict(X[lo:hi]) if hi > lo else np.empty(0, y.dtype)}

            model = result.model
            buffer.clear()
            buf_start = r + 1
            pending_since = None
            ref_start, ref_ready = r + 1, False
            adapted_at = r
            break
        row = adapted_at + 1 if adapted_at is not None else end

    if pend is not None:
        segments.append((np.asarray(pend["true"]), np.asarray(pend["sel"]), np.asarray(pend["def"])))
    return segments


def main() -> None:
    out = {}
    for dataset in ("covtype", "elec2"):
        segments = []
        for model_name in ("xgb", "rf", "sgd", "gnb"):
            for seed in ([0] if model_name in DETERMINISTIC else [0, 1, 2, 3, 4]):
                segs = run_cell(dataset, model_name, seed)
                segments.extend(segs)
                print(f"  {dataset:8} {model_name:4} seed {seed}: {sum(len(s[0]) for s in segs):7,}"
                      f" horizon rows in {len(segs)} alarms", flush=True)
        # DEFER-perfect, decided PER ALARM exactly as in Phase 10: defer when deferring
        # costs no extra errors over that alarm's horizon.
        T = np.concatenate([s[0] for s in segments])
        S = np.concatenate([s[1] for s in segments])
        P = np.concatenate([(d if (d != t).sum() <= (s != t).sum() else s)
                            for t, s, d in segments])
        out[dataset] = {
            "rows": int(len(T)),
            "selector_plain": float(accuracy_score(T, S)),
            "defer_perfect_plain": float(accuracy_score(T, P)),
            "selector_balanced": float(balanced_accuracy_score(T, S)),
            "defer_perfect_balanced": float(balanced_accuracy_score(T, P)),
        }
        o = out[dataset]
        o["delta_plain_pp"] = (o["defer_perfect_plain"] - o["selector_plain"]) * 100
        o["delta_balanced_pp"] = (o["defer_perfect_balanced"] - o["selector_balanced"]) * 100
        print(f"{dataset}: plain {o['selector_plain']:.4f} -> {o['defer_perfect_plain']:.4f} "
              f"({o['delta_plain_pp']:+.2f} pp) | balanced {o['selector_balanced']:.4f} -> "
              f"{o['defer_perfect_balanced']:.4f} ({o['delta_balanced_pp']:+.2f} pp)", flush=True)

    payload = {"label": "declared follow-up",
               "note": "post-alarm horizon rows, pooled over families; both metrics reported",
               "per_stream": out}
    p = ROOT / "results/analysis/phase10_candidate_a_balanced.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
