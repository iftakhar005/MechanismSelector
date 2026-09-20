"""Synthetic extreme: how much of the old concept does ONE nudge destroy?

"Nudge" implies a small adjustment to an existing model. This script asks, on a
deliberately extreme synthetic case, whether that is what each family's cheapest
incremental update actually does.

Setup: concept A is `y = 1[x0 > 0]`; concept B is `y = 1[x1 > 0]` -- a complete
relabelling, no gradual shift. A model is fitted on 2,000 rows of A, then given
ONE nudge on `n_new` rows of B. It is then scored on held-out test sets for both
concepts, and compared with a model fitted from scratch on the same B rows.

Retention = accuracy on the A test set after the nudge. A gentle update keeps it
high; an update indistinguishable from replacement drops it to the from-scratch
model's level.

This is a synthetic worst case, not a measurement on the benchmark streams: an
abrupt total relabelling is the strongest possible pressure to overwrite. It
bounds the behaviour, it does not describe elec2, covtype or insects.

Output: results/analysis/nudge_overwrite_demo.json
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechanisms import GRID_MODELS, NudgeMechanism, anchor_missing_classes, make_spec  # noqa: E402

N_A = 2000
N_TEST = 1000
ROW_COUNTS = (100, 200, 400, 800)
CLASSES = np.array([0, 1])


def concepts(rng):
    def make(n, concept):
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] > 0).astype(int) if concept == "A" else (X[:, 1] > 0).astype(int)
        return X, y
    return make


def fit(spec, X, y, seed):
    model = spec.factory(seed)
    X_fit, y_fit, kwargs = anchor_missing_classes(X, y, spec.classes)
    model.fit(X_fit, y_fit, **kwargs)
    return model


def main() -> None:
    rng = np.random.default_rng(0)
    make = concepts(rng)
    XA, yA = make(N_A, "A")
    XA_test, yA_test = make(N_TEST, "A")
    XB_test, yB_test = make(N_TEST, "B")

    results = {}
    for name in GRID_MODELS:
        spec = make_spec(name, CLASSES)
        base = fit(spec, XA, yA, seed=0)
        rows = {
            "baseline_on_A": float(accuracy_score(yA_test, base.predict(XA_test))),
            "baseline_on_B": float(accuracy_score(yB_test, base.predict(XB_test))),
            "by_rows": {},
        }
        for n_new in ROW_COUNTS:
            XB, yB = make(n_new, "B")
            nudged, cost = NudgeMechanism(spec, seed=0).apply(copy.deepcopy(base), XB, yB, None)
            fresh = fit(spec, XB, yB, seed=0)
            rows["by_rows"][str(n_new)] = {
                "nudged_on_A": float(accuracy_score(yA_test, nudged.predict(XA_test))),
                "nudged_on_B": float(accuracy_score(yB_test, nudged.predict(XB_test))),
                "from_scratch_on_A": float(accuracy_score(yA_test, fresh.predict(XA_test))),
                "from_scratch_on_B": float(accuracy_score(yB_test, fresh.predict(XB_test))),
                "nudge_work_units": int(cost.work_units),
                "nudge_passes": int(cost.n_passes),
            }
        results[name] = rows

    out = ROOT / "results/analysis/nudge_overwrite_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "design": "fit on 2,000 rows of concept A; one nudge on n rows of concept B; "
                  "score on held-out A and B test sets; compare with a from-scratch fit on the same B rows",
        "caveat": "synthetic worst case (total relabelling), not a measurement on the benchmark streams",
        "results": results,
    }, indent=2), encoding="utf-8")

    print(f"{'model':6}{'rows':>6}{'A after nudge':>15}{'A from scratch':>16}{'B after nudge':>15}{'passes':>8}")
    for name, r in results.items():
        for n_new, v in r["by_rows"].items():
            print(f"{name:6}{n_new:>6}{v['nudged_on_A']:>15.3f}{v['from_scratch_on_A']:>16.3f}"
                  f"{v['nudged_on_B']:>15.3f}{v['nudge_passes']:>8}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
