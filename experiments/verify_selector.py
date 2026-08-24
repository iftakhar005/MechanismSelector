"""Phase 4 acceptance check: a manual, single-pass run of MechanismSelector
over elec2 with XGBoost.

This is NOT the stream runner (that's Phase 6 -- no ADWIN, no baselines, no
CSV grid). It is a lightweight walk over the stream in fixed-size windows,
just to observe that all three actions are reachable on real data and to
report a SKIP/NUDGE/REBUILD tally, per PHASE4_PROMPT.md Sec. 10.

Each window is treated as if a drift alarm had just fired on it. After each
`adapt()` call, `reference_accuracy` is re-measured on the next N_REF unseen
rows, per the Sec. 5 rule -- never on the just-adapted holdout, which would be
data the model (in the REBUILD case) just trained on.

Run:  python experiments/verify_selector.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from mechanisms import NudgeMechanism, RebuildMechanism, make_spec  # noqa: E402
from selector import MechanismSelector, NUDGE, REBUILD, SKIP  # noqa: E402

warnings.filterwarnings("ignore")

INIT_ROWS = 4000
WINDOW = 1000
N_REF = 200
SEED = 0


def main() -> int:
    X, y = ds.load_stream("elec2")
    classes = np.unique(y)
    spec = make_spec("xgb", classes)

    model = spec.factory(SEED).fit(X[:INIT_ROWS], y[:INIT_ROWS])

    ref_X, ref_y = X[INIT_ROWS : INIT_ROWS + N_REF], y[INIT_ROWS : INIT_ROWS + N_REF]
    reference_accuracy = float(np.mean(model.predict(ref_X) == ref_y))
    cursor = INIT_ROWS + N_REF

    sel = MechanismSelector(
        model_factory=lambda: spec.factory(SEED),
        nudge_mechanism=NudgeMechanism(spec, seed=SEED),
        rebuild_mechanism=RebuildMechanism(spec, seed=SEED),
        seed=SEED,
    )

    counts = {SKIP: 0, NUDGE: 0, REBUILD: 0}
    degraded = 0
    rows = []

    print(f"elec2: {len(X)} rows total. Initial model trained on first {INIT_ROWS}.")
    print(f"Starting reference_accuracy = {reference_accuracy:.4f}\n")

    header = f"{'#':>3} {'window':>16} {'action':<8}{'acc_before':>11}{'acc_nudged':>11}{'acc_after':>10}{'work_units':>12}"
    print(header)
    print("-" * len(header))

    n_alarm = 0
    while cursor + WINDOW <= len(X):
        X_win, y_win = X[cursor : cursor + WINDOW], y[cursor : cursor + WINDOW]
        n_alarm += 1

        result = sel.adapt(model, X_win, y_win, reference_accuracy)
        counts[result.action] += 1
        degraded += int(result.degraded_to_rebuild)
        model = result.model

        acc_nudged_str = f"{result.acc_nudged:.4f}" if result.acc_nudged is not None else "  --  "
        print(
            f"{n_alarm:>3} {cursor:>7}:{cursor + WINDOW:<8}{result.action:<8}"
            f"{result.acc_before:>11.4f}{acc_nudged_str:>11}{result.acc_after:>10.4f}"
            f"{result.cost.work_units:>12,}"
        )
        rows.append(
            {
                "alarm": n_alarm,
                "action": result.action,
                "acc_before": result.acc_before,
                "acc_nudged": result.acc_nudged,
                "acc_after": result.acc_after,
                "floor": result.floor,
                "work_units": result.cost.work_units,
                "degraded_to_rebuild": result.degraded_to_rebuild,
            }
        )

        cursor += WINDOW

        # Sec. 5: re-measure reference_accuracy on unseen future rows only.
        if cursor + N_REF <= len(X):
            ref_X, ref_y = X[cursor : cursor + N_REF], y[cursor : cursor + N_REF]
            reference_accuracy = float(np.mean(model.predict(ref_X) == ref_y))
            cursor += N_REF
        # else: fewer than N_REF rows remain -- carry reference_accuracy forward.

    print(f"\n{n_alarm} windows processed.")
    print(f"SKIP={counts[SKIP]}  NUDGE={counts[NUDGE]}  REBUILD={counts[REBUILD]}"
          f"  (degraded_to_rebuild={degraded})")

    out = ROOT / "results" / "selector_manual_run_elec2_xgb.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"counts": counts, "degraded": degraded, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
