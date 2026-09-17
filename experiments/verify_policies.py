"""Phase 5 acceptance check: all five policies end to end on elec2 with XGBoost.

Uses the same lightweight walk as `verify_selector.py` -- fixed 1,000-row windows,
each treated as if a drift alarm had just fired, with `reference_accuracy`
re-measured on the next 200 unseen rows after each alarm. Every policy starts
from the same initial model and sees the identical sequence of windows, so their
totals are directly comparable.

This is NOT the Phase 6 runner: no ADWIN, no prequential accuracy, one seed, one
stream. The accuracy reported here is the mean over the 200-row reference slices
-- out-of-sample, but a proxy, not the prequential accuracy Phase 6 will report.

Cross-check: the selector's per-window actions must match
`results/selector_manual_run_elec2_xgb.json` from Phase 4 exactly, which confirms
this harness walks the stream the same way.

Run:  python experiments/verify_policies.py
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
from mechanisms import make_spec  # noqa: E402
from policies import POLICY_KEYS, make_policy  # noqa: E402
from selector import NUDGE, REBUILD, SKIP  # noqa: E402

warnings.filterwarnings("ignore")

INIT_ROWS = 4000
WINDOW = 1000
N_REF = 200
SEED = 0


def run_policy(key: str, X: np.ndarray, y: np.ndarray, spec) -> dict:
    model = spec.factory(SEED).fit(X[:INIT_ROWS], y[:INIT_ROWS])
    ref_X, ref_y = X[INIT_ROWS : INIT_ROWS + N_REF], y[INIT_ROWS : INIT_ROWS + N_REF]
    reference_accuracy = float(np.mean(model.predict(ref_X) == ref_y))
    cursor = INIT_ROWS + N_REF

    policy = make_policy(key, spec, seed=SEED)
    counts = {SKIP: 0, NUDGE: 0, REBUILD: 0}
    actions, ref_accs = [], []
    total_wu = 0

    while cursor + WINDOW <= len(X):
        X_win, y_win = X[cursor : cursor + WINDOW], y[cursor : cursor + WINDOW]
        result = policy.adapt(model, X_win, y_win, reference_accuracy)
        model = result.model
        counts[result.action] += 1
        actions.append(result.action)
        total_wu += result.cost.work_units
        cursor += WINDOW

        if cursor + N_REF <= len(X):
            ref_X, ref_y = X[cursor : cursor + N_REF], y[cursor : cursor + N_REF]
            reference_accuracy = float(np.mean(model.predict(ref_X) == ref_y))
            ref_accs.append(reference_accuracy)
            cursor += N_REF

    return {
        "policy": policy.name,
        "key": key,
        "n_alarms": len(actions),
        "counts": counts,
        "actions": actions,
        "total_work_units": total_wu,
        "mean_reference_accuracy": float(np.mean(ref_accs)),
        "final_reference_accuracy": ref_accs[-1],
    }


def check_definitions(runs: dict) -> list[str]:
    """Each baseline must have behaved exactly as defined on real data."""
    problems = []
    n = runs["never_adapt"]["n_alarms"]

    if runs["never_adapt"]["counts"][SKIP] != n or runs["never_adapt"]["total_work_units"] != 0:
        problems.append("NeverAdapt did something other than SKIP at zero cost")
    if runs["always_rebuild"]["counts"][REBUILD] != n:
        problems.append("AlwaysRebuild did not rebuild on every alarm")
    if runs["always_nudge"]["counts"][NUDGE] != n:
        problems.append("AlwaysNudge did not nudge on every alarm")

    expected = [REBUILD if (i + 1) % 3 == 0 else NUDGE for i in range(n)]
    if runs["fixed_schedule"]["actions"] != expected:
        problems.append("FixedSchedule(k=3) did not follow NUDGE, NUDGE, REBUILD")

    if any(r["n_alarms"] != n for r in runs.values()):
        problems.append("policies did not all see the same number of alarms")
    return problems


def check_against_phase4(selector_actions: list[str]) -> str | None:
    path = ROOT / "results" / "selector_manual_run_elec2_xgb.json"
    if not path.exists():
        return "Phase 4 run not found; cross-check skipped (run verify_selector.py)"
    phase4 = [row["action"] for row in json.loads(path.read_text(encoding="utf-8"))["rows"]]
    if phase4 != selector_actions:
        return "MISMATCH: selector actions differ from the Phase 4 run"
    return None


def main() -> int:
    X, y = ds.load_stream("elec2")
    spec = make_spec("xgb", np.unique(y))

    runs = {}
    for key in POLICY_KEYS:
        runs[key] = run_policy(key, X, y, spec)

    baseline = runs["always_rebuild"]["total_work_units"]
    print(f"elec2 + XGBoost, {runs['never_adapt']['n_alarms']} alarms per policy, seed {SEED}\n")
    header = (f"{'policy':<22}{'SKIP':>6}{'NUDGE':>7}{'REBUILD':>9}"
              f"{'work units':>13}{'% rebuild':>11}{'mean ref acc':>14}")
    print(header)
    print("-" * len(header))
    for key in POLICY_KEYS:
        r = runs[key]
        pct = 100 * r["total_work_units"] / baseline
        print(f"{r['policy']:<22}{r['counts'][SKIP]:>6}{r['counts'][NUDGE]:>7}"
              f"{r['counts'][REBUILD]:>9}{r['total_work_units']:>13,}{pct:>10.1f}%"
              f"{r['mean_reference_accuracy']:>14.4f}")

    out = ROOT / "results" / "policies_manual_run_elec2_xgb.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")

    print("\n--- PHASE 5 GATE ---")
    problems = check_definitions(runs)
    cross = check_against_phase4(runs["mechanism_selector"]["actions"])
    if cross and cross.startswith("MISMATCH"):
        problems.append(cross)
    elif cross:
        print(f"  note: {cross}")
    else:
        print("  cross-check: selector actions match the Phase 4 run window for window")

    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
        return 1
    print("  PASS: all five policies ran end to end on elec2 + XGBoost and each "
          "behaved as defined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
