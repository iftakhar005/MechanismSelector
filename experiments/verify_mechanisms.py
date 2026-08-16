"""Phase 3 acceptance check + measured rebuild:nudge cost ratios.

The ratio is a measured quantity, not a constant implied by the 5% nudge rule.
Two reasons it departs from the naive 20:1 the rule suggests:

1. The selector nudges on the 80% train part but rebuilds on the full window,
   so even for tree ensembles the ratio is ~25:1, not 20:1.
2. For SGD, MLP and GaussianNB the nudge is not "5% of capacity" at all -- it
   is one epoch, or one statistics update. Those ratios are set by convergence
   behaviour and vary by dataset.

Run:  python experiments/verify_mechanisms.py
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
from mechanisms import (  # noqa: E402
    GRID_MODELS,
    MechanismUnavailable,
    NudgeMechanism,
    RebuildMechanism,
    make_spec,
    nudge_size,
)

warnings.filterwarnings("ignore")

WINDOW = 1000
HOLDOUT_FRAC = 0.2
INIT_ROWS = 4000
STREAMS = ("elec2", "insects_abrupt", "covtype")


def measure_ratio(stream: str, model_name: str, seed: int = 0) -> dict:
    X, y = ds.load_stream(stream)
    classes = np.unique(y)
    spec = make_spec(model_name, classes)

    # Initial model, as the runner will build it.
    X_init, y_init = X[:INIT_ROWS], y[:INIT_ROWS]
    model = spec.factory(seed).fit(X_init, y_init)

    # A drift window, split by time exactly as the selector will split it.
    X_win, y_win = X[INIT_ROWS : INIT_ROWS + WINDOW], y[INIT_ROWS : INIT_ROWS + WINDOW]
    split = int(len(X_win) * (1 - HOLDOUT_FRAC))
    X_train, y_train = X_win[:split], y_win[:split]

    rng = np.random.default_rng(seed)

    _, nudge_cost = NudgeMechanism(spec, seed=seed).apply(model, X_train, y_train, rng)
    _, rebuild_cost = RebuildMechanism(spec, seed=seed).apply(model, X_win, y_win, rng)

    ratio = (
        rebuild_cost.work_units / nudge_cost.work_units
        if nudge_cost.work_units
        else float("nan")
    )
    return {
        "stream": stream,
        "model": model_name,
        "nudge_passes": nudge_cost.n_passes,
        "nudge_rows": nudge_cost.n_rows_processed,
        "nudge_work_units": nudge_cost.work_units,
        "rebuild_passes": rebuild_cost.n_passes,
        "rebuild_rows": rebuild_cost.n_rows_processed,
        "rebuild_work_units": rebuild_cost.work_units,
        "ratio": round(ratio, 2),
        "nudge_pct_of_rebuild": round(100 * nudge_cost.work_units / rebuild_cost.work_units, 2),
        "pass_source": nudge_cost.pass_source,
        "nudge_capacity_added": nudge_size(spec),
    }


def check_unsupported() -> list[dict]:
    rows = []
    X = np.random.default_rng(0).normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    for name in ("svc", "knn"):
        spec = make_spec(name, np.unique(y))
        model = spec.factory(0).fit(X, y)
        try:
            NudgeMechanism(spec).apply(model, X, y, np.random.default_rng(0))
        except MechanismUnavailable as exc:
            rows.append({"model": name, "raised": "MechanismUnavailable", "msg": str(exc)})
        else:
            rows.append({"model": name, "raised": "NOTHING -- GATE FAILURE"})
    return rows


def main() -> int:
    results = []
    print(f"Window {WINDOW} rows; nudge trains on {int(WINDOW * (1 - HOLDOUT_FRAC))} "
          f"(80%), rebuild on {WINDOW} (100%).\n")

    header = f"{'stream':<16}{'model':<7}{'nudge WU':>12}{'rebuild WU':>13}{'ratio':>9}{'nudge %':>10}"
    for stream in STREAMS:
        print(header)
        print("-" * len(header))
        for model_name in GRID_MODELS:
            row = measure_ratio(stream, model_name)
            results.append(row)
            print(
                f"{row['stream']:<16}{row['model']:<7}"
                f"{row['nudge_work_units']:>12,}{row['rebuild_work_units']:>13,}"
                f"{row['ratio']:>8.1f}x{row['nudge_pct_of_rebuild']:>9.1f}%"
            )
        print()

    unsupported = check_unsupported()
    print("Unsupported families:")
    for row in unsupported:
        print(f"  {row['model']:<5} -> {row['raised']}")

    out = ROOT / "results" / "mechanism_ratios.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ratios": results, "unsupported": unsupported}, indent=2),
                   encoding="utf-8")
    print(f"\nWritten to {out}")

    print("\n--- PHASE 3 GATE ---")
    failures = []
    for row in results:
        if row["model"] == "gnb":
            continue  # documented negative control
        if row["ratio"] <= 1.0:
            failures.append(f"{row['stream']}/{row['model']} ratio {row['ratio']}")
    if any(r["raised"] != "MechanismUnavailable" for r in unsupported):
        failures.append("unsupported models did not raise")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print("  PASS: nudge cheaper than rebuild for all supported models "
          "(gnb parity by design); unsupported families raise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
