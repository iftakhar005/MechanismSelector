"""The paper's core numbers, per stream and model, from the grid event logs.

1. **Nudge improvement per attempt**: `acc_nudged - acc_before` on the window
   holdout, both out-of-sample. Reported twice:
   - *unconditional*, from `AlwaysNudge`, which nudges at every alarm -- the
     distribution of what a nudge does;
   - *conditional*, from the selector, which nudges only when the model is below
     its floor -- what a nudge does when it is actually needed.
2. **Accuracy deficit at alarm time**: `reference_accuracy_used - acc_before`,
   how far below its own recent accuracy the model sits when ADWIN fires.
   Events whose reference was carried forward are excluded.
3. **Share of the deficit a nudge recovers**: median improvement over median
   deficit, reported only when the median deficit is at least
   `MIN_DEFICIT_PP` points -- below that the ratio is dominated by noise
   (a +12.9 gain over a +1.3 deficit is not a 1,000% recovery).
4. **Rebuild vs no adaptation**: mean prequential accuracy of `AlwaysRebuild`
   against `NeverAdapt`, with the rows the initial model was trained on and the
   size of the windows rebuilds actually used.

Accuracies are in each stream's scorer (plain for binary, balanced for
multi-class), in percentage points.

    python experiments/contribution_numbers.py [grid_dir]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import median, quantiles

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results" / "grid"
MIN_DEFICIT_PP = 2.0


def number(value: str) -> float | None:
    return None if value in ("", "None") else float(value)


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    q = quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return {
        "n": len(values),
        "median": median(values),
        "p25": q[0],
        "p75": q[2],
        "share_positive": sum(v > 0 for v in values) / len(values),
    }


def events_for(grid: Path, dataset: str, model: str, policy: str) -> list[dict]:
    rows = []
    for path in sorted((grid / "events").glob(f"{dataset}__{model}__{policy}__*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main(grid: Path = GRID) -> int:
    with (grid / "grid.csv").open(newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    cells = sorted({(r["dataset"], r["model"]) for r in runs})
    by_key: dict[tuple, list[dict]] = {}
    for r in runs:
        by_key.setdefault((r["dataset"], r["model"], r["policy_key"]), []).append(r)

    def accuracy(r):
        field = "mean_prequential_accuracy" if r["scorer"] == "accuracy" else "mean_prequential_balanced_accuracy"
        return 100 * float(r[field])

    report = {}
    print(f"{'cell':<26}{'nudge gain (AlwaysNudge)':>30}{'deficit at alarm':>24}{'recovered':>11}"
          f"{'Rebuild acc':>13}{'Never acc':>11}{'init rows':>11}{'rebuild window':>16}")
    for dataset, model in cells:
        nudge_events = events_for(grid, dataset, model, "always_nudge")
        selector_events = events_for(grid, dataset, model, "mechanism_selector")
        rebuild_events = events_for(grid, dataset, model, "always_rebuild")

        unconditional = [100 * (number(e["acc_nudged"]) - number(e["acc_before"]))
                         for e in nudge_events if number(e["acc_nudged"]) is not None]
        conditional = [100 * (number(e["acc_nudged"]) - number(e["acc_before"]))
                       for e in selector_events
                       if number(e["acc_nudged"]) is not None and e["degraded_to_rebuild"] == "False"]
        deficit = [100 * (number(e["reference_accuracy_used"]) - number(e["acc_before"]))
                   for e in nudge_events if e["reference_carried"] == "False"]
        windows = [int(e["buffer_rows"]) for e in rebuild_events]

        gain_u, gain_c, deficit_s = summary(unconditional), summary(conditional), summary(deficit)
        recovered = (gain_u["median"] / deficit_s["median"]
                     if gain_u.get("n") and deficit_s.get("n") and deficit_s["median"] >= MIN_DEFICIT_PP else None)
        rebuild_acc = [accuracy(r) for r in by_key.get((dataset, model, "always_rebuild"), [])]
        never_acc = [accuracy(r) for r in by_key.get((dataset, model, "never_adapt"), [])]
        init_rows = int(by_key[(dataset, model, "never_adapt")][0]["n_init_train_rows"])

        report[f"{dataset}/{model}"] = {
            "nudge_gain_unconditional": gain_u,
            "nudge_gain_selector_attempts": gain_c,
            "deficit_at_alarm": deficit_s,
            "median_share_of_deficit_recovered": recovered,
            "rebuild_accuracy": median(rebuild_acc) if rebuild_acc else None,
            "never_adapt_accuracy": median(never_acc) if never_acc else None,
            "initial_training_rows": init_rows,
            "rebuild_window_rows": summary([float(w) for w in windows]),
        }

        def fmt(s):
            return f"{s['median']:+5.1f} [{s['p25']:+5.1f},{s['p75']:+5.1f}]" if s.get("n") else "--"

        rw = report[f"{dataset}/{model}"]["rebuild_window_rows"]
        print(f"{dataset + ' ' + model:<26}{fmt(gain_u):>30}{fmt(deficit_s):>24}"
              f"{(f'{100 * recovered:.0f}%' if recovered is not None else '--'):>11}"
              f"{(median(rebuild_acc) if rebuild_acc else float('nan')):>12.1f}%"
              f"{(median(never_acc) if never_acc else float('nan')):>10.1f}%{init_rows:>11,}  "
              f"{(f'{rw['median']:.0f} [{rw['p25']:.0f}-{rw['p75']:.0f}]' if rw.get('n') else '--'):>16}")

    out = grid / "contribution_numbers.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nmedian [IQR], percentage points; 'recovered' = median gain / median deficit")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else GRID))
