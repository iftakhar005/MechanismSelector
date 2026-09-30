"""Phase 9.3 DIAGNOSTIC — do alarms line up with real drift?

DIAGNOSTIC of existing results. Reads the committed event logs only; runs no
model and writes only new files.

## Ground truth, verified at source

Souza, Reis, Maletzke & Batista (2020), "Challenges in Benchmarking Stream
Learning Algorithms with Real-world Data", Data Mining and Knowledge Discovery
34:1805-1858; arXiv:2005.00113, **Table 2, page 37**, quoted verbatim:

    Abrupt (bal.)   52,848   14352; 19500; 33240; 38682; 39510

The instance count matches `results/dataset_manifest.json` for `insects_abrupt`
(52,848), so the stream we load is the one the change points refer to.

## What this measures

Each logged alarm on insects_abrupt is classified as a **true detection** if it
falls within `tol` rows after a change point, otherwise a **false alarm**.
Tolerance is reported at 500, 1,000 and 2,000 rows because the choice is
arbitrary and the answer moves with it. Alarms are then cross-tabulated against
whether the model was at or above its reference accuracy at the time
(`accuracy_deficit < 0`), which is the quantity Finding 2 is about.

Also reported: detection delay for each change point (rows from the change point
to the first alarm after it), and false alarms per 10,000 rows.

Output: results/analysis/phase9_change_points.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Souza et al. 2020, Table 2 (arXiv:2005.00113, p. 37) -- INSECTS Abrupt (balanced).
CHANGE_POINTS = (14352, 19500, 33240, 38682, 39510)
N_INSTANCES = 52_848
TOLERANCES = (500, 1000, 2000)
DETERMINISTIC = ("xgb", "gnb")
POLICIES = ("never_adapt", "always_rebuild", "fixed_schedule", "mechanism_selector", "always_nudge")


def load_events(grid: Path, policy: str) -> list[dict]:
    rows = []
    for f in sorted((grid / "events").glob(f"insects_abrupt__*__{policy}__*.csv")):
        _, model, _, seed = f.stem.split("__")
        if model in DETERMINISTIC and int(seed) != 0:
            continue
        for e in csv.DictReader(f.open()):
            if e.get("accuracy_deficit", "") == "":
                continue
            rows.append({"model": model, "seed": int(seed), "row": int(e["alarm_row"]),
                         "above": float(e["accuracy_deficit"]) < 0})
    return rows


def classify(rows: list[dict], tol: int) -> dict:
    table = defaultdict(int)
    for r in rows:
        true_det = any(cp <= r["row"] <= cp + tol for cp in CHANGE_POINTS)
        table[("true" if true_det else "false", "above" if r["above"] else "below")] += 1
    n = len(rows)
    n_true = table[("true", "above")] + table[("true", "below")]
    n_false = table[("false", "above")] + table[("false", "below")]
    return {
        "tolerance": tol,
        "n_alarms": n,
        "true_detections": n_true,
        "false_alarms": n_false,
        "share_true": n_true / n if n else None,
        "cross_tab": {f"{k[0]}_{k[1]}": v for k, v in sorted(table.items())},
        "share_above_given_true": (table[("true", "above")] / n_true) if n_true else None,
        "share_above_given_false": (table[("false", "above")] / n_false) if n_false else None,
    }


def delays(rows: list[dict]) -> dict:
    """Rows from each change point to the first alarm after it, per run."""
    per_run = defaultdict(list)
    for r in rows:
        per_run[(r["model"], r["seed"])].append(r["row"])
    out = {str(cp): [] for cp in CHANGE_POINTS}
    for _, alarm_rows in per_run.items():
        alarm_rows = sorted(alarm_rows)
        for cp in CHANGE_POINTS:
            later = [a for a in alarm_rows if a >= cp]
            if later:
                out[str(cp)].append(later[0] - cp)
    return {cp: {"median_delay_rows": float(np.median(v)) if v else None, "n_runs": len(v)}
            for cp, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/analysis/phase9_change_points.json")
    args = ap.parse_args()

    payload = {
        "label": "diagnostic",
        "ground_truth": {
            "source": "Souza et al. 2020, DMKD 34:1805-1858; arXiv:2005.00113, Table 2, p. 37",
            "quote": "Abrupt (bal.)   52,848   14352; 19500; 33240; 38682; 39510",
            "change_points": list(CHANGE_POINTS),
            "instances": N_INSTANCES,
            "verified_against": "results/dataset_manifest.json insects_abrupt n_rows = 52848",
        },
        "detectors": {},
    }

    for det, grid in (("adwin", ROOT / "results/grid"), ("ddm", ROOT / "results/grid_ddm")):
        per_policy = {}
        for policy in POLICIES:
            rows = load_events(grid, policy)
            if not rows:
                continue
            stream_rows = N_INSTANCES - int(N_INSTANCES * 0.10)
            n_runs = len({(r["model"], r["seed"]) for r in rows})
            per_policy[policy] = {
                "n_runs": n_runs,
                "by_tolerance": {str(t): classify(rows, t) for t in TOLERANCES},
                "false_alarms_per_10k_rows_tol1000": (
                    classify(rows, 1000)["false_alarms"] / n_runs / stream_rows * 10000),
                "first_alarm_delay": delays(rows),
            }
        payload["detectors"][det] = per_policy

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for det, pol in payload["detectors"].items():
        print(f"== {det.upper()}")
        for name, v in pol.items():
            c = v["by_tolerance"]["1000"]
            print(f"  {name:20} {c['n_alarms']:4} alarms | true {c['share_true']:.1%} "
                  f"| above-ref given true {c['share_above_given_true']:.1%} "
                  f"vs given false {c['share_above_given_false']:.1%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
