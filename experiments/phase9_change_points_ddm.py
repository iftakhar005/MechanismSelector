"""Item 3 - DDM alarms against the published INSECTS-abrupt change points.

The same check already reported for ADWIN (27.4% of alarms within 1,000 rows of
a published change point), now for DDM, from the committed event logs only.
Nothing is re-executed.

Scope, as declared: insects_abrupt, policy NeverAdapt (the model is never
changed, so the alarms are the detector's own behaviour and not a reaction to
our repairs), all four families, seed 0 for the deterministic families (XGBoost,
Gaussian NB) and all five seeds for Random Forest and SGD.

Ground truth: Souza et al. 2020, DMKD 34:1805-1858 (arXiv:2005.00113), Table 2,
p. 37, change points 14352; 19500; 33240; 38682; 39510 over 52,848 instances.

ADWIN is recomputed here from the same logs so the two sit side by side, and is
also reported restricted to the family x seed cells in which DDM alarmed at all,
since the two detectors do not fire in the same cells.

Writes `results/analysis/phase9_change_points_ddm.json`. [DIAGNOSTIC]
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGE_POINTS = (14352, 19500, 33240, 38682, 39510)
N_INSTANCES = 52_848
TOLERANCES = (500, 1000, 2000)
DETERMINISTIC = ("xgb", "gnb")
GRIDS = {"adwin": "grid", "ddm": "grid_ddm"}
POLICY = "never_adapt"


def load(grid_name: str) -> list[dict]:
    rows = []
    d = ROOT / "results" / grid_name / "events"
    for f in sorted(d.glob(f"insects_abrupt__*__{POLICY}__*.csv")):
        _, model, _, seed = f.stem.split("__")
        if model in DETERMINISTIC and int(seed) != 0:
            continue
        for e in csv.DictReader(f.open()):
            if e.get("accuracy_deficit", "") == "":
                continue
            rows.append({"cell": (model, int(seed)), "model": model,
                         "row": int(e["alarm_row"])})
    return rows


def shares(rows: list[dict]) -> dict:
    out = {}
    for tol in TOLERANCES:
        n_true = sum(any(cp <= r["row"] <= cp + tol for cp in CHANGE_POINTS) for r in rows)
        out[str(tol)] = {
            "tolerance": tol, "n_alarms": len(rows), "within": n_true,
            "outside": len(rows) - n_true,
            "share_within": n_true / len(rows) if rows else None,
            "false_alarms_per_10k_rows": (len(rows) - n_true) / (
                len({r["cell"] for r in rows}) * N_INSTANCES / 10_000) if rows else None,
        }
    return out


def main() -> None:
    ev = {d: load(g) for d, g in GRIDS.items()}
    cells = {d: {r["cell"] for r in rows} for d, rows in ev.items()}
    ddm_cells = cells["ddm"]

    payload = {
        "label": "[DIAGNOSTIC] DDM alarms vs published change points, insects_abrupt",
        "scope": {"stream": "insects_abrupt", "policy": POLICY,
                  "families": "all four; seed 0 for xgb and gnb, all seeds for rf and sgd",
                  "ground_truth": "Souza et al. 2020, Table 2 (arXiv:2005.00113)",
                  "change_points": list(CHANGE_POINTS), "instances": N_INSTANCES},
        "detectors": {}, "adwin_on_ddm_cells": {}, "per_family": {},
    }

    print(f"change points {CHANGE_POINTS} over {N_INSTANCES:,} rows, policy {POLICY}\n")
    print(f"{'detector':10}{'cells':>7}{'alarms':>8}" +
          "".join(f"{'<=' + str(t):>12}" for t in TOLERANCES))
    for d in ("adwin", "ddm"):
        s = shares(ev[d])
        payload["detectors"][d] = {"cells_with_alarms": len(cells[d]), **s}
        print(f"{d:10}{len(cells[d]):>7}{len(ev[d]):>8}" +
              "".join(f"{s[str(t)]['share_within']:>11.1%} " for t in TOLERANCES))

    sub = [r for r in ev["adwin"] if r["cell"] in ddm_cells]
    s = shares(sub)
    payload["adwin_on_ddm_cells"] = {"cells": sorted(f"{m}:{x}" for m, x in ddm_cells), **s}
    print(f"{'adwin*':10}{len(ddm_cells):>7}{len(sub):>8}" +
          "".join(f"{s[str(t)]['share_within']:>11.1%} " for t in TOLERANCES))
    print("\n* ADWIN restricted to the", len(ddm_cells),
          "family x seed cells in which DDM alarmed at all:")
    print("  ", ", ".join(sorted(f"{m}:{x}" for m, x in ddm_cells)))
    missing = sorted(f"{m}:{x}" for m, x in cells["adwin"] - ddm_cells)
    print("   cells where ADWIN alarmed and DDM did not:", ", ".join(missing) or "none")
    payload["cells_adwin_only"] = missing

    print(f"\nper family, share within 1,000 rows")
    print(f"{'family':8}{'adwin n':>9}{'adwin':>9}{'ddm n':>8}{'ddm':>9}")
    for fam in ("xgb", "rf", "sgd", "gnb"):
        rec = {}
        line = f"{fam:8}"
        for d in ("adwin", "ddm"):
            rows = [r for r in ev[d] if r["model"] == fam]
            sh = shares(rows)["1000"]["share_within"] if rows else None
            rec[d] = {"n_alarms": len(rows), "share_within_1000": sh}
            line += f"{len(rows):>9}" + (f"{sh:>9.1%}" if sh is not None else f"{'n/a':>9}")
        payload["per_family"][fam] = rec
        print(line)

    fa = payload["detectors"]
    print("\nfalse alarms per 10,000 rows at tolerance 1,000: "
          f"ADWIN {fa['adwin']['1000']['false_alarms_per_10k_rows']:.3f}, "
          f"DDM {fa['ddm']['1000']['false_alarms_per_10k_rows']:.3f}")

    out = ROOT / "results/analysis/phase9_change_points_ddm.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
