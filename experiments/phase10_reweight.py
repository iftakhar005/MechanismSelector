"""Item 2 - re-weight training against inference inside the oracle's J.

Training work units and inference operations are not the same quantity, so the
oracle's J, which adds them, implicitly weights them 1:1. This recomputes J with

    J_w(a) = w * train_ops + eval_ops + infer_per_row * H + lambda * errors

for w in {1, 10, 100}, on all five streams, from the stored per-branch records.
Nothing is re-executed. `eval_ops` is inference units times holdout rows, so it
is inference work and stays unweighted; only `train` is multiplied by w.

w = 1 reproduces `phase10_per_stream.py` exactly and is kept as a self-check.

Reads `results/analysis/phase10_oracle.json` and `..._oracle_insects.json`.
Writes `results/analysis/phase10_reweight.json`. [DIAGNOSTIC]
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
O1 = ("SKIP", "NUDGE", "REBUILD")
ACTIONS = ("SKIP", "NUDGE", "REBUILD", "DEFER")
WEIGHTS = (1, 10, 100)
LAMBDAS = [0.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
STREAMS = ("elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental")


def load_rows() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for name in ("phase10_oracle.json", "phase10_oracle_insects.json"):
        data = json.loads((ROOT / "results/analysis" / name).read_text(encoding="utf-8"))
        for cell in data["cells"]:
            for r in cell["per_alarm"]:
                r = dict(r)
                r["model"], r["seed"] = cell["model"], cell["seed"]
                out.setdefault(cell["dataset"], []).append(r)
    return out


def jw(r: dict, a: str, lam: float, w: float, hname: str = "next") -> float:
    b = r["branch_ops"][a]
    h = r["h"][hname][a]
    return w * b["train"] + b["eval"] + b["infer_per_row"] * h[2] + lam * h[1]


def analyse(rows: list[dict], lam: float, w: float, hname: str = "next") -> dict:
    sel_j = 0.0
    short_total = 0.0
    by_branch: dict[str, float] = defaultdict(float)
    picks: Counter[str] = Counter()
    for r in rows:
        best = min(O1, key=lambda a: jw(r, a, lam, w, hname))
        picks[best] += 1
        pol_j = jw(r, r["policy_action"], lam, w, hname)
        sel_j += pol_j
        short = pol_j - jw(r, best, lam, w, hname)
        short_total += short
        if best != r["policy_action"]:
            by_branch[best] += short
    return {
        "n_alarms": len(rows),
        "G1": short_total,
        "G1_share_of_selector_J": short_total / sel_j if sel_j else None,
        "oracle_pick_counts": dict(picks),
        "advantage_by_branch_share": {
            a: (by_branch.get(a, 0.0) / short_total if short_total else None) for a in ACTIONS},
    }


def main() -> None:
    rows = load_rows()
    payload = {
        "label": "[DIAGNOSTIC] training/inference re-weighting of the oracle objective",
        "definition": "J_w(a) = w*train + eval + infer_per_row*H + lambda*errors, horizon next",
        "note": "eval_ops is inference work (units x holdout rows) and is not weighted",
        "weights": list(WEIGHTS), "lambda_grid": LAMBDAS,
        "streams": {},
    }
    for name in STREAMS:
        payload["streams"][name] = {
            str(w): {f"{lam:.0e}": analyse(rows[name], lam, w) for lam in LAMBDAS}
            for w in WEIGHTS}

    print("G1 share of selector J, range over the lambda grid")
    print(f"{'stream':22}{'n':>5}" + "".join(f"{'w=' + str(w):>18}" for w in WEIGHTS))
    for name in STREAMS:
        line = f"{name:22}{len(rows[name]):>5}"
        for w in WEIGHTS:
            vals = [payload["streams"][name][str(w)][f"{lam:.0e}"]["G1_share_of_selector_J"]
                    for lam in LAMBDAS]
            line += f"{min(vals):8.1%}-{max(vals):<9.1%}"
        print(line)

    print("\nSKIP share of the oracle's advantage at lambda = 1e3")
    print(f"{'stream':22}" + "".join(f"{'w=' + str(w):>10}" for w in WEIGHTS))
    for name in STREAMS:
        line = f"{name:22}"
        for w in WEIGHTS:
            s = payload["streams"][name][str(w)]["1e+03"]["advantage_by_branch_share"]["SKIP"]
            line += f"{s:10.1%}" if s is not None else f"{'n/a':>10}"
        print(line)

    print("\nfull branch mix at lambda = 1e3")
    for name in STREAMS:
        for w in WEIGHTS:
            m = payload["streams"][name][str(w)]["1e+03"]["advantage_by_branch_share"]
            print(f"   {name:22} w={w:<4} " + "  ".join(
                f"{a} {m[a]:6.1%}" if m[a] is not None else f"{a} n/a" for a in ACTIONS))

    out = ROOT / "results/analysis/phase10_reweight.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
