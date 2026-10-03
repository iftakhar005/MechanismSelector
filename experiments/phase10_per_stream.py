"""Phase 10, per stream — selector vs oracle, branch attribution, guard attribution.

Reads `results/analysis/phase10_oracle.json`. Every table is per stream; nothing
is pooled across streams on its own, because elec2 and covtype behave
differently and a pooled row hides that.

Reported per stream, at each lambda on the grid and at horizon "next":

1. **Selector vs oracle (O1)** split into training ops, inference ops, combined
   ops, and accuracy over the horizon (1 - errors / rows scored).
2. **Branch attribution** - of the oracle's total advantage, how much comes from
   each alarm where it picks SKIP, NUDGE, REBUILD or DEFER instead of what the
   selector did.
3. **Guard attribution** - the share of the selector's shortfall that occurs at
   alarms the selector itself flagged `degraded_to_rebuild`, i.e. where the
   minimum-window guard forced the rebuild.
4. **G1 against the 5% decision threshold.**

Also evaluates **Candidate A** on its own terms: a "DEFER-perfect" policy that
defers at exactly those alarms where deferring costs no extra errors over the
horizon, and does whatever the selector did elsewhere. Reported as the training-op
saving and the accuracy difference against the selector.

Output: results/analysis/phase10_per_stream.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("SKIP", "NUDGE", "REBUILD", "DEFER")
O1 = ("SKIP", "NUDGE", "REBUILD")


def j(h: dict, a: str, lam: float) -> float:
    ops, errs, _ = h[a]
    return ops + lam * errs


def stream_rows(data, dataset):
    out = []
    for cell in data["cells"]:
        if cell["dataset"] != dataset:
            continue
        for r in cell["per_alarm"]:
            r = dict(r)
            r["model"], r["seed"] = cell["model"], cell["seed"]
            out.append(r)
    return out


def split_ops(r, a, hname):
    """Training, inference and combined ops for one branch at this horizon."""
    b = r["branch_ops"][a]
    h = r["h"][hname][a][2]
    infer = b["infer_per_row"] * h
    return b["train"], b["eval"] + infer, b["train"] + b["eval"] + infer


def analyse(rows, lam, hname="next"):
    sel = {"train": 0.0, "infer": 0.0, "ops": 0.0, "errors": 0, "scored": 0}
    orc = {"train": 0.0, "infer": 0.0, "ops": 0.0, "errors": 0, "scored": 0}
    by_branch = defaultdict(float)
    guard_short, total_short = 0.0, 0.0
    picks = Counter()

    for r in rows:
        h = r["h"][hname]
        best = min(O1, key=lambda a: j(h, a, lam))
        picks[best] += 1
        for who, action, acc in ((sel, r["policy_action"], None), (orc, best, None)):
            tr, inf, ops = split_ops(r, action, hname)
            who["train"] += tr
            who["infer"] += inf
            who["ops"] += ops
            who["errors"] += h[action][1]
            who["scored"] += h[action][2]
        short = j(h, r["policy_action"], lam) - j(h, best, lam)
        total_short += short
        if r["degraded_to_rebuild"]:
            guard_short += short
        if best != r["policy_action"]:
            by_branch[best] += short

    for d in (sel, orc):
        d["accuracy"] = 1 - d["errors"] / d["scored"] if d["scored"] else None
    return {
        "n_alarms": len(rows),
        "selector": sel, "oracle_O1": orc,
        "G1": total_short,
        "G1_share_of_selector_J": total_short / (sel["ops"] + lam * sel["errors"])
        if (sel["ops"] + lam * sel["errors"]) else None,
        "oracle_pick_counts": dict(picks),
        "advantage_by_branch": {a: by_branch.get(a, 0.0) for a in ACTIONS},
        "advantage_by_branch_share": {
            a: (by_branch.get(a, 0.0) / total_short if total_short else None) for a in ACTIONS},
        "guard_share_of_shortfall": (guard_short / total_short) if total_short else None,
        "guard_shortfall": guard_short,
        "n_guard_alarms": sum(1 for r in rows if r["degraded_to_rebuild"]),
    }


def candidate_a(rows, hname="next"):
    """DEFER-perfect: defer wherever deferring costs no extra errors over the horizon."""
    sel_train = sel_err = sel_scored = 0
    def_train = def_err = def_scored = 0
    n_deferred = 0
    for r in rows:
        h = r["h"][hname]
        pol = r["policy_action"]
        tr, _, _ = split_ops(r, pol, hname)
        sel_train += tr
        sel_err += h[pol][1]
        sel_scored += h[pol][2]
        defer_ok = h["DEFER"][1] <= h[pol][1]
        action = "DEFER" if defer_ok else pol
        n_deferred += int(defer_ok)
        tr2, _, _ = split_ops(r, action, hname)
        def_train += tr2
        def_err += h[action][1]
        def_scored += h[action][2]
    sel_acc = 1 - sel_err / sel_scored if sel_scored else None
    def_acc = 1 - def_err / def_scored if def_scored else None
    return {
        "n_alarms": len(rows), "n_deferred": n_deferred,
        "selector_training_ops": sel_train, "defer_perfect_training_ops": def_train,
        "training_ops_saved_share": (sel_train - def_train) / sel_train if sel_train else None,
        "selector_accuracy": sel_acc, "defer_perfect_accuracy": def_acc,
        "accuracy_delta_pp": (def_acc - sel_acc) * 100 if (sel_acc is not None) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", default="results/analysis/phase10_oracle.json")
    ap.add_argument("--out", default="results/analysis/phase10_per_stream.json")
    args = ap.parse_args()

    data = json.loads((ROOT / args.oracle).read_text(encoding="utf-8"))
    lambdas = data["lambda_grid"]
    datasets = sorted({c["dataset"] for c in data["cells"]})

    payload = {"label": "declared follow-up", "subset": data["subset"],
               "lambda_grid": lambdas, "per_stream": {}}

    for dsname in datasets:
        rows = stream_rows(data, dsname)
        payload["per_stream"][dsname] = {
            "by_lambda": {str(lam): analyse(rows, lam) for lam in lambdas},
            "by_family": {fam: {str(lam): analyse([r for r in rows if r["model"] == fam], lam)
                                for lam in lambdas}
                          for fam in sorted({r["model"] for r in rows})},
            "candidate_A_defer_perfect": candidate_a(rows),
        }

    (ROOT / args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for dsname in datasets:
        s = payload["per_stream"][dsname]
        print(f"\n================ {dsname} ({s['by_lambda'][str(lambdas[0])]['n_alarms']} alarms)")
        print(f"{'lambda':>9}{'sel train':>13}{'sel infer':>13}{'sel acc':>9}"
              f"{'orc train':>13}{'orc infer':>13}{'orc acc':>9}{'G1 share':>10}{'guard share':>12}")
        for lam in lambdas:
            a = s["by_lambda"][str(lam)]
            sel, orc = a["selector"], a["oracle_O1"]
            print(f"{lam:9.0f}{sel['train']:13,.0f}{sel['infer']:13,.0f}{sel['accuracy']:9.3f}"
                  f"{orc['train']:13,.0f}{orc['infer']:13,.0f}{orc['accuracy']:9.3f}"
                  f"{a['G1_share_of_selector_J']:10.3f}"
                  f"{(a['guard_share_of_shortfall'] if a['guard_share_of_shortfall'] is not None else float('nan')):12.3f}")
        print("  oracle advantage by branch (share of G1), lambda = 1000:")
        a = s["by_lambda"]["1000.0"] if "1000.0" in s["by_lambda"] else s["by_lambda"]["1000"]
        for act in ACTIONS:
            sh = a["advantage_by_branch_share"][act]
            print(f"    {act:8} {sh if sh is None else f'{sh:.3f}'}")
        ca = s["candidate_A_defer_perfect"]
        print(f"  candidate A: training ops saved {ca['training_ops_saved_share']:.3f}, "
              f"accuracy {ca['selector_accuracy']:.4f} -> {ca['defer_perfect_accuracy']:.4f} "
              f"({ca['accuracy_delta_pp']:+.2f} pp), deferred at {ca['n_deferred']}/{ca['n_alarms']}")
    print(f"\nwrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
