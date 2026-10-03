"""Phase 10 analysis — headroom (G1, G2) and whether any of it is learnable.

Reads `results/analysis/phase10_oracle.json` (per-alarm ops and errors for every
branch, at three horizons) and answers Gate 10:

- **G1** = cost(selector) - cost(O1), the best of SKIP/NUDGE/REBUILD per alarm
  with the guard kept: the most any better decision could save with today's
  mechanisms.
- **G2** = cost(selector) - cost(O2), best of all four actions including DEFER:
  what changing the guard and the action set could save.
- cost(NeverAdapt) - cost(O1): does any adaptation beat doing nothing, even in
  hindsight?

All reported over the lambda grid rather than at one chosen value, and broken
down by model family.

**Learnability.** A deliberately overfit gradient-boosted classifier is fitted
*in sample* to predict the oracle action from decision-time features, together
with two trivial rules -- "family only" (that family's most common oracle action)
and "deficit only" (a stump on the accuracy deficit). Each is scored by the share
of G1 it captures: (cost(selector) - cost(rule)) / (cost(selector) - cost(O1)).
An in-sample fit is an upper bound by construction; if it captures little, no
honest model will capture more.

Output: results/analysis/phase10_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("SKIP", "NUDGE", "REBUILD", "DEFER")
O1_ACTIONS = ("SKIP", "NUDGE", "REBUILD")


def j(row_h: dict, action: str, lam: float) -> float:
    ops, errs = row_h[action][0], row_h[action][1]
    return ops + lam * errs


def totals(rows, hname, lam):
    pol = o1 = o2 = never = 0.0
    o1_mix, o2_mix = Counter(), Counter()
    for r in rows:
        h = r["h"][hname]
        best1 = min(O1_ACTIONS, key=lambda a: j(h, a, lam))
        best2 = min(ACTIONS, key=lambda a: j(h, a, lam))
        pol += j(h, r["policy_action"], lam)
        o1 += j(h, best1, lam)
        o2 += j(h, best2, lam)
        never += j(h, "SKIP", lam)
        o1_mix[best1] += 1
        o2_mix[best2] += 1
    return {"policy": pol, "O1": o1, "O2": o2, "never": never,
            "G1": pol - o1, "G2": pol - o2,
            "G1_share": (pol - o1) / pol if pol else None,
            "G2_share": (pol - o2) / pol if pol else None,
            "never_minus_O1": never - o1,
            "o1_mix": dict(o1_mix), "o2_mix": dict(o2_mix)}


def learnability(rows, hname, lam):
    """How much of G1 an in-sample overfit model, and two trivial rules, capture."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    fams = sorted({r["model"] for r in rows})
    X, y, costs = [], [], []
    for r in rows:
        h = r["h"][hname]
        best1 = min(O1_ACTIONS, key=lambda a: j(h, a, lam))
        X.append([fams.index(r["model"]), r["window_rows"], r["deficit"],
                  r["rows_since_previous_alarm"] or 0, int(r["degraded_to_rebuild"]),
                  r["reference"]])
        y.append(best1)
        costs.append({a: j(h, a, lam) for a in O1_ACTIONS})
    X, y = np.array(X, dtype=float), np.array(y)

    pol = sum(c[r["policy_action"]] if r["policy_action"] in c else min(c.values())
              for c, r in zip(costs, rows))
    oracle = sum(min(c.values()) for c in costs)
    denom = pol - oracle

    def captured(pred):
        got = sum(c[p] for c, p in zip(costs, pred))
        return {"cost": got, "captured_share_of_G1": ((pol - got) / denom) if denom else None,
                "accuracy_vs_oracle": float(np.mean(pred == y))}

    out = {"n": len(rows), "policy_cost": pol, "oracle_cost": oracle, "G1": denom}
    if len(set(y)) < 2:
        out["note"] = "the oracle chooses one action everywhere; nothing to learn"
        out["family_only"] = captured(np.array([y[0]] * len(y)))
        return out

    gbm = GradientBoostingClassifier(n_estimators=400, max_depth=6, learning_rate=0.2)
    gbm.fit(X, y)
    out["overfit_gbm_in_sample"] = captured(gbm.predict(X))

    fam_choice = {}
    for f in range(len(fams)):
        mask = X[:, 0] == f
        fam_choice[f] = Counter(y[mask]).most_common(1)[0][0] if mask.any() else y[0]
    out["family_only"] = captured(np.array([fam_choice[int(v)] for v in X[:, 0]]))

    stump = DecisionTreeClassifier(max_depth=1).fit(X[:, [2]], y)
    out["deficit_only_stump"] = captured(stump.predict(X[:, [2]]))
    out["always_skip"] = captured(np.array(["SKIP"] * len(y)))
    out["oracle_mix"] = dict(Counter(y))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", default="results/analysis/phase10_oracle.json")
    ap.add_argument("--out", default="results/analysis/phase10_summary.json")
    ap.add_argument("--learn-lambda", type=float, default=1e3)
    args = ap.parse_args()

    data = json.loads((ROOT / args.oracle).read_text(encoding="utf-8"))
    rows = []
    for cell in data["cells"]:
        for r in cell["per_alarm"]:
            r = dict(r)
            r.update({"dataset": cell["dataset"], "model": cell["model"], "seed": cell["seed"]})
            rows.append(r)

    lambdas = data["lambda_grid"]
    summary = {"label": "declared follow-up (oracle upper bound)", "subset": data["subset"],
               "n_alarms": len(rows), "lambda_grid": lambdas, "by_horizon": {}}

    for hname in ("next", "500", "1000"):
        per_lambda, per_family = {}, {}
        for lam in lambdas:
            per_lambda[str(lam)] = totals(rows, hname, lam)
            per_family[str(lam)] = {
                fam: totals([r for r in rows if r["model"] == fam], hname, lam)
                for fam in sorted({r["model"] for r in rows})}
        summary["by_horizon"][hname] = {"pooled": per_lambda, "by_family": per_family}

    # learnability is lambda-dependent: a rule that helps at one trade-off can hurt at another
    summary["learnability"] = {
        "horizon": "next",
        "by_lambda": {str(lam): learnability(rows, "next", lam)
                      for lam in (1e2, 1e3, 1e4, 1e5)},
    }

    out = ROOT / args.out
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{len(rows)} alarms · elec2 + covtype · ADWIN\n")
    for hname in ("next", "500", "1000"):
        print(f"== horizon {hname}")
        print(f"{'lambda':>10}{'G1 share':>11}{'G2 share':>11}{'never-O1':>16}{'O1 mix':>34}")
        for lam in lambdas:
            t = summary["by_horizon"][hname]["pooled"][str(lam)]
            mix = " ".join(f"{a[:2]}:{t['o1_mix'].get(a, 0)}" for a in O1_ACTIONS)
            print(f"{lam:10.0f}{t['G1_share']:11.3f}{t['G2_share']:11.3f}"
                  f"{t['never_minus_O1']:16,.0f}   {mix:>28}")
        print()
    print("learnability, horizon next (share of G1 captured; in-sample GBM is an upper bound)")
    print(f"{'lambda':>10}{'G1':>16}{'overfit GBM':>14}{'family only':>14}{'deficit stump':>15}"
          f"{'always SKIP':>14}")
    for lam, L in summary["learnability"]["by_lambda"].items():
        def g(k):
            v = L.get(k)
            return "n/a" if not v or v["captured_share_of_G1"] is None else f"{v['captured_share_of_G1']:.1%}"
        print(f"{float(lam):10.0f}{L['G1']:16,.0f}{g('overfit_gbm_in_sample'):>14}"
              f"{g('family_only'):>14}{g('deficit_only_stump'):>15}{g('always_skip'):>14}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
