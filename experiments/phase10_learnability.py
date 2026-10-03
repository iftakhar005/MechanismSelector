"""Phase 10 learnability — out-of-sample, not in-sample.

The in-sample GBM number in the first pass is uninformative: it is fitted and
scored on the same alarms, so it bounds what is fittable and nothing else. This
script replaces it with tests that can fail:

- **cross-stream**: fit on elec2, test on covtype, and the reverse.
- **leave-one-seed-out**: fit on four seeds, test on the fifth, within each
  stream (the deterministic families contribute one seed, so this is driven by
  RandomForest and SGD).

Each is scored by the share of G1 it captures on the held-out alarms:
`(cost(selector) - cost(rule)) / (cost(selector) - cost(oracle))`, alongside
`always SKIP` and `family only` as reference rules, at every lambda on the grid.

Output: results/analysis/phase10_learnability.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
O1 = ("SKIP", "NUDGE", "REBUILD")
FEATURES = ("family", "window_rows", "deficit", "rows_since_previous_alarm", "degraded", "reference")


def j(h, a, lam):
    return h[a][0] + lam * h[a][1]


def rows_of(data):
    fams = sorted({c["model"] for c in data["cells"]})
    out = []
    for cell in data["cells"]:
        for r in cell["per_alarm"]:
            out.append({
                "dataset": cell["dataset"], "model": cell["model"], "seed": cell["seed"],
                "x": [fams.index(cell["model"]), r["window_rows"], r["deficit"],
                      r["rows_since_previous_alarm"] or 0, int(r["degraded_to_rebuild"]),
                      r["reference"]],
                "policy_action": r["policy_action"], "h": r["h"]["next"],
            })
    return out


def score(rows, predict, lam):
    """Share of G1 captured on `rows` by a rule that maps a row to an action."""
    pol = sum(j(r["h"], r["policy_action"], lam) for r in rows)
    orc = sum(min(j(r["h"], a, lam) for a in O1) for r in rows)
    got = sum(j(r["h"], predict(r), lam) for r in rows)
    denom = pol - orc
    return {"captured_share_of_G1": (pol - got) / denom if denom else None,
            "policy_cost": pol, "oracle_cost": orc, "rule_cost": got,
            "accuracy_vs_oracle": float(np.mean(
                [predict(r) == min(O1, key=lambda a: j(r["h"], a, lam)) for r in rows]))}


def fit_predict(train, test, lam):
    y = np.array([min(O1, key=lambda a: j(r["h"], a, lam)) for r in train])
    X = np.array([r["x"] for r in train], dtype=float)
    if len(set(y)) < 2:
        choice = y[0]
        return lambda r: choice
    gbm = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.1).fit(X, y)
    preds = dict(zip(range(len(test)), gbm.predict(np.array([r["x"] for r in test], dtype=float))))
    index = {id(r): i for i, r in enumerate(test)}
    return lambda r: preds[index[id(r)]]


def family_rule(train, lam):
    y = [min(O1, key=lambda a: j(r["h"], a, lam)) for r in train]
    best = {}
    for r, a in zip(train, y):
        best.setdefault(r["model"], Counter())[a] += 1
    choice = {m: c.most_common(1)[0][0] for m, c in best.items()}
    fallback = Counter(y).most_common(1)[0][0]
    return lambda r: choice.get(r["model"], fallback)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", default="results/analysis/phase10_oracle.json")
    ap.add_argument("--out", default="results/analysis/phase10_learnability.json")
    args = ap.parse_args()

    data = json.loads((ROOT / args.oracle).read_text(encoding="utf-8"))
    rows = rows_of(data)
    lambdas = data["lambda_grid"]
    streams = sorted({r["dataset"] for r in rows})

    out = {"label": "declared follow-up", "note": "all numbers here are out-of-sample",
           "lambda_grid": lambdas, "cross_stream": {}, "leave_one_seed_out": {},
           "reference_rules": {}}

    for lam in lambdas:
        key = str(lam)
        out["cross_stream"][key] = {}
        for train_ds in streams:
            test_ds = [d for d in streams if d != train_ds][0]
            train = [r for r in rows if r["dataset"] == train_ds]
            test = [r for r in rows if r["dataset"] == test_ds]
            out["cross_stream"][key][f"train_{train_ds}_test_{test_ds}"] = {
                "n_train": len(train), "n_test": len(test),
                "gbm": score(test, fit_predict(train, test, lam), lam),
                "family_only": score(test, family_rule(train, lam), lam),
            }
        out["reference_rules"][key] = {
            ds_: score([r for r in rows if r["dataset"] == ds_], lambda r: "SKIP", lam)
            for ds_ in streams}

        loo = {}
        for ds_ in streams:
            sub = [r for r in rows if r["dataset"] == ds_]
            seeds = sorted({r["seed"] for r in sub})
            shares = []
            for s in seeds:
                train = [r for r in sub if r["seed"] != s]
                test = [r for r in sub if r["seed"] == s]
                if not train or not test:
                    continue
                res = score(test, fit_predict(train, test, lam), lam)
                if res["captured_share_of_G1"] is not None:
                    shares.append(res["captured_share_of_G1"])
            loo[ds_] = {"n_folds": len(shares),
                        "median_captured_share_of_G1": float(np.median(shares)) if shares else None,
                        "folds": shares}
        out["leave_one_seed_out"][key] = loo

    (ROOT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"{'lambda':>9}  {'train->test':<28}{'GBM':>9}{'family':>9}{'always SKIP (test)':>21}")
    for lam in lambdas:
        k = str(lam)
        for name, v in out["cross_stream"][k].items():
            test_ds = name.split("_test_")[1]
            skip = out["reference_rules"][k][test_ds]["captured_share_of_G1"]
            print(f"{lam:9.0f}  {name:<28}{v['gbm']['captured_share_of_G1']:9.3f}"
                  f"{v['family_only']['captured_share_of_G1']:9.3f}{skip:21.3f}")
    print(f"\n{'lambda':>9}  leave-one-seed-out median capture")
    for lam in lambdas:
        k = str(lam)
        line = "  ".join(f"{d}: {v['median_captured_share_of_G1']:.3f}"
                         if v["median_captured_share_of_G1"] is not None else f"{d}: n/a"
                         for d, v in out["leave_one_seed_out"][k].items())
        print(f"{lam:9.0f}  {line}")
    print(f"\nwrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
