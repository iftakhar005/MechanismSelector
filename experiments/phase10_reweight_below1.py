"""Item 2 - extend the oracle re-weighting below 1:1, to w = 0.1 and w = 0.01.

`phase10_reweight.py` weighted training *above* inference (w in {1, 10, 100}).
This runs the same objective in the other direction, where training is cheap
relative to inference:

    J_w(a) = w * train_ops + eval_ops + infer_per_row * H + lambda * errors

for w in {0.1, 0.01}, all five streams, lambda = 1e3, horizon "next",
recomputed from the stored per-branch records. Nothing is re-executed, and the
definition is unchanged: `eval_ops` is inference work and stays unweighted, so
only `train` is multiplied by w.

w = 1 is recomputed here as well and checked against the committed
`phase10_reweight.json` values, so the extension is known to sit on the same
objective.

The reading was fixed before the numbers were looked at: if SKIP is still the
largest single source of the oracle's advantage on all five streams at w = 0.1,
the decomposition is robust in both directions down to w = 0.1; otherwise the
stream and the weight at which another branch overtakes SKIP are named. w = 0.01
is reported either way and does not change that reading.

Writes `results/analysis/phase10_reweight_below1.json`. [DIAGNOSTIC]
"""

from __future__ import annotations

import json
from pathlib import Path

from phase10_reweight import ACTIONS, LAMBDAS, STREAMS, analyse, load_rows

ROOT = Path(__file__).resolve().parents[1]
NEW_WEIGHTS = (0.1, 0.01)
ALL_WEIGHTS = (100, 10, 1, 0.1, 0.01)
LAM_KEY = "1e+03"


def main() -> None:
    rows = load_rows()
    payload = {
        "label": "[DIAGNOSTIC] oracle re-weighting below 1:1 (training cheaper than inference)",
        "definition": "J_w(a) = w*train + eval + infer_per_row*H + lambda*errors, horizon next",
        "note": "eval_ops is inference work (units x holdout rows) and is not weighted",
        "declared_before_results": {
            "weights": list(NEW_WEIGHTS), "lam": 1e3, "horizon": "next",
            "streams": list(STREAMS),
            "reading": "SKIP largest on all five streams at w=0.1 -> robust in both "
                       "directions down to w=0.1; otherwise name the stream(s) and the w "
                       "at which another branch overtakes SKIP",
        },
        "weights": list(ALL_WEIGHTS), "lambda_grid": LAMBDAS, "streams": {},
    }
    for name in STREAMS:
        payload["streams"][name] = {
            str(w): {f"{lam:.0e}": analyse(rows[name], lam, w) for lam in LAMBDAS}
            for w in ALL_WEIGHTS}

    # ---- self-check: w = 1 must reproduce the committed file exactly
    old = json.loads((ROOT / "results/analysis/phase10_reweight.json").read_text(encoding="utf-8"))
    diffs = []
    for name in STREAMS:
        for lam in LAMBDAS:
            k = f"{lam:.0e}"
            a = payload["streams"][name]["1"][k]["G1"]
            b = old["streams"][name]["1"][k]["G1"]
            if abs(a - b) > 1e-6 * max(1.0, abs(b)):
                diffs.append((name, k, a, b))
    payload["self_check_w1_matches_committed"] = not diffs
    print("self-check, w=1 reproduces the committed phase10_reweight.json:", not diffs)
    if diffs:
        for d in diffs:
            print("   MISMATCH", d)

    # ---- the requested table
    print(f"\nBranch share of the oracle's advantage at lambda = 1e3, horizon \"next\"")
    print(f"{'stream':22}{'w':>8}  " + "".join(f"{a:>10}" for a in ACTIONS) + "   largest")
    overtaken = []
    for name in STREAMS:
        for w in ALL_WEIGHTS:
            m = payload["streams"][name][str(w)][LAM_KEY]["advantage_by_branch_share"]
            defined = {a: v for a, v in m.items() if v is not None}
            top = max(defined, key=defined.get) if defined else None
            cells = "".join(f"{m[a]:10.1%}" if m[a] is not None else f"{'n/a':>10}"
                            for a in ACTIONS)
            print(f"{name:22}{w:>8}  {cells}   {top}")
            if top not in (None, "SKIP"):
                overtaken.append((name, w, top, defined[top], defined.get("SKIP")))
        print()

    payload["skip_largest_at_w_0_1"] = not [o for o in overtaken if o[1] == 0.1]
    payload["branch_overtaking_skip"] = [
        {"stream": s, "w": w, "branch": b, "share": v, "skip_share": sk}
        for s, w, b, v, sk in overtaken]

    print("verdict on the pre-declared reading:")
    if payload["skip_largest_at_w_0_1"]:
        print("   SKIP is the largest single source on all five streams at w = 0.1")
        print("   -> robust in both directions down to w = 0.1")
    else:
        for s, w, b, v, sk in overtaken:
            if w == 0.1:
                print(f"   {s}: {b} overtakes SKIP at w = {w} ({v:.1%} vs {sk:.1%})")
    others = [o for o in overtaken if o[1] != 0.1]
    if others:
        print("   at other weights:")
        for s, w, b, v, sk in others:
            print(f"   {s}: {b} largest at w = {w} ({v:.1%} vs SKIP "
                  + (f"{sk:.1%})" if sk is not None else "n/a)"))

    out = ROOT / "results/analysis/phase10_reweight_below1.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
