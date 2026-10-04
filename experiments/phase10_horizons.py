"""Oracle results at all three horizons [DIAGNOSTIC].

The main tables report horizon "next" (rows until the next alarm, capped at
1,000). The oracle also stored fixed H = 500 and H = 1,000 for every alarm, so
this recomputes the headline quantities at each horizon without re-executing
anything:

- G1 as a share of the selector's J, range over the lambda grid
- branch decomposition of the oracle's advantage at lambda = 1e3
- "always SKIP" capture of G1 at lambda = 1e2, 1e3, 1e4

Reads `results/analysis/phase10_oracle.json` and `..._oracle_insects.json`.
Writes `results/analysis/phase10_horizons.json`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from phase10_analysis import O1_ACTIONS, j  # noqa: E402

ACTIONS = ("SKIP", "NUDGE", "REBUILD", "DEFER")
HORIZONS = ("next", "500", "1000")
LAMBDAS = [0.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
CAPTURE_LAMBDAS = [1e2, 1e3, 1e4]
STREAMS = ("elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental")


def load_rows() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for name in ("phase10_oracle.json", "phase10_oracle_insects.json"):
        data = json.loads((ROOT / "results/analysis" / name).read_text(encoding="utf-8"))
        for cell in data["cells"]:
            out.setdefault(cell["dataset"], []).extend(cell["per_alarm"])
    return out


def analyse(rows: list[dict], lam: float, hname: str) -> dict:
    sel = short_total = 0.0
    by_branch: dict[str, float] = defaultdict(float)
    picks: Counter[str] = Counter()
    for r in rows:
        h = r["h"][hname]
        best = min(O1_ACTIONS, key=lambda a: j(h, a, lam))
        picks[best] += 1
        pol = j(h, r["policy_action"], lam)
        sel += pol
        short = pol - j(h, best, lam)
        short_total += short
        if best != r["policy_action"]:
            by_branch[best] += short
    return {
        "G1": short_total,
        "G1_share": short_total / sel if sel else None,
        "branch_share": {a: (by_branch.get(a, 0.0) / short_total if short_total else None)
                         for a in ACTIONS},
        "oracle_picks": dict(picks),
    }


def capture(rows: list[dict], lam: float, hname: str) -> float | None:
    pol = o1 = skip = 0.0
    for r in rows:
        h = r["h"][hname]
        pol += j(h, r["policy_action"], lam)
        o1 += j(h, min(O1_ACTIONS, key=lambda a: j(h, a, lam)), lam)
        skip += j(h, "SKIP", lam)
    denom = pol - o1
    return (pol - skip) / denom if denom else None


def main() -> None:
    rows = load_rows()
    payload: dict = {"label": "[DIAGNOSTIC] oracle at all three horizons",
                     "horizons": list(HORIZONS), "lambda_grid": LAMBDAS, "streams": {}}

    for name in STREAMS:
        payload["streams"][name] = {
            h: {"n_alarms": len(rows[name]),
                "by_lambda": {f"{lam:.0e}": analyse(rows[name], lam, h) for lam in LAMBDAS},
                "always_skip_capture": {f"{lam:.0e}": capture(rows[name], lam, h)
                                        for lam in CAPTURE_LAMBDAS}}
            for h in HORIZONS}

    print("G1 share of the selector's J, range over the lambda grid")
    print(f"{'stream':22}{'n':>5}" + "".join(f"{'H=' + h:>20}" for h in HORIZONS))
    for name in STREAMS:
        line = f"{name:22}{len(rows[name]):>5}"
        for h in HORIZONS:
            vals = [payload["streams"][name][h]["by_lambda"][f"{lam:.0e}"]["G1_share"]
                    for lam in LAMBDAS]
            line += f"{min(vals):9.1%}-{max(vals):<10.1%}"
        print(line)

    print("\nSKIP share of the oracle's advantage at lambda = 1e3")
    print(f"{'stream':22}" + "".join(f"{'H=' + h:>12}" for h in HORIZONS))
    for name in STREAMS:
        line = f"{name:22}"
        for h in HORIZONS:
            s = payload["streams"][name][h]["by_lambda"]["1e+03"]["branch_share"]["SKIP"]
            line += f"{s:12.1%}" if s is not None else f"{'n/a':>12}"
        print(line)

    print("\nfull branch mix at lambda = 1e3")
    for name in STREAMS:
        for h in HORIZONS:
            m = payload["streams"][name][h]["by_lambda"]["1e+03"]["branch_share"]
            print(f"   {name:22} H={h:<6} " + "  ".join(
                f"{a} {m[a]:6.1%}" if m[a] is not None else f"{a} n/a" for a in ACTIONS))

    print("\n'always SKIP' capture of G1")
    print(f"{'stream':22}{'H':>7}" + "".join(f"{'lam=' + f'{l:.0e}':>14}" for l in CAPTURE_LAMBDAS))
    for name in STREAMS:
        for h in HORIZONS:
            cells = "".join(
                f"{(f'{v:.1%}' if v is not None else 'n/a'):>14}"
                for v in (payload["streams"][name][h]["always_skip_capture"][f"{l:.0e}"]
                          for l in CAPTURE_LAMBDAS))
            print(f"{name:22}{h:>7}{cells}")

    out = ROOT / "results/analysis/phase10_horizons.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
