"""Per-stream 'always SKIP' capture of G1, using phase10_analysis's own formula.

capture = (cost(selector) - cost(always-SKIP)) / (cost(selector) - cost(O1)),
horizon "next", pooled over the four families of a stream.
Elec2/Covertype are recomputed the same way so the numbers are comparable.
"""
import json
import sys
from pathlib import Path

ROOT = Path("E:/My-Projects/MechanismSelector")
sys.path.insert(0, str(ROOT / "experiments"))
from phase10_analysis import O1_ACTIONS, j  # noqa: E402

LAMBDAS = [1e2, 1e3, 1e4]


def rows_by_stream(path):
    payload = json.loads(Path(path).read_text())
    out = {}
    for c in payload["cells"]:
        out.setdefault(c["dataset"], []).extend(c["per_alarm"])
    return out


def capture(rows, lam, hname="next"):
    pol = o1 = skip = 0.0
    for r in rows:
        h = r["h"][hname]
        pol += j(h, r["policy_action"], lam)
        o1 += j(h, min(O1_ACTIONS, key=lambda a: j(h, a, lam)), lam)
        skip += j(h, "SKIP", lam)
    denom = pol - o1
    return (pol - skip) / denom if denom else None


streams = {}
streams.update(rows_by_stream(ROOT / "results/analysis/phase10_oracle_insects.json"))
streams.update(rows_by_stream(ROOT / "results/analysis/phase10_oracle.json"))

print(f"{'stream':22}{'n':>5}" + "".join(f"{'lam=' + f'{l:.0e}':>14}" for l in LAMBDAS))
res = {}
for name in ("elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental"):
    rows = streams.get(name, [])
    vals = [capture(rows, l) for l in LAMBDAS]
    res[name] = {f"{l:.0e}": v for l, v in zip(LAMBDAS, vals)}
    cells = "".join(f"{(f'{v:.1%}' if v is not None else 'n/a'):>14}" for v in vals)
    print(f"{name:22}{len(rows):>5}{cells}")

(ROOT / "results/analysis/phase10_always_skip_per_stream.json").write_text(
    json.dumps({"horizon": "next", "statistic":
                "(cost(selector)-cost(always SKIP))/(cost(selector)-cost(O1)), pooled over families",
                "capture": res}, indent=2), encoding="utf-8")
print("\nwrote results/analysis/phase10_always_skip_per_stream.json")
