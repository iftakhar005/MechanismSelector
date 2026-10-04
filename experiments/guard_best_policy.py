"""Q1/Q2: is any variant the best policy on any stream, and does the finding hold?

Per stream, median over the 12 (family x seed) blocks of training work units and
prequential balanced accuracy, for the five frozen-grid policies and each guard
variant. "Best" is reported two ways: highest median Table IV accuracy, and
non-dominated on (training lower better, accuracy higher better).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("E:/My-Projects/MechanismSelector")
sys.path.insert(0, str(ROOT / "src"))
import analysis as an  # noqa: E402

an.METRICS["balanced"] = lambda r: 100.0 * float(r["mean_prequential_balanced_accuracy"])
an.HIGHER_IS_BETTER["balanced"] = True

STREAMS = ("elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental")
VARIANTS = {"A": "guard_A", "B": "guard_B", "C": "guard_C", "D": "guard_D"}

rows = an.load_grid(ROOT / "results/grid")
for v, d in VARIANTS.items():
    for r in an.load_grid(ROOT / "results" / d):
        r = dict(r)
        r["policy_key"] = f"guard_{v}"
        rows.append(r)

policies = tuple(an.POLICIES) + tuple(f"guard_{v}" for v in VARIANTS)
label = dict(an.LABELS)
label.update({f"guard_{v}": f"selector+{v}" for v in VARIANTS})

print("median training work units / median Table IV accuracy, 12 blocks per stream\n")
answers = {}
for stream in STREAMS:
    cells = [c for c in an.cells_of(rows) if c[0] == stream]
    blk = {m: an.build_blocks(rows, m, "n60", policies, cells) for m in ("cost", "accuracy")}
    med = {p: (float(np.median(blk["cost"].values[:, blk["cost"].policies.index(p)])),
               float(np.median(blk["accuracy"].values[:, blk["accuracy"].policies.index(p)])))
           for p in policies}
    frontier = an.pareto_frontier(med)
    best_acc = max(med, key=lambda p: med[p][1])
    print(f"=== {stream}   best accuracy: {label[best_acc]}   frontier: "
          f"{sorted(label[p] for p in frontier)}")
    for p in policies:
        mark = "  <- frontier" if p in frontier else ""
        print(f"    {label[p]:20}{med[p][0]:14,.0f}{med[p][1]:9.2f}{mark}")
    answers[stream] = (best_acc, frontier)
    print()

print("Q1: does any variant become the best policy on any stream?")
for stream, (best_acc, frontier) in answers.items():
    vwin = [p for p in frontier if p.startswith("guard_")]
    print(f"   {stream:22} best accuracy = {label[best_acc]:20} "
          f"variants on frontier: {sorted(label[p] for p in vwin) or 'none'}")

print("\nQ2: does the selector (or any variant of it) win on any stream?")
for stream, (best_acc, frontier) in answers.items():
    sel_family = {"mechanism_selector"} | {f"guard_{v}" for v in VARIANTS}
    print(f"   {stream:22} best accuracy is a selector variant: "
          f"{best_acc in sel_family}")
