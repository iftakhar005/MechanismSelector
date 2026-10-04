"""Figures for the paper, from committed results only.

Okabe--Ito palette, used consistently: the same colour means the same thing in
every figure. White background, black text and axes, no gradients or shadows;
hatching is added where two colours sit side by side so the figures still read in
black-and-white print. No text is smaller than 7 pt at the size the figure is
placed. Every figure is written as PDF (vector) and PNG at 300 dpi.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe--Ito
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERM, PINK, SKY = "#D55E00", "#CC79A7", "#56B4E9"
GREY, LIGHT, BLACK = "#555555", "#CCCCCC", "#000000"

POLICY = {"never_adapt": BLUE, "always_rebuild": VERM, "always_nudge": SKY,
          "fixed_schedule": PINK, "mechanism_selector": GREEN,
          "waitandcheck": ORANGE, "confirmmd3": GREY}
BRANCH = {"SKIP": GREEN, "NUDGE": ORANGE, "REBUILD": VERM}
COL, FULL = 3.4, 7.0
DET = ("xgb", "gnb")

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": BLACK, "text.color": BLACK, "axes.labelcolor": BLACK,
    "xtick.color": BLACK, "ytick.color": BLACK, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "font.family": "serif",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def _inf_module():
    spec = importlib.util.spec_from_file_location(
        "p9", ROOT / "experiments" / "phase9_inference_cost.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def grid_runs(grid_dir: Path):
    INF = _inf_module()
    out = []
    for r in csv.DictReader((grid_dir / "grid.csv").open()):
        if r["model"] in DET and int(r["seed"]) != 0:
            continue
        traj = INF.size_trajectory(grid_dir / "events", r["dataset"], r["model"],
                                   r["policy_key"], int(r["seed"]))
        infer, _ = INF.inference_ops(r, traj)
        out.append({"dataset": r["dataset"], "model": r["model"], "policy": r["policy_key"],
                    "training": float(r["total_work_units"]),
                    "infer_per_pred": infer / float(r["n_stream_rows"]),
                    "acc": float(r["mean_prequential_balanced_accuracy"]) * 100})
    return out


# ---------------------------------------------------------------- F1 decision flow
def f1_pipeline():
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.4); ax.axis("off")

    def box(x, y, w, h, text, fill="white", edge=BLACK, fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor=fill, edgecolor=edge, linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=BLACK)

    def down(x, y1, y2, label=None):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=6,
                                     color=BLACK, linewidth=0.7, shrinkA=0, shrinkB=0))
        if label:
            ax.text(x + 0.18, (y1 + y2) / 2, label, fontsize=7.0, color=GREY,
                    ha="left", va="center")

    def out(y, text, colour):
        ax.add_patch(FancyArrowPatch((6.9, y), (7.8, y), arrowstyle="-|>", mutation_scale=6,
                                     color=BLACK, linewidth=0.7, shrinkA=0, shrinkB=0))
        box(7.85, y - 0.38, 4.05, 0.76, text, fill=colour + "33", edge=colour, fs=7.0)

    box(0.1, 6.5, 11.8, 0.8, "stream row: predict, score, buffer, ADWIN", fill=LIGHT + "99")
    down(3.5, 6.5, 6.0, "alarm")
    box(0.1, 5.2, 6.8, 0.8, "split by time (80/20)")
    down(3.5, 5.2, 4.7)
    box(0.1, 3.9, 6.8, 0.8, "window $<$ 250 rows?"); out(4.3, "REBUILD (guard)", VERM)
    down(3.5, 3.9, 3.4, "no")
    box(0.1, 2.6, 6.8, 0.8, "model $\\geq$ floor?"); out(3.0, "SKIP", GREEN)
    down(3.5, 2.6, 2.1, "no")
    box(0.1, 1.3, 6.8, 0.8, "nudge $\\geq$ floor?"); out(1.7, "NUDGE", ORANGE)
    down(3.5, 1.3, 0.8, "no")
    box(0.1, 0.0, 6.8, 0.76, "rebuild, pays for both"); out(0.38, "REBUILD", VERM)
    save(fig, "f1_decision_flow")


# ---------------------------------------------------------------- F2 alarm validity
def f2_alarm_validity():
    direction = load("results/analysis/phase9_direction_null.json")
    shuffle = load("results/analysis/phase9_block_shuffle.json")
    obs, null, shuf = {}, {}, {}
    for c in direction["observed_per_cell"]:
        obs[c["dataset"]] = obs.get(c["dataset"], 0) + c["alarms"]
    for c in direction["null_per_cell"]:
        null[c["dataset"]] = null.get(c["dataset"], 0) + c["nulls"]["ar1"]["alarms_mean"]
    for c in shuffle["per_cell"]:
        shuf[c["dataset"]] = shuf.get(c["dataset"], 0) + c["alarms"]["adwin"]["block_shuffled"]

    names = ["elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental"]
    labels = ["Elec2", "Covertype", "INSECTS\nabrupt", "INSECTS\ngradual", "INSECTS\nincr."]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(COL, 2.15))
    ax.bar(x - w, [obs[n] for n in names], w, color=BLUE, label="observed")
    ax.bar(x, [null.get(n, 0) for n in names], w, color=ORANGE, hatch="//",
           edgecolor="white", linewidth=0.4, label="no-drift Markov null")
    ax.bar(x + w, [shuf.get(n, np.nan) for n in names], w, color=GREY, hatch="xx",
           edgecolor="white", linewidth=0.4, label="block-shuffled")
    ax.set_yscale("log")                       # plain log: no stray zero tick
    ax.set_ylim(0.5, 3000)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_ylabel("ADWIN alarms (NeverAdapt)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              handlelength=1.1, columnspacing=0.9, borderpad=0.1)
    save(fig, "f2_alarm_validity")


# ---------------------------------------------------------------- F3 oracle branches
def f3_oracle_branches():
    d = load("results/analysis/phase10_per_stream.json")
    streams, hatches = ["elec2", "covtype"], {"SKIP": "", "NUDGE": "//", "REBUILD": "xx"}
    fig, ax = plt.subplots(figsize=(COL, 1.45))
    left = np.zeros(len(streams))
    small = []
    for b in ("SKIP", "NUDGE", "REBUILD"):
        vals = [d["per_stream"][s]["by_lambda"]["1000.0"]["advantage_by_branch_share"][b] * 100
                for s in streams]
        ax.barh(np.arange(len(streams)), vals, 0.5, left=left, color=BRANCH[b],
                hatch=hatches[b], edgecolor="white", linewidth=0.5, label=b)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 8:                      # white text on a solid patch of the same colour
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                        color="white", zorder=5,
                        bbox=dict(facecolor=BRANCH[b], edgecolor="none", pad=0.8))
            else:
                small.append((streams[i], b, v))
        left += np.array(vals)
    ax.set_yticks(np.arange(len(streams))); ax.set_yticklabels(["Elec2", "Covertype"])
    ax.set_xlabel("share of the oracle's advantage (%), $\\lambda=10^3$")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              handlelength=1.1, columnspacing=1.0, borderpad=0.1)
    save(fig, "f3_oracle_branches")
    print("   segments below 8% (for the caption):",
          ", ".join(f"{s} {b} {v:.1f}%" for s, b, v in small))


# ---------------------------------------------------------------- F4 cost, separate units
def f4_cost_split():
    runs = grid_runs(ROOT / "results/grid")
    pols = ["never_adapt", "always_nudge", "fixed_schedule", "mechanism_selector",
            "always_rebuild"]
    short = ["Never", "Nudge", "Fixed", "Select", "Rebuild"]
    fams = ["xgb", "rf", "sgd", "gnb"]
    famlab = ["XGB", "RF", "SGD", "GNB"]
    x = np.arange(len(fams)); w = 0.16

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 1.95),
                                 gridspec_kw={"wspace": 0.28})
    for i, pol in enumerate(pols):
        tr = [np.median([r["training"] for r in runs if r["model"] == f and r["policy"] == pol])
              for f in fams]
        inf = [np.mean([r["infer_per_pred"] for r in runs if r["model"] == f and r["policy"] == pol])
               for f in fams]
        off = (i - 2) * w
        a1.bar(x + off, np.maximum(tr, 1), w, color=POLICY[pol], label=short[i],
               edgecolor="white", linewidth=0.3)
        a2.bar(x + off, inf, w, color=POLICY[pol], edgecolor="white", linewidth=0.3)
    for ax, lab in ((a1, "training work units"), (a2, "inference ops per prediction")):
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(famlab)
        ax.set_ylabel(lab, fontsize=7.5)
    a1.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(1.14, 1.02),
              handlelength=1.0, columnspacing=1.2, borderpad=0.1)
    save(fig, "f4_cost_split")


# ---------------------------------------------------------------- F5 WaitAndCheck
def f5_waitandcheck():
    d = json.loads((ROOT / "paper" / "numbers.json").read_text(encoding="utf-8"))["tab3"]
    names = ["elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental"]
    labels = ["Elec2", "Covertype", "INSECTS\nabrupt", "INSECTS\ngradual", "INSECTS\nincr."]
    ratio = [d[f"{n}|wait_200"][0] for n in names]
    acc = [d[f"{n}|wait_200"][1] for n in names]
    colours = [LIGHT, LIGHT, ORANGE, ORANGE, ORANGE]
    x = np.arange(len(names))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.7), sharex=True,
                                 gridspec_kw={"hspace": 0.15})
    a1.bar(x, acc, 0.6, color=colours, edgecolor=BLACK, linewidth=0.4)
    a1.axhline(0, color=BLACK, linewidth=0.6)
    a1.axhline(-1, color=BLACK, linewidth=0.8, linestyle="--")
    a1.set_ylabel("median $\\Delta$ balanced\naccuracy (pp)", fontsize=7.2)

    a2.bar(x, ratio, 0.6, color=colours, edgecolor=BLACK, linewidth=0.4)
    a2.axhline(1, color=BLACK, linewidth=0.6)
    a2.axhline(0.75, color=BLACK, linewidth=0.8, linestyle="--")
    a2.set_ylabel("median training ops\n(ratio to selector)", fontsize=7.2)
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_ylim(0, 3.6)
    save(fig, "f5_waitandcheck")


# ---------------------------------------------------------------- F6 nudge failure
def f6_nudge_failure():
    ev = load("results/analysis/phase7.json")["event_split"]
    fams = ["xgb", "rf", "sgd", "gnb"]
    famlab = ["XGB", "RF", "SGD", "GNB"]
    share = [ev[f]["always_nudge_all"]["share_no_prediction_changed"] * 100 for f in fams]
    gain = [ev[f]["always_nudge_all"]["median_gain_pp"] for f in fams]
    n = [ev[f]["always_nudge_all"]["n"] for f in fams]
    x = np.arange(len(fams))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 1.9),
                                 gridspec_kw={"wspace": 0.22})
    a1.bar(x, share, 0.6, color=BLUE, edgecolor=BLACK, linewidth=0.4)
    a1.set_ylabel("no-change nudges (%)", fontsize=7.2)
    a2.bar(x, gain, 0.6, color=ORANGE, hatch="//", edgecolor=BLACK, linewidth=0.4)
    a2.axhline(0, color=BLACK, linewidth=0.6)
    a2.set_ylabel("median $\\Delta$acc (pp)", fontsize=7.2)
    for ax in (a1, a2):                     # family on one line, sample size beneath it
        ax.set_xticks(x)
        ax.set_xticklabels([l + chr(10) + "$n$=" + format(c, ",") for l, c in zip(famlab, n)],
                           fontsize=7)
    save(fig, "f6_nudge_failure")


# ---------------------------------------------------------------- F7 accuracy vs cost
def f7_accuracy_cost():
    nums = json.loads((ROOT / "paper" / "numbers.json").read_text(encoding="utf-8"))["main"]
    streams = ["elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental"]
    titles = ["Elec2", "Covertype", "INSECTS abrupt", "INSECTS gradual", "INSECTS incr."]
    pols = [("never_adapt", "Never", "o"), ("always_rebuild", "Rebuild", "s"),
            ("always_nudge", "Nudge", "^"), ("fixed_schedule", "Fixed", "D"),
            ("mechanism_selector", "Selector", "v"), ("waitandcheck_200", "Wait", "P"),
            ("confirmmd3_200", "MD3", "X")]
    colour = dict(POLICY, waitandcheck_200=POLICY["waitandcheck"],
                  confirmmd3_200=POLICY["confirmmd3"])

    fig, axes = plt.subplots(1, 5, figsize=(FULL, 1.9), gridspec_kw={"wspace": 0.42})
    for ax, s, title in zip(axes, streams, titles):
        for key, lab, mark in pols:
            acc, train = nums[f"{s}|{key}"]
            ax.scatter(max(train, 1e3), acc, s=22, marker=mark, color=colour[key],
                       edgecolor=BLACK, linewidth=0.3, label=lab, zorder=3)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("median balanced\naccuracy (%)", fontsize=7.2)
    fig.supxlabel("training work units, log", fontsize=7.5, y=-0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=7, loc="lower center",
               bbox_to_anchor=(0.5, 1.02), handlelength=1.0, columnspacing=1.2)
    save(fig, "f7_accuracy_cost")


if __name__ == "__main__":
    f1_pipeline()
    f2_alarm_validity()
    f3_oracle_branches()
    f4_cost_split()
    f5_waitandcheck()
    f6_nudge_failure()
    f7_accuracy_cost()
