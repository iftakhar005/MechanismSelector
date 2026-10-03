"""Figures for the paper, from committed results only.

Plain style: greyscale plus one accent, no titles inside the axes, sized for a
single IEEE column (3.4 in) or the full width (7.0 in). Each figure is written
as PDF (vector, for LaTeX) and PNG at 300 dpi (for slides).
"""

from __future__ import annotations

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

INK, MID, LIGHT, ACCENT = "#000000", "#777777", "#cccccc", "#b45309"
COL, FULL = 3.4, 7.0

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "figure.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "font.family": "serif",
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", name)


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- F1 decision flow
def f1_pipeline():
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.6); ax.axis("off")

    def box(x, y, w, h, text, fill="white", edge=INK, lw=0.9, fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=fill, edgecolor=edge, linewidth=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, label=None, dx=0.12):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     color=INK, linewidth=0.8, shrinkA=0, shrinkB=0))
        if label:
            ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2, label, fontsize=6.4, color=MID,
                    ha="left", va="center")

    box(0.2, 6.6, 9.6, 0.8, "stream row: predict, score, buffer (cap 1000), feed ADWIN", fill=LIGHT)
    arrow(5.0, 6.6, 5.0, 6.0, "alarm, buffer $\\geq$ 100")
    box(1.4, 5.1, 7.2, 0.85, "split window by time: older 80% train / newest 20% holdout")
    arrow(5.0, 5.1, 5.0, 4.6)
    box(1.4, 3.7, 7.2, 0.85, "window $<$ 250 rows?")
    arrow(8.6, 4.1, 9.4, 4.1); ax.text(9.5, 4.1, "REBUILD\n(guard)", fontsize=6.4, va="center")
    arrow(5.0, 3.7, 5.0, 3.2)
    box(1.4, 2.3, 7.2, 0.85, "model $\\geq$ floor (reference $-$ 2 pp)?")
    arrow(8.6, 2.7, 9.4, 2.7); ax.text(9.5, 2.7, "SKIP", fontsize=6.4, va="center")
    arrow(5.0, 2.3, 5.0, 1.8)
    box(1.4, 0.9, 7.2, 0.85, "nudge a copy on train part; nudge $\\geq$ floor?")
    arrow(8.6, 1.3, 9.4, 1.3); ax.text(9.5, 1.3, "NUDGE", fontsize=6.4, va="center")
    arrow(5.0, 0.9, 5.0, 0.45); ax.text(5.15, 0.3, "REBUILD (pays for the failed nudge)", fontsize=6.4)
    save(fig, "f1_decision_flow")


# ---------------------------------------------------------------- F2 alarm validity
def f2_alarm_validity():
    direction = load("results/analysis/phase9_direction_null.json")
    shuffle = load("results/analysis/phase9_block_shuffle.json")

    obs, null = {}, {}
    for c in direction["observed_per_cell"]:
        obs[c["dataset"]] = obs.get(c["dataset"], 0) + c["alarms"]
    for c in direction["null_per_cell"]:
        null[c["dataset"]] = null.get(c["dataset"], 0) + c["nulls"]["ar1"]["alarms_mean"]
    shuf = {}
    for c in shuffle["per_cell"]:
        shuf.setdefault(c["dataset"], [0, 0])
        shuf[c["dataset"]][0] += c["alarms"]["adwin"]["original"]
        shuf[c["dataset"]][1] += c["alarms"]["adwin"]["block_shuffled"]

    names = ["elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental"]
    labels = ["Elec2", "Covertype", "INSECTS\nabrupt", "INSECTS\ngradual", "INSECTS\nincr."]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ax.bar(x - w, [obs[n] for n in names], w, color=INK, label="observed")
    ax.bar(x, [null.get(n, 0) for n in names], w, color=MID, label="no-drift AR(1) null")
    ax.bar(x + w, [shuf[n][1] if n in shuf else np.nan for n in names], w,
           color="white", edgecolor=INK, hatch="///", linewidth=0.7, label="block-shuffled")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("ADWIN alarms (NeverAdapt)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    save(fig, "f2_alarm_validity")


# ---------------------------------------------------------------- F3 oracle branches
def f3_oracle_branches():
    d = load("results/analysis/phase10_per_stream.json")
    streams = ["elec2", "covtype"]
    branches = ["SKIP", "NUDGE", "REBUILD"]
    fig, ax = plt.subplots(figsize=(COL, 1.8))
    left = np.zeros(len(streams))
    shades = [INK, MID, LIGHT]
    for b, sh in zip(branches, shades):
        vals = [d["per_stream"][s]["by_lambda"]["1000.0"]["advantage_by_branch_share"][b] * 100
                for s in streams]
        ax.barh(streams, vals, left=left, color=sh, edgecolor="white", linewidth=0.6, label=b)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 4:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=6.6,
                        color="white" if sh == INK else "black")
        left += np.array(vals)
    ax.set_xlabel("share of the oracle's advantage over the selector (%), $\\lambda=10^3$")
    ax.set_yticklabels(["Elec2", "Covertype"])
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.1, columnspacing=1.2)
    save(fig, "f3_oracle_branches")


# ---------------------------------------------------------------- F4 training vs inference
def f4_cost_split():
    d = load("results/analysis/phase9_inference_cost.json")["per_run"]
    pols = ["never_adapt", "always_nudge", "fixed_schedule", "mechanism_selector", "always_rebuild"]
    short = ["Never", "Nudge", "Fixed", "Selector", "Rebuild"]
    fams = ["xgb", "rf", "sgd", "gnb"]
    famlab = ["XGBoost", "RandomForest", "SGD", "GaussianNB"]

    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.7), sharey=True)
    for ax, fam, lab in zip(axes, fams, famlab):
        shares = []
        for pol in pols:
            rows = [r for r in d if r["model"] == fam and r["policy"] == pol]
            tr = sum(r["training_work_units"] for r in rows)
            inf = sum(r["inference_units_total"] for r in rows)
            shares.append(100 * tr / (tr + inf) if (tr + inf) else 0)
        ax.bar(range(len(pols)), shares, color=INK, width=0.65)
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels(short, rotation=45, ha="right")
        ax.set_xlabel(lab)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("training share of\noperations (%)")
    save(fig, "f4_cost_split")


# ---------------------------------------------------------------- F5 WaitAndCheck
def f5_waitandcheck():
    d = load("results/analysis/phase14_waitandcheck.json")["verdict_inputs"]["200"]
    names = ["elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental"]
    labels = ["Elec2*", "Covertype*", "INSECTS\nabrupt", "INSECTS\ngradual", "INSECTS\nincr."]
    acc = [d[n]["accuracy_delta_pp"] for n in names]
    ratio = [d[n]["median_training_ratio"] for n in names]
    x = np.arange(len(names))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 2.8), sharex=True,
                                 gridspec_kw={"hspace": 0.18})
    a1.bar(x, acc, 0.6, color=[MID if n in ("elec2", "covtype") else INK for n in names])
    a1.axhline(0, color="#444444", linewidth=0.7)
    a1.axhline(-1, color=ACCENT, linewidth=0.8, linestyle="--")
    a1.text(-0.45, -1.0, "$-1$ pp rule", fontsize=6.2, color=ACCENT, va="bottom", ha="left")
    a1.set_ylabel("$\\Delta$ balanced\naccuracy (pp)")

    a2.bar(x, ratio, 0.6, color=[MID if n in ("elec2", "covtype") else INK for n in names])
    a2.axhline(1, color="#444444", linewidth=0.7)
    a2.axhline(0.75, color=ACCENT, linewidth=0.8, linestyle="--")
    a2.text(4.45, 0.78, "$0.75\\times$ rule", fontsize=6.2, color=ACCENT, va="bottom", ha="right")
    a2.set_ylabel("training ops\n(ratio to selector)")
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_ylim(0, 3.5)
    for i, v in enumerate(ratio):
        if v > 3.4:
            a2.text(i, 3.3, f"{v:.1f}", ha="center", fontsize=6.2)
    save(fig, "f5_waitandcheck")


if __name__ == "__main__":
    f1_pipeline()
    f2_alarm_validity()
    f3_oracle_branches()
    f4_cost_split()
    f5_waitandcheck()
