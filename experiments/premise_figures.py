"""Figures for whether drift-triggered adaptation pays, from `detector_premise.json`.

    python experiments/premise_figures.py

Uses the palette and conventions of `make_figures.py` (validated palettes, single
hue for one-quantity charts, data CSV beside every figure).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import make_figures as mf  # noqa: E402  (sets rcParams; functions are not run)
from make_figures import BLUE, INK, INK_2, ORANGE, SURFACE, plt, save  # noqa: E402

PREMISE = json.loads((ROOT / "results" / "analysis" / "detector_premise.json").read_text())
ADAPTIVE = ("always_rebuild", "fixed_schedule", "always_nudge", "mechanism_selector")
LABELS = {"always_rebuild": "AlwaysRebuild", "fixed_schedule": "FixedSchedule",
          "always_nudge": "AlwaysNudge", "mechanism_selector": "MechanismSelector"}
ORDER = [f"{d}/{m}" for d in mf.DATASETS for m in mf.MODELS]


def pretty(cell: str) -> str:
    d, m = cell.split("/")
    return f"{mf.DATASET_NAMES[d]} · {mf.MODEL_NAMES[m]}"


def fig_never_adapt():
    dom = PREMISE["never_adapt_dominance"]
    records = []
    fig, axes = plt.subplots(1, 4, figsize=(12, 6.4), sharey=True)
    y_of = {cell: len(ORDER) - 1 - i for i, cell in enumerate(ORDER)}
    for ax, policy in zip(axes, ADAPTIVE):
        v = dom[policy]
        robust = set(v["cells_dominated_robust"])
        for c in v["cells"]:
            y = y_of[c["cell"]]
            x = c["never_minus_policy_pp"]
            if c["seed_min_pp"] is not None:
                ax.hlines(y, c["seed_min_pp"], c["seed_max_pp"], color=BLUE, linewidth=1.0, alpha=0.55)
            filled = c["cell"] in robust
            ax.plot(x, y, "o", markersize=6, color=BLUE, markerfacecolor=BLUE if filled else SURFACE,
                    markeredgewidth=1.3, zorder=3)
            records.append({"policy": LABELS[policy], "cell": c["cell"], **{k: c[k] for k in (
                "never_minus_policy_pp", "seed_min_pp", "seed_max_pp", "never_better_in_seeds", "n_seeds",
                "never_adapt_dominates")}, "robust_dominance": filled})
        ax.axvline(0, color=INK_2, linewidth=0.8)
        for b in range(1, len(mf.DATASETS)):
            ax.axhline(b * len(mf.MODELS) - 0.5, color=mf.GRIDLINE, linewidth=0.8)
        ax.set_title(f"vs {LABELS[policy]}\ndominated in {v['n_cells_dominated']}/20 · robust {v['n_cells_dominated_robust']}",
                     fontsize=8.5)
        ax.set_xlabel("NeverAdapt minus policy\naccuracy (points)")
        ax.set_xlim(-45, 45)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.text(22, len(ORDER) - 0.2, "NeverAdapt better →", ha="center", fontsize=7, color=INK_2)
        ax.text(-22, len(ORDER) - 0.2, "← adaptation better", ha="center", fontsize=7, color=INK_2)
    axes[0].set_yticks([y_of[c] for c in ORDER], [pretty(c) for c in ORDER], fontsize=7.5)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=BLUE, markerfacecolor=BLUE, label="robust: ≥1 point, and ≥4/5 seeds for RF/SGD"),
               plt.Line2D([], [], marker="o", linestyle="", color=BLUE, markerfacecolor=SURFACE, label="not robust"),
               plt.Line2D([], [], color=BLUE, alpha=0.55, label="range over 5 seeds (RF, SGD)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.suptitle("Never adapting vs each adaptive policy, per cell: accuracy difference at zero cost",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig9_never_adapt_vs_adaptive", pd.DataFrame(records))


def fig_alarm_quality():
    aq = PREMISE["alarm_quality_vs_payoff"]
    cells = pd.DataFrame(aq["cells"])
    streams = pd.DataFrame(aq["streams"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))

    left.axhline(0, color=INK_2, linewidth=0.8)
    left.plot(100 * cells.share_falling, cells.rebuild_gain_over_never_pp, "o", color=BLUE, markersize=6,
              markeredgecolor=SURFACE, markeredgewidth=0.8)
    left.set_title(f"20 cells (not independent: 5 streams)\nSpearman ρ = {aq['cell_level']['spearman_rho']:+.2f}, "
                   f"p = {aq['cell_level']['p']:.2g}", fontsize=8.5)
    left.set_xlabel("NeverAdapt alarms fired on falling error (%)")
    left.set_ylabel("AlwaysRebuild accuracy gain\nover NeverAdapt (points)")

    right.axhline(0, color=INK_2, linewidth=0.8)
    right.plot(100 * streams.share_falling, streams.mean_rebuild_gain_pp, "o", color=BLUE, markersize=8,
               markeredgecolor=SURFACE, markeredgewidth=1.0)
    for _, s in streams.iterrows():
        right.annotate(mf.DATASET_NAMES[s.dataset], (100 * s.share_falling, s.mean_rebuild_gain_pp),
                       xytext=(7, 0 if s.dataset != "insects_abrupt" else -9), textcoords="offset points",
                       va="center", fontsize=8, color=INK)
    right.set_title(f"5 streams (alarm-weighted share; mean gain over models)\nSpearman ρ = "
                    f"{aq['stream_level']['spearman_rho']:+.2f}, p = {aq['stream_level']['p']:.2g}  — only 5 points",
                    fontsize=8.5)
    right.set_xlabel("NeverAdapt alarms fired on falling error (%)")
    right.set_xlim(10, 60)
    fig.suptitle("AlwaysRebuild's accuracy gain over NeverAdapt against the share of alarms fired on falling error",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig10_alarm_quality_vs_payoff",
         pd.concat([cells.assign(level="cell"), streams.assign(level="stream")], ignore_index=True))


def fig_cost_on_falling():
    ad = PREMISE["adaptive_cost_on_falling_error"]
    if not ad:
        print("  fig11 skipped: direction-logged runs not available")
        return
    runs = pd.DataFrame(ad["per_run"])
    deterministic = runs.model.isin(["xgb", "gnb"])
    runs = runs[~deterministic | (runs.seed == 0)]
    groups = [("all streams", runs)] + [(mf.DATASET_NAMES[d], runs[runs.dataset == d]) for d in mf.DATASETS]
    policies = ["always_rebuild", "fixed_schedule", "mechanism_selector"]
    records = []
    fig, axes = plt.subplots(1, len(groups), figsize=(12, 2.9), sharey=True)
    for ax, (name, sub) in zip(axes, groups):
        for i, policy in enumerate(policies):
            p = sub[sub.policy == policy]
            n_adapt, n_fall = p.adaptations.sum(), p.on_falling_error.sum()
            work, work_fall = p.work_units.sum(), p.work_units_on_falling_error.sum()
            share_a = 100 * n_fall / n_adapt if n_adapt else np.nan
            share_w = 100 * work_fall / work if work else np.nan
            y = len(policies) - 1 - i
            ax.plot([share_a], [y + 0.12], "o", color=BLUE, markersize=6, label="adaptations" if i == 0 else None)
            ax.plot([share_w], [y - 0.12], "s", color=ORANGE, markersize=6, label="work units" if i == 0 else None)
            records.append({"group": name, "policy": LABELS[policy], "adaptations": int(n_adapt),
                            "on_falling_error": int(n_fall), "share_adaptations_pct": share_a,
                            "share_work_units_pct": share_w})
        ax.set_title(name, fontsize=8.5)
        ax.set_xlim(-3, 80)
        ax.set_yticks(range(len(policies)), [LABELS[p] for p in policies[::-1]], fontsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.supxlabel("% triggered by alarms that fired on a falling error rate", fontsize=9, color=INK_2, y=0.02)
    fig.suptitle("Adaptive policies: share of adaptations, and of their cost, spent on falling-error alarms",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig11_cost_on_falling_error_alarms", pd.DataFrame(records))


def main() -> int:
    print("writing premise figures:")
    fig_never_adapt()
    fig_alarm_quality()
    fig_cost_on_falling()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
