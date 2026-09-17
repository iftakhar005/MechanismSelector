"""Phase 7 figures, drawn from the grid and `results/analysis/phase7.json`.

    python experiments/make_figures.py

Writes PNGs to `figures/` and, for each, the plotted values as CSV to
`results/analysis/figure_data/` (the table view that the light-surface contrast
of some series requires).

Palette (validated with the dataviz checker, light surface #fcfcfb):
- Scatter / Pareto (all-pairs): three hues only -- MechanismSelector, FixedSchedule,
  AlwaysRebuild. AlwaysNudge and NeverAdapt are neutral grey, distinguished by
  marker shape and direct label.
- Lines and stacked bars (adjacent pairs): up to five hues in fixed slot order.
- One-quantity charts: a single hue, faceted, never colour-coded by category.
Only RF and SGD carry error bars; XGBoost and GaussianNB are deterministic.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analysis as an  # noqa: E402

warnings.filterwarnings("ignore")
GRID = ROOT / "results" / "grid"
ANALYSIS = ROOT / "results" / "analysis"
FIG = ROOT / "figures"
DATA = ANALYSIS / "figure_data"

SURFACE, INK, INK_2, GRIDLINE, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#8a8984"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

DATASETS = ["elec2", "covtype", "insects_abrupt", "insects_gradual", "insects_incremental"]
MODELS = ["xgb", "rf", "sgd", "gnb"]
MODEL_NAMES = {"xgb": "XGBoost", "rf": "Random Forest", "sgd": "SGD", "gnb": "Gaussian NB"}
DATASET_NAMES = {"elec2": "elec2", "covtype": "covtype", "insects_abrupt": "insects abrupt",
                 "insects_gradual": "insects gradual", "insects_incremental": "insects incremental"}
POLICY_STYLE = {  # colour, marker, short label
    "mechanism_selector": (BLUE, "o", "Selector"),
    "fixed_schedule": (ORANGE, "s", "FixedSched"),
    "always_rebuild": (AQUA, "D", "Rebuild"),
    "always_nudge": (NEUTRAL, "^", "Nudge"),
    "never_adapt": (NEUTRAL, "v", "Never"),
}
LINE_COLOURS = {"mechanism_selector": BLUE, "fixed_schedule": ORANGE, "always_rebuild": AQUA,
                "always_nudge": YELLOW, "never_adapt": MAGENTA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRIDLINE, "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.5,
})


def save(fig, name: str, table: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(DATA / f"{name}.csv", index=False)
    print(f"  {name}.png  ({len(table)} rows of data)")


def events(dataset, model, policy, seeds):
    return an.load_events(GRID, dataset, model, policy, seeds)


# --- 1. alarm direction (lead) ----------------------------------------------------------------


def fig_alarm_direction():
    cells = json.loads((GRID / "alarm_direction.json").read_text())["cells"]
    rows = [{"dataset": d, "model": m, "alarms": v["alarms"], "falling_error": v["error_down"],
             "share_on_falling_error": v["share_on_improvement"]}
            for key, v in cells.items() for d, m in [key.split("/")]]
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = table[table.model == model].set_index("dataset").reindex(DATASETS[::-1])
        y = np.arange(len(sub))
        share = 100 * sub.share_on_falling_error.fillna(0).to_numpy()
        ax.axvline(50, color=GRIDLINE, linewidth=1.2, zorder=0)
        ax.hlines(y, 0, share, color=BLUE, linewidth=1.5, zorder=2)
        ax.plot(share, y, "o", color=BLUE, markersize=6, zorder=3)
        for yi, s, n in zip(y, share, sub.alarms.to_numpy()):
            ax.annotate(f"{s:.0f}%  (n={n:,})", (s, yi), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK_2, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.1", facecolor=SURFACE, edgecolor="none"))
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 100)
        ax.set_title(MODEL_NAMES[model])
        ax.set_yticks(y, [DATASET_NAMES[d] for d in sub.index])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("% of alarms on falling error")
    fig.suptitle("ADWIN alarms fired while the error rate was falling, model unchanged (NeverAdapt replay)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_alarm_direction", table)


# --- 2. nudge failure modes --------------------------------------------------------------------


def fig_failure_modes(rows):
    records = []
    for model in MODELS:
        seeds = an.analysis_seeds(rows, model)
        for dataset in DATASETS:
            for e in events(dataset, model, "always_nudge", seeds):
                if e["acc_nudged"]:
                    records.append({"model": model, "dataset": dataset, "seed": e["seed"],
                                    "prediction_change_pct": 100 * float(e["nudge_prediction_change"]),
                                    "accuracy_gain_pp": 100 * (float(e["acc_nudged"]) - float(e["acc_before"]))})
    table = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6))
    for col, model in enumerate(MODELS):
        sub = table[table.model == model]
        for row, (field, xlabel, bins) in enumerate([
            ("prediction_change_pct", "holdout predictions changed by the nudge (%)", np.linspace(0, 100, 41)),
            ("accuracy_gain_pp", "holdout accuracy change (points)", np.linspace(-100, 100, 41)),
        ]):
            ax = axes[row, col]
            values = sub[field]
            weights = np.full(len(values), 100 / max(len(values), 1))
            ax.hist(values, bins=bins, weights=weights, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
            zero = 100 * np.mean(sub[field] == 0)
            ax.text(0.97, 0.93, f"exactly 0: {zero:.0f}%\nmedian: {sub[field].median():+.1f}\nn = {len(sub):,}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK_2)
            ax.set_xlabel(xlabel, fontsize=7.5)
            if col == 0:
                ax.set_ylabel("% of nudges")
            if row == 0:
                ax.set_title(MODEL_NAMES[model])
    fig.suptitle("What one nudge does, per model family (AlwaysNudge, every alarm)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_nudge_failure_modes", table)


# --- 3. Pareto per cell ---------------------------------------------------------------------------


def seed_ranges(rows, dataset, model):
    """Min-max over seeds for RF/SGD (cost as % of AlwaysRebuild's mean). Empty for
    deterministic models, which have no seed variation to show."""
    if model in an.DETERMINISTIC_MODELS:
        return {}
    runs = an.index_runs(rows)
    seeds = an.seeds_of(rows)
    rebuild = np.mean([an.cost_of(runs[(dataset, model, "always_rebuild", s)]) for s in seeds])
    out = {}
    for policy in an.POLICIES:
        costs = [100 * an.cost_of(runs[(dataset, model, policy, s)]) / rebuild for s in seeds]
        accs = [an.accuracy_of(runs[(dataset, model, policy, s)]) for s in seeds]
        out[policy] = {"cost_min": min(costs), "cost_max": max(costs), "acc_min": min(accs), "acc_max": max(accs)}
    return out


def fig_pareto(rows):
    pareto = json.loads((ANALYSIS / "phase7.json").read_text())["pareto"]
    by_cell = {(c["dataset"], c["model"]): c for c in pareto}
    records = []
    fig, axes = plt.subplots(len(DATASETS), len(MODELS), figsize=(11, 13))
    for r, dataset in enumerate(DATASETS):
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            cell = by_cell[(dataset, model)]
            ranges = seed_ranges(rows, dataset, model)
            frontier = [(v["cost_pct_of_rebuild"], v["accuracy"]) for p, v in cell["policies"].items()
                        if p in cell["frontier"]]
            fx, fy = zip(*sorted(frontier))
            ax.step(fx, fy, where="post", color=INK_2, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
            for policy, v in cell["policies"].items():
                colour, marker, label = POLICY_STYLE[policy]
                rng = ranges.get(policy)
                xerr = yerr = None
                if rng:
                    x, yv = v["cost_pct_of_rebuild"], v["accuracy"]
                    xerr = [[x - rng["cost_min"]], [rng["cost_max"] - x]]
                    yerr = [[yv - rng["acc_min"]], [rng["acc_max"] - yv]]
                ax.errorbar(v["cost_pct_of_rebuild"], v["accuracy"], xerr=xerr, yerr=yerr,
                            fmt=marker, color=colour, markersize=6.5, markeredgecolor=SURFACE,
                            markeredgewidth=1.0, elinewidth=0.9, capsize=0, zorder=3,
                            label=an.LABELS[policy])
                records.append({"dataset": dataset, "model": model, "policy": an.LABELS[policy],
                                "cost_pct_of_rebuild": v["cost_pct_of_rebuild"], "cost_pct_sd": v["cost_pct_sd"],
                                "accuracy": v["accuracy"], "accuracy_sd": v["accuracy_sd"],
                                "on_frontier": policy in cell["frontier"]})
            status = "selector ON frontier" if cell["selector_on_frontier"] else "selector dominated"
            share = cell["selector_frontier_share_across_seeds"]
            if share is not None:
                status += f" · frontier {round(share * cell['n_seeds'])}/{cell['n_seeds']} seeds"
            ax.set_title(f"{DATASET_NAMES[dataset]} · {MODEL_NAMES[model]}\n{status}", fontsize=7.5)
            if c == 0:
                ax.set_ylabel("mean prequential accuracy (%)")
            if r == len(DATASETS) - 1:
                ax.set_xlabel("adaptation cost (% of AlwaysRebuild)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.985), fontsize=8,
               handletextpad=0.3, columnspacing=1.4)
    fig.suptitle("Cost-accuracy operating points per cell; dotted step = Pareto frontier; "
                 "bars (RF, SGD only) = min-max over 5 seeds",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "fig3_pareto_per_cell", pd.DataFrame(records))


# --- 4. selector actions -----------------------------------------------------------------------------


def fig_actions(rows):
    runs = an.index_runs(rows)
    records = []
    for dataset in DATASETS:
        for model in MODELS:
            seeds = an.analysis_seeds(rows, model)
            sel = [runs[(dataset, model, "mechanism_selector", s)] for s in seeds]
            mean = lambda f: float(np.mean([int(r[f]) for r in sel]))  # noqa: E731
            degraded = mean("n_degraded")
            records.append({"dataset": dataset, "model": model, "SKIP": mean("n_skip"), "NUDGE": mean("n_nudge"),
                            "REBUILD after failed nudge": mean("n_rebuild") - degraded,
                            "REBUILD, window too small": degraded})
    table = pd.DataFrame(records)
    parts = [("SKIP", BLUE), ("NUDGE", ORANGE), ("REBUILD after failed nudge", AQUA),
             ("REBUILD, window too small", YELLOW)]

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=False)
    for ax, model in zip(axes, MODELS):
        sub = table[table.model == model].set_index("dataset").reindex(DATASETS)
        totals = sub[[p for p, _ in parts]].sum(axis=1).to_numpy()
        left = np.zeros(len(sub))
        y = np.arange(len(sub))[::-1]
        for part, colour in parts:
            share = 100 * sub[part].to_numpy() / np.where(totals == 0, 1, totals)
            ax.barh(y, share, left=left, color=colour, height=0.62, edgecolor=SURFACE, linewidth=1.0, label=part)
            left += share
        for yi, total in zip(y, totals):
            ax.text(101, yi, f"{total:.1f}" if total < 1 else f"{total:.0f}", va="center", fontsize=7, color=INK_2)
        ax.set_yticks(y, [DATASET_NAMES[d] for d in sub.index])
        ax.set_xlim(0, 112)
        ax.set_title(MODEL_NAMES[model])
        ax.set_xlabel("% of selector adaptations")
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle("What the selector did at each alarm (numbers: adaptations per run)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_selector_actions", table)


# --- 5. cumulative cost ------------------------------------------------------------------------------


def fig_cumulative_cost(rows):
    manifest = {m["name"]: m["n_rows"] for m in json.loads((ROOT / "results" / "dataset_manifest.json").read_text())}
    runs = an.index_runs(rows)
    records = []
    fig, axes = plt.subplots(len(DATASETS), len(MODELS), figsize=(11, 12))
    for r, dataset in enumerate(DATASETS):
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            n_stream = int(runs[(dataset, model, "always_rebuild", 0)]["n_stream_rows"])
            start = manifest[dataset] - n_stream
            final_rebuild = float(runs[(dataset, model, "always_rebuild", 0)]["total_work_units"]) or 1.0
            for policy in an.POLICIES:
                ev = events(dataset, model, policy, [0])
                x = [0.0] + [100 * (int(e["adapt_row"]) - start) / n_stream for e in ev] + [100.0]
                cum = [0.0] + [100 * float(e["cumulative_work_units"]) / final_rebuild for e in ev]
                cum.append(cum[-1])
                ax.step(x, cum, where="post", color=LINE_COLOURS[policy], label=an.LABELS[policy])
                records += [{"dataset": dataset, "model": model, "policy": an.LABELS[policy],
                             "stream_position_pct": xi, "cumulative_cost_pct_of_rebuild_total": yi}
                            for xi, yi in zip(x, cum)]
            ax.set_title(f"{DATASET_NAMES[dataset]} · {MODEL_NAMES[model]}", fontsize=7.5)
            if not float(runs[(dataset, model, "always_rebuild", 0)]["total_work_units"]):
                ax.cla()
                ax.text(0.5, 0.5, "no ADWIN alarms in seed 0:\nno policy adapted", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color=INK_2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{DATASET_NAMES[dataset]} · {MODEL_NAMES[model]}", fontsize=7.5)
            if c == 0:
                ax.set_ylabel("cumulative cost\n(% of AlwaysRebuild total)")
            if r == len(DATASETS) - 1:
                ax.set_xlabel("position in stream (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.suptitle("Cumulative adaptation cost along the stream (seed 0)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig5_cumulative_cost", pd.DataFrame(records))


# --- 6. nudge success by drift type -----------------------------------------------------------------


def fig_nudge_success(rows):
    runs = an.index_runs(rows)
    records = []
    for dataset in DATASETS:
        for model in MODELS:
            seeds = an.analysis_seeds(rows, model)
            sel = [runs[(dataset, model, "mechanism_selector", s)] for s in seeds]
            attempts = sum(int(r["n_nudge_attempts"]) for r in sel)
            wins = sum(int(r["n_nudge"]) for r in sel)
            records.append({"dataset": dataset, "model": model, "attempts": attempts, "successes": wins,
                            "success_rate": wins / attempts if attempts else None})
    table = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = table[table.model == model].set_index("dataset").reindex(DATASETS[::-1])
        y = np.arange(len(sub))
        rate = 100 * sub.success_rate.fillna(0).to_numpy()
        ax.hlines(y, 0, rate, color=BLUE, linewidth=1.5)
        ax.plot(rate, y, "o", color=BLUE, markersize=6)
        for yi, rt, s, a in zip(y, rate, sub.successes.to_numpy(), sub.attempts.to_numpy()):
            ax.text(min(rt + 3, 70), yi, f"{s}/{a}" if a else "no attempts", va="center", fontsize=7.5, color=INK_2)
        ax.set_xlim(0, 100)
        ax.set_yticks(y, [DATASET_NAMES[d] for d in sub.index])
        ax.set_title(MODEL_NAMES[model])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Nudge success rate by stream (drift type); labels = successes / attempts, all seeds",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.supxlabel("selector nudges that cleared the accuracy floor (%)", fontsize=9, color=INK_2)
    fig.tight_layout()
    save(fig, "fig6_nudge_success_by_drift", table)


# --- 7. critical-difference diagrams -------------------------------------------------------------------


def fig_cd():
    import scikit_posthocs as sp

    stats_all = json.loads((ANALYSIS / "phase7.json").read_text())["statistics"]["all"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 4.4))
    records = []
    for r, level in enumerate(("n60", "n20")):
        for c, metric in enumerate(("accuracy", "cost")):
            ax = axes[r, c]
            m = stats_all[level]["metrics"][metric]
            ranks = pd.Series(m["mean_ranks"])
            sig = pd.DataFrame(m["nemenyi_p"])
            sp.critical_difference_diagram(
                ranks, sig, ax=ax,
                label_props={"fontsize": 8},
                marker_props={"color": INK_2, "s": 22},
                elbow_props={"color": INK_2, "linewidth": 0.8},
                crossbar_props={"color": BLUE, "linewidth": 2.0},
            )
            ax.set_title(f"{metric}, {level} (n={stats_all[level]['n']}): Friedman p = "
                         f"{m['friedman']['p']:.2g}, CD = {m['critical_difference']:.2f}", fontsize=8)
            for policy, rank in ranks.items():
                records.append({"level": level, "metric": metric, "policy": policy, "mean_rank": rank,
                                "friedman_p": m["friedman"]["p"], "critical_difference": m["critical_difference"]})
    fig.suptitle("Mean rank per policy (1 = best); bars join policies not significantly different (Nemenyi, alpha 0.05)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_critical_difference", pd.DataFrame(records))


# --- 8. decay diagnostic ---------------------------------------------------------------------------------


def fig_decay():
    decay = json.loads((ANALYSIS / "phase7.json").read_text())["decay"]
    records = [{"model": model, "dataset": run["dataset"], "seed": run["seed"],
                "trend_gain": run["trend_gain"], "trend_prediction_change": run["trend_prediction_change"]}
               for model, v in decay.items() for run in v["runs"]]
    table = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
    for ax, (field, label) in zip(axes, [("trend_gain", "accuracy gain"),
                                         ("trend_prediction_change", "prediction change")]):
        for i, model in enumerate(MODELS):
            vals = table[(table.model == model)][field].dropna().to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
            ax.plot(vals, i + jitter, "o", color=BLUE, markersize=5, alpha=0.8,
                    markeredgecolor=SURFACE, markeredgewidth=0.6)
            if len(vals):
                ax.plot([np.median(vals)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2.0)
            ax.text(1.05, i, f"n={len(vals)}", va="center", fontsize=7.5, color=INK_2)
        ax.axvline(0, color=INK_2, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_xlim(-1.05, 1.25)
        ax.set_yticks(range(len(MODELS)), [MODEL_NAMES[m] for m in MODELS])
        ax.set_xlabel(f"trend in AlwaysNudge minus FixedSchedule\n{label} along the stream (Spearman rho)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Nudge decay with accumulation: one dot per run, bar = median (< 0 would mean decay)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig8_nudge_decay_diagnostic", table)


def main() -> int:
    rows = an.load_grid(GRID)
    print("writing figures:")
    fig_alarm_direction()
    fig_failure_modes(rows)
    fig_pareto(rows)
    fig_actions(rows)
    fig_cumulative_cost(rows)
    fig_nudge_success(rows)
    fig_cd()
    fig_decay()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
