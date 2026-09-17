"""Phase 7 analysis: statistics over the finished grid.

Everything here is a pure function of the grid CSV and event logs, so every
number in the paper can be regenerated and every rule below is unit-tested.

## Observations, not runs

XGBoost and GaussianNB use no randomness at these settings: their five seeds
reproduce one result (verified on the grid, and re-verified here before
anything is collapsed -- `verify_determinism` raises otherwise). Counting them
five times would be pseudo-replication, so the unit of analysis is:

| level | Random Forest, SGD | XGBoost, GaussianNB | total |
|---|---|---|---|
| `n60` (primary) | one block per (dataset, model, seed) | one block per (dataset, model) | 60 |
| `n20` (sensitivity) | one block per (dataset, model), seeds averaged | one block per (dataset, model) | 20 |

`n20` follows Demsar (2006): one observation per dataset. RF/SGD seeds within a
cell replay the same stream and share every drift point, so even `n60` is
optimistic; conclusions that do not survive `n20` must be reported as such.

## Metrics

- **accuracy**: mean prequential accuracy in the stream's own scorer (plain for
  binary streams, balanced for multi-class), in percent. Higher is better.
- **cost**: total adaptation work units. Lower is better. Statistics use raw
  work units: every test ranks within a block, so this is equivalent to "% of
  AlwaysRebuild" without dividing by zero where ADWIN never fired.

## Uncertainty

Only RF and SGD have seed variance. Deterministic cells are point values and are
never given error bars.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

POLICIES = ("always_rebuild", "mechanism_selector", "fixed_schedule", "always_nudge", "never_adapt")
LABELS = {
    "always_rebuild": "AlwaysRebuild",
    "mechanism_selector": "MechanismSelector",
    "fixed_schedule": "FixedSchedule",
    "always_nudge": "AlwaysNudge",
    "never_adapt": "NeverAdapt",
}
DETERMINISTIC_MODELS = ("xgb", "gnb")
STOCHASTIC_MODELS = ("rf", "sgd")
HIGHER_IS_BETTER = {"accuracy": True, "cost": False}

#: Studentised range statistic / sqrt(2) at alpha = 0.05, indexed by k (Demsar 2006, Table 5).
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

VOLATILE_FIELDS = {
    "seed", "total_wall_clock_s", "run_wall_clock_s", "total_energy_kwh",
    "initial_predict_us_per_row", "final_predict_us_per_row", "config_json",
}


# --- loading -------------------------------------------------------------------


def load_grid(grid_dir: Path) -> list[dict]:
    with (Path(grid_dir) / "grid.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def accuracy_of(row: dict) -> float:
    field = "mean_prequential_accuracy" if row["scorer"] == "accuracy" else "mean_prequential_balanced_accuracy"
    return 100.0 * float(row[field])


def cost_of(row: dict) -> float:
    return float(row["total_work_units"])


METRICS = {"accuracy": accuracy_of, "cost": cost_of}


def index_runs(rows: list[dict]) -> dict[tuple, dict]:
    return {(r["dataset"], r["model"], r["policy_key"], int(r["seed"])): r for r in rows}


def cells_of(rows: list[dict]) -> list[tuple[str, str]]:
    return sorted({(r["dataset"], r["model"]) for r in rows})


def seeds_of(rows: list[dict]) -> list[int]:
    return sorted({int(r["seed"]) for r in rows})


# --- determinism -----------------------------------------------------------------


class DeterminismViolation(AssertionError):
    """A model assumed deterministic produced different results across seeds."""


def verify_determinism(rows: list[dict], models=DETERMINISTIC_MODELS) -> int:
    """Raise unless every outcome field is identical across seeds for `models`.

    Returns the number of (dataset, model, policy) groups checked.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["model"] in models:
            groups[(r["dataset"], r["model"], r["policy_key"])].append(r)
    fields = [c for c in rows[0] if c not in VOLATILE_FIELDS]
    for key, runs in groups.items():
        for field in fields:
            if len({r[field] for r in runs}) > 1:
                raise DeterminismViolation(f"{key}: '{field}' differs across seeds")
    return len(groups)


# --- blocks ------------------------------------------------------------------------


@dataclass
class Blocks:
    """A blocks x policies matrix for one metric at one level of aggregation."""

    level: str
    metric: str
    block_ids: list[tuple]
    policies: tuple[str, ...]
    values: np.ndarray  # shape (n_blocks, n_policies)

    @property
    def n(self) -> int:
        return len(self.block_ids)


def build_blocks(rows: list[dict], metric: str, level: str = "n60",
                 policies=POLICIES, cells: list[tuple] | None = None) -> Blocks:
    """Assemble the observation matrix. See the module docstring for the levels."""
    if level not in ("n60", "n20"):
        raise ValueError(f"unknown level {level!r}")
    runs = index_runs(rows)
    value_of = METRICS[metric]
    seeds = seeds_of(rows)
    ids, matrix = [], []
    for dataset, model in (cells if cells is not None else cells_of(rows)):
        if model in DETERMINISTIC_MODELS:
            ids.append((dataset, model))
            matrix.append([value_of(runs[(dataset, model, p, seeds[0])]) for p in policies])
        elif level == "n60":
            for seed in seeds:
                ids.append((dataset, model, seed))
                matrix.append([value_of(runs[(dataset, model, p, seed)]) for p in policies])
        else:
            ids.append((dataset, model))
            matrix.append([float(np.mean([value_of(runs[(dataset, model, p, s)]) for s in seeds]))
                           for p in policies])
    return Blocks(level, metric, ids, tuple(policies), np.asarray(matrix, dtype=float))


# --- tests and effect sizes ------------------------------------------------------------


def mean_ranks(blocks: Blocks) -> dict[str, float]:
    """Mean within-block rank per policy; 1 = best."""
    sign = -1.0 if HIGHER_IS_BETTER[blocks.metric] else 1.0
    ranks = np.vstack([stats.rankdata(sign * row) for row in blocks.values])
    return {p: float(ranks[:, i].mean()) for i, p in enumerate(blocks.policies)}


def friedman(blocks: Blocks) -> dict:
    statistic, p = stats.friedmanchisquare(*blocks.values.T)
    return {"statistic": float(statistic), "p": float(p), "n": blocks.n, "k": len(blocks.policies)}


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Critical difference in mean rank (Demsar 2006)."""
    if alpha != 0.05:
        raise ValueError("only alpha = 0.05 is tabulated")
    return NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6.0 * n))


def nemenyi(blocks: Blocks):
    """Pairwise Nemenyi p-values as a policy-labelled DataFrame."""
    import scikit_posthocs as sp

    table = sp.posthoc_nemenyi_friedman(blocks.values)
    table.index = table.columns = [LABELS[p] for p in blocks.policies]
    return table


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b) over all pairs. Treats the samples as unpaired."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return float((greater - less) / (len(a) * len(b)))


def rank_biserial_paired(a, b) -> float:
    """Matched-pairs rank-biserial correlation: the effect size for a paired
    Wilcoxon. +1 means every non-zero difference favours `a`."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    positive, negative = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((positive - negative) / (positive + negative))


def paired_comparisons(blocks_by_metric: dict[str, Blocks], target: str, against: list[str]) -> list[dict]:
    """Wilcoxon signed-rank of `target` against each policy, per metric, Holm-adjusted
    across every test in the family."""
    results = []
    for metric, blocks in blocks_by_metric.items():
        t = blocks.values[:, blocks.policies.index(target)]
        for other in against:
            o = blocks.values[:, blocks.policies.index(other)]
            diff = t - o
            nonzero = int(np.count_nonzero(diff))
            p = float(stats.wilcoxon(t, o, zero_method="wilcox").pvalue) if nonzero else 1.0
            results.append({
                "metric": metric, "target": target, "against": other, "n": blocks.n,
                "n_nonzero": nonzero, "median_difference": float(np.median(diff)),
                "target_better_in": int(np.sum(diff > 0 if HIGHER_IS_BETTER[metric] else diff < 0)),
                "target_worse_in": int(np.sum(diff < 0 if HIGHER_IS_BETTER[metric] else diff > 0)),
                "p": p, "cliffs_delta": cliffs_delta(t, o), "rank_biserial": rank_biserial_paired(t, o),
            })
    for r, adj in zip(results, holm([r["p"] for r in results])):
        r["p_holm"] = adj
    return results


# --- Pareto ------------------------------------------------------------------------------


def pareto_frontier(points: dict[str, tuple[float, float]]) -> set[str]:
    """Non-dominated policies for (cost: lower better, accuracy: higher better).

    A dominates B when A costs no more, is no less accurate, and is strictly
    better on at least one of the two.
    """
    frontier = set()
    for name, (cost, acc) in points.items():
        dominated = any(
            c2 <= cost and a2 >= acc and (c2 < cost or a2 > acc)
            for other, (c2, a2) in points.items() if other != name
        )
        if not dominated:
            frontier.add(name)
    return frontier


def dominators(points: dict[str, tuple[float, float]], name: str) -> list[str]:
    cost, acc = points[name]
    return sorted(o for o, (c2, a2) in points.items()
                  if o != name and c2 <= cost and a2 >= acc and (c2 < cost or a2 > acc))


def cell_pareto(rows: list[dict], dataset: str, model: str, comparable_pp: float = 1.0) -> dict:
    """Per-cell cost/accuracy operating points and the selector's frontier status.

    Cost is shown as % of AlwaysRebuild's mean work units (ratio of means, defined
    whenever AlwaysRebuild adapted in any seed). For RF/SGD the frontier is also
    recomputed per seed, to show how stable the selector's status is.
    """
    runs = index_runs(rows)
    seeds = [seeds_of(rows)[0]] if model in DETERMINISTIC_MODELS else seeds_of(rows)

    def per_seed(policy, fn):
        return [fn(runs[(dataset, model, policy, s)]) for s in seeds]

    rebuild_mean = float(np.mean(per_seed("always_rebuild", cost_of)))
    points, detail = {}, {}
    for p in POLICIES:
        costs, accs = per_seed(p, cost_of), per_seed(p, accuracy_of)
        cost_pct = 100.0 * float(np.mean(costs)) / rebuild_mean if rebuild_mean else float("nan")
        points[p] = (float(np.mean(costs)), float(np.mean(accs)))
        detail[p] = {
            "cost_pct_of_rebuild": cost_pct, "accuracy": float(np.mean(accs)),
            "accuracy_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
            "cost_pct_sd": (float(np.std([100.0 * c / rebuild_mean for c in costs], ddof=1))
                            if len(costs) > 1 and rebuild_mean else None),
        }

    frontier = pareto_frontier(points)
    seed_frontier_share = None
    if len(seeds) > 1:
        on = 0
        for s in seeds:
            pts = {p: (cost_of(runs[(dataset, model, p, s)]), accuracy_of(runs[(dataset, model, p, s)]))
                   for p in POLICIES}
            on += "mechanism_selector" in pareto_frontier(pts)
        seed_frontier_share = on / len(seeds)

    sel, fs = points["mechanism_selector"], points["fixed_schedule"]
    acc_gap = sel[1] - fs[1]
    if sel[0] <= fs[0] and acc_gap >= 0:
        versus = "selector dominates FixedSchedule"
    elif fs[0] <= sel[0] and acc_gap <= 0:
        versus = "FixedSchedule dominates selector"
    elif abs(acc_gap) < comparable_pp:
        versus = ("selector cheaper, comparable accuracy" if sel[0] < fs[0]
                  else "FixedSchedule cheaper, comparable accuracy")
    elif sel[0] > fs[0]:
        versus = "different operating point: selector more accurate, costs more"
    else:
        versus = "different operating point: selector cheaper, less accurate"

    seed_consistency = None
    if len(seeds) > 1:
        diffs = [accuracy_of(runs[(dataset, model, "mechanism_selector", s)])
                 - accuracy_of(runs[(dataset, model, "fixed_schedule", s)]) for s in seeds]
        seed_consistency = max(sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)) / len(seeds)

    return {
        "dataset": dataset, "model": model, "n_seeds": len(seeds), "policies": detail,
        "frontier": sorted(frontier),
        "selector_on_frontier": "mechanism_selector" in frontier,
        "selector_dominated_by": dominators(points, "mechanism_selector"),
        "selector_frontier_share_across_seeds": seed_frontier_share,
        "selector_vs_fixed_schedule": versus,
        "accuracy_gap_vs_fixed_schedule_pp": acc_gap,
        "accuracy_gap_seed_sign_consistency": seed_consistency,
    }


# --- placeholders ---------------------------------------------------------------------------


def load_events(grid_dir: Path, dataset: str, model: str, policy: str, seeds) -> list[dict]:
    rows = []
    for seed in seeds:
        path = Path(grid_dir) / "events" / f"{dataset}__{model}__{policy}__{seed}.csv"
        with path.open(newline="", encoding="utf-8") as f:
            for e in csv.DictReader(f):
                e["seed"] = seed
                rows.append(e)
    return rows


def analysis_seeds(rows: list[dict], model: str) -> list[int]:
    seeds = seeds_of(rows)
    return [seeds[0]] if model in DETERMINISTIC_MODELS else seeds


def placeholder_exposure(rows: list[dict], grid_dir: Path) -> dict[tuple, dict]:
    """Per cell: share of adaptations, across all policies, that used placeholders,
    and whether initial training did."""
    exposure = {}
    for dataset, model in cells_of(rows):
        seeds = analysis_seeds(rows, model)
        events = [e for p in POLICIES for e in load_events(grid_dir, dataset, model, p, seeds)]
        affected = sum(e["placeholders_injected"] == "True" for e in events)
        initial = any(int(r["initial_placeholder_rows"]) > 0 for r in rows
                      if r["dataset"] == dataset and r["model"] == model)
        exposure[(dataset, model)] = {
            "adaptations": len(events), "with_placeholders": affected,
            "share": affected / len(events) if events else 0.0,
            "initial_training_affected": initial,
            "affected": affected > 0 or initial,
        }
    return exposure


# --- nudge-decay diagnostic -------------------------------------------------------------------


def _num(value) -> float | None:
    return None if value in (None, "", "None") else float(value)


def nudge_points(events: list[dict], stream_start: int, stream_rows: int) -> list[dict]:
    """NUDGE events as (position in stream 0..1, holdout gain pp, prediction change %, depth)."""
    out = []
    for e in events:
        if e["action"] != "NUDGE" or _num(e["acc_nudged"]) is None:
            continue
        out.append({
            "position": (int(e["adapt_row"]) - stream_start) / stream_rows,
            "gain": 100.0 * (_num(e["acc_nudged"]) - _num(e["acc_before"])),
            "prediction_change": 100.0 * _num(e["nudge_prediction_change"]),
            "depth": int(e["nudges_since_rebuild"]),
            "placeholders": e["placeholders_injected"] == "True",
        })
    return out


def matched_position_trend(accumulating: list[dict], resetting: list[dict], field: str,
                           n_bins: int = 5, min_bins: int = 3) -> float | None:
    """Does the accumulating policy fall further behind as the stream goes on?

    Bins stream position into `n_bins` equal-width bins. In each bin that both
    policies nudged in, takes accumulating-minus-resetting mean `field`. Returns
    Spearman's rho between bin index and that difference: negative means the
    accumulating policy's nudges get relatively worse at later positions, which
    position alone cannot explain because the resetting policy sees the same
    positions. None if fewer than `min_bins` bins are shared.
    """
    def by_bin(points):
        bins = defaultdict(list)
        for pt in points:
            bins[min(int(pt["position"] * n_bins), n_bins - 1)].append(pt[field])
        return {b: float(np.mean(v)) for b, v in bins.items()}

    a, r = by_bin(accumulating), by_bin(resetting)
    shared = sorted(set(a) & set(r))
    if len(shared) < min_bins:
        return None
    diffs = [a[b] - r[b] for b in shared]
    if len(set(diffs)) == 1:
        return 0.0
    return float(stats.spearmanr(shared, diffs).statistic)
