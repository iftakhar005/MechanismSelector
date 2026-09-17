"""Tests for Phase 7 analysis: every statistical rule the paper relies on."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import (  # noqa: E402
    POLICIES,
    DeterminismViolation,
    build_blocks,
    cell_pareto,
    cliffs_delta,
    dominators,
    holm,
    matched_position_trend,
    mean_ranks,
    nemenyi_cd,
    pareto_frontier,
    rank_biserial_paired,
    verify_determinism,
)


def synthetic_grid(deterministic_noise: float = 0.0):
    """2 datasets x 4 models x 5 policies x 5 seeds, with known structure."""
    rows = []
    rng = np.random.default_rng(0)
    for d, dataset in enumerate(("elec2", "covtype")):
        for model in ("xgb", "rf", "sgd", "gnb"):
            for p, policy in enumerate(POLICIES):
                for seed in range(5):
                    noise = 0.0
                    if model in ("rf", "sgd"):
                        noise = rng.normal(0, 0.5)
                    elif deterministic_noise and seed == 3:
                        noise = deterministic_noise
                    rows.append({
                        "dataset": dataset, "model": model, "policy_key": policy, "seed": str(seed),
                        "scorer": "accuracy",
                        "mean_prequential_accuracy": str((70 + 2 * p + d + noise) / 100),
                        "mean_prequential_balanced_accuracy": "0.5",
                        "total_work_units": str(1000 * (5 - p) + int(10 * noise)),
                        "initial_placeholder_rows": "0",
                        "total_wall_clock_s": str(rng.random()),
                    })
    return rows


# --- determinism and blocks -------------------------------------------------------------


def test_determinism_passes_when_deterministic_models_repeat_exactly():
    assert verify_determinism(synthetic_grid()) == 2 * 2 * 5


def test_determinism_violation_stops_the_analysis():
    with pytest.raises(DeterminismViolation):
        verify_determinism(synthetic_grid(deterministic_noise=1.0))


def test_primary_level_collapses_deterministic_cells_only():
    rows = synthetic_grid()
    blocks = build_blocks(rows, "accuracy", "n60")
    # per dataset: rf and sgd x 5 seeds = 10, xgb and gnb collapsed = 2
    assert blocks.n == 2 * (10 + 2)
    assert blocks.values.shape == (24, 5)


def test_full_grid_shape_gives_sixty_and_twenty():
    rows = []
    for dataset in ("a", "b", "c", "d", "e"):
        for model in ("xgb", "rf", "sgd", "gnb"):
            for policy in POLICIES:
                for seed in range(5):
                    rows.append({"dataset": dataset, "model": model, "policy_key": policy, "seed": str(seed),
                                 "scorer": "accuracy", "mean_prequential_accuracy": "0.7",
                                 "total_work_units": "1"})
    assert build_blocks(rows, "cost", "n60").n == 60
    assert build_blocks(rows, "cost", "n20").n == 20


def test_sensitivity_level_averages_seeds():
    rows = synthetic_grid()
    b20 = build_blocks(rows, "accuracy", "n20")
    b60 = build_blocks(rows, "accuracy", "n60")
    rf_elec2 = [i for i, bid in enumerate(b60.block_ids) if bid[:2] == ("elec2", "rf")]
    row20 = b20.block_ids.index(("elec2", "rf"))
    assert np.allclose(b20.values[row20], b60.values[rf_elec2].mean(axis=0))


# --- ranks and tests ------------------------------------------------------------------------


def test_ranks_reward_high_accuracy_and_low_cost():
    rows = synthetic_grid()
    acc = mean_ranks(build_blocks(rows, "accuracy"))
    cost = mean_ranks(build_blocks(rows, "cost"))
    # policy index p: accuracy rises with p, work units fall with p
    assert acc["never_adapt"] == pytest.approx(1.0)
    assert cost["never_adapt"] == pytest.approx(1.0)
    assert acc["always_rebuild"] == pytest.approx(5.0)


def test_nemenyi_critical_difference_matches_demsar():
    # k=5, N=60: 2.728 * sqrt(5*6 / (6*60))
    assert nemenyi_cd(5, 60) == pytest.approx(2.728 * math.sqrt(30 / 360))
    assert nemenyi_cd(5, 20) > nemenyi_cd(5, 60), "fewer observations, wider critical difference"


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03, 0.20]) == pytest.approx([0.04, 0.09, 0.09, 0.20])


def test_holm_is_monotone_and_capped():
    adj = holm([0.5, 0.4, 0.9])
    assert all(0 <= a <= 1 for a in adj)
    assert adj[1] <= adj[0] <= adj[2]


def test_cliffs_delta_extremes_and_symmetry():
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == 1.0
    assert cliffs_delta([0, 1, 2], [3, 4, 5]) == -1.0
    a, b = [1, 5, 2, 8], [3, 3, 7, 1]
    assert cliffs_delta(a, b) == pytest.approx(-cliffs_delta(b, a))


def test_rank_biserial_matches_wilcoxon_rank_sums():
    a = np.array([5.0, 3.0, 8.0, 2.0, 9.0, 4.0])
    b = np.array([4.0, 4.0, 6.0, 2.0, 5.0, 1.0])
    d = a - b
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    expected = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    assert rank_biserial_paired(a, b) == pytest.approx(expected)
    assert rank_biserial_paired(a, a) == 0.0


# --- Pareto ------------------------------------------------------------------------------------


def test_pareto_frontier_keeps_only_non_dominated_points():
    points = {
        "cheap_bad": (0.0, 50.0),
        "mid": (50.0, 70.0),
        "dominated": (60.0, 65.0),       # costs more and less accurate than mid
        "pricey_best": (100.0, 80.0),
        "tie_worse_acc": (50.0, 69.0),   # same cost as mid, worse accuracy
    }
    assert pareto_frontier(points) == {"cheap_bad", "mid", "pricey_best"}
    assert dominators(points, "dominated") == ["mid", "tie_worse_acc"], "(50, 69) is cheaper and more accurate than (60, 65)"
    assert dominators(points, "tie_worse_acc") == ["mid"]


def test_identical_points_do_not_dominate_each_other():
    assert pareto_frontier({"a": (1.0, 1.0), "b": (1.0, 1.0)}) == {"a", "b"}


def test_cell_pareto_labels_a_more_accurate_costlier_selector_as_a_different_operating_point():
    rows = synthetic_grid()
    for r in rows:
        if r["model"] == "gnb" and r["dataset"] == "elec2":
            if r["policy_key"] == "mechanism_selector":
                r["mean_prequential_accuracy"], r["total_work_units"] = "0.90", "3000"
            if r["policy_key"] == "fixed_schedule":
                r["mean_prequential_accuracy"], r["total_work_units"] = "0.80", "1000"
    cell = cell_pareto(rows, "elec2", "gnb")
    assert cell["selector_vs_fixed_schedule"].startswith("different operating point: selector more accurate")
    assert cell["selector_on_frontier"]
    assert cell["n_seeds"] == 1 and cell["policies"]["mechanism_selector"]["accuracy_sd"] is None


def test_cell_pareto_reports_seed_stability_for_stochastic_models():
    cell = cell_pareto(synthetic_grid(), "elec2", "rf")
    assert cell["n_seeds"] == 5
    assert 0.0 <= cell["selector_frontier_share_across_seeds"] <= 1.0
    assert cell["policies"]["mechanism_selector"]["accuracy_sd"] is not None


# --- decay diagnostic ------------------------------------------------------------------------------


def points(gains_by_position):
    return [{"position": pos, "gain": g, "prediction_change": g} for pos, g in gains_by_position]


def test_trend_is_negative_when_only_the_accumulating_policy_decays():
    positions = [0.05, 0.25, 0.45, 0.65, 0.85]
    accumulating = points([(p, 10 - 10 * p) for p in positions])   # decays along the stream
    resetting = points([(p, 5.0) for p in positions])              # flat
    assert matched_position_trend(accumulating, resetting, "gain") == pytest.approx(-1.0)


def test_trend_is_zero_when_both_decay_with_position_alike():
    positions = [0.05, 0.25, 0.45, 0.65, 0.85]
    both = points([(p, 10 - 10 * p) for p in positions])
    assert matched_position_trend(both, both, "gain") == 0.0, "position alone must not register as decay"


def test_trend_needs_enough_shared_positions():
    early = points([(0.05, 1.0), (0.1, 2.0)])
    late = points([(0.9, 1.0)])
    assert matched_position_trend(early, late, "gain") is None
