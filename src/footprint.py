"""Model footprint: size and per-prediction inference cost.

`accounting.py` measures what it costs to *train*. This module measures what a
model costs to *use*, because a policy can buy cheap training with permanently
more expensive inference: every XGBoost nudge appends boosting rounds that every
later prediction must evaluate. Counting training work alone would report that
policy as a large saving while hiding a cost paid on every prediction, forever.

## What is measured

| Family | Size | Inference units per row |
|---|---|---|
| `XGBClassifier` | boosting rounds | tree node visits (exact, mean over a probe set) |
| `RandomForestClassifier` | trees | tree node visits (exact, mean over a probe set) |
| `SGDClassifier` | parameters | parameters read (= all parameters) |
| `GaussianNB` | parameters | parameters read (= all parameters) |
| `MLPClassifier` | parameters | parameters read (= all parameters) |

**Node visits, not tree counts.** Random Forest keeps a constant number of trees,
but its trees are grown without a depth limit, so a forest rebuilt on a
1,000-row window is shallower than one trained on 4,500 rows. Counting the nodes
a prediction actually visits -- every split comparison plus one leaf per tree --
captures both more trees and deeper trees. For XGBoost the count comes from the
leaf each row lands in and that leaf's depth; for Random Forest from
`decision_path`. Both are exact integers per row, averaged over a fixed probe set.

**The probe set is fixed per run.** Initial and final models are measured on the
same rows, so a change in inference units reflects the model changing, not the
data. Path lengths depend on inputs, so values are comparable within a
(dataset, model) cell, not across datasets.

**Units are not comparable across families**, the same caveat as work units: a
node visit and a parameter read are different amounts of computation.

Measured prediction time is recorded alongside as a secondary figure. Like
wall-clock training time and energy, it is hardware-dependent and never used to
decide anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

UNIT_ROUNDS = "rounds"
UNIT_TREES = "trees"
UNIT_PARAMS = "params"
INFER_NODE_VISITS = "node_visits"
INFER_PARAMS_READ = "params_read"


@dataclass(frozen=True)
class Footprint:
    """Size and inference cost of one fitted model.

    Attributes:
        size: rounds (XGBoost), trees (Random Forest) or parameters (others).
        size_unit: which of those `size` counts.
        n_nodes: total tree nodes, the memory footprint of a tree ensemble;
            None for non-tree models.
        inference_units: mean node visits per prediction row for tree models,
            parameters read per row for the others.
        inference_unit: which of those `inference_units` counts.
        predict_us_per_row: measured prediction time. Secondary, never decisive.
    """

    size: int
    size_unit: str
    n_nodes: int | None
    inference_units: float
    inference_unit: str
    predict_us_per_row: float


def measure_footprint(model: Any, X_probe: np.ndarray, timing_repeats: int = 3) -> Footprint:
    """Measure a fitted model's size and per-row inference cost on `X_probe`."""
    if len(X_probe) == 0:
        raise ValueError("probe set is empty")

    if hasattr(model, "get_booster"):
        size, unit, n_nodes, units, infer = _xgb(model, X_probe)
    elif hasattr(model, "estimators_") and hasattr(model, "decision_path"):
        size, unit, n_nodes, units, infer = _forest(model, X_probe)
    else:
        size = _param_count(model)
        unit, n_nodes, units, infer = UNIT_PARAMS, None, float(size), INFER_PARAMS_READ

    return Footprint(
        size=int(size),
        size_unit=unit,
        n_nodes=n_nodes,
        inference_units=float(units),
        inference_unit=infer,
        predict_us_per_row=_time_predict(model, X_probe, timing_repeats),
    )


def model_size(model: Any) -> tuple[int, str]:
    """Cheap size only -- no probe, no tree traversal. For per-adaptation logging."""
    if hasattr(model, "get_booster"):
        return int(model.get_booster().num_boosted_rounds()), UNIT_ROUNDS
    if hasattr(model, "estimators_") and hasattr(model, "decision_path"):
        return len(model.estimators_), UNIT_TREES
    return _param_count(model), UNIT_PARAMS


def xgb_leaf_depths(booster) -> np.ndarray:
    """Depth of every node, as a (n_trees, max_node_id + 1) array; -1 = absent.

    Depth counts split comparisons from the root, so a root-only tree has a
    leaf at depth 0.
    """
    df = booster.trees_to_dataframe()
    trees = df["Tree"].to_numpy()
    nodes = df["Node"].to_numpy()
    n_trees = int(trees.max()) + 1
    depth = np.full((n_trees, int(nodes.max()) + 1), -1, dtype=np.int64)
    depth[np.unique(trees), 0] = 0

    internal = df[df["Feature"] != "Leaf"].sort_values(["Tree", "Node"])
    t = internal["Tree"].to_numpy()
    parent = internal["Node"].to_numpy()
    yes = internal["Yes"].str.split("-").str[1].astype(int).to_numpy()
    no = internal["No"].str.split("-").str[1].astype(int).to_numpy()

    for ti, pi, yi, ni in zip(t, parent, yes, no):
        d = depth[ti, pi]
        if d < 0:
            raise AssertionError(
                "XGBoost node visited before its parent; children are assumed to "
                "have larger node ids than their parent"
            )
        depth[ti, yi] = d + 1
        depth[ti, ni] = d + 1
    return depth


def _xgb(model, X_probe):
    import xgboost as xgb

    booster = model.get_booster()
    depth = xgb_leaf_depths(booster)
    leaves = booster.predict(xgb.DMatrix(X_probe), pred_leaf=True).astype(np.int64)
    if leaves.ndim == 1:
        leaves = leaves[:, None]
    if leaves.shape[1] != depth.shape[0]:
        raise AssertionError(
            f"pred_leaf returned {leaves.shape[1]} trees, structure has {depth.shape[0]}"
        )

    leaf_depths = depth[np.arange(depth.shape[0])[None, :], leaves]
    if np.any(leaf_depths < 0):
        raise AssertionError("pred_leaf pointed at a node absent from the tree structure")

    # a row visits (depth + 1) nodes in each tree: its comparisons plus the leaf
    visits_per_row = (leaf_depths + 1).sum(axis=1)
    n_nodes = int((depth >= 0).sum())
    return booster.num_boosted_rounds(), UNIT_ROUNDS, n_nodes, visits_per_row.mean(), INFER_NODE_VISITS


def _forest(model, X_probe):
    indicator, _ = model.decision_path(X_probe)
    visits_per_row = np.asarray(indicator.getnnz(axis=1))
    n_nodes = int(sum(est.tree_.node_count for est in model.estimators_))
    return len(model.estimators_), UNIT_TREES, n_nodes, visits_per_row.mean(), INFER_NODE_VISITS


def _param_count(model) -> int:
    """Every learned array a prediction reads."""
    if hasattr(model, "coefs_"):  # MLP
        return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))
    if hasattr(model, "theta_"):  # GaussianNB
        return int(model.theta_.size + model.var_.size + model.class_prior_.size)
    if hasattr(model, "coef_"):  # linear models
        return int(model.coef_.size + np.asarray(model.intercept_).size)
    raise TypeError(f"no footprint rule for {type(model).__name__}")


def _time_predict(model, X_probe, repeats: int) -> float:
    timings = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        model.predict(X_probe)
        timings.append(time.perf_counter() - start)
    return 1e6 * float(np.median(timings)) / len(X_probe)
