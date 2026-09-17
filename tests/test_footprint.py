"""Tests for model footprint: size and per-row inference cost.

The node-visit counts are checked against brute-force walks of every tree, so
they are proven exact rather than merely plausible.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footprint import (  # noqa: E402
    INFER_NODE_VISITS,
    INFER_PARAMS_READ,
    UNIT_PARAMS,
    UNIT_ROUNDS,
    UNIT_TREES,
    measure_footprint,
)
from mechanisms import NudgeMechanism, RebuildMechanism, make_spec, nudge_size  # noqa: E402

warnings.filterwarnings("ignore")


def dataset(n_classes):
    rng = np.random.default_rng(n_classes)
    X = rng.normal(size=(1200, 6))
    y = rng.integers(0, n_classes, 1200)
    return X, y, np.arange(n_classes)


# --- exactness against brute force --------------------------------------------


def brute_force_xgb_visits(model, X):
    """Walk every JSON-dumped tree by hand for every row.

    Comparisons are done in float32, as XGBoost does them. A float64 walk
    disagrees with XGBoost's own traversal on values within float32 rounding of
    a threshold (e.g. -0.105259778 vs -0.105259776).
    """
    total = np.zeros(len(X), dtype=np.int64)
    X32 = X.astype(np.float32)
    for tree_json in model.get_booster().get_dump(dump_format="json"):
        tree = json.loads(tree_json)
        for i, row in enumerate(X32):
            node, visits = tree, 1
            while "children" in node:
                feature = int(node["split"].lstrip("f"))
                go = node["yes"] if row[feature] < np.float32(node["split_condition"]) else node["no"]
                node = next(c for c in node["children"] if c["nodeid"] == go)
                visits += 1
            total[i] += visits
    return total.mean()


@pytest.mark.parametrize("n_classes", [2, 4])
def test_xgb_node_visits_match_a_brute_force_walk(n_classes):
    X, y, classes = dataset(n_classes)
    model = make_spec("xgb", classes, n_estimators=20).factory(0).fit(X, y)
    probe = X[:60]

    fp = measure_footprint(model, probe)

    assert fp.inference_unit == INFER_NODE_VISITS
    assert fp.inference_units == pytest.approx(brute_force_xgb_visits(model, probe), abs=1e-9)


@pytest.mark.parametrize("n_classes", [2, 4])
def test_forest_node_visits_match_a_per_tree_walk(n_classes):
    X, y, classes = dataset(n_classes)
    model = make_spec("rf", classes, n_estimators=15).factory(0).fit(X, y)
    probe = X[:80]

    expected = np.zeros(len(probe))
    for est in model.estimators_:
        t = est.tree_
        for i, row in enumerate(probe.astype(np.float32)):  # sklearn trees compare float32 inputs
            node, visits = 0, 1
            while t.children_left[node] != -1:
                node = t.children_left[node] if row[t.feature[node]] <= t.threshold[node] else t.children_right[node]
                visits += 1
            expected[i] += visits

    fp = measure_footprint(model, probe)
    assert fp.inference_units == pytest.approx(expected.mean(), abs=1e-9)
    assert fp.n_nodes == sum(e.tree_.node_count for e in model.estimators_)


# --- the property the reviewer objection is about ------------------------------


def test_xgb_nudges_grow_the_model_and_its_inference_cost():
    X, y, classes = dataset(3)
    spec = make_spec("xgb", classes)
    model = spec.factory(0).fit(X[:800], y[:800])
    probe = X[800:]
    before = measure_footprint(model, probe)

    nudge = NudgeMechanism(spec)
    for _ in range(4):
        model, _ = nudge.apply(model, X[:800], y[:800], None)
    after = measure_footprint(model, probe)

    assert before.size_unit == UNIT_ROUNDS
    assert after.size == before.size + 4 * nudge_size(spec)
    assert after.inference_units > before.inference_units
    assert after.n_nodes > before.n_nodes


def test_forest_nudges_keep_tree_count_constant():
    X, y, classes = dataset(3)
    spec = make_spec("rf", classes)
    model = spec.factory(0).fit(X[:800], y[:800])
    probe = X[800:]
    before = measure_footprint(model, probe)

    for _ in range(4):
        model, _ = NudgeMechanism(spec).apply(model, X[:800], y[:800], None)
    after = measure_footprint(model, probe)

    assert before.size_unit == UNIT_TREES
    assert after.size == before.size


@pytest.mark.parametrize("name", ["sgd", "gnb"])
def test_parameter_models_do_not_grow_and_read_every_parameter(name):
    X, y, classes = dataset(3)
    spec = make_spec(name, classes)
    model = spec.factory(0).fit(X[:800], y[:800])
    probe = X[800:]
    before = measure_footprint(model, probe)

    for _ in range(4):
        model, _ = NudgeMechanism(spec).apply(model, X[:800], y[:800], None)
    after = measure_footprint(model, probe)

    assert before.size_unit == UNIT_PARAMS and before.inference_unit == INFER_PARAMS_READ
    assert before.n_nodes is None
    assert after.size == before.size
    assert after.inference_units == before.size


def test_parameter_counts_are_exact():
    X, y, classes = dataset(3)
    sgd = make_spec("sgd", classes).factory(0).fit(X, y)
    gnb = make_spec("gnb", classes).factory(0).fit(X, y)
    assert measure_footprint(sgd, X[:10]).size == 3 * 6 + 3
    assert measure_footprint(gnb, X[:10]).size == 3 * 6 + 3 * 6 + 3


def test_a_rebuilt_forest_is_shallower_on_a_small_window():
    """Why node visits rather than tree counts: same tree count, different cost."""
    X, y, classes = dataset(2)
    spec = make_spec("rf", classes)
    big = spec.factory(0).fit(X, y)
    small, _ = RebuildMechanism(spec).apply(None, X[:150], y[:150], None)

    fp_big, fp_small = measure_footprint(big, X[:200]), measure_footprint(small, X[:200])
    assert fp_big.size == fp_small.size
    assert fp_small.inference_units < fp_big.inference_units


def test_structural_measurements_are_deterministic():
    X, y, classes = dataset(3)
    model = make_spec("xgb", classes).factory(0).fit(X, y)
    a, b = measure_footprint(model, X[:100]), measure_footprint(model, X[:100])
    assert (a.size, a.n_nodes, a.inference_units) == (b.size, b.n_nodes, b.inference_units)


def test_empty_probe_is_rejected():
    X, y, classes = dataset(2)
    model = make_spec("gnb", classes).factory(0).fit(X, y)
    with pytest.raises(ValueError):
        measure_footprint(model, X[:0])
