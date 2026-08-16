"""Phase 3 acceptance tests for the mechanism library."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechanisms import (  # noqa: E402
    GRID_MODELS,
    MechanismUnavailable,
    NudgeMechanism,
    RebuildMechanism,
    SkipMechanism,
    make_spec,
    nudge_size,
)

warnings.filterwarnings("ignore")

CLASSES = np.array([0, 1])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def fitted(name, data, n_estimators=None):
    X, y = data
    spec = make_spec(name, CLASSES, n_estimators=n_estimators)
    model = spec.factory(0).fit(X, y)
    return spec, model


@pytest.mark.parametrize("name", GRID_MODELS)
def test_gate_nudge_is_cheaper_than_rebuild(name, data):
    """PHASE 3 GATE (part 1): nudge measurably cheaper than rebuild.

    GaussianNB is the documented exception -- its rebuild is already one
    closed-form pass, so parity is the correct result, not a failure.
    """
    X, y = data
    spec, model = fitted(name, data)
    rng = np.random.default_rng(0)

    _, nudge_cost = NudgeMechanism(spec).apply(model, X, y, rng)
    _, rebuild_cost = RebuildMechanism(spec).apply(model, X, y, rng)

    if name == "gnb":
        assert nudge_cost.work_units == rebuild_cost.work_units, (
            "GaussianNB is the negative control: nudge and rebuild are both "
            "a single pass, so they must cost the same"
        )
    else:
        assert nudge_cost.work_units < rebuild_cost.work_units, (
            f"{name}: nudge {nudge_cost.work_units} not cheaper than "
            f"rebuild {rebuild_cost.work_units}"
        )


def test_gate_rf_keeps_constant_tree_count_over_ten_nudges(data):
    """PHASE 3 GATE (part 2): the forest must not dilute."""
    X, y = data
    spec, model = fitted("rf", data)
    original = len(model.estimators_)
    rng = np.random.default_rng(0)
    nudge = NudgeMechanism(spec)

    for i in range(10):
        model, _ = nudge.apply(model, X, y, rng)
        assert len(model.estimators_) == original, (
            f"forest diluted at nudge {i + 1}: "
            f"{len(model.estimators_)} trees, expected {original}"
        )

    assert model.n_estimators == original


def test_gate_rf_ten_nudges_still_charge_full_cost(data):
    """Truncating the forest must not make repeated nudges look free."""
    X, y = data
    spec, model = fitted("rf", data)
    k = nudge_size(spec)
    rng = np.random.default_rng(0)
    nudge = NudgeMechanism(spec)

    for _ in range(10):
        model, cost = nudge.apply(model, X, y, rng)
        assert cost.n_estimators_fitted == k, (
            "each nudge trains k new trees and must be charged for them "
            "even though k old trees are dropped afterwards"
        )
        assert cost.work_units == k * len(X)


def test_gate_rf_replacement_actually_swaps_trees(data):
    """The retained trees must be the newest, not the same ones each time."""
    X, y = data
    spec, model = fitted("rf", data)
    k = nudge_size(spec)
    before = list(model.estimators_)
    rng = np.random.default_rng(0)

    model, _ = NudgeMechanism(spec).apply(model, X, y, rng)
    after = list(model.estimators_)

    assert after[: len(before) - k] == before[k:], "oldest k trees should be dropped"
    assert all(t not in before for t in after[len(before) - k :]), (
        "newest k trees should be freshly fitted"
    )


@pytest.mark.parametrize("name", ["svc", "knn"])
def test_gate_unsupported_models_raise(name, data):
    """PHASE 3 GATE (part 3)."""
    X, y = data
    spec, model = fitted(name, data)
    rng = np.random.default_rng(0)

    with pytest.raises(MechanismUnavailable):
        NudgeMechanism(spec).apply(model, X, y, rng)


def test_skip_is_free_and_does_not_touch_the_model(data):
    X, y = data
    spec, model = fitted("rf", data)
    rng = np.random.default_rng(0)

    returned, cost = SkipMechanism().apply(model, X, y, rng)

    assert returned is model
    assert cost.work_units == 0
    assert cost.n_rows_processed == 0


def test_nudge_size_is_five_percent_of_original(data):
    spec = make_spec("rf", CLASSES, n_estimators=300)
    assert nudge_size(spec) == 15

    spec100 = make_spec("xgb", CLASSES, n_estimators=100)
    assert nudge_size(spec100) == 5


def test_nudge_size_does_not_compound_across_nudges(data):
    """5% is of the *original* size, so repeated nudges add a constant amount."""
    X, y = data
    spec, model = fitted("xgb", data)
    k = nudge_size(spec)
    rng = np.random.default_rng(0)
    nudge = NudgeMechanism(spec)

    for _ in range(3):
        model, cost = nudge.apply(model, X, y, rng)
        assert cost.n_estimators_fitted == k, "nudge size must stay constant"


def test_xgb_nudge_appends_to_existing_booster(data):
    """The nudge must extend the model, not silently retrain it."""
    X, y = data
    spec, model = fitted("xgb", data)
    original_rounds = model.get_booster().num_boosted_rounds()
    k = nudge_size(spec)
    rng = np.random.default_rng(0)

    nudged, cost = NudgeMechanism(spec).apply(model, X, y, rng)

    assert nudged.get_booster().num_boosted_rounds() == original_rounds + k
    assert cost.n_estimators_fitted == k, "must charge only the added rounds"


def test_rebuild_carries_no_state_from_the_old_model(data):
    """Rebuild is from scratch: the old model must be untouched and unused."""
    X, y = data
    spec, model = fitted("rf", data)
    old_trees = list(model.estimators_)

    rebuilt, _ = RebuildMechanism(spec).apply(model, X, y, np.random.default_rng(0))

    assert rebuilt is not model
    assert not any(t in old_trees for t in rebuilt.estimators_)
    assert list(model.estimators_) == old_trees, "old model must not be mutated"


def test_partial_fit_survives_a_window_missing_a_class(data):
    """Spec trap #3: classes= must be passed on every call."""
    X, y = data
    spec, model = fitted("sgd", data)

    single_class = y == 0
    X_one, y_one = X[single_class][:100], y[single_class][:100]
    assert len(np.unique(y_one)) == 1

    nudged, cost = NudgeMechanism(spec).apply(
        model, X_one, y_one, np.random.default_rng(0)
    )
    assert cost.n_passes == 1
    assert set(nudged.classes_) == {0, 1}


@pytest.mark.parametrize("name", GRID_MODELS)
def test_all_grid_models_are_nudgeable(name, data):
    X, y = data
    spec, model = fitted(name, data)
    nudged, cost = NudgeMechanism(spec).apply(model, X, y, np.random.default_rng(0))
    assert nudged is not None
    assert cost.work_units > 0
