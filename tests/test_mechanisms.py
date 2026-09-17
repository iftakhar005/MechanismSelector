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
    anchor_missing_classes,
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


# --- RebuildMechanism ignores its model argument -----------------------------
#
# The selector hands RebuildMechanism.apply() `None` and relies on the
# mechanism to construct everything itself. If apply() ever started reading
# its model argument -- most dangerously as `baseline=snapshot(model)` -- the
# old model's estimator count would be subtracted from the fresh model's and
# the rebuild billed at or near zero. These tests pin that it does not.


class Tripwire:
    """Fails the test on any attribute access at all, so 'ignores its model
    argument' means never even read -- not merely that the output looks right."""

    def __getattribute__(self, name):
        raise AssertionError(
            f"RebuildMechanism.apply() read .{name} from its model argument; "
            "it must construct from its own spec and ignore the argument"
        )


@pytest.mark.parametrize("name", GRID_MODELS)
def test_rebuild_never_reads_its_model_argument(name, data):
    X, y = data
    spec = make_spec(name, CLASSES)

    rebuilt, cost = RebuildMechanism(spec).apply(
        Tripwire(), X, y, np.random.default_rng(0)
    )

    assert rebuilt is not None
    assert cost.work_units > 0


@pytest.mark.parametrize("name", GRID_MODELS)
def test_rebuild_cost_and_model_independent_of_model_argument(name, data):
    """Same window, three very different model arguments -> identical rebuild.

    The foreign model is deliberately a *larger* ensemble (300 vs 100) fitted on
    different data: if its estimator count were ever used as a cost baseline,
    max(0, 100 - 300) would bill the rebuild at zero and this test would fail.
    """
    X, y = data
    spec = make_spec(name, CLASSES)

    foreign_spec = make_spec(name, CLASSES, n_estimators=300)
    X_other, y_other = X[::-1][:500], y[::-1][:500]
    foreign = foreign_spec.factory(99).fit(X_other, y_other)

    outcomes = []
    for arg in (None, Tripwire(), foreign):
        rebuilt, cost = RebuildMechanism(spec).apply(arg, X, y, np.random.default_rng(0))
        outcomes.append((rebuilt, cost))
        assert rebuilt is not arg

    costs = [c for _, c in outcomes]
    assert len({c.work_units for c in costs}) == 1, (
        f"rebuild cost depended on the model argument: {[c.work_units for c in costs]}"
    )
    assert len({c.n_passes for c in costs}) == 1

    predictions = [m.predict(X) for m, _ in outcomes]
    assert all(np.array_equal(predictions[0], p) for p in predictions[1:]), (
        "rebuilt model differed depending on the model argument -- state leaked in"
    )

    if spec.original_size is not None:
        assert costs[0].n_estimators_fitted == spec.original_size, (
            "a rebuild must be charged for its full ensemble, never a delta"
        )


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


# --- windows missing a class ---------------------------------------------------
#
# 92% of 1,000-row covtype windows lack at least one class. Before placeholder
# rows, an XGBoost nudge on such a window raised, and a Random Forest rebuilt on
# a class subset crashed at prediction after its next nudge.

SEVEN = np.arange(7)


def missing_class_windows():
    rng = np.random.default_rng(7)
    return {
        "missing_3_to_6": (rng.normal(size=(300, 5)), rng.integers(0, 3, 300)),
        "non_contiguous": (rng.normal(size=(300, 5)), rng.choice([0, 2, 5], 300)),
        "single_class": (rng.normal(size=(300, 5)), np.full(300, 4)),
    }


@pytest.fixture(scope="module")
def full_class_data():
    rng = np.random.default_rng(8)
    return rng.normal(size=(2000, 5)), rng.integers(0, 7, 2000)


@pytest.mark.parametrize("window_kind", ["missing_3_to_6", "non_contiguous", "single_class"])
@pytest.mark.parametrize("name", GRID_MODELS)
def test_rebuild_then_nudge_survive_windows_missing_classes(name, window_kind, full_class_data):
    X_full, y_full = full_class_data
    X_win, y_win = missing_class_windows()[window_kind]
    spec = make_spec(name, SEVEN)
    rng = np.random.default_rng(0)

    rebuilt, rebuild_cost = RebuildMechanism(spec).apply(None, X_win, y_win, rng)
    assert list(rebuilt.classes_) == list(SEVEN), "rebuild must know every stream class"
    assert rebuild_cost.n_rows_processed == len(X_win), "placeholders are not charged"

    nudged, _ = NudgeMechanism(spec).apply(rebuilt, X_full[:300], y_full[:300], rng)
    assert np.all(np.isin(nudged.predict(X_full), SEVEN))

    full = spec.factory(0).fit(X_full, y_full)
    nudged_full, nudge_cost = NudgeMechanism(spec).apply(full, X_win, y_win, rng)
    assert nudge_cost.n_rows_processed == len(X_win)
    assert np.all(np.isin(nudged_full.predict(X_full), SEVEN))


@pytest.mark.parametrize("name", GRID_MODELS)
def test_placeholder_classes_are_never_predicted(name, full_class_data):
    X_full, _ = full_class_data
    X_win, y_win = missing_class_windows()["missing_3_to_6"]
    rebuilt, _ = RebuildMechanism(make_spec(name, SEVEN)).apply(
        None, X_win, y_win, np.random.default_rng(0)
    )

    assert not np.any(np.isin(rebuilt.predict(X_full), [3, 4, 5, 6]))
    if name != "sgd":
        mass = rebuilt.predict_proba(X_full)[:, 3:].max()
        limit = 0.0 if name in ("rf", "gnb") else 1e-4
        assert mass <= limit, f"placeholder classes carry probability mass {mass}"


@pytest.mark.parametrize("name", ["xgb", "gnb"])
def test_zero_weight_placeholders_are_exactly_inert(name):
    """For XGBoost and GaussianNB a zero-weight row changes nothing at all.
    (Random Forest and SGD are documented as not bit-for-bit inert: the rows
    shift the bootstrap / shuffle random stream without carrying weight.)"""
    rng = np.random.default_rng(3)
    X, y = rng.normal(size=(600, 5)), rng.integers(0, 4, 600)
    X_test = rng.normal(size=(300, 5))
    spec = make_spec(name, np.arange(4))

    plain = spec.factory(0).fit(X, y)
    padded = spec.factory(0).fit(
        np.vstack([X, X[:4]]), np.concatenate([y, y[:4]]),
        sample_weight=np.concatenate([np.ones(600), np.zeros(4)]),
    )
    assert np.allclose(plain.predict_proba(X_test), padded.predict_proba(X_test))


def test_anchor_is_a_no_op_when_every_class_is_present():
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 1, 2, 0, 1, 2])
    X_out, y_out, kwargs = anchor_missing_classes(X, y, np.arange(3))
    assert X_out is X and y_out is y and kwargs == {}


def test_anchor_adds_one_zero_weight_copy_per_missing_class():
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0, 0, 2, 2])
    X_out, y_out, kwargs = anchor_missing_classes(X, y, np.arange(5))

    assert list(y_out[4:]) == [1, 3, 4]
    assert np.array_equal(X_out[4:], np.repeat(X[:1], 3, axis=0)), "copies, not zeros"
    assert list(kwargs["sample_weight"]) == [1, 1, 1, 1, 0, 0, 0]


def test_anchor_refuses_an_empty_window():
    with pytest.raises(ValueError):
        anchor_missing_classes(np.empty((0, 3)), np.empty(0, dtype=int), np.arange(2))


@pytest.mark.parametrize("name", GRID_MODELS)
def test_placeholder_count_is_recorded_on_the_cost(name, full_class_data):
    X_full, y_full = full_class_data
    X_win, y_win = missing_class_windows()["missing_3_to_6"]     # 4 classes absent
    spec = make_spec(name, SEVEN)
    rng = np.random.default_rng(0)

    _, rebuild_cost = RebuildMechanism(spec).apply(None, X_win, y_win, rng)
    assert rebuild_cost.n_placeholder_rows == 4

    full = spec.factory(0).fit(X_full, y_full)
    _, nudge_cost = NudgeMechanism(spec).apply(full, X_win, y_win, rng)
    expected = 4 if name in ("xgb", "rf") else 0   # partial_fit declares classes itself
    assert nudge_cost.n_placeholder_rows == expected

    _, clean = RebuildMechanism(spec).apply(None, X_full, y_full, rng)
    assert clean.n_placeholder_rows == 0, "no class missing, no placeholders"
