"""Phase 5 tests for the five policies.

Three groups:

1. Each baseline does exactly what its name says, on every alarm.
2. The fairness invariant: every policy performs the *same* nudge and the
   *same* rebuild as the selector, on the same rows, at the same cost. If this
   breaks, differences between policies stop being attributable to the
   decision rule.
3. All five run end to end on every grid model family.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from accounting import PASS_TREES, CostRecord  # noqa: E402
from mechanisms import GRID_MODELS, make_spec  # noqa: E402
from policies import (  # noqa: E402
    POLICY_KEYS,
    AlwaysNudge,
    AlwaysRebuild,
    FixedSchedule,
    MechanismSelector,
    NeverAdapt,
    make_policy,
)
from selector import NUDGE, REBUILD, SKIP, AdaptResult, temporal_split  # noqa: E402

warnings.filterwarnings("ignore")


# --- fakes ---------------------------------------------------------------


class FakeModel:
    """Predicts a constant label. With an all-ones window, label 1 is 100%
    accurate and label 0 is 0% -- accuracy is fully controlled."""

    def __init__(self, label: int):
        self.label = label

    def predict(self, X):
        return np.full(len(X), self.label)


class RecordingMechanism:
    """A mechanism stub that records exactly which rows it was given."""

    def __init__(self, cost: CostRecord, transform):
        self.cost = cost
        self.transform = transform
        self.calls: list[tuple] = []

    def apply(self, model, X, y, rng):
        self.calls.append((model, X.copy(), y.copy()))
        return self.transform(model, X, y), self.cost


def mutate_in_place(model, X, y):
    """Simulates partial_fit: mutates and returns the same object."""
    model.label = 1 - model.label
    return model


def fresh_correct(model, X, y):
    return FakeModel(label=1)


NUDGE_COST = CostRecord(5, 5, PASS_TREES, 800, 5 * 800, 0.1, None)
REBUILD_COST = CostRecord(100, 100, PASS_TREES, 1000, 100 * 1000, 1.0, None)


def window(n=1000):
    """Column 0 is a row index, so rows are identifiable after slicing."""
    X = np.random.default_rng(0).normal(size=(n, 4))
    X[:, 0] = np.arange(n, dtype=float)
    y = np.ones(n, dtype=int)
    return X, y


def mechanisms():
    return (
        RecordingMechanism(NUDGE_COST, mutate_in_place),
        RecordingMechanism(REBUILD_COST, fresh_correct),
    )


# --- 1. each baseline does what it says ---------------------------------------


def test_never_adapt_skips_at_zero_cost_and_touches_nothing():
    nudge, rebuild = mechanisms()
    policy = NeverAdapt(nudge, rebuild)
    X, y = window()
    model = FakeModel(label=0)  # badly wrong -- must still be ignored

    for _ in range(5):
        result = policy.adapt(model, X, y, reference_accuracy=1.0)
        assert result.action == SKIP
        assert result.cost == CostRecord.zero()
        assert result.model is model
        assert result.acc_after == result.acc_before

    assert nudge.calls == [] and rebuild.calls == []
    assert model.label == 0


def test_always_rebuild_rebuilds_every_alarm_on_the_full_window():
    nudge, rebuild = mechanisms()
    policy = AlwaysRebuild(nudge, rebuild)
    X, y = window()
    model = FakeModel(label=1)  # already perfect -- must rebuild regardless

    for _ in range(3):
        result = policy.adapt(model, X, y, reference_accuracy=1.0)
        assert result.action == REBUILD
        assert result.cost == REBUILD_COST, "no nudge was attempted, so none is charged"
        assert result.acc_nudged is None

    assert nudge.calls == []
    assert len(rebuild.calls) == 3
    for passed_model, X_seen, _ in rebuild.calls:
        assert passed_model is None, "rebuild must be handed nothing to inherit"
        assert np.array_equal(X_seen, X), "rebuild trains on the full window"


def test_always_nudge_nudges_every_alarm_and_never_rebuilds():
    nudge, rebuild = mechanisms()
    policy = AlwaysNudge(nudge, rebuild)
    X, y = window()

    for _ in range(4):
        result = policy.adapt(FakeModel(label=1), X, y, reference_accuracy=1.0)
        assert result.action == NUDGE
        assert result.cost == NUDGE_COST
        assert result.acc_nudged is not None

    assert len(nudge.calls) == 4
    assert rebuild.calls == []


def test_fixed_schedule_k3_rebuilds_on_every_third_alarm():
    nudge, rebuild = mechanisms()
    policy = FixedSchedule(nudge, rebuild, k=3)
    X, y = window()

    actions = [policy.adapt(FakeModel(0), X, y, 1.0).action for _ in range(7)]

    assert actions == [NUDGE, NUDGE, REBUILD, NUDGE, NUDGE, REBUILD, NUDGE]
    assert len(nudge.calls) == 5
    assert len(rebuild.calls) == 2


def test_fixed_schedule_rebuild_alarms_are_not_charged_a_nudge():
    nudge, rebuild = mechanisms()
    policy = FixedSchedule(nudge, rebuild, k=3)
    X, y = window()

    costs = [policy.adapt(FakeModel(0), X, y, 1.0).cost for _ in range(3)]

    assert costs == [NUDGE_COST, NUDGE_COST, REBUILD_COST]


def test_fixed_schedule_never_looks_at_accuracy():
    """The contrast with the selector: a perfect model still gets acted on."""
    nudge, rebuild = mechanisms()
    schedule = FixedSchedule(nudge, rebuild, k=3)
    selector = MechanismSelector(*mechanisms())
    X, y = window()

    perfect = FakeModel(label=1)
    assert selector.adapt(perfect, X, y, 1.0).action == SKIP
    assert schedule.adapt(perfect, X, y, 1.0).action == NUDGE


def test_fixed_schedule_k1_is_equivalent_to_always_rebuild():
    policy = FixedSchedule(*mechanisms(), k=1)
    X, y = window()
    assert all(policy.adapt(FakeModel(0), X, y, 1.0).action == REBUILD for _ in range(4))


@pytest.mark.parametrize("bad_k", [0, -1, 2.5, True, "3", None])
def test_fixed_schedule_rejects_invalid_k(bad_k):
    with pytest.raises(ValueError):
        FixedSchedule(*mechanisms(), k=bad_k)


def test_fixed_schedule_instances_do_not_share_state():
    X, y = window()
    first = FixedSchedule(*mechanisms(), k=3)
    for _ in range(2):
        first.adapt(FakeModel(0), X, y, 1.0)

    second = FixedSchedule(*mechanisms(), k=3)
    assert second.adapt(FakeModel(0), X, y, 1.0).action == NUDGE
    assert first.adapt(FakeModel(0), X, y, 1.0).action == REBUILD


def test_fixed_schedule_name_records_k():
    assert FixedSchedule(*mechanisms(), k=3).name == "FixedSchedule(k=3)"
    assert FixedSchedule(*mechanisms(), k=5).name == "FixedSchedule(k=5)"


@pytest.mark.parametrize("policy_cls", [NeverAdapt, AlwaysRebuild, AlwaysNudge, FixedSchedule])
def test_baselines_apply_no_floor_and_never_degrade(policy_cls):
    """Baselines make no accuracy decision, so a floor would be fictitious and
    the small-window guard -- which protects a decision -- does not apply."""
    X, y = window(n=60)  # would degrade the selector
    result = policy_cls(*mechanisms()).adapt(FakeModel(0), X, y, 1.0)

    assert math.isnan(result.floor)
    assert result.degraded_to_rebuild is False


@pytest.mark.parametrize("policy_cls", [AlwaysNudge, FixedSchedule])
def test_baseline_nudges_never_train_on_the_holdout(policy_cls):
    """The selector's leakage guarantee, extended to every policy that nudges."""
    nudge, rebuild = mechanisms()
    policy = policy_cls(nudge, rebuild)
    X, y = window()

    policy.adapt(FakeModel(0), X, y, 1.0)

    X_train, _, X_holdout, _ = temporal_split(X, y, policy.holdout_frac)
    trained_on = set(nudge.calls[0][1][:, 0])
    assert trained_on.isdisjoint(set(X_holdout[:, 0])), "a holdout row leaked into a nudge"
    assert np.array_equal(nudge.calls[0][1], X_train)


@pytest.mark.parametrize(
    "policy_cls", [NeverAdapt, AlwaysRebuild, AlwaysNudge, FixedSchedule, MechanismSelector]
)
def test_no_policy_mutates_the_callers_model(policy_cls):
    X, y = window()
    model = FakeModel(label=0)

    for _ in range(3):  # covers both FixedSchedule actions
        policy_cls(*mechanisms()).adapt(model, X, y, 1.0)

    assert model.label == 0


# --- 2. fairness: identical actions under every policy -------------------------


def test_every_policy_hands_its_mechanisms_the_same_rows_as_the_selector():
    X, y = window()
    still_wrong = lambda model, X, y: FakeModel(label=0)  # noqa: E731

    sel_nudge = RecordingMechanism(NUDGE_COST, still_wrong)
    sel_rebuild = RecordingMechanism(REBUILD_COST, fresh_correct)
    selector_result = MechanismSelector(sel_nudge, sel_rebuild).adapt(FakeModel(0), X, y, 1.0)
    assert selector_result.action == REBUILD, "precondition: selector tried both actions"

    nudge_only, _ = mechanisms()
    AlwaysNudge(nudge_only, RecordingMechanism(REBUILD_COST, fresh_correct)).adapt(
        FakeModel(0), X, y, 1.0
    )
    _, rebuild_only = mechanisms()
    AlwaysRebuild(RecordingMechanism(NUDGE_COST, still_wrong), rebuild_only).adapt(
        FakeModel(0), X, y, 1.0
    )

    for got, want in [(nudge_only.calls[0], sel_nudge.calls[0]),
                      (rebuild_only.calls[0], sel_rebuild.calls[0])]:
        assert np.array_equal(got[1], want[1]), "different X than the selector"
        assert np.array_equal(got[2], want[2]), "different y than the selector"


@pytest.mark.parametrize("name", GRID_MODELS)
def test_actions_cost_exactly_the_same_under_every_policy(name):
    """With real models: the selector's REBUILD outcome costs precisely one
    AlwaysNudge alarm plus one AlwaysRebuild alarm, and rebuilds the same model.

    Labels carry 20% noise so no model can reach the 0.98 floor on the holdout,
    which forces the selector down the nudge-then-rebuild path.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    flip = rng.random(1000) < 0.20
    y[flip] = 1 - y[flip]

    spec = make_spec(name, np.array([0, 1]))
    model = spec.factory(0).fit(X[:500], y[:500])

    selector = make_policy("mechanism_selector", spec, seed=0).adapt(model, X, y, 1.0)
    nudged = make_policy("always_nudge", spec, seed=0).adapt(model, X, y, 1.0)
    rebuilt = make_policy("always_rebuild", spec, seed=0).adapt(model, X, y, 1.0)

    assert selector.action == REBUILD, "precondition: floor unreachable, so selector escalates"
    assert selector.acc_nudged == pytest.approx(nudged.acc_nudged), "same nudge, same holdout"
    assert selector.cost.work_units == nudged.cost.work_units + rebuilt.cost.work_units
    assert np.array_equal(selector.model.predict(X), rebuilt.model.predict(X)), (
        "the selector's rebuild and AlwaysRebuild must produce the same model"
    )


# --- 3. all five, end to end, every grid model ---------------------------------


def test_make_policy_builds_all_five_in_spec_order():
    spec = make_spec("gnb", np.array([0, 1]))
    names = [make_policy(key, spec).name for key in POLICY_KEYS]
    assert names == [
        "NeverAdapt",
        "AlwaysRebuild",
        "AlwaysNudge",
        "FixedSchedule(k=3)",
        "MechanismSelector",
    ]


def test_make_policy_rejects_unknown_key():
    with pytest.raises(ValueError):
        make_policy("always_skip", make_spec("gnb", np.array([0, 1])))


def test_make_policy_returns_a_fresh_instance_each_call():
    spec = make_spec("gnb", np.array([0, 1]))
    assert make_policy("fixed_schedule", spec) is not make_policy("fixed_schedule", spec)


@pytest.mark.parametrize("key", POLICY_KEYS)
@pytest.mark.parametrize("name", GRID_MODELS)
def test_all_five_policies_run_on_every_grid_model(key, name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    spec = make_spec(name, np.array([0, 1]))
    model = spec.factory(0).fit(X[:500], y[:500])
    policy = make_policy(key, spec, seed=0)

    for _ in range(3):  # three alarms: exercises FixedSchedule's rebuild
        result = policy.adapt(model, X, y, reference_accuracy=0.9)
        assert isinstance(result, AdaptResult)
        assert result.action in (SKIP, NUDGE, REBUILD)
        assert result.cost.work_units >= 0
        model = result.model
        assert hasattr(model, "predict")


def test_baseline_prediction_change_recorded_for_nudges_only():
    X, y = window()
    assert AlwaysNudge(*mechanisms()).adapt(FakeModel(0), X, y, 1.0).prediction_change == 1.0
    assert AlwaysRebuild(*mechanisms()).adapt(FakeModel(0), X, y, 1.0).prediction_change is None
    assert NeverAdapt(*mechanisms()).adapt(FakeModel(0), X, y, 1.0).prediction_change is None
