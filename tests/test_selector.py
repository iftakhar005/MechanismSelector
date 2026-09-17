"""Phase 4 acceptance tests for the selector -- the project's contribution.

Leakage tests come first (spec trap #1): a leaking selector produces
excellent-looking results that are worthless, and nothing crashes to tell you.
These must pass before anything else is trusted.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from accounting import CostRecord, PASS_TREES  # noqa: E402
from mechanisms import GRID_MODELS, NudgeMechanism, RebuildMechanism, make_spec  # noqa: E402
from selector import AdaptResult, MechanismSelector, NUDGE, REBUILD, SKIP  # noqa: E402

warnings_filter = pytest.mark.filterwarnings("ignore")


# --- fakes ---------------------------------------------------------------
#
# The leakage / action-reachability / cost tests need full control over
# whether a nudge "succeeds," which a real model on random data does not
# give us deterministically. These stand-ins are plain data objects: they
# expose exactly the surface the selector touches (`predict`, `apply`) and
# nothing else, so a bug in the selector can't hide behind incidental
# behaviour of a real estimator.


class FakeModel:
    """Predicts a constant label for every row. `label` is mutated in place
    by `mutate_nudge` to simulate a real partial_fit changing model state --
    this is what lets the mutation tests catch a missing deepcopy."""

    def __init__(self, label: int):
        self.label = label

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.label)


class FakeMechanism:
    """A `mechanisms.py`-shaped stub with an injectable fixed cost and a
    caller-supplied transform. Records every call for inspection."""

    def __init__(self, cost: CostRecord, transform, name: str = "fake"):
        self.cost = cost
        self.transform = transform
        self.name = name
        self.calls: list[tuple] = []

    def apply(self, model, X, y, rng):
        self.calls.append((model, X, y, rng))
        return self.transform(model, X, y), self.cost


def mutate_in_place(model: FakeModel, X, y) -> FakeModel:
    """Simulates partial_fit: mutates and returns the SAME object."""
    model.label = 1 - model.label
    return model


def flip_label(model: FakeModel, X, y) -> FakeModel:
    """A fresh, correctly-fixed model -- used for the rebuild mechanism,
    which does not need to be tested for success/failure (it's unconditional)."""
    return FakeModel(label=1)


NUDGE_COST = CostRecord(15, 15, PASS_TREES, 800, 15 * 800, 0.1, None)
REBUILD_COST = CostRecord(300, 300, PASS_TREES, 1000, 300 * 1000, 1.0, None)


def make_window(n=1000, seed=0, marker=False):
    """y is constant (all 1s) so a FakeModel's accuracy is fully controlled
    by its `label`: label==1 -> 100% accurate, label==0 -> 0% accurate."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    if marker:
        X[:, 0] = np.arange(n, dtype=float)  # increasing marker column
    y = np.ones(n, dtype=int)
    return X, y


def build_selector(nudge_mechanism, rebuild_mechanism, **kwargs):
    return MechanismSelector(
        nudge_mechanism=nudge_mechanism,
        rebuild_mechanism=rebuild_mechanism,
        **kwargs,
    )


# --- leakage tests (the gate) ---------------------------------------------


def test_holdout_rows_never_in_nudge_training():
    X, y = make_window(n=1000, seed=1)
    nudge = FakeMechanism(NUDGE_COST, mutate_in_place)
    rebuild = FakeMechanism(REBUILD_COST, flip_label)
    sel = build_selector(nudge, rebuild)

    model = FakeModel(label=0)  # wrong -> forces an attempt to nudge
    sel.adapt(model, X, y, reference_accuracy=1.0)

    assert nudge.calls, "nudge should have been attempted"
    captured_X = nudge.calls[0][1]
    k = int(len(X) * (1 - sel.holdout_frac))
    holdout_rows = {tuple(row) for row in X[k:]}
    captured_rows = {tuple(row) for row in captured_X}
    assert holdout_rows.isdisjoint(captured_rows), (
        "a holdout row leaked into the nudge's training data"
    )


def test_holdout_is_the_most_recent_slice():
    X, y = make_window(n=200, seed=2, marker=True)
    sel = build_selector(
        FakeMechanism(NUDGE_COST, mutate_in_place),
        FakeMechanism(REBUILD_COST, flip_label),
    )

    X_train, _, X_holdout, _ = sel._split(X, y)

    assert X_train[:, 0].max() < X_holdout[:, 0].min(), (
        "holdout must contain the highest (most recent) markers"
    )


def test_split_is_deterministic():
    X, y = make_window(n=500, seed=3)
    nudge = FakeMechanism(NUDGE_COST, mutate_in_place)
    rebuild = FakeMechanism(REBUILD_COST, flip_label)

    for seed in (0, 1, 42):
        sel = build_selector(
            FakeMechanism(NUDGE_COST, mutate_in_place),
            FakeMechanism(REBUILD_COST, flip_label),
            seed=seed,
        )
        splits = [sel._split(X, y) for _ in range(3)]
        for X_train, y_train, X_holdout, y_holdout in splits[1:]:
            assert np.array_equal(X_train, splits[0][0])
            assert np.array_equal(X_holdout, splits[0][2])

    # full adapt() is deterministic too: same (window, model, ref_acc, seed)
    # must give the same action and cost every time (spec Sec. 8).
    results = []
    for _ in range(3):
        sel = build_selector(
            FakeMechanism(NUDGE_COST, mutate_in_place),
            FakeMechanism(REBUILD_COST, flip_label),
            seed=7,
        )
        model = FakeModel(label=0)
        results.append(sel.adapt(model, X, y, reference_accuracy=1.0))
    assert len({r.action for r in results}) == 1
    assert len({r.cost.work_units for r in results}) == 1


def test_scoring_uses_only_holdout():
    X, y = make_window(n=300, seed=4)
    seen_lengths = []

    def spy_scorer(y_true, y_pred):
        seen_lengths.append(len(y_true))
        return float(np.mean(y_true == y_pred))

    sel = build_selector(
        FakeMechanism(NUDGE_COST, mutate_in_place),
        FakeMechanism(REBUILD_COST, flip_label),
        scorer=spy_scorer,
    )
    model = FakeModel(label=0)
    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert seen_lengths, "scorer was never called"
    assert all(n == result.n_holdout_rows for n in seen_lengths)


# --- action-reachability tests ---------------------------------------------


def test_skip_when_model_still_good():
    X, y = make_window(n=1000, seed=5)
    sel = build_selector(
        FakeMechanism(NUDGE_COST, mutate_in_place),
        FakeMechanism(REBUILD_COST, flip_label),
    )
    model = FakeModel(label=1)  # matches y exactly -> 100% on holdout

    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.action == SKIP
    assert result.cost == CostRecord.zero()
    assert result.model is model, "SKIP must return the identical object"


def test_nudge_when_repair_recovers():
    X, y = make_window(n=1000, seed=6)
    nudge = FakeMechanism(NUDGE_COST, mutate_in_place)  # label 0 -> 1: fixes it
    sel = build_selector(nudge, FakeMechanism(REBUILD_COST, flip_label))
    model = FakeModel(label=0)  # wrong -> triggers an attempt

    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.action == NUDGE
    assert result.cost == NUDGE_COST


def test_rebuild_when_nudge_insufficient():
    X, y = make_window(n=1000, seed=7)
    # nudge "fixes" 0 -> 1, but the model was already wrong in a way the
    # nudge can't repair: use a transform that keeps it wrong.
    still_wrong = lambda model, X, y: FakeModel(label=0)
    nudge = FakeMechanism(NUDGE_COST, still_wrong)
    sel = build_selector(nudge, FakeMechanism(REBUILD_COST, flip_label))
    model = FakeModel(label=0)

    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.action == REBUILD


# --- cost-accounting tests ---------------------------------------------


def test_rebuild_cost_includes_failed_nudge():
    X, y = make_window(n=1000, seed=8)
    still_wrong = lambda model, X, y: FakeModel(label=0)
    nudge = FakeMechanism(NUDGE_COST, still_wrong)
    rebuild = FakeMechanism(REBUILD_COST, flip_label)
    sel = build_selector(nudge, rebuild)
    model = FakeModel(label=0)

    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.action == REBUILD
    assert result.cost.work_units == NUDGE_COST.work_units + REBUILD_COST.work_units
    assert result.cost.work_units > REBUILD_COST.work_units, (
        "must be strictly greater than a standalone rebuild"
    )


def test_skip_cost_is_exactly_zero():
    X, y = make_window(n=1000, seed=9)
    sel = build_selector(
        FakeMechanism(NUDGE_COST, mutate_in_place),
        FakeMechanism(REBUILD_COST, flip_label),
    )
    model = FakeModel(label=1)

    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.action == SKIP
    assert result.cost == CostRecord.zero()


# --- safety tests ---------------------------------------------


def test_input_model_not_mutated_on_rebuild():
    X, y = make_window(n=1000, seed=10)
    still_wrong = lambda model, X, y: FakeModel(label=0)
    nudge = FakeMechanism(NUDGE_COST, still_wrong)  # would mutate if not copied
    rebuild = FakeMechanism(REBUILD_COST, flip_label)
    sel = build_selector(nudge, rebuild)
    model = FakeModel(label=0)
    label_before = model.label

    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.action == REBUILD
    assert model.label == label_before, "caller's model must be untouched"


def test_input_model_not_mutated_on_skip():
    X, y = make_window(n=1000, seed=11)
    sel = build_selector(
        FakeMechanism(NUDGE_COST, mutate_in_place),
        FakeMechanism(REBUILD_COST, flip_label),
    )
    model = FakeModel(label=1)
    label_before = model.label

    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.action == SKIP
    assert model.label == label_before


def test_small_window_degrades_to_rebuild():
    X, y = make_window(n=60, seed=12)  # default thresholds: 60*0.2=12<50
    sel = build_selector(
        FakeMechanism(NUDGE_COST, mutate_in_place),
        FakeMechanism(REBUILD_COST, flip_label),
    )
    model = FakeModel(label=1)  # would SKIP at full size; must not here

    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.action == REBUILD
    assert result.degraded_to_rebuild is True


@pytest.mark.parametrize(
    "n, real_holdout, should_degrade",
    [
        # Default holdout_frac=0.2, min_holdout_rows=50. For n in 246..249 the
        # approximation n * 0.2 falls just under 50 while _split really yields
        # exactly 50 holdout rows -- the old guard degraded these wrongly.
        (246, 50, False),
        (249, 50, False),
        (250, 50, False),
        (245, 49, True),
    ],
)
def test_degraded_guard_uses_real_split_lengths(n, real_holdout, should_degrade):
    X, y = make_window(n=n, seed=14)
    sel = build_selector(
        FakeMechanism(NUDGE_COST, mutate_in_place),
        FakeMechanism(REBUILD_COST, flip_label),
    )
    _, _, X_holdout, _ = sel._split(X, y)
    assert len(X_holdout) == real_holdout, "fixture assumption about _split broke"

    model = FakeModel(label=1)  # 100% accurate: SKIPs whenever not degraded
    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.degraded_to_rebuild is should_degrade
    assert result.action == (REBUILD if should_degrade else SKIP)
    assert result.n_holdout_rows == real_holdout, (
        "the recorded holdout size must be the one the decision was based on"
    )


def test_rebuild_is_handed_no_model_to_inherit_from():
    """The rebuild mechanism owns construction. The selector must not hand it
    the live model: a rebuild that snapshotted it as a cost baseline would
    subtract the old model's estimators and bill itself at near zero."""
    X, y = make_window(n=1000, seed=15)
    still_wrong = lambda model, X, y: FakeModel(label=0)
    rebuild = FakeMechanism(REBUILD_COST, flip_label)
    sel = build_selector(FakeMechanism(NUDGE_COST, still_wrong), rebuild)

    result = sel.adapt(FakeModel(label=0), X, y, reference_accuracy=1.0)

    assert result.action == REBUILD
    assert len(rebuild.calls) == 1
    assert rebuild.calls[0][0] is None


def test_degraded_rebuild_is_also_handed_no_model():
    X, y = make_window(n=60, seed=16)
    rebuild = FakeMechanism(REBUILD_COST, flip_label)
    sel = build_selector(FakeMechanism(NUDGE_COST, mutate_in_place), rebuild)

    result = sel.adapt(FakeModel(label=1), X, y, reference_accuracy=1.0)

    assert result.degraded_to_rebuild is True
    assert rebuild.calls[0][0] is None


def test_floor_is_recorded():
    X, y = make_window(n=1000, seed=13)
    sel = build_selector(
        FakeMechanism(NUDGE_COST, mutate_in_place),
        FakeMechanism(REBUILD_COST, flip_label),
        floor_drop=0.03,
    )
    model = FakeModel(label=1)

    result = sel.adapt(model, X, y, reference_accuracy=0.9)

    assert result.floor == pytest.approx(0.9 - 0.03)


# --- integration test ---------------------------------------------


@pytest.mark.parametrize("name", GRID_MODELS)
def test_all_four_model_families(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    classes = np.array([0, 1])

    spec = make_spec(name, classes)
    model = spec.factory(0).fit(X, y)

    sel = MechanismSelector(
        nudge_mechanism=NudgeMechanism(spec),
        rebuild_mechanism=RebuildMechanism(spec),
        seed=0,
    )

    result = sel.adapt(model, X, y, reference_accuracy=1.0)

    assert result.action in (SKIP, NUDGE, REBUILD)
    assert isinstance(result, AdaptResult)
