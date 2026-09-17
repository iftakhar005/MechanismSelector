"""Phase 2 acceptance tests for operation accounting."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from accounting import (  # noqa: E402
    PASS_N_ITER,
    PASS_SINGLE,
    PASS_TREES,
    Baseline,
    CostRecord,
    count_passes,
    estimator_count,
    iter_count,
    measure,
    resolve_iter_passes,
    snapshot,
)

warnings.filterwarnings("ignore")

N_ROWS = 10_000


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N_ROWS, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def test_gate_300_trees_is_exactly_30x_10_trees(data):
    """PHASE 2 GATE: 300-tree fit records exactly 30x the work of a 10-tree fit."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = data

    def fit(n):
        return lambda: RandomForestClassifier(
            n_estimators=n, max_depth=3, random_state=0, n_jobs=1
        ).fit(X, y)

    _, small = measure(fit(10), n_rows=len(X))
    _, big = measure(fit(300), n_rows=len(X))

    assert small.work_units == 10 * N_ROWS
    assert big.work_units == 300 * N_ROWS
    assert big.work_units == 30 * small.work_units


def test_gate_holds_for_xgboost(data):
    """Same exactness requirement for boosting rounds."""
    from xgboost import XGBClassifier

    X, y = data

    def fit(n):
        return lambda: XGBClassifier(
            n_estimators=n, max_depth=3, random_state=0, n_jobs=1, verbosity=0
        ).fit(X, y)

    _, small = measure(fit(10), n_rows=len(X))
    _, big = measure(fit(300), n_rows=len(X))

    assert small.n_estimators_fitted == 10
    assert big.n_estimators_fitted == 300
    assert big.work_units == 30 * small.work_units


def test_work_units_is_deterministic(data):
    """Identical operations must produce identical work units across runs."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = data
    runs = [
        measure(
            lambda: RandomForestClassifier(
                n_estimators=25, max_depth=3, random_state=0, n_jobs=1
            ).fit(X, y),
            n_rows=len(X),
        )[1].work_units
        for _ in range(3)
    ]
    assert len(set(runs)) == 1


def test_incremental_fit_charges_only_added_trees(data):
    """warm_start accumulates; only the delta may be charged."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = data
    model = RandomForestClassifier(
        n_estimators=100, max_depth=3, random_state=0, n_jobs=1, warm_start=True
    ).fit(X, y)
    base = snapshot(model)
    assert base.n_estimators == 100

    def grow():
        model.n_estimators += 5
        return model.fit(X, y)

    _, cost = measure(grow, n_rows=len(X), baseline=base)
    assert cost.n_estimators_fitted == 5, "must charge added trees, not all 105"
    assert cost.work_units == 5 * N_ROWS


# --- n_iter_ semantics -------------------------------------------------------
#
# scikit-learn does not document whether repeated partial_fit accumulates
# n_iter_. These tests pin the behaviour we measured (1.9.0: it resets) and
# prove the accounting stays correct if a future version starts accumulating.


@pytest.mark.parametrize("kind", ["sgd", "mlp"])
def test_repeated_partial_fit_charges_one_pass_each_time(data, kind):
    """Five consecutive nudges on one model must cost five passes, not one."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier

    X, y = data
    model = (
        SGDClassifier(random_state=0)
        if kind == "sgd"
        else MLPClassifier(hidden_layer_sizes=(8,), random_state=0)
    )

    charged = []
    for _ in range(5):
        base = snapshot(model)
        _, cost = measure(
            lambda: model.partial_fit(X, y, classes=[0, 1]),
            n_rows=len(X),
            baseline=base,
        )
        charged.append(cost.n_passes)

    assert charged == [1, 1, 1, 1, 1], (
        f"each partial_fit is one epoch and must be charged as such; got {charged}"
    )


@pytest.mark.parametrize("kind", ["sgd", "mlp"])
def test_partial_fit_does_not_accumulate_n_iter_on_this_sklearn(data, kind):
    """Documents the observed 1.9.0 behaviour. If this fails, sklearn changed
    its convention -- resolve_iter_passes already handles it, but the README
    and the docstring in accounting.py need updating."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier

    X, y = data
    model = (
        SGDClassifier(random_state=0)
        if kind == "sgd"
        else MLPClassifier(hidden_layer_sizes=(8,), random_state=0)
    )

    observed = []
    for _ in range(5):
        model.partial_fit(X, y, classes=[0, 1])
        observed.append(iter_count(model))

    assert observed == [1, 1, 1, 1, 1], (
        f"sklearn n_iter_ convention changed: {observed}"
    )


def test_resolve_iter_passes_handles_accumulating_counters():
    """If n_iter_ accumulates (1, 2, 3...), charge the delta."""
    assert resolve_iter_passes(current=1, baseline=0) == 1
    assert resolve_iter_passes(current=2, baseline=1) == 1
    assert resolve_iter_passes(current=3, baseline=2) == 1
    assert resolve_iter_passes(current=36, baseline=0) == 36


def test_resolve_iter_passes_handles_resetting_counters():
    """If n_iter_ resets (1, 1, 1...), charge the raw value.

    A naive delta would give 1 - 1 = 0 here and record every nudge as free,
    which is worse than an overcharge: it would fabricate the result.
    """
    assert resolve_iter_passes(current=1, baseline=1) == 1
    assert resolve_iter_passes(current=1, baseline=5) == 1


def test_accounting_is_correct_under_a_hypothetical_accumulating_sklearn(data):
    """End-to-end proof the delta rule works if the convention flips."""

    class FakeAccumulatingModel:
        """Stands in for an sklearn version where partial_fit accumulates."""

        def __init__(self):
            self.n_iter_ = 0

        def partial_fit(self, *_args, **_kwargs):
            self.n_iter_ += 1
            return self

    model = FakeAccumulatingModel()
    charged = []
    for _ in range(5):
        base = snapshot(model)
        _, cost = measure(lambda: model.partial_fit(), n_rows=N_ROWS, baseline=base)
        charged.append(cost.n_passes)

    assert model.n_iter_ == 5, "fixture should accumulate"
    assert charged == [1, 1, 1, 1, 1], (
        f"accumulating counter must still charge one pass per call; got {charged}"
    )


def test_snapshot_captures_both_dimensions(data):
    from sklearn.ensemble import RandomForestClassifier

    X, y = data
    rf = RandomForestClassifier(n_estimators=7, max_depth=2, random_state=0).fit(X, y)
    assert snapshot(rf) == Baseline(n_estimators=7, n_iter=0)
    assert snapshot(None) == Baseline(n_estimators=0, n_iter=0)


def test_sgd_fit_and_partial_fit_are_distinguishable(data):
    """The core reason the flat 'one pass per call' convention was rejected."""
    from sklearn.linear_model import SGDClassifier

    X, y = data

    _, full = measure(
        lambda: SGDClassifier(max_iter=1000, tol=1e-3, random_state=0).fit(X, y),
        n_rows=len(X),
    )

    def one_epoch():
        m = SGDClassifier(random_state=0)
        m.partial_fit(X, y, classes=[0, 1])
        return m

    _, incremental = measure(one_epoch, n_rows=len(X))

    assert incremental.n_passes == 1
    assert full.n_passes > 1, "fit must record its real epoch count"
    assert full.work_units > incremental.work_units
    assert full.pass_source == PASS_N_ITER
    assert incremental.pass_source == PASS_N_ITER


def test_mlp_fit_and_partial_fit_are_distinguishable(data):
    from sklearn.neural_network import MLPClassifier

    X, y = data

    _, full = measure(
        lambda: MLPClassifier(
            hidden_layer_sizes=(8,), max_iter=30, random_state=0
        ).fit(X, y),
        n_rows=len(X),
    )

    def one_epoch():
        m = MLPClassifier(hidden_layer_sizes=(8,), random_state=0)
        m.partial_fit(X, y, classes=[0, 1])
        return m

    _, incremental = measure(one_epoch, n_rows=len(X))

    assert incremental.n_passes == 1
    assert full.n_passes > 1
    assert full.work_units > incremental.work_units


def test_gaussian_nb_is_a_single_pass(data):
    """GaussianNB is closed-form: one traversal, no epochs, no estimators."""
    from sklearn.naive_bayes import GaussianNB

    X, y = data
    _, cost = measure(lambda: GaussianNB().fit(X, y), n_rows=len(X))

    assert cost.n_passes == 1
    assert cost.n_estimators_fitted == 0
    assert cost.pass_source == PASS_SINGLE
    assert cost.work_units == N_ROWS


def test_pass_source_is_recorded_for_audit(data):
    """Every record must say where its pass count came from."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = data
    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0).fit(X, y)
    _, _, source = count_passes(model)
    assert source == PASS_TREES


def test_cost_records_accumulate():
    """A failed nudge followed by a rebuild must sum, not overwrite."""
    nudge = CostRecord(15, 15, PASS_TREES, 800, 15 * 800, 0.5, None)
    rebuild = CostRecord(300, 300, PASS_TREES, 1000, 300 * 1000, 9.0, None)
    total = nudge + rebuild

    assert total.work_units == 15 * 800 + 300 * 1000
    assert total.n_estimators_fitted == 315
    assert total.n_rows_processed == 1800
    assert total.wall_clock_s == pytest.approx(9.5)


def test_zero_record_is_additive_identity():
    rec = CostRecord(10, 10, PASS_TREES, 500, 5000, 1.0, None)
    assert (CostRecord.zero() + rec).work_units == rec.work_units
    assert (CostRecord.zero() + rec).pass_source == PASS_TREES


def test_energy_is_never_required(data):
    """Energy may be None and must not break accounting."""
    from sklearn.naive_bayes import GaussianNB

    X, y = data
    _, cost = measure(lambda: GaussianNB().fit(X, y), n_rows=len(X))
    assert cost.energy_kwh is None
    assert cost.work_units > 0


def test_placeholder_rows_accumulate_and_are_never_charged():
    nudge = CostRecord(5, 5, PASS_TREES, 800, 5 * 800, 0.1, None, n_placeholder_rows=2)
    rebuild = CostRecord(100, 100, PASS_TREES, 1000, 100 * 1000, 1.0, None, n_placeholder_rows=3)
    total = nudge + rebuild

    assert total.n_placeholder_rows == 5
    assert total.n_rows_processed == 1800, "placeholders are not training rows"
    assert total.work_units == 5 * 800 + 100 * 1000
    assert CostRecord.zero().n_placeholder_rows == 0
