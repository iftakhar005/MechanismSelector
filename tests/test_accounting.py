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
    CostRecord,
    count_passes,
    estimator_count,
    measure,
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
    baseline = estimator_count(model)
    assert baseline == 100

    def grow():
        model.n_estimators += 5
        return model.fit(X, y)

    _, cost = measure(grow, n_rows=len(X), baseline_estimators=baseline)
    assert cost.n_estimators_fitted == 5, "must charge added trees, not all 105"
    assert cost.work_units == 5 * N_ROWS


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
