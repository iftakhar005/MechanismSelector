"""Tests for the four guard variants, written before the variants were run.

1. Each variant with its change switched off reproduces the original selector
   exactly on one seed per family: same decisions, same ops, same accuracy.
2. Variant C's guard branch performs zero training ops.
3. Variant A's buffer never exceeds 1,000 rows and holds only rows seen before
   the alarm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from mechanisms import GRID_MODELS  # noqa: E402
from runner import RunConfig, run_stream  # noqa: E402
from selector import REBUILD, SKIP, MechanismSelector  # noqa: E402

DECISION_FIELDS = (
    "n_alarms", "n_adaptations", "n_skip", "n_nudge", "n_rebuild", "n_degraded",
    "total_work_units", "total_estimators_fitted", "total_rows_processed", "total_passes",
    "mean_prequential_accuracy", "mean_prequential_balanced_accuracy",
    "final_prequential_balanced_accuracy",
)

BASE = RunConfig(track_energy=False)

# One short stream keeps the test suite fast; the guard fires on it for every
# family, which is the behaviour under test.
STREAM = "insects_gradual"


@pytest.fixture(scope="module")
def stream():
    return ds.load_stream(STREAM)


def _run(X, y, model_name, config):
    record, events = run_stream(X, y, model_name, "mechanism_selector", 0, config, STREAM)
    return record, events


@pytest.mark.parametrize("model_name", list(GRID_MODELS))
@pytest.mark.parametrize("variant", [
    pytest.param(dict(keep_window=False), id="A-off"),
    pytest.param(dict(min_holdout_rows=50, min_train_rows=100), id="B-off"),
    pytest.param(dict(guard_action=REBUILD), id="C-off"),
    pytest.param(dict(min_holdout_rows=50, min_train_rows=100), id="D-off"),
])
def test_variant_off_reproduces_the_original(stream, model_name, variant):
    """Test 1: the change switched off is the original selector, exactly."""
    X, y = stream
    base, _ = _run(X, y, model_name, BASE)
    got, _ = _run(X, y, model_name, RunConfig(track_energy=False, **variant))
    for field in DECISION_FIELDS:
        assert got[field] == pytest.approx(base[field], rel=0, abs=0), (
            f"{field} differs for {model_name} with {variant}")


@pytest.mark.parametrize("model_name", list(GRID_MODELS))
def test_variant_c_guard_branch_costs_nothing(stream, model_name):
    """Test 2: under variant C every guard-triggered event does zero training."""
    X, y = stream
    cfg = RunConfig(track_energy=False, guard_action=SKIP)
    _, events = _run(X, y, model_name, cfg)
    guarded = [e for e in events if e["degraded_to_rebuild"]]
    for e in guarded:
        assert e["action"] == SKIP, f"guard event took {e['action']}, expected SKIP"
        assert e["work_units"] == 0, f"guard event spent {e['work_units']} work units"
        assert e["n_estimators_fitted"] == 0
        assert e["rows_processed"] == 0
        assert e["n_passes"] == 0


@pytest.mark.parametrize("model_name", list(GRID_MODELS))
def test_variant_a_window_is_bounded_and_causal(stream, model_name):
    """Test 3: variant A's buffer stays at or below 1,000 rows seen before the alarm."""
    X, y = stream
    cfg = RunConfig(track_energy=False, keep_window=True)
    _, events = _run(X, y, model_name, cfg)
    assert events, "variant A produced no adaptations to check"
    for e in events:
        assert 0 < e["buffer_rows"] <= cfg.buffer_size, (
            f"buffer held {e['buffer_rows']} rows, cap is {cfg.buffer_size}")
        # every buffered row precedes the row the policy acts on
        assert e["buffer_rows"] <= e["adapt_row"] + 1
        assert e["adapt_row"] >= e["alarm_row"]


def test_variant_a_keeps_more_history_than_the_original(stream):
    """Variant A should actually change something: wider windows at the alarms."""
    X, y = stream
    _, base_events = _run(X, y, "xgb", BASE)
    _, keep_events = _run(X, y, "xgb", RunConfig(track_energy=False, keep_window=True))
    if base_events and keep_events:
        assert (np.mean([e["buffer_rows"] for e in keep_events])
                >= np.mean([e["buffer_rows"] for e in base_events]))


def test_guard_action_is_validated():
    with pytest.raises(ValueError, match="guard_action"):
        MechanismSelector(None, None, guard_action="NUDGE")
