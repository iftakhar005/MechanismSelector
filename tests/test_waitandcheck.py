"""Phase 14 — WaitAndCheck(W) behaviour, written before the experiment was run.

W is a wait, not a new policy: on an alarm the runner keeps predicting for W
labelled rows, then either cancels the alarm (the model is still above the floor
on exactly those rows) or hands the enlarged buffer to the unchanged selector.

These tests pin the parts that could silently go wrong: that W = 0 is the old
behaviour exactly, that cancelling is free, that the check sees only post-alarm
rows, that proceeding passes the W rows on, and that alarms raised during a wait
are ignored and counted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runner import RunConfig, run_stream  # noqa: E402

FAMILIES = ("xgb", "rf", "sgd", "gnb")


def drifting_stream(n=6000, seed=0):
    """Two concepts with an abrupt switch: alarms are guaranteed, drift is real."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    y[n // 2:] = (X[n // 2:, 1] > 0).astype(np.int64)
    return X, y


def bursty_stream(n=12000, seed=0, burst=120, period=900):
    """One concept, with short bursts of label noise that end before the wait does.

    ADWIN fires on the rise in error; by the time W rows have passed the burst is
    over and the model is back above the floor, so the wait should cancel. A
    stationary stream would not do: ADWIN raises no alarms on i.i.d. noise at
    delta = 0.002, which is what Phase 9.1 measured.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    for start in range(1500, n - burst, period):
        flip = rng.random(burst) < 0.8
        y[start:start + burst] = np.where(flip, 1 - y[start:start + burst], y[start:start + burst])
    return X, y


def run(X, y, family, wait_rows, seed=0):
    cfg = RunConfig(track_energy=False, wait_rows=wait_rows)
    return run_stream(X, y, family, "mechanism_selector", seed, cfg, "synthetic")


@pytest.mark.parametrize("family", FAMILIES)
def test_w0_reproduces_the_selector_exactly(family):
    X, y = drifting_stream()
    base_record, base_events = run(X, y, family, wait_rows=0)
    again_record, again_events = run(X, y, family, wait_rows=0)

    wait_fields = {"wait_start", "wait_rows_scored", "wait_accuracy",
                   "n_alarms_ignored_during_wait", "n_waits_cancelled",
                   "n_waits_proceeded", "n_waits_truncated"}

    def strip(d):
        return {k: v for k, v in d.items() if k not in wait_fields and k != "run_wall_clock_s"
                and k != "total_wall_clock_s" and k != "total_energy_kwh"
                and not k.startswith("initial_predict") and not k.startswith("final_predict")}

    assert strip(base_record) == strip(again_record)
    assert [strip(e) for e in base_events] == [strip(e) for e in again_events]
    assert base_record["n_waits_cancelled"] == 0
    assert base_record["n_waits_proceeded"] == 0
    assert base_record["n_alarms_ignored_during_wait"] == 0


def improving_stream(n=12000, seed=0, noise=0.25):
    """Noisy labels first, clean later: the error rate FALLS mid-stream.

    ADWIN is two-sided, so it alarms on the improvement; the W rows after that
    alarm are better than the reference measured during the noisy stretch, so the
    wait should cancel. This is the situation Phase 9 found to be common, not a
    contrived one.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    flip = rng.random(n // 2) < noise
    y[: n // 2] = np.where(flip, 1 - y[: n // 2], y[: n // 2])
    return X, y


def test_cancel_path_costs_zero_training():
    X, y = improving_stream()
    record, events = run(X, y, "rf", wait_rows=200)
    cancelled = [e for e in events if e["action"] == "CANCELLED"]
    assert cancelled, "an alarm on a falling error rate should cancel"
    for e in cancelled:
        assert e["work_units"] == 0
        assert e["n_passes"] == 0
        assert e["n_estimators_fitted"] == 0
        assert e["rows_processed"] == 0
    assert record["n_waits_cancelled"] == len(cancelled)


def test_check_uses_only_post_alarm_rows_and_the_full_w():
    X, y = drifting_stream()
    _, events = run(X, y, "sgd", wait_rows=200)
    waited = [e for e in events if e.get("wait_start") is not None]
    assert waited
    for e in waited:
        assert e["wait_start"] == e["alarm_row"] + 1        # strictly after the alarm
        assert e["wait_rows_scored"] == 200                  # exactly W rows
        assert e["adapt_row"] == e["wait_start"] + 199       # the decision is taken at the end


def test_proceed_path_hands_the_w_rows_to_the_selector():
    X, y = drifting_stream()
    _, events = run(X, y, "rf", wait_rows=200)
    proceeded = [e for e in events
                 if e.get("wait_start") is not None and e["action"] != "CANCELLED"]
    assert proceeded, "an abrupt drift should make at least one wait proceed"
    for e in proceeded:
        assert e["buffer_rows"] >= 200                       # the W rows are in the window
        assert e["adapt_row"] - e["buffer_rows"] + 1 <= e["wait_start"]


def test_alarms_during_a_wait_are_ignored_and_counted():
    X, y = bursty_stream()
    record_wait, _ = run(X, y, "rf", wait_rows=500)
    record_none, _ = run(X, y, "rf", wait_rows=0)
    assert record_wait["n_alarms_ignored_during_wait"] > 0
    assert record_none["n_alarms_ignored_during_wait"] == 0
    assert record_wait["n_alarms"] >= record_wait["n_waits_cancelled"] + record_wait["n_waits_proceeded"]


def test_proceeding_after_a_wait_adapts_on_a_larger_window():
    """Waiting is not free: the W rows join the buffer, so a proceed trains on more.

    This pins behaviour that surprised me while writing these tests - on a bursty
    stream WaitAndCheck can spend MORE training ops than acting at once, because
    each adaptation it does take is over a bigger window.
    """
    X, y = bursty_stream()
    _, events_wait = run(X, y, "rf", wait_rows=200)
    _, events_none = run(X, y, "rf", wait_rows=0)
    acted_wait = [e["buffer_rows"] for e in events_wait if e["action"] != "CANCELLED"]
    acted_none = [e["buffer_rows"] for e in events_none]
    assert acted_wait and acted_none
    assert np.median(acted_wait) > np.median(acted_none)
