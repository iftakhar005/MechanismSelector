"""Confirm-before-acting and exact inference tracking (docs/confirm/PREREGISTRATION.md)."""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import synthetic  # noqa: E402
from runner import run_stream  # noqa: E402
from test_runner import CFG, TIMING_FIELDS, drifting_stream, n_init_of, scripted  # noqa: E402

warnings.filterwarnings("ignore")

EXTRA = {"n_alarms_dismissed", "dismissed_alarms", "total_inference_ops", "config_json", "prequential_errors"}


def _strip(record):
    return {k: v for k, v in record.items() if k not in TIMING_FIELDS | EXTRA}


def _strip_events(events):
    return [{k: v for k, v in e.items() if k != "inference_units_after"} for e in events]


@pytest.mark.parametrize("policy_key", ["always_rebuild", "mechanism_selector"])
def test_neither_option_changes_behaviour_when_off_or_tracking_only(policy_key):
    X, y = drifting_stream(n=3000)
    steps = [400, 1200, 2100]
    base_r, base_e = run_stream(X, y, "xgb", policy_key, 0, CFG, detector_factory=scripted(steps))
    off_r, off_e = run_stream(X, y, "xgb", policy_key, 0, replace(CFG, confirm_rows=0),
                              detector_factory=scripted(steps))
    trk_r, trk_e = run_stream(X, y, "xgb", policy_key, 0, replace(CFG, track_inference=True),
                              detector_factory=scripted(steps))
    assert _strip(base_r) == _strip(off_r) == _strip(trk_r)
    assert _strip_events(base_e) == _strip_events(off_e) == _strip_events(trk_e)
    assert base_r["n_alarms_dismissed"] == 0 and base_r["total_inference_ops"] is None


def test_alarm_on_a_stationary_stream_is_dismissed_at_no_cost():
    X, y = drifting_stream(n=3000, n_segments=1)
    cfg = replace(CFG, confirm_rows=100)
    record, events = run_stream(X, y, "rf", "always_rebuild", 0, cfg, detector_factory=scripted([600]))
    assert events == []
    assert record["total_work_units"] == 0
    assert record["n_alarms_dismissed"] == 1
    d = record["dismissed_alarms"][0]
    n_init = n_init_of(X)
    assert d["alarm_row"] == n_init + 600
    assert d["decide_row"] - d["alarm_row"] == 100           # decided on exactly W rows, all already labelled
    assert d["confirm_score"] >= d["reference_accuracy"] - CFG.floor_drop


def test_alarm_after_a_real_change_is_confirmed_and_adapts_W_rows_later():
    X, y = drifting_stream(n=3000, n_segments=2)
    n_init = n_init_of(X)
    step = 1500 - n_init + 5                                   # just after the change at row 1500
    cfg = replace(CFG, confirm_rows=100)
    record, events = run_stream(X, y, "rf", "always_rebuild", 0, cfg, detector_factory=scripted([step]))
    assert record["n_alarms_dismissed"] == 0
    assert len(events) == 1
    assert events[0]["adapt_row"] - events[0]["alarm_row"] == 100
    assert events[0]["buffer_rows"] >= 100                     # the confirmation rows are in the buffer


def test_alarms_during_the_wait_coalesce():
    X, y = drifting_stream(n=3000, n_segments=2)
    n_init = n_init_of(X)
    first = 1500 - n_init + 5
    cfg = replace(CFG, confirm_rows=100)
    record, events = run_stream(X, y, "rf", "always_rebuild", 0, cfg,
                                detector_factory=scripted([first, first + 30, first + 60]))
    assert len(events) == 1 and record["n_alarms_coalesced"] == 2


def test_inference_ops_are_exact_rows_times_units():
    X, y = drifting_stream(n=3000)
    cfg = replace(CFG, track_inference=True)
    never, _ = run_stream(X, y, "xgb", "never_adapt", 0, cfg, detector_factory=scripted([]))
    assert never["total_inference_ops"] == pytest.approx(never["n_stream_rows"] * never["initial_inference_units"])

    rec, events = run_stream(X, y, "xgb", "always_rebuild", 0, cfg, detector_factory=scripted([400, 1500]))
    n, n_init = len(X), n_init_of(X)
    bounds = [n_init] + [e["adapt_row"] + 1 for e in events] + [n]
    units = [rec["initial_inference_units"]] + [e["inference_units_after"] for e in events]
    expected = sum(u * (b - a) for u, a, b in zip(units, bounds[:-1], bounds[1:]))
    assert rec["total_inference_ops"] == pytest.approx(expected)
    assert units[-1] == pytest.approx(rec["final_inference_units"])


@pytest.mark.parametrize("name", synthetic.STREAMS)
def test_synthetic_streams_are_deterministic_and_well_formed(name):
    X1, y1, cp1 = synthetic.make_stream(name, 3, n_rows=4000)
    X2, y2, cp2 = synthetic.make_stream(name, 3, n_rows=4000)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2) and cp1 == cp2
    assert len(X1) == len(y1) == 4000
    assert (cp1 == ()) == ("_none_" in name)


def test_sticky_sampling_adds_label_persistence_without_changing_rows():
    _, y_iid, _ = synthetic.make_stream("sea_abrupt_iid", 0, n_rows=8000)
    _, y_sticky, _ = synthetic.make_stream("sea_abrupt_sticky", 0, n_rows=8000)
    persist = lambda y: np.mean(y[1:] == y[:-1])  # noqa: E731
    assert persist(y_sticky) > persist(y_iid) + 0.25
