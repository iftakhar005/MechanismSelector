"""Phase 6 tests for the experiment runner.

Most tests replace ADWIN with a scripted detector so alarms fire on exact rows,
and wrap the real policy in a spy that records what `adapt()` was given. The
chunked-prediction test uses real ADWIN on a genuinely drifting stream.
"""

from __future__ import annotations

import csv
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechanisms import make_spec  # noqa: E402
from policies import POLICY_KEYS, make_policy  # noqa: E402
from runner import (  # noqa: E402
    POLICY_ORDER,
    RECORD_FIELDS,
    RunConfig,
    grid_plan,
    run_experiment,
    run_grid,
    run_stream,
)
from selector import NUDGE, REBUILD  # noqa: E402

warnings.filterwarnings("ignore")

CFG = RunConfig(
    init_frac=0.10, buffer_size=400, min_buffer=50, n_ref=100,
    eval_block=300, probe_rows=300, chunk_size=500, track_energy=False,
)
TIMING_FIELDS = {
    "total_wall_clock_s", "run_wall_clock_s", "total_energy_kwh",
    "initial_predict_us_per_row", "final_predict_us_per_row",
}


def drifting_stream(n=3000, n_classes=2, seed=0, marker=False, n_segments=3):
    """Abrupt concept changes between equal segments, so ADWIN genuinely alarms."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = np.empty(n, dtype=np.int64)
    edges = np.linspace(0, n, n_segments + 1).astype(int)
    for start, stop in zip(edges[:-1], edges[1:]):
        scores = X[start:stop] @ rng.normal(size=(n_classes, 5)).T
        y[start:stop] = scores.argmax(axis=1)
    if marker:
        X[:, 0] = np.arange(n, dtype=float)
    return X, y


class ScriptedDetector:
    """Alarms on exact update steps; step 0 is the first stream row."""

    def __init__(self, alarm_steps):
        self.alarm_steps = set(alarm_steps)
        self.step = -1
        self._fired = False

    def update(self, _error):
        self.step += 1
        self._fired = self.step in self.alarm_steps

    @property
    def drift_detected(self):
        return self._fired


def scripted(steps):
    return lambda _config: ScriptedDetector(steps)


def spying(calls):
    def factory(key, spec, **kwargs):
        inner = make_policy(key, spec, **kwargs)

        class Spy:
            name = inner.name

            def adapt(self, model, X, y, reference_accuracy):
                result = inner.adapt(model, X, y, reference_accuracy)
                calls.append({"X": X.copy(), "y": y.copy(), "ref": reference_accuracy,
                              "returned": result.model, "result": result})
                return result

        return Spy()

    return factory


def n_init_of(X, cfg=CFG):
    return int(len(X) * cfg.init_frac)


# --- chunked prediction is exact -------------------------------------------------


@pytest.mark.parametrize(
    "model_name, policy_key",
    [("xgb", "mechanism_selector"), ("sgd", "fixed_schedule"), ("gnb", "always_nudge")],
)
def test_chunked_prediction_is_exactly_equivalent_to_row_by_row(model_name, policy_key):
    X, y = drifting_stream(n=4000, n_segments=8)
    row_by_row = RunConfig(**{**CFG.__dict__, "chunk_size": 1})
    chunked = RunConfig(**{**CFG.__dict__, "chunk_size": 700})

    rec_a, ev_a = run_stream(X, y, model_name, policy_key, 0, row_by_row)
    rec_b, ev_b = run_stream(X, y, model_name, policy_key, 0, chunked)

    assert rec_a["n_adaptations"] >= 2, "precondition: the stream must actually trigger adaptations"
    strip = lambda r: {k: v for k, v in r.items() if k not in TIMING_FIELDS | {"config_json"}}  # noqa: E731
    assert strip(rec_a) == strip(rec_b)
    assert ev_a == ev_b


# --- minimum-buffer guard ----------------------------------------------------------


def test_alarm_on_a_small_buffer_is_deferred_until_min_buffer():
    X, y = drifting_stream()
    n_init = n_init_of(X)
    record, events = run_stream(X, y, "gnb", "always_rebuild", 0, CFG, detector_factory=scripted({10}))

    assert len(events) == 1
    ev = events[0]
    assert ev["alarm_row"] == n_init + 10
    assert ev["adapt_row"] == n_init + CFG.min_buffer - 1, "adapts the moment the buffer reaches min_buffer"
    assert ev["buffer_rows"] == CFG.min_buffer
    assert ev["deferred_rows"] == CFG.min_buffer - 1 - 10
    assert record["n_alarms_deferred"] == 1


def test_alarms_during_a_deferral_are_coalesced_into_one_adaptation():
    X, y = drifting_stream()
    record, events = run_stream(X, y, "gnb", "always_rebuild", 0, CFG,
                                detector_factory=scripted({5, 15, 25}))
    assert len(events) == 1
    assert record["n_alarms"] == 3
    assert record["n_alarms_coalesced"] == 2


@pytest.mark.parametrize("policy_key", POLICY_KEYS)
def test_no_policy_ever_receives_a_window_below_min_buffer(policy_key):
    X, y = drifting_stream()
    calls = []
    run_stream(X, y, "gnb", policy_key, 0, CFG, detector_factory=scripted(range(0, 3000, 7)),
               policy_factory=spying(calls))

    assert len(calls) > 5
    assert all(CFG.min_buffer <= len(c["X"]) <= CFG.buffer_size for c in calls)


def test_pending_alarm_that_never_reaches_min_buffer_is_unserved():
    X, y = drifting_stream()
    n_stream = len(X) - n_init_of(X)
    # adapt near the end, then alarm with too few rows left to fill min_buffer
    record, events = run_stream(X, y, "gnb", "always_rebuild", 0, CFG,
                                detector_factory=scripted({n_stream - 30, n_stream - 20}))
    assert len(events) == 1
    assert record["n_alarms_unserved"] == 1


# --- the window handed to adapt() -------------------------------------------------


def test_window_is_the_contiguous_most_recent_rows_since_the_last_adaptation():
    X, y = drifting_stream(marker=True)
    calls = []
    _, events = run_stream(X, y, "gnb", "always_rebuild", 0, CFG,
                           detector_factory=scripted({60, 130, 900, 905}), policy_factory=spying(calls))

    previous_adapt = n_init_of(X) - 1
    for call, ev in zip(calls, events):
        rows = call["X"][:, 0].astype(int)
        assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)), "window must be contiguous and in order"
        assert rows[-1] == ev["adapt_row"], "window must end at the current row"
        assert rows[0] == max(previous_adapt + 1, ev["adapt_row"] + 1 - CFG.buffer_size)
        previous_adapt = ev["adapt_row"]


# --- reference accuracy -----------------------------------------------------------


def test_initial_reference_is_scored_on_the_init_holdout():
    X, y = drifting_stream()
    n_init = n_init_of(X)
    n_train = int(n_init * (1 - CFG.init_holdout_frac))
    calls = []
    run_stream(X, y, "gnb", "always_rebuild", 0, CFG, detector_factory=scripted({200}),
               policy_factory=spying(calls))

    initial = make_spec("gnb", np.unique(y)).factory(0).fit(X[:n_train], y[:n_train])
    expected = accuracy_score(y[n_train:n_init], initial.predict(X[n_train:n_init]))
    assert calls[0]["ref"] == pytest.approx(expected)


def test_reference_is_the_returned_models_score_on_the_next_n_ref_rows():
    X, y = drifting_stream()
    calls = []
    _, events = run_stream(X, y, "gnb", "always_rebuild", 0, CFG,
                           detector_factory=scripted({100, 700}), policy_factory=spying(calls))

    first_adapt = events[0]["adapt_row"]
    ref_rows = slice(first_adapt + 1, first_adapt + 1 + CFG.n_ref)
    expected = accuracy_score(y[ref_rows], calls[0]["returned"].predict(X[ref_rows]))

    assert events[1]["reference_carried"] is False
    assert calls[1]["ref"] == pytest.approx(expected)


def test_reference_is_carried_forward_when_the_next_alarm_comes_too_soon():
    X, y = drifting_stream()
    calls = []
    record, events = run_stream(X, y, "gnb", "always_rebuild", 0, CFG,
                                detector_factory=scripted({100, 170}), policy_factory=spying(calls))

    assert events[1]["adapt_row"] - events[0]["adapt_row"] < CFG.n_ref, "precondition"
    assert events[1]["reference_carried"] is True
    assert calls[1]["ref"] == calls[0]["ref"]
    assert record["n_reference_carried"] == 1


# --- record integrity -------------------------------------------------------------


@pytest.mark.parametrize("policy_key", POLICY_KEYS)
def test_record_totals_agree_with_the_event_log(policy_key):
    X, y = drifting_stream(n_classes=3)
    record, events = run_stream(X, y, "xgb", policy_key, 0, CFG, detector_factory=scripted(range(0, 3000, 150)))

    assert set(RECORD_FIELDS) <= set(record)
    assert record["n_skip"] + record["n_nudge"] + record["n_rebuild"] == record["n_adaptations"] == len(events)
    assert record["total_work_units"] == sum(e["work_units"] for e in events)
    if events:
        assert events[-1]["cumulative_work_units"] == record["total_work_units"]
    assert record["scorer"] == "balanced_accuracy"


def test_never_adapt_prequential_accuracy_matches_an_independent_computation():
    X, y = drifting_stream()
    n_init = n_init_of(X)
    n_train = int(n_init * (1 - CFG.init_holdout_frac))
    record, _ = run_stream(X, y, "gnb", "never_adapt", 0, CFG)

    model = make_spec("gnb", np.unique(y)).factory(0).fit(X[:n_train], y[:n_train])
    preds = model.predict(X[n_init:])
    assert record["mean_prequential_accuracy"] == pytest.approx(accuracy_score(y[n_init:], preds))
    assert record["final_prequential_accuracy"] == pytest.approx(
        accuracy_score(y[-CFG.eval_block:], preds[-CFG.eval_block:])
    )


# --- footprint: the inference-cost objection ---------------------------------------


def test_always_nudge_xgboost_grows_by_exactly_its_nudges():
    X, y = drifting_stream()
    record, _ = run_stream(X, y, "xgb", "always_nudge", 0, CFG, detector_factory=scripted(range(0, 3000, 120)))

    assert record["n_nudge"] > 3
    assert record["model_size_unit"] == "rounds"
    assert record["final_model_size"] == record["initial_model_size"] + 5 * record["n_nudge"]
    assert record["inference_growth"] > 1.0


def test_selector_xgboost_size_resets_on_rebuild_and_grows_only_by_later_nudges():
    X, y = drifting_stream()
    record, events = run_stream(X, y, "xgb", "mechanism_selector", 0, CFG,
                                detector_factory=scripted(range(0, 3000, 120)))

    for ev in events:
        if ev["action"] == REBUILD:
            assert ev["model_size_after"] == 100
    last_rebuild = max((i for i, e in enumerate(events) if e["action"] == REBUILD), default=-1)
    nudges_since = sum(1 for e in events[last_rebuild + 1:] if e["action"] == NUDGE)
    base = 100 if last_rebuild >= 0 else record["initial_model_size"]
    assert record["final_model_size"] == base + 5 * nudges_since


def test_random_forest_stays_the_same_size_under_always_nudge():
    X, y = drifting_stream()
    record, _ = run_stream(X, y, "rf", "always_nudge", 0, CFG, detector_factory=scripted(range(0, 3000, 400)))
    assert record["n_nudge"] > 2
    assert record["final_model_size"] == record["initial_model_size"] == 100


# --- robustness: windows missing classes ---------------------------------------------


@pytest.mark.parametrize("model_name", ["xgb", "rf", "sgd", "gnb"])
@pytest.mark.parametrize("policy_key", ["always_rebuild", "always_nudge", "mechanism_selector"])
def test_single_class_stretches_and_an_init_missing_a_class_do_not_crash(model_name, policy_key):
    X, y = drifting_stream(n=3000, n_classes=3)
    y[:300] = np.where(y[:300] == 2, 0, y[:300])      # class 2 absent from the init slice
    y[1200:1900] = 1                                    # a long single-class stretch
    alarms = {950, 1100, 1300, 1500, 1750}              # adaptations inside that stretch

    record, events = run_stream(X, y, model_name, policy_key, 0, CFG, detector_factory=scripted(alarms))

    assert record["n_adaptations"] == len(events) >= 3


# --- spec interface and grid ------------------------------------------------------------


def test_run_experiment_keeps_the_spec_signature():
    params = list(inspect.signature(run_experiment).parameters)
    assert params[:6] == ["dataset", "model_name", "policy", "seed", "init_frac", "buffer_size"]


def test_grid_plan_is_seed_outermost_with_key_comparisons_first():
    plan = grid_plan(range(2), ["d1", "d2"], ["xgb", "rf"], POLICY_KEYS)
    assert len(plan) == 2 * 2 * 2 * 5
    assert {s for *_, s in plan[:20]} == {0}, "every seed-0 run comes before any seed-1 run"
    assert [p for _, _, p, _ in plan[:5]] == list(POLICY_ORDER)


def tiny_loader(_name):
    return drifting_stream(n=1500)


def test_grid_writes_one_row_per_run_and_resumes_without_duplicates(tmp_path):
    cfg = RunConfig(**{**CFG.__dict__, "probe_rows": 200, "eval_block": 200})
    kwargs = dict(seeds=(0, 1), datasets=("tiny",), models=("gnb",), policies=POLICY_KEYS,
                  config=cfg, log=lambda _m: None, loader=tiny_loader)

    first = run_grid(tmp_path, **kwargs)
    second = run_grid(tmp_path, **kwargs)

    with (tmp_path / "grid.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert first == {"planned": 10, "ran": 10, "failed": 0, "skipped": 0}
    assert second == {"planned": 10, "ran": 0, "failed": 0, "skipped": 10}
    assert len(rows) == 10
    assert len({(r["policy_key"], r["seed"]) for r in rows}) == 10
    assert len(list((tmp_path / "events").glob("*.csv"))) == 10


def test_grid_isolates_a_failing_run_and_keeps_going(tmp_path):
    def flaky(X, y, model_name, policy_key, seed, config, dataset):
        if policy_key == "always_nudge":
            raise RuntimeError("boom")
        return run_stream(X, y, model_name, policy_key, seed, config, dataset)

    cfg = RunConfig(**{**CFG.__dict__, "probe_rows": 200, "eval_block": 200})
    result = run_grid(tmp_path, seeds=(0,), datasets=("tiny",), models=("gnb",), config=cfg,
                      log=lambda _m: None, run_fn=flaky, loader=tiny_loader)

    assert result == {"planned": 5, "ran": 4, "failed": 1, "skipped": 0}
    with (tmp_path / "grid_failures.csv").open(newline="", encoding="utf-8") as f:
        failures = list(csv.DictReader(f))
    assert len(failures) == 1 and "boom" in failures[0]["error"]


# --- placeholder logging --------------------------------------------------------------


def missing_class_stream():
    """3-class stream with stretches that lack classes, and an init slice missing one."""
    X, y = drifting_stream(n=3000, n_classes=3)
    y[:300] = np.where(y[:300] == 2, 0, y[:300])
    y[1200:1900] = 1
    return X, y, np.arange(3)


def n_missing(y_part, classes):
    return len(np.setdiff1d(classes, np.unique(y_part)))


@pytest.mark.parametrize("model_name", ["xgb", "rf", "sgd"])
@pytest.mark.parametrize("policy_key", ["always_rebuild", "always_nudge"])
def test_event_placeholder_counts_match_the_rows_each_operation_received(model_name, policy_key):
    X, y, classes = missing_class_stream()
    calls = []
    record, events = run_stream(X, y, model_name, policy_key, 0, CFG,
                                detector_factory=scripted({500, 950, 1100, 1300, 1500, 1750, 2200}),
                                policy_factory=spying(calls))

    for call, ev in zip(calls, events):
        if policy_key == "always_rebuild":
            expected = n_missing(call["y"], classes)               # rebuild fits the full window
        elif model_name in ("xgb", "rf"):
            k = int(len(call["y"]) * (1 - CFG.holdout_frac))
            expected = n_missing(call["y"][:k], classes)           # nudge fits the train part
        else:
            expected = 0                                           # partial_fit needs none
        assert ev["n_placeholder_rows"] == expected
        assert ev["placeholders_injected"] is (expected > 0)

    assert any(e["placeholders_injected"] for e in events) or model_name == "sgd" and policy_key == "always_nudge", (
        "precondition: this stream must actually trigger placeholders"
    )
    assert record["total_placeholder_rows"] == sum(e["n_placeholder_rows"] for e in events)
    assert record["n_adaptations_with_placeholders"] == sum(e["placeholders_injected"] for e in events)


def test_selector_rebuild_event_counts_placeholders_from_both_operations():
    X, y, classes = missing_class_stream()
    calls = []
    _, events = run_stream(X, y, "xgb", "mechanism_selector", 0, CFG,
                           detector_factory=scripted({950, 1100, 1300, 1500, 1750}),
                           policy_factory=spying(calls))

    checked = 0
    for call, ev in zip(calls, events):
        if ev["action"] != REBUILD or ev["degraded_to_rebuild"]:
            continue
        k = int(len(call["y"]) * (1 - CFG.holdout_frac))
        assert ev["n_placeholder_rows"] == n_missing(call["y"][:k], classes) + n_missing(call["y"], classes)
        checked += 1
    assert checked >= 1, "precondition: at least one nudge-then-rebuild on an affected window"


def test_initial_placeholder_rows_are_recorded():
    X, y, _ = missing_class_stream()                # class 2 absent from the init slice
    record, _ = run_stream(X, y, "xgb", "never_adapt", 0, CFG)
    assert record["initial_placeholder_rows"] == 1

    X2, y2 = drifting_stream(n=3000, n_classes=3)
    assert run_stream(X2, y2, "xgb", "never_adapt", 0, CFG)[0]["initial_placeholder_rows"] == 0


def test_nudges_since_rebuild_counts_consecutive_nudges_and_resets_on_rebuild():
    X, y = drifting_stream()
    _, events = run_stream(X, y, "gnb", "fixed_schedule", 0, CFG, detector_factory=scripted(range(0, 3000, 150)))

    depths = [e["nudges_since_rebuild"] for e in events]
    assert depths[:6] == [1, 2, 0, 1, 2, 0], "k=3: NUDGE, NUDGE, REBUILD repeating"


@pytest.mark.parametrize("policy_key", ["never_adapt", "always_nudge", "mechanism_selector"])
def test_every_event_logs_the_deficit_at_alarm_time(policy_key):
    X, y = drifting_stream()
    _, events = run_stream(X, y, "xgb", policy_key, 0, CFG, detector_factory=scripted(range(0, 3000, 150)))

    assert len(events) > 5
    for ev in events:
        assert ev["accuracy_deficit"] == pytest.approx(ev["reference_accuracy_used"] - ev["acc_before"]), (
            "positive deficit = model below its reference; negative = alarm fired on an improvement"
        )
        nudged = ev["acc_nudged"] is not None
        assert (ev["nudge_prediction_change"] is not None) == nudged
        if nudged:
            assert 0.0 <= ev["nudge_prediction_change"] <= 1.0


# --- alarm direction -----------------------------------------------------------------------


def replay_directions(X, y, model_name, seed, cfg):
    """Independent replay: NeverAdapt's error stream through a fresh ADWIN."""
    from river.drift import ADWIN

    from mechanisms import anchor_missing_classes

    n = len(X)
    n_init = int(n * cfg.init_frac)
    n_train = int(n_init * (1 - cfg.init_holdout_frac))
    X0, y0, kw = anchor_missing_classes(X[:n_train], y[:n_train], np.unique(y))
    model = make_spec(model_name, np.unique(y)).factory(seed).fit(X0, y0, **kw)
    errors = (model.predict(X[n_init:]) != y[n_init:]).astype(int)
    detector, out = ADWIN(delta=cfg.adwin_delta), {}
    for step, err in enumerate(errors):
        before = detector.estimation
        detector.update(int(err))
        if detector.drift_detected:
            out[n_init + step] = "error_down" if detector.estimation < before else (
                "error_up" if detector.estimation > before else "flat")
    return out


def test_logged_alarm_direction_matches_an_independent_detector_replay():
    X, y = drifting_stream(n=4000, n_segments=8)
    _, events = run_stream(X, y, "gnb", "never_adapt", 0, CFG)
    replay = replay_directions(X, y, "gnb", 0, CFG)

    assert len(events) >= 3, "precondition: real ADWIN alarms"
    for ev in events:
        assert ev["alarm_direction"] == replay[ev["alarm_row"]]
        assert (ev["alarm_error_after"] < ev["alarm_error_before"]) == (ev["alarm_direction"] == "error_down")


@pytest.mark.parametrize("policy_key", ["always_rebuild", "fixed_schedule", "mechanism_selector"])
def test_cost_on_falling_error_alarms_is_the_sum_over_those_events(policy_key):
    X, y = drifting_stream(n=4000, n_segments=8)
    record, events = run_stream(X, y, "xgb", policy_key, 0, CFG)

    down = [e for e in events if e["alarm_direction"] == "error_down"]
    assert record["n_adaptations_on_falling_error"] == len(down)
    assert record["work_units_on_falling_error"] == sum(e["work_units"] for e in down)
    assert all(e["alarm_direction"] in ("error_up", "error_down", "flat") for e in events)


def test_detectors_without_an_estimate_log_no_direction():
    X, y = drifting_stream()
    record, events = run_stream(X, y, "gnb", "always_rebuild", 0, CFG, detector_factory=scripted({200, 900}))
    assert events and all(e["alarm_direction"] is None for e in events)
    assert record["n_adaptations_on_falling_error"] == 0
