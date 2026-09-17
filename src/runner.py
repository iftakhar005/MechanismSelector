"""Experiment runner: one policy, one model, one stream, prequentially.

## The loop

1. Train the initial model on the first `init_frac` of the stream, holding out
   the most recent `init_holdout_frac` of that slice; score the model on the
   held-out part to get the first `reference_accuracy`.
2. Walk every remaining row in order. For each row: predict, record whether the
   prediction was right (prequential evaluation), add the row to a rolling
   buffer of the most recent `buffer_size` rows, and feed the error to ADWIN.
3. When ADWIN raises an alarm, call `policy.adapt()` on the buffer, adopt the
   returned model, and clear the buffer.
4. After the last row, measure the final model's footprint and return totals.

## Guards and rules added beyond the spec

**Minimum buffer.** An alarm that fires while the buffer holds fewer than
`min_buffer` rows is *deferred*, not dropped: the policy is called as soon as
the buffer reaches `min_buffer`. Adapting on a handful of rows would leave the
80% training part empty or single-class. Several alarms during one deferral are
coalesced into one adaptation. The default of 100 sits well below the selector's
own small-window threshold (~250 rows), so the selector's degrade-to-rebuild
behaviour is still exercised rather than pre-empted. Every deferral is counted.

**Reference accuracy, measured causally.** After each adaptation, the returned
model's predictions on the next `n_ref` rows are scored as those rows arrive,
and the result becomes `reference_accuracy` for the next alarm. The rows are
not consumed -- they are ordinary stream rows, predicted and evaluated as usual.
This is the rule in `selector.py`'s docstring, evaluated as the rows arrive
rather than by looking ahead, so no decision ever uses a label the stream has
not produced yet. If the next alarm is served before `n_ref` rows have arrived,
the previous reference is carried forward and the event is flagged.

**Scorer.** Binary streams use accuracy; multi-class streams use balanced
accuracy, for both the selector's floor and the reference. Plain and balanced
prequential accuracy are both recorded for every run.

## Chunked prediction is exact, not approximate

Predicting one row at a time is 430x-1,300x slower than predicting a batch.
Between adaptations the model does not change, so predicting a chunk ahead gives
exactly the predictions a row-at-a-time loop would. After an adaptation, the
remaining predictions in the chunk are discarded and re-made with the new model.
A test runs the same stream with chunk size 1 and chunk size 2,000 and requires
identical records and events.

## Costs

`total_work_units` covers adaptations only. Initial training is identical for
every policy in a cell, so it is reported separately as `initial_work_units`
rather than added to every total. `total_wall_clock_s` is adaptation training
time; `run_wall_clock_s` is the whole run. Energy covers the whole run and is
secondary. Model size and per-row inference cost are recorded for the initial
and final model (see `footprint.py`), and size after every adaptation in the
event log.

## ADWIN is two-sided

ADWIN flags a significant change in error rate in either direction, so a clear
improvement after an adaptation can itself raise an alarm. The spec feeds ADWIN
the raw error stream and does not reset it; this runner does the same, for every
policy alike. Each adaptation records the direction of the alarm that triggered
it (`alarm_direction`), read from the detector's own error estimate, and each run
records how much adaptation cost was spent on alarms fired by a *falling* error
rate.
"""

from __future__ import annotations

import csv
import json
import os
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import datasets as ds
from accounting import measure
from footprint import measure_footprint, model_size
from mechanisms import GRID_MODELS, anchor_missing_classes, make_spec
from policies import POLICY_KEYS, make_policy
from selector import NUDGE, REBUILD, SKIP

#: Grid order within one (seed, dataset, model) cell. The comparisons that
#: matter most finish first, so a partial grid is already informative.
POLICY_ORDER = (
    "always_rebuild",
    "mechanism_selector",
    "fixed_schedule",
    "always_nudge",
    "never_adapt",
)

RECORD_FIELDS = [
    "dataset", "model", "policy", "policy_key", "seed",
    "n_stream_rows", "n_init_train_rows", "scorer",
    "n_alarms", "n_adaptations", "n_skip", "n_nudge", "n_rebuild", "n_degraded",
    "n_alarms_deferred", "n_alarms_coalesced", "n_alarms_unserved", "n_reference_carried",
    "n_nudge_attempts", "n_nudge_successes",
    "n_adaptations_with_placeholders", "total_placeholder_rows", "initial_placeholder_rows",
    "n_adaptations_on_falling_error", "work_units_on_falling_error",
    "total_work_units", "total_estimators_fitted", "total_rows_processed", "total_passes",
    "total_wall_clock_s", "total_energy_kwh", "initial_work_units",
    "final_prequential_accuracy", "mean_prequential_accuracy",
    "final_prequential_balanced_accuracy", "mean_prequential_balanced_accuracy",
    "model_size_unit", "initial_model_size", "final_model_size",
    "initial_n_nodes", "final_n_nodes",
    "inference_unit", "initial_inference_units", "final_inference_units", "inference_growth",
    "initial_predict_us_per_row", "final_predict_us_per_row",
    "run_wall_clock_s", "config_json",
]

EVENT_FIELDS = [
    "dataset", "model", "policy_key", "seed",
    "event_index", "alarm_row", "adapt_row", "deferred_rows", "buffer_rows",
    "action", "degraded_to_rebuild",
    "reference_accuracy_used", "reference_carried", "floor",
    "acc_before", "acc_nudged", "acc_after",
    "work_units", "n_passes", "n_estimators_fitted", "rows_processed",
    "cumulative_work_units", "model_size_after", "model_size_unit",
    "placeholders_injected", "n_placeholder_rows", "nudges_since_rebuild",
    "accuracy_deficit", "nudge_prediction_change",
    "alarm_error_before", "alarm_error_after", "alarm_direction",
]


@dataclass(frozen=True)
class RunConfig:
    init_frac: float = 0.10
    init_holdout_frac: float = 0.20
    buffer_size: int = 1000
    min_buffer: int = 100
    n_ref: int = 200
    holdout_frac: float = 0.20
    floor_drop: float = 0.02
    k: int = 3
    eval_block: int = 1000
    probe_rows: int = 1000
    chunk_size: int = 2000
    adwin_delta: float = 0.002
    track_energy: bool = True
    energy_country_iso: str = "USA"  # affects only the CO2 figure, which is not recorded


def alarm_direction(before: float | None, after: float | None) -> str | None:
    """Which way the error rate moved at an alarm, read from the detector itself.

    ADWIN drops the older part of its window when it alarms. Its error estimate
    just before the update covers the whole window; just after, only the
    retained recent part. A lower estimate afterwards means the change it
    detected was a *fall* in error -- the model got better, or the stream got
    easier. None when the detector exposes no estimate (e.g. a scripted test
    detector). Reading the estimate never changes the detector.
    """
    if before is None or after is None:
        return None
    if after < before:
        return "error_down"
    if after > before:
        return "error_up"
    return "flat"


def scorer_for(classes: np.ndarray) -> tuple[Callable, str]:
    if len(classes) == 2:
        return accuracy_score, "accuracy"
    return balanced_accuracy_score, "balanced_accuracy"


def default_detector(config: RunConfig):
    from river.drift import ADWIN

    return ADWIN(delta=config.adwin_delta)


def run_stream(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    policy_key: str,
    seed: int,
    config: RunConfig = RunConfig(),
    dataset: str = "",
    *,
    detector_factory: Callable[[RunConfig], Any] = default_detector,
    policy_factory: Callable[..., Any] = make_policy,
) -> tuple[dict, list[dict]]:
    """Run one policy over one stream. Returns ``(record, events)``."""
    run_start = time.perf_counter()
    tracker = _start_energy(config)
    try:
        record, events = _run(
            X, y, model_name, policy_key, seed, config, dataset,
            detector_factory, policy_factory,
        )
    finally:
        energy = _stop_energy(tracker)
    record["total_energy_kwh"] = energy
    record["run_wall_clock_s"] = time.perf_counter() - run_start
    return record, events


def run_experiment(
    dataset: str,
    model_name: str,
    policy: str,
    seed: int,
    init_frac: float = 0.10,
    buffer_size: int = 1000,
    config: RunConfig | None = None,
) -> dict:
    """The spec's interface: load a named stream and run one policy over it."""
    base = config or RunConfig()
    cfg = RunConfig(**{**asdict(base), "init_frac": init_frac, "buffer_size": buffer_size})
    X, y = ds.load_stream(dataset)
    record, _ = run_stream(X, y, model_name, policy, seed, cfg, dataset)
    return record


def _run(X, y, model_name, policy_key, seed, config, dataset, detector_factory, policy_factory):
    warnings.filterwarnings("ignore")
    n = len(X)
    classes = np.unique(y)
    spec = make_spec(model_name, classes)
    scorer, scorer_name = scorer_for(classes)

    n_init = int(n * config.init_frac)
    n_init_train = int(n_init * (1 - config.init_holdout_frac))
    if n_init_train < 1 or n_init - n_init_train < 1 or n_init >= n:
        raise ValueError(f"stream of {n} rows too short for init_frac={config.init_frac}")

    model = spec.factory(seed)
    X0, y0, kw = anchor_missing_classes(X[:n_init_train], y[:n_init_train], classes)
    model, init_cost = measure(lambda: model.fit(X0, y0, **kw), n_rows=n_init_train)
    initial_placeholder_rows = len(y0) - n_init_train
    reference_accuracy = float(scorer(y[n_init_train:n_init], model.predict(X[n_init_train:n_init])))

    probe = X[max(0, n - config.probe_rows):]
    initial_fp = measure_footprint(model, probe)

    policy = policy_factory(
        policy_key, spec, seed=seed, holdout_frac=config.holdout_frac,
        floor_drop=config.floor_drop, k=config.k, scorer=scorer,
    )
    detector = detector_factory(config)

    preds = np.empty(n - n_init, dtype=y.dtype)
    events: list[dict] = []
    counts = {SKIP: 0, NUDGE: 0, REBUILD: 0}
    n_alarms = n_degraded = n_deferred = n_coalesced = n_carried = 0
    n_nudge_attempts = 0
    n_on_falling = work_on_falling = 0
    alarm_before = alarm_after = None  # detector error estimate around the alarm that opened the episode
    n_with_placeholders = total_placeholders = 0
    nudges_since_rebuild = 0  # consecutive-nudge depth; the initial model counts as a rebuild
    totals = dict(work_units=0, estimators=0, rows=0, passes=0, wall=0.0)

    buf_start = n_init
    pending_since: int | None = None
    deferral_counted = False
    ref_start: int | None = None      # first row of the reference slice in progress
    ref_ready = True                   # the initial reference comes from the init holdout

    t = n_init
    while t < n:
        chunk_end = min(t + config.chunk_size, n)
        chunk_preds = model.predict(X[t:chunk_end])
        adapted_at = None

        for offset, row in enumerate(range(t, chunk_end)):
            pred = chunk_preds[offset]
            preds[row - n_init] = pred

            if not ref_ready and row - ref_start + 1 == config.n_ref:
                reference_accuracy = float(scorer(y[ref_start:row + 1], preds[ref_start - n_init:row + 1 - n_init]))
                ref_ready = True

            estimate_before = getattr(detector, "estimation", None)
            detector.update(int(pred != y[row]))
            if detector.drift_detected:
                n_alarms += 1
                if pending_since is None:
                    pending_since = row
                    deferral_counted = False
                    alarm_before, alarm_after = estimate_before, getattr(detector, "estimation", None)
                else:
                    n_coalesced += 1

            if pending_since is None:
                continue

            window_start = max(buf_start, row + 1 - config.buffer_size)
            buffer_rows = row + 1 - window_start
            if buffer_rows < config.min_buffer:
                if not deferral_counted:
                    n_deferred += 1
                    deferral_counted = True
                continue

            carried = not ref_ready
            n_carried += int(carried)
            result = policy.adapt(model, X[window_start:row + 1], y[window_start:row + 1], reference_accuracy)
            model = result.model

            cost = result.cost
            counts[result.action] += 1
            n_degraded += int(result.degraded_to_rebuild)
            n_nudge_attempts += int(result.acc_nudged is not None)
            n_with_placeholders += int(cost.n_placeholder_rows > 0)
            total_placeholders += cost.n_placeholder_rows
            if result.action == REBUILD:
                nudges_since_rebuild = 0
            elif result.action == NUDGE:
                nudges_since_rebuild += 1
            totals["work_units"] += cost.work_units
            direction = alarm_direction(alarm_before, alarm_after)
            if direction == "error_down":
                n_on_falling += 1
                work_on_falling += cost.work_units
            totals["estimators"] += cost.n_estimators_fitted
            totals["rows"] += cost.n_rows_processed
            totals["passes"] += cost.n_passes
            totals["wall"] += cost.wall_clock_s

            size_after, size_unit = model_size(model)
            events.append({
                "dataset": dataset, "model": model_name, "policy_key": policy_key, "seed": seed,
                "event_index": len(events), "alarm_row": pending_since, "adapt_row": row,
                "deferred_rows": row - pending_since, "buffer_rows": buffer_rows,
                "action": result.action, "degraded_to_rebuild": result.degraded_to_rebuild,
                "reference_accuracy_used": reference_accuracy, "reference_carried": carried,
                "floor": result.floor, "acc_before": result.acc_before,
                "acc_nudged": result.acc_nudged, "acc_after": result.acc_after,
                "work_units": cost.work_units, "n_passes": cost.n_passes,
                "n_estimators_fitted": cost.n_estimators_fitted, "rows_processed": cost.n_rows_processed,
                "cumulative_work_units": totals["work_units"],
                "model_size_after": size_after, "model_size_unit": size_unit,
                "placeholders_injected": cost.n_placeholder_rows > 0,
                "n_placeholder_rows": cost.n_placeholder_rows,
                "nudges_since_rebuild": nudges_since_rebuild,
                "accuracy_deficit": reference_accuracy - result.acc_before,
                "nudge_prediction_change": result.prediction_change,
                "alarm_error_before": alarm_before, "alarm_error_after": alarm_after,
                "alarm_direction": direction,
            })

            buf_start = row + 1
            pending_since = None
            ref_start, ref_ready = row + 1, False
            adapted_at = row
            break

        t = adapted_at + 1 if adapted_at is not None else chunk_end

    final_fp = measure_footprint(model, probe)
    y_stream = y[n_init:]
    block = slice(max(0, len(y_stream) - config.eval_block), len(y_stream))
    is_selector = policy_key == "mechanism_selector"

    record = {
        "dataset": dataset, "model": model_name, "policy": policy.name, "policy_key": policy_key,
        "seed": seed, "n_stream_rows": n - n_init, "n_init_train_rows": n_init_train,
        "scorer": scorer_name,
        "n_alarms": n_alarms, "n_adaptations": len(events),
        "n_skip": counts[SKIP], "n_nudge": counts[NUDGE], "n_rebuild": counts[REBUILD],
        "n_degraded": n_degraded, "n_alarms_deferred": n_deferred,
        "n_alarms_coalesced": n_coalesced, "n_alarms_unserved": int(pending_since is not None),
        "n_reference_carried": n_carried,
        "n_nudge_attempts": n_nudge_attempts,
        "n_nudge_successes": counts[NUDGE] if is_selector else None,
        "n_adaptations_with_placeholders": n_with_placeholders,
        "n_adaptations_on_falling_error": n_on_falling,
        "work_units_on_falling_error": work_on_falling,
        "total_placeholder_rows": total_placeholders,
        "initial_placeholder_rows": initial_placeholder_rows,
        "total_work_units": totals["work_units"], "total_estimators_fitted": totals["estimators"],
        "total_rows_processed": totals["rows"], "total_passes": totals["passes"],
        "total_wall_clock_s": totals["wall"], "total_energy_kwh": None,
        "initial_work_units": init_cost.work_units,
        "final_prequential_accuracy": float(accuracy_score(y_stream[block], preds[block])),
        "mean_prequential_accuracy": float(accuracy_score(y_stream, preds)),
        "final_prequential_balanced_accuracy": float(balanced_accuracy_score(y_stream[block], preds[block])),
        "mean_prequential_balanced_accuracy": float(balanced_accuracy_score(y_stream, preds)),
        "model_size_unit": initial_fp.size_unit,
        "initial_model_size": initial_fp.size, "final_model_size": final_fp.size,
        "initial_n_nodes": initial_fp.n_nodes, "final_n_nodes": final_fp.n_nodes,
        "inference_unit": initial_fp.inference_unit,
        "initial_inference_units": initial_fp.inference_units,
        "final_inference_units": final_fp.inference_units,
        "inference_growth": final_fp.inference_units / initial_fp.inference_units,
        "initial_predict_us_per_row": initial_fp.predict_us_per_row,
        "final_predict_us_per_row": final_fp.predict_us_per_row,
        "run_wall_clock_s": None,
        "config_json": json.dumps(asdict(config), sort_keys=True),
    }
    return record, events


# --- energy (secondary, never decisive) ----------------------------------------


def _start_energy(config: RunConfig):
    if not config.track_energy:
        return None
    try:
        import logging

        from codecarbon import OfflineEmissionsTracker

        logging.getLogger("codecarbon").setLevel(logging.ERROR)
        tracker = OfflineEmissionsTracker(
            country_iso_code=config.energy_country_iso, save_to_file=False,
            log_level="error", tracking_mode="process",
        )
        tracker.start()
        return tracker
    except Exception:  # noqa: BLE001 - energy is optional; its absence is recorded as None
        return None


def _stop_energy(tracker) -> float | None:
    if tracker is None:
        return None
    try:
        tracker.stop()
        return float(tracker.final_emissions_data.energy_consumed)
    except Exception:  # noqa: BLE001
        return None


# --- grid ---------------------------------------------------------------------


def run_key(record: dict) -> tuple:
    return (record["dataset"], record["model"], record["policy_key"], int(record["seed"]))


def completed_keys(summary_path: Path) -> set[tuple]:
    if not summary_path.exists():
        return set()
    with summary_path.open(newline="", encoding="utf-8") as f:
        return {run_key(row) for row in csv.DictReader(f)}


def _append_row(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
        f.flush()
        os.fsync(f.fileno())


def _write_events(events_dir: Path, key: tuple, events: list[dict]) -> None:
    events_dir.mkdir(parents=True, exist_ok=True)
    final = events_dir / ("__".join(str(p) for p in key) + ".csv")
    tmp = final.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for ev in events:
            writer.writerow({k: ("" if ev.get(k) is None else ev.get(k)) for k in EVENT_FIELDS})
    os.replace(tmp, final)


def grid_plan(seeds, datasets, models, policies) -> list[tuple]:
    """Seed outermost: after seed 0, every (dataset, model) cell has one full
    replicate of all five policies, so the early picture covers the whole grid."""
    order = [p for p in POLICY_ORDER if p in policies]
    return [(d, m, p, s) for s in seeds for d in datasets for m in models for p in order]


def run_grid(
    out_dir: Path,
    seeds=range(5),
    datasets=ds.STREAMS,
    models=GRID_MODELS,
    policies=POLICY_KEYS,
    config: RunConfig = RunConfig(),
    log: Callable[[str], None] = print,
    run_fn: Callable[..., tuple[dict, list[dict]]] = run_stream,
    loader: Callable[[str], tuple[np.ndarray, np.ndarray]] = ds.load_stream,
) -> dict:
    """Run every (dataset, model, policy, seed) not already in the summary CSV.

    Each row is written and fsynced the moment its run finishes, so a crash or
    power cut loses at most the run in progress; rerunning resumes. A run that
    raises is logged to the failures file and the grid moves on.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, failures, events_dir = out_dir / "grid.csv", out_dir / "grid_failures.csv", out_dir / "events"

    plan = grid_plan(seeds, datasets, models, policies)
    done = completed_keys(summary)
    todo = [k for k in plan if k not in done]
    log(f"grid: {len(plan)} runs planned, {len(plan) - len(todo)} already complete, {len(todo)} to run")

    cache: dict[str, tuple] = {}
    n_ok = n_failed = 0
    for i, (dataset, model_name, policy_key, seed) in enumerate(todo, 1):
        if dataset not in cache:
            cache.clear()
            cache[dataset] = loader(dataset)
        X, y = cache[dataset]
        key = (dataset, model_name, policy_key, seed)
        started = time.perf_counter()
        try:
            record, events = run_fn(X, y, model_name, policy_key, seed, config, dataset)
            _write_events(events_dir, key, events)
            _append_row(summary, RECORD_FIELDS, record)
            n_ok += 1
            log(f"[{i}/{len(todo)}] {dataset} {model_name} {policy_key} seed={seed}: "
                f"{record['n_adaptations']} adaptations, {record['total_work_units']:,} WU, "
                f"acc {record['mean_prequential_accuracy']:.4f} ({time.perf_counter() - started:.0f}s)")
        except Exception as exc:  # noqa: BLE001 - isolate failures so the grid keeps going
            n_failed += 1
            _append_row(failures, ["dataset", "model", "policy_key", "seed", "error", "traceback"], {
                "dataset": dataset, "model": model_name, "policy_key": policy_key, "seed": seed,
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            })
            log(f"[{i}/{len(todo)}] FAILED {key}: {type(exc).__name__}: {exc}")

    return {"planned": len(plan), "ran": n_ok, "failed": n_failed, "skipped": len(plan) - len(todo)}
