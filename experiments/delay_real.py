"""Delay-before-adapting on the real streams (docs/confirm/PREREGISTRATION_REAL.md).

The interventional test of the stale-window finding: acting at alarm time trains the replacement
model on a window that is mostly pre-change data. Waiting W rows after the alarm, then always
adapting, moves the window past the change. Nothing is dismissed.

Grid: 5 real streams x {rf, sgd: seeds 0-4; xgb, gnb: seed 0, being deterministic} x
{always_rebuild, mechanism_selector} x W in {0, 200, 500}, plus never_adapt once: 420 runs. ADWIN, the study's RunConfig defaults.

Needs the real data (elec2, the three INSECTS variants, covtype). They download through river
and scikit-learn on first use; run it where those hosts are reachable:

    python experiments/delay_real.py                 # full grid, 4 workers
    python experiments/delay_real.py --workers 8
    python experiments/delay_real.py --substitute    # dry run on synthetic stand-ins, no download

Outputs, under results/delay_real/ (or results/delay_real_substitute/ for a dry run):
  runs.csv       one row per run
  episodes.csv   one row per adaptation (insects_abrupt rows carry change-point fields)
  versions.json  the package versions the run used
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
from runner import RunConfig, run_stream  # noqa: E402

MODELS = ("xgb", "rf", "sgd", "gnb")
SEEDS = range(5)
DELAYS = (0, 200, 500)
ADAPTIVE = ("always_rebuild", "mechanism_selector")
DETERMINISTIC = ("xgb", "gnb")   # as in Phase 9.5: one seed
BASE = RunConfig(track_energy=False, track_inference=True)
POST_WINDOW = 2000
TRUE_WINDOW = 1000
CHANGE_POINTS = {
    "insects_abrupt": tuple(json.loads((ROOT / "results/analysis/phase9_change_points.json").read_text())
                            ["ground_truth"]["change_points"]),
}

RUN_FIELDS = [
    "stream", "model", "policy_key", "delay_rows", "seed", "n_stream_rows", "scorer",
    "n_alarms", "n_adaptations", "n_skip", "n_nudge", "n_rebuild", "n_degraded",
    "total_work_units", "total_inference_ops", "initial_inference_units", "final_inference_units",
    "mean_prequential_accuracy", "mean_prequential_balanced_accuracy", "post_change_accuracy",
    "run_wall_clock_s",
]
EPISODE_FIELDS = ["stream", "model", "policy_key", "delay_rows", "seed", "alarm_row", "adapt_row",
                  "buffer_rows", "action", "alarm_direction", "work_units",
                  "change_point", "detect_lag", "true_detection", "post_change_share"]


def load(stream: str, substitute: bool):
    if not substitute:
        return ds.load_stream(stream)
    import synthetic   # dry run only: a stand-in of the same shape class, never reported
    X, y, _ = synthetic.make_stream("agrawal_abrupt_sticky" if "insects" in stream else "sea_abrupt_iid",
                                    seed=0, n_rows=8000)
    return X, y


def plan():
    jobs = []
    for stream in ds.STREAMS:
        for seed in SEEDS:
            for model in MODELS:
                if model in DETERMINISTIC and seed != 0:
                    continue   # identical across seeds: extra copies would be pseudo-replication
                jobs.append((stream, model, "never_adapt", 0, seed))
                for policy in ADAPTIVE:
                    for w in DELAYS:
                        jobs.append((stream, model, policy, w, seed))
    return jobs


_cache: dict = {}


def run_one(job, substitute: bool):
    stream, model, policy, w, seed = job
    if stream not in _cache:
        _cache.clear()
        _cache[stream] = load(stream, substitute)
    X, y = _cache[stream]
    cfg = replace(BASE, confirm_rows=w, confirm_dismiss=False)
    record, events = run_stream(X, y, model, policy, seed, cfg, stream)

    cps = np.asarray(CHANGE_POINTS.get(stream, ()) if not substitute else ())
    n_init = len(X) - record["n_stream_rows"]
    err = record["prequential_errors"]
    post_acc = (float(1 - np.concatenate([err[max(0, c - n_init):c - n_init + POST_WINDOW] for c in cps]).mean())
                if len(cps) else None)

    episodes = []
    for e in events:
        ep = {"stream": stream, "model": model, "policy_key": policy, "delay_rows": w, "seed": seed,
              "alarm_row": e["alarm_row"], "adapt_row": e["adapt_row"], "buffer_rows": e["buffer_rows"],
              "action": e["action"], "alarm_direction": e["alarm_direction"], "work_units": e["work_units"]}
        if len(cps):
            past = cps[cps <= e["alarm_row"]]
            if len(past):
                c = int(past.max())
                ep.update(change_point=c, detect_lag=e["alarm_row"] - c,
                          true_detection=e["alarm_row"] - c < TRUE_WINDOW,
                          post_change_share=min(max(e["adapt_row"] + 1 - c, 0), e["buffer_rows"]) / e["buffer_rows"])
        episodes.append(ep)

    row = {k: record.get(k) for k in RUN_FIELDS if k in record}
    row.update(stream=stream, policy_key=policy, delay_rows=w, seed=seed, post_change_accuracy=post_acc)
    return row, episodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--substitute", action="store_true", help="dry run on synthetic stand-ins (no download)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = ROOT / (args.out or ("results/delay_real_substitute" if args.substitute else "results/delay_real"))
    out.mkdir(parents=True, exist_ok=True)
    import river, sklearn, xgboost  # noqa: E401
    (out / "versions.json").write_text(json.dumps({
        "python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "river": river.__version__, "substitute": args.substitute,
    }, indent=2), encoding="utf-8")

    if not args.substitute:   # fail fast, before any compute, if a stream cannot be loaded
        for s in ds.STREAMS:
            ds.load_stream(s)

    jobs = plan()[: args.limit]
    started = time.perf_counter()
    rows, eps = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_one, j, args.substitute) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            row, episodes = f.result()
            rows.append(row)
            eps.extend(episodes)
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {time.perf_counter() - started:.0f}s", flush=True)

    key = lambda r: (r["stream"], r["model"], r["policy_key"], r["delay_rows"], r["seed"])  # noqa: E731
    rows.sort(key=key)
    eps.sort(key=lambda e: (*key(e), e["alarm_row"]))
    for path, fields, data in ((out / "runs.csv", RUN_FIELDS, rows), (out / "episodes.csv", EPISODE_FIELDS, eps)):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in data:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print(f"wrote {len(rows)} runs, {len(eps)} episodes to {out} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
