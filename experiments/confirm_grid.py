"""Confirm-before-acting grid on synthetic streams (docs/confirm/PREREGISTRATION.md).

6 streams x 4 models x 5 seeds x {always_rebuild, mechanism_selector} x W in {0, 200, 500},
plus never_adapt once per (stream, model, seed): 840 runs. Every run executes fresh;
the frozen Phase 6 grid is neither read nor written.

Outputs, under results/confirm/:
  runs.csv      one row per run: costs, accuracy, post-change accuracy, alarm counts
  episodes.csv  one row per alarm episode: adapted or dismissed, true or false detection
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import synthetic  # noqa: E402
from runner import RunConfig, run_stream  # noqa: E402

MODELS = ("xgb", "rf", "sgd", "gnb")
SEEDS = range(5)
CONFIRM = (0, 200, 500)
ADAPTIVE = ("always_rebuild", "mechanism_selector")
TRUE_WINDOW = 1000      # an episode starting within this many rows after a change point is a true detection
POST_WINDOW = 2000      # accuracy over this many rows after each change point
BASE = RunConfig(track_energy=False, track_inference=True)

RUN_FIELDS = [
    "stream", "model", "policy_key", "confirm_rows", "seed",
    "n_alarms", "n_episodes", "n_adaptations", "n_alarms_dismissed", "n_skip", "n_nudge", "n_rebuild",
    "n_degraded", "total_work_units", "total_inference_ops", "initial_inference_units",
    "final_inference_units", "mean_prequential_accuracy", "post_change_accuracy",
    "n_true_episodes", "n_false_episodes", "n_true_dismissed", "n_false_dismissed", "run_wall_clock_s",
]
EPISODE_FIELDS = ["stream", "model", "policy_key", "confirm_rows", "seed", "alarm_row",
                  "outcome", "action", "truth", "alarm_direction", "work_units"]


def truth(alarm_row: int, change_points) -> str:
    return "true" if any(0 <= alarm_row - c < TRUE_WINDOW for c in change_points) else "false"


def plan():
    jobs = []
    for stream in synthetic.STREAMS:
        for seed in SEEDS:
            for model in MODELS:
                jobs.append((stream, model, "never_adapt", 0, seed))
                for policy in ADAPTIVE:
                    for w in CONFIRM:
                        jobs.append((stream, model, policy, w, seed))
    return jobs


_cache: dict = {}


def run_one(job, delay_only=False):
    stream, model, policy, w, seed = job
    if (stream, seed) not in _cache:
        _cache.clear()
        _cache[(stream, seed)] = synthetic.make_stream(stream, seed)
    X, y, cps = _cache[(stream, seed)]
    record, events = run_stream(X, y, model, policy, seed, replace(BASE, confirm_rows=w, confirm_dismiss=not delay_only), stream)

    n_init = len(X) - record["n_stream_rows"]
    err = record["prequential_errors"]
    post = [err[max(0, c - n_init):c - n_init + POST_WINDOW] for c in cps]
    post_acc = float(1 - np.concatenate(post).mean()) if post else None

    episodes = []
    for e in events:
        episodes.append({"alarm_row": e["alarm_row"], "outcome": "adapted", "action": e["action"],
                         "alarm_direction": e["alarm_direction"], "work_units": e["work_units"]})
    for d in record["dismissed_alarms"]:
        episodes.append({"alarm_row": d["alarm_row"], "outcome": "dismissed", "action": "",
                         "alarm_direction": d["alarm_direction"], "work_units": 0})
    for ep in episodes:
        ep.update(stream=stream, model=model, policy_key=policy, confirm_rows=w, seed=seed,
                  truth=truth(ep["alarm_row"], cps))

    count = lambda t, o=None: sum(1 for ep in episodes if ep["truth"] == t and (o is None or ep["outcome"] == o))  # noqa: E731
    row = {k: record.get(k) for k in RUN_FIELDS if k in record}
    row.update(stream=stream, policy_key=policy, confirm_rows=w, seed=seed,
               n_episodes=len(episodes), post_change_accuracy=post_acc,
               n_true_episodes=count("true"), n_false_episodes=count("false"),
               n_true_dismissed=count("true", "dismissed"), n_false_dismissed=count("false", "dismissed"))
    return row, episodes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/confirm")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N jobs (smoke test)")
    ap.add_argument("--delay-only", action="store_true",
                    help="post-hoc control: W in {200, 500} with no dismissal, adaptive policies only")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    jobs = plan()[: args.limit]
    if args.delay_only:
        jobs = [j for j in jobs if j[3] > 0]
    started = time.perf_counter()
    rows, eps = [], []
    # chunked by (stream, seed) so each worker reuses its generated stream
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_one, j, args.delay_only): j for j in jobs}
        for i, f in enumerate(as_completed(futures), 1):
            row, episodes = f.result()
            rows.append(row)
            eps.extend(episodes)
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {time.perf_counter() - started:.0f}s", flush=True)

    key = lambda r: (r["stream"], r["model"], r["policy_key"], r["confirm_rows"], r["seed"])  # noqa: E731
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
