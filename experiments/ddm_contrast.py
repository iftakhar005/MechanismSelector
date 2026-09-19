"""Phase 7b: DDM (one-sided) as a contrast to ADWIN (two-sided).

    python experiments/ddm_contrast.py

Reads the ADWIN grid (`results/grid`, `results/grid_direction`) and the DDM grid
(`results/grid_ddm`), neither of which it modifies, and writes
`results/analysis/ddm_contrast.json`.

## Why direction is measured with a window rule

ADWIN's direction was read from its own error estimate before and after it
drops the old part of its window. DDM has no such estimate: it fires when its
running error rate (since its last reset) exceeds its running minimum by three
standard deviations, so by its *own* statistic every DDM alarm is a rise, by
construction. Scoring DDM that way would make "DDM shows no improvement alarms"
true by definition.

Direction is therefore measured with one rule for both detectors: the
prequential error rate over the W rows ending at the alarm against the W rows
before those. W = 200 is primary (the project's reference length); 100 and 500
are sensitivity checks. All three were fixed before any DDM run. For NeverAdapt
the error stream is identical under both detectors -- the model never changes --
so that comparison is exact. How well the window rule agrees with ADWIN's own
measure is reported, so the two sets of ADWIN numbers can be related.

## What is reported

1. Alarm counts per cell, both detectors, NeverAdapt replay (verified against
   both grids' alarm counts).
2. Direction split under both detectors, by the window rule.
3. Adaptation cost attributable to falling-error alarms, per adaptive policy.
   DDM uses the window rule; the ADWIN follow-up runs predate the window fields
   and use ADWIN's own measure -- the agreement in (2) bounds that difference.
4. Rows between an adaptation and the next alarm, split by direction, uncapped.
5. Whether the policy comparison changes under DDM.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from river.drift import ADWIN
from river.drift.binary import DDM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analysis as an  # noqa: E402
import datasets as ds  # noqa: E402
from mechanisms import GRID_MODELS, anchor_missing_classes, make_spec  # noqa: E402
from runner import DIRECTION_WINDOWS, RunConfig, window_error_rates  # noqa: E402

warnings.filterwarnings("ignore")
ADWIN_GRID = ROOT / "results" / "grid"
ADWIN_DIRECTION = ROOT / "results" / "grid_direction"
DDM_GRID = ROOT / "results" / "grid_ddm"
OUT = ROOT / "results" / "analysis"
ADAPTIVE = ("always_rebuild", "fixed_schedule", "mechanism_selector")
PRIMARY_W = 200


def direction_from_windows(recent, prior):
    if recent is None or prior is None:
        return None
    return "error_down" if recent < prior else "error_up" if recent > prior else "flat"


def never_adapt_errors(X, y, model_name, seed, config):
    n = len(X)
    n_init = int(n * config.init_frac)
    n_train = int(n_init * (1 - config.init_holdout_frac))
    X0, y0, kw = anchor_missing_classes(X[:n_train], y[:n_train], np.unique(y))
    model = make_spec(model_name, np.unique(y)).factory(seed).fit(X0, y0, **kw)
    return (model.predict(X[n_init:]) != y[n_init:]).astype(np.int8)


def replay(errors, detector):
    alarms = []
    for step, err in enumerate(errors):
        before = getattr(detector, "estimation", None)
        detector.update(int(err))
        if detector.drift_detected:
            after = getattr(detector, "estimation", None)
            own = None if before is None else ("error_down" if after < before else "error_up" if after > before else "flat")
            windows = {w: direction_from_windows(*window_error_rates(errors, step, w)) for w in DIRECTION_WINDOWS}
            alarms.append({"step": step, "own": own, **{f"w{w}": d for w, d in windows.items()}})
    return alarms


def share(alarms, key, value="error_down"):
    known = [a[key] for a in alarms if a[key] in ("error_up", "error_down", "flat")]
    return (sum(v == value for v in known) / len(known)) if known else None, len(known)


def section_replay(adwin_rows, ddm_rows):
    config = RunConfig()
    adwin_counts = {(r["dataset"], r["model"], int(r["seed"])): int(r["n_alarms"]) for r in adwin_rows if r["policy_key"] == "never_adapt"}
    ddm_counts = {(r["dataset"], r["model"], int(r["seed"])): int(r["n_alarms"]) for r in ddm_rows if r["policy_key"] == "never_adapt"}
    cells, mismatches, agreement = {}, [], {"agree": 0, "disagree": 0}
    for dataset in ds.STREAMS:
        X, y = ds.load_stream(dataset)
        for model in GRID_MODELS:
            seeds = [0] if model in an.DETERMINISTIC_MODELS else range(5)
            pooled = {"adwin": [], "ddm": []}
            for seed in seeds:
                errors = never_adapt_errors(X, y, model, seed, config)
                a = replay(errors, ADWIN(delta=config.adwin_delta))
                d = replay(errors, DDM())
                if len(a) != adwin_counts[(dataset, model, seed)]:
                    mismatches.append(f"ADWIN {dataset}/{model}/{seed}: replay {len(a)} vs grid {adwin_counts[(dataset, model, seed)]}")
                if len(d) != ddm_counts[(dataset, model, seed)]:
                    mismatches.append(f"DDM {dataset}/{model}/{seed}: replay {len(d)} vs grid {ddm_counts[(dataset, model, seed)]}")
                for alarm in a:
                    if alarm["own"] and alarm[f"w{PRIMARY_W}"]:
                        agreement["agree" if alarm["own"] == alarm[f"w{PRIMARY_W}"] else "disagree"] += 1
                pooled["adwin"] += a
                pooled["ddm"] += d
            entry = {"seeds": list(seeds)}
            for det, alarms in pooled.items():
                entry[det] = {"alarms": len(alarms), "alarms_per_seed": len(alarms) / len(seeds)}
                for w in DIRECTION_WINDOWS:
                    s, n = share(alarms, f"w{w}")
                    entry[det][f"share_falling_w{w}"] = s
                    entry[det][f"alarms_with_direction_w{w}"] = n
                if det == "adwin":
                    entry[det]["share_falling_own_estimate"] = share(alarms, "own")[0]
            cells[f"{dataset}/{model}"] = entry
    total = agreement["agree"] + agreement["disagree"]
    return {
        "cells": cells, "replay_mismatches": mismatches,
        "adwin_own_vs_window_agreement": {**agreement, "rate": agreement["agree"] / total if total else None},
    }


def event_direction(event, grid):
    if grid == "ddm":
        return direction_from_windows(_f(event.get(f"window_error_recent_{PRIMARY_W}")), _f(event.get(f"window_error_prior_{PRIMARY_W}")))
    return event["alarm_direction"] or None


def _f(value):
    return None if value in (None, "", "None") else float(value)


def section_cost_and_timing(ddm_rows):
    manifest = {m["name"]: m["n_rows"] for m in json.loads((ROOT / "results" / "dataset_manifest.json").read_text())}
    sources = {"ddm": (DDM_GRID, [r for r in ddm_rows if r["policy_key"] in ADAPTIVE])}
    with (ADWIN_DIRECTION / "grid.csv").open(newline="", encoding="utf-8") as f:
        sources["adwin"] = (ADWIN_DIRECTION, list(csv.DictReader(f)))
    out = {}
    for det, (grid, rows) in sources.items():
        per = defaultdict(lambda: {"adaptations": 0, "directed": 0, "falling": 0, "work": 0.0, "work_falling": 0.0,
                                   "gap_up": [], "gap_down": []})
        for r in rows:
            if r["model"] in an.DETERMINISTIC_MODELS and int(r["seed"]) != 0:
                continue
            n_init = manifest[r["dataset"]] - int(r["n_stream_rows"])
            previous = n_init - 1
            for e in an.load_events(grid, r["dataset"], r["model"], r["policy_key"], [int(r["seed"])]):
                acc = per[r["policy_key"]]
                direction = event_direction(e, det)
                work = float(e["work_units"])
                acc["adaptations"] += 1
                acc["work"] += work
                gap = int(e["alarm_row"]) - previous
                if direction in ("error_up", "error_down", "flat"):
                    acc["directed"] += 1
                if direction == "error_down":
                    acc["falling"] += 1
                    acc["work_falling"] += work
                    acc["gap_down"].append(gap)
                elif direction == "error_up":
                    acc["gap_up"].append(gap)
                previous = int(e["adapt_row"])
        out[det] = {an.LABELS[p]: {
            "adaptations": v["adaptations"], "with_direction": v["directed"],
            "share_on_falling_error": v["falling"] / v["directed"] if v["directed"] else None,
            "share_of_work_units_on_falling_error": v["work_falling"] / v["work"] if v["work"] else None,
            "median_rows_since_adaptation_falling": float(np.median(v["gap_down"])) if v["gap_down"] else None,
            "median_rows_since_adaptation_rising": float(np.median(v["gap_up"])) if v["gap_up"] else None,
            "direction_measure": f"window W={PRIMARY_W}" if det == "ddm" else "ADWIN own estimate",
        } for p, v in per.items()}
    return out


def section_policies(adwin_rows, ddm_rows):
    out = {}
    for det, rows in (("adwin", adwin_rows), ("ddm", ddm_rows)):
        an.verify_determinism(rows)
        res = {}
        for level in ("n60", "n20"):
            blocks = {m: an.build_blocks(rows, m, level) for m in ("accuracy", "cost")}
            res[level] = {
                "friedman_accuracy_p": an.friedman(blocks["accuracy"])["p"],
                "friedman_cost_p": an.friedman(blocks["cost"])["p"],
                "mean_ranks_accuracy": {an.LABELS[p]: r for p, r in an.mean_ranks(blocks["accuracy"]).items()},
                "mean_ranks_cost": {an.LABELS[p]: r for p, r in an.mean_ranks(blocks["cost"]).items()},
                "selector_comparisons": an.paired_comparisons(blocks, "mechanism_selector", ["always_rebuild", "fixed_schedule"]),
            }
        runs = an.index_runs(rows)
        dominance = {}
        for policy in ("always_rebuild", "fixed_schedule", "always_nudge", "mechanism_selector"):
            count = 0
            for dataset, model in an.cells_of(rows):
                seeds = an.analysis_seeds(rows, model)
                never = np.mean([an.accuracy_of(runs[(dataset, model, "never_adapt", s)]) for s in seeds])
                pol = np.mean([an.accuracy_of(runs[(dataset, model, policy, s)]) for s in seeds])
                spent = np.mean([an.cost_of(runs[(dataset, model, policy, s)]) for s in seeds]) > 0
                count += int(never - pol >= 0 and (spent or never > pol))
            dominance[an.LABELS[policy]] = count
        res["never_adapt_dominates_cells"] = dominance
        res["alarms_total_never_adapt"] = sum(int(r["n_alarms"]) for r in rows if r["policy_key"] == "never_adapt"
                                              and (r["model"] not in an.DETERMINISTIC_MODELS or r["seed"] == "0"))
        out[det] = res
    return out


def main() -> int:
    adwin_rows = an.load_grid(ADWIN_GRID)
    ddm_rows = an.load_grid(DDM_GRID)
    failures = DDM_GRID / "grid_failures.csv"
    report = {
        "ddm_grid": {"runs": len(ddm_rows), "unique": len({(r["dataset"], r["model"], r["policy_key"], r["seed"]) for r in ddm_rows}),
                     "failures_file_present": failures.exists(),
                     "detector_params": {"warm_start": 30, "warning_threshold": 2.0, "drift_threshold": 3.0}},
        "replay": section_replay(adwin_rows, ddm_rows),
        "cost_and_timing": section_cost_and_timing(ddm_rows),
        "policies": section_policies(adwin_rows, ddm_rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ddm_contrast.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    g = report["ddm_grid"]
    print(f"DDM grid: {g['runs']} runs ({g['unique']} unique), failures file present: {g['failures_file_present']}")
    rp = report["replay"]
    print(f"replay alarm-count mismatches (both detectors): {len(rp['replay_mismatches'])}")
    for m in rp["replay_mismatches"][:10]:
        print("   ", m)
    ag = rp["adwin_own_vs_window_agreement"]
    print(f"ADWIN own-estimate vs window-{PRIMARY_W} direction agreement: {ag['agree']}/{ag['agree'] + ag['disagree']}"
          f" ({100 * ag['rate']:.0f}%)\n")

    print(f"{'cell':<26}{'ADWIN alarms':>13}{'DDM alarms':>11}{'ADWIN falling (w100/200/500)':>31}{'DDM falling (w100/200/500)':>29}")
    tot = {"adwin": 0, "ddm": 0}
    for cell, e in rp["cells"].items():
        def trio(det):
            vals = [e[det][f"share_falling_w{w}"] for w in DIRECTION_WINDOWS]
            return " / ".join("--" if v is None else f"{100 * v:.0f}%" for v in vals)
        tot["adwin"] += e["adwin"]["alarms"]; tot["ddm"] += e["ddm"]["alarms"]
        print(f"{cell:<26}{e['adwin']['alarms']:>13,}{e['ddm']['alarms']:>11,}{trio('adwin'):>31}{trio('ddm'):>29}")
    print(f"{'total':<26}{tot['adwin']:>13,}{tot['ddm']:>11,}")
    for det in ("adwin", "ddm"):
        alarms = sum(e[det][f"alarms_with_direction_w{PRIMARY_W}"] for e in rp["cells"].values())
        falling = sum((e[det][f"share_falling_w{PRIMARY_W}"] or 0) * e[det][f"alarms_with_direction_w{PRIMARY_W}"] for e in rp["cells"].values())
        print(f"  pooled {det.upper()} (window {PRIMARY_W}): {falling:.0f} of {alarms} alarms on falling error ({100 * falling / alarms if alarms else float('nan'):.0f}%)")

    print("\ncost on falling-error alarms and rows since previous adaptation:")
    for det, pols in report["cost_and_timing"].items():
        for pol, v in pols.items():
            s = lambda x: "--" if x is None else f"{100 * x:.0f}%"  # noqa: E731
            r = lambda x: "--" if x is None else f"{x:.0f}"  # noqa: E731
            print(f"  {det.upper():<6}{pol:<19} adaptations {v['adaptations']:>6}  on falling {s(v['share_on_falling_error']):>5}"
                  f"  cost on falling {s(v['share_of_work_units_on_falling_error']):>5}"
                  f"  rows since adaptation: falling {r(v['median_rows_since_adaptation_falling']):>6}, rising {r(v['median_rows_since_adaptation_rising']):>6}")

    print("\npolicy comparison:")
    for det, res in report["policies"].items():
        print(f"  {det.upper()}: NeverAdapt alarms {res['alarms_total_never_adapt']:,}; NeverAdapt dominates {res['never_adapt_dominates_cells']}")
        for level in ("n60", "n20"):
            r = res[level]
            ranks = ", ".join(f"{k} {v:.2f}" for k, v in sorted(r["mean_ranks_accuracy"].items(), key=lambda kv: kv[1]))
            print(f"    {level}: Friedman accuracy p={r['friedman_accuracy_p']:.2g}, cost p={r['friedman_cost_p']:.2g}; accuracy ranks {ranks}")
            for c in r["selector_comparisons"]:
                print(f"      selector vs {an.LABELS[c['against']]:<14}{c['metric']:<9} better/worse {c['target_better_in']:>2}/{c['target_worse_in']:<2} Holm p={c['p_holm']:.2g} r={c['rank_biserial']:+.2f}")
    return 1 if rp["replay_mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
