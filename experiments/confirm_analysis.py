"""Analysis of the confirm-before-acting grid, by the criteria fixed in
docs/confirm/PREREGISTRATION.md (nothing here is tuned to the results).

Per (policy, model, W > 0), paired with W = 0 over stream x seed (30 pairs):
  helps   : median training-cost reduction >= 20%, mean-accuracy loss <= 0.5 pp,
            and post-change accuracy loss <= 1.0 pp (abrupt streams)
  hurts   : post-change accuracy loss > 1.0 pp
  neither : otherwise
Pairs whose W = 0 run spent no training are excluded from the median reduction and counted.

Output: results/confirm/summary.json, and tables printed for the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
KEY = ["stream", "model", "seed"]


def wil(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return None
    return float(wilcoxon(a, b).pvalue)


def main() -> None:
    runs = pd.read_csv(ROOT / "results/confirm/runs.csv")
    eps = pd.read_csv(ROOT / "results/confirm/episodes.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)

    # alarm-count validity check (pre-registered: < 5 alarms/run under W = 0 on a sticky stream)
    base = runs[(runs.confirm_rows == 0) & (runs.policy_key != "never_adapt")]
    alarms = base.groupby(["stream", "model"]).n_alarms.mean().unstack().round(1)
    low = [(s, m) for s in alarms.index for m in alarms.columns
           if "sticky" in s and alarms.loc[s, m] < 5]

    verdicts = []
    for (policy, model), grp in runs[runs.policy_key != "never_adapt"].groupby(["policy_key", "model"]):
        zero = grp[grp.confirm_rows == 0].set_index(KEY)
        for w in sorted(grp.confirm_rows.unique()):
            if w == 0:
                continue
            cur = grp[grp.confirm_rows == w].set_index(KEY).loc[zero.index]
            usable = [k for k in zero.index if (k[0], k[1]) not in low]
            z, c = zero.loc[usable], cur.loc[usable]
            spent = z.total_work_units > 0
            reduction = 1 - c.total_work_units[spent] / z.total_work_units[spent]
            acc_loss = 100 * (z.mean_prequential_accuracy - c.mean_prequential_accuracy)
            abrupt = [k for k in usable if "_abrupt_" in k[0]]
            post_loss = 100 * (zero.loc[abrupt].post_change_accuracy - cur.loc[abrupt].post_change_accuracy)
            infer_change = c.total_inference_ops / z.total_inference_ops - 1
            med_red, mean_acc, mean_post = float(reduction.median()), float(acc_loss.mean()), float(post_loss.mean())
            if mean_post > 1.0:
                verdict = "hurts"
            elif med_red >= 0.20 and mean_acc <= 0.5 and mean_post <= 1.0:
                verdict = "helps"
            else:
                verdict = "neither"
            verdicts.append({
                "policy": policy, "model": model, "W": int(w), "pairs": len(usable),
                "pairs_without_training_at_W0": int((~spent).sum()),
                "median_training_reduction": med_red,
                "total_training_reduction": float(1 - c.total_work_units.sum() / z.total_work_units.sum())
                if z.total_work_units.sum() else None,
                "mean_accuracy_loss_pp": mean_acc, "mean_post_change_loss_pp": mean_post,
                "median_inference_change": float(infer_change.median()),
                "p_training": wil(z.total_work_units, c.total_work_units),
                "p_accuracy": wil(z.mean_prequential_accuracy, c.mean_prequential_accuracy),
                "verdict": verdict,
            })
    v = pd.DataFrame(verdicts)

    # per stream detail, means over seeds
    detail = runs.groupby(["stream", "model", "policy_key", "confirm_rows"]).agg(
        alarms=("n_alarms", "mean"), adaptations=("n_adaptations", "mean"),
        dismissed=("n_alarms_dismissed", "mean"), train_wu=("total_work_units", "mean"),
        infer_ops=("total_inference_ops", "mean"), acc=("mean_prequential_accuracy", "mean"),
        post_acc=("post_change_accuracy", "mean")).reset_index()

    # which episodes get dismissed: true vs false detections
    e = eps[eps.confirm_rows > 0]
    dismiss = e.groupby(["confirm_rows", "truth"]).outcome.apply(lambda s: (s == "dismissed").mean()).unstack()
    dismiss_n = e.groupby(["confirm_rows", "truth"]).size().unstack()
    by_dir = e.groupby(["confirm_rows", "alarm_direction"]).outcome.apply(
        lambda s: (s == "dismissed").mean()).unstack()

    out = ROOT / "results/confirm/summary.json"
    out.write_text(json.dumps({
        "label": "pre-registered test (docs/confirm/PREREGISTRATION.md)",
        "alarms_per_run_W0": alarms.reset_index().to_dict(orient="records"),
        "excluded_low_alarm_cells": [list(x) for x in low],
        "verdicts": verdicts,
        "dismissed_share_by_truth": dismiss.reset_index().to_dict(orient="records"),
        "episodes_by_truth": dismiss_n.reset_index().to_dict(orient="records"),
        "dismissed_share_by_direction": by_dir.reset_index().to_dict(orient="records"),
        "detail": detail.to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")

    print("alarms per run at W=0:\n", alarms, "\nexcluded (sticky, <5 alarms):", low)
    print("\nverdicts:\n", v.round(3).to_string(index=False))
    print("\nshare of episodes dismissed, by truth:\n", dismiss.round(3), "\n", dismiss_n)
    print("\nshare dismissed, by alarm direction:\n", by_dir.round(3))
    print("\ndetail:\n", detail.round(3).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
