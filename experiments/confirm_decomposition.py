"""POST-HOC decomposition of the confirm-before-acting result. Not pre-registered.

The pre-registered grid changed two things at once: *when* the adaptation window is taken
(W rows later) and *which* alarms are acted on (dismissal). The delay-only control
(`confirm_grid.py --delay-only`: wait W rows, always adapt) separates them:

  delay  - now      effect of taking the window later
  confirm - delay   effect of dismissing alarms, given the delay

It also measures the mechanism directly: for true detections that were acted on, how far after
the change ADWIN alarmed, and what share of the (up to) 1,000-row window was post-change data.
The share assumes a full 1,000-row buffer, so it is an upper-bound-style approximation where the
buffer was cleared less than 1,000 rows earlier.

Output: results/confirm/decomposition.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import synthetic  # noqa: E402

KEY = ["stream", "model", "policy_key", "seed"]


def main() -> None:
    r = pd.read_csv(ROOT / "results/confirm/runs.csv")
    d = pd.read_csv(ROOT / "results/confirm_delay_only/runs.csv")
    r = r[r.policy_key != "never_adapt"].assign(cond=lambda x: np.where(x.confirm_rows == 0, "now", "confirm"))
    a = pd.concat([r, d.assign(cond="delay")], ignore_index=True)
    a["kind"] = a.stream.str.split("_").str[1] + "_" + a.stream.str.split("_").str[2]
    now = a[a.cond == "now"].set_index(KEY)

    parts = []
    for w in (200, 500):
        for cond in ("delay", "confirm"):
            x = a[(a.cond == cond) & (a.confirm_rows == w)].set_index(KEY).loc[now.index]
            parts.append(pd.DataFrame({
                "W": w, "cond": cond, "kind": x.kind,
                "d_acc_pp": 100 * (x.mean_prequential_accuracy - now.mean_prequential_accuracy),
                "d_post_pp": 100 * (x.post_change_accuracy - now.post_change_accuracy),
                "train_ratio": x.total_work_units / now.total_work_units.replace(0, np.nan),
                "d_adaptations": x.n_adaptations - now.n_adaptations,
            }, index=now.index))
    t = pd.concat(parts).reset_index()

    by_model = (t.groupby(["policy_key", "model", "W", "cond"])
                 .agg(d_acc_pp=("d_acc_pp", "mean"), d_post_pp=("d_post_pp", "mean"),
                      median_train_ratio=("train_ratio", "median"), d_adaptations=("d_adaptations", "mean"))
                 .reset_index())
    by_kind = (t.groupby(["kind", "W", "cond"])
                .agg(d_acc_pp=("d_acc_pp", "mean"), d_post_pp=("d_post_pp", "mean"),
                     median_train_ratio=("train_ratio", "median"))
                .reset_index())

    e = pd.read_csv(ROOT / "results/confirm/episodes.csv")
    de = pd.read_csv(ROOT / "results/confirm_delay_only/episodes.csv").assign(cond="delay")
    e = e.assign(cond=np.where(e.confirm_rows == 0, "now", "confirm"))
    ep = pd.concat([e, de], ignore_index=True)
    ep = ep[(ep.truth.astype(str).str.lower() == "true") & (ep.outcome == "adapted")
            & (ep.policy_key != "never_adapt")].copy()
    cps = np.asarray(synthetic.CHANGE_POINTS)
    diff = ep.alarm_row.to_numpy()[:, None] - cps[None, :]
    ep["detect_lag"] = np.where((diff >= 0) & (diff < 1000), diff, 10**9).min(axis=1)
    ep["post_change_share"] = np.clip(ep.detect_lag + ep.confirm_rows, 0, 1000) / 1000
    window = (ep.groupby(["cond", "confirm_rows"])
                .agg(n=("detect_lag", "size"), median_detect_lag=("detect_lag", "median"),
                     mean_post_change_share=("post_change_share", "mean"))
                .reset_index())

    out = ROOT / "results/confirm/decomposition.json"
    out.write_text(json.dumps({
        "label": "post-hoc, not pre-registered",
        "by_policy_model": by_model.to_dict(orient="records"),
        "by_stream_kind": by_kind.to_dict(orient="records"),
        "training_window_at_true_detections": window.to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(by_kind.round(2).to_string(index=False))
    print(window.round(2).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
