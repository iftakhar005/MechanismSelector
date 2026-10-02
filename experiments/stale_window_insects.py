"""Stale training windows on real data: INSECTS abrupt. DIAGNOSTIC of committed results.

Motivated by the post-hoc finding in docs/confirm/REPORT.md: on synthetic abrupt streams, acting
at alarm time trains the replacement model on a window that is mostly pre-change data. This
checks the same thing on the one real stream with published change points (insects_abrupt;
Souza et al. 2020 Table 2, verified in Phase 9.3), using the frozen Phase 6 ADWIN grid. It reads
event logs only and runs no model.

For every adaptation episode:
  lag              rows from the most recent published change point to the alarm
  true detection   lag < 1,000 (Phase 9.3's window)
  post_share       share of the training window (the logged buffer) that lies after that change
  next200          the returned model's balanced accuracy on the next 200 stream rows -- the
                   runner's own causal reference measurement, read from the *next* event, and
                   used only when that event did not carry the reference forward

Observational: post_share is not assigned, so this shows association, not effect. The
interventional test is the delay-only run on the real streams.

Output: results/analysis/stale_window_insects.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BINS = [-0.01, 0.10, 0.25, 0.50, 1.0]


def rho(x: pd.DataFrame) -> dict | None:
    if len(x) < 6:
        return None
    r, p = spearmanr(x.post_share, x.next200)
    return {"n": int(len(x)), "rho": float(r), "p": float(p)}


def main() -> None:
    cps = np.asarray(json.loads((ROOT / "results/analysis/phase9_change_points.json").read_text())
                     ["ground_truth"]["change_points"])
    ev = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "results/grid/events/insects_abrupt__*.csv"))],
                   ignore_index=True)
    ev = ev[~(ev.model.isin(["xgb", "gnb"]) & (ev.seed != 0))]       # deterministic families: one seed
    ev = ev.sort_values(["model", "policy_key", "seed", "event_index"]).reset_index(drop=True)

    g = ev.groupby(["model", "policy_key", "seed"])
    ev["next200"] = g.reference_accuracy_used.shift(-1)
    carried = g.reference_carried.shift(-1).astype(str) != "False"
    ev.loc[carried, "next200"] = np.nan

    diff = ev.alarm_row.to_numpy()[:, None] - cps[None, :]
    diff = np.where(diff >= 0, diff, 10**9)
    ev["lag"] = diff.min(axis=1)
    ev["change_point"] = cps[diff.argmin(axis=1)]
    ev["true_detection"] = ev.lag < 1000
    ev["post_share"] = np.clip(ev.adapt_row + 1 - ev.change_point, 0, ev.buffer_rows) / ev.buffer_rows

    t = ev[ev.true_detection]
    window = (t.groupby("model")
               .agg(n=("lag", "size"), median_lag=("lag", "median"), median_buffer=("buffer_rows", "median"),
                    mean_post_share=("post_share", "mean"),
                    share_below_half=("post_share", lambda s: float((s < 0.5).mean())))
               .reset_index())

    r = t[(t.action == "REBUILD") & t.next200.notna()].copy()
    r["bin"] = pd.cut(r.post_share, BINS).astype(str)
    by_bin = (r.groupby("bin").agg(n=("next200", "size"), next200=("next200", "mean"),
                                    acc_before=("acc_before", "mean")).reset_index())
    never = t[(t.policy_key == "never_adapt") & t.next200.notna()]

    payload = {
        "label": "diagnostic (observational)",
        "change_points": cps.tolist(),
        "episodes": int(len(ev)), "true_detections": int(len(t)),
        "training_window_at_true_detections": window.to_dict(orient="records"),
        "rebuilds_after_true_detections": int(len(r)),
        "next200_by_post_share_bin": by_bin.to_dict(orient="records"),
        "spearman_all": rho(r),
        "spearman_by_model": {m: rho(x) for m, x in r.groupby("model")},
        "spearman_within_change_point": {int(c): rho(x) for c, x in r.groupby("change_point")},
        "never_adapt_next200_after_true_detections": {"n": int(len(never)), "mean": float(never.next200.mean())},
        "rebuild_next200_post_share_below_0.25": float(r[r.post_share < 0.25].next200.mean()),
        "rebuild_next200_post_share_at_least_0.5": float(r[r.post_share >= 0.5].next200.mean()),
    }
    out = ROOT / "results/analysis/stale_window_insects.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(window.round(2).to_string(index=False))
    print(by_bin.round(3).to_string(index=False))
    print(json.dumps({k: payload[k] for k in ("spearman_all", "spearman_by_model", "spearman_within_change_point",
                                              "never_adapt_next200_after_true_detections")}, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
