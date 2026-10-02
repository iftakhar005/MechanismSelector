"""Analysis of delay_real.py by the criteria fixed in docs/confirm/PREREGISTRATION_REAL.md.

Primary score: mean prequential accuracy for binary streams (elec2), balanced accuracy for
multi-class streams -- the study's scorer convention. Per (policy, model, W > 0), paired with
W = 0 over stream x seed:

  rf, sgd (25 pairs: 5 streams x 5 seeds)
    supported    median gain >= +1.0 pp, Wilcoxon p < 0.05, mean gain > 0 in >= 3 of 5 streams
    contradicted median gain <= -1.0 pp and Wilcoxon p < 0.05
  xgb, gnb (5 pairs, one seed: deterministic)
    supported    median gain >= +1.0 pp and gain > 0 in >= 4 of 5 streams
    contradicted median gain <= -1.0 pp and gain < 0 in >= 4 of 5 streams
  inconclusive otherwise

H1 (primary) is judged on rf and xgb at W = 500, for each policy. Everything else is reported
without a verdict.

    python experiments/delay_real_analysis.py [--dir results/delay_real]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
KEY = ["stream", "model", "seed"]
DETERMINISTIC = ("xgb", "gnb")


def score(df: pd.DataFrame) -> pd.Series:
    return np.where(df.scorer == "accuracy", df.mean_prequential_accuracy, df.mean_prequential_balanced_accuracy)


def p_value(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return None if np.allclose(d, 0) else float(wilcoxon(a, b).pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/delay_real")
    args = ap.parse_args()
    d = ROOT / args.dir
    runs = pd.read_csv(d / "runs.csv")
    eps = pd.read_csv(d / "episodes.csv")
    runs["score"] = score(runs)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)

    results = []
    for (policy, model), grp in runs[runs.policy_key != "never_adapt"].groupby(["policy_key", "model"]):
        now = grp[grp.delay_rows == 0].set_index(KEY).sort_index()
        for w in (200, 500):
            cur = grp[grp.delay_rows == w].set_index(KEY).loc[now.index]
            gain = 100 * (cur.score - now.score)
            by_stream = gain.groupby(level="stream").mean()
            med, p = float(gain.median()), p_value(cur.score, now.score)
            n_up, n_down = int((by_stream > 0).sum()), int((by_stream < 0).sum())
            if model in DETERMINISTIC:   # 5 pairs: no Wilcoxon can reach 0.05, so judge by sign
                up = med >= 1.0 and n_up >= 4
                down = med <= -1.0 and n_down >= 4
            else:
                up = med >= 1.0 and p is not None and p < 0.05 and n_up >= 3
                down = med <= -1.0 and p is not None and p < 0.05
            verdict = "supported" if up else "contradicted" if down else "inconclusive"
            results.append({
                "policy": policy, "model": model, "W": w, "pairs": int(len(gain)),
                "median_gain_pp": med, "mean_gain_pp": float(gain.mean()), "p": p,
                "streams_with_gain": int((by_stream > 0).sum()),
                "gain_by_stream_pp": by_stream.round(3).to_dict(),
                "median_training_ratio": float((cur.total_work_units / now.total_work_units.replace(0, np.nan)).median()),
                "median_inference_ratio": float((cur.total_inference_ops / now.total_inference_ops).median()),
                "verdict": verdict,
                "h1": model in ("rf", "xgb") and w == 500,
            })
    v = pd.DataFrame(results)

    never = runs[runs.policy_key == "never_adapt"].set_index(KEY).score
    vs_never = (runs[runs.policy_key != "never_adapt"].assign(gain_over_never=lambda x: 100 * (
        x.score.to_numpy() - never.loc[list(zip(x.stream, x.model, x.seed))].to_numpy())))
    vs_never = vs_never.groupby(["policy_key", "model", "delay_rows"]).gain_over_never.mean().unstack()

    window = None
    if "post_change_share" in eps and eps.post_change_share.notna().any():
        t = eps[eps.true_detection.astype(str).str.lower() == "true"]
        window = t.groupby(["delay_rows"]).agg(n=("detect_lag", "size"), median_lag=("detect_lag", "median"),
                                               post_change_share=("post_change_share", "mean")).reset_index()
    post = runs[runs.post_change_accuracy.notna()].groupby(["policy_key", "model", "delay_rows"]).post_change_accuracy.mean().unstack()

    h1 = v[v.h1]
    out = d / "summary.json"
    out.write_text(json.dumps({
        "label": "pre-registered (docs/confirm/PREREGISTRATION_REAL.md)",
        "h1_verdicts": h1[["policy", "model", "verdict"]].to_dict(orient="records"),
        "results": results,
        "gain_over_never_adapt_pp": vs_never.reset_index().to_dict(orient="records"),
        "insects_abrupt_window_at_true_detections": None if window is None else window.to_dict(orient="records"),
        "insects_abrupt_post_change_accuracy": None if post.empty else post.reset_index().to_dict(orient="records"),
    }, indent=2, default=float), encoding="utf-8")

    print("H1 (rf, xgb at W=500):\n", h1[["policy", "model", "median_gain_pp", "p", "streams_with_gain", "verdict"]].round(3).to_string(index=False))
    print("\nall:\n", v.drop(columns=["gain_by_stream_pp", "h1"]).round(3).to_string(index=False))
    print("\ngain over NeverAdapt (pp):\n", vs_never.round(2))
    if window is not None:
        print("\ninsects_abrupt training window at true detections:\n", window.round(2).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
