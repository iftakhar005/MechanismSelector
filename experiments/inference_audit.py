"""Audit of the Phase 9.5 inference-cost table. DIAGNOSTIC; reads committed results only.

Answers four review questions about "AlwaysRebuild costs less in total than NeverAdapt":

1. What the pooled totals are sums over: they sum 60 runs (5 seeds for rf/sgd, seed 0 for the
   deterministic xgb/gnb), so dividing by mean stream length gives no per-prediction figure.
2. Per family x dataset, operations per prediction for the initial model vs the model
   AlwaysRebuild ends with, with node counts and visits per tree (a mean-path-depth proxy).
3. Combined training + inference per family, never pooled across families, under two
   estimates for AlwaysRebuild: Phase 9.5's mean-of-endpoints, and final-model units throughout.
4. Whether the inference saving tracks initial-training size / window size.

Output: results/analysis/inference_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"elec2": 2, "covtype": 7, "insects_abrupt": 6, "insects_gradual": 6, "insects_incremental": 6}


def main() -> None:
    g = pd.read_csv(ROOT / "results/grid/grid.csv")
    g = g[~(g.model.isin(["xgb", "gnb"]) & (g.seed != 0))]   # same run set as Phase 9.5

    na = g[g.policy_key == "never_adapt"]
    pooled_never = float((na.n_stream_rows * na.initial_inference_units).sum())
    sanity = {
        "runs_in_pooled_total": int(len(na)),
        "never_adapt_inference_total": pooled_never,
        "per_run": pooled_never / len(na),
        "per_prediction_row": pooled_never / float(na.n_stream_rows.sum()),
        "naive_total_over_mean_rows": pooled_never / float(na.n_stream_rows.mean()),
    }

    rows = []
    for (model, dataset), ar in g[g.policy_key == "always_rebuild"].groupby(["model", "dataset"]):
        nv = na[(na.model == model) & (na.dataset == dataset)]
        rows_ = float(ar.n_stream_rows.mean())
        u0, u1 = float(ar.initial_inference_units.mean()), float(ar.final_inference_units.mean())
        trees = 1 if model != "xgb" or CLASSES[dataset] == 2 else CLASSES[dataset]   # xgb: one tree per class per round
        size0, size1 = float(ar.initial_model_size.mean()), float(ar.final_model_size.mean())
        train = float(ar.total_work_units.mean())
        never = float(nv.initial_inference_units.mean()) * float(nv.n_stream_rows.mean())
        mid, final = rows_ * (u0 + u1) / 2, rows_ * u1
        rows.append({
            "model": model, "dataset": dataset, "init_train_rows": float(ar.n_init_train_rows.mean()),
            "units_per_prediction_initial": u0, "units_per_prediction_rebuilt": u1, "ratio": u1 / u0,
            "nodes_initial": None if pd.isna(ar.initial_n_nodes).all() else float(ar.initial_n_nodes.mean()),
            "nodes_rebuilt": None if pd.isna(ar.final_n_nodes).all() else float(ar.final_n_nodes.mean()),
            "visits_per_tree_initial": u0 / (size0 * trees) if model in ("xgb", "rf") else None,
            "visits_per_tree_rebuilt": u1 / (size1 * trees) if model in ("xgb", "rf") else None,
            "never_combined": never, "rebuild_training": train,
            "rebuild_combined_mid": train + mid, "rebuild_combined_final": train + final,
            "rebuild_cheaper_mid": train + mid < never, "rebuild_cheaper_final": train + final < never,
        })
    cells = pd.DataFrame(rows)
    for col in ("rebuild_cheaper_mid", "rebuild_cheaper_final"):
        cells[col] = cells[col].astype(int)          # counted per family below
    per_family = cells.groupby("model")[["never_combined", "rebuild_training", "rebuild_combined_mid",
                                         "rebuild_combined_final", "rebuild_cheaper_mid",
                                         "rebuild_cheaper_final"]].sum()
    # Spearman rank correlation between initial training size and the inference ratio, per tree family
    ratio_vs_size = {
        m: float(cells[cells.model == m][["init_train_rows", "ratio"]].corr(method="spearman").iloc[0, 1])
        for m in ("xgb", "rf")
    }

    payload = {
        "label": "diagnostic",
        "pooled_total_sanity": sanity,
        "cells": rows,
        "per_family": per_family.reset_index().to_dict(orient="records"),
        "spearman_init_rows_vs_ratio": ratio_vs_size,
        "caveats": [
            "rebuilt-model units are measured once, on the final model, not along the stream",
            "xgb and gnb have one seed per cell",
            "window size and window class composition are not separated by this audit",
        ],
    }
    out = ROOT / "results/analysis/inference_audit.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(json.dumps(sanity, indent=2))
    print(cells[["model", "dataset", "init_train_rows", "units_per_prediction_initial",
                 "units_per_prediction_rebuilt", "ratio", "visits_per_tree_initial",
                 "visits_per_tree_rebuilt"]].round(2).to_string(index=False))
    counts = ["rebuild_cheaper_mid", "rebuild_cheaper_final"]
    print(per_family.drop(columns=counts).round(-3).to_string())
    print(per_family[counts].to_string())
    print("Spearman(initial training rows, rebuilt/initial ratio):", ratio_vs_size)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
