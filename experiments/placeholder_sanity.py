"""Are zero-weight placeholders harmless, or merely harmless-looking?

Placeholders are injected when a window lacks a class -- 92% of covtype windows
-- so "a different, equally valid draw" has to be demonstrated, not asserted.
Showing that elec2 results are unchanged proves nothing, because elec2 never
needs a placeholder. This forces them where they are not needed and measures
the effect.

## Design

elec2 (binary; no window ever lacks a class) with Random Forest and SGD -- the
two families for which placeholders are not bit-for-bit inert, because zero-
weight rows still occupy bootstrap / shuffle indices. All five policies, seeds
0-4, run twice:

- **normal**: the grid's code, untouched. No placeholders ever occur on elec2.
- **forced**: every `fit` during adaptation gets one zero-weight placeholder
  row -- a copy of the window's first row with its own (present) label, exactly
  the form a real placeholder takes. Initial training is not forced, so the two
  conditions start from the same model and any difference comes from
  placeholders during adaptation alone.

## Questions, in order

1. **Controls.** Runs that never `fit` during adaptation must be identical:
   `NeverAdapt` (never adapts) and SGD `AlwaysNudge` (only `partial_fit`). If
   they differ, the harness is broken and nothing else here means anything.
2. **Bit-for-bit.** Are the other runs identical? Expected no -- the placeholder
   shifts the random stream.
3. **Bias.** Is the forced-minus-normal difference centred on zero? A systematic
   shift in one direction would make placeholders harmful, not just noisy.
4. **Magnitude.** Is it no larger than changing the seed? The reference is the
   difference between consecutive seeds of the *same* policy in the normal
   condition: a pure change of random stream, which is what a harmless
   placeholder should amount to.

Run:  python experiments/placeholder_sanity.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import datasets as ds  # noqa: E402
import mechanisms  # noqa: E402
from policies import POLICY_KEYS  # noqa: E402
from runner import RunConfig, run_stream  # noqa: E402

warnings.filterwarnings("ignore")

MODELS = ("rf", "sgd")
SEEDS = range(5)
OUTCOMES = ("mean_prequential_accuracy", "final_prequential_accuracy", "total_work_units")
EXACT_FIELDS = (
    "n_adaptations", "n_skip", "n_nudge", "n_rebuild", "total_work_units",
    "mean_prequential_accuracy", "final_prequential_accuracy", "final_model_size",
)
_original_anchor = mechanisms.anchor_missing_classes


def forced_anchor(X, y, classes):
    """Real placeholders when a class is missing; otherwise force one anyway."""
    X_out, y_out, kwargs = _original_anchor(X, y, classes)
    if kwargs:
        return X_out, y_out, kwargs
    X_forced = np.vstack([X, X[:1]])
    y_forced = np.concatenate([y, y[:1]])
    weight = np.concatenate([np.ones(len(y)), np.zeros(1)])
    return X_forced, y_forced, {"sample_weight": weight}


def run_condition(X, y, forced: bool) -> dict:
    mechanisms.anchor_missing_classes = forced_anchor if forced else _original_anchor
    config = RunConfig(track_energy=False)
    records = {}
    try:
        for model in MODELS:
            for policy in POLICY_KEYS:
                for seed in SEEDS:
                    record, events = run_stream(X, y, model, policy, seed, config, "elec2")
                    record["_forced_placeholder_events"] = sum(e["n_placeholder_rows"] > 0 for e in events)
                    records[(model, policy, seed)] = record
                    print(f"  {'forced' if forced else 'normal'} {model} {policy} seed={seed}: "
                          f"acc {record['mean_prequential_accuracy']:.4f}, "
                          f"{record['total_work_units']:,} WU, placeholder events "
                          f"{record['_forced_placeholder_events']}", flush=True)
    finally:
        mechanisms.anchor_missing_classes = _original_anchor
    return records


def is_control(model: str, policy: str) -> bool:
    return policy == "never_adapt" or (model == "sgd" and policy == "always_nudge")


def main() -> int:
    X, y = ds.load_stream("elec2")
    started = time.perf_counter()
    normal = run_condition(X, y, forced=False)
    forced = run_condition(X, y, forced=True)

    report = {"controls": {}, "exact": {}, "bias": {}, "magnitude": {}}
    ok = True

    print("\n=== 1. controls: runs with no adaptation fit must be identical ===")
    for (model, policy, seed), rec in normal.items():
        if not is_control(model, policy):
            continue
        other = forced[(model, policy, seed)]
        same = all(rec[f] == other[f] for f in EXACT_FIELDS)
        injected = other["_forced_placeholder_events"]
        report["controls"][f"{model}/{policy}/{seed}"] = {"identical": same, "forced_events": injected}
        if not same or injected:
            ok = False
            print(f"  CONTROL FAILED {model}/{policy}/seed{seed}: identical={same}, forced events={injected}")
    print("  all controls identical, no placeholders injected" if ok else "  HARNESS INVALID")

    print("\n=== 2. bit-for-bit: adaptation runs ===")
    for model in MODELS:
        keys = [k for k in normal if k[0] == model and not is_control(*k[:2])]
        identical = sum(all(normal[k][f] == forced[k][f] for f in EXACT_FIELDS) for k in keys)
        injected = sum(forced[k]["_forced_placeholder_events"] > 0 for k in keys)
        report["exact"][model] = {"runs": len(keys), "identical": identical, "runs_with_forced_placeholders": injected}
        print(f"  {model}: {identical}/{len(keys)} identical; placeholders forced in {injected}/{len(keys)} runs")

    print("\n=== 3-4. bias and magnitude, per model, over adapting policies x seeds ===")
    for model in MODELS:
        report["bias"][model], report["magnitude"][model] = {}, {}
        for outcome in OUTCOMES:
            scale = 100.0 if "accuracy" in outcome else 1.0
            unit = "pp" if "accuracy" in outcome else "WU"
            adapting = [p for p in POLICY_KEYS if not is_control(model, p)]

            diffs = [scale * (float(forced[(model, p, s)][outcome]) - float(normal[(model, p, s)][outcome]))
                     for p in adapting for s in SEEDS]
            seed_diffs = [scale * (float(normal[(model, p, s + 1)][outcome]) - float(normal[(model, p, s)][outcome]))
                          for p in adapting for s in list(SEEDS)[:-1]]

            nonzero = [d for d in diffs if d != 0]
            p_wilcoxon = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 6 else float("nan")
            ci = stats.t.interval(0.95, len(diffs) - 1, loc=mean(diffs),
                                  scale=stdev(diffs) / len(diffs) ** 0.5) if stdev(diffs) > 0 else (0.0, 0.0)
            mad_forced = mean(abs(d) for d in diffs)
            mad_seed = mean(abs(d) for d in seed_diffs)
            p_scale = float(stats.mannwhitneyu([abs(d) for d in diffs], [abs(d) for d in seed_diffs]).pvalue)

            report["bias"][model][outcome] = {
                "mean_diff": mean(diffs), "ci95": list(ci), "wilcoxon_p": p_wilcoxon, "n": len(diffs),
            }
            report["magnitude"][model][outcome] = {
                "mean_abs_forced_diff": mad_forced, "mean_abs_seed_diff": mad_seed,
                "ratio": mad_forced / mad_seed if mad_seed else float("nan"), "mannwhitney_p": p_scale,
            }
            print(f"  {model} {outcome:<28} forced-normal mean {mean(diffs):+9.3f} {unit} "
                  f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  Wilcoxon p={p_wilcoxon:.3f}  |  "
                  f"mean |diff| forced {mad_forced:.3f} vs seed {mad_seed:.3f} (ratio {mad_forced / mad_seed if mad_seed else float('nan'):.2f}, "
                  f"MWU p={p_scale:.3f})")

    report["runtime_min"] = (time.perf_counter() - started) / 60
    report["normal"] = {"/".join(map(str, k)): {f: v[f] for f in EXACT_FIELDS} for k, v in normal.items()}
    report["forced"] = {"/".join(map(str, k)): {f: v[f] for f in EXACT_FIELDS} for k, v in forced.items()}
    out = ROOT / "results" / "placeholder_sanity.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}  ({report['runtime_min']:.1f} min)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
