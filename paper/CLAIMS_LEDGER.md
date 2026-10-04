# Claims ledger

Every number in `paper.tex` with its source file, metric, scope and label.
**[V]** = pre-declared verdict (judged against rules fixed before the results
were seen). **[D]** = diagnostic (a measurement that carries no verdict).

Shared scope unless stated otherwise: 5 streams (Elec2, INSECTS abrupt/gradual/
incremental, Covertype 100k prefix) × 4 families (XGBoost, RandomForest, SGD,
GaussianNB) × 5 seeds, ADWIN δ = 0.002 two-sided on the raw 0/1 error stream.
Deterministic families (XGBoost, GaussianNB) contribute seed 0 only, so n = 60
blocks overall and n = 12 per stream.

Accuracy metrics used in the paper:
- **whole-stream prequential, balanced** — balanced on multi-class streams; the
  metric the floor uses. Default for policy comparisons.
- **whole-stream prequential, plain** — plain 0/1.
- **horizon accuracy, plain** — post-alarm rows only, pooled over families
  (Phase 10 oracle tables).

---

## Method section

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| 500-run grid; 5 × 4 × 5 × 5 | `results/grid/grid.csv` | run count | [D] |
| Replicate grid under DDM | `results/grid_ddm/grid.csv` | run count (500) | [D] |
| Nudge = 4% of a rebuild (trees), 0.6–1.0% (SGD), 80% (GNB) | `results/mechanism_ratios.json` | work units, per family | [D] |
| 92% of 1,000-row Covertype windows lack a class | `src/mechanisms.py` docstring, measured in Phase 3 | share of windows | [D] |
| Guard thresholds 50 holdout / 100 train ≈ 250-row window | `src/selector.py` | configuration | — |
| Floor = reference − 2 points; reference from next 200 rows; buffer cap 1,000 | `src/runner.py`, `src/selector.py` | configuration | — |
| n = 60 overall, n = 12 per stream | `results/analysis/phase7.json` (`determinism_groups_verified` = 50) | block counts | [D] |

## Results 1 — the selector does not win

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| Friedman accuracy p = 0.18 (n60), 0.45 (n20) | `phase7.json` | whole-stream prequential, balanced on multi-class | [D] |
| Friedman cost p = 2.9e−43 | `phase7.json` | total work units | [D] |
| Selector vs AlwaysRebuild: median −1.3 points, Holm p = 0.017 | `phase7.json` | accuracy, n = 60 | [D] |
| Selector cheaper in 51 of 57 blocks, r = −0.96 | `phase7.json` | work units, n = 60 | [D] |
| Selector dearer than FixedSchedule, Holm p = 0.0087; n20 p = 0.088; DDM p = 0.065 | `phase7.json`, `ddm_contrast.json` | work units | [D] |
| NeverAdapt at least as accurate at zero cost in 10 of 20 cells | `detector_premise.json` | accuracy, per cell | [D] |
| Per-stream medians: +20.3 (gradual), +36.2 (incremental), −14.8 (Covertype) | `detector_premise.json` | AlwaysRebuild − NeverAdapt, accuracy points | [D] |

## Results 2 — how cheap repair fails

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| 2,830 adaptations: 841 SKIP, 164 NUDGE, 1,825 REBUILD | `results/grid/grid.csv` | counts, selector, 60 runs | [D] |
| 15.2% of 1,079 nudge attempts cleared the floor | `grid.csv` | counts | [D] |
| 69% of 915 failures missed by > 10 points; median deficit 22.5 points in that group | `results/grid/events/` | floor − nudged accuracy, points | [D] |
| RF changed no prediction in 60% of 1,876 nudges; GNB 77% of 364 | `phase7.json` → `event_split` | prediction change, AlwaysNudge | [D] |
| SGD median 9.4% of predictions changed, 0.0 point gain; XGB +2.5 points over 212 | `phase7.json` → `event_split` | per-nudge medians | [D] |
| Full-window nudge: median 0.00 points over 1,225 paired alarms; XGB +2.28 | `nudge_window_diagnostic.json` | paired per-alarm, next-200-row scoring | [D] |
| 910 of 1,825 rebuilds guard-forced (32% of adaptations) | `grid.csv` (`n_degraded`) | counts | [D] |

## Results 3 — what the alarms are

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| 27.4% of ADWIN alarms within 1,000 rows of a change point | `phase9_change_points.json` | NeverAdapt, INSECTS-abrupt only, tolerance 1,000 | [D] |
| Change points 14352; 19500; 33240; 38682; 39510 | Souza et al. 2020, Table 2 (arXiv:2505.00113 p. 37), verified | ground truth | — |
| Block-shuffling removes 40% (ADWIN) / 69% (DDM) of alarms | `phase9_block_shuffle.json` | Elec2 + Covertype only, blocks of 50 | [D] |
| Error-stream lag-1 autocorrelation +0.58 to +0.90 | `phase9_block_shuffle.json` | Elec2 + Covertype | [D] |
| AR(1) no-drift null reproduces falling-error share: 0.421 vs 0.416 observed | `phase9_direction_null.json` | ADWIN, NeverAdapt, all 5 streams | [D] |
| i.i.d. null produced 0 ADWIN alarms | `phase9_direction_null.json` | 20 cells | [D] |
| 38.1% of 2,515 ADWIN alarms significantly above reference; 36.5% n_eff-corrected | `phase9_reference_noise.json` | one-sided two-proportion test, α = 0.05, plain accuracy | [D] |
| DDM equivalent 8.6% / 3.5% | `phase9_reference_noise.json` | same test, DDM grid | [D] |
| DDM `p_min` below our reference error in 79% of alarms, median 14.7 points | `phase9_ddm_internals.json` | NeverAdapt + DDM, 20 cells | [D] |

## Results 4 — the oracle

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| G1 = 16.8–25.5% (Elec2), 24.8–37.1% (Covertype) | `phase10_per_stream.json` | share of selector's J, λ grid, horizon "next" | [D] |
| 2,595 alarms | `phase10_oracle.json` | ADWIN, Elec2 + Covertype, seeds 0–4 | [D] |
| SKIP 80.8% (Covertype) / 78.9% (Elec2) of the advantage; NUDGE ~16%; REBUILD 3–5% | `phase10_per_stream.json` | λ = 1e3, horizon "next" | [D] |
| Always-SKIP captures 95.3% / 78.1% / −3.6% at λ = 1e2 / 1e3 / 1e4 | `phase10_summary.json` | **in-sample reference values**, labelled as such | [D] |
| Cross-stream transfer −0.122 / +0.162 at λ = 1e4 | `phase10_learnability.json` | out-of-sample, GBM, train one stream test the other | [D] |
| Leave-one-seed-out 0.645 (Covertype) / 0.085 (Elec2) at λ = 1e4 | `phase10_learnability.json` | out-of-sample | [D] |

## Results 5 — inference dominates

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| Inference 81–90% of the selector's combined ops | `phase10_per_stream.json` | Elec2 + Covertype, horizon "next" | [D] |
| Inference 85–100% pooled across policies | `phase9_inference_cost.json` | whole streams, all 5, mean horizon 50,279 predictions/run | [D] |
| Mean 779 operations per prediction (range 9–8,046) | `phase9_inference_cost.json` + `grid.csv` | per run, divided by that run's stream length | [D] |
| 32 kept XGBoost nudges (24 Covertype); 17 RF | `nudge_growth_diagnostic.json` | selector trajectory, all seeds | [D] |
| XGBoost nudge adds +18.8 to +184.9 ops/prediction | `nudge_growth_diagnostic.json` | growth term, per-cell medians | [D] |
| RF adds +4.8, +8.0, +1.3, −0.2, −2.1 ops/prediction | `nudge_growth_diagnostic.json` | growth term, per-cell medians | [D] |
| Break-even ≈ 300 predictions; gaps 640–17,014 | `nudge_growth_diagnostic.json` | per-cell medians | [D] |
| Rebuilt-model cheapness is a window-size artifact (control/initial 0.57–0.88) | `inference_audit.json` | ops/prediction on a common 1,000-row probe | [D] |

## Results 6 — WaitAndCheck

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| **Verdict NO** (rule 1 fails 2 of 5; rule 3 fails) | `phase14_waitandcheck.json` | pre-declared rules, W = 200 | **[V]** |
| Training ratios 0.494, 0.578, 0.867, 1.188, 3.235 | `phase14_waitandcheck.json` | median over family × seed, per stream | **[V]** input |
| INSECTS-incremental −1.07 points ("near, but fails the pre-declared rule") | `phase14_waitandcheck.json` | whole-stream prequential, balanced | **[V]** input |
| Accuracy +1.98 to +9.40 points on 4 of 5 | `phase14_waitandcheck.json` | whole-stream prequential, balanced | [D] |
| Pooled median +2.8 points, Holm p = 0.0004 | `phase14_waitandcheck.json` | Wilcoxon, n = 60 | [D] |
| Per stream: abrupt Holm p = 0.0024 (accuracy), gradual 0.0098 | `phase14_breakdown.json` | Wilcoxon, n = 12 per stream, Holm across 5 | [D] |
| INSECTS-abrupt training +332,806 median, Holm p = 0.0024 | `phase14_breakdown.json` | work units | [D] |
| Guard-forced rebuilds 752 → 40 (Covertype), 143 → 10 (Elec2) | `phase14_waitandcheck.json` | counts, W = 200 | [D] |
| 23 of 79 cancelled alarms within 1,000 rows of a change point | `phase14_waitandcheck.json` | INSECTS-abrupt only | [D] |
| Loses to NeverAdapt on Covertype by 8.6 points | `phase14_waitandcheck.json` | balanced accuracy | [D] |

## Results 7 — ConfirmMD3 (comparison only)

Rule: cancel unless `reference − acc_W > Θ·σ`, Θ = 2, σ = sqrt(p(1−p)/W).
Approximates MD3 (Sethi & Kantardzic 2017); our σ comes from the reference
estimate over the W rows rather than a training-set distribution, and balanced
accuracy is not a binomial proportion, so σ is approximate on multi-class
streams. **No verdict is attached.**

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| Training ratios 0.54, 0.26, 2.11, 0.92, 0.48 (Elec2, Covertype, abrupt, gradual, incremental) | `md3_compare.json` | ratio to selector, W = 200 | [D] |
| Accuracy deltas +1.76, +7.35, +6.90, +5.47, −1.79 | `md3_compare.json` | whole-stream prequential, balanced | [D] |
| Cheaper than WaitAndCheck on every stream; significant after Holm at W = 500 on Covertype (p = 0.049) and Elec2 (p = 0.023) | `md3_compare.json` | Wilcoxon on training ops, n = 12 per stream, Holm across 5 | [D] |
| Grids: `results/grid_md3_200/`, `results/grid_md3_500/` | run logs | 100 runs each, ADWIN | [D] |

## Numbers quoted elsewhere

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| Candidate A (DEFER-perfect) −52.2% / −44.7% training ops | `phase10_per_stream.json` | Covertype / Elec2, oracle foresight (upper bound) | [D] |
| Candidate A accuracy +3.30 / +2.72 points plain; +2.80 / +2.55 balanced | `phase10_candidate_a_balanced.json` | post-alarm horizon rows, pooled over families | [D] |

---

## Revision of 4 October — statistic, new tables and figures

**One statistic throughout for WaitAndCheck and ConfirmMD3.** Table~III and
Section V-E now use the **median over the twelve (family x seed) blocks** of a
stream, which is the statistic the pre-declared training rule uses. The earlier
Table III used a ratio of means over the same blocks, and the earlier Section V-E
quoted the median for training but the *mean* for accuracy. The two differ
because per-block ratios are right-skewed: a cell where WaitAndCheck trains much
more (INSECTS-abrupt/gnb, 4.12x) pulls the mean up but not the median, and on
covtype the mean accuracy gain (+1.98) exceeds the median (+0.55) because two
families gain heavily and two barely move. The pre-declared verdict is unchanged:
it was decided on its own stated statistics, which are quoted in the verdict
paragraph with that fact noted.

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| Table I setup values | `src/runner.py`, `src/selector.py`, `src/mechanisms.py` | configuration, fixed before the grid | - |
| Run times: 78 min ADWIN grid, 52+12 min DDM, 8.5+9.0 min WaitAndCheck, 7.7+7.8 min MD3 | `results/*/run.log` | wall clock, 12-core AMD, Windows 11 | [D] |
| Table III (main results): median balanced accuracy and median training work units, 5 streams x 7 policies | `results/grid/`, `grid_wait200/`, `grid_md3_200/` | median over 12 blocks per stream | [D] |
| Table IV (related work) | the cited papers only; "?" where we could not verify | - | - |
| Fig. 6 nudge failure: no-prediction-change share 11.8 / 60.2 / 77.2 / 40.1%, median gain +2.52 / 0.00 / 0.00 / 0.00 pp (xgb / rf / gnb / sgd), n = 212 / 1,876 / 364 / 1,816 | `phase7.json` -> `event_split` | per nudge, AlwaysNudge, ADWIN | [D] |
| Fig. 7 cost-accuracy scatter | same as Table III | median over 12 blocks | [D] |
| Per-prediction inference 201 (SGD) to 4,009 (XGBoost under AlwaysNudge) | `phase9_inference_cost.json` + `grid.csv` | mean per run, own stream length | [D] |
| WaitAndCheck medians: training 0.58 / 0.49 / 3.23 / 1.19 / 0.71; accuracy +1.72 / +0.55 / +8.34 / +6.04 / -0.66 | `results/grid_wait200/` | median over 12 blocks | [D] |
| ConfirmMD3 medians: training 0.51 / 0.43 / 2.76 / 0.92 / 0.43; accuracy +2.76 / +9.24 / +5.59 / +5.19 / -2.44 | `results/grid_md3_200/` | median over 12 blocks | [D] |
| INSECTS-incremental -1.07 pp (rule's mean) vs -0.66 pp (median) | `phase14_waitandcheck.json`, `paper/numbers.json` | both stated in the text | **[V]** input |

## Claims removed in this revision

1. **"Inference, not training, is 81-100% of operations"** - removed from the
   abstract, contribution 5, the introduction, Section V-D, the discussion and
   the conclusion. Training work units and inference operations are not
   commensurable, so a share of their sum is not a valid quantity. Figure 4 now
   plots the two on separate axes in their own units.
2. **"Mean 779 operations per prediction (range 9-8,046)"** - removed as a pooled
   cross-family figure; per-family values are given instead.
3. **"Break-even near 300 predictions"** and **"median net cost is negative on
   three of four cells"** - removed. Both depended on trading training work units
   against inference operations at a 1:1 rate, which the units do not support.
4. **"A No-Change classifier is hard to beat"** as a description of
   Bifet 2017 - replaced with the verified wording about a No-Change *Detector*
   signalling every 60 instances.
5. **"(in-sample reference values)"** attached to the always-SKIP numbers -
   removed; always-SKIP is a fixed rule, not a fitted one.
6. The unqualified **"adapt less"** advice - now scoped to Elec2 and Covertype,
   with the INSECTS gradual and incremental counter-examples (+20.3 and +36.2
   points for adaptation) stated in the same paragraph.

---

## Rule 1 and Rule 3 statistics, stated openly

**Rule 1** was pre-declared as "training operations <= 0.75x the selector's on >= 4
of 5 datasets (median over family x seed cells)". The rule named the statistic but
not what to do with a cell where the selector performs no training at all, which
makes the ratio undefined. That happens in exactly three blocks, all
INSECTS-incremental SGD (seeds 0, 1 and 2), where the selector takes no
adaptation and so trains nothing.

| stream | undefined ranked highest (decision-time computation) | undefined dropped |
|---|---|---|
| Elec2 | 0.58 | 0.58 |
| Covertype | 0.49 | 0.49 |
| INSECTS abrupt | 3.23 | 3.23 |
| INSECTS gradual | 1.19 | 1.19 |
| INSECTS incremental | **0.87** | **0.71** |
| streams passing <= 0.75x | **2 of 5** | **3 of 5** |

The 0.867 in the earlier Phase 14 report and the 0.71 in the first draft of this
paper are these two computations. **The pre-declared one is the first**: it is the
computation that produced the numbers the verdict was taken on. The paper, Table
III and Fig. 5 now all use it, and the caption names it. Rule 1 fails under both,
since neither reaches 4 of 5.

**Rule 3** did not specify an accuracy statistic. INSECTS-incremental is -1.07
points on the mean over blocks and -0.66 on the median. Rule 3 therefore fails
under the mean and passes under the median. We report both and do not pick the
favourable one. The verdict is NO regardless, because Rule 1 fails under both
treatments.

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| Training ratios 0.58 / 0.49 / 3.23 / 1.19 / 0.87 (WaitAndCheck, W=200) | `results/grid_wait200/`, `results/grid/` | median over 12 blocks, undefined ranked highest | **[V]** input |
| Alternative 0.71 for INSECTS-incremental | same | median over 12 blocks, undefined dropped | [D] |
| ConfirmMD3 ratios 0.51 / 0.43 / 2.76 / 0.92 / 0.60 | `results/grid_md3_200/` | same statistic as above | [D] |
| Rule 3: -1.07 (mean) vs -0.66 (median) | `phase14_waitandcheck.json`, `paper/numbers.json` | both stated in the paper | **[V]** input |

## Item 1 - oracle on the three INSECTS streams [DIAGNOSTIC]

Run 4 October. The Phase 10 oracle unchanged (same branches, horizon "next",
same lambda grid), ADWIN, seed 0, four families per stream. Sources:
`results/analysis/phase10_oracle_insects.json`,
`results/analysis/phase10_per_stream_insects.json`,
`results/analysis/phase10_always_skip_per_stream.json`. Alarm counts reproduce
the frozen grid exactly (43 / 25 / 16 alarms over the four families).

| Claim | Source | Metric / scope | Label |
|---|---|---|---|
| G1 range 9.5-17.7% (INSECTS-abrupt) | `phase10_per_stream_insects.json` | share of selector cost, over the whole lambda grid, pooled over 4 families, 43 alarms | [D] |
| G1 range 11.5-25.2% (INSECTS-gradual) | same | same, 25 alarms | [D] |
| G1 range 1.6-15.5% (INSECTS-incremental) | same | same, 16 alarms | [D] |
| Branch shares at lambda=1e3, abrupt: SKIP 70.3%, NUDGE 2.0%, REBUILD 27.7% | same | share of G1, DEFER 0.0% | [D] |
| Branch shares at lambda=1e3, gradual: SKIP 52.0%, NUDGE 8.3%, REBUILD 39.7% | same | same | [D] |
| Branch shares at lambda=1e3, incremental: SKIP 98.9%, NUDGE 0.0%, REBUILD 1.1% | same | same | [D] |
| Always-SKIP capture, abrupt: 18.6% / -47.3% / -122.0% | `phase10_always_skip_per_stream.json` | (cost(selector)-cost(always SKIP))/G1 at lambda=1e2/1e3/1e4 | [D] |
| Always-SKIP capture, gradual: 58.9% / -8.4% / -276.6% | same | same | [D] |
| Always-SKIP capture, incremental: 86.7% / 64.4% / -938.5% | same | same | [D] |
| Per-stream always-SKIP capture, Elec2: 91.0% / 57.4% / -22.8% | same | same, recomputed for comparability | [D] |
| Per-stream always-SKIP capture, Covertype: 96.2% / 83.0% / 3.4% | same | same | [D] |

The capture formula reproduces the published pooled Elec2+Covertype values
95.3% / 78.1% / -3.6% exactly, so the INSECTS figures are computed identically.

**Pre-declared description rule.** Fixed before the run: SKIP at or above 60% of
the advantage on all three INSECTS streams would license "skipping dominates on
all five streams". It is 70.3% (abrupt) and 98.9% (incremental) but 52.0%
(gradual), so the licensed wording is the second one: **skipping dominates on
Elec2 and Covertype but not on INSECTS-gradual**, where REBUILD carries 39.7%.

**Caveat carried with every number above.** 43, 25 and 16 alarms respectively,
against 2,595 for Elec2 and Covertype. The original Phase 10 subset excluded
these streams for that reason, and excluding them was declared in advance. These
are diagnostics on thin data, not a revision of the Phase 10 verdict.

---

## Item 2 - re-weighting training against inference in the oracle [DIAGNOSTIC]

Run 4 October, from the stored per-branch records; nothing was re-executed.
J_w(a) = w * train_ops + eval_ops + infer_per_row * H + lambda * errors, horizon
"next", w in {1, 10, 100}, all five streams. `eval_ops` is inference units times
holdout rows, so it is inference work and is not weighted; only `train` is.
Source: `experiments/phase10_reweight.py`,
`results/analysis/phase10_reweight.json`. w = 1 reproduces
`phase10_per_stream.py` exactly on all five streams and is kept as a self-check.

SKIP share of the oracle's advantage at lambda = 1e3:

| stream | w=1 | w=10 | w=100 |
|---|---|---|---|
| Elec2 | 78.9% | 87.1% | 96.2% |
| Covertype | 80.8% | 85.5% | 89.5% |
| INSECTS-abrupt | 70.3% | 87.5% | 99.0% |
| INSECTS-gradual | 52.0% | 83.0% | 99.4% |
| INSECTS-incremental | 98.9% | 99.7% | 100.0% |

G1 as a share of the selector's J, range over the whole lambda grid:

| stream | w=1 | w=10 | w=100 |
|---|---|---|---|
| Elec2 | 16.8-25.5% | 18.3-59.5% | 18.5-92.9% |
| Covertype | 24.8-37.1% | 36.5-73.0% | 37.7-96.3% |
| INSECTS-abrupt | 9.5-17.7% | 10.9-27.0% | 13.7-77.0% |
| INSECTS-gradual | 11.5-25.2% | 11.4-36.8% | 13.0-81.0% |
| INSECTS-incremental | 1.6-15.5% | 1.5-33.6% | 1.5-80.0% |

**Does the main finding change?** No. The SKIP share rises monotonically with w
on every stream: at w = 10 all five are at or above 83%, at w = 100 all are at or
above 89.5%. The INSECTS-gradual exception recorded under Item 1 holds only at
the 1:1 weighting and disappears by w = 10.

**The test is one-sided, and that limits what it licenses.** Raising w makes
training more expensive, which mechanically favours SKIP, the only zero-training
action. So this shows the SKIP finding is not an artifact of under-weighting
training; it does not show the finding survives the opposite error. The weighting
that could falsify it is w < 1, which was not in the declared set and has not
been run.

---

## Style pass

The prose was rewritten for readability on 4 October. No number, claim, scope or
citation was changed in that pass. The sentences where we were unsure whether the
rewrite shifted meaning are listed in the hand-off notes and were each checked
against the source number before the rewrite was kept.

---

## Summary

- **Verdict claims [V]: 3** (the WaitAndCheck verdict and its two rule inputs).
- **Diagnostic claims [D]: 58.**
- Configuration statements with no claim attached: 3.

## Withdrawn claims that must not appear

These were measured, found wrong, and withdrawn. They appear nowhere in the paper:
the raw "45% of alarms on healthy models" framing; the DDM "above reference"
claim; "~56k operations per prediction"; "compact rebuild" as a method (it is a
window-size artifact, reported as a diagnostic only); and "false alarms carry
most of the falling-error share".
