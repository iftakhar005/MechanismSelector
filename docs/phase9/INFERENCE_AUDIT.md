# Inference-cost audit (review of Phase 9.5)

**Diagnostic.** Answers the four review questions on "AlwaysRebuild costs less in total than
NeverAdapt (2.75e9 vs 2.81e9) because its inference is cheaper (2.33e9 vs 2.81e9)".
Scripts: `experiments/inference_audit.py` (committed results only) and
`experiments/window_size_control.py` (new, stationary synthetic data). Outputs:
`results/analysis/inference_audit.json`, `results/analysis/window_size_control.json`.

## 1. What one "operation" is, and the ~56k per prediction

| Family | Counted per predicted row (`src/footprint.py`) |
|---|---|
| XGBoost | tree nodes visited (split comparisons + the leaf) summed over **every tree**, i.e. rounds × classes for multi-class |
| Random Forest | tree nodes visited, from `decision_path`, summed over the 100 trees |
| SGD, GaussianNB | every learned parameter read (coefficients + intercepts; θ, σ², priors) |

**The ~56k figure is an artifact of the division, not a per-prediction cost.** The pooled
2.81e9 is a **sum over 60 runs** (5 seeds for rf/sgd, seed 0 for the deterministic xgb/gnb, over
5 datasets). It reproduces exactly (2,806,977,006.96). Divided by the mean stream length it gives
55,828, which mixes a 60-run total with a one-run length. The plausible figures are:

- **4.68e7 operations per run**, and
- **930 operations per prediction** (total ops / total rows),
- within the range of individual cells: 9–763 for sgd/gnb, 569–3,970 for the tree families.

## 2. Initial vs rebuilt models, per family × dataset

Operations per prediction; "rebuilt" is the model AlwaysRebuild ends with (measured, not
interpolated). Visits per tree is a mean-path-length proxy (= mean depth + 1); XGBoost is divided
by trees, not rounds.

| model | dataset | initial training rows | initial | rebuilt | ratio | visits/tree initial → rebuilt |
|---|---|---|---|---|---|---|
| rf | covtype | 8,000 | 1,568 | 646 | 0.41 | 15.7 → 6.5 |
| rf | insects_abrupt | 4,227 | 1,307 | 994 | 0.76 | 13.1 → 9.9 |
| rf | insects_incremental | 4,560 | 1,245 | 1,012 | 0.81 | 12.5 → 10.1 |
| rf | elec2 | 3,624 | 817 | 672 | 0.82 | 8.2 → 6.7 |
| rf | insects_gradual | 1,932 | 871 | 878 | 1.01 | 8.7 → 8.8 |
| xgb | covtype | 8,000 | 3,970 | 1,288 | 0.32 | 5.7 → 1.8 |
| xgb | insects_abrupt | 4,227 | 3,673 | 2,357 | 0.64 | 6.1 → 3.9 |
| xgb | insects_incremental | 4,560 | 3,854 | 2,714 | 0.70 | 6.4 → 4.5 |
| xgb | elec2 | 3,624 | 569 | 446 | 0.78 | 5.7 → 4.5 |
| xgb | insects_gradual | 1,932 | 1,833 | 1,732 | 0.94 | 3.1 → 2.9 |
| sgd, gnb | all five | — | — | — | **1.00** | fixed cost |

Rebuilt tree models are shallower, and node counts fall with them (rf covtype 287k → 9.3k
nodes; xgb covtype 34k → 3.8k). The ratio tracks the initial training size closely but not
exactly. Spearman ρ = −0.9 for both families; insects_abrupt and insects_incremental, whose
initial sizes differ by 7%, swap places. The smallest gap between initial set and 1,000-row
window (insects_gradual, 1,932 rows) shows almost no effect.

## 3. Combined cost per family — not pooled

Mean per cell, summed over the five datasets. Two estimates for AlwaysRebuild bracket the
truth: Phase 9.5's mean of the initial and final units ("mid", which overstates a rebuilt model
because the first rebuild comes early), and the final model's units for every row ("final", which
assumes all rebuilt models are like the last).

| family | NeverAdapt | AlwaysRebuild, mid | AlwaysRebuild, final | cells where rebuild is cheaper (mid / final) |
|---|---|---|---|---|
| xgb | 7.93e8 | 6.20e8 | 4.36e8 | 4 / 5 — 5 / 5 |
| rf | 3.19e8 | 2.74e8 | 2.16e8 | 3 / 5 — 4 / 5 |
| sgd | 5.96e7 | 1.29e8 | 1.29e8 | 0 / 5 |
| gnb | 1.19e8 | 1.19e8 | 1.19e8 | 0 / 5 |

The pooled 2.75e9 vs 2.81e9 (a 2% gap) is two opposite effects netted: tree families save on
inference, sgd pays rebuild training for no inference change, and the 5-seed weighting of rf and
sgd sets the balance. **A pooled row must not be reported alone.** Training work units and
inference units are different quantities, so the combined column carries that caveat.

## 4. Window size or class mix? A controlled test

On the real streams, a rebuild window is both *smaller* and, under drift, often *narrower in
classes*. `window_size_control.py` separates the two on stationary synthetic data (5 seeds each,
the grid's own model factories and footprint measure):

| effect isolated | xgb | rf |
|---|---|---|
| size only: 1,000 vs 8,000 i.i.d. rows, SEA | 0.94 | 0.63 |
| size only: 1,000 vs 8,000 i.i.d. rows, Agrawal | 0.70 | 0.73 |
| class mix only: 1,000 rows, 2 of 7 classes vs all 7 (RandomRBF) | 0.47 | 0.87 |

**Both mechanisms are real, and their weight differs by family.** For Random Forest, window
size alone gives most of the observed ratio. For multi-class XGBoost, a narrow class mix alone
halves the cost: trees for absent classes collapse to near-stumps. That is the likely main
contributor to covtype's 0.32, where 1,000-row windows contain few of five rare classes.

## Verdict for the findings page

**Holds for tree families, as a diagnostic finding:** rebuilds on small windows produce
cheaper-to-serve models, through two separable mechanisms (fewer rows → shallower trees; fewer
classes → trivial per-class trees in XGBoost). Not a headline: it is absent for sgd/gnb, it rests
on final-model measurements for the rebuilt side (the confirm grid now measures the whole
trajectory exactly; see `docs/confirm/`), and xgb has one seed per cell. A rebuild being "cheaper"
can also mean a model that has forgotten rare classes, which is an accuracy question, not a
saving.
