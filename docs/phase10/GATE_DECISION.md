# Phase 10 finished, inference audit, and the A/B/C/D decision

Thresholds in the decision table are **our own choices, not standard values**. Results close to a
threshold are reported as "near" rather than forced to a side.

All numbers come from committed files: `results/analysis/phase10_oracle.json`,
`phase10_per_stream.json`, `inference_audit.json`, `phase9_inference_cost.json`,
`results/grid/grid.csv`.

---

## Surprises first

**1. A reporting error of mine, now corrected.** The "~56k ops per prediction" figure came from
dividing a 60-run pooled total by one run's row count. The correct mean is **779 ops per
prediction**, ranging from 9 (elec2/sgd) to 8,046 (a covtype/xgb run). Every per-prediction number
below divides each run by its own stream length.

**2. Deferring perfectly makes the selector more accurate, not just cheaper.** On both streams a
DEFER-perfect policy *gains* accuracy — covtype +3.30 pp, elec2 +2.72 pp — while cutting training
ops roughly in half. Adapting is not merely expensive on these streams, it is often harmful.

**3. 80% of the oracle's advantage comes from SKIP alone.** NUDGE contributes ~16%, REBUILD 3–5%.
The headroom is in not acting, not in choosing a better action.

**4. The compact-rebuild effect is a window-size artifact.** A model trained on 1,000 rows from the
*initial* era is as cheap to query as a rebuild from late in the stream (control/initial 0.57–0.88
against rebuilt/initial 0.28–1.42). Rebuilding is not what makes the model cheap; training on less
data is.

---

## 0. What each accuracy number means

Three different accuracies appear across these reports. They are all correct; they are not
interchangeable.

| name | rows scored | metric | grouping | where |
|---|---|---|---|---|
| **horizon accuracy** | only the rows after each alarm, up to the next alarm or 1,000 | plain 0/1 | pooled over families within a stream | Phase 10 tables |
| **whole-stream prequential** | every row after the initial slice | plain, or balanced on multi-class | per family, averaged over seeds | `grid.csv`, inference tables |
| **next-1,000 accuracy** | the 1,000 rows following that model's own training slice | the stream's own scorer | one fitted model, not a policy | inference audit |

Reconciliation of the two numbers that looked inconsistent, both for the selector on covtype:

| family | horizon accuracy (plain) | whole-stream plain | whole-stream balanced |
|---|---|---|---|
| xgb | 0.890 | 88.9% | 52.1% |
| rf | 0.875 | 87.5% | 69.9% |
| sgd | 0.629 | 46.4% | 34.8% |
| gnb | 0.771 | 78.3% | 51.7% |
| **pooled** | **0.780** | — | — |

So Phase 10's 0.780 is plain accuracy on post-alarm rows pooled over families, and e257208's 52.1%
is balanced accuracy over the whole stream for XGBoost alone. For xgb and rf the horizon and
whole-stream plain figures agree to within 0.1 points; for sgd they differ by 16 points, because
SGD scores far better in the first 1,000 rows after an adaptation than it does later. **No number
changes; the labels do.**

## 1. Phase 10, per stream (ADWIN, horizon "next", 2,595 alarms)

### covtype — 1,860 alarms

Accuracy below is **horizon accuracy**: plain 0/1 over post-alarm rows, pooled over families.

| λ | selector train | selector infer | selector acc | oracle train | oracle infer | oracle acc | G1 share | guard share of shortfall |
|---|---|---|---|---|---|---|---|---|
| 0 | 144,928,665 | 633,780,789 | 0.780 | 2,199,600 | 554,703,436 | 0.754 | 28.5% | 41.3% |
| 1e2 | 144,928,665 | 633,780,789 | 0.780 | 2,304,188 | 555,247,456 | 0.767 | 27.7% | 41.9% |
| 1e3 | 144,928,665 | 633,780,789 | 0.780 | 6,339,753 | 563,008,319 | 0.813 | 24.8% | 45.8% |
| 1e4 | 144,928,665 | 633,780,789 | 0.780 | 56,852,902 | 629,915,629 | 0.858 | 27.9% | 46.3% |
| 1e5 | 144,928,665 | 633,780,789 | 0.780 | 103,961,056 | 646,763,363 | 0.862 | 35.7% | 40.4% |
| 1e6 | 144,928,665 | 633,780,789 | 0.780 | 118,646,966 | 647,970,712 | 0.862 | 37.1% | 39.5% |

### elec2 — 735 alarms

| λ | selector train | selector infer | selector acc | oracle train | oracle infer | oracle acc | G1 share | guard share of shortfall |
|---|---|---|---|---|---|---|---|---|
| 0 | 16,972,304 | 157,192,957 | 0.676 | 1,241,200 | 128,525,930 | 0.657 | 25.5% | 5.4% |
| 1e2 | 16,972,304 | 157,192,957 | 0.676 | 1,405,589 | 128,721,441 | 0.691 | 23.8% | 5.6% |
| 1e3 | 16,972,304 | 157,192,957 | 0.676 | 3,156,747 | 130,469,226 | 0.713 | 18.2% | 10.0% |
| 1e4 | 16,972,304 | 157,192,957 | 0.676 | 12,184,380 | 149,986,002 | 0.735 | 16.8% | 19.7% |
| 1e5 | 16,972,304 | 157,192,957 | 0.676 | 15,795,944 | 160,768,337 | 0.737 | 18.4% | 20.1% |
| 1e6 | 16,972,304 | 157,192,957 | 0.676 | 16,175,204 | 162,574,232 | 0.737 | 18.6% | 20.1% |

Combined ops are the sum of the two op columns; inference is 81–90% of the selector's combined
total on both streams.

### Where the oracle's advantage comes from (λ = 1e3)

| branch | covtype | elec2 |
|---|---|---|
| SKIP | 80.8% | 78.9% |
| NUDGE | 15.9% | 16.3% |
| REBUILD | 3.3% | 4.7% |
| DEFER | 0.0% | 0.0% |

DEFER is 0.0% **by construction, not by measurement**: in a per-alarm frame it pays no training and
carries the current model's errors, so it is identical to SKIP and can never be strictly better.
Its real value — the alarm staying pending while the buffer fills — needs a trajectory-level
experiment this design cannot express.

### G1 against the 5% threshold

| stream | G1 range over λ | verdict |
|---|---|---|
| covtype | 24.8–37.1% | far above 5% |
| elec2 | 16.8–25.5% | far above 5% |

---

## 2. Inference cost

### What one unit is, per family (from `src/footprint.py`)

| family | unit | counted as |
|---|---|---|
| XGBoost | tree node visits | the leaf each row reaches plus that leaf's depth, summed over every boosting round |
| RandomForest | tree node visits | `decision_path` length per row summed over trees: every split comparison plus one leaf per tree |
| SGDClassifier | parameter reads | `n_classes × n_features + n_classes`, read once per prediction |
| GaussianNB | parameter reads | `2 × n_classes × n_features + n_classes` |

Comparable **within** a family only.

### Plausibility of the magnitudes

They follow from the definitions. elec2/sgd = 2 × 8 + 2 = 9 ✓. covtype/gnb = 2 × 7 × 54 + 7 = 763 ✓.
covtype/xgb = 100 rounds × 41.3 mean path = 4,133 ✓ against a measured 34,130 total nodes.
covtype/rf = 100 trees × 14.5 = 1,448 ✓ against 285,840 total nodes. Nothing approaches 56,000.

### Initial vs rebuilt vs control (common 1,000-row probe, seed 0)

`acc next 1,000` is that model's accuracy on the 1,000 rows following its own training slice, in
the stream's own scorer (balanced on multi-class) — a property of one fitted model, not of a policy.

| cell | model | rows trained | ops/pred | size | nodes | path/tree | acc next 1,000 |
|---|---|---|---|---|---|---|---|
| elec2/xgb | initial | 3,624 | 573 | 100 | 4,628 | 5.73 | 0.800 |
| | rebuilt | 1,000 | 420 | 100 | 2,278 | 4.20 | 0.815 |
| | control | 1,000 | 402 | 100 | — | — | — |
| elec2/rf | initial | 3,624 | 789 | 100 | 66,150 | 7.89 | 0.779 |
| | rebuilt | 1,000 | 669 | 100 | 16,370 | 6.69 | 0.839 |
| | control | 1,000 | 605 | 100 | — | — | — |
| covtype/xgb | initial | 8,000 | 4,133 | 100 | 34,130 | 41.33 | 0.587 |
| | rebuilt | 1,000 | 1,174 | 100 | 3,272 | 11.74 | 0.694 |
| | control | 1,000 | 2,358 | 100 | — | — | — |
| covtype/rf | initial | 8,000 | 1,448 | 100 | 285,840 | 14.48 | 0.604 |
| | rebuilt | 1,000 | 583 | 100 | 9,284 | 5.83 | 0.602 |
| | control | 1,000 | 1,059 | 100 | — | — | — |
| insects_abrupt/xgb | initial | 4,227 | 3,728 | 100 | 29,416 | 37.28 | 0.730 |
| | rebuilt | 1,000 | 1,598 | 100 | 5,432 | 15.98 | 0.701 |
| | control | 1,000 | 2,604 | 100 | — | — | — |
| insects_abrupt/rf | initial | 4,227 | 1,309 | 100 | 122,898 | 13.09 | 0.720 |
| | rebuilt | 1,000 | 836 | 100 | 13,988 | 8.36 | 0.705 |
| | control | 1,000 | 1,005 | 100 | — | — | — |
| insects_gradual/xgb | initial | 1,932 | 1,888 | 100 | 8,232 | 18.88 | 0.613 |
| | rebuilt | 1,000 | 2,676 | 100 | 14,776 | 26.76 | 0.550 |
| | control | 1,000 | 1,577 | 100 | — | — | — |
| insects_gradual/rf | initial | 1,932 | 1,024 | 100 | 26,588 | 10.24 | 0.575 |
| | rebuilt | 1,000 | 973 | 100 | 39,764 | 9.73 | 0.569 |
| | control | 1,000 | 899 | 100 | — | — | — |
| insects_incremental/xgb | initial | 4,560 | 3,934 | 100 | 36,642 | 39.34 | 0.607 |
| | rebuilt | 1,000 | 2,788 | 100 | 14,122 | 27.88 | 0.680 |
| | control | 1,000 | 2,838 | 100 | — | — | — |
| insects_incremental/rf | initial | 4,560 | 1,346 | 100 | 190,346 | 13.46 | 0.590 |
| | rebuilt | 1,000 | 975 | 100 | 38,308 | 9.75 | 0.700 |
| | control | 1,000 | 1,003 | 100 | — | — | — |

Policy-level accuracy beside per-prediction cost, per cell, is in the chat report and in
`results/analysis/phase9_inference_cost.json` with `results/grid/grid.csv`.

---

## 3. The decision table

### A. Wait-and-check — **YES (on the intended reading; see the caveat)**

Rule: on both streams, perfect DEFER cuts the selector's training ops by ≥25% with accuracy within
1 pp.

| stream | training ops saved | selector acc | DEFER-perfect acc | Δ | deferred at |
|---|---|---|---|---|---|
| covtype | **52.2%** | 0.7799 | 0.8129 | **+3.30 pp** | 1,103 / 1,860 |
| elec2 | **44.7%** | 0.6765 | 0.7036 | **+2.72 pp** | 440 / 735 |

Both streams clear 25% by a wide margin. On accuracy, "within 1 pp" was confirmed to mean "no more
than 1 pp worse", so the condition is met with room to spare: the difference is +2.7 to +3.3 pp in
the policy's favour. **A = YES, unambiguously.**

**Caveat that limits the strength of this yes:** "perfectly" means oracle foresight about whether
deferring will cost errors, so these are upper bounds; and because DEFER coincides with SKIP in
this frame, what is measured is really "skip when skipping is free", not a deferral that re-decides
later.

### B. Compact rebuild — **NO**

Rule: in both tree families, on ≥3 of 5 datasets, rebuilt ≤0.8× initial ops **and** policy accuracy
not below NeverAdapt's; must survive the window-size control.

- Ratio alone: XGBoost 4 of 5 (0.73, 0.43, 0.71, 0.28; insects_gradual 1.42 fails), RandomForest 3
  of 5 (0.64, 0.72, 0.40; elec2 0.85 and insects_gradual 0.95 fail). This part passes.
- With the accuracy condition: AlwaysRebuild is below NeverAdapt on covtype by 15.7 points (xgb)
  and 17.9 (rf), and on insects_abrupt/xgb by 0.8. Cells passing **both** conditions: XGBoost 2 of
  5 (elec2, insects_incremental), RandomForest 2 of 5 (insects_abrupt, insects_incremental).
  Below the ≥3 requirement in both families.
- The control settles it anyway: a model trained on 1,000 rows from the initial era costs
  0.57–0.88× the initial model, and on elec2 and insects_incremental it is *cheaper* than the
  rebuild. The gap is window size, which the rule says makes the answer no.

### C. Guard fix — **NO (covtype near)**

Rule: guard-forced rebuilds ≥50% of the selector's shortfall, on both streams.

| stream | guard share of shortfall (λ = 0 … 1e6) | verdict |
|---|---|---|
| covtype | 39.5–46.3% | **near**, peaks at 46.3% (λ = 1e4) |
| elec2 | 5.4–20.1% | clearly below |

Fails on both the "both streams" requirement and, strictly, on covtype as well — though covtype is
close enough that it should be called near rather than a clean no.

### D. None — **does not apply**, since A passes.

---

## What this supports

The measurement study stands, and one intervention is justified by the numbers: **wait-and-check**.
Its support is "skip more, and skip when skipping is free", which is also where 80% of the oracle's
advantage sits. Nothing here justifies compact rebuilds, and the guard — the thing Finding 1 is
about — turns out to explain under half the shortfall on covtype and very little on elec2. Finding
1 remains true as a description of the policy's behaviour; it is not the main source of its cost.

Nothing in A, B or C has been implemented.
