# Phase 9 — cheap diagnostics on existing logs

All five sub-phases are **diagnostics**: they explain existing results, introduce no new
design, and write only new files. The pre-registered grid is untouched — the checksum over
every CSV in `results/grid`, `results/grid_ddm` and `results/grid_direction` is
`a8ba3cebeecff2b61727d4145807907ce5fd87aaaa09992a1bc438afe1e897bb` before and after.

Every replay reproduced its logged alarm count exactly (e.g. elec2/xgb: 79 under ADWIN, 128
under DDM), so the diagnostics are measuring the same runs the grid measured.

---

## Falsifications first

**1. The DDM half of Finding 2 does not survive a significance test.** Our Phase 7b claim was
that a one-sided detector also fires on models at or above their reference, at 24–33%. Once the
comparison is a test rather than a raw `>=`, the DDM rate falls to **8.6%**, and to **3.5%**
with the autocorrelation correction — at or below the 5% false-positive rate the test itself
produces. Under DDM there is essentially no evidence that alarms fire on genuinely improved
models. The sentence "a one-sided detector does it too" must be withdrawn.

**2. The raw "above reference" rate is uninformative on its own.** Under a null where nothing
changes, comparing two noisy estimates gives **50.6%** at or above reference. Our observed raw
ADWIN rate is **40.2%** — *below* chance. Quoting 45% as if it were high was wrong; the number
carries information only once it is tested.

**3. What survives is the ADWIN result, and it is stronger than before.** 38.1% of ADWIN alarms
fire while the model is **significantly** above its reference (36.5% with the n_eff correction),
against the 5% such a test yields by construction. That is a real effect, roughly sevenfold.

**4. Counting inference reverses "never adapting is cheapest."** In all 10 tree-model cells the
cheapest policy changes once per-prediction cost is included, because a model rebuilt on a
1,000-row window is shallower than the initial model trained on 8,000 rows.

---

## 9.1 Is "alarm above reference" just noise?

`results/analysis/phase9_reference_noise.json` · `experiments/phase9_reference_noise.py`

120 NeverAdapt replays (5 streams × 4 models × seeds, both detectors). Rescored without
retraining. `n` is smaller than the alarm count because alarms whose reference slice never
reached 200 rows are excluded rather than guessed.

| measure | ADWIN | DDM |
|---|---|---|
| as logged (scorer, `>=`, 200-row reference) | 2425/4680 = **51.8%** | 1784/4946 = **36.1%** |
| plain accuracy, 200-row reference | 1010/2515 = 40.2% | 378/2019 = 18.7% |
| plain accuracy, 500-row reference | 345/802 = 43.0% | 28/270 = 10.4% |
| plain accuracy, 1000-row reference | 165/397 = 41.6% | 14/169 = 8.3% |
| **significantly above (α = 0.05)** | 959/2515 = **38.1%** | 173/2019 = **8.6%** |
| significantly above, n_eff corrected | 917/2515 = 36.5% | 70/2019 = **3.5%** |
| **null: expected under no change** | **50.6%** | 50.7% |

Per stream (plain / significant / significant+n_eff / null):

| detector | stream | raw | signif | sig+n_eff | null | n |
|---|---|---|---|---|---|---|
| adwin | covtype | 0.391 | 0.375 | 0.362 | 0.505 | 1612 |
| adwin | elec2 | 0.411 | 0.382 | 0.355 | 0.505 | 722 |
| adwin | insects_abrupt | 0.505 | 0.475 | 0.475 | 0.519 | 101 |
| adwin | insects_gradual | 0.441 | 0.441 | 0.441 | 0.517 | 59 |
| adwin | insects_incremental | 0.238 | 0.190 | 0.190 | 0.524 | 21 |
| ddm | covtype | 0.196 | 0.104 | 0.037 | 0.506 | 1392 |
| ddm | elec2 | 0.160 | 0.041 | 0.023 | 0.507 | 563 |
| ddm | insects_abrupt | 0.400 | 0.200 | 0.200 | 0.521 | 20 |
| ddm | insects_gradual | 0.233 | 0.033 | 0.033 | 0.520 | 30 |
| ddm | insects_incremental | 0.000 | 0.000 | 0.000 | 0.524 | 14 |

Enlarging the reference from 200 to 1,000 rows barely moves ADWIN (40.2% → 41.6%) but halves
DDM (18.7% → 8.3%), which is what noise-driven numbers do.

Two bookkeeping notes. The logged rate uses `>=` and the runner's scorer (balanced accuracy on
multi-class streams); the strictly-above count we quoted previously (2101/4680 = 44.9%) differs
from 51.8% only by 324 exact ties. The null is computed on plain accuracy, since balanced
accuracy is not a binomial proportion — both observed forms are given so the comparison is like
for like.

**Verdict: partly explains it.** It removes the DDM claim entirely and removes the raw-rate
framing, but the tested ADWIN effect stands at roughly seven times the test's own error rate.

## 9.2 What is DDM comparing against?

`results/analysis/phase9_ddm_internals.json` · `experiments/phase9_ddm_internals.py`

Verified from the installed source, `river/drift/binary/ddm.py` (river 0.25.0), quoted:

- `update()` begins `if self.drift_detected: self._reset()` (lines 122–124) — **the reset is
  lazy**, it happens on the update *after* the signal.
- `_reset()` clears `_p = stats.Mean()`, `_p_min`, `_s_min`, `_ps_min = inf` (lines 109–120).
- `p_i = self._p.get()` is a **cumulative mean over all samples since the reset**, not a window;
  `s_i = sqrt(p_i (1 - p_i) / n)` with `n = self._p.n` (lines 129–132).
- Statistics are evaluated only when `n > warm_start` = 30 (line 134). **There is no other
  lockout**: DDM can re-arm 31 rows after a reset.
- Drift fires when `p_i + s_i > p_min + 3 * s_min` (lines 145–146).

Measured over 20 cells (NeverAdapt + DDM):

| quantity | value |
|---|---|
| observed alarms vs stationary-null alarms | 1,515 vs 5.0 (**303×**) |
| cells where the null produced zero alarms | 11 of 20 |
| median share of alarms within 100 rows of a reset | 20% |
| **median share where `p_min` is below our reference error** | **79%** |
| median gap, our reference error − `p_min` | 14.7 points |
| median `p_i − p_min` at the alarm | 5.6 points |

**Verdict: explains the apparent contradiction.** DDM compares the cumulative error rate since
its own last reset against the best such rate it has seen; our policy compares against accuracy
measured on the 200 rows after the last adaptation. In 79% of alarms DDM's baseline is 14.7
points stricter than ours, so "DDM fired while the model was above our reference" describes two
different baselines, not a misbehaving detector. The stationary null rules out noise: on i.i.d.
data at the same error rate, DDM is silent.

## 9.3 Do alarms line up with real drift?

`results/analysis/phase9_change_points.json` · `experiments/phase9_change_points.py`

Ground truth verified at source — Souza, Reis, Maletzke & Batista (2020), *Challenges in
Benchmarking Stream Learning Algorithms with Real-world Data*, DMKD 34:1805–1858,
arXiv:2005.00113, **Table 2, p. 37**, quoted verbatim:

    Abrupt (bal.)   52,848   14352; 19500; 33240; 38682; 39510

The instance count matches `results/dataset_manifest.json` (52,848), confirming the stream we
load is the one those change points refer to.

Tolerance 1,000 rows, NeverAdapt: **27.4% of ADWIN alarms** and **10.3% of DDM alarms** follow a
real change point. At tolerance 2,000 the ADWIN share rises but stays a minority.

The cross-tabulation carries a warning for our own framing: alarms classified as *true
detections* are **more** likely to read "above reference" (87.1%) than false alarms (47.6%). The
reason is that the reference re-anchors after every alarm, so during a sustained post-drift
degradation the model is repeatedly "above" a reference that has already fallen. **Above
reference means better than recently, not healthy.** Sample size is small — 113 alarms over the
insects_abrupt cells — so this is indicative, not conclusive.

**Verdict: partly explains it.** A majority of alarms are not near any published change point,
and the moving reference makes "above reference" a weaker statement than we have been treating it
as.

## 9.4 Temporal dependence

`results/analysis/phase9_block_shuffle.json` · `experiments/phase9_block_shuffle.py`

Rows after the initial training slice permuted within blocks of 50; the initial model and the
overall order are unchanged, so only the error stream's local structure differs.

| detector | alarms before | after | retained |
|---|---|---|---|
| ADWIN | 1,369 | 822 | 60% |
| DDM | 1,472 | 455 | **31%** |

Lag-1 autocorrelation of the error stream falls from +0.58…+0.90 to +0.15…+0.49. Per cell, DDM
loses the most on covtype/gnb (397 → 40) and covtype/xgb (205 → 34).

**Verdict: explains a large share.** Two fifths of ADWIN's alarms and more than two thirds of
DDM's exist only because consecutive errors are correlated. Elec2 and Covertype are the streams
where this matters most, which is also why the persistence baseline in Phase 12.3 is worth
running.

## 9.5 Inference cost

`results/analysis/phase9_inference_cost.json` · `experiments/phase9_inference_cost.py`

Units per row are interpolated linearly in model size between the two measured endpoints in
`grid.csv` and integrated over the logged size trajectory. Exact at the endpoints and exact for
SGD/GaussianNB; an estimate in between. Training work units (row-passes) and inference units
(node visits / parameter reads) are different quantities — the combined column is reported
because the instructions ask for it, not as a claim that they are commensurable.

| policy | training | inference | inference share |
|---|---|---|---|
| AlwaysRebuild | 419,758,504 | 2,333,004,294 | 84.8% |
| MechanismSelector | 173,545,094 | 2,409,235,150 | 93.3% |
| FixedSchedule | 156,362,938 | 2,296,032,740 | 93.6% |
| AlwaysNudge | 3,731,918 | 2,933,693,174 | 99.9% |
| NeverAdapt | 0 | 2,806,977,007 | 100% |

The cheapest policy changes in **10 of 20 cells** — every XGBoost and Random Forest cell — always
away from NeverAdapt. Mean inference operations per run, XGBoost: NeverAdapt 1.59e8 vs
AlwaysRebuild 1.22e8. Those two numbers rest on measured endpoints, not on the interpolation,
because NeverAdapt's model never changes.

**Verdict: does not explain the alarms, but changes a headline.** "Never adapting is free" is only
true if inference is not counted.

## 9.6 Direction on the null, and the true/false split (added at review)

`results/analysis/phase9_direction_null.json` · `experiments/phase9_direction_null.py`

Asked before rewriting Finding 2: if ADWIN's two-sidedness makes spurious alarms land on falling
error about half the time, is the observed 42% just that?

**The i.i.d. null cannot answer it: ADWIN fired 0 alarms in all 20 cells.** At δ = 0.002 it does
not produce spurious alarms on independent noise, so a second null was built — a two-state Markov
chain with each cell's own error rate *and* its own lag-1 autocorrelation, and no drift anywhere.

| stream population | alarms | share on falling error |
|---|---|---|
| observed (NeverAdapt, ADWIN) | 1,369 | **0.416** |
| AR(1) null, no drift, same autocorrelation | 789 | **0.421** |
| i.i.d. null, same error rate | 0 | undefined |

The match is close to exact: a stream with no drift at all, carrying only the dependence the real
streams have, produces the same falling-error share. **The 42% figure is what an
autocorrelation-driven alarm population produces, and carries no evidence about improvements.**

The second half of the hypothesis is **not** supported. On insects_abrupt, splitting by whether an
alarm follows a published change point:

| tolerance | falling share, true detections | falling share, false alarms | share of falling alarms that are false |
|---|---|---|---|
| 500 | 0.931 (n = 29) | 0.298 (n = 84) | 0.481 |
| 1000 | 0.903 (n = 31) | 0.293 (n = 82) | 0.462 |
| 2000 | 0.879 (n = 33) | 0.287 (n = 80) | 0.442 |

Conditional on being a true detection an alarm is *more* likely to be falling-error, not less —
the opposite of the expected pattern. Because false alarms are three times as numerous, they still
carry about half the falling-error alarms in absolute terms. Why alarms near these change points
are overwhelmingly falling is unexplained; n is 29–33, and the insects temperature shifts may make
the new concept easier for an unchanged model. Reported, not explained.

**Verdict: supports the aggregate reframing, not the conditional one.** The defensible sentence is
that most ADWIN alarms on these benchmarks are not responses to published drift, and the
falling-error share is reproduced by a no-drift AR(1) null — not that falling-error alarms are the
spurious ones.

---

## What Phase 9 means for the four findings

- **Finding 1 (the guard)** — untouched.
- **Finding 2 (alarms above reference)** — must be rewritten. Drop DDM. Drop the raw rate.
  State the tested ADWIN result (38.1%, or 36.5% corrected, against a 5% test rate), and add the
  9.2 explanation: DDM's baseline is its own since-reset minimum, ours is the last 200 rows.
- **Finding 3 (per-family repair)** — untouched.
- **Finding 4 (inference cost)** — strengthened, and now quantified over the whole stream rather
  than at the endpoints.
- **New, from 9.4** — a large share of alarms on Elec2 and Covertype is manufactured by label
  autocorrelation. This is a property of the benchmarks, not of the policies.

## Timing estimate for Phase 10

Full-information replay branches four actions per alarm and scores each over a horizon. Cost per
alarm is dominated by one rebuild (~100 trees on ≤1,000 rows) plus four prediction passes over
H ≤ 1,000 rows.

- Alarms to branch: ~4,680 (ADWIN) and ~4,946 (DDM) on the NeverAdapt trajectory; ~2,830 on the
  selector trajectory.
- Measured comparison: the Phase 9 nudge-window diagnostic did 1,225 alarms × 2 nudges in ~25
  minutes, and a rebuild costs roughly 25× a nudge on the tree families.
- **Estimate: 6–10 CPU-hours per trajectory per detector**, so 12–20 hours for the ADWIN
  trajectories alone, and roughly double that with DDM and the three horizon settings.

That is close to the 20-hour threshold in standing rule 7, so before running I will time 10
alarms per family and give a measured total. Say whether to (a) time it and proceed on ADWIN
only, (b) time it and run both detectors, or (c) restrict to the declared subset the
instructions allow — all of Elec2 plus one Insects stream.
