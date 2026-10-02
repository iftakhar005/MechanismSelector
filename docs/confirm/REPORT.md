# Confirm-before-acting — report

Pre-registration: `docs/confirm/PREREGISTRATION.md` (committed before any run, `9089762`).
Everything in **Pre-registered result** follows it exactly. **Post-hoc** sections are labelled
and were not planned.

Scripts: `experiments/confirm_grid.py`, `experiments/confirm_analysis.py` (pre-registered),
`experiments/confirm_decomposition.py` (post-hoc). Results: `results/confirm/`,
`results/confirm_delay_only/`. Environment: Python 3.11, river 0.26.1, scikit-learn 1.9.1,
xgboost 3.2.0, numpy 2.4.6 — not the study's pinned versions. All 265 tests pass on it.

## Surprises first

1. **The rule's effect on accuracy comes from *when* it adapts, not from *which* alarms it drops.**
   Waiting W rows before acting raises accuracy sharply on abrupt drift. At W = 500 the gain is
   +8 to +12 points for the tree models, but only because the replacement model then trains on
   newer data. Dismissing alarms, on top of the wait, gives some of that gain back (post-hoc
   control below).
2. **Acting immediately trains the replacement on mostly stale data.** For alarms that are true
   detections, ADWIN fires a median 223 rows after the change. The 1,000-row window then holds
   about **33% post-change rows**. Waiting 500 rows raises that to about 73–80%. On these streams,
   rebuilding on every alarm the moment it fires is barely better than never adapting for XGBoost
   (0.745 vs 0.697 mean accuracy; the selector 0.698).
3. **The false-alarm half of the design did not work.** Sticky sampling at ρ = 0.8 produced almost
   no alarms for the tree models on no-drift streams (0–0.2 per run). The pre-registered validity
   rule excludes 14 of 24 stream × model cells, so the "save on false alarms" question is mostly
   untested here. It needs either the real streams or a stronger dependence model.

## Pre-registered result

**Alarm counts per run at W = 0** (adaptive policies, mean over seeds):

| stream | gnb | rf | sgd | xgb |
|---|---|---|---|---|
| agrawal_abrupt_iid | 3.0 | 2.6 | 2.3 | 2.4 |
| agrawal_abrupt_sticky | 11.6 | 3.7 | 11.2 | 3.2 |
| agrawal_none_sticky | 4.3 | 0.0 | 4.0 | 0.0 |
| sea_abrupt_iid | 2.6 | 2.8 | 2.7 | 2.8 |
| sea_abrupt_sticky | 2.6 | 3.0 | 2.2 | 3.0 |
| sea_none_sticky | 0.4 | 0.0 | 0.6 | 0.2 |

The rule excludes every sticky cell below 5 alarms per run: all of `sea_*_sticky`,
`agrawal_none_sticky`, and the tree models on `agrawal_abrupt_sticky`. That leaves 15 pairs for
gnb/sgd and 10 for rf/xgb, instead of 30.

**Verdicts**, per the fixed criteria: training reduction ≥ 20% and accuracy loss ≤ 0.5 pp →
*helps*; post-change loss > 1 pp → *hurts*. Accuracy *loss* is negative when accuracy rose.

| policy | model | W | median training reduction | mean acc. loss (pp) | post-change loss (pp) | verdict |
|---|---|---|---|---|---|---|
| always_rebuild | gnb | 200 | 0% | −0.9 | −0.7 | neither |
| always_rebuild | gnb | 500 | 0% | −4.1 | −1.6 | neither |
| always_rebuild | rf | 200 | 33% | +2.8 | −0.9 | neither |
| always_rebuild | rf | 500 | 0% | −9.8 | −12.0 | neither |
| always_rebuild | sgd | 200 | 25% | +2.1 | +1.0 | neither |
| always_rebuild | sgd | 500 | 24% | +0.6 | −0.3 | neither |
| always_rebuild | xgb | 200 | 33% | −0.9 | −2.2 | **helps** |
| always_rebuild | xgb | 500 | 0% | −11.3 | −11.5 | neither |
| mechanism_selector | gnb | 200 | −7% | −0.5 | −0.5 | neither |
| mechanism_selector | gnb | 500 | 19% | −1.9 | +0.2 | neither |
| mechanism_selector | rf | 200 | 0% | −1.1 | −0.8 | neither |
| mechanism_selector | rf | 500 | −100% | −13.2 | −11.5 | neither |
| mechanism_selector | sgd | 200 | 0% | +3.0 | +4.8 | **hurts** |
| mechanism_selector | sgd | 500 | 22% | −0.6 | +0.9 | **helps** |
| mechanism_selector | xgb | 200 | 0% | −3.1 | −2.4 | neither |
| mechanism_selector | xgb | 500 | −3% | −12.8 | −10.7 | neither |

**By the pre-registered criteria the rule is not a cost-saving method:** helps in 2 of 16
settings, hurts in 1, and has no material effect in 13. The two "helps" do not share a W or a
policy. Most "neither" verdicts fail on *cost*, not accuracy. Waiting changes the whole
trajectory, and with a better replacement model ADWIN often fires more alarms later, so training
cost stays flat or rises. Accuracy did not fall anywhere except sgd at W = 200.

**Does confirmation pick out the false alarms?** Partly. Share of alarm episodes dismissed (all
streams, models, policies):

| W | false detections dismissed | true detections dismissed |
|---|---|---|
| 200 | 70% (of 709) | 23% (of 296) |
| 500 | 82% (of 848) | 31% (of 479) |

It discriminates, but it drops about a quarter to a third of the true detections. Alarms fired on
*falling* error are dismissed 78–84% of the time, against 44–49% for rising error. Caveat: on
abrupt streams "false" means "not within 1,000 rows of a change point". Many of those are
re-alarms after a stale rebuild, not spurious alarms.

## Post-hoc: separating "wait" from "dismiss"

Not pre-registered. Added after seeing results, because the pre-registered grid changes two
things at once. **Delay-only** waits W rows and then always adapts (`--delay-only`, 480 runs).
Changes are relative to acting immediately, pooled over models and both policies:

| stream kind | W | delay-only Δacc | confirm Δacc | delay-only training ratio | confirm training ratio |
|---|---|---|---|---|---|
| abrupt, i.i.d. | 200 | +1.8 | +0.3 | 1.00 | 1.00 |
| abrupt, i.i.d. | 500 | **+10.8** | +8.1 | 1.39 | 1.00 |
| abrupt, sticky | 200 | +1.4 | −0.8 | 1.00 | 0.81 |
| abrupt, sticky | 500 | +5.8 | +2.7 | 1.19 | 0.94 |
| none, sticky | 200 | +0.1 | −0.5 | 1.00 | 0.44 |
| none, sticky | 500 | +0.4 | +0.5 | 1.00 | 0.00 |

Training ratios are medians. In "none, sticky" most cells have very few alarms.

Training window at the moment of adapting, for true detections:

| condition | episodes | median rows from change to alarm | post-change share of window |
|---|---|---|---|
| act now | 283 | 223 | 0.33 |
| delay 200 | 315 | 243 | 0.53 |
| delay 500 | 472 | 319 | 0.80 |
| confirm 200 | 228 | 207 | 0.49 |
| confirm 500 | 329 | 171 | 0.73 |

The share assumes a full 1,000-row buffer, so it is approximate where the buffer was cleared
recently.

**Reading:** waiting buys accuracy and costs training; dismissing saves training and costs some
of that accuracy. Confirm-at-500 sits between the two, with near-flat training cost and +3 to +8
points of accuracy on abrupt streams. That is an accuracy improvement at equal cost, not the cost
saving the rule was proposed for.

## What this means for the study

- **A candidate explanation for the main negative results, untested on real data.** On these
  streams, acting at alarm time trains on a window that is about two-thirds pre-change. That is
  consistent with Finding 3 (nudges recover little of the deficit) and with NeverAdapt being as
  accurate in 10 of 20 real cells. **It is not shown for elec2, covtype or INSECTS**, whose
  changes are not abrupt and mostly unlabelled. The real-stream test is to run `--delay-only` on
  the real data once the hosts are reachable.
- **Not new as a mechanism.** Training replacement models only on post-change data is the
  rationale of classic detector designs: DDM's warning zone, and ADWIN's own shrinking window.
  The contribution available here is the measurement: how stale the window is under
  alarm-triggered retraining, and what that does to cost and accuracy.
- **Confirmation as a cost-saver: not supported** by the pre-registered test on synthetic data,
  and its false-alarm arm was underpowered.

## Limits

- Synthetic streams, abrupt changes only, one dependence level. ρ = 0.8 was not tuned, and it
  turned out too weak to make tree models alarm.
- 10–15 pairs per verdict after the exclusions. The Wilcoxon p-values are in
  `results/confirm/summary.json`, and several "neither" cells have p < 0.05 on accuracy.
- Package versions differ from the study's pinned set, so the W = 0 runs here are not
  comparable number-for-number with the frozen Phase 6 grid. They are only comparable with each
  other.
