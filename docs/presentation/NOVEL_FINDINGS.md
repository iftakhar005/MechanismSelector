# Novel findings — MechanismSelector

Scope: 5 streams × 4 model families, 60 runs per policy per detector. ADWIN: two-sided,
δ = 0.002, raw 0/1 error stream. DDM: library defaults. Costs are given as training, inference
and combined, over a mean horizon of 50,279 predictions per run, and always beside accuracy.

### 1. A policy's own safety guard, not its decision logic, drove half its rebuilds

**Measured:** 910 of the selector's 1,825 rebuilds (49.9%) were forced by its minimum-window
guard rather than by a failed nudge — 32% of all 2,830 adaptations. (`results/grid/grid.csv`)

**New:** a precondition meant to protect the decision — refuse to judge a holdout under ~250
rows — interacts with the buffer supplying those windows and becomes the policy's dominant
behaviour. **Means:** the window-supply rule is part of the policy; clearing the buffer at every
adaptation guarantees the guard fires on the next close alarm.

### 2. A third of ADWIN alarms fire while the model is significantly better than its post-adaptation reference

**Measured:** 38.1% of 2,515 ADWIN alarms (36.5% corrected for autocorrelation) fired while
accuracy was *significantly* above the reference taken on the 200 rows after the previous
adaptation — against the 5% such a test yields by construction. Under DDM: 8.6%, and 3.5%
corrected, indistinguishable from the test's own rate. This is an ADWIN result only.
(`results/analysis/phase9_reference_noise.json`)

**New:** the excess survives significance testing against a moving baseline on one detector and
vanishes on another. DDM's baseline is the minimum cumulative error since its own last reset
(river 0.25.0 source), below our reference error in 79% of its alarms by a median 14.7 points —
the two answer different questions rather than disagreeing.

**Context:** most ADWIN alarms here are not responses to published drift: 73% fall outside 1,000
rows of any change point on insects_abrupt (verified against Souza et al. 2020, Table 2), 40%
vanish when local label dependence is destroyed, and a no-drift AR(1) null carrying only the
observed autocorrelation reproduces the falling-error share (42% vs 42%). **Means:** an alarm is
a request to check, not evidence of degradation; "above reference" means better than recently,
not healthy.

### 3. Cheap repair fails differently in each model family, at the prediction level

**Measured:** per nudge under AlwaysNudge — Random Forest changed **no** prediction in 60% of
1,876 nudges, Gaussian NB in 77% of 364; SGD changed a median 9.4% of predictions across 1,816
nudges for a median gain of 0.0 points; only XGBoost had a positive median (+2.5 points, 212
nudges). (`results/analysis/phase7.json`)

**New:** that cheap repair is limited is established; that the limit differs in kind — inert for
bagged and closed-form learners, active but directionless for SGD — is not. Refitting on the full
window changes nothing (median 0.00 pp over 1,225 paired alarms; XGBoost +2.28 pp). **Means:**
`partial_fit` and `warm_start` are not one capability — check that the update moves predictions
at all before building a policy on it.

### 4. Training cost is the minor term, and adapting can buy cheaper inference

**Measured:** inference is 85–100% of combined cost, and counting it changes the cheapest policy
in all 10 tree-model cells — away from NeverAdapt, because a model rebuilt on 1,000 rows is
shallower than one trained on 8,000. Repeated nudging runs the other way: an XGBoost booster
grows 100 → 365 rounds on elec2, raising per-prediction node visits 2.87× (3.18× on covtype).

| policy | accuracy | training | inference | combined |
|---|---|---|---|---|
| NeverAdapt | 43.5% | 0 | 2.81e9 | 2.81e9 |
| AlwaysNudge | 44.8% | 3.7e6 | 2.93e9 | 2.94e9 |
| FixedSchedule | 44.5% | 1.56e8 | 2.30e9 | 2.45e9 |
| MechanismSelector | 47.0% | 1.74e8 | 2.41e9 | 2.58e9 |
| AlwaysRebuild | 48.2% | 4.20e8 | 2.33e9 | 2.75e9 |

**New:** retraining-energy studies count training, which at this horizon is a few percent of the
total and can reverse sign once inference is included. **Means:** report training, inference and
combined with the horizon stated, and never rank on cost alone.

## What we do not claim

The method does not win: less accurate than always rebuilding (median −1.3 points, Holm
p = 0.017), not reliably cheaper than a fixed schedule, and no policy wins on accuracy under
either detector. Comparisons are detector-dependent. Five streams is a small sample and cells
within a stream share drift points. Finding 2 gives no mechanism for the ADWIN excess; it reports
the rate and rules out two explanations. Finding 1 describes this implementation's defaults.
Inference between the two measured endpoints is interpolated from the logged size trajectory, and
the combined column adds row-passes to node visits, which are different units.
