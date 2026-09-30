# Phase 10 — oracle upper bound (Gate 10)

**Declared follow-up**, not a diagnostic: it measures headroom against the existing design. The
pre-registered grid is untouched; the trajectory is the MechanismSelector run exactly as the grid
ran it (alarm counts match, e.g. elec2/xgb 71), and the extra branches are executed from the saved
state and discarded.

**Declared subset, as agreed:** ADWIN only, elec2 and covtype, all four model families, seeds 0–4
for RandomForest and SGD and seed 0 for the deterministic families. **2,595 alarms.** Insects cells
were excluded: 5–11 alarms each is too few to estimate headroom.

**Measured cost, reported before committing compute:** 10 alarms per family took 0.004–0.98 s per
alarm of branching, projecting to **0.32 CPU-hours**; the full run took **~15 minutes wall**. My
Gate 9 estimate of 6–10 hours per trajectory was wrong by about 40× — branching is cheap because
rebuilds are on ≤1,000 rows and horizons are ≤1,000 rows.

`J(a) = Ops(a) + λ · Errors(a)`, where `Ops` is training work units **plus** evaluation ops
(inference units × holdout rows, per scoring the action performs) **plus** inference over the
horizon (inference units × H), as required. Inference units come from `measure_footprint` on a
200-row probe from the window, so a shallower rebuilt model is correctly cheaper to query.

## G1 and G2

| horizon | λ = 0 | 1e2 | 1e3 | 1e4 | 1e5 | 1e6 |
|---|---|---|---|---|---|---|
| next alarm | 27.9% | 27.0% | 23.2% | 23.7% | 28.3% | 29.0% |
| 500 rows | 26.6% | 25.7% | 22.8% | 25.1% | 29.9% | 30.6% |
| 1,000 rows | 17.6% | 16.9% | 16.3% | 23.5% | 27.8% | 28.4% |

**G1 is 16–31% of the selector's cost at every λ and every horizon — far above the 5% threshold.**

**G2 equals G1 exactly, everywhere, and that is a limitation of this design rather than a
result.** DEFER keeps the model and pays no training, so in a per-alarm frame its ops and errors
are identical to SKIP and it can never win. Measuring what the guard really costs requires DEFER
to change the *trajectory* — the alarm stays pending, the buffer keeps filling, the next decision
happens later — which full-information replay at fixed snapshots cannot express. **Gate 10's "G2
much larger than G1" case cannot be tested by this experiment.** A trajectory-level variant is
needed, which is what 13.1's defer-until-minimum-buffer guard would do.

## Does any adaptation beat doing nothing, in hindsight?

Yes, at every λ and horizon: `cost(NeverAdapt) − cost(O1)` is positive throughout, from 1.1e7 at
λ = 0 to 1.2e11 at λ = 1e6. Note this holds **even at λ = 0**, where errors are not priced at all:
the oracle still rebuilds 119 times purely to make the model cheaper to query, which is the
Phase 9.5 effect appearing again.

## Where the headroom is

Oracle action mix at λ = 1e3, horizon "next": SKIP 1,943, NUDGE 367, REBUILD 285 — against the
selector's actual 2,595 adaptations. Most headroom is over-adaptation: alarms where skipping was
the better call. By family at λ = 1e3 the oracle's nudge share differs sharply: SGD 332 of 1,212
alarms, GaussianNB 24 of 307, RandomForest 1 of 916, XGBoost 10 of 160 — the same per-family split
Finding 3 reports at the prediction level.

## Learnability

In-sample overfit GBM is an upper bound by construction. Share of G1 captured, horizon "next":

| λ | G1 | overfit GBM | family only | deficit stump | always SKIP |
|---|---|---|---|---|---|
| 1e2 | 2.65e8 | 99.9% | 95.3% | 95.3% | 95.3% |
| 1e3 | 2.90e8 | 99.1% | 78.1% | 78.1% | 78.1% |
| 1e4 | 9.31e8 | 94.7% | 20.4% | 9.6% | **−3.6%** |
| 1e5 | 8.68e9 | 94.0% | 12.9% | 25.9% | **−30.3%** |

Two things matter here. First, **at low λ the trivial rules and "always SKIP" are the same rule** —
they capture 78–95% of the headroom simply by adapting less, so most of G1 at those trade-offs is
not a decision problem at all. Second, **at high λ "always SKIP" is worse than the current policy**
(negative capture), so the headroom there is genuinely about *which* action to take. The
in-sample GBM captures 94–100% throughout, which bounds what is learnable but says nothing about
generalisation — it is fitted and scored on the same alarms, and a leave-one-stream-out version
is the only honest test.

## Which Gate 10 case are we in?

Not "G1 small": G1 is 16–31%, so a better decision component is not ruled out.
Not testable as "G2 ≫ G1": DEFER collapses onto SKIP in this frame, as above.
Closest to **"G1 large, and partly learnable"** — but with the caveat that at the low-λ end the
learnable part is almost entirely "adapt less", which 13.1's Bayesian rule targets directly and
which needs no learned model.

**Recommendation for Gate 10:** run 13.1 in full, including the defer-until-minimum-buffer guard,
since that is the only way to measure the G2 question this phase could not reach. Treat 13.2 as
conditional on the high-λ regime being the one we care about: at λ ≤ 1e3 a logistic model has
little to add over "adapt less", while at λ ≥ 1e4 there is real per-alarm structure (the trivial
rules fail, the fitted model does not). Deciding that means choosing the λ region the paper
argues about, which is a framing decision rather than a measurement.

Files: `experiments/phase10_oracle.py`, `experiments/phase10_analysis.py`,
`results/analysis/phase10_oracle.json`, `results/analysis/phase10_summary.json`.
