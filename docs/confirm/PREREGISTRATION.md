# Confirm-before-acting — pre-registration

Written and committed **before** any confirmation run. Parameters below are fixed; nothing is
tuned after results are seen. Anything added later is labelled as post-hoc.

## Question

Phase 9 found that much of the ADWIN alarm stream is produced by temporal dependence rather than
concept change, and Phase 10 found that most of the oracle's headroom over the selector is
over-adaptation ("SKIP was the better call"). Phase 10 could not measure DEFER, because in a
per-alarm replay it collapses onto SKIP. This test changes the *trajectory*: the alarm stays
pending while the stream keeps flowing, and the decision is made later on rows that arrived
after the alarm.

**Does confirming an alarm on fresh rows before acting save adaptation cost without losing
accuracy on real drift?**

## The rule (`confirm_rows = W`)

1. ADWIN alarms at row `a`. The model keeps predicting rows `a+1 … a+W` unchanged (prequential,
   so every label used has already arrived).
2. At row `a+W`, score the current model on those `W` rows with the run's scorer (accuracy for
   binary streams, balanced accuracy for multi-class).
3. **Confirmed** if that score is below `reference_accuracy − floor_drop` (the selector's own
   floor, `floor_drop = 0.02`). Then the policy adapts at row `a+W`, on a buffer that includes the
   `W` confirmation rows.
4. **Dismissed** otherwise: no adaptation, no cost, buffer and reference kept.
5. Alarms during the wait coalesce into the pending one. An alarm still waiting at stream end is
   unserved.

`W = 0` is the existing runner, bit-for-bit (tested).

**Fixed values:** `W ∈ {200, 500}`. 200 is the study's reference length (`n_ref`); 500 is the
longer sensitivity check. No other value will be run for the headline.

## Streams — synthetic only, with known change points

The real streams cannot be downloaded from this environment (network policy). Synthetic streams
are the part of the design that has ground truth anyway. 50,000 rows each; abrupt concept changes
at rows 12,500, 25,000 and 37,500; stream seed = run seed.

| Stream | Generator | Concepts by segment | Sampling |
|---|---|---|---|
| `sea_abrupt_iid` | river `SEA`, noise 0.1 | variants 0, 2, 1, 3 | i.i.d. |
| `sea_abrupt_sticky` | same | same | sticky |
| `sea_none_sticky` | same | variant 0 throughout | sticky |
| `agrawal_abrupt_iid` | river `Agrawal`, perturbation 0.05 | functions 0, 2, 4, 6 | i.i.d. |
| `agrawal_abrupt_sticky` | same | same | sticky |
| `agrawal_none_sticky` | same | function 0 throughout | sticky |

**Sticky sampling** adds the temporal dependence Phase 9 identified, without changing any concept.
Within a segment, rows are drawn from the generator into per-class pools. The class sequence is
a Markov chain that keeps the previous class with probability `ρ = 0.8` and otherwise draws from
the segment's empirical class prior. The next row of that class is taken from its pool in order.
`P(y | x)` is untouched, so **every alarm on a `none` stream is false with respect to the
concept**. An i.i.d. no-drift stream is omitted: Phase 9.6 found ADWIN fires no alarms on one.

## Grid

- Models: xgb, rf, sgd, gnb (the study's four).
- Seeds: 0–4.
- Policies: `always_rebuild` and `mechanism_selector`, each at `W ∈ {0, 200, 500}`; `never_adapt`
  once as a reference.
- Total: 6 streams × 4 models × 5 seeds × 7 = 840 runs. All runs, including `W = 0`, are executed
  fresh in the same environment. The frozen Phase 6 grid is not reused or modified.

## Measurements

- **Training cost:** adaptation work units, as everywhere in the study.
- **Inference cost, exact:** the footprint is re-measured after every adaptation, and each model
  is charged for the rows it actually served. This removes the Phase 9.5 interpolation. It is
  reported separately from training, never summed without the unit caveat.
- **Accuracy:** mean prequential accuracy over the stream, plus accuracy over the 2,000 rows after
  each change point (the cost of reacting late).
- **Alarm truth:** an alarm episode is a *true detection* if it starts within 1,000 rows after a
  change point (Phase 9.3's window), otherwise *false*. For each `W`, report the share of true
  and of false episodes that were dismissed.

## Decision criteria, fixed now

Per (policy, model), paired over stream × seed, `W > 0` vs `W = 0`:

- **Helps:** median training-cost reduction ≥ 20% **and** mean-accuracy loss ≤ 0.5 percentage
  points, with the accuracy loss also ≤ 1.0 point in the post-change windows.
- **Hurts:** post-change accuracy loss > 1.0 point, whatever the saving.
- **Otherwise:** no material effect.

Reported per family and per stream, never pooled alone (Simpson's caveat from Phase 9). Paired
Wilcoxon signed-rank tests on training cost and mean accuracy, n = 30 pairs per (policy, model, W).
Both `W` values are always reported. Neither is promoted after the fact.

## Known limits, stated in advance

- Synthetic streams decide nothing about elec2, covtype or INSECTS. The real-stream run is a
  follow-up once the data hosts are reachable.
- `ρ = 0.8` is one level of dependence, chosen before running. It is not tuned to produce alarms.
  If a `sticky` stream averages fewer than 5 alarms per run under `W = 0`, that is reported, and
  its savings are not used for the verdict.
- The rule uses the selector's floor. A different confirmation statistic could behave
  differently, and only this one is tested.
