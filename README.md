# MechanismSelector — The Third Axis

Mechanism-aware retraining for energy-efficient concept-drift adaptation.

When drift is detected, current practice rebuilds the model from scratch. This
project tests a cheaper policy: try a small incremental update first, measure
whether it worked on held-out data, and rebuild only if it did not.

**The economic argument.** A nudge costs ~5% of a rebuild. If it succeeds we save
~95%; if it fails we wasted 5% and rebuild anyway. Try-first therefore wins as
long as nudges succeed more than ~5% of the time. The headline number in the
results is the **nudge success rate**.

**Primary metric is not energy.** It is `work_units = n_estimators_fitted ×
n_rows_processed` — exact, deterministic, and identical on any machine. Energy
from CodeCarbon is logged as secondary corroboration only; on CPU without RAPL
it is an estimate with up to 40% error, and it never influences any decision.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy pandas scikit-learn xgboost river scipy codecarbon scikit-posthocs matplotlib pytest
```

Isolated in its own venv deliberately — this stack pulls newer numpy/pandas than
some neighbouring projects pin.

Verified working set (2026-08-16): Python 3.12.3, river 0.25.0, xgboost 3.4.1,
scikit-learn 1.9.0, numpy 2.5.2, pandas 3.0.5.

## Build status

| Phase | Module | Gate | Status |
|---|---|---|---|
| 1 | `datasets.py` | all five loaders return correctly shaped arrays | **PASS** |
| 2 | `accounting.py` | 300-tree fit = 30× work units of 10-tree fit | **PASS** |
| 3 | `mechanisms.py` | nudge cheaper than rebuild; RF tree count constant; SVC raises | **PASS** |
| 4 | `selector.py` | holdout rows never appear in training data | **PASS** |
| 5 | `policies.py` | all five policies run end-to-end | pending |
| 6 | `runner.py` | full grid, 500-row CSV | pending |
| 7 | `analysis.py` | four figures, Friedman + Nemenyi | pending |
| 8 | packaging | minimal installable package, public API = `MechanismSelector.adapt()` | pending |

## Phase 1 notes — data layer

Run the acceptance check with:

```bash
.venv/Scripts/python.exe experiments/verify_datasets.py
```

Measured on 2026-08-16:

| Stream | Rows | Features | Classes | Source |
|---|---|---|---|---|
| `elec2` | 45,312 | 8 | 2 | `river.datasets.Elec2` |
| `insects_abrupt` | 52,848 | 33 | 6 | `river.datasets.Insects("abrupt_balanced")` |
| `insects_gradual` | 24,150 | 33 | 6 | `river.datasets.Insects("gradual_balanced")` |
| `insects_incremental` | 57,018 | 33 | 6 | `river.datasets.Insects("incremental_balanced")` |
| `covtype` | 100,000 (capped) | 54 | 7 | `sklearn.datasets.fetch_covtype` |

### Design decisions

**Covtype does not come from river.** The spec's table listed
`river.datasets.Covtype()`, which does not exist in river 0.25.0 — the library
has no covtype loader at all. It is sourced from `sklearn.datasets.fetch_covtype`
with `shuffle=False` passed explicitly, preserving original file order.

**Never shuffle; capping takes a temporal prefix.** Temporal order is the entire
point of the study. `load_stream(name, cap)` truncates to the first `cap` rows
and never samples.

**Column order is validated, not assumed.** River yields `(dict, label)` pairs.
Column order is fixed from the first record and every later record is asserted
to match, so a single reordered record raises instead of silently scrambling
features.

**Labels are encoded by sorted order**, not first-seen order, so a class maps to
the same integer regardless of where the stream is capped.

### Known caveat — covtype class balance

The first 100,000 rows of covtype in original order are heavily skewed:

```
class 0: 22,026   class 1: 66,751   class 2: 2,160   class 3: 2,160
class 4:  2,583   class 5:  2,160   class 6: 2,160
```

Covtype's native file order groups records geographically, so a temporal prefix
is dominated by two cover types. All seven classes are present, but five of them
are rare. This makes covtype's "drift" largely class-prior shift, which is the
standard way the dataset is used in the drift literature — but it means covtype
accuracy figures should be read alongside the balance above, not as if the
stream were balanced. Reported, not corrected: rebalancing would require
reordering, which is forbidden.

## Phase 2 notes — operation accounting

```bash
.venv/Scripts/python.exe -m pytest tests/test_accounting.py -q
```

`work_units = n_passes × n_rows_processed`. Exact, deterministic, hardware-independent.

### What counts as a pass

A pass is one traversal of the training data. The count is read from the fitted
model, never assumed:

| Model | Passes | Source |
|---|---|---|
| `XGBClassifier` | boosting rounds added | `booster.num_boosted_rounds()` delta |
| `RandomForestClassifier` | trees added | `len(estimators_)` delta |
| `SGDClassifier` | epochs actually run | `n_iter_` |
| `MLPClassifier` | epochs actually run | `n_iter_` |
| `GaussianNB` | 1 | closed-form, single pass |

**Why not simply count one pass per fit call.** That would make `fit` and
`partial_fit` indistinguishable for SGD and MLP — and their difference *is* the
saving this study measures. Measured on a 10,000-row synthetic set,
`SGDClassifier.fit` runs 36 epochs where `partial_fit` runs exactly 1. Flattening
both to 1 would erase a real 36× cost difference and silently overstate the
nudge's advantage for those models.

Tree ensembles accumulate under `warm_start` / `xgb_model`, so those are
measured as a delta against a baseline captured before the fit — charging a
nudge for all 315 trees when it only added 15 would destroy the economic
argument.

### The `n_iter_` convention is not assumed

scikit-learn does not document whether repeated `partial_fit` calls accumulate
`n_iter_`. Measured on 1.9.0, five consecutive calls on the same model:

```
SGD:  n_iter_ = 1, 1, 1, 1, 1
MLP:  n_iter_ = 1, 1, 1, 1, 1     (loss 0.751 → 0.744 → 0.737 → 0.731 → 0.724)
```

Both **reset** rather than accumulate, including with `warm_start=True`. The
falling loss confirms the model carries state across calls — only the counter
resets.

The accounting does not hardcode this. A plain delta would be actively wrong
under reset semantics (`1 - 1 = 0`), recording every nudge as free — worse than
an overcharge, because it would fabricate the headline result. `resolve_iter_passes`
charges the delta when the counter grew and the raw value when it did not, which
is correct under either convention. Two tests pin the observed 1.9.0 behaviour so
a future version change fails loudly, and a third proves the rule still holds
against a fake accumulating estimator.

`CostRecord` stores `n_passes` and `pass_source` alongside `n_estimators_fitted`
so every number in the results is auditable back to where it came from.

### GaussianNB is a control case, not a candidate for savings

GaussianNB is expected to show **minimal or no savings**, by construction. Its
rebuild is already a single closed-form pass over the data, so a nudge
(`partial_fit`, also one pass) costs essentially the same as a rebuild. There is
no 20:1 ratio to exploit.

It stays in the grid deliberately. Which model families admit a meaningful
mechanism choice is itself a reportable finding, and GaussianNB is the negative
control that shows the selector is not manufacturing savings where none exist.
A policy that "saves" on GaussianNB would be evidence of a measurement bug.

### Cross-model comparison

Work units are **not** comparable across model families — one XGBoost tree and
one GaussianNB pass are not the same unit of work. Cross-model comparison goes
through the "% of `AlwaysRebuild`" normalisation, which is the headline table
anyway. Within a (dataset, model) cell the comparison is exact.

### Secondary metrics

`wall_clock_s` and `energy_kwh` are recorded but never read by decision logic.
Energy is left `None` at the operation level and captured once per run by the
runner, because starting a CodeCarbon tracker per fit would cost more than the
fits themselves.

## Phase 3 notes — mechanism library

```bash
.venv/Scripts/python.exe -m pytest tests/test_mechanisms.py -q
.venv/Scripts/python.exe experiments/verify_mechanisms.py
```

### A nudge is not one operation

The spec's "add capacity equal to 5% of the model's original size" describes the
tree ensembles exactly. It does not describe the others, because they have no
capacity to add:

| Family | What a nudge actually is | Governed by |
|---|---|---|
| `XGBClassifier` | append `ceil(0.05 × original)` boosting rounds | the 5% rule |
| `RandomForestClassifier` | append `ceil(0.05 × original)` trees, drop the same number of oldest | the 5% rule |
| `SGDClassifier` | one extra epoch (`partial_fit`) | convergence behaviour |
| `MLPClassifier` | one extra epoch (`partial_fit`) | convergence behaviour |
| `GaussianNB` | update sufficient statistics (`partial_fit`) | closed form |

A nudge is therefore **each family's cheapest available incremental update**,
not a uniform 5% of anything. The paper should describe it that way. The
rebuild:nudge ratio is a measured quantity per family, not a constant the 5%
rule implies.

### Measured rebuild:nudge ratios

Window 1,000 rows; nudge trains on 800 (the 80% train part), rebuild on all
1,000 — the configuration the selector actually uses.

| Stream | Model | Nudge WU | Rebuild WU | Ratio | Nudge as % of rebuild |
|---|---|---|---|---|---|
| elec2 | xgb | 4,000 | 100,000 | 25.0× | 4.0% |
| elec2 | rf | 4,000 | 100,000 | 25.0× | 4.0% |
| elec2 | sgd | 800 | 78,000 | 97.5× | 1.0% |
| elec2 | gnb | 800 | 1,000 | 1.2× | 80.0% |
| insects_abrupt | xgb | 4,000 | 100,000 | 25.0× | 4.0% |
| insects_abrupt | rf | 4,000 | 100,000 | 25.0× | 4.0% |
| insects_abrupt | sgd | 800 | 126,000 | 157.5× | 0.6% |
| insects_abrupt | gnb | 800 | 1,000 | 1.2× | 80.0% |
| covtype | xgb | 4,000 | 100,000 | 25.0× | 4.0% |
| covtype | rf | 4,000 | 100,000 | 25.0× | 4.0% |
| covtype | sgd | 800 | 96,000 | 120.0× | 0.8% |
| covtype | gnb | 800 | 1,000 | 1.2× | 80.0% |

**Trees are 25×, not 20×.** The 5% rule alone would give 20:1, but the nudge
trains on the 80% train part while the rebuild uses the full window:
`0.05 × 0.8 = 0.04`, so 4% and 25:1. Quote the measured figure, not the
construction figure.

**SGD varies from 97× to 158× across streams**, because its ratio is set by how
many epochs `fit` needs to converge on that data (78, 126 and 96 epochs
respectively) rather than by any fixed rule. This is a per-dataset property, and
reporting a single SGD ratio would misrepresent it.

**GaussianNB's 1.2× is not a saving.** Both operations are exactly one pass; the
entire difference is that the nudge sees 800 rows and the rebuild sees 1,000.
That is a window-size artifact, not a mechanism advantage — read as "no saving
available," which is what a negative control should show. If GaussianNB ever
reports a large saving in the final results, that is a measurement bug, not a
finding.

### Design decision — Random Forest tree replacement

`warm_start` only ever *adds* trees. Under sustained drift the forest fills with
stale trees and the nudge stops working — the model appears to adapt early and
fail later, which looks like a real finding but is an implementation artifact.
This is dilution, not adaptation.

`NudgeMechanism` therefore adds `k` trees and drops the `k` oldest, holding the
forest at its original size. Tests pin all three properties that matter: the
count stays constant over 10 nudges, the retained trees are genuinely the newest
(not the same ones each time), and each nudge is still charged for the `k` trees
it fitted — truncation must not make repeated nudges look free.

### Unsupported families

`SVC` and `KNeighborsClassifier` raise `MechanismUnavailable`. No incremental
mechanism exists for either. This is a recorded outcome rather than an error to
work around: which model families admit a mechanism choice is itself a
reportable finding.

## Phase 4 notes — the selector

```bash
.venv/Scripts/python.exe -m pytest tests/test_selector.py -q
.venv/Scripts/python.exe experiments/verify_selector.py
```

### Framing: try-and-escalate, not prediction

`MechanismSelector` is a **cost-ordered try-and-escalate policy under an
accuracy constraint**. The decision variable at runtime is accuracy — a nudge
is kept if it clears the floor on held-out data, discarded and escalated to a
rebuild if it does not. Cost does not decide anything; it only sets the order
in which options are attempted (cheapest first) and is what makes attempting
the cheap option rational: a nudge costs a few percent of a rebuild, so trying
and occasionally failing is cheaper than building a predictor of when it would
succeed. Earlier phrasing described this as "selection by measured compute" —
that overstates what the mechanism does and has been corrected everywhere,
including in `selector.py`'s docstrings.

### Procedure

1. Split the adaptation window by **time**: `train_part` is the older
   `1 - holdout_frac` prefix, `holdout` is the most recent `holdout_frac`
   suffix. Never a random split.
2. Score the current model on `holdout`. Good enough (within `floor_drop` of
   `reference_accuracy`) → `SKIP`, zero cost, model returned unchanged.
3. Otherwise nudge a **copy** of the model on `train_part` only (the holdout
   never appears in any training data) and score the nudge on the same
   holdout. Clears the floor → `NUDGE`.
4. Otherwise rebuild from scratch on the full window (holdout included) and
   use that. The failed nudge's cost is **added** to the rebuild's, not
   discarded — see below.

### The reference-accuracy problem

`reference_accuracy` is an input to `adapt()`, not state the selector
maintains. After a `REBUILD` the model has trained on the entire window, so no
clean data remains inside it to certify a new baseline — scoring on the
post-rebuild holdout would be scoring on data the model just trained on and
would set an inflated baseline, against which every subsequent `SKIP` would be
judged unfairly.

**The runner owns `reference_accuracy` and must re-measure it from unseen
future rows** after every `adapt()` call: take the next `N_REF` (default 200)
rows the stream hasn't produced yet, score the returned model on them, and use
that as `reference_accuracy` for the next alarm. Those rows then continue
through the normal stream loop as usual — they are not consumed. If fewer than
`N_REF` rows remain, carry the previous `reference_accuracy` forward and log
that this happened. `experiments/verify_selector.py` implements this rule as a
manual, single-pass illustration (not the Phase 6 runner).

### Guard condition — degrading to REBUILD on small windows

Below `min_holdout_rows` (default 50) or `min_train_rows` (default 100), a
holdout is too small for its accuracy read to be signal rather than noise —
with a 20-row holdout, a single misclassification moves accuracy by 5 points,
larger than the default 2-point floor.

**This bypasses both the `SKIP` and the `NUDGE` checks, not just the nudge.**
The spec text only explicitly says "do not attempt a nudge," but the stated
rationale — the accuracy estimate is noise — applies just as much to trusting
a tiny holdout's "the model is still fine" as it does to trusting its "the
nudge worked." Since a drift alarm already fired to get here, we don't have a
reliable signal to justify doing nothing, so the safe default is to spend the
cost and rebuild. `acc_before` is still computed and recorded on the small
holdout for audit purposes, but the decision does not use it.
`AdaptResult.degraded_to_rebuild` flags every time this happened, so the
analysis can count it.

### REBUILD's `acc_after` is optimistic

A `REBUILD` trains on the full window, holdout included, then is scored on
that same holdout — data it just trained on. `acc_after` for a `REBUILD` is
therefore **not comparable** to `acc_before` / `acc_nudged`, which are always
scored on data the model being evaluated has never seen. This is correct
behaviour (the rebuild is the final answer and should use everything
available) but downstream analysis must not treat it as apples-to-apples.

### Cost accounting

A `REBUILD` outcome's cost is `nudge_cost + rebuild_cost`, never `rebuild_cost`
alone — the wasted nudge attempt is not discarded. Silently dropping it would
flatter the method's headline number (spec trap #5). A `SKIP`'s cost is an
exact `CostRecord.zero()`, not a near-zero approximation.

### Scoring

Default scorer is plain accuracy, but it is injectable
(`scorer: Callable = accuracy_score`) because the multi-class streams
(`insects_*`, `covtype`) are severely imbalanced. Experiments on those streams
should use `balanced_accuracy_score`; `elec2` uses plain accuracy. `floor_drop`
is in the units of whichever scorer is chosen, and every result row must
record which scorer was used — do not silently mix them across a grid.

### Manual verification — elec2 + XGBoost, one pass

34 fixed-size 1,000-row windows walked across `elec2`, each treated as if a
drift alarm had just fired on it (this is a lightweight stand-in for Phase 6's
ADWIN-triggered loop, not the loop itself):

```
SKIP=14  NUDGE=10  REBUILD=10  (degraded_to_rebuild=0)
```

**Nudge success rate: 10 / 20 = 50%** of the times a nudge was attempted, it
recovered accuracy above the floor without needing a rebuild. The core
economic argument only requires this to exceed ~4% (the measured nudge cost as
a fraction of rebuild cost for XGBoost, see Phase 3 table) for try-first to be
worth it on expectation — 50% clears that by a wide margin on this stream. Full
JSON in `results/selector_manual_run_elec2_xgb.json`.
