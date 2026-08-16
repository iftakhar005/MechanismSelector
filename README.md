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
| 2 | `accounting.py` | 300-tree fit = 30× work units of 10-tree fit | pending |
| 3 | `mechanisms.py` | nudge cheaper than rebuild; RF tree count constant; SVC raises | pending |
| 4 | `selector.py` | holdout rows never appear in training data | pending |
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
