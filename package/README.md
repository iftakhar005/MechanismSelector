# mechanism-selector 0.1.0

**A research artifact, published so the results it produced can be reproduced
and scrutinised. It is not an energy-saving tool, and it did not work as hoped.**

When a drift detector fires, `MechanismSelector` tries a cheap incremental update
of the model (a "nudge") first, checks it against held-out recent data, and
rebuilds the model from scratch only if the nudge fails to restore accuracy.

## Read this before using it

Evaluated on 5 data streams (elec2; covtype; insects abrupt, gradual and
incremental) × 4 model families (XGBoost, Random Forest, SGD, Gaussian NB) × 5
seeds, with ADWIN as the drift detector and a DDM contrast:

- **It does not beat simpler policies.** A fixed schedule — nudge, nudge, rebuild
  — was cheaper in 16 of 20 settings. The selector sat on the cost–accuracy Pareto
  frontier in only 4 of 20. Never adapting at all was at least as accurate, at zero
  cost, in 10 of 20.
- **It is frequently defeated by its own minimum-window guard.** When the recent
  window is too small to evaluate a nudge reliably (under ~250 rows by default),
  it rebuilds unconditionally. Half of all its rebuilds (910 of 1,825) came from
  that guard rather than from a failed nudge. A caller that clears its buffer after
  every adaptation will hit this constantly.
- **Cheap incremental repair recovers only a fraction of the accuracy deficit.**
  Only 15% of nudge attempts cleared the accuracy floor. For XGBoost a nudge
  recovered a median 11–20% of the deficit; for Random Forest and Gaussian NB the
  nudge left predictions unchanged in most attempts; for SGD it changed many
  predictions without a systematic gain.
- **Nudging XGBoost grows the model without bound.** Each nudge appends boosting
  rounds that every later prediction must evaluate: over 34 nudges, per-prediction
  cost grew 2.45×. Training cost alone hides this.
- **Many alarms are not degradations.** With ADWIN, about 45% of alarms fired
  while the error rate was falling, and a quarter to a third of adaptation compute
  went on reacting to them. These are properties of detector-triggered adaptation
  in general, not just of this policy.

## Install

```bash
pip install mechanism-selector            # numpy + scikit-learn
pip install "mechanism-selector[xgboost]" # adds XGBoost support
```

From a clone of the repository: `pip install -e .`

## Example

```python
import numpy as np
from mechanism_selector import MechanismSelector
from mechanism_selector.mechanisms import NudgeMechanism, RebuildMechanism, make_spec

rng = np.random.default_rng(0)
X, y = rng.normal(size=(3000, 5)), rng.integers(0, 2, 3000)
spec = make_spec("rf", classes=np.array([0, 1]))
model = spec.factory(0).fit(X[:2000], y[:2000])

selector = MechanismSelector(NudgeMechanism(spec, seed=0), RebuildMechanism(spec, seed=0))
result = selector.adapt(model, X[2000:], y[2000:], reference_accuracy=0.90)
print(result.action, result.cost.work_units, round(result.acc_before, 3))
```

`result.action` is `"SKIP"`, `"NUDGE"` or `"REBUILD"`; `result.model` is the model
to use next. `result.cost.work_units` is the exact training work done (passes over
the data × rows), including a failed nudge when the outcome is a rebuild.

## Public interface

`MechanismSelector` and its `adapt()` method. The constructor takes a nudge
mechanism and a rebuild mechanism, which are built with
`mechanism_selector.mechanisms` (`make_spec`, `NudgeMechanism`,
`RebuildMechanism`) — that is the interface the experiments used, and it is
published unchanged. Supported model families: `"xgb"`, `"rf"`, `"sgd"`,
`"gnb"`, `"mlp"`; `"svc"` and `"knn"` have no incremental update and raise
`MechanismUnavailable` when nudged.

You maintain `reference_accuracy` yourself: after each `adapt()`, score the
returned model on the next rows the stream produces and pass that in next time.
Never reuse `result.acc_after` — after a rebuild it is measured on training data.

## Known issues, published as tested

These are left exactly as they were when the results were produced.

- `MechanismSelector(seed=...)` has no effect with the provided mechanisms. It
  seeds a generator passed to them as `rng`, which they ignore; set
  `NudgeMechanism(seed=...)` and `RebuildMechanism(seed=...)` instead.
- On windows missing some classes, fits add zero-weight placeholder rows so the
  model keeps every class. This is exactly inert for XGBoost and Gaussian NB; for
  Random Forest and SGD it shifts the random stream without carrying weight.
- `MechanismSelector` also has a public `name` class attribute, used by the
  experiment harness to label results.

## Citing

A DOI will be added when the repository is archived. Until then, cite the
repository and version 0.1.0.
