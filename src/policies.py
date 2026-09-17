"""Baseline policies: the five strategies the selector is measured against.

| Policy | On every drift alarm | Role in the study |
|---|---|---|
| `NeverAdapt` | do nothing | lower bound on cost |
| `AlwaysRebuild` | full rebuild | current practice -- the thing to beat |
| `AlwaysNudge` | cheap update | upper bound on savings, likely poor accuracy |
| `FixedSchedule(k=3)` | nudge, but rebuild every k-th alarm | the timer competitor |
| `MechanismSelector` | try the nudge, measure it, escalate if it fails | ours (`selector.py`) |

All five expose the same `adapt(model, X_window, y_window, reference_accuracy)
-> AdaptResult` signature, so the runner treats them identically.

## Mechanisms are held fixed; only the decision rule varies

Every policy that nudges performs *the same nudge* the selector performs: a copy
of the model, trained on the older `1 - holdout_frac` of the window. Every
policy that rebuilds performs *the same rebuild*: a fresh model, trained on the
full window, handed `None` as its model argument. The split comes from the one
`temporal_split` function the selector also uses.

This is deliberate. Had the baselines trained on the whole window, `AlwaysNudge`
and `FixedSchedule` would see 25% more data per nudge than the selector does,
and any difference in accuracy or cost could be down to data quantity rather
than to the policy. Holding the actions fixed means a difference between two
policies is attributable to *when each chose to nudge or rebuild* -- which is the
claim under test.

It is also why the baselines still set aside a holdout they never use to decide
anything: it keeps each nudge identical to the selector's, and lets every policy
report the same out-of-sample diagnostics.

## Why `FixedSchedule` matters

It answers the strongest objection a reviewer will raise: *why measure anything,
when a timer that rebuilds every k-th alarm and nudges in between is simpler?*
The selector has to beat it on cost at comparable accuracy, or the paper has to
report honestly that it does not.

## Reading `AdaptResult` for a baseline

The fields keep the selector's meanings, because the split and actions are the
same:

- `acc_before` -- incoming model on the holdout. Out-of-sample for every policy,
  so alarm-time accuracy is directly comparable across all five.
- `acc_nudged` -- nudged model on the holdout, whenever a nudge was performed;
  otherwise `None`. Out-of-sample: the nudge never trains on the holdout.
  Recorded even though no baseline acts on it, so the analysis can ask how often
  a timer's nudges would have failed the selector's floor.
- `acc_after` -- returned model on the holdout. In-sample after a rebuild, which
  trained on the full window -- the same caveat the selector's REBUILD carries.
- `floor` -- `NaN`. Baselines apply no accuracy floor; recording one they never
  used would misrepresent the decision.
- `degraded_to_rebuild` -- always `False`. The small-window guard protects a
  *decision* made from a noisy holdout; baselines make no such decision.

Accuracy is scored for diagnostics only and costs no work units -- prediction is
not training.

## One instance per run

`FixedSchedule` counts alarms, so a fresh instance must be built for every run;
`make_policy` does this. The other four hold no state between calls.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Protocol

import numpy as np
from sklearn.metrics import accuracy_score

from accounting import CostRecord
from mechanisms import ModelSpec, NudgeMechanism, RebuildMechanism
from selector import NUDGE, REBUILD, SKIP, AdaptResult, MechanismSelector, temporal_split

__all__ = [
    "Policy",
    "NeverAdapt",
    "AlwaysRebuild",
    "AlwaysNudge",
    "FixedSchedule",
    "MechanismSelector",
    "POLICY_KEYS",
    "make_policy",
]


class Policy(Protocol):
    """What the runner relies on. `MechanismSelector` satisfies it structurally."""

    name: str

    def adapt(
        self,
        model: Any,
        X_window: np.ndarray,
        y_window: np.ndarray,
        reference_accuracy: float,
    ) -> AdaptResult: ...


class _FixedActionPolicy:
    """Shared machinery: the selector's split, nudge and rebuild, without its
    decision rule. Subclasses decide which action to take on each alarm."""

    name = "abstract"

    def __init__(
        self,
        nudge_mechanism: Any,
        rebuild_mechanism: Any,
        holdout_frac: float = 0.20,
        scorer: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
        seed: int = 0,
    ):
        self.nudge_mechanism = nudge_mechanism
        self.rebuild_mechanism = rebuild_mechanism
        self.holdout_frac = holdout_frac
        self.scorer = scorer
        self.seed = seed

    def _score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.scorer(y, model.predict(X)))

    def _skip(self, model, X_window, y_window) -> AdaptResult:
        _, _, X_holdout, y_holdout = temporal_split(X_window, y_window, self.holdout_frac)
        acc_before = self._score(model, X_holdout, y_holdout)
        return AdaptResult(
            model=model,
            action=SKIP,
            cost=CostRecord.zero(),
            acc_before=acc_before,
            acc_nudged=None,
            acc_after=acc_before,
            floor=math.nan,
            n_train_rows=0,
            n_holdout_rows=len(X_holdout),
        )

    def _nudge(self, model, X_window, y_window) -> AdaptResult:
        """Identical to the selector's nudge: a copy, trained on train_part only."""
        rng = np.random.default_rng(self.seed)
        X_train, y_train, X_holdout, y_holdout = temporal_split(
            X_window, y_window, self.holdout_frac
        )
        acc_before = self._score(model, X_holdout, y_holdout)

        candidate = copy.deepcopy(model)  # the caller's model is never mutated
        candidate, cost = self.nudge_mechanism.apply(candidate, X_train, y_train, rng)
        acc_nudged = self._score(candidate, X_holdout, y_holdout)

        return AdaptResult(
            model=candidate,
            action=NUDGE,
            cost=cost,
            acc_before=acc_before,
            acc_nudged=acc_nudged,
            acc_after=acc_nudged,
            floor=math.nan,
            n_train_rows=len(X_train),
            n_holdout_rows=len(X_holdout),
        )

    def _rebuild(self, model, X_window, y_window) -> AdaptResult:
        """Identical to the selector's rebuild: fresh, full window, handed None.

        Charged for the rebuild alone. Unlike the selector, a baseline that
        rebuilds never attempted a nudge first, so there is no wasted nudge
        to carry.
        """
        rng = np.random.default_rng(self.seed)
        _, _, X_holdout, y_holdout = temporal_split(X_window, y_window, self.holdout_frac)
        acc_before = self._score(model, X_holdout, y_holdout)

        fresh, cost = self.rebuild_mechanism.apply(None, X_window, y_window, rng)
        acc_after = self._score(fresh, X_holdout, y_holdout)

        return AdaptResult(
            model=fresh,
            action=REBUILD,
            cost=cost,
            acc_before=acc_before,
            acc_nudged=None,
            acc_after=acc_after,
            floor=math.nan,
            n_train_rows=len(X_window),
            n_holdout_rows=len(X_holdout),
        )


class NeverAdapt(_FixedActionPolicy):
    """Ignore every alarm. The lower bound on cost: zero work units, always.

    Takes the mechanisms anyway so all five policies share one constructor
    shape; it never calls them.
    """

    name = "NeverAdapt"

    def adapt(self, model, X_window, y_window, reference_accuracy) -> AdaptResult:
        return self._skip(model, X_window, y_window)


class AlwaysRebuild(_FixedActionPolicy):
    """Rebuild from scratch on every alarm. Current practice -- the thing to beat."""

    name = "AlwaysRebuild"

    def adapt(self, model, X_window, y_window, reference_accuracy) -> AdaptResult:
        return self._rebuild(model, X_window, y_window)


class AlwaysNudge(_FixedActionPolicy):
    """Nudge on every alarm, never rebuild. The upper bound on savings.

    Expected to lose accuracy under sustained drift, since nothing ever resets
    the model. Measuring how much is the point.
    """

    name = "AlwaysNudge"

    def adapt(self, model, X_window, y_window, reference_accuracy) -> AdaptResult:
        return self._nudge(model, X_window, y_window)


class FixedSchedule(_FixedActionPolicy):
    """Nudge, but rebuild on every k-th alarm. The timer competitor.

    Alarms are counted from 1, so with k=3 the sequence is
    NUDGE, NUDGE, REBUILD, NUDGE, NUDGE, REBUILD, ... The schedule is fixed in
    advance and never looks at accuracy -- that is exactly the contrast with
    the selector. k=1 rebuilds every time and is equivalent to `AlwaysRebuild`.

    Stateful: build one instance per run.
    """

    def __init__(self, nudge_mechanism, rebuild_mechanism, k: int = 3, **kwargs):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        super().__init__(nudge_mechanism, rebuild_mechanism, **kwargs)
        self.k = k
        self.n_alarms = 0

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"FixedSchedule(k={self.k})"

    def adapt(self, model, X_window, y_window, reference_accuracy) -> AdaptResult:
        self.n_alarms += 1
        if self.n_alarms % self.k == 0:
            return self._rebuild(model, X_window, y_window)
        return self._nudge(model, X_window, y_window)


#: Keys accepted by `make_policy`, in the spec's order.
POLICY_KEYS = (
    "never_adapt",
    "always_rebuild",
    "always_nudge",
    "fixed_schedule",
    "mechanism_selector",
)


def make_policy(
    key: str,
    spec: ModelSpec,
    seed: int = 0,
    *,
    holdout_frac: float = 0.20,
    floor_drop: float = 0.02,
    k: int = 3,
    scorer: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
) -> Policy:
    """Build a fresh policy for one run, wired to identical mechanisms.

    Every policy gets its own `NudgeMechanism` and `RebuildMechanism` built from
    the same spec and seed, and the same `holdout_frac` and `scorer`, so the
    runner cannot accidentally give one policy different actions from another.
    `floor_drop` applies only to the selector; `k` only to `FixedSchedule`.
    """
    nudge = NudgeMechanism(spec, seed=seed)
    rebuild = RebuildMechanism(spec, seed=seed)
    common = dict(holdout_frac=holdout_frac, scorer=scorer, seed=seed)

    if key == "never_adapt":
        return NeverAdapt(nudge, rebuild, **common)
    if key == "always_rebuild":
        return AlwaysRebuild(nudge, rebuild, **common)
    if key == "always_nudge":
        return AlwaysNudge(nudge, rebuild, **common)
    if key == "fixed_schedule":
        return FixedSchedule(nudge, rebuild, k=k, **common)
    if key == "mechanism_selector":
        return MechanismSelector(nudge, rebuild, floor_drop=floor_drop, **common)
    raise ValueError(f"Unknown policy {key!r}; expected one of {list(POLICY_KEYS)}")
