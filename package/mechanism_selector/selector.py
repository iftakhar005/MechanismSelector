"""MechanismSelector: try a cheap model update first, rebuild only if it fails.

## Framing

This is a **cost-ordered try-and-escalate policy under an accuracy
constraint**, not "selection by measured compute." The decision variable at
runtime is accuracy: a nudge is kept if it clears the floor, discarded (and
escalated to a rebuild) if it does not. Cost only determines the *order* in
which options are attempted -- cheapest first -- and is what makes attempting
the cheap option rational in the first place: a nudge costs a few percent of a
rebuild, so trying and occasionally failing is cheaper than building a
predictor of when it would succeed.

## Procedure (see `MechanismSelector.adapt`)

1. Split the adaptation window by time -- train_part (older) / holdout (most
   recent) -- never randomly.
2. Score the current model on holdout. Good enough -> SKIP, zero cost.
3. Otherwise nudge a *copy* of the model on train_part only, score the nudge
   on the same holdout. Cleared the floor -> NUDGE, keep it.
4. Otherwise rebuild from scratch on the full window (holdout included) and
   use that. The failed nudge's cost is not discarded -- it is added to the
   rebuild's, so the reported cost of a REBUILD is the true cost of trying
   the cheap option first.

## The reference-accuracy problem

`reference_accuracy` is an input, not something this module maintains. After a
REBUILD the model has trained on the entire window, so no clean holdout
remains inside it to certify a new baseline -- that would require scoring on
data the model has already seen.

The caller therefore owns the baseline. The rule used in the experiments this
code was tested with, after every `adapt()` call:

1. Score the *returned* model on the next N rows the stream has not yet
   produced -- rows no model has trained on -- and use that as the
   `reference_accuracy` for the next alarm. Default N = 200.
2. Those N rows are not consumed: they continue through the normal stream loop
   afterwards (predicted, counted in prequential accuracy, buffered) like any
   other row. Skipping them would remove them from prequential accuracy.
3. If fewer than N rows remain, carry the previous `reference_accuracy`
   forward rather than scoring on a shorter, noisier slice, and log that this
   happened.

It must never use `AdaptResult.acc_after` as the new baseline: for a REBUILD
that number is scored on training data (see below) and would inflate the floor
for every later decision.

## Guard condition

With very small windows a holdout of a handful of rows makes accuracy noise,
not signal -- one misclassification can swing it past the floor in either
direction. Below `min_holdout_rows` / `min_train_rows` this module does not
trust *any* decision computed from that holdout, so it bypasses both the SKIP
and NUDGE checks and goes straight to REBUILD (`degraded_to_rebuild=True`).
`acc_before` is still recorded (from the small holdout) for audit purposes,
but it is not used to decide anything in this branch.

The thresholds are checked against the row counts `_split` actually produces,
not against `len(window) * holdout_frac`. The two differ by rounding: a
249-row window splits into 199 train and 50 holdout rows, while
`249 * 0.2 = 49.8` would wrongly report it as under the 50-row minimum.

**This guard dominates in practice.** A caller that clears its buffer after each
adaptation will often call `adapt()` on small windows, and every such call
rebuilds unconditionally. In the benchmarks this code was evaluated on, half of
all rebuilds came from this guard rather than from a failed nudge.

## Rebuild construction

The rebuild mechanism owns model construction; this module never builds a
model itself. It passes `None` as the model argument to
`rebuild_mechanism.apply`, because there is nothing a rebuild should inherit.
Passing the live model instead would invite a rebuild to snapshot it as a
cost baseline -- subtracting the old model's estimators from the new one's and
billing the rebuild at or near zero.

## REBUILD's `acc_after` is optimistic

A REBUILD trains on the full window, holdout included, then is scored on that
same holdout -- data it just trained on. `acc_after` for a REBUILD is
therefore not comparable to `acc_before` / `acc_nudged`, which are always
scored on data the model being evaluated has never trained on. This is
correct (the rebuild is the final answer and should use all available data)
but must not be compared directly with `acc_before` or `acc_nudged`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score

from .accounting import CostRecord

SKIP = "SKIP"
NUDGE = "NUDGE"
REBUILD = "REBUILD"


def temporal_split(
    X: np.ndarray, y: np.ndarray, holdout_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Temporal split: train_part is the older prefix, holdout the recent
    suffix. Never randomised -- a shuffle here would leak future rows into
    training and invalidate every downstream result.

    Module-level so every policy in `policies.py` splits a window exactly as
    the selector does. The baselines perform the same nudge on the same rows,
    so a comparison between policies isolates the decision rule rather than
    differences in how much data each policy trained on.
    """
    k = int(len(X) * (1 - holdout_frac))
    return X[:k], y[:k], X[k:], y[k:]


@dataclass
class AdaptResult:
    """Everything a caller needs to act on, and log, one adaptation decision.

    Attributes:
        model: the model to use going forward.
        action: "SKIP" | "NUDGE" | "REBUILD".
        cost: cumulative cost -- includes a failed nudge when action is REBUILD.
        acc_before: current model's accuracy on the holdout, before any action.
        acc_nudged: nudged model's accuracy on the holdout, or None when the
            nudge was never attempted (SKIP, or degraded_to_rebuild).
        acc_after: accuracy of the returned model. For REBUILD this is
            measured on data the model trained on -- see module docstring.
        floor: reference_accuracy - floor_drop, the threshold actually used.
        n_train_rows: rows in train_part (never seen by the holdout check).
        n_holdout_rows: rows in the holdout.
        degraded_to_rebuild: True if the window was too small to trust a
            holdout-based decision, so REBUILD was taken unconditionally.
        prediction_change: share of holdout rows whose predicted label the
            nudge changed, whenever a nudge was performed; otherwise None.
            Distinguishes a nudge that moved predictions without improving
            accuracy from one that left predictions untouched -- accuracy
            alone cannot, because flips can cancel.
    """

    model: Any
    action: str
    cost: CostRecord
    acc_before: float
    acc_nudged: float | None
    acc_after: float
    floor: float
    n_train_rows: int
    n_holdout_rows: int
    degraded_to_rebuild: bool = False
    prediction_change: float | None = None


class MechanismSelector:
    """Cost-ordered try-and-escalate policy under an accuracy constraint.

    Args:
        nudge_mechanism: a `mechanisms.py` Mechanism applied to a *copy* of the
            current model, trained on `train_part` only.
        rebuild_mechanism: a `mechanisms.py` Mechanism that constructs a fresh
            model itself and trains it on the full window. It is handed `None`
            as its model argument -- see "Rebuild construction" in the module
            docstring.
        holdout_frac: fraction of the window held out as the most recent slice.
        floor_drop: how far below `reference_accuracy` is still acceptable.
            In the units of `scorer` -- e.g. 0.02 means 2 percentage points
            for accuracy, 2 points of balanced accuracy if that scorer is used.
        min_holdout_rows / min_train_rows: below these, the holdout is too
            small to trust; see the guard condition in the module docstring.
        scorer: `(y_true, y_pred) -> float`. Default plain accuracy. Pass
            `balanced_accuracy_score` for imbalanced multi-class data.
        seed: seeds a `numpy.random.Generator` that is passed to both
            mechanisms as their `rng` argument. **The mechanisms in
            `mechanism_selector.mechanisms` ignore that argument**; their
            randomness is set by their own `seed`. With them, this parameter
            has no effect -- set `NudgeMechanism(seed=...)` and
            `RebuildMechanism(seed=...)` instead. Given identical inputs and
            mechanism seeds, `adapt()` is deterministic.
    """

    name = "MechanismSelector"

    def __init__(
        self,
        nudge_mechanism: Any,
        rebuild_mechanism: Any,
        holdout_frac: float = 0.20,
        floor_drop: float = 0.02,
        min_holdout_rows: int = 50,
        min_train_rows: int = 100,
        scorer: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
        seed: int = 0,
    ):
        self.nudge_mechanism = nudge_mechanism
        self.rebuild_mechanism = rebuild_mechanism
        self.holdout_frac = holdout_frac
        self.floor_drop = floor_drop
        self.min_holdout_rows = min_holdout_rows
        self.min_train_rows = min_train_rows
        self.scorer = scorer
        self.seed = seed

    def _score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        return self._predict_score(model, X, y)[1]

    def _predict_score(self, model: Any, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        predictions = model.predict(X)
        return predictions, float(self.scorer(y, predictions))

    def _split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Temporal split; see `temporal_split`."""
        return temporal_split(X, y, self.holdout_frac)

    def _is_degraded(self, X: np.ndarray, y: np.ndarray) -> bool:
        """True if the split this window actually produces is too small to trust.

        Checks the real row counts from `_split`, not `len(X) * holdout_frac`:
        the two disagree by rounding, and the disagreement falls exactly on
        the small windows this guard exists for.
        """
        X_train, _, X_holdout, _ = self._split(X, y)
        return (
            len(X_holdout) < self.min_holdout_rows
            or len(X_train) < self.min_train_rows
        )

    def adapt(
        self,
        model: Any,
        X_window: np.ndarray,
        y_window: np.ndarray,
        reference_accuracy: float,
    ) -> AdaptResult:
        """Decide how to respond to a drift alarm, and return the model to use.

        Splits the window by time into an older training part and a recent
        holdout (`holdout_frac`), then:

        1. **SKIP** at zero cost if the current model's holdout score is at least
           `reference_accuracy - floor_drop`.
        2. Otherwise **NUDGE**: apply `nudge_mechanism` to a copy of the model on
           the training part only, and keep it if its holdout score clears the
           same floor.
        3. Otherwise **REBUILD**: apply `rebuild_mechanism` to the full window.
           The reported cost includes the failed nudge.

        If the split leaves fewer than `min_holdout_rows` holdout rows or
        `min_train_rows` training rows, steps 1 and 2 are skipped and it rebuilds
        unconditionally (`degraded_to_rebuild=True`).

        Args:
            model: the current fitted model. Never modified; the nudge works on
                a copy.
            X_window, y_window: the recent rows, oldest first. Never shuffled.
            reference_accuracy: the model's recent accuracy, in the units of
                `scorer`, as maintained by the caller (see the module docstring).

        Returns:
            An `AdaptResult` with the model to use going forward, the action
            taken, its cost, and the accuracies it was based on.
        """
        floor = reference_accuracy - self.floor_drop
        rng = np.random.default_rng(self.seed)

        X_train, y_train, X_holdout, y_holdout = self._split(X_window, y_window)
        n_train_rows, n_holdout_rows = len(X_train), len(X_holdout)

        if self._is_degraded(X_window, y_window):
            # Holdout too small to trust for any decision (SKIP or NUDGE) --
            # go straight to REBUILD. acc_before is recorded for audit only.
            acc_before = self._score(model, X_holdout, y_holdout)
            fresh, rebuild_cost = self.rebuild_mechanism.apply(
                None, X_window, y_window, rng
            )
            acc_after = self._score(fresh, X_holdout, y_holdout)
            return AdaptResult(
                model=fresh,
                action=REBUILD,
                cost=rebuild_cost,
                acc_before=acc_before,
                acc_nudged=None,
                acc_after=acc_after,
                floor=floor,
                n_train_rows=n_train_rows,
                n_holdout_rows=n_holdout_rows,
                degraded_to_rebuild=True,
            )

        pred_before, acc_before = self._predict_score(model, X_holdout, y_holdout)

        if acc_before >= floor:
            return AdaptResult(
                model=model,
                action=SKIP,
                cost=CostRecord.zero(),
                acc_before=acc_before,
                acc_nudged=None,
                acc_after=acc_before,
                floor=floor,
                n_train_rows=n_train_rows,
                n_holdout_rows=n_holdout_rows,
            )

        candidate = copy.deepcopy(model)  # protects the caller's model on REBUILD
        candidate, nudge_cost = self.nudge_mechanism.apply(
            candidate, X_train, y_train, rng
        )
        pred_nudged, acc_nudged = self._predict_score(candidate, X_holdout, y_holdout)
        prediction_change = float(np.mean(pred_nudged != pred_before))

        if acc_nudged >= floor:
            return AdaptResult(
                model=candidate,
                action=NUDGE,
                cost=nudge_cost,
                acc_before=acc_before,
                acc_nudged=acc_nudged,
                acc_after=acc_nudged,
                floor=floor,
                n_train_rows=n_train_rows,
                n_holdout_rows=n_holdout_rows,
                prediction_change=prediction_change,
            )

        fresh, rebuild_cost = self.rebuild_mechanism.apply(
            None, X_window, y_window, rng
        )
        acc_after = self._score(fresh, X_holdout, y_holdout)

        return AdaptResult(
            model=fresh,
            action=REBUILD,
            cost=nudge_cost + rebuild_cost,  # the wasted nudge is not discarded
            acc_before=acc_before,
            acc_nudged=acc_nudged,
            acc_after=acc_after,
            floor=floor,
            n_train_rows=n_train_rows,
            n_holdout_rows=n_holdout_rows,
            prediction_change=prediction_change,
        )
