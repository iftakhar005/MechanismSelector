"""MechanismSelector: the project's contribution.

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
   rebuild's, because that honesty is what makes the economic argument
   credible (spec trap #5).

## The reference-accuracy problem

`reference_accuracy` is an input, not something this module maintains. After a
REBUILD the model has trained on the entire window, so no clean holdout
remains inside it to certify a new baseline -- that would require scoring on
data the model has already seen. The caller (the runner, from Phase 6) is
responsible for re-measuring `reference_accuracy` on unseen future rows after
`adapt()` returns. See `PHASE4_PROMPT.md` Section 5.

## Guard condition

With very small windows a holdout of a handful of rows makes accuracy noise,
not signal -- one misclassification can swing it past the floor in either
direction. Below `min_holdout_rows` / `min_train_rows` this module does not
trust *any* decision computed from that holdout, so it bypasses both the SKIP
and NUDGE checks and goes straight to REBUILD (`degraded_to_rebuild=True`).
`acc_before` is still recorded (from the small holdout) for audit purposes,
but it is not used to decide anything in this branch.

## REBUILD's `acc_after` is optimistic

A REBUILD trains on the full window, holdout included, then is scored on that
same holdout -- data it just trained on. `acc_after` for a REBUILD is
therefore not comparable to `acc_before` / `acc_nudged`, which are always
scored on data the model being evaluated has never trained on. This is
correct (the rebuild is the final answer and should use all available data)
but must not be treated as an apples-to-apples number by the analysis.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score

from accounting import CostRecord

SKIP = "SKIP"
NUDGE = "NUDGE"
REBUILD = "REBUILD"


@dataclass
class AdaptResult:
    """Everything the runner needs to log one adaptation decision.

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


class MechanismSelector:
    """Cost-ordered try-and-escalate policy under an accuracy constraint.

    Args:
        model_factory: zero-arg callable building a fresh, unfitted model with
            identical hyperparameters and no carried state. Used for REBUILD --
            never `copy.deepcopy` of the live model, which could carry stale
            fitted state forward.
        nudge_mechanism: a `mechanisms.py` Mechanism applied to a *copy* of the
            current model, trained on `train_part` only.
        rebuild_mechanism: a `mechanisms.py` Mechanism applied to a fresh model
            from `model_factory`, trained on the full window.
        holdout_frac: fraction of the window held out as the most recent slice.
        floor_drop: how far below `reference_accuracy` is still acceptable.
            In the units of `scorer` -- e.g. 0.02 means 2 percentage points
            for accuracy, 2 points of balanced accuracy if that scorer is used.
        min_holdout_rows / min_train_rows: below these, the holdout is too
            small to trust; see the guard condition in the module docstring.
        scorer: `(y_true, y_pred) -> float`. Default plain accuracy. Pass
            `balanced_accuracy_score` for the imbalanced multi-class streams.
        seed: every random operation is derived from this, so the same
            (window, model, reference_accuracy, seed) reproduces exactly.
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        nudge_mechanism: Any,
        rebuild_mechanism: Any,
        holdout_frac: float = 0.20,
        floor_drop: float = 0.02,
        min_holdout_rows: int = 50,
        min_train_rows: int = 100,
        scorer: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
        seed: int = 0,
    ):
        self.model_factory = model_factory
        self.nudge_mechanism = nudge_mechanism
        self.rebuild_mechanism = rebuild_mechanism
        self.holdout_frac = holdout_frac
        self.floor_drop = floor_drop
        self.min_holdout_rows = min_holdout_rows
        self.min_train_rows = min_train_rows
        self.scorer = scorer
        self.seed = seed

    def _score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        return float(self.scorer(y, model.predict(X)))

    def _split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Temporal split: train_part is the older prefix, holdout the recent
        suffix. Never randomised -- a shuffle here would leak future rows into
        training and invalidate every downstream result."""
        k = int(len(X) * (1 - self.holdout_frac))
        return X[:k], y[:k], X[k:], y[k:]

    def _is_degraded(self, n: int) -> bool:
        return (
            n * self.holdout_frac < self.min_holdout_rows
            or n * (1 - self.holdout_frac) < self.min_train_rows
        )

    def adapt(
        self,
        model: Any,
        X_window: np.ndarray,
        y_window: np.ndarray,
        reference_accuracy: float,
    ) -> AdaptResult:
        floor = reference_accuracy - self.floor_drop
        n = len(X_window)
        rng = np.random.default_rng(self.seed)

        X_train, y_train, X_holdout, y_holdout = self._split(X_window, y_window)
        n_train_rows, n_holdout_rows = len(X_train), len(X_holdout)

        if self._is_degraded(n):
            # Holdout too small to trust for any decision (SKIP or NUDGE) --
            # go straight to REBUILD. acc_before is recorded for audit only.
            acc_before = self._score(model, X_holdout, y_holdout)
            fresh = self.model_factory()
            fresh, rebuild_cost = self.rebuild_mechanism.apply(
                fresh, X_window, y_window, rng
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

        acc_before = self._score(model, X_holdout, y_holdout)

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
        acc_nudged = self._score(candidate, X_holdout, y_holdout)

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
            )

        fresh = self.model_factory()
        fresh, rebuild_cost = self.rebuild_mechanism.apply(
            fresh, X_window, y_window, rng
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
        )
