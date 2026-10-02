

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


def temporal_split(
    X: np.ndarray, y: np.ndarray, holdout_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  
    k = int(len(X) * (1 - holdout_frac))
    return X[:k], y[:k], X[k:], y[k:]


@dataclass
class AdaptResult:
   
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
