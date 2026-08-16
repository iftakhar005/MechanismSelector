"""Operation accounting: the primary metric.

`work_units = n_passes * n_rows_processed` is exact, deterministic and
identical on any machine. Wall-clock and energy are captured alongside it but
must NEVER be read by decision logic -- they are hardware-dependent estimates,
reported only as secondary corroboration.

## What counts as a pass

A "pass" is one traversal of the training data. The count is taken from the
model itself after fitting, never assumed:

| Model | Passes | Source |
|---|---|---|
| `XGBClassifier` | boosting rounds added | `booster.num_boosted_rounds()` delta |
| `RandomForestClassifier` | trees added | `len(estimators_)` delta |
| `SGDClassifier` | epochs actually run | `n_iter_` |
| `MLPClassifier` | epochs actually run | `n_iter_` |
| `GaussianNB` | 1 | closed-form, single pass |

This distinction is the point of the metric. `SGDClassifier.fit` runs many
epochs (36 on a small synthetic set) while `partial_fit` runs exactly one --
a 36x cost difference. Collapsing both to "one fit call" would erase precisely
the saving this study measures.

`n_iter_` is per-call rather than cumulative for both SGD and MLP (verified
against scikit-learn 1.9.0), so it is read directly. Tree ensembles accumulate
under `warm_start` / `xgb_model`, so those are measured as a delta against a
baseline captured before the fit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

# Provenance tags for n_passes, recorded so every number is auditable.
PASS_TREES = "trees"
PASS_N_ITER = "n_iter_"
PASS_SINGLE = "single_pass"
PASS_NONE = "none"
PASS_MIXED = "mixed"


@dataclass(frozen=True)
class CostRecord:
    """Exact cost of one fitting operation.

    Attributes:
        n_estimators_fitted: trees / boosting rounds added. 0 for non-ensemble
            models, which have no estimators -- use ``n_passes`` for those.
        n_passes: raw passes over the data. The actual cost driver, and the
            multiplicand in ``work_units``. Recorded separately from
            ``n_estimators_fitted`` so the two are independently auditable.
        pass_source: where ``n_passes`` came from, for audit.
        n_rows_processed: training rows seen by the operation.
        work_units: ``n_passes * n_rows_processed``. The headline number.
        wall_clock_s: secondary. Never used in decision logic.
        energy_kwh: secondary, often None. Never used in decision logic.
    """

    n_estimators_fitted: int
    n_passes: int
    pass_source: str
    n_rows_processed: int
    work_units: int
    wall_clock_s: float
    energy_kwh: float | None

    @staticmethod
    def zero() -> "CostRecord":
        """A no-cost record, e.g. for SKIP."""
        return CostRecord(
            n_estimators_fitted=0,
            n_passes=0,
            pass_source=PASS_NONE,
            n_rows_processed=0,
            work_units=0,
            wall_clock_s=0.0,
            energy_kwh=None,
        )

    def __add__(self, other: "CostRecord") -> "CostRecord":
        """Accumulate two operations.

        Used where a REBUILD must carry the cost of the failed nudge that
        preceded it. ``work_units`` sums the operands rather than being
        recomputed, because each operation processed a different row count.
        """
        if not isinstance(other, CostRecord):
            return NotImplemented
        if self.n_passes == 0 and self.n_rows_processed == 0:
            source = other.pass_source
        elif other.n_passes == 0 and other.n_rows_processed == 0:
            source = self.pass_source
        elif self.pass_source == other.pass_source:
            source = self.pass_source
        else:
            source = PASS_MIXED
        energies = [e for e in (self.energy_kwh, other.energy_kwh) if e is not None]
        return CostRecord(
            n_estimators_fitted=self.n_estimators_fitted + other.n_estimators_fitted,
            n_passes=self.n_passes + other.n_passes,
            pass_source=source,
            n_rows_processed=self.n_rows_processed + other.n_rows_processed,
            work_units=self.work_units + other.work_units,
            wall_clock_s=self.wall_clock_s + other.wall_clock_s,
            energy_kwh=sum(energies) if energies else None,
        )

    def with_energy(self, energy_kwh: float | None) -> "CostRecord":
        """Attach an externally measured energy figure. Secondary metric only."""
        return replace(self, energy_kwh=energy_kwh)


def estimator_count(model: Any) -> int:
    """Current number of fitted estimators, or 0 if the model has none/unfitted.

    Used to snapshot a baseline before an incremental fit so that added
    capacity can be measured as a delta.
    """
    if model is None:
        return 0

    # XGBoost: ask the booster, which reflects rounds actually present.
    getter = getattr(model, "get_booster", None)
    if callable(getter):
        try:
            return int(getter().num_boosted_rounds())
        except Exception:  # noqa: BLE001 - unfitted booster
            return 0

    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        return len(estimators)

    return 0


def count_passes(model: Any, baseline_estimators: int = 0) -> tuple[int, int, str]:
    """Derive (n_estimators_fitted, n_passes, pass_source) from a fitted model.

    Args:
        model: the model *after* fitting.
        baseline_estimators: estimator count captured before the fit, so
            incremental fits report only the capacity they added.
    """
    total_estimators = estimator_count(model)
    if total_estimators > 0:
        added = max(0, total_estimators - baseline_estimators)
        return added, added, PASS_TREES

    n_iter = getattr(model, "n_iter_", None)
    if n_iter is not None:
        # MLP exposes an int; some estimators expose an array (one per target).
        passes = int(max(n_iter)) if hasattr(n_iter, "__iter__") else int(n_iter)
        return 0, passes, PASS_N_ITER

    # Closed-form estimators (GaussianNB): exactly one traversal of the data.
    return 0, 1, PASS_SINGLE


def measure(
    fn: Callable[[], Any],
    *,
    n_rows: int,
    baseline_estimators: int = 0,
    energy_kwh: float | None = None,
) -> tuple[Any, CostRecord]:
    """Run a fitting operation and record its exact cost.

    Args:
        fn: zero-arg callable that performs the fit and returns the model.
        n_rows: training rows the operation was given.
        baseline_estimators: estimator count before the fit; pass
            ``estimator_count(model)`` for incremental fits so only added
            capacity is charged.
        energy_kwh: optional externally measured energy. Secondary only.

    Returns:
        ``(model, CostRecord)``.

    Note:
        The spec's signature is ``measure(fn)``; ``n_rows`` and
        ``baseline_estimators`` are required because neither is recoverable
        from the fitted model alone.
    """
    start = time.perf_counter()
    model = fn()
    elapsed = time.perf_counter() - start

    n_estimators, n_passes, source = count_passes(model, baseline_estimators)

    return model, CostRecord(
        n_estimators_fitted=n_estimators,
        n_passes=n_passes,
        pass_source=source,
        n_rows_processed=int(n_rows),
        work_units=int(n_passes) * int(n_rows),
        wall_clock_s=elapsed,
        energy_kwh=energy_kwh,
    )
