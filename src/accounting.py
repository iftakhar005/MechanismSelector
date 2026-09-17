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

Tree ensembles accumulate under `warm_start` / `xgb_model`, so those are
measured as a delta against a baseline captured before the fit.

`n_iter_` is per-call rather than cumulative for both SGD and MLP on
scikit-learn 1.9.0: five consecutive `partial_fit` calls on the same model
report ``n_iter_ = 1`` every time, including with ``warm_start=True``, while the
loss falls monotonically -- the model carries state, only the counter resets.
This is undocumented and could change, so the pass count is not hardcoded to
either behaviour. See `resolve_iter_passes`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, NamedTuple

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
        n_placeholder_rows: zero-weight rows appended so the fit saw every
            class (see `mechanisms.anchor_missing_classes`). Audit metadata,
            like `pass_source`: never charged, never counted in
            `n_rows_processed`, recorded so results can be reported with and
            without the operations it affected.
    """

    n_estimators_fitted: int
    n_passes: int
    pass_source: str
    n_rows_processed: int
    work_units: int
    wall_clock_s: float
    energy_kwh: float | None
    n_placeholder_rows: int = 0

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
            n_placeholder_rows=self.n_placeholder_rows + other.n_placeholder_rows,
        )

    def with_energy(self, energy_kwh: float | None) -> "CostRecord":
        """Attach an externally measured energy figure. Secondary metric only."""
        return replace(self, energy_kwh=energy_kwh)


class Baseline(NamedTuple):
    """Model state captured immediately before a fit, so an incremental fit
    is charged only for the work it actually added."""

    n_estimators: int
    n_iter: int


def snapshot(model: Any) -> Baseline:
    """Capture the pre-fit baseline. Pass the result to ``measure(baseline=...)``."""
    return Baseline(n_estimators=estimator_count(model), n_iter=iter_count(model))


def iter_count(model: Any) -> int:
    """Current ``n_iter_``, or 0 if absent/unfitted."""
    if model is None:
        return 0
    n_iter = getattr(model, "n_iter_", None)
    if n_iter is None:
        return 0
    return int(max(n_iter)) if hasattr(n_iter, "__iter__") else int(n_iter)


def resolve_iter_passes(current: int, baseline: int) -> int:
    """Charge epochs correctly whether ``n_iter_`` accumulates or resets.

    scikit-learn does not document whether repeated ``partial_fit`` calls
    accumulate ``n_iter_``. On 1.9.0 SGD and MLP both reset it to 1 per call,
    but relying on that would silently misprice every nudge if a future
    version changed it -- and a plain delta would be worse than wrong: under
    reset semantics ``1 - 1 = 0`` would record nudges as free.

    The rule below is correct under both conventions:

    - counter grew  -> it accumulates, charge the difference
    - counter did not grow -> it reset, charge the value as-is
    """
    if current > baseline:
        return current - baseline
    return current


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


def count_passes(model: Any, baseline: Baseline | None = None) -> tuple[int, int, str]:
    """Derive (n_estimators_fitted, n_passes, pass_source) from a fitted model.

    Args:
        model: the model *after* fitting.
        baseline: state captured before the fit via ``snapshot``, so an
            incremental fit reports only the work it added. ``None`` means a
            fit from scratch.
    """
    base = baseline or Baseline(0, 0)

    total_estimators = estimator_count(model)
    if total_estimators > 0:
        added = max(0, total_estimators - base.n_estimators)
        return added, added, PASS_TREES

    if getattr(model, "n_iter_", None) is not None:
        passes = resolve_iter_passes(iter_count(model), base.n_iter)
        return 0, passes, PASS_N_ITER

    # Closed-form estimators (GaussianNB): exactly one traversal of the data.
    return 0, 1, PASS_SINGLE


def measure(
    fn: Callable[[], Any],
    *,
    n_rows: int,
    baseline: Baseline | None = None,
    energy_kwh: float | None = None,
) -> tuple[Any, CostRecord]:
    """Run a fitting operation and record its exact cost.

    Args:
        fn: zero-arg callable that performs the fit and returns the model.
        n_rows: training rows the operation was given.
        baseline: state captured before the fit via ``snapshot(model)``, so an
            incremental fit is charged only for what it added. ``None`` means
            a fit from scratch.
        energy_kwh: optional externally measured energy. Secondary only.

    Returns:
        ``(model, CostRecord)``.

    Note:
        The spec's signature is ``measure(fn)``; ``n_rows`` and ``baseline``
        are required because neither is recoverable from the fitted model
        alone.
    """
    start = time.perf_counter()
    model = fn()
    elapsed = time.perf_counter() - start

    n_estimators, n_passes, source = count_passes(model, baseline)

    return model, CostRecord(
        n_estimators_fitted=n_estimators,
        n_passes=n_passes,
        pass_source=source,
        n_rows_processed=int(n_rows),
        work_units=int(n_passes) * int(n_rows),
        wall_clock_s=elapsed,
        energy_kwh=energy_kwh,
    )
