"""Mechanism library: skip / nudge / rebuild behind one interface.

## The nudge is not one operation

The spec's "add capacity equal to 5% of the model's original size" describes
the tree ensembles exactly. It does not describe the others, because they have
no capacity to add:

| Family | What a nudge actually is | Governed by |
|---|---|---|
| `XGBClassifier` | append `ceil(0.05 * original)` boosting rounds | the 5% rule |
| `RandomForestClassifier` | append `ceil(0.05 * original)` trees, drop the same number of oldest | the 5% rule |
| `SGDClassifier` | one extra epoch (`partial_fit`) | convergence behaviour |
| `MLPClassifier` | one extra epoch (`partial_fit`) | convergence behaviour |
| `GaussianNB` | update sufficient statistics (`partial_fit`) | closed form |

So a nudge is **each family's cheapest available incremental update**, not a
uniform 5% of anything. The rebuild:nudge cost ratio is therefore a measured
quantity that differs per family, not a constant implied by the 5% rule --
see `experiments/verify_mechanisms.py`, which records it.

`GaussianNB` is a deliberate negative control: its rebuild is already a single
closed-form pass, so nudge and rebuild cost the same and no saving is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from accounting import CostRecord, measure, snapshot

DEFAULT_ENSEMBLE_SIZE = 100
DEFAULT_NUDGE_FRAC = 0.05


class MechanismUnavailable(Exception):
    """Raised when a model family supports no incremental update.

    Which families admit a mechanism choice is itself a reportable finding,
    so this is a first-class outcome rather than an error to work around.
    """


@dataclass
class ModelSpec:
    """Everything a mechanism needs to know about a model family.

    Attributes:
        name: short key, e.g. ``"xgb"``.
        factory: ``seed -> fresh unfitted model`` with canonical hyperparameters.
        classes: the full class list for the stream. Required on every
            ``partial_fit`` call, or the call crashes when a window happens to
            lack a class (spec trap #3).
        original_size: ensemble size at build time. Fixed for the run: the 5%
            rule is defined against the *original* size, so repeated nudges
            add a constant amount rather than compounding.
    """

    name: str
    factory: Callable[[int], Any]
    classes: np.ndarray
    original_size: int | None = None
    supports_nudge: bool = True


class Mechanism(Protocol):
    name: str

    def apply(
        self, model: Any, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[Any, CostRecord]: ...


# --- model factories ---------------------------------------------------------


def _xgb_factory(n_estimators: int = DEFAULT_ENSEMBLE_SIZE) -> Callable[[int], Any]:
    def build(seed: int):
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.3,
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
            random_state=seed,
        )

    return build


def _rf_factory(n_estimators: int = DEFAULT_ENSEMBLE_SIZE) -> Callable[[int], Any]:
    def build(seed: int):
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=1, random_state=seed
        )

    return build


def _sgd_factory() -> Callable[[int], Any]:
    def build(seed: int):
        from sklearn.linear_model import SGDClassifier

        return SGDClassifier(max_iter=1000, tol=1e-3, random_state=seed)

    return build


def _gnb_factory() -> Callable[[int], Any]:
    def build(_seed: int):
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB()

    return build


def _mlp_factory() -> Callable[[int], Any]:
    def build(seed: int):
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=seed)

    return build


def _svc_factory() -> Callable[[int], Any]:
    def build(seed: int):
        from sklearn.svm import SVC

        return SVC(random_state=seed)

    return build


def _knn_factory() -> Callable[[int], Any]:
    def build(_seed: int):
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier()

    return build


#: Models in the experimental grid, plus MLP and the two unsupported families.
MODEL_BUILDERS: dict[str, tuple[Callable[[], Callable[[int], Any]], int | None, bool]] = {
    "xgb": (_xgb_factory, DEFAULT_ENSEMBLE_SIZE, True),
    "rf": (_rf_factory, DEFAULT_ENSEMBLE_SIZE, True),
    "sgd": (_sgd_factory, None, True),
    "gnb": (_gnb_factory, None, True),
    "mlp": (_mlp_factory, None, True),
    "svc": (_svc_factory, None, False),
    "knn": (_knn_factory, None, False),
}

GRID_MODELS = ("xgb", "rf", "sgd", "gnb")


def make_spec(name: str, classes: np.ndarray, n_estimators: int | None = None) -> ModelSpec:
    """Build a ModelSpec for a named model family."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {name!r}; expected one of {list(MODEL_BUILDERS)}")
    builder, default_size, supports_nudge = MODEL_BUILDERS[name]
    size = n_estimators if n_estimators is not None else default_size
    factory = builder(size) if default_size is not None else builder()
    return ModelSpec(
        name=name,
        factory=factory,
        classes=np.asarray(classes),
        original_size=size,
        supports_nudge=supports_nudge,
    )


def nudge_size(spec: ModelSpec, nudge_frac: float = DEFAULT_NUDGE_FRAC) -> int:
    """Capacity added by one nudge. Defined against the *original* size."""
    if spec.original_size is None:
        return 0
    return max(1, math.ceil(nudge_frac * spec.original_size))


# --- mechanisms --------------------------------------------------------------


@dataclass
class SkipMechanism:
    """Return the model untouched at zero cost. Does not look at the data."""

    name: str = "skip"

    def apply(self, model, X, y, rng) -> tuple[Any, CostRecord]:  # noqa: ARG002
        return model, CostRecord.zero()


@dataclass
class RebuildMechanism:
    """Train a fresh model of the same family and hyperparameters from scratch."""

    spec: ModelSpec
    seed: int = 0
    name: str = "rebuild"

    def apply(self, model, X, y, rng) -> tuple[Any, CostRecord]:  # noqa: ARG002
        fresh = self.spec.factory(self.seed)
        return measure(lambda: fresh.fit(X, y), n_rows=len(X))


@dataclass
class NudgeMechanism:
    """Each family's cheapest available incremental update.

    Dispatches on model family. Raises ``MechanismUnavailable`` for families
    with no incremental path, which is a recorded outcome, not a failure.
    """

    spec: ModelSpec
    nudge_frac: float = DEFAULT_NUDGE_FRAC
    seed: int = 0
    name: str = "nudge"

    def apply(self, model, X, y, rng) -> tuple[Any, CostRecord]:
        family = _family_of(model)
        if family == "xgb":
            return self._nudge_xgb(model, X, y)
        if family == "rf":
            return self._nudge_rf(model, X, y)
        if family == "partial_fit":
            return self._nudge_partial(model, X, y)
        raise MechanismUnavailable(
            f"{type(model).__name__} supports no incremental update; "
            "excluded from mechanism selection by design"
        )

    def _nudge_xgb(self, model, X, y) -> tuple[Any, CostRecord]:
        """Append boosting rounds on top of the existing booster."""
        k = nudge_size(self.spec, self.nudge_frac)
        base = snapshot(model)
        booster = model.get_booster()

        extended = self.spec.factory(self.seed)
        extended.set_params(n_estimators=k)

        return measure(
            lambda: extended.fit(X, y, xgb_model=booster),
            n_rows=len(X),
            baseline=base,
        )

    def _nudge_rf(self, model, X, y) -> tuple[Any, CostRecord]:
        """Add k trees and drop the k oldest, keeping the forest a fixed size.

        `warm_start` alone only ever *adds* trees, so under sustained drift the
        forest fills with stale trees and the nudge stops working. That is
        dilution, not adaptation -- it looks like a finding but is an
        implementation artifact. The replacement policy is a deliberate design
        decision; see README.
        """
        k = nudge_size(self.spec, self.nudge_frac)
        base = snapshot(model)

        def grow():
            model.set_params(warm_start=True)
            model.n_estimators = base.n_estimators + k
            return model.fit(X, y)

        # Cost is measured while the k new trees are still present; truncation
        # afterwards must not erase the work that was genuinely done.
        fitted, cost = measure(grow, n_rows=len(X), baseline=base)

        fitted.estimators_ = fitted.estimators_[k:]  # drop k oldest, retain k newest
        fitted.n_estimators = len(fitted.estimators_)
        return fitted, cost

    def _nudge_partial(self, model, X, y) -> tuple[Any, CostRecord]:
        """One incremental update via partial_fit.

        `classes=` is passed on every call. Omitting it crashes as soon as a
        window lacks one of the stream's classes (spec trap #3).
        """
        base = snapshot(model)
        return measure(
            lambda: model.partial_fit(X, y, classes=self.spec.classes),
            n_rows=len(X),
            baseline=base,
        )


def _family_of(model: Any) -> str:
    """Classify a model by the incremental mechanism it supports."""
    if hasattr(model, "get_booster"):
        return "xgb"
    if hasattr(model, "estimators_") and hasattr(model, "warm_start"):
        return "rf"
    if hasattr(model, "partial_fit"):
        return "partial_fit"
    return "unsupported"
