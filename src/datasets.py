"""Data layer: load drift streams as arrays in original temporal order.

Critical invariant: never shuffle. Temporal order is the entire point of the
study -- any reordering destroys the drift the experiment is built to measure.
Capping a stream therefore takes a temporal *prefix*, never a random sample.
"""

from __future__ import annotations

import numpy as np

# Loader names verified against river 0.25.0 on 2026-08-16 (spec Task 3.1).
# The spec's table listed `river.datasets.Covtype()`, which does not exist in
# river 0.25.0; covtype is sourced from scikit-learn instead. See README.
STREAMS = (
    "elec2",
    "insects_abrupt",
    "insects_gradual",
    "insects_incremental",
    "covtype",
)

_INSECTS_VARIANTS = {
    "insects_abrupt": "abrupt_balanced",
    "insects_gradual": "gradual_balanced",
    "insects_incremental": "incremental_balanced",
}

COVTYPE_DEFAULT_CAP = 100_000


def _river_to_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Materialise a river stream into arrays, preserving yield order.

    River yields ``(dict, label)`` pairs. Dict iteration order is insertion
    order, but we do not trust it to be stable across records -- a single
    reordered record would silently scramble columns. Instead we fix the
    column order from the first record and assert every later record matches.
    """
    feature_names: list[str] | None = None
    rows: list[list[float]] = []
    raw_labels: list = []

    for x, y in dataset:
        if feature_names is None:
            feature_names = list(x.keys())
        elif list(x.keys()) != feature_names:
            raise ValueError(
                f"Inconsistent feature keys at row {len(rows)}: "
                f"expected {feature_names}, got {list(x.keys())}"
            )
        rows.append([float(x[k]) for k in feature_names])
        raw_labels.append(y)

    if feature_names is None:
        raise ValueError("Stream yielded no records")

    X = np.asarray(rows, dtype=np.float64)
    y = _encode_labels(raw_labels)
    return X, y


def _encode_labels(raw_labels: list) -> np.ndarray:
    """Map labels to contiguous ints 0..K-1 via a deterministic sorted mapping.

    Sorted (not first-seen) so the same stream always yields the same integer
    for the same class regardless of where the run starts or how it is capped.
    """
    classes = sorted(set(raw_labels))
    lookup = {c: i for i, c in enumerate(classes)}
    return np.asarray([lookup[v] for v in raw_labels], dtype=np.int64)


def _load_covtype() -> tuple[np.ndarray, np.ndarray]:
    """Covtype via scikit-learn; river 0.25.0 has no Covtype loader.

    ``shuffle=False`` is the default but is passed explicitly because shuffling
    here would silently invalidate every downstream result.
    """
    from sklearn.datasets import fetch_covtype

    bunch = fetch_covtype(shuffle=False, download_if_missing=True)
    X = np.asarray(bunch.data, dtype=np.float64)
    y = _encode_labels(list(bunch.target))
    return X, y


def load_stream(name: str, cap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return X, y as arrays in original temporal order. Never shuffle.

    Args:
        name: one of ``STREAMS``.
        cap: keep at most this many rows, taken as a temporal prefix. Covtype
            defaults to 100,000 per the spec; pass an explicit value to override.
    """
    if name not in STREAMS:
        raise ValueError(f"Unknown stream {name!r}; expected one of {list(STREAMS)}")

    if name == "elec2":
        from river.datasets import Elec2

        X, y = _river_to_arrays(Elec2())
    elif name in _INSECTS_VARIANTS:
        from river.datasets import Insects

        X, y = _river_to_arrays(Insects(variant=_INSECTS_VARIANTS[name]))
    else:
        X, y = _load_covtype()
        if cap is None:
            cap = COVTYPE_DEFAULT_CAP

    if cap is not None and cap < len(X):
        X, y = X[:cap], y[:cap]  # temporal prefix, never a sample

    return X, y


def describe(name: str, cap: int | None = None) -> dict:
    """Load a stream and summarise it. Used by the Phase 1 acceptance check."""
    X, y = load_stream(name, cap=cap)
    classes, counts = np.unique(y, return_counts=True)
    return {
        "name": name,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(classes.size),
        "class_balance": {int(c): int(n) for c, n in zip(classes, counts)},
        "dtype": str(X.dtype),
        "has_nan": bool(np.isnan(X).any()),
    }
