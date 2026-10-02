"""Synthetic drift streams with known change points (confirm-before-acting test).

Every stream is 50,000 rows with abrupt concept changes at rows 12,500, 25,000
and 37,500 (or none), generated offline by river's synthetic generators. See
docs/confirm/PREREGISTRATION.md for why each stream exists.

**Sticky sampling** adds temporal dependence without changing any concept.
Within a segment, rows are drawn from the generator into per-class pools; the
class sequence is a Markov chain that keeps the previous class with probability
`rho` and otherwise draws from the segment's empirical class prior; the next
row of that class is taken from its pool in generation order. `P(y | x)` is
untouched, so on a `none` stream every alarm is false with respect to the
concept.
"""

from __future__ import annotations

import numpy as np

N_ROWS = 50_000
CHANGE_POINTS = (12_500, 25_000, 37_500)
RHO = 0.8

STREAMS = (
    "sea_abrupt_iid",
    "sea_abrupt_sticky",
    "sea_none_sticky",
    "agrawal_abrupt_iid",
    "agrawal_abrupt_sticky",
    "agrawal_none_sticky",
)

_SEA_VARIANTS = (0, 2, 1, 3)
_AGRAWAL_FUNCTIONS = (0, 2, 4, 6)


def _generator(family: str, concept: int, seed: int):
    from river.datasets import synth

    if family == "sea":
        return synth.SEA(variant=concept, noise=0.1, seed=seed)
    if family == "agrawal":
        return synth.Agrawal(classification_function=concept, perturbation=0.05, seed=seed)
    raise ValueError(family)


def _draw(family: str, concept: int, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    names, X, y = None, [], []
    for x, label in _generator(family, concept, seed).take(n):
        if names is None:
            names = list(x)
        X.append([float(x[k]) for k in names])
        y.append(int(label))
    return np.asarray(X), np.asarray(y, dtype=np.int64)


def _sticky(X: np.ndarray, y: np.ndarray, n: int, rho: float, rng: np.random.Generator):
    classes, counts = np.unique(y, return_counts=True)
    prior = counts / counts.sum()
    pools = {c: list(np.flatnonzero(y == c)) for c in classes}
    pos = {c: 0 for c in classes}
    out, current = [], rng.choice(classes, p=prior)
    for _ in range(n):
        if rng.random() >= rho:
            current = rng.choice(classes, p=prior)
        if pos[current] >= len(pools[current]):   # pool exhausted: fall back to any class with rows left
            current = next(c for c in classes if pos[c] < len(pools[c]))
        out.append(pools[current][pos[current]])
        pos[current] += 1
    idx = np.asarray(out)
    return X[idx], y[idx]


def make_stream(name: str, seed: int, n_rows: int = N_ROWS) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Return ``X, y, change_points`` for a named synthetic stream."""
    if name not in STREAMS:
        raise ValueError(f"unknown synthetic stream {name!r}")
    family, drift, sampling = name.split("_")
    concepts = (_SEA_VARIANTS if family == "sea" else _AGRAWAL_FUNCTIONS)
    if drift == "none":
        concepts = (concepts[0],) * len(concepts)
        change_points: tuple[int, ...] = ()
    else:
        change_points = tuple(int(c * n_rows / N_ROWS) for c in CHANGE_POINTS)

    bounds = (0, *(int(c * n_rows / N_ROWS) for c in CHANGE_POINTS), n_rows)
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for i, concept in enumerate(concepts):
        seg = bounds[i + 1] - bounds[i]
        gen_seed = seed * 100 + i       # distinct draws per segment even when the concept repeats
        if sampling == "iid":
            Xi, yi = _draw(family, concept, gen_seed, seg)
        else:
            Xp, yp = _draw(family, concept, gen_seed, 4 * seg)
            Xi, yi = _sticky(Xp, yp, seg, RHO, rng)
        Xs.append(Xi)
        ys.append(yi)
    return np.vstack(Xs), np.concatenate(ys), change_points
