"""Per-oracle normalization — the engineering lynchpin of the disagreement penalty.

Heterogeneous oracles are not comparable in raw units: Boltz-2's affinity (log-pIC50-like)
and AutoDock-Vina's energy (kcal/mol) live on different scales even after the sign is
unified. Computing the cross-oracle disagreement ``sigma_a`` on *raw* scores would measure
unit mismatch, not genuine disagreement. We therefore map each oracle's scores onto a
common ``[0, 1]`` (rank) or zero-mean/unit-variance (z-score) scale *within the population*
**before** any cross-oracle statistic is taken. This module is intentionally numpy-only.
"""

from __future__ import annotations

import numpy as np

_NORMALIZERS = ("rank", "zscore")


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map ``values`` to ``[0, 1]`` by average rank (ties share the mean rank).

    Rank-normalization is scale- and monotone-transform-invariant, which is exactly what we
    want when fusing oracles whose units/calibration differ but whose *ordering* is the
    meaningful signal. A population of size 1 maps to ``[0.5]`` (no information).
    """
    arr = np.asarray(values, dtype=float).ravel()
    n = arr.size
    if n == 0:
        raise ValueError("rank_normalize requires a non-empty array")
    if n == 1:
        return np.array([0.5])
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # average ranks for ties so equal scores get equal normalized values
    _average_ties(arr, ranks)
    return ranks / (n - 1)


def _average_ties(arr: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    i = 0
    n = arr.size
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            mean_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = mean_rank
        i = j


def zscore_normalize(values: np.ndarray) -> np.ndarray:
    """Map ``values`` to zero mean and unit variance. Constant input maps to all-zeros."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("zscore_normalize requires a non-empty array")
    std = arr.std()
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def normalize(values: np.ndarray, method: str = "rank") -> np.ndarray:
    """Dispatch to :func:`rank_normalize` (default) or :func:`zscore_normalize`."""
    if method == "rank":
        return rank_normalize(values)
    if method == "zscore":
        return zscore_normalize(values)
    raise ValueError(f"unknown normalize method {method!r}; choose from {_NORMALIZERS}")
