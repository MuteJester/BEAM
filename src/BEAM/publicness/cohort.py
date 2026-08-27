"""Repertoire-depth grouping and the FFT length it implies."""
from __future__ import annotations

from typing import Any

import numpy as np




def _next_power_of_two_above(n: int) -> int:
    k = 2
    while k <= n:
        k *= 2
    return k


def _depth_groups(
    depths: Any, depth_bins: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collapse repertoire depths into (representative depth, multiplicity).

    The PGF product runs over these groups rather than over every repertoire.
    Grouping is *exact* whenever the cohort has at most ``depth_bins`` distinct
    depths; only beyond that are depths quantile-binned, with the geometric
    mean as the representative because detection probability is driven by
    log-depth over most of the range, so it distorts ``1 - (1-p)**N`` least.
    """
    d = np.asarray(depths, dtype=np.float64).reshape(-1)
    if d.size == 0:
        raise ValueError("depths must name at least one repertoire")
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("every depth must be finite and non-negative")

    unique, counts = np.unique(d, return_counts=True)
    if unique.size <= depth_bins:
        return unique, counts.astype(np.float64), 1.0

    # Zero-depth repertoires contribute pi = 0 at every p; they are held out of
    # the quantile binning so that a geometric mean never sees a zero, and are
    # carried as their own group so the repertoire count still adds up.
    n_zero = int(np.count_nonzero(d <= 0))
    positive = np.sort(d[d > 0])
    edges = np.unique(
        np.quantile(positive, np.linspace(0.0, 1.0, depth_bins + 1))
    )
    index = np.clip(np.digitize(positive, edges[1:-1]), 0, edges.size - 2)

    representatives: list[float] = []
    multiplicities: list[float] = []
    spread = 1.0
    for b in range(edges.size - 1):
        group = positive[index == b]
        if group.size == 0:
            continue
        representatives.append(float(np.exp(np.mean(np.log(group)))))
        multiplicities.append(float(group.size))
        spread = max(spread, float(group[-1] / group[0]))
    if n_zero:
        representatives.append(0.0)
        multiplicities.append(float(n_zero))

    return (
        np.asarray(representatives, dtype=np.float64),
        np.asarray(multiplicities, dtype=np.float64),
        spread,
    )
