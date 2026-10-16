"""Stand-in repertoires for the documentation's examples. NOT real data.

Real labelled repertoires (cell types, conditions, cohorts) cannot be shipped
with the documentation, so the group examples use stand-ins whose differences
are built in and known. Each population is drawn from a reference model and
then reweighted:

    "A"  the reference population itself;
    "B"  tilted towards longer CDR3s (each extra residue x1.6 as likely);
    "C"  enriched for the junction motif "PG" (x8 as likely when present).

A stand-in donor is `population(reference, kind, cells, seed)`: that many
cells of one population, drawn with replacement, so common sequences repeat
as they do in a real sample. `expand` then gives clones heavy-tailed sizes, as
clonal expansion does.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

MOTIF = "PG"
LENGTH_TILT = np.log(1.6)        # per residue above 14, for population B
MOTIF_BOOST = 8.0                # for population C
KINDS = ("A", "B", "C")


def has_motif(sequence: str) -> bool:
    """Whether the junction (all but the first and last three residues) carries MOTIF."""
    return MOTIF in sequence[3:-3]


def _weights(pool: list[str], kind: str) -> np.ndarray:
    if kind == "A":
        return np.ones(len(pool))
    if kind == "B":
        return np.exp(LENGTH_TILT * (np.array([len(s) for s in pool]) - 14))
    if kind == "C":
        return np.where([has_motif(s) for s in pool], MOTIF_BOOST, 1.0)
    raise ValueError(f"unknown stand-in population {kind!r}; use one of {KINDS}")


def population(reference, kind: str, cells: int, seed: int, *, pool_factor: int = 4) -> list[str]:
    """`cells` cells of stand-in population `kind`, drawn from `reference`."""
    pool = list(reference.sample(pool_factor * cells, seed=seed))
    w = _weights(pool, kind)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=cells, replace=True, p=w / w.sum())
    return [pool[i] for i in idx]


def expand(cells: list[str], *, seed: int, strength: float) -> tuple[list[str], list[int]]:
    """Distinct sequences of `cells` and their clone sizes after expansion.

    Each clone's sampled count is multiplied by ceil(exp(strength * |Z|)), Z a
    standard normal variate: strength 0 leaves the counts as sampled; at
    strength 2.5 the largest clone holds a few percent of the cells and the ten
    largest about a seventh, as in an expanded repertoire.
    """
    counts = Counter(cells)
    seqs = sorted(counts)
    if strength == 0:
        return seqs, [counts[s] for s in seqs]
    rng = np.random.default_rng(seed)
    factor = np.ceil(np.exp(strength * np.abs(rng.standard_normal(len(seqs))))).astype(int)
    return seqs, [int(counts[s] * f) for s, f in zip(seqs, factor)]
