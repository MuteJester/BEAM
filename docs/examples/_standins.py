"""Stand-in repertoires for the documentation's examples. NOT real data.

Real labelled repertoires (cell types, conditions, cohorts) cannot be shipped
with the documentation, so the group examples use stand-ins whose differences
are built in and known. Each population is a reference model's distribution, changed in one stated
way, and sampled exactly:

    "A"  the reference population itself;
    "B"  tilted towards longer CDR3s: each extra residue x1.6 as likely;
    "C"  enriched for the junction motif "PG": x8 as likely when present.

A stand-in donor is `population(reference, kind, cells, seed)`: that many
independent cells of one population. Sequences repeat only as often as
independent draws from the population repeat them. `expand` then gives clones
heavy-tailed sizes, as clonal expansion does.
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


def population(reference, kind: str, cells: int, seed: int) -> list[str]:
    """`cells` independent cells of stand-in population `kind`, from `reference`.

    Exact for each kind, with no resampling: B draws each cell's length from the
    tilted length distribution and then a sequence of that length; C draws
    motif carriers and other sequences in the enriched proportion.
    """
    rng = np.random.default_rng(seed)
    if kind == "A":
        return list(reference.sample(cells, seed=seed))
    if kind == "B":
        dist = reference.length_distribution
        lengths = sorted(dist)
        w = np.array([dist[L] * np.exp(LENGTH_TILT * (L - 14)) for L in lengths])
        counts = rng.multinomial(cells, w / w.sum())
        out = []
        for L, n in zip(lengths, counts):
            if n:
                out += list(reference.sample(int(n), length=L, seed=seed * 1000 + L))
        return [out[i] for i in rng.permutation(len(out))]
    if kind == "C":
        draws = list(reference.sample(8 * cells, seed=seed))
        carriers = [s for s in draws if has_motif(s)]
        others = [s for s in draws if not has_motif(s)]
        b = len(carriers) / len(draws)
        n_carriers = int(rng.binomial(cells, MOTIF_BOOST * b / (MOTIF_BOOST * b + 1 - b)))
        if n_carriers > len(carriers):
            raise RuntimeError("too few motif carriers drawn; increase the draw")
        out = carriers[:n_carriers] + others[:cells - n_carriers]
        return [out[i] for i in rng.permutation(len(out))]
    raise ValueError(f"unknown stand-in population {kind!r}; use one of {KINDS}")


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
