"""Gapped repertoire fingerprints: bounded native search, exact event masses.

The returned categories partition complete sequences, not overlapping windows.
The pair-adaptive score is a dissimilarity; no triangle inequality is claimed.
All probabilities describe fitted models, not statistical confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Iterable

import numpy as np

from .. import _cbeam
from ..errors import BeamValueError
from ..mixture import Mixture
from ..model import alphabets

__all__ = ["compare", "Fingerprint", "FingerprintLeaf", "LengthFingerprint"]


@dataclass(frozen=True)
class FingerprintLeaf:
    length: int
    node: int
    masks: tuple[int, ...]
    alphabet: str
    conditional_p: float
    conditional_q: float
    p: float
    q: float

    @property
    def contribution(self) -> float:
        """This disjoint category's contribution to squared distance."""
        return 0.5 * (math.sqrt(self.p) - math.sqrt(self.q)) ** 2

    @property
    def pattern(self) -> str:
        """One token per position; bracket tokens occupy ONE position."""
        all_mask = (1 << len(self.alphabet)) - 1
        parts = []
        for mask in self.masks:
            if mask == all_mask:
                parts.append("_")
            else:
                yes = "".join(a for i, a in enumerate(self.alphabet) if mask & (1 << i))
                no = "".join(a for i, a in enumerate(self.alphabet) if not mask & (1 << i))
                parts.append(yes if len(yes) == 1 else
                             "[^" + no + "]" if len(no) < len(yes) else "[" + yes + "]")
        return "".join(parts)

    def matches(self, sequence: str) -> bool:
        if len(sequence) != self.length:
            return False
        for a, mask in zip(sequence, self.masks):
            i = self.alphabet.find(a)
            if i < 0 or not mask & (1 << i):
                return False
        return True


@dataclass(frozen=True)
class LengthFingerprint:
    length: int
    nodes: np.ndarray  # parent, yes, no, position, residue, depth
    conditional_masses: np.ndarray  # nodes x (P,Q)
    leaves: tuple[FingerprintLeaf, ...]
    squared_distance: float  # conditional on this length
    queries: int
    budget_exhausted: bool
    scratch_bytes: int
    searched: bool


@dataclass(frozen=True)
class Fingerprint:
    by_length: dict[int, LengthFingerprint]
    squared_distance: float
    length_squared_distance: float
    full_squared_distance: float | None
    config: dict

    @property
    def distance(self) -> float:
        return math.sqrt(self.squared_distance)

    @property
    def within_squared_distance(self) -> float:
        return max(0.0, self.squared_distance - self.length_squared_distance)

    @property
    def residual(self) -> float | None:
        if self.full_squared_distance is None:
            return None
        return max(0.0, self.full_squared_distance - self.squared_distance)

    @property
    def coverage(self) -> float | None:
        if self.full_squared_distance is None or self.full_squared_distance <= 1e-12:
            return None
        return min(1.0, self.squared_distance / self.full_squared_distance)

    @property
    def within_coverage(self) -> float | None:
        if self.full_squared_distance is None:
            return None
        denominator = self.full_squared_distance - self.length_squared_distance
        if denominator <= 1e-12:
            return None
        return min(1.0, self.within_squared_distance / denominator)

    @property
    def leaves(self) -> tuple[FingerprintLeaf, ...]:
        return tuple(leaf for row in self.by_length.values() for leaf in row.leaves)

    def counts(self, sequences: Iterable[str]) -> dict[tuple[int, int], int]:
        """Route each input sequence once through the frozen tree.

        Duplicates count repeatedly; pass distinct strings for presence weights.
        Reject unsupported lengths/letters instead of silently losing observations.
        """
        out = {(leaf.length, leaf.node): 0 for leaf in self.leaves}
        alphabet = self.leaves[0].alphabet
        residues = {a: i for i, a in enumerate(alphabet)}
        for sequence in sequences:
            L = len(sequence)
            if L not in self.by_length or any(a not in residues for a in sequence):
                raise BeamValueError("sequence outside the fingerprint's length/alphabet support")
            nodes = self.by_length[L].nodes
            node = 0
            while nodes[node, 1] >= 0:
                _, yes, no, position, residue, _ = nodes[node]
                node = int(yes if residues[sequence[position]] == residue else no)
            out[L, node] += 1
        return out


def _integer(value, name, minimum=0, maximum=2**32-1):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise BeamValueError(f"{name} must be an integer") from exc
    if isinstance(value, (bool, np.bool_)) or not minimum <= result <= maximum:
        raise BeamValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _leaves(L, nodes, masses, symbols, pi, qi):
    all_mask = (1 << len(symbols)) - 1
    result = []
    for i, row in enumerate(nodes):
        if row[1] >= 0:
            continue
        masks = [all_mask] * L
        at = i
        while nodes[at, 0] >= 0:
            parent = int(nodes[at, 0])
            _, yes, _, position, residue, _ = nodes[parent]
            bit = 1 << int(residue)
            masks[position] &= bit if yes == at else all_mask ^ bit
            at = parent
        p, q = map(float, masses[i])
        result.append(FingerprintLeaf(L, i, tuple(masks), symbols, p, q, pi*p, qi*q))
    return tuple(result)


def compare(a, b, *, lengths=None, max_leaves=16, max_depth=4,
            lookahead=2, beam_width=32, max_queries=256,
            min_mass=1e-5, min_gain=1e-7, compute_full=True) -> Fingerprint:
    """Search for disjoint gapped categories contrasting two repertoire laws.

    Search/cache/conditional sweeps run in C, releasing the GIL. Each length
    is processed sequentially and its cache freed before the next. ``lengths``
    restricts SEARCH only: unsearched lengths retain a root/remainder category
    in the probability law and distance. Start with ``lengths=[14]`` to price
    a pilot. Defaults are engineering budgets, not validated biological choices.

    ``max_queries`` counts pairs of conditional marginal sweeps per length.
    ``lookahead`` explores positive conjunctions of up to three questions per
    proposal; existing tree paths also include exclusions. A bounded candidate
    list can miss interactions. ``max_depth`` limits all questions on a path,
    including exclusions. ``max_leaves`` includes remainder leaves.

    The exploratory half of the beam retains candidates by positional diversity
    and support even with zero current contrast. The other half ranks gain per
    added leaf. No globally optimal motif or exhaustive pair screen is promised.

    Same alphabet, order and maximum length are currently required by the
    whole-law Mixture API. Full coverage refers to whole-sequence Hellinger,
    not the earlier local five-block metric. ``compute_full=False`` skips that
    ceiling calculation, leaving coverage/residual undefined.
    """
    max_leaves = _integer(max_leaves, "max_leaves", 1, (2**31-1)//2)
    max_depth = _integer(max_depth, "max_depth")
    lookahead = _integer(lookahead, "lookahead", 1, 3)
    beam_width = _integer(beam_width, "beam_width", 1)
    max_queries = _integer(max_queries, "max_queries", 1)
    min_mass, min_gain = float(min_mass), float(min_gain)
    if not math.isfinite(min_mass) or not 0 <= min_mass <= 1:
        raise BeamValueError("min_mass must be finite and in [0,1]")
    if not math.isfinite(min_gain) or min_gain <= 0:
        raise BeamValueError("min_gain must be finite and positive")
    mix = Mixture([a, b])  # also validates finalization/configuration agreement
    symbols = alphabets.symbols(a.alphabet)
    supported = sorted(set(a.observed_lengths) | set(b.observed_lengths))
    selected = set(supported) if lengths is None else {
        _integer(L, "length", 1) for L in lengths}
    if selected - set(supported):
        raise BeamValueError("requested search length has no support in either model")
    config = dict(lengths=sorted(selected), max_leaves=max_leaves,
                  max_depth=max_depth, lookahead=lookahead, beam_width=beam_width,
                  max_queries=max_queries, min_mass=min_mass, min_gain=min_gain)
    full = None
    if compute_full:
        log_bc = mix.log_bhattacharyya(0, 1)
        if not math.isfinite(log_bc) and log_bc != -math.inf:
            raise ArithmeticError("non-finite whole-law affinity")
        if log_bc > 1e-10:
            raise ArithmeticError("whole-law affinity exceeds one")
        full = max(0.0, -math.expm1(log_bc))
    rows = {}
    length_terms = []
    for L in supported:
        pi, qi = math.exp(a.log_length_prior(L)), math.exp(b.log_length_prior(L))
        length_terms.append(0.5 * (math.sqrt(pi) - math.sqrt(qi))**2)
        searched = L in selected and pi > 0 and qi > 0
        if searched:
            raw, probs, calls, stopped, memory, h2 = _cbeam.fingerprint_search(
                a._g, b._g, L, max_leaves, min(max_depth, L), lookahead,
                beam_width, max_queries, min_mass, min_gain)
            nodes = np.frombuffer(raw, dtype=np.int32).reshape(-1, 6)
            masses = np.frombuffer(probs, dtype=np.float64).reshape(-1, 2)
        else:
            nodes = np.array([[-1, -1, -1, -1, -1, 0]], dtype=np.int32)
            masses = np.ones((1, 2))
            nodes.flags.writeable = masses.flags.writeable = False
            calls, stopped, memory, h2 = 0, False, 0, 0.0
        leaves = _leaves(L, nodes, masses, symbols, pi, qi)
        rows[L] = LengthFingerprint(L, nodes, masses, leaves, h2,
                                    calls, bool(stopped), memory, searched)
    score = math.fsum(leaf.contribution for row in rows.values() for leaf in row.leaves)
    length_score = math.fsum(length_terms)
    if full is not None and score > full + 1e-9:
        raise ArithmeticError("fingerprint distance exceeds whole-law ceiling")
    return Fingerprint(rows, score, length_score, full, config)
