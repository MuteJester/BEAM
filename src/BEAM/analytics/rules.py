"""Exact sequence rules: which positions constrain which, and how strongly.

A **rule** is a set of ``(position, residue)`` constraints, not necessarily
contiguous, at any order from two upwards. ``__________N_QF_`` is a rule; so is
``CASS___________``. Everything here is computed exactly from the model's own
analytical layer, with no sampling and no fitting:

    :meth:`~BEAM.BeamModel.log_clamp_probability`   exact log P of a rule
    :meth:`~BEAM.BeamModel.positional_marginals`    exact conditionals given it

**Why not rank by lift over independence.** The obvious score,
``P(S) / prod_i P(r_i)``, conflates every order at once. A four-position rule
scores highly merely because one of its pairs is strong, so ranking by it
rediscovers pairs dressed up as higher-order rules. Measured on real
repertoires, the top of such a ranking is entirely three-way restatements of
two-way facts.

**So the score is the minimum conditional lift.** For each member of a rule ask
how much the REST of the rule predicts it,

.. math::
    \\mathrm{cl}_i(S) = \\frac{P(r_i \\mid S \\setminus \\{i\\})}{P(r_i)}
                      = \\frac{P(S)}{P(S \\setminus \\{i\\})\\,P(r_i)}

and score by :math:`\\min_i \\mathrm{cl}_i(S)`. A rule is only as strong as its
weakest member, so one passenger position caps the score at that passenger
however strong the rest is. That is what makes an order-``k`` rule an order-``k``
statement rather than a pair with decoration.

**Search cost is O(rules kept), not O(subsets).** One clamp on a rule returns
the conditional at every remaining position and residue simultaneously, so
extending an entire beam by one order costs one clamp per kept rule. The
expensive part is the clamped dynamic program, which is already native; what
lives here is bookkeeping.

**Lengths and anchoring.** Positions mean different things at different CDR3
lengths, so rules are extracted per length and recorded in both coordinate
systems: ``pos`` from the N terminus and ``rpos`` from the C terminus.
:func:`canonicalise` then collapses a rule seen at many lengths into one
entry, inferring whether it is start- or end-anchored from which coordinate
stays constant. The two are indistinguishable within a single length, so a
rule seen at only one length is left length-specific rather than guessed at.
Breadth, the number of lengths a rule holds at, is retained because it is
evidence rather than redundancy.

**Significance is reproducibility.** A fitted model is a deterministic function
of its training data, so an exact probability carries no error bar of its own.
:func:`reproducible` takes two dictionaries from disjoint halves of a corpus
and keeps only rules both find, with agreeing strength. On real repertoires
that filter removes about a third of candidates, and its pass rate tracks the
probability mass of the length: at modal lengths roughly nine rules in ten
survive, and past length twenty essentially none do.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import log
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..model import alphabets

__all__ = ["Rule", "CanonicalRule", "extract", "canonicalise", "reproducible"]

#: The amino-acid alphabet, in the order the analytical layer reports. Kept
#: for callers that used it; :func:`extract` reads the alphabet from the model
#: instead, since a nucleotide model reports five residues in a different
#: order and clamping an amino acid on one is refused outright.
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_INDEX = {a: i for i, a in enumerate(ALPHABET)}

#: Floor for probabilities entering a logarithm or a denominator.
_TINY = 1e-300


@dataclass(frozen=True)
class Rule:
    """One rule at one CDR3 length.

    Attributes:
        length: the CDR3 length this rule was extracted at.
        pos: constrained positions, counted from the N terminus, ascending.
        res: the residue required at each entry of ``pos``.
        p: exact probability that a sequence of this length satisfies the rule.
        min_cond_lift: the score; see the module docstring.
        lift_indep: ``P(S) / prod_i P(r_i)``, reported for comparison only.
    """

    length: int
    pos: tuple[int, ...]
    res: tuple[str, ...]
    p: float
    min_cond_lift: float
    lift_indep: float

    @property
    def order(self) -> int:
        """How many positions the rule constrains."""
        return len(self.pos)

    @property
    def rpos(self) -> tuple[int, ...]:
        """Constrained positions counted from the C terminus."""
        return tuple(self.length - 1 - p for p in self.pos)

    @property
    def pattern(self) -> str:
        """The rule as a length-``length`` string, ``_`` where unconstrained."""
        out = ["_"] * self.length
        for p, r in zip(self.pos, self.res):
            out[p] = r
        return "".join(out)

    @property
    def score(self) -> float:
        """Ranking key: probability weighted by the strength of the rule."""
        return self.p * abs(log(max(self.min_cond_lift, _TINY)))

    def key(self) -> tuple:
        """Identity of the rule within its length."""
        return self.pos, self.res


@dataclass(frozen=True)
class CanonicalRule:
    """One rule, collapsed across every length it holds at.

    Attributes:
        anchor: ``"start"``, ``"end"``, or ``"length-specific"``.
        offsets: positions in the anchored coordinate system.
        res: the residue required at each entry of ``offsets``.
        lengths: the lengths at which the rule was found, ascending.
        mass: total probability mass of those lengths, when supplied.
        min_cond_lift: per-length score.
        mcl_weighted: mass-weighted mean score.
    """

    anchor: str
    offsets: tuple[int, ...]
    res: tuple[str, ...]
    lengths: tuple[int, ...]
    mass: float
    min_cond_lift: Mapping[int, float]
    mcl_weighted: float
    example: str

    @property
    def order(self) -> int:
        return len(self.offsets)

    @property
    def breadth(self) -> int:
        """How many lengths the rule holds at. Evidence, not redundancy."""
        return len(self.lengths)

    @property
    def mcl_range(self) -> tuple[float, float]:
        v = list(self.min_cond_lift.values())
        return min(v), max(v)


def _norm(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.clip(x.sum(axis=-1, keepdims=True), _TINY, None)


def extract(
    model,
    *,
    lengths: Iterable[int] | None = None,
    max_order: int = 4,
    beam: int = 250,
    top_k: int = 200,
    min_p: float = 2e-3,
    min_marginal: float = 5e-3,
    mass_floor: float = 1e-2,
) -> dict[int, list[Rule]]:
    """Extract rules from a fitted model, by beam search over orders.

    Args:
        model: a fitted :class:`~BEAM.BeamModel`, over either alphabet. The
            residue letters in the returned rules are that model's own.
        lengths: CDR3 lengths to search. Defaults to every length carrying at
            least ``mass_floor`` of the model's length distribution.
        max_order: highest rule order to search. ``2`` gives pairs only.
        beam: rules carried from one order into the next.
        top_k: rules reported per length per order.
        min_p: a rule must occur at least this often to be kept.
        min_marginal: a residue must carry at least this much marginal
            probability at a position before it is clamped there.
        mass_floor: skip lengths carrying less than this share of the mass.
            The default of 1% is deliberate: rare lengths are the most
            expensive to search and almost nothing found there reproduces.

    Returns:
        ``{length: [Rule, ...]}``, each list sorted by descending score.

    Raises:
        ValueError: if ``max_order`` is below 2.
    """
    if max_order < 2:
        raise ValueError(f"max_order must be at least 2, got {max_order}")

    # The model's own alphabet, not the amino-acid one. Residue indices mean
    # different letters under each, so a hard-coded alphabet turns a
    # nucleotide model's `G` into a `D` and the clamp is then refused.
    syms = alphabets.symbols(model.alphabet)
    index = {a: i for i, a in enumerate(syms)}

    pmf = np.asarray(model.length_pmf(), dtype=float)
    pmf = pmf / max(pmf.sum(), _TINY)
    if lengths is None:
        lengths = [int(L) for L in range(len(pmf)) if pmf[L] >= mass_floor]

    out: dict[int, list[Rule]] = {}
    for L in sorted(int(x) for x in lengths):
        if L < 3:
            continue
        m = _norm(model.positional_marginals(L))
        # Every rule evaluated is retained so that the denominators
        # P(S \ {i}) are dictionary lookups rather than extra clamps.
        seen: dict[frozenset, tuple[tuple, float]] = {}

        frontier: list[tuple[tuple, float]] = []
        for i in range(L):
            for ai in np.where(m[i] >= min_marginal)[0]:
                entry = (((i, syms[ai]),), float(m[i][ai]))
                seen[frozenset(entry[0])] = entry
                frontier.append(entry)

        found: list[Rule] = []
        for _order in range(2, max_order + 1):
            grown: list[tuple[tuple, float]] = []
            for pairs, p_parent in frontier:
                clamp = {p: r for p, r in pairs}
                cond = _norm(model.positional_marginals(L, clamp=clamp))
                for j in range(L):
                    if j in clamp:
                        continue
                    for bj in np.where(cond[j] >= min_marginal)[0]:
                        p = p_parent * float(cond[j][bj])
                        if p < min_p:
                            continue
                        nxt = tuple(sorted(pairs + ((j, syms[bj]),)))
                        k = frozenset(nxt)
                        if k in seen:
                            continue
                        seen[k] = (nxt, p)
                        grown.append(seen[k])

            scored: list[tuple[Rule, tuple, float]] = []
            for pairs, p in grown:
                lifts = []
                for member in pairs:
                    rest = frozenset(x for x in pairs if x != member)
                    sub = seen.get(rest)
                    if sub is None:          # pruned before we needed it
                        lifts = []
                        break
                    lifts.append(p / max(sub[1] * m[member[0]][index[member[1]]],
                                         _TINY))
                if not lifts:
                    continue
                indep = float(np.prod([m[p_][index[r_]] for p_, r_ in pairs]))
                rule = Rule(length=L,
                            pos=tuple(p_ for p_, _ in pairs),
                            res=tuple(r_ for _, r_ in pairs),
                            p=float(p),
                            min_cond_lift=float(min(lifts)),
                            lift_indep=float(p / max(indep, _TINY)))
                scored.append((rule, pairs, p))

            scored.sort(key=lambda t: -t[0].score)
            found.extend(r for r, _, _ in scored[:top_k])
            frontier = [(pairs, p) for _, pairs, p in scored[:beam]]
            if not frontier:
                break

        found.sort(key=lambda r: -r.score)
        out[L] = found
    return out


def reproducible(
    a: Mapping[int, Sequence[Rule]],
    b: Mapping[int, Sequence[Rule]],
    *,
    tol: float = 0.25,
) -> dict[int, list[Rule]]:
    """Keep rules that two independently fitted models agree on.

    Args:
        a, b: rule dictionaries from models fitted on disjoint data.
        tol: maximum allowed absolute difference in ``log(min_cond_lift)``.

    Returns:
        ``{length: [Rule, ...]}`` holding the entries from ``a`` that survive.
    """
    out: dict[int, list[Rule]] = {}
    for L in sorted(set(a) & set(b)):
        other = {r.key(): r for r in b[L]}
        keep = []
        for r in a[L]:
            o = other.get(r.key())
            if o is None:
                continue
            if abs(log(max(r.min_cond_lift, _TINY))
                   - log(max(o.min_cond_lift, _TINY))) > tol:
                continue
            keep.append(r)
        out[L] = keep
    return out


def canonicalise(
    rules: Mapping[int, Sequence[Rule]],
    length_mass: Mapping[int, float] | None = None,
) -> list[CanonicalRule]:
    """Collapse a per-length dictionary into one entry per rule.

    A rule recurs at many lengths: the ``GYT`` skeleton of the TRBJ1-2 motif
    is found at every length from ten to nineteen. Left uncollapsed the
    dictionary is inflated roughly threefold and a repertoire comparison would
    count that rule ten times.

    The recurrence is kept as ``breadth`` rather than discarded, because a rule
    holding at ten lengths is a stronger statement than one holding at one, and
    its strength genuinely varies across them.

    Args:
        rules: ``{length: [Rule, ...]}``, typically the output of
            :func:`reproducible`.
        length_mass: ``{length: probability}``, used to weight the summary
            score. Without it every length weighs equally.

    Returns:
        A list of :class:`CanonicalRule`, sorted by breadth then score.
    """
    mass = dict(length_mass or {})
    by_start: dict[tuple, list[Rule]] = defaultdict(list)
    by_end: dict[tuple, list[Rule]] = defaultdict(list)
    for L, rs in rules.items():
        for r in rs:
            by_start[(r.pos, r.res)].append(r)
            by_end[(r.rpos, r.res)].append(r)

    claimed: set[tuple[int, str]] = set()
    out: list[CanonicalRule] = []
    # End-anchored first: on TCRB the J-side family is by far the largest, so
    # resolving it first stops a J rule being mis-assigned to the V frame.
    for source, anchor in ((by_end, "end"), (by_start, "start")):
        for (offsets, res), hits in sorted(source.items(),
                                           key=lambda kv: -len(kv[1])):
            hits = [r for r in hits if (r.length, r.pattern) not in claimed]
            if len(hits) < 2:
                continue
            for r in hits:
                claimed.add((r.length, r.pattern))
            mcl = {r.length: r.min_cond_lift for r in hits}
            w = {L: mass.get(L, 1.0) for L in mcl}
            tw = sum(w.values()) or 1.0
            out.append(CanonicalRule(
                anchor=anchor, offsets=offsets, res=res,
                lengths=tuple(sorted(mcl)),
                mass=float(sum(mass.get(L, 0.0) for L in mcl)),
                min_cond_lift=mcl,
                mcl_weighted=float(sum(mcl[L] * w[L] for L in mcl) / tw),
                example=hits[0].pattern))

    for L, rs in rules.items():
        for r in rs:
            if (r.length, r.pattern) in claimed:
                continue
            out.append(CanonicalRule(
                anchor="length-specific", offsets=r.pos, res=r.res,
                lengths=(L,), mass=float(mass.get(L, 0.0)),
                min_cond_lift={L: r.min_cond_lift},
                mcl_weighted=r.min_cond_lift, example=r.pattern))

    out.sort(key=lambda r: (-r.breadth, -r.mcl_weighted))
    return out
