"""Local-window repertoire distance and its additive explanation.

The squared distance separates differences in length probabilities from
differences in conditional window probabilities. Every supported length and
window enters the calculation, even when only the strongest pattern terms
are returned. This is the local-window metric, not whole-sequence Hellinger.

Overlap groups are a deterministic display heuristic over the returned
terms. Their scores sum existing coordinates of the metric; they are neither
disjoint sequence events nor contributions of the complete consensus event.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Any

import numpy as np

from .. import _cbeam
from ..errors import BeamTypeError, BeamValueError, InternalError, ModelMismatch
from ..model import alphabets

__all__ = [
    "compare", "group_overlaps", "DistanceComparison", "LengthContribution",
    "WindowContribution", "PatternContribution", "ModelSnapshot",
    "PatternGroup", "GroupedDistanceComparison", "PatternProbabilities",
]


@dataclass(frozen=True)
class ModelSnapshot:
    """Immutable configuration/count summary, not a model-content fingerprint."""

    alphabet: str
    order: int
    max_length: int
    n_observed: int
    total_weight: float
    config: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class LengthContribution:
    """Length probabilities and both contributions at this length."""

    length: int
    p: float
    q: float
    contribution: float
    pattern_contribution: float

    @property
    def squared_distance(self) -> float:
        return self.contribution + self.pattern_contribution


@dataclass(frozen=True)
class WindowContribution:
    """Contribution of all words at one zero-based window start."""

    length: int
    start: int
    width: int
    contribution: float

    @property
    def end(self) -> int:
        """Exclusive end, following Python slice notation."""
        return self.start + self.width


@dataclass(frozen=True)
class PatternContribution:
    """One word's contribution, with probabilities conditional on length.

    ``direction`` compares conditional probabilities, not length-weighted
    probabilities. ``start`` is zero-based and ``end`` is exclusive.
    """

    length: int
    start: int
    word: str
    conditional_p: float
    conditional_q: float
    contribution: float

    @property
    def width(self) -> int:
        return len(self.word)

    @property
    def end(self) -> int:
        return self.start + self.width

    @property
    def delta(self) -> float:
        return self.conditional_p - self.conditional_q

    @property
    def direction(self) -> str:
        """``'p'``, ``'q'`` or ``'equal'``, according to conditional probability."""
        return "p" if self.delta > 0 else "q" if self.delta < 0 else "equal"


@dataclass(frozen=True)
class DistanceComparison:
    """Exact distance, complete length/window totals, and bounded top terms.

    ``length_squared_distance + reported_pattern_squared_distance + remainder``
    reconstructs ``squared_distance`` to floating-point precision. ``remainder``
    is solely the contribution of omitted pattern coordinates, not a new event.
    """

    alphabet: str
    order: int
    width: int
    top_k: int
    squared_distance: float
    length_squared_distance: float
    pattern_squared_distance: float
    reported_pattern_squared_distance: float
    remainder: float
    scratch_bytes: int
    lengths: tuple[LengthContribution, ...]
    windows: tuple[WindowContribution, ...]
    patterns: tuple[PatternContribution, ...]
    model_p: ModelSnapshot | None = None
    model_q: ModelSnapshot | None = None

    @property
    def distance(self) -> float:
        return math.sqrt(self.squared_distance)

    @property
    def coverage(self) -> float | None:
        """Fraction of the pattern component returned; None if it is zero."""
        if self.pattern_squared_distance == 0:
            return None
        return min(1.0, self.reported_pattern_squared_distance /
                   self.pattern_squared_distance)

    @property
    def total_coverage(self) -> float | None:
        """Fraction explained by length plus returned patterns; None at zero."""
        if self.squared_distance == 0:
            return None
        return min(1.0, (self.length_squared_distance +
                         self.reported_pattern_squared_distance) /
                   self.squared_distance)

    def group_overlaps(self, *, min_overlap: int | None = None
                       ) -> GroupedDistanceComparison:
        """Group the returned coordinates without changing the distance."""
        return group_overlaps(self, min_overlap=min_overlap)


@dataclass(frozen=True)
class PatternProbabilities:
    """Complete-consensus event probabilities in explicitly supplied models.

    ``conditional_p`` and ``conditional_q`` condition on the specified length;
    either is None when that model has no probability mass at this length.
    ``p`` and ``q`` include the length probabilities and are then zero.
    These probabilities carry no distance contribution.
    """

    conditional_p: float | None
    conditional_q: float | None
    p: float
    q: float
    length_p: float
    length_q: float


@dataclass(frozen=True)
class PatternGroup:
    """Compatible overlapping coordinates, not an inferred biological family.

    ``consensus`` constrains the contiguous span ``[start:end]``. Its complete
    event can be queried separately, but the group's ``contribution`` always
    sums its original members, including receptors matching only some members.
    ``direction`` describes the members' conditional contrasts. The complete
    consensus event can have a different direction; query it directly.
    """

    length: int
    start: int
    consensus: str
    direction: str
    members: tuple[PatternContribution, ...]
    alphabet: str
    order: int

    @property
    def width(self) -> int:
        return len(self.consensus)

    @property
    def end(self) -> int:
        return self.start + self.width

    @property
    def contribution(self) -> float:
        """Sum of original window-word terms, not a complete-event score."""
        return math.fsum(member.contribution for member in self.members)

    def probabilities(self, a, b) -> PatternProbabilities:
        """Query the complete consensus in the two explicitly supplied models.

        Their alphabet/order must match the explanation. This method evaluates
        the supplied models as they are now; it does not verify that their
        parameters are unchanged since comparison or that they are its original
        pair. Configuration/count summaries are not model-content hashes.
        Queries are exact, performed on demand, and do not alter attribution.
        """
        ca, cb = _check_models(a, b)
        if ca["alphabet"] != self.alphabet or ca["order"] != self.order:
            raise ModelMismatch("query models must match the group's alphabet and order")
        from ..query import BeamQuery, Condition

        condition = Condition(self.length, self.alphabet).run(self.start, self.consensus)

        def evaluate(model, config):
            if self.length > config["max_length"]:
                return None, 0.0, 0.0
            prior = math.exp(model.log_length_prior(self.length))
            if prior == 0:
                return None, 0.0, 0.0
            conditional = BeamQuery(model).probability(condition)
            return conditional, prior * conditional, prior

        p, joint_p, length_p = evaluate(a, ca)
        q, joint_q, length_q = evaluate(b, cb)
        return PatternProbabilities(p, q, joint_p, joint_q, length_p, length_q)


@dataclass(frozen=True)
class GroupedDistanceComparison:
    """A partition of returned attribution coordinates, not of sequences."""

    comparison: DistanceComparison
    groups: tuple[PatternGroup, ...]
    min_overlap: int

    @property
    def distance(self) -> float:
        return self.comparison.distance

    @property
    def squared_distance(self) -> float:
        return self.comparison.squared_distance

    @property
    def length_squared_distance(self) -> float:
        return self.comparison.length_squared_distance

    @property
    def pattern_squared_distance(self) -> float:
        return self.comparison.pattern_squared_distance

    @property
    def reported_pattern_squared_distance(self) -> float:
        return math.fsum(group.contribution for group in self.groups)

    @property
    def remainder(self) -> float:
        return self.comparison.remainder

    @property
    def coverage(self) -> float | None:
        return self.comparison.coverage

    @property
    def total_coverage(self) -> float | None:
        return self.comparison.total_coverage


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise BeamTypeError(f"{name} must be an integer, not a boolean")
    try:
        result = operator.index(value)
    except TypeError:
        raise BeamTypeError(f"{name} must be an integer") from None
    if not minimum <= result <= maximum:
        raise BeamValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _check_models(a, b):
    from ..model.facade import BeamModel

    for name, model in (("a", a), ("b", b)):
        if not isinstance(model, BeamModel):
            raise BeamTypeError(f"{name} must be a BeamModel, got {type(model).__name__}")
    ca, cb = a.config, b.config
    if ca["alphabet"] != cb["alphabet"] or ca["order"] != cb["order"]:
        raise ModelMismatch("distance requires models with the same alphabet and order")
    return ca, cb


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _snapshot(model, config) -> ModelSnapshot:
    return ModelSnapshot(config["alphabet"], config["order"], config["max_length"],
                         model.n_observed, model.total_weight, _freeze(config))


def _decode_word(index: int, width: int, symbols: str) -> str:
    letters = [""] * width
    for i in range(width - 1, -1, -1):
        index, residue = divmod(index, len(symbols))
        letters[i] = symbols[residue]
    return "".join(letters)


def compare(a, b, *, width: int | None = None,
            top_k: int = 100) -> DistanceComparison:
    r"""Compute the exact local-window distance and its strongest coordinates.

    Models must have the same alphabet and context order; their maximum lengths
    may differ. The default window width is ``order + 1``; explicit widths from
    one through ``order + 1`` define corresponding local-window distances.
    For a shorter sequence, the window covers the whole sequence.
    Widths smaller than ``order + 1`` can miss dependencies, so different
    complete sequence laws can then have zero local-window distance.

    For length probabilities ``pi, qi``, conditional word probabilities ``p, q``
    and ``n`` windows at that length, each word contributes
    ``sqrt(pi*qi) * (sqrt(p)-sqrt(q))**2 / (2*n)``. The length term is
    ``(sqrt(pi)-sqrt(qi))**2 / 2``. All these nonnegative terms sum to ``D**2``.

    The native implementation releases the GIL, processes all supported lengths
    and windows, and retains at most ``top_k`` strictly positive word terms.
    ``top_k=0`` still computes the complete distance and length/window totals.
    Terms sort by decreasing contribution, then length, start and word in the
    model alphabet's order. Model inputs and caller-owned results are untouched.
    Omitted terms are reported explicitly as ``remainder``. Grouping and
    complete-consensus probability queries are optional separate operations.
    """
    ca, cb = _check_models(a, b)
    order = int(ca["order"])
    requested_width = 0 if width is None else _integer(width, "width", 1, order + 1)
    top_k = _integer(top_k, "top_k", 0, 2**32 - 1)
    raw = _cbeam.local_distance(a._g, b._g, requested_width, top_k)
    symbols = alphabets.symbols(ca["alphabet"])
    lengths = tuple(LengthContribution(*row) for row in raw["lengths"])
    windows = tuple(WindowContribution(*row) for row in raw["windows"])
    patterns = tuple(PatternContribution(length, start,
                                        _decode_word(index, actual_width, symbols),
                                        p, q, contribution)
                     for length, start, actual_width, index, p, q, contribution
                     in raw["patterns"])
    values = {name: float(raw[name]) for name in
              ("squared_distance", "length_squared_distance", "pattern_squared_distance",
               "reported_pattern_squared_distance", "remainder")}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise InternalError("native distance returned a non-finite or negative contribution")
    return DistanceComparison(ca["alphabet"], order, int(raw["width"]), top_k,
                              **values, scratch_bytes=int(raw["scratch_bytes"]),
                              lengths=lengths, windows=windows, patterns=patterns,
                              model_p=_snapshot(a, ca), model_q=_snapshot(b, cb))


def group_overlaps(comparison: DistanceComparison, *,
                   min_overlap: int | None = None) -> GroupedDistanceComparison:
    """Conservatively group compatible returned windows, strongest first.

    A term may join a group only at the same sequence length, in the same
    conditional enrichment direction, with at least ``min_overlap`` residues
    shared with a member and with no residue conflict anywhere in the full
    consensus. The default overlap is ``max(1, width - 1)``. Width-one terms
    consequently remain separate. Opposite directions and contradictory
    branches stay separate even when a chain of other windows links them.

    At each step, assignment prefers the eligible group with the largest
    accumulated contribution, then earliest length/start/consensus in alphabet
    order. A bridging window also merges other eligible groups when their full
    consensus remains consistent; conflicting branches stay separate.
    This is a deterministic greedy heuristic, not a unique or optimal motif
    reconstruction. Changing ``top_k`` or ``min_overlap`` can change grouping.
    No biological independence, enrichment test or clonotype identity is
    inferred. Each returned coordinate occurs exactly once; the unreturned
    remainder remains separate. Groups may match some of the same sequences.
    """
    if not isinstance(comparison, DistanceComparison):
        raise BeamTypeError("comparison must be a DistanceComparison")
    width = _integer(comparison.width, "comparison.width", 1, 2**32 - 1)
    overlap = (max(1, width - 1) if min_overlap is None else
               _integer(min_overlap, "min_overlap", 1, width))
    symbols = alphabets.symbols(comparison.alphabet)
    indices = {letter: index for index, letter in enumerate(symbols)}

    totals = (comparison.squared_distance, comparison.length_squared_distance,
              comparison.pattern_squared_distance,
              comparison.reported_pattern_squared_distance, comparison.remainder)
    if any(not math.isfinite(value) or value < 0 for value in totals):
        raise BeamValueError("comparison totals must be finite and nonnegative")
    if (not math.isclose(comparison.squared_distance,
                         comparison.length_squared_distance + comparison.pattern_squared_distance,
                         rel_tol=1e-10, abs_tol=1e-14) or
            not math.isclose(comparison.pattern_squared_distance,
                             comparison.reported_pattern_squared_distance + comparison.remainder,
                             rel_tol=1e-10, abs_tol=1e-14)):
        raise BeamValueError("comparison components must reconstruct its squared distance")

    def word_key(word):
        return tuple(indices[letter] for letter in word)

    seen = set()
    for pattern in comparison.patterns:
        if not isinstance(pattern, PatternContribution):
            raise BeamTypeError("comparison.patterns must contain PatternContribution objects")
        if (not pattern.word or any(letter not in indices for letter in pattern.word) or
                pattern.start < 0 or pattern.end > pattern.length or
                pattern.width != min(width, pattern.length)):
            raise BeamValueError("pattern lies outside its length/alphabet/window support")
        if (not math.isfinite(pattern.contribution) or pattern.contribution < 0 or
                any(not math.isfinite(p) or not 0 <= p <= 1 + 1e-12
                    for p in (pattern.conditional_p, pattern.conditional_q))):
            raise BeamValueError("pattern probabilities and contribution must be finite and valid")
        key = pattern.length, pattern.start, pattern.word
        if key in seen:
            raise BeamValueError("each pattern coordinate must occur only once")
        seen.add(key)

    if not math.isclose(math.fsum(p.contribution for p in comparison.patterns),
                        comparison.reported_pattern_squared_distance,
                        rel_tol=1e-10, abs_tol=1e-14):
        raise BeamValueError("pattern contributions must sum to the reported pattern total")

    def pattern_key(pattern):
        return (-pattern.contribution, pattern.length, pattern.start, word_key(pattern.word))

    patterns = sorted(comparison.patterns, key=pattern_key)
    groups: list[PatternGroup] = []

    def group_key(group):
        return (-group.contribution, group.length, group.start, word_key(group.consensus))

    def compatible(group, pattern):
        if group.length != pattern.length or group.direction != pattern.direction:
            return False
        if not any(min(member.end, pattern.end) - max(member.start, pattern.start) >= overlap
                   for member in group.members):
            return False
        lo, hi = max(group.start, pattern.start), min(group.end, pattern.end)
        return (group.consensus[lo - group.start:hi - group.start] ==
                pattern.word[lo - pattern.start:hi - pattern.start])

    def combine(group, members):
        joined = tuple(sorted(group.members + members, key=pattern_key))
        start = min(member.start for member in joined)
        end = max(member.end for member in joined)
        residues = [""] * (end - start)
        for member in joined:
            residues[member.start - start:member.end - start] = member.word
        return PatternGroup(group.length, start, "".join(residues), group.direction,
                            joined, group.alphabet, group.order)

    for pattern in patterns:
        candidates = [i for i, group in enumerate(groups) if compatible(group, pattern)]
        if not candidates:
            groups.append(PatternGroup(pattern.length, pattern.start, pattern.word,
                                       pattern.direction, (pattern,), comparison.alphabet,
                                       comparison.order))
            continue
        candidates.sort(key=lambda i: group_key(groups[i]))
        at = candidates[0]
        combined = combine(groups[at], (pattern,))
        consumed = {at}
        for i in candidates[1:]:
            other = groups[i]
            lo, hi = max(combined.start, other.start), min(combined.end, other.end)
            if (combined.consensus[lo - combined.start:hi - combined.start] ==
                    other.consensus[lo - other.start:hi - other.start]):
                combined = combine(combined, other.members)
                consumed.add(i)
        groups = [group for i, group in enumerate(groups) if i not in consumed]
        groups.append(combined)

    groups.sort(key=group_key)
    return GroupedDistanceComparison(comparison, tuple(groups), overlap)
