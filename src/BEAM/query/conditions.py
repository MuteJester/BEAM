"""Conditions: the events a query conditions on, or asks the probability of.

A condition denotes a set of sequences of one length. It has two tiers and the
split is a cost ladder, not a convenience.

**Per-position residue masks** are free. They delete edges from the trellis and
add no states, so a condition built only from :meth:`Condition.at`,
:meth:`Condition.oneof`, :meth:`Condition.exclude` and :meth:`Condition.run`
costs exactly what an unconstrained query costs.

**An automaton** is needed for anything relating two positions to each other --
a motif anywhere, a bounded count, a disjunction, a complement -- and the query
then runs on a product graph costing a factor of its state count. The
distinction is visible in :meth:`Condition.is_free`.

Composition follows the tiers. ``&`` intersects masks where it can, which is
free. ``|`` and ``~`` cannot: the union of two per-position restrictions is not
a per-position restriction, so both operands are lifted to automata first.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from ..model import alphabets
from .errors import UnsupportedQuery

__all__ = ["Condition"]


class Condition:
    """A set of sequences of one length, built by narrowing.

    Every builder returns a NEW condition rather than mutating this one, so a
    partly built condition can be shared and extended in several directions.

    Args:
        length: the sequence length this condition is expressed over.
        alphabet: ``"aa"`` or ``"nt"``. Residues are named by their letters
            throughout; the indices the C layer uses never surface.
    """

    __slots__ = ("_length", "_alphabet", "_syms", "_index", "_masks", "_spec")

    def __init__(self, length: int, alphabet: str = "aa") -> None:
        if length < 1:
            raise UnsupportedQuery(f"length must be at least 1, got {length}")
        self._length = int(length)
        self._alphabet = alphabet
        self._syms = alphabets.symbols(alphabet)
        self._index = {a: i for i, a in enumerate(self._syms)}
        self._masks = [(1 << len(self._syms)) - 1] * self._length
        self._spec: tuple | None = None

    # ── inspection ──────────────────────────────────────────────

    @property
    def length(self) -> int:
        return self._length

    @property
    def alphabet(self) -> str:
        return self._alphabet

    @property
    def is_free(self) -> bool:
        """True when nothing has been narrowed, so this selects everything."""
        full = (1 << len(self._syms)) - 1
        return self._spec is None and all(m == full for m in self._masks)

    @property
    def needs_product(self) -> bool:
        """True when answering this costs a product graph rather than a sweep."""
        return self._spec is not None

    def __repr__(self) -> str:
        return (f"Condition(length={self._length}, alphabet={self._alphabet!r}, "
                f"pattern={self.pattern()!r}, "
                f"automaton={'yes' if self._spec else 'no'})")

    def pattern(self) -> str:
        """The mask tier as a string: a letter where pinned, else ``.`` or ``_``.

        ``_`` is an unnarrowed position, a letter is one pinned to it, and
        ``.`` is one narrowed to several residues. The automaton tier does not
        appear, having no per-position picture.
        """
        full = (1 << len(self._syms)) - 1
        out = []
        for m in self._masks:
            if m == full:
                out.append("_")
            elif m and (m & (m - 1)) == 0:
                out.append(self._syms[m.bit_length() - 1])
            else:
                out.append(".")
        return "".join(out)

    # ── the free tier ───────────────────────────────────────────

    def _clone(self) -> "Condition":
        c = Condition.__new__(Condition)
        c._length, c._alphabet = self._length, self._alphabet
        c._syms, c._index = self._syms, self._index
        c._masks, c._spec = list(self._masks), self._spec
        return c

    def _bit(self, residue: str) -> int:
        try:
            return 1 << self._index[residue]
        except KeyError:
            raise UnsupportedQuery(
                f"{residue!r} is not one of the {self._alphabet!r} residues "
                f"{self._syms!r}") from None

    def _check_pos(self, position: int) -> int:
        if not 0 <= position < self._length:
            raise UnsupportedQuery(
                f"position {position} is outside 0..{self._length - 1}, the "
                f"positions of a length-{self._length} condition")
        return position

    def at(self, position: int, residue: str) -> "Condition":
        """Require `residue` at `position`."""
        c = self._clone()
        c._masks[self._check_pos(position)] &= self._bit(residue)
        return c

    def oneof(self, position: int, residues: Iterable[str]) -> "Condition":
        """Require one of `residues` at `position`."""
        keep = 0
        for r in residues:
            keep |= self._bit(r)
        c = self._clone()
        c._masks[self._check_pos(position)] &= keep
        return c

    def exclude(self, position: int, residues: Iterable[str] | str) -> "Condition":
        """Forbid `residues` at `position`."""
        drop = 0
        for r in ([residues] if isinstance(residues, str) else residues):
            drop |= self._bit(r)
        c = self._clone()
        c._masks[self._check_pos(position)] &= ~drop
        return c

    def run(self, start: int, residues: str) -> "Condition":
        """Require the residues starting at `start` to be exactly `residues`.

        One primitive rather than separate prefix and suffix builders: a prefix
        is ``start=0`` and a suffix is ``start=length - len(residues)``.
        """
        if start < 0 or start + len(residues) > self._length:
            raise UnsupportedQuery(
                f"a run of {len(residues)} from position {start} runs past the "
                f"end of a length-{self._length} condition")
        c = self._clone()
        for k, r in enumerate(residues):
            c._masks[start + k] &= self._bit(r)
        return c

    # ── the automaton tier ──────────────────────────────────────

    def contains(self, motif: str) -> "Condition":
        """Require `motif` to occur somewhere, at any position."""
        if not motif:
            raise UnsupportedQuery(
                "an empty motif is contained in everything; omit the call")
        if len(motif) > self._length:
            raise UnsupportedQuery(
                f"a motif of {len(motif)} residues cannot occur in a sequence "
                f"of {self._length}")
        spec = ("motif", tuple(self._index_of(r) for r in motif))
        return self._with_spec(spec)

    def count(self, residues: Iterable[str], lo: int, hi: int) -> "Condition":
        """Require between `lo` and `hi` positions to hold one of `residues`."""
        if lo > hi:
            raise UnsupportedQuery(f"the count range {lo}..{hi} is empty")
        of = 0
        for r in residues:
            of |= self._bit(r)
        return self._with_spec(("count", of, int(lo), int(hi)))

    def _index_of(self, residue: str) -> int:
        try:
            return self._index[residue]
        except KeyError:
            raise UnsupportedQuery(
                f"{residue!r} is not one of the {self._alphabet!r} residues "
                f"{self._syms!r}") from None

    def _with_spec(self, spec: tuple) -> "Condition":
        c = self._clone()
        c._spec = spec if c._spec is None else ("and", c._spec, spec)
        return c

    def _as_spec(self) -> tuple:
        """This whole condition as one automaton description.

        Used by the operations masks cannot express in place. A condition that
        narrows nothing lifts to a counter satisfied by EVERYTHING -- zero
        residues drawn from the empty set, required to be zero -- so that
        complementing it gives nothing, which is what the complement of
        everything is.
        """
        full = (1 << len(self._syms)) - 1
        parts = []
        if any(m != full for m in self._masks):
            parts.append(("masks", tuple(self._masks)))
        if self._spec is not None:
            parts.append(self._spec)
        if not parts:
            return ("count", 0, 0, 0)      # the empty set counted zero times
        spec = parts[0]
        for p in parts[1:]:
            spec = ("and", spec, p)
        return spec

    def _compatible(self, other: "Condition") -> None:
        if self._length != other._length or self._alphabet != other._alphabet:
            raise UnsupportedQuery(
                f"cannot combine a length-{self._length} {self._alphabet!r} "
                f"condition with a length-{other._length} {other._alphabet!r} one")

    # ── boolean algebra ─────────────────────────────────────────

    def __and__(self, other: "Condition") -> "Condition":
        """Both. Masks intersect for free; automata compose on the product."""
        self._compatible(other)
        c = self._clone()
        c._masks = [a & b for a, b in zip(self._masks, other._masks)]
        if other._spec is not None:
            c._spec = other._spec if c._spec is None \
                else ("and", c._spec, other._spec)
        return c

    def __or__(self, other: "Condition") -> "Condition":
        """Either. Both sides lift to automata: a union of per-position
        restrictions is not a per-position restriction."""
        self._compatible(other)
        c = Condition(self._length, self._alphabet)
        c._spec = ("or", self._as_spec(), other._as_spec())
        return c

    def __invert__(self) -> "Condition":
        """Everything this does not select. Lifts, for the same reason."""
        c = Condition(self._length, self._alphabet)
        c._spec = ("not", self._as_spec())
        return c

    # ── what the bindings need ──────────────────────────────────

    def _payload(self) -> tuple[int, list[int], tuple | None]:
        return self._length, self._masks, self._spec
