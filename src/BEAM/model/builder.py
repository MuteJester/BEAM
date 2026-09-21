"""Builder: incremental construction from streamed sequences.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from .facade import BeamModel

from collections.abc import Iterable, Sequence
from typing import Any


from .config import _config_args
from .validation import _check_int

from .. import _cbeam as _c
from .._simulation_result import SimulationResult


def _as_weights(abundances):
    """Weights for `observe`: a float64 buffer untouched, anything else listed.

    A contiguous float64 array goes to C as bytes. Rebuilding it as Python
    floats first cost four passes over the abundance column of a parquet
    corpus -- 926,909,993 objects created and destroyed to move numbers that
    were already contiguous. Lists, tuples and generators take the original
    path, which is what most callers pass.
    """
    if abundances is None:
        return None
    dt = getattr(abundances, "dtype", None)
    if dt is not None and getattr(dt, "name", None) == "float64" \
            and getattr(abundances, "flags", None) is not None \
            and abundances.flags.c_contiguous:
        return abundances
    try:
        mv = memoryview(abundances)
    except TypeError:
        pass
    else:
        if mv.format == "d" and mv.c_contiguous:
            return abundances
    return [float(a) for a in abundances]


class Builder:
    """Accumulate BeamModel counts across batches, then finalize once.

    The counterpart of building an :class:`BeamModel` in one call, for a corpus
    that does not fit in memory or a model that has to be measured while it is
    still being fed. Counts are additive and ``finalize`` recomputes the tables
    from scratch, so the model this produces is **byte-identical** to one built
    in a single call over the concatenated batches — a property the tests
    check rather than assume.

    Example:
        >>> b = BeamModel.builder(order=2, max_length=10)
        >>> b.observe(["CASSF", "CASSY"], [3.0, 1.0])
        >>> b.observe(["CASSW"])
        >>> model = b.build()
    """

    __slots__ = ("_g", "_cargs", "_spent")

    def __init__(
        self,
        *,
        alphabet: str = "aa",
        order: int | None = None,
        weights: tuple[float, float] = (0.5, 0.5),
        max_length: int | None = None,
        strict: bool = True,
        sparse: bool | None = None,
    ) -> None:
        self._cargs = _config_args(
            alphabet=alphabet, order=order, max_length=max_length,
            weights=weights, strict=strict, sparse=sparse)
        self._g = _c.new(*self._cargs)
        self._spent = False

    def __repr__(self) -> str:
        state = "spent" if self._spent else f"observed={self.n_observed:,}"
        return f"Builder(order={self._cargs[0]}, {state})"

    def _live(self) -> Any:
        if self._spent:
            raise RuntimeError(
                "this builder has already been build()'d; its counts were "
                "handed to the BeamModel it returned. Start a new builder, or use "
                "snapshot() instead of build() to keep feeding it.")
        return self._g

    def observe(self, sequences: Iterable[str],
                abundances: Iterable[float] | None = None) -> Builder:
        """Accumulate one batch. Returns ``self`` so calls chain.

        Does NOT finalize: a streamed corpus pays for one table build at the
        end rather than one per batch.
        """
        seqs = list(sequences)
        ab = _as_weights(abundances)
        if ab is not None and len(ab) != len(seqs):
            raise ValueError(
                f"abundances has {len(ab)} entries but sequences has "
                f"{len(seqs)}")
        _c.observe(self._live(), seqs, ab)
        return self

    @property
    def n_observed(self) -> int:
        """Sequences accepted so far."""
        return int(_c.n_observed(self._live()))

    @property
    def n_skipped(self) -> int:
        """Sequences rejected so far as too long or outside the alphabet."""
        return int(_c.n_skipped(self._live()))

    @property
    def total_weight(self) -> float:
        """Total abundance accumulated so far."""
        return float(_c.total_weight(self._live()))

    @property
    def n_nonzero_cells(self) -> int:
        """Cells carrying a non-zero count so far."""
        return int(_c.n_nonzero_cells(self._live()))

    @property
    def memory_bytes(self) -> int:
        """Approximate resident bytes of the tables accumulated so far."""
        return int(_c.memory_bytes(self._live()))

    def build(self) -> BeamModel:
        """Finalize and hand the counts over. The builder is spent afterwards.

        Raises:
            ValueError: if nothing usable was ever observed, which is what a
                failed finalize means here.
        """
        # Local: the builder hands back a BeamModel and BeamModel.builder()
        # hands back a Builder, so a top-level import is a cycle.
        from .facade import BeamModel
        g = self._live()
        _c.finalize(g)
        self._spent = True
        return BeamModel._from_capsule(g)

    def snapshot(self) -> BeamModel:
        """A finalized COPY of the counts so far, without consuming them.

        The builder keeps accumulating afterwards, and the snapshot is
        independent of everything that happens next. This is what a
        convergence curve needs: ingest, measure, keep ingesting, with the
        measured model unaffected by the ingestion that follows it.

        Costs a full table build and a second copy of the counts, so take
        snapshots at the checkpoints you mean to measure, not per batch.
        """
        # Local: the builder hands back a BeamModel and BeamModel.builder()
        # hands back a Builder, so a top-level import is a cycle.
        from .facade import BeamModel
        dst = _c.new(*self._cargs)
        _c.add(dst, self._live())
        return BeamModel._from_capsule(dst)
