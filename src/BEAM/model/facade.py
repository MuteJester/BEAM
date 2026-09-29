"""BeamModel: the public model class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from .builder import Builder
    from ..analytics.pseq import PseqAnalysis
    from ..analytics.calibration import Calibration

import datetime as _datetime
from importlib.metadata import version as _dist_version, PackageNotFoundError
import math
import operator
import os
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..constants import (BEAM_ALPHABET, BEAM_MAX_LENGTH,
                         BEAM_MAX_MOMENT, BEAM_MAX_ORDER, BEAM_RESIDUES)
from . import alphabets
from .validation import _check_int, _is_scalar_arg
from .config import _config_args, _config_args_from_dict
from ..io.readers import _column, _iter_delimited, _iter_parquet
from ..math.saddlepoint import _normal_pdf, _normal_cdf
from ..runtime import effective_threads, memory_budget, set_threads, threads

from .. import _cbeam as _c
from ..errors import BeamTypeError, BeamValueError
from .._simulation_result import SimulationResult
from .._banded_result import BandedMarginals


def _default_column(column: str | None, config: dict) -> str:
    """The sequence column to read when the caller names none.

    `cdr3_aa` for amino acids and `cdr3` for nucleotides, which are the AIRR
    fields for a junction and its translation. Reading the wrong one is not a
    subtle failure -- an amino-acid model handed nucleotides skips every row --
    but it is a silent one until the corpus comes back empty, so the default
    follows the alphabet rather than making the caller remember.
    """
    if column is not None:
        return column
    alphabet = config.get("alphabet", alphabets.DEFAULT_ALPHABET)
    try:
        return alphabets.DEFAULT_COLUMN[alphabet]
    except KeyError:
        raise ValueError(
            f"no default column for alphabet {alphabet!r}; name one "
            f"explicitly") from None


def _orders_vector(orders: Iterable[float]) -> np.ndarray:
    """``orders`` as a fresh one-dimensional float64 array of finite values.

    Fresh on purpose, not merely contiguous: the C side reads it with the GIL
    released, and an array the caller still holds could be written by another
    thread while it does.
    """
    if isinstance(orders, (str, bytes)):
        raise BeamTypeError("orders must be a sequence of numbers, not a string")
    try:
        seq = orders if isinstance(orders, np.ndarray) else list(orders)
        q = np.array(seq, dtype=np.float64)
    except (TypeError, ValueError):
        raise BeamTypeError(
            f"orders must be a sequence of numbers, got "
            f"{type(orders).__name__}") from None
    if q.ndim != 1:
        raise BeamValueError(
            f"orders must be one-dimensional, got shape {q.shape}")
    bad = q[~np.isfinite(q)]
    if bad.size:
        raise BeamValueError(
            f"orders must be finite, got {bad[:3].tolist()}")
    return np.ascontiguousarray(q)


def _hill_from_mellin(q: float, log_mellin: float) -> float:
    """``D_q`` from ``ln sum P**q``, by the expression ``hill_number`` ends in.

    ``beam_model_hill_number`` returns 0.0 for a model with no live length,
    where ``log_mellin`` is ``-inf``; above ``q = 1`` the division would turn
    that into ``+inf`` instead, so it is special-cased the same way. C's
    ``exp`` saturates to ``inf`` where Python's raises, and a support too
    large for a double is an answer, not an error.
    """
    if log_mellin == -math.inf:
        return 0.0
    try:
        return math.exp(log_mellin / (1.0 - q))
    except OverflowError:
        return math.inf


class BeamModel:
    """Bidirectional anchored DAG over CDR3 sequences.

    The model is the same whichever alphabet it is built over -- one
    anchored table per direction, geometric mixing, exact dynamic programs.
    Only the symbols change, and with them the size of every table.

    Args:
        sequences: Training CDR3 strings. Sequences longer than ``max_length``
            or containing a character outside the model's alphabet are
            skipped; see :attr:`n_skipped`.
        abundances: Per-sequence observed counts; defaults to 1 each.
        alphabet: ``"aa"`` for the twenty amino acids, ``"nt"`` for the five
            nucleotide residues ``ACGTN``. Nucleotide input is case-folded;
            amino-acid input is not, since case carries no meaning for a base
            and folding residues would silently widen what is accepted.
            ``N`` is a residue rather than a wildcard, so a read carrying one
            is data; the other IUPAC ambiguity codes are refused.
        order: Context order, from 2 up to the alphabet's own ceiling -- 4
            for amino acids, 9 for nucleotides, because the real bound is on
            ``radix ** order``. Defaults to the alphabet's measured optimum:
            4 and 6 respectively.
        coefficients: Log-linear mixing weights for the wide, mid and
            length-conditioned experts. **They must sum to 1.** The sum is not
            a free parameter: it is a pure cross-length calibration knob, and
            moving it off 1 leaves within-length ranking untouched while
            wrecking global ranking.
        max_length: Longest sequence the model represents. Defaults to 27 for
            amino acids and 81 for nucleotides, three bases to the residue.
        alpha: Prior strengths of the hierarchical Bayesian smoothing, one
            per context level 0..order, or one number for all of them.
            Every sequence up to ``max_length`` has positive probability, and
            a row nothing was observed at predicts exactly what its shorter
            context predicts::

                P_k(x) = (n_k(x) + alpha[k] * P_{k-1}(x)) / (n_k + alpha[k])

            with level 0 the depth alone and the uniform ``1/20`` below it.
            ``alpha[k]`` is how many pseudo-observations the shorter context
            is worth. ``None`` (the default) or ``"auto"`` estimates them from
            the model's own counts at every finalize, as the strengths that
            maximise the counts' evidence -- the Dirichlet-multinomial
            marginal likelihood -- level by level from the bottom. The
            estimates are reported in :attr:`config`. Pass numbers to fix
            them, or :func:`BEAM.fit_alpha` to fit them to held-out reads.
        alpha_length: Prior strength of the length distribution,
            ``P(L) = (n_L + alpha_length / max_length) / (N + alpha_length)``,
            so a length the corpus never produced is improbable rather than
            impossible. Estimated the same way unless given.
        sparse: Store only the rows something was observed at, and resolve
            every other row through its parent. ``False`` materialises every
            row -- the same model, bit for bit, at gigabytes for order 4 --
            and exists for equivalence checks.

    Example:
        >>> model = BeamModel(sequences, abundances=counts)
        >>> model.log_pseq("CASSLGQGAEQYF")
        -13.268...
    """

    __slots__ = ("_g", "_compacted")

    def __init__(
        self,
        sequences: Iterable[str],
        *,
        abundances: Iterable[float] | None = None,
        alphabet: str = "aa",
        order: int | None = None,
        weights: tuple[float, float] = (0.5, 0.5),
        max_length: int | None = None,
        alpha: float | Iterable[float] | None = None,
        alpha_length: float | None = None,
        sparse: bool | None = None,
    ) -> None:
        seqs = list(sequences)
        if not seqs:
            raise ValueError("BeamModel requires at least one training sequence")
        abund = None
        if abundances is not None:
            abund = np.asarray(list(abundances), dtype=np.float64)
            if len(abund) != len(seqs):
                raise ValueError(
                    f"abundances has {len(abund)} entries but sequences has "
                    f"{len(seqs)}")
        cargs = _config_args(alphabet=alphabet, order=order,
                             max_length=max_length, weights=weights,
                             alpha=alpha, alpha_length=alpha_length,
                             sparse=sparse)
        max_length = cargs[1]

        try:
            self._g = _c.build(seqs, abund, *cargs)
        except ValueError as exc:
            if "configuration" in str(exc):
                raise                       # already says what is wrong
            raise ValueError(self._build_failure_reason(
                seqs, abund, max_length, alphabet, exc)) from None
        self._compacted = False

    @staticmethod
    def _build_failure_reason(seqs, abund, max_length, alphabet, exc) -> str:
        """Turn the C layer's generic refusal into something actionable.

        The C entry points do not populate the library's error-message buffer,
        so every failure arrives as a bare "invalid argument". Almost all of
        them mean one thing — nothing in the corpus survived encoding — and
        saying so is worth the one pass it costs on a corpus that has already
        failed.
        """
        usable = 0
        for i, s in enumerate(seqs):
            if not isinstance(s, str) or not s or len(s) > max_length:
                continue
            if not alphabets.accepts(alphabet, s):
                continue
            if abund is not None and not (abund[i] > 0 and np.isfinite(abund[i])):
                continue
            usable += 1
            break
        # The corpus is not always the problem. Now that the C layer's own
        # message reaches here -- an order past its ceiling, a table too large
        # to address, an allocation that failed -- blaming the sequences for it
        # would bury the real reason under a guess.
        if usable > 0:
            return str(exc)
        if usable == 0:
            syms = alphabets.symbols(alphabet)
            return (
                f"no training sequence survived encoding: every one of "
                f"{len(seqs)} is empty, longer than max_length={max_length}, "
                f"contains a character the {alphabet!r} alphabet does not "
                f"accept (its residues are {syms!r}), or carries a "
                f"non-positive abundance")
        return f"BeamModel could not be built from this corpus ({exc})"

    # ── Streaming construction ──────────────────────────────────

    @classmethod
    def builder(cls, **config: Any) -> Builder:
        """An :class:`Builder` that accumulates counts batch by batch.

        Use this when the corpus does not fit in memory, or when the model has
        to be measured while it is still being fed. Takes the same keyword
        configuration :class:`BeamModel` does, minus the sequences.

        Example:
            >>> b = BeamModel.builder(order=2, max_length=10)
            >>> b.observe(["CASSF", "CASSY"], [3.0, 1.0])
            >>> b.observe(["CASSW"])
            >>> model = b.build()
            >>> model.n_observed
            3
        """
        # Imported here rather than at module scope: this class is
        # constructed with the model, so a top-level import would be a
        # cycle.
        from .builder import Builder
        return Builder(**config)

    @classmethod
    def from_sequence_file(
        cls,
        path: str | os.PathLike,
        *,
        sep: str = "\t",
        sequence_column: int = 0,
        abundance_column: int | None = 1,
        has_header: bool = False,
        chunk_size: int = 1_000_000,
        **config: Any,
    ) -> BeamModel:
        """Train from a delimited text file without materialising it.

        The file the graph builders in this library take: one record per line,
        ``sequence<TAB>abundance``. A line with only one field is read as an
        abundance of 1, so a plain list of sequences works too.

        Args:
            sep: Field separator. ``None`` splits on any whitespace.
            sequence_column: Zero-based index of the sequence field.
            abundance_column: Index of the weight field, or ``None`` to weight
                every record 1.
            has_header: Skip the first line.
            chunk_size: Lines handed to the C layer per call.
            **config: Passed to :class:`BeamModel`.
        """
        # Local: constructed with the model, so a top-level import cycles.
        from .builder import Builder
        b = Builder(**config)
        for seqs, ab in _iter_delimited(
                path, sep=sep, sequence_column=sequence_column,
                abundance_column=abundance_column, has_header=has_header,
                chunk_size=chunk_size):
            b.observe(seqs, ab)
        return b.build()

    @classmethod
    def from_parquet(
        cls,
        path: str | os.PathLike,
        *,
        sequence_column: str | None = None,
        abundance_column: str | None = None,
        batch_size: int = 1_000_000,
        **config: Any,
    ) -> BeamModel:
        """Train straight from a parquet file, one record batch at a time.

        Requires ``pyarrow``. Nothing larger than ``batch_size`` rows is ever
        a Python object at once, so a corpus far larger than memory trains in
        constant space.

        Args:
            sequence_column: Column holding the amino-acid sequence.
            abundance_column: Column holding the weight, or ``None`` to weight
                every row 1.
            batch_size: Rows per record batch.
            **config: Passed to :class:`BeamModel`.
        """
        # Local: constructed with the model, so a top-level import cycles.
        from .builder import Builder
        sequence_column = _default_column(sequence_column, config)
        b = Builder(**config)
        for seqs, ab in _iter_parquet(path, sequence_column, abundance_column,
                                      batch_size):
            b.observe(seqs, ab)
        return b.build()

    @classmethod
    def from_dataframe(
        cls,
        data: Any,
        *,
        sequence_column: str | None = None,
        abundance_column: str | None = None,
        **config: Any,
    ) -> BeamModel:
        """Train from anything column-shaped: a DataFrame, an Arrow table, or
        a plain ``dict`` of lists.

        Duck-typed rather than isinstance-checked, so it does not drag in a
        dependency to recognise one.

        Args:
            sequence_column: Column holding the amino-acid sequence.
            abundance_column: Column holding the weight, or ``None`` for 1 each.
            **config: Passed to :class:`BeamModel`.
        """
        sequence_column = _default_column(sequence_column, config)
        seqs = _column(data, sequence_column)
        ab = None if abundance_column is None else [
            float(x) for x in _column(data, abundance_column)]
        return cls(seqs, abundances=ab, **config)

    # ── Construction ────────────────────────────────────────────

    @classmethod
    def _from_capsule(cls, capsule: Any) -> BeamModel:
        obj = cls.__new__(cls)
        obj._g = capsule
        obj._compacted = False
        return obj

    #: Conventional suffix for a saved model.
    #:
    #: Applied by :meth:`save` when the path carries no suffix of its own. An
    #: explicit suffix is always respected, so files written before this
    #: convention existed still load and can still be overwritten by name.
    SUFFIX = ".beam"

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> BeamModel:
        """Load a model written by :meth:`save`.

        Any path is accepted. The suffix is convention, not a format check --
        the file is identified by its magic number and validated by CRC.
        """
        return cls._from_capsule(_c.load(os.fspath(path)))

    def save(self, path: str | os.PathLike) -> None:
        """Write the model's counts and configuration to ``path``.

        Counts are written rather than the derived tables, so a loaded model
        can still be subtracted from and re-finalized under different
        smoothing.

        A path with no suffix gets :attr:`SUFFIX` (``.beam``). A path that
        already has one keeps it, so an existing ``.model`` file can still be
        written by name.
        """
        q = Path(os.fspath(path))
        if not q.suffix:
            q = q.with_suffix(self.SUFFIX)
        _c.save(self._g, os.fspath(q))

    def __repr__(self) -> str:
        cfg = self.config
        return (f"BeamModel(order={cfg['order']}, sequences={self.n_observed:,}, "
                f"states={self.n_states:,}, "
                f"memory={self.memory_bytes / 2**20:.0f}MB)")

    def __len__(self) -> int:
        return int(self.n_observed)

    def __contains__(self, sequence: str) -> bool:
        return math.isfinite(self.log_pseq(sequence))

    # ── Introspection ───────────────────────────────────────────

    @property
    def config(self) -> dict[str, Any]:
        """The configuration this model was built with."""
        return _c.config(self._g)

    @property
    def alphabet(self) -> str:
        """The alphabet this model reads: ``"aa"`` or ``"nt"``.

        It decides what the model accepts, the width of every table, and the
        legal range of :attr:`config`'s ``order``.
        """
        return str(self.config["alphabet"])

    @property
    def n_residues(self) -> int:
        """Residues this model's alphabet emits: 20 for ``aa``, 5 for ``nt``."""
        return int(alphabets.info(self.alphabet)["residues"])

    @property
    def n_observed(self) -> int:
        """Sequences accepted during training."""
        return int(_c.n_observed(self._g))

    @property
    def n_skipped(self) -> int:
        """Sequences rejected as too long or outside the alphabet."""
        return int(_c.n_skipped(self._g))

    @property
    def total_weight(self) -> float:
        """Total abundance observed during training."""
        return float(_c.total_weight(self._g))

    @property
    def n_states(self) -> int:
        """Nodes per layer of the DAG, ``21 ** order``.

        A state holds exactly the residues one context needs. It was
        ``max(order, 3)`` while a second expert of fixed order 3 existed,
        because a narrower state could not have supplied its context.
        """
        return int(_c.n_states(self._g))

    @property
    def n_cells(self) -> int:
        """Total table cells across both directions."""
        return int(_c.n_cells(self._g))

    @property
    def n_nonzero_cells(self) -> int:
        """Cells carrying a non-zero count. Unavailable after :meth:`compact`.

        Raises rather than reporting the 0 the C layer returns once the counts
        are gone, which is indistinguishable from a genuinely empty model.
        """
        if self._compacted:
            raise RuntimeError(
                "n_nonzero_cells is unavailable after compact(): the raw "
                "counts it inspects have been released")
        return int(_c.n_nonzero_cells(self._g))

    # ── Cross-model summary names ───────────────────────────────
    #
    # These carry the spelling the other models in this library use, so code
    # that summarises several of them side by side does not have to special
    # case BeamModel. Each one is a real quantity, not a stub.

    @property
    def variant(self) -> str:
        """``"beam"``. The tag a summary table reports for this model."""
        return "beam"

    @property
    def n_sequences(self) -> float:
        """Total abundance ingested. Alias of :attr:`total_weight`.

        The number a corpus-integrity check compares against a release's
        summed abundance column: it is what the model actually ingested, so a
        silently dropped sequence shows up here.
        """
        return self.total_weight

    @property
    def observed_lengths(self) -> tuple[int, ...]:
        """Lengths the model can produce, ascending."""
        return tuple(L for L in range(int(self.config["max_length"]) + 1)
                     if math.isfinite(self.log_length_prior(L)))

    @property
    def length_distribution(self) -> dict[int, float]:
        """``{length: pi(L)}`` over the lengths the model produces.

        The observed length distribution, NORMALISED — a probability, not a
        count. BeamModel keeps the prior rather than the raw per-length totals, so
        a count would have to be reconstructed and would be a reconstruction.
        Iterating gives the lengths, so ``min``/``max`` over it read as they
        do on the other models.
        """
        return {L: math.exp(self.log_length_prior(L))
                for L in self.observed_lengths}

    @staticmethod
    def _pkg_version() -> str:
        try:
            return _dist_version("beam-model")
        except PackageNotFoundError:            # an uninstalled source tree
            return "unknown"

    def metadata(self, *, lengths: bool = True,
                 support: bool = False) -> dict[str, Any]:
        """Everything the model knows about itself.

        Three of these are STORED, because a finished model cannot recompute
        them: when it was first built, and how much wall time went into
        observing and into finalizing. They survive :meth:`save` and
        :meth:`from_file`, so a model loaded from disk reports when it was
        built rather than when it was loaded.

        A re-finalize the CALLER asks for -- after :meth:`add` or
        :meth:`subtract` -- adds to the finalize total. The one every load
        performs does not: the file stores counts rather than tables, so
        rebuilding them is the format's cost and not this model's, and
        charging it would make the build time rise every time the model was
        opened.

        Everything else is derived on demand. Storing it would be a second
        copy to keep in step with the tables, and the tables are the truth.

        Args:
            lengths: include the per-length breakdown. One sweep over the
                tables, so it costs something on a large model.
            support: include ``path_count`` per length, which is how many
                distinct sequences of that length the model can spell. A
                separate dynamic program PER LENGTH, so it is off by default
                and is the expensive part of this call by a wide margin.

        The per-length entries answer "how much of what this length admits
        has this model actually seen". A sequence of length ``L`` is scored
        through forward depths ``0..L-1`` and backward depths ``0..L-1``, and
        at depth ``d`` only ``min(d, order)`` context slots are free -- the
        rest are the sentinel padding that direction's anchor. So the rows it
        can REACH number ``2 * sum_d R ** min(d, order)``, and
        ``row_coverage`` is the fraction of those the model holds. ``cells``
        is the same question one level down, counting residues within rows
        rather than rows.

        The denominator counts what is reachable rather than what the
        alphabet can spell, and the difference is not small: at length 4 the
        looser count is 1,555,848 rows where 16,842 exist, so a model built
        from every sequence of that length reads as 1.08% rather than 100%.

        ``share`` is that length's mass under the length prior, so the
        coverage figures can be weighted by how much of the model actually
        lives at each length.

        >>> m = BeamModel(["CASSLGEQYF", "CASSQDTQYF"], order=2, max_length=12)
        >>> meta = m.metadata()
        >>> meta["config"]["order"]
        2
        >>> meta["lengths"][10]["rows_possible"]
        8820
        """
        cfg = self.config
        created_unix, obs_ns, fin_ns = _c.provenance(self._g)
        radix = alphabets.info(self.alphabet)["radix"]
        n_res = self.n_residues
        width = int(cfg["max_length"])
        order = int(cfg["order"])
        cells = self.n_cells
        # `n_nonzero_cells` inspects the raw counts, which `compact()`
        # releases. A compacted model is a normal thing to be handed -- it is
        # the form a large model is deployed in -- so this reports None rather
        # than raising, and `compacted` says why. None rather than a zero or a
        # dense stand-in, because both would read as a real occupancy.
        nonzero = None if self._compacted else self.n_nonzero_cells

        out: dict[str, Any] = {
            # Zero means the file predates provenance (format 6). Report it
            # as unknown rather than as 1970, which is what an epoch stamp
            # rendered literally would claim.
            "created": (_datetime.datetime.fromtimestamp(
                created_unix, _datetime.timezone.utc).isoformat()
                if created_unix else None),
            "created_unix": int(created_unix) or None,
            "beam_version": self._pkg_version(),
            "variant": self.variant,
            "finalized": True,
            "config": cfg,
            "build": {
                "observe_seconds": obs_ns / 1e9,
                "finalize_seconds": fin_ns / 1e9,
                "total_seconds": (obs_ns + fin_ns) / 1e9,
                "n_observed": self.n_observed,
                "n_skipped": self.n_skipped,
                "n_sequences": self.n_sequences,
                "total_weight": self.total_weight,
            },
            "size": {
                "n_states": self.n_states,
                "n_cells": cells,
                "n_nonzero_cells": nonzero,
                "occupancy": ((nonzero / cells) if cells else 0.0)
                             if nonzero is not None else None,
                "memory_bytes": self.memory_bytes,
                "compacted": self._compacted,
            },
        }
        if not lengths:
            return out

        # A compacted model cannot answer this: the prior made every finalized
        # cell positive and compact() released the counts that knew which were
        # reached. Reported as absent rather than as a refusal, because the
        # rest of the per-length entry is still exact.
        try:
            rows, cell_counts = _c.depth_occupancy(self._g)
        except Exception:
            rows = cell_counts = None
        prior = self.length_distribution
        per: dict[int, dict[str, Any]] = {}
        for L in self.observed_lengths:
            # Depths 0..L-1 in each direction are what a length-L sequence
            # reads, so those are the rows it could possibly be scored through.
            known = rows is not None
            seen_rows = (sum(rows[d * width + k] for d in (0, 1)
                             for k in range(L)) if known else None)
            seen_cells = (sum(cell_counts[d * width + k] for d in (0, 1)
                              for k in range(L)) if known else None)
            # The REACHABLE ceiling, not every context the alphabet can
            # spell. At depth `d` a context has only `min(d, order)` free
            # slots -- the rest are the sentinel padding that direction's
            # anchor -- so `R ** min(d, order)` rows exist there and the
            # others are unaddressable rather than merely unobserved.
            #
            # This was `2 * L * radix ** order`, which counts rows no
            # sequence can reach. At depth 0 exactly ONE row exists, the
            # all-sentinel context, and a model that had found it was
            # reported at 1/194481 of that depth rather than at all of it.
            #
            # Checked against brute force rather than argued: enumerate every
            # sequence over a small alphabet and `n_nonzero_cells` IS the
            # reachable set. See
            # `test_the_possible_cell_count_is_what_brute_force_occupies`.
            poss_rows = 2 * sum(n_res ** min(d, order) for d in range(L))
            poss_cells = poss_rows * n_res
            entry: dict[str, Any] = {
                "share": float(prior.get(L, 0.0)),
                "rows_possible": poss_rows,
                "rows_observed": int(seen_rows) if known else None,
                "row_coverage": (seen_rows / poss_rows
                                 if poss_rows else 0.0) if known else None,
                "cells_possible": poss_cells,
                "cells_observed": int(seen_cells) if known else None,
                "cell_coverage": (seen_cells / poss_cells
                                  if poss_cells else 0.0) if known else None,
                "log_partition": self.log_partition(L),
            }
            if support:
                pc = self.path_count(L)
                entry["path_count"] = pc
                # Of every string of this length, not every string the model
                # was shown. log10 because the denominator reaches 1e35.
                entry["log10_spellable_fraction"] = (
                    math.log10(pc) - L * math.log10(n_res)
                    if pc > 0.0 else float("-inf"))
            per[int(L)] = entry
        out["lengths"] = per
        return out

    def pgen_diagnostics(self) -> dict[str, Any]:
        """Measure, rather than assert, that this is a proper distribution.

        Two independent identities are checked by actually evaluating them:
        ``sum_s P(s) == 1`` through the Mellin transform at ``q=1``, which is
        a full sweep over the trellis, and ``sum_L pi(L) == 1`` through the
        length prior. A model that normalised only per length, or whose length
        prior had leaked, would fail one of them.

        The name is the one :class:`FlashBackGraph` uses; the quantity here is
        Pseq, and BeamModel has no generation model, so nothing in it is a Pgen.
        """
        mass = math.exp(self.log_mellin(1.0))
        prior = float(sum(self.length_distribution.values()))
        worst = max(abs(mass - 1.0), abs(prior - 1.0))
        return {
            "is_proper": worst < 1e-9,
            "total_mass": mass,
            "length_prior_mass": prior,
            "max_abs_deviation": worst,
            "measure": "Pseq",
        }

    @property
    def threads(self) -> int:
        """The thread cap in force, or 0 for the runtime's own default.

        .. warning::
           This is a **process-wide** setting, not a per-model one. Assigning
           to it on any model changes it for every model in the process, and
           for every other part of the library that runs a BeamModel parallel
           region. It is exposed here because this is where a caller looks for
           it; :func:`set_threads` is the spelling that says so.

        Results do not depend on it — reductions are order-independent or
        summed in fixed index order — so pinning it to 1 for a runtime
        comparison changes the timing and not the answer.

        Example:
            >>> model.threads = 1        # pin for a fair benchmark
            >>> model.effective_threads
            1
            >>> model.threads = 0        # back to the runtime's choice
        """
        return threads()

    @threads.setter
    def threads(self, n: int) -> None:
        set_threads(n)

    @property
    def effective_threads(self) -> int:
        """Threads a parallel region would actually start right now.

        The cap if one is set, otherwise the OpenMP runtime's maximum, and
        always 1 in a build without OpenMP. Unlike :attr:`threads`, which
        reports what was ASKED for, this reports what would happen.
        """
        return effective_threads()

    @property
    def memory_bytes(self) -> int:
        """Approximate resident bytes of the model's TABLES.

        Exact to within 0.2 MB of what the build actually allocates. It does
        NOT include the live cache, which appears after the first analytic and
        can be larger than this figure -- see :attr:`live_cache_bytes`.
        """
        return int(_c.memory_bytes(self._g))

    @property
    def live_cache_bytes(self) -> int:
        """Bytes held by the per-length live cache, excluded from
        :attr:`memory_bytes`.

        The cache memoises the trellis structure every analytic needs; without
        it each one rebuilds that structure, which is most of their cost. But
        it is real memory, it appears only once an analytic has run, and it is
        bounded by a quarter of the memory budget -- one gigabyte at the
        default. Measured: 15.1 MB on a per-repertoire model whose tables are
        13.8 MB, and 382.1 MB on a foundation whose tables are 806.7 MB.

        Zero before the first analytic, and after :meth:`drop_caches`.
        """
        return int(_c.live_cache_bytes(self._g))

    def drop_caches(self) -> None:
        """Release everything the model can rebuild.

        Only the live cache today. Changes no result -- the next analytic
        rebuilds what it needs and pays the build cost again. For a caller
        holding more models than fit, or done analysing one it must keep.

        Not safe to call while another thread is inside an analytic on this
        model, which is the rule for every other mutation here.
        """
        _c.drop_caches(self._g)

    def compact(self) -> None:
        """Drop the raw counts, keeping only the finalized tables.

        Roughly halves resident memory. After this the model can be scored and
        analysed but not subtracted from, saved, or re-finalized.
        """
        _c.compact(self._g)
        self._compacted = True

    # ── Argument checking ───────────────────────────────────────

    def _check_length(self, length: int, *, allow_none: bool = False) -> int:
        """Validate a sequence length and return it in C form (0 == "all").

        The C layer takes lengths as ``uint32_t``, so a negative one wraps to
        billions and comes back as a bare "invalid argument" or, worse, as a
        silent ``-inf``. Naming the offending value here is the difference
        between a typo that is obvious and one that is not.
        """
        if length is None:
            if allow_none:
                return 0
            raise TypeError("length must be an integer")
        try:
            v = int(length)
        except (TypeError, ValueError):
            raise TypeError(
                f"length must be an integer, got {length!r}") from None
        hi = int(self.config["max_length"])
        if not 0 <= v <= hi:
            raise ValueError(
                f"length must be between 0 and max_length={hi}, got {v}")
        return v

    def _check_count(self, name: str, n: int) -> int:
        """Validate a "how many" argument.

        The C layer takes these as ``uint32_t``, and the binding used to parse
        them with ``PyArg_ParseTuple``'s ``"I"``, which truncates modulo 2**32
        without error. So this checked only the sign -- while its own docstring
        described the magnitude problem it was not preventing: ``sample(-1)``
        was rejected, and ``sample(2**32 + 5)`` became a request for 5.

        The bindings now refuse an out-of-range integer themselves, and refuse
        a request too large for the memory budget. This stays as the first
        line, because a Python-level message can name the argument as the
        caller wrote it.
        """
        try:
            v = int(n)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be an integer, got {n!r}") from None
        if v < 0:
            raise ValueError(f"{name} must be non-negative, got {v}")
        if v > 0xFFFFFFFF:
            raise ValueError(
                f"{name} must be at most 4294967295, got {v}")
        return v

    def _check_bins(self, bins: int, spacing: float | None) -> int:
        """Validate a spectrum grid size.

        When the spacing is chosen automatically the C layer reserves
        ``max_length + 9`` bins of headroom for the linear splitting, so the
        real floor is well above the nominal 2 and asking for less fails with
        a message that does not mention it.
        """
        try:
            v = int(bins)
        except (TypeError, ValueError):
            raise TypeError(f"bins must be an integer, got {bins!r}") from None
        if v < 2:
            raise ValueError(f"bins must be at least 2, got {v}")
        if spacing is None:
            floor = int(self.config["max_length"]) + 10
            if v < floor:
                raise ValueError(
                    f"bins must be at least {floor} when the spacing is chosen "
                    f"automatically (got {v}): the transport reserves "
                    f"max_length + 9 bins of headroom for linear splitting. "
                    f"Pass an explicit `spacing` to use a smaller grid.")
        return v

    def _grid_spacing(self, bins: int) -> float:
        """The bin width the C layer would choose for this many bins.

        Reproduced here, rather than read back from a probe spectrum, so that
        several spectra can be asked for on a grid that is known to be shared
        BEFORE any of them is built. The headroom term is the transport's:
        ``max_length + 9`` bins are reserved for linear splitting.
        """
        bins = self._check_bins(bins, None)
        hi = self.accumulator_max()
        if not (hi > 0):
            raise ValueError("model has no producible sequences")
        return hi * 1.02 / (bins - (int(self.config["max_length"]) + 9))

    # ── Scoring ─────────────────────────────────────────────────

    def log_pseq(self, sequence: str) -> float:
        """Natural log of the sequence probability.

        Returns ``-inf``, rather than raising, for a sequence the model
        cannot produce: empty, longer than ``max_length``, containing a
        non-amino-acid character, or of a length never observed in training.
        Use :meth:`explain`, which raises and names the rule, when the reason
        matters.
        """
        return float(_c.log_pseq(self._g, sequence))

    def log_pseq_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Vectorised :meth:`log_pseq`. Parallel where OpenMP is available."""
        return np.asarray(_c.log_pseq_batch(self._g, list(sequences)),
                          dtype=np.float64)

    def pseq(self, sequence, *, log: bool = False):
        """Sequence probability, for one sequence or many.

        Args:
            sequence: A single string, or any sequence of strings.
            log: Return ``ln P`` instead of ``P``. ``log=True`` with a list is
                the spelling :class:`FlashBackGraph` uses, so pipeline code
                written against that reads across unchanged.

        Returns:
            ``float`` for a single string, ``np.ndarray`` for anything else.

        Note:
            The default is ``log=False`` — the LINEAR probability — which is
            the opposite of :class:`FlashBackGraph`'s default. BeamModel named the
            log form :meth:`log_pseq` from the start and that name is what its
            own callers use, so flipping this default would silently rescale
            every existing script rather than break it.
        """
        if isinstance(sequence, str):
            v = self.log_pseq(sequence)
            return float(v) if log else math.exp(v)
        v = self.log_pseq_batch(list(sequence))
        return v if log else np.exp(v)

    # ── Streaming and file-backed scoring ───────────────────────

    def log_pseq_chunks(
        self, sequences: Iterable[str], *, chunk_size: int = 1_000_000,
    ):
        """Score an arbitrary iterable of sequences, yielding one array per
        chunk and never holding more than ``chunk_size`` of them.

        The primitive the file and parquet scorers are built on, and the one
        to reach for when the sequences come from somewhere else entirely.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        buf: list[str] = []
        for seq in sequences:
            buf.append(seq)
            if len(buf) >= chunk_size:
                yield self.log_pseq_batch(buf)
                buf = []
        if buf:
            yield self.log_pseq_batch(buf)

    def log_pseq_file(
        self,
        path: str | os.PathLike,
        *,
        sep: str = "\t",
        column: int = 0,
        has_header: bool = False,
        chunk_size: int = 1_000_000,
        out: Any = None,
        dtype: Any = np.float64,
    ) -> np.ndarray:
        """Score every sequence in a delimited text file.

        Args:
            column: Zero-based index of the sequence field.
            out: A preallocated array or ``numpy`` memmap to fill. Its length
                must match the file's. ``None`` grows a list of chunks and
                concatenates, which needs the whole result in memory.
            dtype: Element type when ``out`` is ``None``. ``float32`` halves
                the result's footprint and costs about 1e-7 relative on a
                log-probability.
        """
        return self._collect(
            (self.log_pseq_batch(chunk) for chunk, _ in _iter_delimited(
                path, sep=sep, sequence_column=column, abundance_column=None,
                has_header=has_header, chunk_size=chunk_size)),
            out, dtype)

    def log_pseq_parquet(
        self,
        path: str | os.PathLike,
        *,
        column: str | None = None,
        batch_size: int = 1_000_000,
        out: Any = None,
        dtype: Any = np.float64,
    ) -> np.ndarray:
        """Score every sequence in a parquet column, one record batch at a time.

        Requires ``pyarrow``. Row order is the file's, so the result lines up
        positionally with any other column of the same file — which is how the
        aligned (sequence, publicness) releases this library is used on have to
        be read.

        Args:
            column: Column holding the sequences. Defaults to this model's
                alphabet: ``cdr3_aa`` for amino acids, ``cdr3`` for
                nucleotides.
            out: A preallocated array or memmap to fill, of the file's length.
            dtype: Element type when ``out`` is ``None``.
        """
        column = _default_column(column, {"alphabet": self.alphabet})
        return self._collect(
            (self.log_pseq_batch(chunk)
             for chunk, _ in _iter_parquet(path, column, None, batch_size)),
            out, dtype)

    @staticmethod
    def _collect(chunks, out, dtype) -> np.ndarray:
        """Fill ``out`` from an iterator of score arrays, or concatenate."""
        if out is None:
            parts = list(chunks)
            if not parts:
                return np.empty(0, dtype=dtype)
            return np.concatenate(parts).astype(dtype, copy=False)
        at = 0
        for part in chunks:
            end = at + len(part)
            if end > len(out):
                raise ValueError(
                    f"`out` holds {len(out)} elements but the source has at "
                    f"least {end}; it is not the right length for this input")
            out[at:end] = part
            at = end
        if at != len(out):
            raise ValueError(
                f"`out` holds {len(out)} elements but the source supplied "
                f"{at}; it is not the right length for this input")
        return out

    def explain(self, sequence: str) -> dict[str, Any]:
        """Per-residue decomposition of a score.

        Returns the residues, and each one's N-anchored and C-anchored
        natural-log probability before mixing. This is what makes the model
        legible: the N-anchored expert is confident at the start, the
        C-anchored expert at the end, and both are surprised across the
        junction.

        The two arrays reconstruct ``log_pseq`` exactly, using the mixing
        weights from :attr:`config`::

            wf, wb = model.config["weights"]
            raw = sum(wf * f + wb * b
                      for f, b in zip(e["forward"], e["backward"]))
            raw + model.log_length_prior(L) - model.log_partition(L)

        with ONE exception: a sequence of a length the model never observed
        has ``log_pseq == -inf`` while its per-residue terms stay finite,
        because the length prior — not any residue — is what rules it out. The
        returned ``length_is_producible`` flag says which case you are in.

        Raises:
            ValueError: if the sequence is empty, longer than ``max_length``,
                or contains a character outside the model's alphabet. Use
                :meth:`log_pseq`, which returns ``-inf`` instead, when a
                verdict rather than a decomposition is what is wanted.
        """
        self._check_scoreable(sequence)
        fwd, bwd = _c.explain(self._g, sequence)
        fwd = np.asarray(fwd, dtype=np.float64)
        bwd = np.asarray(bwd, dtype=np.float64)
        return {
            "sequence": sequence,
            "residues": list(sequence),
            "forward": fwd,
            "backward": bwd,
            "depth_from_n": np.arange(len(sequence)),
            "depth_from_c": np.arange(len(sequence))[::-1],
            "log_pseq": self.log_pseq(sequence),
            "length_is_producible": math.isfinite(
                self.log_length_prior(len(sequence))),
        }

    def _check_scoreable(self, sequence: Any) -> None:
        """Reject a sequence the encoder would refuse, saying which rule it hit.

        The C layer collapses all three refusals into one error code and does
        not populate a message, so without this the caller is told only
        "invalid argument".
        """
        if not isinstance(sequence, str):
            raise TypeError(f"sequence must be a string, got "
                            f"{type(sequence).__name__}")
        hi = int(self.config["max_length"])
        if not sequence:
            raise ValueError("sequence is empty")
        if len(sequence) > hi:
            raise ValueError(
                f"sequence has length {len(sequence)}, longer than "
                f"max_length={hi}")
        bad = alphabets.offenders(self.alphabet, sequence)
        if bad:
            raise ValueError(
                f"sequence contains {bad!r}, which the {self.alphabet!r} "
                f"alphabet does not accept; its residues are "
                f"{alphabets.symbols(self.alphabet)!r}")

    # ── Per-length quantities ───────────────────────────────────

    def log_partition(self, length: int) -> float:
        """``ln Z(L)``, the total weight of every length-``L`` path.

        Needed because a geometric product of two experts is not normalised on
        its own.
        """
        return float(_c.log_partition(self._g, self._check_length(length)))

    def log_pseq_by_position(self, sequence: str) -> np.ndarray:
        """``ln Pseq(s)`` broken down per position. **The sum IS the score.**

        Returns ``len(sequence)`` values whose total is :meth:`log_pseq` of the
        same sequence. Stack the vectors from ``K`` per-donor models and the
        result is a ``K x L`` matrix whose row sums are those models' scores,
        with a per-position account of how each one arrived there.

        Entry ``i`` is::

            w_f * ln q_f(s_i | i, ctx) + w_b * ln q_b(s_i | L-1-i, ctx)
              - (that position's share of ln Z(L))
              + (ln pi(L), on the final position only)

        and it is ``-inf`` exactly where the model cannot spell the transition
        at ``i`` -- a context it never observed, or a residue that never
        followed one it did. One such position makes the whole row ``-inf``,
        which is what :meth:`log_pseq` returns for that sequence.

        WHERE THE TWO CONSTANTS GO, and why they are not alike.

        ``ln P(s) = ln W(s) + ln pi(L) - ln Z(L)``. Both trailing terms belong
        to the sequence's LENGTH rather than to any position, so a per-position
        vector has to place them. They are not equally placeable.

        ``ln Z(L)`` **has** positional structure: it is a forward sum over the
        trellis and a forward sum factorises by layer, so each position takes
        the model's own share of it -- see :meth:`log_partition_by_layer`. It
        is far from even; on a small fixture one layer carries more than half.

        ``ln pi(L)`` **has none**. It is one number read off the observed
        length histogram, and no decomposition of it exists to find. It is
        placed on the FINAL position, for the same reason the sink is: the
        length is only settled once the sequence ends. That is a convention,
        not a derivation, and it is the one arbitrary choice in this vector.
        A caller wanting it elsewhere has :meth:`log_partition_by_layer` and
        :meth:`log_length_prior` and can rebuild the vector to taste.

        Costs one forward sweep for ``ln Z(L)``'s layers, plus the explain
        pass. Per MODEL and per LENGTH, not per sequence -- score many
        sequences of one length against one model and the sweep is amortised.
        """
        L = len(sequence)
        if L == 0:
            raise BeamValueError("log_pseq_by_position: the sequence is empty")
        ex = self.explain(sequence)
        w_f, w_b = (float(x) for x in self.config["weights"])
        psi = (w_f * np.asarray(ex["forward"], dtype=np.float64)
               + w_b * np.asarray(ex["backward"], dtype=np.float64))

        out = psi.copy()
        if not ex["length_is_producible"]:
            # No length prior and no partition function to share out. The
            # score is -inf and the breakdown says so at the position the
            # prior would have occupied.
            out[-1] = -np.inf
            return out

        running = self.log_partition_by_layer(L)
        shares = np.diff(np.concatenate(([0.0], running)))
        # -inf - finite stays -inf; only subtract where psi is finite, so a
        # dead position is never turned into a nan.
        live = np.isfinite(out)
        out[live] -= shares[live]
        out[-1] += self.log_length_prior(L)
        return out

    def log_partition_by_layer(self, length: int) -> np.ndarray:
        """``ln Z(L)`` split across the positions it is a product over.

        Returns ``length`` running log-masses: entry ``i`` is the log of the
        trellis mass once residue ``i`` has been emitted. **Difference them**
        to get each position's share of ``ln Z(L)``.

        WHY THIS IS NOT A UNIFORM SPLIT. ``ln P(s) = ln W(s) + ln pi(L) -
        ln Z(L)``, and the last two terms belong to the sequence's LENGTH, not
        to any position in it. Anything reading a per-position cell has to put
        ``-ln Z(L)`` somewhere, and spreading it evenly over ``L`` is
        arithmetically valid and analytically arbitrary -- any weights summing
        to one give the same row total and a different matrix. But ``Z(L)`` is
        a forward sum over the trellis and a forward sum factorises by layer,
        so the split is the model's own rather than a choice.

        It is uneven enough to matter: on a small fixture one layer carries
        more than half of ``-ln Z`` while four carry essentially none.

        The last entry is :meth:`log_partition` exactly, bit for bit, so read
        ``ln Z(L)`` off it rather than re-summing the differences, which
        accumulate about one ULP over ``L`` terms.

        The sink -- the backward scores of the final ``min(order, length)``
        residues, which never get a full right-hand context -- is attributed
        to the last layer, as :meth:`explain` attributes ``by_depth[-1]``.

        Costs one forward sweep, where :meth:`log_partition` reads a value
        cached at finalize.
        """
        return np.asarray(
            _c.log_partition_by_layer(self._g, self._check_length(length)),
            dtype=np.float64)

    def log_length_prior(self, length: int) -> float:
        """``ln pi(L)``, from the observed length distribution."""
        return float(_c.log_length_prior(self._g,
                                              self._check_length(length)))

    def length_pmf(self) -> np.ndarray:
        """The length prior as a probability vector indexed by length."""
        n = int(self.config["max_length"]) + 1
        return np.array([math.exp(self.log_length_prior(L)) for L in range(n)])

    # ── Analytics ───────────────────────────────────────────────

    def path_count(self, length: int | None = None) -> float:
        """Distinct sequences the model can produce, by exact path counting.

        .. warning::
           This is exactly ``R ** length`` for ``R`` residues: under the
           hierarchical prior every table cell is positive, so every string of
           the length is possible, and the number describes the alphabet
           rather than the corpus. Sparse and dense storage agree.

           It is kept because it is the exact ``q=0`` corner of the Hill
           family and because it is what the counting spectrum integrates to,
           so the two have to agree.
        """
        if length is None:
            return float(_c.path_count_total(self._g))
        return float(_c.path_count(self._g, self._check_length(length)))

    def entropy(self, *, base: float = math.e) -> float:
        """Shannon entropy of the sequence distribution. Exact."""
        if not (base > 0.0) or base == 1.0:
            raise ValueError(f"base must be positive and not 1, got {base!r}")
        h = float(_c.entropy(self._g))
        return h if base == math.e else h / math.log(base)

    def hill_number(self, alpha: float) -> float:
        """Effective number of sequences at order ``alpha``.

        ``alpha=0`` is the support size, ``alpha=1`` the exponential of the
        entropy, ``alpha=2`` the inverse collision probability.
        """
        return float(_c.hill_number(self._g, float(alpha)))

    def accumulator_max(self) -> float:
        """Largest value the spectrum DP's accumulator reaches, in nats.

        This is **not** the largest surprisal. The DP accumulates ``-w`` along
        the path, which is monotone, and applies the per-length constant
        ``ln Z(L) - ln pi(L) - tail`` only at the sink; when that constant is
        negative the final surprisal is smaller than the accumulator ever was.
        Grids must be sized by this, not by the surprisal.
        """
        return float(_c.accumulator_max(self._g))

    def spectrum(
        self,
        *,
        measure: str = "generated",
        length: int | None = None,
        bins: int = 1024,
        spacing: float | None = None,
    ) -> dict[str, Any]:
        """Exact surprisal spectrum by forward grid transport. No sampling.

        Args:
            measure: ``'counting'`` weights every supported sequence 1, so the
                spectrum integrates to the support size. ``'generated'``
                weights each sequence by its own probability, so it integrates
                to 1.
            length: A single length, or ``None`` to pool over all of them.
            bins: Resolution of the surprisal axis.
            spacing: Bin width in nats, or ``None`` to size it automatically
                from :meth:`accumulator_max`.

        Returns a dict with ``surprisal`` (bin left edges, ``-ln P``),
        ``weights``, ``spacing``, and ``total``.

        "Exact" here means no sampling: the measure transported is the true
        one. It does NOT mean every bin is exact. Mass is split linearly
        between adjacent bins at EVERY layer, so a sequence's weight diffuses
        over a few bins rather than landing on the two adjacent to its own
        surprisal. What that splitting preserves exactly is the total and the
        FIRST moment — the mean surprisal read off this grid matches the exact
        one to machine precision — while the variance comes out inflated
        by at most ``(L + 1) * spacing**2 / 6``, one split per layer plus the
        sink, which shrinks as the SQUARE of the bin width. Read shape at grid
        resolution, and take moments from :meth:`surprisal_moments` when they
        matter.

        Raises:
            ValueError: if ``bins`` is below 2, or below ``max_length + 10``
                when ``spacing`` is ``None`` — the transport reserves
                ``max_length + 9`` bins of headroom for the linear splitting,
                so the effective floor is higher than the nominal one.

        .. note::
           Cost scales with the number of *live* states per layer, and there
           is no pruning: every state is live at every layer, 194,481 of them
           at order 4, 9,261 at order 3 and 441 at order 2 -- it is
           ``radix ** order`` now, not ``radix ** max(order, 3)``. A pooled
           spectrum over a
           full-range order-4 model needs several GB and minutes. Ask for one
           length at a time, or build at a lower ``order``, when that matters.
        """
        if measure not in ("counting", "generated"):
            raise ValueError(
                f"measure must be 'counting' or 'generated', got {measure!r}")
        bins = self._check_bins(bins, spacing)
        if spacing is not None and not (float(spacing) > 0.0):
            raise ValueError(
                f"spacing must be positive, got {spacing!r}; pass None to have "
                f"it chosen from accumulator_max()")
        w, ds = _c.spectrum(
            self._g, measure == "counting",
            self._check_length(length, allow_none=True),
            bins, 0.0 if spacing is None else float(spacing))
        w = np.asarray(w, dtype=np.float64)
        return {
            "surprisal": np.arange(bins) * ds,
            "weights": w,
            "spacing": ds,
            "total": float(w.sum()),
            "measure": measure,
        }

    def spectrum_pair(
        self,
        *,
        length: int | None = None,
        bins: int = 1024,
        spacing: float | None = None,
    ) -> dict[str, Any]:
        """Both spectrum measures through ONE transport.

        Returns ``surprisal``, ``counting``, ``generated``, ``spacing``.
        About 1.5x faster than two :meth:`spectrum` calls, and bit-identical
        to them.

        Speed is the smaller reason it exists. Linear splitting spreads each
        sequence over adjacent bins, and ``generated[b] / counting[b]`` is the
        exact mass-weighted mean probability of the sequences in bin ``b``
        **only if both measures were split by the same fractions**. Two
        separate calls agree today because they recompute the same arithmetic
        from the same inputs; nothing holds them together, and everything
        downstream of that ratio — see :meth:`PseqAnalysis.atoms` —
        assumes they are. One transport makes it structural.
        """
        bins = self._check_bins(bins, spacing)
        if spacing is not None and not (float(spacing) > 0.0):
            raise ValueError(
                f"spacing must be positive, got {spacing!r}; pass None to have "
                f"it chosen from accumulator_max()")
        cb, gb, ds = _c.spectrum_pair(
            self._g, self._check_length(length, allow_none=True), bins,
            0.0 if spacing is None else float(spacing))
        return {
            "surprisal": np.arange(bins) * ds,
            "counting": np.frombuffer(cb, dtype=np.float64).copy(),
            "generated": np.frombuffer(gb, dtype=np.float64).copy(),
            "spacing": ds,
        }

    def spectra_by_length(
        self,
        *,
        measure: str = "generated",
        max_length: int | None = None,
        bins: int = 1024,
        spacing: float | None = None,
    ) -> dict[str, Any]:
        """Every length's spectrum in one call, on ONE shared grid.

        Returns ``surprisal``, ``weights`` (a ``(max_length + 1, bins)``
        array, zero for lengths the model does not generate), and ``spacing``.
        Summing the rows reproduces the pooled spectrum.

        .. note:: **What this buys, honestly.**
           BeamModel's per-length trellises are genuinely separate dynamic programs
           with no shared work, so fusing them saves no arithmetic. Measured,
           it is 1.26x at order 3 and *parity* at order 4 with 512 bins, where
           each length's scratch is large enough that the memory budget allows
           only two at a time and OpenMP does not nest, so it falls back to
           serial-outer / parallel-inner.

           The value at order 4 is therefore correctness and convenience, not
           speed: one grid shared by every row, sized from one
           :meth:`accumulator_max` instead of one per length, so the rows are
           directly comparable and summable.

        Concurrency is capped by :func:`set_memory_budget`, not by core
        count, because one length's scratch is measured in gigabytes at order
        4. The budget never caps what a single length may allocate and the
        result does not depend on it.
        """
        if measure not in ("counting", "generated"):
            raise ValueError(
                f"measure must be 'counting' or 'generated', got {measure!r}")
        bins = self._check_bins(bins, spacing)
        if spacing is not None and not (float(spacing) > 0.0):
            raise ValueError(
                f"spacing must be positive, got {spacing!r}; pass None to have "
                f"it chosen from accumulator_max()")
        hi = int(self.config["max_length"])
        if max_length is None:
            want = 0                      # 0 means "the model's own maximum"
        else:
            max_length = int(max_length)
            if not 0 <= max_length <= hi:
                raise ValueError(
                    f"max_length must be between 0 and {hi}, got {max_length}")
            want = max_length
        b, rows, cols, ds = _c.spectra_by_length(
            self._g, measure == "counting", want, bins,
            0.0 if spacing is None else float(spacing))
        return {
            "surprisal": np.arange(cols) * ds,
            "weights": np.frombuffer(b, dtype=np.float64).reshape(rows,
                                                                  cols).copy(),
            "spacing": ds,
            "measure": measure,
        }

    def length_moments(self, *, q: float = 1.0, order: int = 4) -> dict[str, Any]:
        """The ``q``-tilted surprisal moments of every length, unmixed.

        Returns ``length``, ``share`` and ``moments``. ``share[L]`` is that
        length's share of the ``q``-tilted law — at ``q=1`` the length prior
        itself — and ``moments[L, k]`` is ``E_q[(-ln P)**k | length L]``, with
        ``moments[L, 0] == 1``.

        This is :meth:`surprisal_moments` with the cross-length mixture left
        undone, and the two are related by exactly that mixture::

            surprisal_moments(q=q, order=k)[k] == (share * moments[:, k]).sum()

        which the tests assert rather than take on trust. It is also the
        numerically stable form: it never multiplies by ``M(q)``, so unlike
        :meth:`PseqAnalysis.length_derivatives` it does not overflow below
        ``q = 1``.
        """
        order = _check_int("order", order, 0, BEAM_MAX_MOMENT)
        b, rows, cols = _c.length_derivatives(self._g, float(q), order)
        raw = np.frombuffer(b, dtype=np.float64).reshape(rows, cols).copy()
        moments = raw.copy()
        share = raw[:, 0].copy()
        # Slot 0 carries the share instead of the constant 1 a raw jet would
        # put there, precisely so the mixture identity above is checkable from
        # the output alone. Restore the 1 in the moment view.
        moments[:, 0] = 1.0
        return {"length": np.arange(rows), "share": share, "moments": moments}

    # ── Mellin transform and diversity ──────────────────────────

    def log_mellin(self, q: float) -> float:
        """``ln sum_s P(s)**q``, exactly.

        The generator every diversity quantity here is built from. ``q=0``
        gives the log support size, ``q=1`` gives 0, and the derivatives with
        respect to ``q`` are the cumulants of ``ln P``.
        """
        return float(_c.log_mellin(self._g, float(q)))

    def log_mellin_many(self, orders: Iterable[float]) -> np.ndarray:
        """:meth:`log_mellin` at every order in ``orders``, in one pass.

        Bit-identical to ``[self.log_mellin(q) for q in orders]`` -- the same
        bits, not the same numbers to rounding -- whatever the thread count,
        the memory budget, or which other orders share the call. What changes
        is the work: each length's trellis is traversed once for all the
        orders instead of once per order, with the layer of edge weights built
        once and every order riding it under its own rescaling.

        Orders must be finite. Returns a float64 array the length of
        ``orders``; an empty sequence returns an empty array.
        """
        q = _orders_vector(orders)
        if q.size == 0:
            return np.empty(0, dtype=np.float64)
        return np.frombuffer(_c.log_mellin_many(self._g, q),
                             dtype=np.float64).copy()

    def power_sum(self, q: float) -> float:
        """``sum_s P(s)**q`` on the linear scale.

        Overflows to ``inf`` for ``q`` well below 1, where the support size
        dominates; use :meth:`log_mellin` there.
        """
        return math.exp(self.log_mellin(q))

    def hill_numbers(self, orders: Iterable[float]) -> np.ndarray:
        """Hill numbers at several orders at once.

        One :meth:`hill_number` per order, each its own pass over every
        length. :meth:`hill_profile` computes the same profile from one pass
        per length and agrees with this to machine precision, not bit for bit;
        keep this where a number has to reproduce one already published.
        """
        return np.array([self.hill_number(float(a)) for a in orders])

    def hill_profile(self, orders: Iterable[float]) -> np.ndarray:
        """Hill numbers at every order in ``orders``, from one pass per length.

        Each order takes the route :meth:`hill_number` takes, chosen by the
        same tests: ``q = 0`` is :meth:`path_count`, ``q`` within ``1e-12`` of
        1 is ``exp(entropy())``, and every other order is
        ``exp(log_mellin(q) / (1 - q))`` with all of those orders' Mellin
        values from one :meth:`log_mellin_many` call.

        The two corners are bit-identical to :meth:`hill_numbers`. The rest
        agree with it to MACHINE PRECISION, NOT BIT FOR BIT: ``hill_number``
        reaches ``sum P**q`` through a different tilted sweep
        (``tilted_log_z`` in math/diversity.c), which rescales each layer by a
        different maximum and so rounds differently. Over nine models -- both
        alphabets, both worlds, orders 2 to 4, and a donor-scale model -- the
        two values of ``ln sum P**q`` differ by at most 7.1e-15, and on the
        default grid of 0 to 3 in steps of 0.1 the Hill numbers by at most
        1.1e-14 relative. Closed-world models mostly agree exactly.

        NEAR ``q = 1`` THE RELATIVE DIFFERENCE GROWS as ``1 / |1 - q|``,
        because ``D_q = exp(ln M / (1 - q))`` magnifies any rounding in
        ``ln M`` by that factor on both routes: 1.3e-12 was measured at
        ``q = 1.001`` on a model with every sequence in its support. Neither route is the more
        accurate one there. Use :meth:`hill_numbers` where a profile has to
        reproduce one already published from it to the last bit.

        On a donor-scale model (180,000 sequences, order 4, 16 threads) the
        default 31-order profile takes 0.42 s here against 4.25 s for
        :meth:`hill_numbers`.
        """
        q = _orders_vector(orders)
        out = np.empty(q.size, dtype=np.float64)
        one = np.abs(q - 1.0) < 1e-12        # `fabs(alpha - 1.0) < 1e-12` in C
        zero = q == 0.0
        rest = ~(one | zero)
        if rest.any():
            qr = q[rest]
            lm = self.log_mellin_many(qr)
            out[rest] = [_hill_from_mellin(float(a), float(v))
                         for a, v in zip(qr, lm)]
        if zero.any():
            out[zero] = self.path_count()
        if one.any():
            out[one] = math.exp(self.entropy())
        return out

    def hill_curve(self, orders: Iterable[float] | None = None) -> dict[str, np.ndarray]:
        """The diversity profile as a curve, for plotting."""
        if orders is None:
            orders = np.concatenate([np.linspace(0.0, 1.0, 11)[:-1],
                                     np.linspace(1.0, 4.0, 13)])
        orders = np.asarray(list(orders), dtype=np.float64)
        return {"orders": orders, "hill": self.hill_numbers(orders)}

    def diversity_profile(self) -> dict[str, float]:
        """The three Hill numbers usually quoted together.

        ``D0`` is the support size, ``D1`` the exponential of the entropy, and
        ``D2`` the inverse collision probability.

        .. warning::
           ``D0`` inherits the caveat on :meth:`path_count`: it is the size
           of all of sequence space, which describes the smoother rather than
           the grammar. ``D1`` is the number to quote if you are quoting one.
        """
        return {
            "D0": self.hill_number(0.0),
            "D1": self.hill_number(1.0),
            "D2": self.hill_number(2.0),
            # Spelled as FlashBackGraph spells it. Cross-model summary code
            # reads this key and would otherwise have to special-case BeamModel.
            "entropy_nats": self.entropy(),
        }

    def effective_diversity(self) -> float:
        """``D1``, the effective number of sequences. Alias for readability."""
        return self.hill_number(1.0)

    # ── Surprisal moments ───────────────────────────────────────

    def surprisal_moments(self, *, q: float = 1.0, order: int = 4) -> np.ndarray:
        """Raw moments of the surprisal ``-ln P`` under the ``q``-tilted law.

        Returns ``order + 1`` values, so ``[0]`` is 1 and ``[1]`` is the mean.
        At ``q=1`` the mean is the entropy.
        """
        order = _check_int("order", order, 0, BEAM_MAX_MOMENT)
        return np.asarray(
            _c.surprisal_moments(self._g, float(q), order),
            dtype=np.float64)

    def surprisal_cumulants(self, *, q: float = 1.0, order: int = 4) -> np.ndarray:
        """Cumulants 1..``order`` of the surprisal: mean, variance, and on up."""
        order = _check_int("order", order, 1, BEAM_MAX_MOMENT)
        return np.asarray(
            _c.surprisal_cumulants(self._g, float(q), order),
            dtype=np.float64)

    def surprisal_stats(self, *, q: float = 1.0) -> dict[str, float]:
        """Mean, variance, standard deviation, skewness and excess kurtosis
        of the surprisal, exactly."""
        k = self.surprisal_cumulants(q=q, order=4)
        var = float(k[1])
        sd = math.sqrt(var) if var > 0 else 0.0
        return {
            "mean": float(k[0]),
            "variance": var,
            "std": sd,
            "skewness": float(k[2]) / sd ** 3 if sd > 0 else 0.0,
            "excess_kurtosis": float(k[3]) / var ** 2 if var > 0 else 0.0,
        }

    def cgf(self, t: float) -> float:
        """Cumulant generating function of ``ln P``, ``ln E[exp(t ln P)]``.

        Equal to ``log_mellin(1 + t)``, since tilting by ``t`` and evaluating
        the Mellin transform at ``1 + t`` are the same operation.
        """
        return self.log_mellin(1.0 + float(t))

    # ── Extremes ────────────────────────────────────────────────

    def surprisal_range(self) -> tuple[float, float]:
        """Smallest and largest surprisal any producible sequence attains.

        Exact, by min-plus and max-plus sweeps. Not an estimate from samples.
        """
        return tuple(_c.surprisal_range(self._g))

    def dynamic_range(self, *, base: float = 10.0) -> float:
        """Orders of magnitude of probability the grammar spans."""
        if not (base > 0.0) or base == 1.0:
            raise ValueError(f"base must be positive and not 1, got {base!r}")
        lo, hi = self.surprisal_range()
        return (hi - lo) / math.log(base)

    # ── Per-length decomposition ────────────────────────────────

    def length_marginals(self) -> dict[str, np.ndarray]:
        """Both spectra decomposed by sequence length.

        ``counting`` sums to the support size and ``generated`` to 1, which is
        what makes this a decomposition rather than a summary.
        ``mean_surprisal`` and ``variance`` are ``E[-ln P | L]`` and
        ``Var[-ln P | L]`` within each length.
        """
        c, g, s, v = _c.length_marginals(self._g)
        return {
            "length": np.arange(len(c)),
            "counting": np.asarray(c, dtype=np.float64),
            "generated": np.asarray(g, dtype=np.float64),
            "mean_surprisal": np.asarray(s, dtype=np.float64),
            "variance": np.asarray(v, dtype=np.float64),
        }

    # ── Most probable sequences ─────────────────────────────────

    def top_k_sequences(self, k: int = 20, *,
                        length: int | None = None) -> list[tuple[str, float]]:
        """The ``k`` most probable sequences, with their log probabilities.

        Exact, not a beam. The search is guided by the backward values, which
        are an exact completion weight rather than a heuristic, so the ``n``-th
        result really is the ``n``-th most probable sequence.
        """
        seqs, lp = _c.top_k(self._g, self._check_count("k", k),
                                 self._check_length(length, allow_none=True))
        return list(zip(seqs, lp))

    # ── Saddlepoint and the surprisal distribution ──────────────

    def cgf_jet(self, t: float) -> dict[str, float]:
        """``K(t)``, ``K'(t)`` and ``K''(t)`` for the surprisal, from one sweep.

        With ``Y = -ln P``, ``K(t) = ln E[exp(tY)] = log_mellin(1 - t)``. All
        three come from a single pass over the trellis, which is what makes a
        Newton solve for the saddlepoint affordable.
        """
        lm, mean, var = _c.surprisal_tilt(self._g, 1.0 - float(t))
        return {"K": lm, "K1": mean, "K2": var}

    #: Largest tilt the Newton solve will travel to. Beyond this the tilted
    #: sweep is dominated by a single path and the reconstruction stops being
    #: informative, so the answer is reported at the boundary instead of being
    #: extrapolated from a degenerate sweep.
    MAX_TILT = 25.0

    def _solve_tilt(self, x, cache, *, tol=1e-9, max_iter=80):
        """Solve ``K'(t) = x`` for the tilt, safeguarded.

        ``K''`` is a variance and therefore positive, so ``K'`` is strictly
        increasing and the root is unique. Plain Newton still misbehaves in the
        tails, where ``K''`` collapses and a step can leap past the root, so
        every iterate is kept inside a bracket and a step that escapes it falls
        back to bisection.

        Returns ``(t, jet, converged)``. A point beyond what the clamped tilt
        range can reach is reported as unconverged rather than answered with a
        tilt that does not solve the equation, because Lugannani-Rice built on
        a wrong saddlepoint produces a confident and meaningless number.
        """
        def jet_at(t):
            key = round(t, 12)
            if key not in cache:
                cache[key] = self.cgf_jet(key)
            return cache[key]

        lo_t, hi_t = -self.MAX_TILT, self.MAX_TILT
        j_lo, j_hi = jet_at(lo_t), jet_at(hi_t)
        if x <= j_lo["K1"]:
            return lo_t, j_lo, False
        if x >= j_hi["K1"]:
            return hi_t, j_hi, False

        t = 0.0
        jet = jet_at(t)
        for _ in range(max_iter):
            f = jet["K1"] - x
            if abs(f) < tol:
                return t, jet, True
            if f > 0.0:
                hi_t = t
            else:
                lo_t = t
            step = f / max(jet["K2"], 1e-300)
            nt = t - step
            if not (lo_t < nt < hi_t):
                nt = 0.5 * (lo_t + hi_t)
            t = nt
            jet = jet_at(t)
        return t, jet, abs(jet["K1"] - x) < 1e-6

    def saddlepoint(self, x, *, tol: float = 1e-9, max_iter: int = 80) -> dict:
        """Lugannani-Rice saddlepoint approximation to the surprisal density.

        Returns ``pdf``, ``cdf``, ``sf``, the solved ``t``, and ``converged``
        at each value of ``x`` (a surprisal in nats).

        The cumulant generating function and its derivatives are **exact**,
        from the jet DP; only the inversion from those to a smooth density is
        approximate. Outside :meth:`surprisal_range` the density is exactly
        zero and that is what is returned, rather than an extrapolation.

        Points so far into a tail that the clamped tilt cannot reach them are
        reported at the boundary with ``converged`` false, rather than answered
        from a tilt that does not solve the saddlepoint equation.

        Cost: each solve is a handful of full passes over the trellis, so this
        is seconds per point on a large model. Ask for the points you need, and
        use :meth:`surprisal_cdf` when a cheap answer will do.
        """
        xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
        lo, hi = self.surprisal_range()
        cache: dict[float, dict[str, float]] = {}

        pdf = np.zeros_like(xs)
        cdf = np.zeros_like(xs)
        tt = np.full_like(xs, np.nan)
        ok = np.zeros(xs.shape, dtype=bool)

        for i, xv in enumerate(xs):
            if xv <= lo:
                continue
            if xv >= hi:
                cdf[i] = 1.0
                continue
            t, jet, converged = self._solve_tilt(xv, cache, tol=tol,
                                                 max_iter=max_iter)
            tt[i] = t
            ok[i] = converged
            if not converged:
                # Beyond the reachable tilt range: the point is deep in a tail,
                # and the honest answer is the boundary it is closest to.
                cdf[i] = 0.0 if t < 0 else 1.0
                continue

            k2 = max(jet["K2"], 1e-300)
            pdf[i] = math.exp(jet["K"] - t * xv) / math.sqrt(2.0 * math.pi * k2)
            w = math.copysign(math.sqrt(max(2.0 * (t * xv - jet["K"]), 0.0)), t)
            u = t * math.sqrt(k2)
            if abs(t) < 1e-7:
                # Lugannani-Rice has a removable singularity at the mean; use
                # the standard limiting form rather than dividing by zero.
                k3 = float(self.surprisal_cumulants(q=1.0, order=3)[2])
                cdf[i] = 0.5 - k3 / (6.0 * math.sqrt(2.0 * math.pi) * k2 ** 1.5)
            elif abs(w) < 1e-10 or abs(u) < 1e-10:
                # Degenerate away from the mean, which the limiting form above
                # does NOT cover: it is the expansion around t = 0 and would
                # answer ~0.5 for a point deep in a tail. `w` vanishes only at
                # the mean, so reaching here with a non-small `t` means the
                # tilted sweep has collapsed onto a single path and the
                # inversion has nothing left to invert. Report that rather
                # than a confident number.
                ok[i] = False
                cdf[i] = 0.0 if t < 0 else 1.0
                pdf[i] = 0.0
            else:
                cdf[i] = _normal_cdf(w) + _normal_pdf(w) * (1.0 / w - 1.0 / u)
            cdf[i] = min(max(cdf[i], 0.0), 1.0)

        out = {"x": xs, "pdf": pdf, "cdf": cdf, "sf": 1.0 - cdf, "t": tt,
               "converged": ok, "range": (lo, hi)}
        if _is_scalar_arg(x):
            out = {k: (v[0] if isinstance(v, np.ndarray) and v.size == 1 else v)
                   for k, v in out.items()}
            out["x"] = float(out["x"])
            for key in ("cdf", "pdf", "sf", "t"):
                out[key] = float(out[key])
            out["converged"] = bool(out["converged"])
        return out

    def surprisal_distribution(self) -> dict[str, np.ndarray]:
        """The surprisal as a mixture over lengths, with exact components.

        Each length contributes a component whose weight is ``pi(L)`` and whose
        mean and variance come from the exact jet DP. The Gaussian shape
        *within* a component is the only approximation; the weights and moments
        are not approximate at all.

        The cheap, interpretable counterpart to :meth:`saddlepoint`: one pass
        per length rather than a Newton solve per evaluation point.
        """
        lm = self.length_marginals()
        keep = lm["generated"] > 0
        return {
            "length": lm["length"][keep],
            "weight": lm["generated"][keep],
            "mean": lm["mean_surprisal"][keep],
            "variance": lm["variance"][keep],
            "std": np.sqrt(lm["variance"][keep]),
        }

    def surprisal_pdf(self, x) -> np.ndarray:
        """Density of the surprisal from the length mixture.

        Approximate in the same single way :meth:`surprisal_distribution` is,
        and far cheaper than :meth:`saddlepoint`.

        Returns a ``float`` for a scalar ``x`` and an array otherwise, which is
        the convention :meth:`saddlepoint` and
        :meth:`Calibration.pvalue` follow too.
        """
        d = self.surprisal_distribution()
        xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
        out = np.zeros_like(xs)
        for w, mu, sd in zip(d["weight"], d["mean"], d["std"]):
            if sd <= 0:
                continue
            z = (xs - mu) / sd
            out += w * np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))
        return float(out[0]) if _is_scalar_arg(x) else out

    def surprisal_cdf(self, x) -> np.ndarray:
        """Distribution function of the surprisal from the length mixture.

        Returns a ``float`` for a scalar ``x`` and an array otherwise.
        """
        d = self.surprisal_distribution()
        xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
        out = np.zeros_like(xs)
        for w, mu, sd in zip(d["weight"], d["mean"], d["std"]):
            if sd <= 0:
                out += w * (xs >= mu)
                continue
            out += w * np.array([_normal_cdf((v - mu) / sd) for v in xs])
        return float(out[0]) if _is_scalar_arg(x) else out

    # ── Where the surprisal sits ────────────────────────────────

    def surprisal_by_depth(self, *, q: float = 1.0) -> dict[str, Any]:
        """Where in the sequence the surprisal is actually incurred.

        The exact analogue of asking which edges of a graph carry the
        information: the conserved termini cost almost nothing and the junction
        carries it.

        Returns ``by_depth``, the expected contribution of each layer, and
        ``length_constant``, the per-length normalisation
        ``E[ln Z(L) - ln pi(L)]``. That constant belongs to the choice of
        length rather than to any position, so it is reported separately
        instead of smeared across depths. ``total`` is their sum and equals the
        mean surprisal exactly, which is the identity worth checking.
        """
        depths, konst = _c.surprisal_by_depth(self._g, float(q))
        depths = np.asarray(depths, dtype=np.float64)
        return {
            "depth": np.arange(len(depths)),
            "by_depth": depths,
            "length_constant": float(konst),
            "total": float(depths.sum() + konst),
        }

    # ── Topology ────────────────────────────────────────────────

    @property
    def is_dag(self) -> bool:
        """Always true, and true by construction rather than by inspection.

        The trellis is layered by position, so no edge can ever point backward.
        Exposed because every analytic method here depends on it.
        """
        return True

    def topology(self, length: int) -> dict[str, Any]:
        """Live nodes and edges of the trellis at one length.

        Live means both forward-reachable and able to reach a valid ending, so
        this is the size of the graph a walk can actually traverse rather than
        the size of the state space. Every state is live at every layer,
        which is the support-size fact seen from the other side.

        Raises:
            ValueError: if ``length`` is below 1, above ``max_length``, or a
                length the model does not generate. There is no pooled
                all-lengths trellis, so ``0`` is not accepted here even though
                it means "all lengths" elsewhere.
        """
        length = self._check_length(length)
        if length < 1:
            raise ValueError(
                "topology() needs a specific length of at least 1; there is "
                "no pooled-over-all-lengths trellis to describe")
        if not math.isfinite(self.log_length_prior(length)):
            raise ValueError(
                f"the model generates no sequences of length {length}, so its "
                f"trellis has no live structure; length_pmf() shows which "
                f"lengths it does generate")
        nodes, edges, layers = _c.topology(self._g, length)
        layers = np.asarray(layers, dtype=np.int64)
        return {
            "length": int(length),
            "n_nodes": int(nodes),
            "n_edges": int(edges),
            "nodes_by_layer": layers,
            "max_layer_nodes": int(layers.max()) if layers.size else 0,
            "mean_layer_nodes": float(layers.mean()) if layers.size else 0.0,
            "density": float(edges / (nodes * self.n_residues)) if nodes else 0.0,
        }

    # ── Positional marginals ────────────────────────────────────

    #: Byte that leaves a position free in the clamp buffer handed to C.
    #: Mirrors ``BEAM_RESIDUE_INVALID``, the one value that can never be a
    #: residue index, so "no residue" and "not a residue" are the same byte.
    _CLAMP_FREE = 0xFF

    def _clamp_buffer(self, length: int, clamp) -> bytes | None:
        """Turn a ``{position: residue}`` mapping into the C clamp buffer.

        Returns ``None`` for an empty or absent clamp, so the C side takes the
        unclamped path rather than one that happens to pin nothing.

        Every position is checked against the length and every residue against
        the alphabet HERE, because the C layer sees only bytes: a position of
        ``-1`` would wrap to four billion and a residue of ``'B'`` would be a
        silent index into a neighbouring cell.
        """
        if clamp is None:
            return None
        try:
            items = list(clamp.items())
        except AttributeError:
            raise TypeError(
                f"clamp must be a mapping of position to residue, got "
                f"{type(clamp).__name__}") from None
        if not items:
            return None
        symbols = alphabets.symbols(self.alphabet)
        buf = bytearray([self._CLAMP_FREE]) * length
        for pos, residue in items:
            # __index__ rather than int(): int(4.7) silently truncates to 4,
            # and a clamp on "position 4.7" is a typo, not a request.
            try:
                p = operator.index(pos)
            except TypeError:
                raise TypeError(
                    f"clamp positions must be integers, got {pos!r}") from None
            if not 0 <= p < length:
                raise ValueError(
                    f"clamp position {p} is outside a length-{length} "
                    f"sequence, whose positions are 0..{length - 1}")
            if isinstance(residue, str):
                if len(residue) != 1 or residue not in symbols:
                    raise ValueError(
                        f"clamp at position {p} names {residue!r}, which is "
                        f"not one of the {self.alphabet!r} residues {symbols!r}")
                idx = symbols.index(residue)
            else:
                try:
                    idx = int(residue)
                except (TypeError, ValueError):
                    raise TypeError(
                        f"clamp values must be residue characters or indices, "
                        f"got {residue!r}") from None
                if not 0 <= idx < len(symbols):
                    raise ValueError(
                        f"clamp at position {p} names residue index {idx}, "
                        f"which is outside 0..{len(symbols) - 1}")
            # Unreachable from a plain dict, where equal keys collapse, but
            # `items()` on an arbitrary mapping may yield a position twice and
            # the second write would silently win.
            if buf[p] != self._CLAMP_FREE and buf[p] != idx:
                raise ValueError(
                    f"clamp pins position {p} to two different residues")
            buf[p] = idx
        return bytes(buf)

    def positional_marginals(self, length: int, *,
                             clamp: dict | None = None) -> np.ndarray:
        """``P(x_i = a | length)`` for every position, exactly.

        Returns an ``(length, n_residues)`` array whose ``[i, a]`` entry is the
        probability that a length-``length`` sequence drawn from the model
        carries residue ``a`` at position ``i``. Columns are in
        :attr:`BEAM_ALPHABET` order, which is what
        ``alphabets.symbols(self.alphabet)`` returns.

        Args:
            length: the sequence length to condition on. Marginals are a
                per-length object: position 5 of a length-12 sequence and
                position 5 of a length-18 one are different distances from the
                C terminus, and the model scores them from different cells, so
                pooling them would average two things that are not the same
                quantity. There is deliberately no pooled form.
            clamp: an optional ``{position: residue}`` mapping pinning
                positions, e.g. ``{4: 'G'}``. The result is then the exact
                conditional ``P(x_i = a | length, clamp)``. Residues may be
                given as characters or as indices into the alphabet.

        Exact, and in a stronger sense than :meth:`spectrum` is. Both avoid
        sampling, but the spectrum splits mass linearly onto a GRID at every
        layer, so its bins carry a controlled error. A marginal has no grid to
        be wrong about: the axis is twenty residues, and the answer is the
        per-layer edge posterior of the trellis summed over states, which is a
        finite sum evaluated in full. Rows sum to 1 to the last bits, and that
        is worth checking rather than trusting -- nothing renormalises them.

        A clamped row comes back one-hot. That is the correct conditional
        rather than a placeholder: given that position 4 is ``G``, the
        probability that it is ``G`` is 1.

        No floor is applied. Under the hierarchical prior no residue is
        impossible, so one the corpus never put at a position comes back
        small, at the probability its shorter contexts give it.

        Raises:
            ValueError: if the model generates no sequences of this length --
                loudly, rather than returning a row of zeros that reads like a
                flat distribution -- or if ``clamp`` pins a configuration of
                probability zero, which is a question with no answer rather
                than a division by zero to be smoothed away.

        Example:
            >>> p = model.positional_marginals(14)
            >>> p.shape
            (14, 20)
            >>> float(p.sum(axis=1).min())          # every row normalised
            1.0
            >>> q = model.positional_marginals(14, clamp={4: 'G'})
            >>> float(q[4, BEAM_ALPHABET.index('G')])
            1.0
        """
        length = self._check_length(length)
        if length < 1:
            raise ValueError(
                "positional_marginals() needs a specific length of at least 1; "
                "marginals are per-length and there is no pooled form")
        if not math.isfinite(self.log_length_prior(length)):
            raise ValueError(
                f"the model generates no sequences of length {length}, so it "
                f"has no positional marginals there; length_pmf() shows which "
                f"lengths it does generate")
        buf = self._clamp_buffer(length, clamp)
        w, n_res, _logz = _c.positional_marginals(self._g, length, buf)
        return np.frombuffer(w, dtype=np.float64).reshape(length, n_res).copy()

    def positional_banded(self, length: int, edges, *,
                          counting: bool = False, bins: int = 512,
                          spacing: float = 0.0) -> BandedMarginals:
        """Per-position residue distributions, one set per band of surprisal.

        What a typical sequence of this length looks like given that it is
        also this rare. :meth:`positional_marginals` answers the same
        question pooled over all rarities and is what a single band covering
        everything returns.

        Args:
            length: the sequence length. Marginals are per-length.
            edges: band edges in nats of surprisal (``-ln P(s)``), strictly
                ascending. ``n`` edges define ``n - 1`` half-open bands.
                Mass outside the outermost edges belongs to no band; pass
                ``0`` and a large number to catch everything. Thinking in
                probabilities, an edge is ``-log(p)``, so bands ascend in
                rarity as ``p`` falls.
            counting: weight every producible sequence 1 rather than by its
                own probability -- the two measures :meth:`spectrum` offers.
            bins, spacing: the surprisal grid, sized as the spectrum's is.
                ``spacing <= 0`` chooses it automatically.

        Returns:
            :class:`~BEAM._banded_result.BandedMarginals`.

        Unlike :meth:`positional_marginals` this is NOT exact. The residue
        axis still is, but a band conditions on a real-valued quantity, so
        surprisal is binned and a sequence near an edge can fall either side.
        Check ``result.resolution_error`` before trusting narrow bands.

        **How many bins a band needs.** Linear splitting spreads one
        sequence over ``length + 1`` adjacent bins, so what governs the
        answer is that kernel against the width of a band::

            kernel  = (length + 1) * spacing
            spacing = accumulator_max * 1.02 / (bins - max_length - 9)

        Keep ``kernel`` well under the narrowest band -- a fifth of it is
        comfortable -- which rearranges to a bin count::

            bins >~ 5 * (length + 1) * accumulator_max / band_width

        Two properties of a model drive that, and neither is its size. The
        first is ``accumulator_max``, the surprisal range the grid must span:
        the more it spans, the wider each bin is at a fixed ``bins``. The
        second is how much mass sits within a kernel of a band edge -- a
        small model whose spectrum is a few dozen spikes has edges that land
        in the gaps between them and pays nothing for a coarse grid, while a
        model fitted to hundreds of millions of sequences is a continuum and
        every edge cuts through mass.

        The prior widens the first: a context the corpus never saw can still
        be spelled, improbably, and the grid has to reach it. But a model
        without that tail is not therefore safe: the foundation below, built
        before the prior, still needed eight times the default.
        ``resolution_error`` is what settles it either way.

        How many bins a band needs, measured on the 927M-sequence foundation
        (order 4, built under the previous closed-world rule) at its modal length 15,
        with quartile bands about
        2.4 nats wide::

            bins   spacing   resolution_error   scratch   time
            1024     0.122         3.4%           16 GB    7.7s
            2048     0.060         2.2%           31 GB     10s
            4096     0.030         1.1%           61 GB     19s
            8192     0.015        0.44%          119 GB     37s

        Halving the spacing halves the error, which is what says it is the
        grid and not a defect. The band fractions are stable to a tenth of a
        point from 4096 on, so that is the setting to reach for there.
        Working the rule above: that model spans 117.9 nats, so at the
        default 512 bins its kernel is 4.04 nats against bands of 2.4 -- the
        smoothing is WIDER than the band it is meant to resolve, and no
        amount of care in the caller recovers that.

        Scratch is refused before it is allocated, and the refusal names the
        figure -- so the cheap way to size a run is to ask for it under a
        small :func:`~BEAM.set_memory_budget` and read the number back.

            >>> import numpy as np
            >>> b = model.positional_banded(14, [0, 30, 34, 1e12])
            >>> b.fraction.round(3)
            array([0.412, 0.317, 0.271])
            >>> b.marginals[0].shape
            (14, 20)
        """
        length = self._check_length(length)
        if length < 1:
            raise ValueError(
                "positional_banded() needs a specific length of at least 1; "
                "marginals are per-length and there is no pooled form")
        if not math.isfinite(self.log_length_prior(length)):
            raise ValueError(
                f"the model generates no sequences of length {length}, so it "
                f"has no banded marginals there; length_pmf() shows which "
                f"lengths it does generate")

        e = np.ascontiguousarray(np.asarray(edges, dtype=np.float64).ravel())
        if e.size < 2:
            raise ValueError(
                f"{e.size} band edge(s) define no band; two ascending edges "
                f"define one")
        if not np.all(np.isfinite(e)):
            raise ValueError(
                "band edges must be finite; a band is bounded in nats at both "
                "ends, so pass a large number rather than inf for an open top")
        if not np.all(np.diff(e) > 0.0):
            bad = int(np.argmin(np.diff(e)))
            raise ValueError(
                f"band edges must ascend strictly, but edge {bad} is "
                f"{e[bad]!r} and edge {bad + 1} is {e[bad + 1]!r}; if you are "
                f"thinking in probabilities, remember -log(p) reverses the "
                f"order")
        bins = self._check_bins(bins, spacing if spacing > 0 else None)

        marg, bmass, pmass, n_res, n_bands, used = _c.positional_banded(
            self._g, length, bool(counting), bins, float(spacing), e.tobytes())
        return BandedMarginals(
            np.frombuffer(marg, dtype=np.float64)
              .reshape(n_bands, length, n_res).copy(),
            np.frombuffer(bmass, dtype=np.float64).copy(),
            np.frombuffer(pmass, dtype=np.float64)
              .reshape(n_bands, length).copy(),
            e, used, bool(counting), length)

    def log_clamp_probability(self, length: int, clamp: dict) -> float:
        """``ln P(clamp | length)``, the exact mass the clamp keeps.

        The partition function of the clamped trellis divided by the
        unclamped one. It is the natural companion to a clamped
        :meth:`positional_marginals` call: the marginals say what the
        conditional looks like, and this says how much of the length's
        probability the condition selects, which is what decides whether the
        conditional describes anything worth looking at.

        Returns ``-inf`` for a configuration the model cannot produce, rather
        than raising, because "this cannot happen" is the answer to the
        question rather than a failure to answer it. That is the one place
        these two methods differ: a marginal conditional on an impossible event
        does not exist, so :meth:`positional_marginals` raises instead.
        """
        length = self._check_length(length)
        if not math.isfinite(self.log_length_prior(length)):
            raise ValueError(
                f"the model generates no sequences of length {length}")
        buf = self._clamp_buffer(length, clamp)
        if buf is None:
            return 0.0
        _w, _n, logz = _c.positional_marginals(self._g, length, buf, True)
        return float(logz) - self.log_partition(length)

    # ── Expected cell usage ─────────────────────────────────────

    #: Block order of the C result: anchor major, expert minor.
    _EXPECTED_BLOCKS = ("forward", "backward")

    def expected_counts(self, *, length: int | None = None) -> dict[str, Any]:
        """Exact expected usage of every table cell under the model itself.

        The same edge posterior :meth:`positional_marginals` sums over states,
        aggregated instead into the cell each edge would have incremented
        during training. It is the E step of an EM iteration written out
        exactly: the model's expected sufficient statistics under its own
        distribution, with no sampling anywhere in the derivation.

        Args:
            length: a single length, or ``None`` to pool over every length the
                model generates, each weighted by its own prior ``pi(L)``.
                Pooled, the answer is the expected usage per sequence DRAWN
                FROM THE MODEL, so multiplying by :attr:`total_weight` puts it
                on the same scale as the counts the corpus produced. It is not
                multiplied here, because the per-sequence form is the one that
                has a check attached to it: see ``expected_length`` below.

        Returns a dict with ``expected_length``, ``n_residues``, ``residues``,
        ``working_set_bytes``, and six blocks keyed ``forward_wide``,
        ``forward_mid``, ``forward_len``, ``backward_wide``, ``backward_mid``,
        ``backward_len``.
        Each block is itself a dict of

            ``rows``    dense row indices, ascending, ``(n_rows,)`` int64
            ``counts``  ``(n_rows, n_residues)`` expected counts
            ``depth``   the row's distance from its anchor
            ``context`` the row's packed context index
            ``length``  the row's sequence length, ``len`` blocks only

        SPARSE BY ROW, and that is forced rather than chosen: an order-4 model
        has ~148 million cells and the support of this expectation is a few
        tens of thousands of rows, so a dense block would be gigabytes of
        zeros. A row that does not appear has expected count exactly zero.

        Every block sums to ``expected_length`` -- ``length`` when one is
        given, and ``E[L]`` when pooled -- because each position of a sequence
        increments exactly one cell of each of the two tables. That identity is
        the cheapest check that the sweep covered the sink and the first ``K``
        layers, and it is returned so it can be run rather than assumed.

        Contexts are packed most-distant-first in BOTH anchors, matching the
        count tables. The finalized C-anchored wide table is repacked
        nearest-first for cache reasons; these are counts, so they are not.

        ``working_set_bytes`` is the most scratch the C sweep held at once,
        MEASURED rather than estimated, and it is reported because it cannot
        be predicted: the accumulators are keyed on the ``(depth, context)``
        pairs the trellis actually reaches, so their size is a property of the
        answer, not of the model's shape. On the published foundation model it
        is over a gigabyte -- larger than the model. It is charged against
        :func:`BEAM.set_memory_budget`, so a call that would exceed the budget
        raises :class:`BudgetExceeded` instead of taking the machine down.

        .. note::
           This is the most expensive analytic here: a full forward-backward
           sweep per length, over every state of every layer with no pruning.
           Ask for one ``length`` when that is what you need, and read
           ``working_set_bytes`` before asking for the pooled answer on a wide
           model.
        """
        want = 0 if length is None else self._check_length(length)
        if want != 0 and not math.isfinite(self.log_length_prior(want)):
            raise ValueError(
                f"the model generates no sequences of length {want}, so it "
                f"uses no cells there; length_pmf() shows which lengths it "
                f"does generate")
        blocks, n_res, total, working_set = _c.expected_counts(self._g, want)

        cfg = self.config
        radix = n_res + 1
        n_ctx = radix ** int(cfg["order"])
        out: dict[str, Any] = {
            "expected_length": float(total),
            "n_residues": int(n_res),
            "residues": alphabets.symbols(self.alphabet),
            "working_set_bytes": int(working_set),
        }
        # Each block's bytes are dropped as soon as its arrays exist, rather
        # than held until the loop ends. `np.frombuffer` is a view and `.copy()`
        # is the array the caller keeps, so holding both buffers to the end
        # means two copies of the whole answer are resident at once -- 112 MB
        # against 112 MB on a 24-length order-3 model, and over a gigabyte on
        # the foundation model. Clearing the list entry is what releases them:
        # `blocks` is the only reference the C layer left behind.
        for i, anchor in enumerate(self._EXPECTED_BLOCKS):
            rows_b, vals_b = blocks[i]
            blocks[i] = None
            rows = np.frombuffer(rows_b, dtype=np.uint32).astype(np.int64)
            del rows_b
            counts = np.frombuffer(vals_b, dtype=np.float64).reshape(
                len(rows), n_res).copy()
            del vals_b
            # row = depth * A**order + context, exactly as indexing.h has it.
            block: dict[str, np.ndarray] = {
                "rows": rows,
                "counts": counts,
                "context": rows % n_ctx,
                "depth": rows // n_ctx,
            }
            out[anchor] = block
        return out

    # ── Generation ──────────────────────────────────────────────

    def sample(
        self,
        n: int,
        *,
        length: int | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Draw ``n`` sequences exactly in proportion to their probability.

        Args:
            length: A single length, or ``None`` to draw it from the length
                prior.
            seed: Seed for reproducibility.
        """
        return _c.sample(
            self._g, self._check_count("n", n),
            self._check_length(length, allow_none=True),
            0 if seed is None else int(seed))

    def simulate(
        self,
        n: int,
        *,
        length: int | None = None,
        seed: int | None = None,
    ) -> SimulationResult:
        """:meth:`sample`, packaged with each draw's log probability.

        Returns the same container the other models return, so downstream code
        that consumes a ``SimulationResult`` works unchanged. ``n_tokens`` is
        the sequence length in residues, which is what a "token" is for a model
        that emits one residue per step.
        """
        seqs = self.sample(n, length=length, seed=seed)
        return SimulationResult(
            sequences=seqs,
            log_probs=self.log_pseq_batch(seqs),
            n_tokens=np.array([len(s) for s in seqs], dtype=np.int64),
        )

    # ── Length-conditioned anomaly scoring ──────────────────────

    # ── Spectrum analytics under the graph models' names ────────

    @property
    def insights(self):
        """Derived, plottable views of this model -- see :mod:`BEAM.insights`.

        An accessor rather than a method per quantity, so the package can
        grow without `BeamModel` growing a method for everything anyone ever
        wants derived from one.

        Example:
            >>> s = model.insights.logos(15, pseq_below=1e-9)
            >>> s[0].consensus_sequence
            'CASSLGGGGGYEQYF'
        """
        from ..insights import _Insights
        return _Insights(self)

    def pseq_analysis(self) -> PseqAnalysis:
        """An :class:`PseqAnalysis` over this model.

        The accessor the graph models in this library expose, so spectrum,
        discovery and length-marginal code written against
        ``graph.pseq_analysis()`` reads across to BeamModel unchanged. The object
        caches its reconstructions, so asking for the same spectrum twice
        costs one transport.

        Example:
            >>> a = model.pseq_analysis()
            >>> h = a.histogram(256, measure="counting")
            >>> round(h.total_mass / model.path_count(), 6)
            1.0
        """
        # Imported here rather than at module scope: this class is
        # constructed with the model, so a top-level import would be a
        # cycle.
        from ..analytics.pseq import PseqAnalysis
        return PseqAnalysis(self)

    def scale_score(self, sequence, calibration: Calibration, *,
                    method: str = "pvalue"):
        """Length-normalised anomaly score; higher means more surprising.

        ``-log10 P(U >= -ln P(s))`` under ``calibration``, which is the model's
        own EXACT per-length surprisal law, reconstructed by dynamic
        programming rather than measured off simulated draws. There is no
        ``n_sim`` and no minimum-count rule, so there is nothing to tune and no
        sampling noise in the reference.

        Because the reference is the exact null, a threshold here IS a false
        positive rate: 2 means 1% at every length. ``method="robust"`` returns
        the historical ``(u - median_L) / IQR_L`` instead, whose cutoffs are a
        different rate at every length. See :meth:`Calibration.scale`.

        This scores the sequence as a whole. For WHICH RESIDUES look wrong, and
        for the two-channel index that sees deletions, use
        :func:`BEAM.analytics.scale.attribute` and :class:`ScaleIndex`.

        A length the calibration does not cover falls back to the pooled null,
        weighted by the length prior.

        Args:
            sequence: A single string, or a sequence of strings.
            calibration: From :meth:`calibrate`.
            method: ``"pvalue"`` (default) or ``"robust"``.

        Returns:
            ``float`` for a single string, ``np.ndarray`` otherwise.
        """
        # Local: constructed with the model, so a top-level import cycles.
        from ..analytics.calibration import Calibration
        if not isinstance(calibration, Calibration):
            raise TypeError(
                f"calibration must be an Calibration from "
                f"BeamModel.calibrate(), got {type(calibration).__name__}")
        single = isinstance(sequence, str)
        seqs = [sequence] if single else list(sequence)
        u = -self.log_pseq_batch(seqs)
        lengths = np.fromiter((len(x) for x in seqs), dtype=np.int64,
                              count=len(seqs))
        out = np.array([calibration.scale(int(L), float(v), method=method)
                        for L, v in zip(lengths, u)], dtype=np.float64)
        return float(out[0]) if single else out

    def calibrate(self, *, bins: int = 512) -> Calibration:
        """Build the EXACT length-conditioned null for the surprisal.

        For each length the model can produce, reconstructs the distribution of
        ``-ln P`` among sequences of exactly that length by dynamic programming
        over the DAG. No simulation, so no sampling noise, no minimum-count
        rule, and nothing to tune: a length is covered if it has support.

        The null is the model's own generated law restricted to a length, which
        is the distribution a sequence drawn from the model follows. That makes
        a p-value here exactly "how surprising is this sequence for its length,
        under this model", with no free parameter standing between the question
        and the answer.

        Args:
            bins: Resolution of the surprisal grid. The only knob, and it
                trades cost against the smallest tail probability that can be
                resolved — see :attr:`Calibration.resolution_floor`.
                Must be at least ``max_length + 10``, the same floor
                :meth:`spectrum` applies when it chooses its own spacing.

        Raises:
            ValueError: if ``bins`` is below that floor, or if no length has
                support.
        """
        # Imported here rather than at module scope: this class is
        # constructed with the model, so a top-level import would be a
        # cycle.
        from ..analytics.calibration import Calibration
        # The grid is sized from the accumulator bound rather than from a
        # probe spectrum: a max-plus sweep is far cheaper than a full
        # transport, and it is the correct bound (see accumulator_max).
        bins = self._check_bins(bins, None)
        hi = self.accumulator_max()
        if not (hi > 0):
            raise ValueError("model has no producible sequences")
        spacing = hi * 1.02 / (bins - (int(self.config["max_length"]) + 9))

        survival: dict[int, np.ndarray] = {}
        # pi(L) restricted to the covered lengths: what a pooled fallback null
        # has to be weighted by, so an uncovered length is scored against the
        # mixture the model actually generates rather than against a flat one.
        prior: dict[int, float] = {}
        for L in range(len(self.length_pmf())):
            if not math.isfinite(self.log_length_prior(L)):
                continue
            w = self.spectrum(measure="generated", length=L, bins=bins,
                              spacing=spacing)["weights"]
            total = w.sum()
            if total <= 0:
                continue
            # Survival is P(U >= u), so the tail accumulates from the top.
            survival[L] = np.cumsum(w[::-1])[::-1] / total
            prior[L] = total

        if not survival:
            raise ValueError("no length has support; nothing to calibrate")
        return Calibration(
            surprisal=np.arange(bins) * spacing,
            spacing=spacing, survival=survival, prior=prior)

    # ── Set operations ──────────────────────────────────────────

    def add(self, other: BeamModel) -> None:
        """Add another model's counts into this one, in place.

        The inverse of :meth:`subtract`, and the reason both exist: counts are
        additive, so a corpus can be counted in shards and combined without
        re-reading anything. The model is re-finalized afterwards.

        Both models must share ``order`` and ``max_length``;
        the smoothing parameters need not match, because counts do not depend
        on them and this model's own smoothing is what gets reapplied.
        """
        self._check_operand("add", other)
        _c.add(self._g, other._g)

    def subtract(self, other: BeamModel) -> None:
        """Subtract another model's counts from this one, in place.

        Counts are additive, which is what makes holding a cohort out of a
        foundation model a subtraction rather than a recount of the corpus.
        Both models must share ``order`` and ``max_length``.
        The model is re-finalized afterwards.

        .. warning::
           :attr:`n_observed` and :attr:`n_skipped` are NOT decremented — the
           C layer adjusts them on :meth:`add` and leaves them alone here — so
           after a subtraction they count everything ever added rather than
           what is currently in the tables. :attr:`total_weight` *is* adjusted
           and is the number to trust.

        .. warning::
           If the subtraction empties the tables the re-finalize fails and the
           model is left unusable: the counts are already gone by then and
           there is no rollback. Build a fresh model rather than subtracting a
           cohort out of itself.
        """
        self._check_operand("subtract", other)
        _c.subtract(self._g, other._g)

    def posterior(self, prior: BeamModel, kappa: float = 100.0) -> None:
        """Fold a prior model's belief into this model's counts, in place.

        A Dirichlet-multinomial update. Every row of every count table is one
        categorical distribution — over the alphabet's residues for the three
        expert tables, over lengths for the length histogram — and this adds
        ``kappa * p_prior[row]`` to each of this model's rows, where
        ``p_prior`` is that row of ``prior`` NORMALISED. A row then reads::

            p_post = (c_own + kappa * p_prior) / (n_own + kappa)

        before BeamModel's own smoothing, which is the convention
        :meth:`FlashBackGraph.posterior` uses for graph edges.

        This is the personalisation primitive: a foundation model as the
        prior, one donor's own sequences as the counts, and ``kappa`` as how
        many pseudo-observations of the foundation that donor's data has to
        argue against.

        ``kappa = 0`` is a no-op. A row the prior never observed is left
        alone: an empty row carries no belief, and spreading ``kappa``
        uniformly over it would invent one.

        The sequence counters do not move. :attr:`total_weight` and
        :attr:`n_observed` report what was OBSERVED, and a pseudo-count is not
        an observation, so a corpus size does not shift when the smoothing
        strength does.

        Both models must share ``order`` and ``max_length``,
        and neither may be compacted. The model is re-finalized afterwards.

        Example:
            >>> donor = BeamModel(donor_sequences, order=2, max_length=10)
            >>> donor.posterior(foundation, kappa=200.0)
        """
        self._check_operand("posterior", prior)
        if not (float(kappa) >= 0.0):
            raise ValueError(f"kappa must be non-negative, got {kappa!r}")
        _c.posterior(self._g, prior._g, float(kappa))

    def set_alpha(self, alpha: float | Iterable[float] | None = None,
                  alpha_length: float | None = None) -> None:
        """Change the hierarchical prior's strengths in place, and re-finalize.

        Nothing is re-observed: the counts are the only state the prior reads,
        so this is what makes fitting the strengths cheap -- one model, many
        re-finalizes. ``None`` keeps the current setting; ``"auto"`` hands the
        strength back to the data (estimated by the evidence of the counts);
        numbers fix it. See :func:`BEAM.fit_alpha`.
        """
        if self._compacted:
            raise RuntimeError(
                "set_alpha() is unavailable after compact(): the counts the "
                "prior is re-applied to have been released")
        from .config import _alpha_length, _alpha_levels
        cfg = self.config
        if alpha is None:
            levels = None if cfg["alpha_auto"] else cfg["alpha"]
        else:
            levels = _alpha_levels(alpha, cfg["order"])
        if alpha_length is None:
            al = None if cfg["alpha_length_auto"] else cfg["alpha_length"]
        else:
            al = _alpha_length(alpha_length)
        _c.set_prior(self._g, levels, al)

    def with_prior(self, prior: BeamModel, kappa: float = 100.0) -> BeamModel:
        """A NEW model equal to this one updated toward ``prior``.

        The non-destructive :meth:`posterior`, and the form a sweep over
        ``kappa`` wants: the individual's own counts survive each step.

        Example:
            >>> curve = [donor.with_prior(foundation, kappa=k)
            ...          for k in (0.0, 10.0, 100.0)]
        """
        out = self.copy()
        out.posterior(prior, kappa)
        return out

    def copy(self) -> BeamModel:
        """An independent model with the same counts and configuration.

        Built by adding this model's counts into a fresh empty one, so it
        shares no storage: the copy can be subtracted from without touching
        the original. That matters because :meth:`subtract` is destructive and
        the counts are the only copy there is.
        """
        if self._compacted:
            raise RuntimeError(
                "copy() is unavailable after compact(): the raw counts it "
                "duplicates have been released")
        dst = _c.new(*_config_args_from_dict(self.config))
        _c.add(dst, self._g)
        return BeamModel._from_capsule(dst)

    def tempered(self, t: float) -> BeamModel:
        """A NEW model holding ``P_t(s) proportional to P(s)**t``.

        The tempered law already drives :meth:`surprisal_moments`, the spectrum
        and :meth:`saddlepoint`, where it is a measure a sweep carries. This
        makes it a model, so everything the ordinary one answers it answers
        too. ``t > 1`` concentrates the distribution on its likeliest
        sequences and ``t < 1`` flattens it toward the rare ones; the SUPPORT
        is unchanged either way, so :meth:`path_count` does not move.

        Exact. The result satisfies
        ``log_pseq_t(s) == t * log_pseq(s) - log_mellin(t)`` identically, and
        its whole Mellin transform follows from this model's:
        ``log_mellin_t(q) == log_mellin(t*q) - q*log_mellin(t)``.

        ``t`` must be finite and strictly positive. At ``t = 0`` every finite
        log weight becomes zero and every ``-inf`` becomes NaN, and a negative
        ``t`` turns an unspellable edge into the likeliest one.

        The result arrives with its COUNTS RELEASED, as after
        :meth:`compact`, and that is a correctness guard rather than a saving:
        a tempered model's distribution is not a function of its counts, so
        :meth:`add`, :meth:`subtract`, :meth:`posterior` and :meth:`copy`
        would each silently rebuild the untempered tables. They refuse
        instead. It is also why tempering does not yet compose by chaining --
        ``tempered`` on a tempered model is refused, not approximated -- and
        carrying the exponent in the configuration is what would lift that.
        """
        out = self.copy()
        _c.temper(out._g, float(t))
        out._compacted = True
        return out

    def merged(self, other: BeamModel) -> BeamModel:
        """A NEW model holding this model's counts plus ``other``'s.

        The non-destructive :meth:`add`.
        """
        self._check_operand("merged", other)
        out = self.copy()
        out.add(other)
        return out

    def without(self, other: Any, *, abundances: Iterable[float] | None = None,
                ) -> BeamModel:
        """A NEW model with ``other``'s contribution removed. Leaves this one
        intact.

        This is the leave-one-cohort-out primitive. ``other`` may be another
        :class:`BeamModel`, or the cohort's sequences themselves, in which case a
        matching model is built from them first.

        Removing a cohort by SUBTRACTING ITS COUNTS is not the same as
        rebuilding without its sequences, and the difference is the point: a
        sequence the held-out cohort shares with the rest of the corpus keeps
        the rest of the corpus's evidence instead of vanishing along with it.

        Example:
            >>> foundation = BeamModel(corpus, order=2, max_length=10)
            >>> held_out = foundation.without(cohort_sequences)

        Raises:
            ValueError: if ``other`` accounts for every sequence in this model.
                The C layer refuses that rather than leaving a model whose
                ``finalize`` can only fail.
        """
        if not isinstance(other, BeamModel):
            cfg = self.config
            other = BeamModel(
                list(other), abundances=abundances, order=cfg["order"],
                max_length=cfg["max_length"], weights=cfg["weights"],
                alpha=cfg["alpha"], alpha_length=cfg["alpha_length"],
                sparse=cfg["sparse"])
        elif abundances is not None:
            raise TypeError(
                "abundances applies only when `other` is a list of sequences; "
                "a BeamModel already carries its own weights")
        out = self.copy()
        out.subtract(other)
        return out

    def _check_operand(self, op: str, other: Any) -> None:
        if not isinstance(other, BeamModel):
            raise TypeError(f"{op}() expects a BeamModel, got "
                            f"{type(other).__name__}")
        if other is self:
            raise ValueError(
                f"{op}() with the model itself is refused: the C layer aliases "
                f"source and destination, so it would silently double the "
                f"counts (add) or zero them and leave the model unusable "
                f"(subtract)")
        if self._compacted or other._compacted:
            raise RuntimeError(
                f"{op}() is unavailable after compact(): the raw counts it "
                f"operates on have been released")
