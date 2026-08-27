"""BeamModel — Ensemble of Dual Graph Experts.

A probability model over CDR3 amino-acid sequences in which **every residue is
scored twice**: once by an N-anchored expert that indexes it by its distance
from the front of the sequence, and once by a C-anchored expert that indexes it
by its distance from the back. The two readings are combined geometrically and
normalised per length by an exact forward dynamic program.

The model consumes only the amino-acid sequence and its observed abundance. It
never sees a V or J gene call, a donor identity, or a sharing statistic.

Why two anchors: a single left-to-right chain learns the V-germline start well
and has forgotten it by the time it reaches the J motif, so it emits
terminations no J gene produces. Anchoring one expert at each terminus recovers
germline structure without ever being told which genes exist, because the V
germline *is* the front-anchored statistic and the J motif *is* the
back-anchored statistic.

For a fixed length the model is a layered DAG whose nodes are a position plus
the last ``order`` residues, so every analytical quantity — the partition
function, the sequence-probability spectrum, the support size, Hill numbers,
generation — is an exact dynamic program with no sampling.

BeamModel is a separate model, not a FlashBack variant. It keeps the geometric fact
the FlashBack peel exposed on amino acids (distance from each terminus is the
informative coordinate) and drops the token fusion. Nothing in the FlashBack
implementation is shared or changed.

See ``.private/edge/METHOD.md`` for the design and the measurements behind
every default.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from . import _cbeam as _c
from ._simulation_result import SimulationResult

#: Amino acids the model emits. Mirrors BEAM_RESIDUES in edge_model.h.
BEAM_RESIDUES = 20

#: Longest sequence any BeamModel model can represent. Mirrors
#: ``BEAM_MAX_LENGTH`` in edge_model.h.
BEAM_MAX_LENGTH = 27

#: Largest supported context order for the wide expert. Mirrors
#: ``BEAM_MAX_ORDER``.
BEAM_MAX_ORDER = 4

#: Largest moment order the jet DP carries. Mirrors ``BEAM_MAX_MOMENT``.
BEAM_MAX_MOMENT = 16

#: The twenty amino acids the model accepts, in table order.
BEAM_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _check_int(name: str, value: Any, lo: int, hi: int) -> int:
    """Range-check a configuration integer before it reaches the C layer.

    The C entry point takes these as ``uint8_t``, so an out-of-range value is
    silently truncated rather than rejected: ``max_length=283`` becomes 27 and
    ``order=260`` becomes 4, both of which then build a perfectly healthy
    model that is not the one that was asked for. Checking here is the only
    place that difference can still be seen.
    """
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {v}")
    return v


def _is_scalar_arg(x: Any) -> bool:
    """True when ``x`` is a single number rather than a container.

    One convention for the whole module: an argument that arrives as a scalar
    leaves as a ``float``, and anything with a shape leaves as an array of the
    same shape. A one-element list is a container and stays one.
    """
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def set_threads(n: int) -> None:
    """Cap the threads BeamModel uses, **process-wide**.

    A library that silently grabs every core is unusable inside a job
    scheduler or alongside another parallel stage, so this is a hard cap
    rather than a hint: every parallel region in the BeamModel implementation runs
    with at most ``n`` threads.

    ``n = 0`` restores the default, which is whatever the OpenMP runtime would
    choose on its own. Negative values are treated as 0. A build without
    OpenMP ignores the setting and always reports 1.

    Results do NOT depend on it: every reduction in the implementation is
    either order-independent or summed in a fixed index order, so a
    one-thread run and a sixteen-thread run agree bit for bit. That is what
    makes pinning threads for a runtime comparison a fair thing to do.

    The setting is global and not synchronised. Set it before starting work,
    not from one thread while another is inside a computation.
    """
    _c.set_threads(int(n))


def threads() -> int:
    """The cap in force, or 0 if none is. The CONFIGURED value."""
    return int(_c.threads())


def effective_threads() -> int:
    """Threads a parallel region would actually start right now.

    The cap if one is set, otherwise the OpenMP runtime's own maximum, and
    always 1 in a build without OpenMP.
    """
    return int(_c.effective_threads())


def set_memory_budget(nbytes: int) -> None:
    """Cap the scratch memory concurrent per-length spectra may hold together.

    Only :meth:`BeamModel.spectra_by_length` is large enough to care: one length's
    histogram grids run to 1.6 GB at order 4 with 512 bins, so running one
    length per core would ask for twenty-five gigabytes. It divides this
    budget by what one length needs and runs that many at a time.

    The default is 4 GB; ``0`` restores it. This caps **concurrency only** — a
    single length always runs whatever its size — so lowering it can never
    turn a working call into a failing one, only a slower one.

    It is a resource knob and nothing else. **Results do not depend on it**,
    which the C unit tests pin at a budget of one byte.
    """
    _c.set_memory_budget(int(nbytes))


def memory_budget() -> int:
    """The scratch budget in force, in bytes."""
    return int(_c.memory_budget())


def _column(data: Any, name: str) -> list:
    """One column of a DataFrame / Arrow table / dict of lists, as a list.

    Duck-typed on purpose: this library carries no DataFrame dependency, and
    an isinstance check would mean acquiring one just to recognise a type.
    """
    if hasattr(data, "column_names") and hasattr(data, "column"):
        if name not in data.column_names:
            raise KeyError(f"no column {name!r}; have {list(data.column_names)}")
        return data.column(name).to_pylist()          # pyarrow.Table
    if hasattr(data, "columns"):
        cols = list(data.columns)
        if name not in cols:
            raise KeyError(f"no column {name!r}; have {cols}")
        col = data[name]
        return col.tolist() if hasattr(col, "tolist") else list(col)
    if isinstance(data, dict):
        if name not in data:
            raise KeyError(f"no column {name!r}; have {sorted(data)}")
        col = data[name]
        return col.tolist() if hasattr(col, "tolist") else list(col)
    raise TypeError(
        f"expected a DataFrame, an Arrow table or a dict of lists, got "
        f"{type(data).__name__}")


def _iter_delimited(path, *, sep, sequence_column, abundance_column,
                    has_header, chunk_size):
    """Yield ``(sequences, abundances_or_None)`` chunks from a text file."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    seqs: list[str] = []
    ab: list[float] | None = None if abundance_column is None else []
    with open(os.fspath(path), encoding="utf-8") as fh:
        if has_header:
            fh.readline()
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split(sep) if sep is not None else line.split()
            seqs.append(parts[sequence_column])
            if ab is not None:
                ab.append(float(parts[abundance_column])
                          if len(parts) > abundance_column else 1.0)
            if len(seqs) >= chunk_size:
                yield seqs, ab
                seqs = []
                ab = None if abundance_column is None else []
    if seqs:
        yield seqs, ab


def _iter_parquet(path, sequence_column, abundance_column, batch_size):
    """Yield ``(sequences, abundances_or_None)`` chunks from a parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:                                  # pragma: no cover
        raise ImportError(
            "reading parquet needs pyarrow; `pip install pyarrow`, or use "
            "from_sequence_file / from_dataframe") from None
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    cols = [sequence_column]
    if abundance_column is not None:
        cols.append(abundance_column)
    pf = pq.ParquetFile(os.fspath(path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        seqs = batch.column(sequence_column).to_pylist()
        ab = (None if abundance_column is None
              else [float(x) for x in batch.column(abundance_column).to_pylist()])
        yield seqs, ab


def _config_args(*, order, depth_cap, max_length, coefficients, kappa,
                 kappa_wide, kappa_len, weights, strict=False,
                 sparse=None) -> tuple:
    """Validate a build configuration and flatten it for the C entry points.

    Shared by :class:`BeamModel` and :class:`Builder` so a streamed build and a
    one-shot build cannot be validated differently, which would make the
    "byte-identical" claim on :meth:`Builder.build` a coincidence.
    """
    if abs(sum(coefficients) - 1.0) > 1e-9:
        raise ValueError(
            f"coefficients must sum to 1, got {sum(coefficients)!r}. The "
            "sum is a cross-length calibration knob, not a free "
            "parameter; moving it off 1 degrades global ranking while "
            "leaving within-length ranking untouched.")
    order = _check_int("order", order, 2, BEAM_MAX_ORDER)
    # 2 is a floor, not an oversight: `depth_cap` must be at least 1 and
    # strictly below `max_length`, so a one-residue model has nowhere to
    # put its wide expert's depth resolution.
    max_length = _check_int("max_length", max_length, 2, BEAM_MAX_LENGTH)
    depth_cap = _check_int("depth_cap", depth_cap, 1, BEAM_MAX_LENGTH - 1)
    # A cap at or beyond the longest sequence resolves every depth exactly,
    # so the C layer clamps it; mirroring that here keeps `config` honest
    # about what was actually built.
    depth_cap = min(depth_cap, max_length - 1)
    coefficients = tuple(float(c) for c in coefficients)
    kappa = tuple(float(k) for k in kappa)
    kappa_wide = float(kappa_wide)
    kappa_len = float(kappa_len)
    weights = tuple(float(w) for w in weights)
    strict = bool(strict)

    zero_backoff = all(k == 0.0 for k in kappa) and (
        kappa_wide == 0.0 and kappa_len == 0.0)
    if sparse is None:
        sparse = strict and zero_backoff
    else:
        sparse = bool(sparse)
        if sparse and not (strict and zero_backoff):
            raise ValueError(
                "sparse=True requires strict=True and all five backoff "
                "strengths (kappa, kappa_wide, and kappa_len) to be zero")

    return (order, depth_cap, max_length, coefficients, kappa,
            kappa_wide, kappa_len, weights, strict, sparse)


def _smoothing_defaults(strict, kappa, kappa_wide, kappa_len):
    """Resolve the backoff strengths, whose default depends on `strict`.

    Open-world (the default) wants strong backoff: 20/20/20 and 200. Strict
    mode wants none, because backoff from a shorter context is exactly the
    thing that gives an unobserved transition mass and stops D0 from meaning
    anything. Passing kappa explicitly always wins, which is what makes the
    partial regime -- strict with backoff still on -- reachable for anyone
    who wants it.
    """
    if kappa is None:
        kappa = (0.0, 0.0, 0.0) if strict else (20.0, 20.0, 20.0)
    if kappa_wide is None:
        kappa_wide = 0.0 if strict else 20.0
    if kappa_len is None:
        kappa_len = 0.0 if strict else 200.0
    return kappa, kappa_wide, kappa_len


def _config_args_from_dict(cfg: dict[str, Any]) -> tuple:
    """The same tuple, rebuilt from what :attr:`BeamModel.config` reports."""
    return _config_args(order=cfg["order"], depth_cap=cfg["depth_cap"],
                        max_length=cfg["max_length"],
                        coefficients=cfg["coefficients"], kappa=cfg["kappa"],
                        kappa_wide=cfg["kappa_wide"],
                        kappa_len=cfg["kappa_len"], weights=cfg["weights"],
                        strict=cfg.get("strict", False),
                        sparse=cfg.get("sparse", False))


class BeamModel:
    """Bidirectional anchored DAG over CDR3 amino-acid sequences.

    Args:
        sequences: Training CDR3 strings. Sequences longer than ``max_length``
            or containing a character outside the twenty amino acids are
            skipped; see :attr:`n_skipped`.
        abundances: Per-sequence observed counts; defaults to 1 each.
        order: Context order of the wide expert, 2 to 4. Order 4 is the
            measured optimum and the default; order 2 or 3 build far smaller
            models (see :attr:`memory_bytes`).
        depth_cap: Depth at which the wide expert stops resolving position
            exactly and pools everything deeper into one bucket.
        coefficients: Log-linear mixing weights for the wide, mid and
            length-conditioned experts. **They must sum to 1.** The sum is not
            a free parameter: it is a pure cross-length calibration knob, and
            moving it off 1 leaves within-length ranking untouched while
            wrecking global ranking.
        max_length: Longest sequence the model represents.
        strict: Remove the order-0 add-one and probability floor. Its
            smoothing defaults are all zero, giving exact closed-world
            support unless a backoff strength is supplied explicitly.
        sparse: Store only occupied rows. ``None`` selects sparse storage when
            ``strict`` is true and all five backoff strengths are zero;
            ``False`` forces the equivalent dense representation. ``True``
            outside that exact closed-world regime is rejected.

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
        order: int = 4,
        depth_cap: int = 15,
        coefficients: tuple[float, float, float] = (0.60, 0.25, 0.15),
        kappa: tuple[float, float, float] | None = None,
        kappa_wide: float | None = None,
        kappa_len: float | None = None,
        weights: tuple[float, float] = (0.5, 0.5),
        max_length: int = 27,
        strict: bool = False,
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
        kappa, kappa_wide, kappa_len = _smoothing_defaults(
            strict, kappa, kappa_wide, kappa_len)
        cargs = _config_args(order=order, depth_cap=depth_cap,
                             max_length=max_length, coefficients=coefficients,
                             kappa=kappa, kappa_wide=kappa_wide,
                             kappa_len=kappa_len, weights=weights,
                             strict=strict, sparse=sparse)
        max_length = cargs[2]

        try:
            self._g = _c.build(seqs, abund, *cargs)
        except ValueError as exc:
            if "configuration" in str(exc):
                raise                       # already says what is wrong
            raise ValueError(self._build_failure_reason(seqs, abund,
                                                        max_length, exc)) from None
        self._compacted = False

    @staticmethod
    def _build_failure_reason(seqs, abund, max_length, exc) -> str:
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
            if any(c not in BEAM_ALPHABET for c in s):
                continue
            if abund is not None and not (abund[i] > 0 and np.isfinite(abund[i])):
                continue
            usable += 1
            break
        if usable == 0:
            return (
                f"no training sequence survived encoding: every one of "
                f"{len(seqs)} is empty, longer than max_length={max_length}, "
                f"contains a character outside {BEAM_ALPHABET!r}, or carries a "
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
        sequence_column: str = "cdr3_aa",
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
        sequence_column: str = "cdr3_aa",
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
    def n_observed(self) -> int:
        """Sequences accepted during training."""
        return int(_c.n_observed(self._g))

    @property
    def n_skipped(self) -> int:
        """Sequences rejected as too long or not amino acids."""
        return int(_c.n_skipped(self._g))

    @property
    def total_weight(self) -> float:
        """Total abundance observed during training."""
        return float(_c.total_weight(self._g))

    @property
    def n_states(self) -> int:
        """Nodes per layer of the DAG, ``21 ** max(order, 3)``.

        Not ``21 ** order``: the mid expert is always order 3, so the state
        can never hold fewer than three residues.
        """
        return int(_c.n_states(self._g))

    @property
    def n_cells(self) -> int:
        """Total table cells across both directions and all three experts."""
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
           for every other part of the library that runs an BeamModel parallel
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
        """Approximate resident bytes of the model's tables."""
        return int(_c.memory_bytes(self._g))

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

        The C layer takes these as ``uint32_t``: ``sample(-1)`` becomes a
        request for 4,294,967,295 draws, which does not raise, it simply never
        returns.
        """
        try:
            v = int(n)
        except (TypeError, ValueError):
            raise TypeError(f"{name} must be an integer, got {n!r}") from None
        if v < 0:
            raise ValueError(f"{name} must be non-negative, got {v}")
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
        column: str = "cdr3_aa",
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
            out: A preallocated array or memmap to fill, of the file's length.
            dtype: Element type when ``out`` is ``None``.
        """
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
                or contains a character outside the twenty amino acids. Use
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
        bad = sorted({c for c in sequence if c not in BEAM_ALPHABET})
        if bad:
            raise ValueError(
                f"sequence contains {bad!r}, outside the twenty amino acids "
                f"{BEAM_ALPHABET!r}")

    # ── Per-length quantities ───────────────────────────────────

    def log_partition(self, length: int) -> float:
        """``ln Z(L)``, the total weight of every length-``L`` path.

        Needed because a geometric product of two experts is not normalised on
        its own.
        """
        return float(_c.log_partition(self._g, self._check_length(length)))

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
           In the default open-world model this will be ``20 ** length``, and
           that is not a bug. Hierarchical backoff leaves every table cell
           strictly positive, so the number describes the smoother rather
           than the grammar. With ``strict=True`` and zero backoff, dead
           transitions stay dead and this is instead the exact support of the
           observed-transition DAG. Sparse and dense storage agree.

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
           at order 4 and 9,261 at order 2 or 3. A pooled spectrum over a
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

    def power_sum(self, q: float) -> float:
        """``sum_s P(s)**q`` on the linear scale.

        Overflows to ``inf`` for ``q`` well below 1, where the support size
        dominates; use :meth:`log_mellin` there.
        """
        return math.exp(self.log_mellin(q))

    def hill_numbers(self, orders: Iterable[float]) -> np.ndarray:
        """Hill numbers at several orders at once."""
        return np.array([self.hill_number(float(a)) for a in orders])

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
            "density": float(edges / (nodes * BEAM_RESIDUES)) if nodes else 0.0,
        }

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
        return PseqAnalysis(self)

    def scale_score(self, sequence, calibration: Calibration):
        """Length-normalised anomaly score; higher means more surprising.

        ``(-ln P(s) - median_L) / IQR_L``, with the median and IQR taken from
        ``calibration`` — which is the model's own EXACT per-length surprisal
        law, reconstructed by dynamic programming rather than measured off a
        few hundred thousand simulated draws. There is no ``n_sim`` and no
        minimum-count rule, so there is nothing to tune and no sampling noise
        in the reference.

        A length the calibration does not cover falls back to the pooled null,
        weighted by the length prior.

        Args:
            sequence: A single string, or a sequence of strings.
            calibration: From :meth:`calibrate`.

        Returns:
            ``float`` for a single string, ``np.ndarray`` otherwise.
        """
        if not isinstance(calibration, Calibration):
            raise TypeError(
                f"calibration must be an Calibration from "
                f"BeamModel.calibrate(), got {type(calibration).__name__}")
        single = isinstance(sequence, str)
        seqs = [sequence] if single else list(sequence)
        u = -self.log_pseq_batch(seqs)
        lengths = np.fromiter((len(x) for x in seqs), dtype=np.int64,
                              count=len(seqs))
        out = np.array([calibration.scale(int(L), float(v))
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

        Both models must share ``order``, ``max_length`` and ``depth_cap``;
        the smoothing parameters need not match, because counts do not depend
        on them and this model's own smoothing is what gets reapplied.
        """
        self._check_operand("add", other)
        _c.add(self._g, other._g)

    def subtract(self, other: BeamModel) -> None:
        """Subtract another model's counts from this one, in place.

        Counts are additive, which is what makes holding a cohort out of a
        foundation model a subtraction rather than a recount of the corpus.
        Both models must share ``order``, ``max_length`` and ``depth_cap``.
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
        categorical distribution — over the twenty residues for the three
        expert tables, over lengths for the length histogram — and this adds
        ``kappa * p_prior[row]`` to each of this model's rows, where
        ``p_prior`` is that row of ``prior`` NORMALISED. A row then reads::

            p_post = (c_own + kappa * p_prior) / (n_own + kappa)

        before BeamModel's own hierarchical backoff, which is the convention
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

        Both models must share ``order``, ``max_length`` and ``depth_cap``,
        and neither may be compacted. The model is re-finalized afterwards.

        Example:
            >>> donor = BeamModel(donor_sequences, order=2, max_length=10)
            >>> donor.posterior(foundation, kappa=200.0)
        """
        self._check_operand("posterior", prior)
        if not (float(kappa) >= 0.0):
            raise ValueError(f"kappa must be non-negative, got {kappa!r}")
        _c.posterior(self._g, prior._g, float(kappa))

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
                depth_cap=cfg["depth_cap"], max_length=cfg["max_length"],
                coefficients=cfg["coefficients"], kappa=cfg["kappa"],
                kappa_wide=cfg["kappa_wide"], kappa_len=cfg["kappa_len"],
                weights=cfg["weights"], strict=cfg["strict"],
                sparse=cfg["sparse"])
        elif abundances is not None:
            raise TypeError(
                "abundances applies only when `other` is a list of sequences; "
                "an BeamModel already carries its own weights")
        out = self.copy()
        out.subtract(other)
        return out

    def _check_operand(self, op: str, other: Any) -> None:
        if not isinstance(other, BeamModel):
            raise TypeError(f"{op}() expects an BeamModel, got "
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
        order: int = 4,
        depth_cap: int = 15,
        coefficients: tuple[float, float, float] = (0.60, 0.25, 0.15),
        kappa: tuple[float, float, float] | None = None,
        kappa_wide: float | None = None,
        kappa_len: float | None = None,
        weights: tuple[float, float] = (0.5, 0.5),
        max_length: int = 27,
        strict: bool = False,
        sparse: bool | None = None,
    ) -> None:
        kappa, kappa_wide, kappa_len = _smoothing_defaults(
            strict, kappa, kappa_wide, kappa_len)
        self._cargs = _config_args(
            order=order, depth_cap=depth_cap, max_length=max_length,
            coefficients=coefficients, kappa=kappa, kappa_wide=kappa_wide,
            kappa_len=kappa_len, weights=weights, strict=strict,
            sparse=sparse)
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
        ab = None if abundances is None else [float(a) for a in abundances]
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
        """Sequences rejected so far as too long or not amino acids."""
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
        dst = _c.new(*self._cargs)
        _c.add(dst, self._live())
        return BeamModel._from_capsule(dst)


class Histogram:
    """One reconstructed surprisal spectrum on a deterministic grid.

    Carries the attribute names :class:`FlashBackGraph`'s ``PseqHistogram``
    carries, so spectrum code written against that reads across unchanged.

    ``weights`` are raw masses: without a length restriction they sum to 1 for
    ``measure='generated'`` and to the support size for ``'counting'``.
    """

    __slots__ = ("surprisal", "weights", "measure", "q", "grid_spacing",
                 "max_rounding_error", "true_max_surprisal", "length")

    def __init__(self, *, surprisal, weights, measure, q, grid_spacing,
                 max_rounding_error, true_max_surprisal, length=None):
        self.surprisal = np.asarray(surprisal, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.measure = measure
        self.q = float(q)
        self.grid_spacing = float(grid_spacing)
        self.max_rounding_error = float(max_rounding_error)
        self.true_max_surprisal = float(true_max_surprisal)
        self.length = length

    def __repr__(self) -> str:
        at = "" if self.length is None else f", length={self.length}"
        return (f"Histogram(measure={self.measure!r}, "
                f"bins={self.weights.size}{at}, "
                f"total_mass={self.total_mass:.6g})")

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.weights, dtype=np.longdouble))

    @property
    def normalized_weights(self) -> np.ndarray:
        total = self.total_mass
        return self.weights / total if total > 0 else np.zeros_like(self.weights)

    @property
    def mean(self) -> float:
        """Mean surprisal on this grid.

        Exact to machine precision: linear transport conserves the first
        moment. The VARIANCE read off the same grid is not — it is inflated by
        up to ``(L + 1) * spacing**2 / 6`` — so take that from
        :meth:`BeamModel.surprisal_moments` when it matters.
        """
        return float(np.dot(self.normalized_weights, self.surprisal))

    @property
    def variance(self) -> float:
        w = self.normalized_weights
        d = self.surprisal - self.mean
        return float(np.dot(w, d * d))

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def cdf(self, x):
        """Normalised CDF in surprisal space."""
        values = np.atleast_1d(np.asarray(x, dtype=np.float64))
        total = self.total_mass
        if total <= 0:
            out = np.zeros_like(values)
        else:
            cum = np.cumsum(self.weights, dtype=np.float64) / total
            out = np.interp(values, self.surprisal, cum, left=0.0, right=1.0)
        return float(out[0]) if _is_scalar_arg(x) else out

    def pdf(self, x):
        """Piecewise-linear normalised density of the reconstruction."""
        values = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self.grid_spacing <= 0 or self.total_mass <= 0:
            out = np.zeros_like(values)
        else:
            density = self.normalized_weights / self.grid_spacing
            out = np.interp(values, self.surprisal, density,
                            left=0.0, right=0.0)
        return float(out[0]) if _is_scalar_arg(x) else out

    def quantile(self, probability):
        """Approximate surprisal quantile of the reconstructed measure."""
        p = np.atleast_1d(np.asarray(probability, dtype=np.float64))
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must be between 0 and 1")
        cum = np.cumsum(self.normalized_weights)
        out = np.interp(p, cum, self.surprisal)
        return float(out[0]) if _is_scalar_arg(probability) else out


class Atoms:
    """A discrete ``(probability, multiplicity)`` description of the model.

    One entry per occupied grid bin: how probable a sequence in that bin is,
    and how many sequences are there. Every rarefaction-style quantity —
    expected richness, novelty, coverage, the Poisson-binomial occupancy a
    sharing prediction needs — is a sum over these pairs.

    **The probability is ``generated / counting``, not ``exp(-surprisal)``.**
    Both spectra come out of ONE transport
    (:meth:`BeamModel.spectrum_pair`), so they are the same weights carrying
    different mass and their ratio is the mass-weighted mean probability of
    the sequences in the bin. That the two share splitting fractions is what
    makes the ratio meaningful, and taking them from one sweep makes it
    structural rather than a coincidence of two calls doing the same
    arithmetic. Reading ``exp(-surprisal)`` off the bin's left edge instead biases
    every atom by up to ``exp(spacing)`` and shows up immediately as a total
    mass that is not 1. With this definition
    ``sum(multiplicities * probabilities)`` is the generated mass exactly,
    which is the identity the tests check.
    """

    __slots__ = ("probabilities", "multiplicities", "surprisal",
                 "grid_spacing", "length")

    def __init__(self, *, probabilities, multiplicities, surprisal,
                 grid_spacing, length=None):
        self.probabilities = np.asarray(probabilities, dtype=np.longdouble)
        self.multiplicities = np.asarray(multiplicities, dtype=np.longdouble)
        self.surprisal = np.asarray(surprisal, dtype=np.float64)
        self.grid_spacing = float(grid_spacing)
        self.length = length

    def __repr__(self) -> str:
        return (f"Atoms(n={self.probabilities.size}, "
                f"support={self.support_size:.6g}, mass={self.mass:.9f})")

    def __len__(self) -> int:
        return int(self.probabilities.size)

    @property
    def support_size(self) -> float:
        """Total multiplicity: the counting measure's integral."""
        return float(np.sum(self.multiplicities, dtype=np.longdouble))

    @property
    def mass(self) -> float:
        """``sum multiplicity * probability``. 1 for an unrestricted spectrum."""
        return float(np.sum(self.multiplicities * self.probabilities,
                            dtype=np.longdouble))

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(probabilities, multiplicities)``, the pair every R(n)-style
        calculation and ``PublicnessModel.expected_counts`` consumes."""
        return self.probabilities, self.multiplicities

    def _survival(self, n: float) -> np.ndarray:
        """``(1 - p)**n``, computed so it survives ``p ~ 1e-80``.

        ``(1 - p)**n`` in float64 rounds ``1 - p`` to 1 and collapses to 1.0;
        ``exp(n * log1p(-p))`` in long double stays representable.
        """
        return np.exp(np.longdouble(n) * np.log1p(-self.probabilities))

    def expected_richness(self, n: float) -> float:
        """``R(n) = sum_s [1 - (1 - P(s))**n]``, distinct sequences after
        ``n`` independent draws.

        ``R(1)`` is exactly the generated mass, 1 for an unrestricted
        spectrum, and ``R(n)`` rises monotonically to the support size.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return float(np.sum(self.multiplicities * (1.0 - self._survival(n)),
                            dtype=np.longdouble))

    def novelty(self, n: float) -> float:
        """``U(n) = sum_s P(s) (1 - P(s))**(n-1)``, the chance draw ``n`` is
        one never drawn before.

        Exactly ``R(n) - R(n-1)``, which is the identity the tests check.
        """
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        return float(np.sum(
            self.multiplicities * self.probabilities * self._survival(n - 1.0),
            dtype=np.longdouble))

    def coverage(self, n: float) -> float:
        """Probability mass carried by the sequences seen in ``n`` draws.

        Good-Turing's target, computed exactly rather than estimated from the
        singletons.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        total = self.mass
        if total <= 0:
            return 0.0
        seen = np.sum(self.multiplicities * self.probabilities
                      * (1.0 - self._survival(n)), dtype=np.longdouble)
        return float(seen / np.longdouble(total))

    def expected_frequency(self, n: int, r: int) -> float:
        """Expected number of distinct sequences drawn exactly ``r`` times in
        ``n`` draws. ``r=1`` and ``r=2`` are Chao's ``f1`` and ``f2``."""
        n, r = int(n), int(r)
        if n < 0 or r < 0 or r > n:
            raise ValueError("require n >= 0 and 0 <= r <= n")
        p = self.probabilities
        keep = (p > 0) & (p < 1)
        log_coef = (math.lgamma(n + 1) - math.lgamma(r + 1)
                    - math.lgamma(n - r + 1))
        terms = np.exp(np.longdouble(log_coef)
                       + np.longdouble(r) * np.log(p[keep])
                       + np.longdouble(n - r) * np.log1p(-p[keep]))
        return float(np.sum(self.multiplicities[keep] * terms,
                            dtype=np.longdouble))

    def discovery_curve(self, draw_counts: Iterable[float]) -> dict[str, np.ndarray]:
        """``R(n)``, ``U(n)`` and coverage over many sampling depths at once.

        One pass of numpy per depth over the atoms, which is what makes a
        400-point curve seconds rather than one spectrum rebuild per point.
        """
        ns = np.asarray(list(draw_counts), dtype=np.float64)
        if np.any(ns < 1):
            raise ValueError("every draw count must be at least 1")
        return {
            "n": ns,
            "expected_richness": np.array([self.expected_richness(n) for n in ns]),
            "novelty": np.array([self.novelty(n) for n in ns]),
            "coverage": np.array([self.coverage(n) for n in ns]),
        }


class PseqAnalysis:
    """Spectrum and diversity analytics, under the names the graph models use.

    Returned by :meth:`BeamModel.pseq_analysis`. Everything here is exact dynamic
    programming over the trellis or arithmetic on a reconstructed spectrum;
    nothing is sampled.

    Spectra are cached per ``(bins, measure, length)``, which is the whole
    reason this object exists rather than a set of free functions: a
    reconstruction is the expensive step and a figure typically wants the same
    one several times.
    """

    __slots__ = ("model", "_hist", "_atoms")

    def __init__(self, model: BeamModel):
        self.model = model
        self._hist: dict[tuple, Histogram] = {}
        self._atoms: dict[tuple, Atoms] = {}

    def __repr__(self) -> str:
        return f"PseqAnalysis({self.model!r})"

    # ── grid ────────────────────────────────────────────────────

    @property
    def true_max_surprisal(self) -> float:
        """Exact largest surprisal any producible sequence attains."""
        return self.model.surprisal_range()[1]

    @property
    def true_min_surprisal(self) -> float:
        """Exact smallest surprisal any producible sequence attains."""
        return self.model.surprisal_range()[0]

    def grid_spacing(self, bins: int) -> float:
        """Bin width the model would choose for this many bins.

        Read out and passed back in explicitly by every spectrum this object
        builds, so that histograms taken at different lengths or measures share
        one grid by construction rather than by both happening to auto-size the
        same way.
        """
        return self.model._grid_spacing(bins)

    # ── spectra ─────────────────────────────────────────────────

    def histogram(self, bins: int = 2048, *, measure: str = "generated",
                  length: int | None = None) -> Histogram:
        """Reconstruct one spectrum on the deterministic grid.

        Args:
            bins: Grid resolution.
            measure: ``'generated'`` weights each sequence by ``P(s)`` and
                totals 1; ``'counting'`` weights every supported sequence 1
                and totals the support size.
            length: Restrict to sequences of exactly this length.
        """
        key = (int(bins), measure, None if length is None else int(length))
        if key not in self._hist:
            m = self.model
            spacing = m._grid_spacing(bins)
            sp = m.spectrum(measure=measure, length=length, bins=bins,
                            spacing=spacing)
            self._hist[key] = Histogram(
                surprisal=sp["surprisal"], weights=sp["weights"],
                measure=measure, q=0.0 if measure == "counting" else 1.0,
                grid_spacing=spacing,
                # One linear split per layer, plus one at the sink.
                max_rounding_error=(int(m.config["max_length"]) + 1) * spacing,
                true_max_surprisal=self.true_max_surprisal,
                length=key[2])
        return self._hist[key]

    def _pair(self, bins: int, length: int | None) -> dict[str, Histogram]:
        """Both measures at one length, from a single fused transport.

        Populates the same cache :meth:`histogram` reads, so a later
        single-measure call at the same key costs nothing.
        """
        bins = int(bins)
        key_c = (bins, "counting", length)
        key_g = (bins, "generated", length)
        if key_c not in self._hist or key_g not in self._hist:
            m = self.model
            spacing = m._grid_spacing(bins)
            sp = m.spectrum_pair(length=length, bins=bins, spacing=spacing)
            err = (int(m.config["max_length"]) + 1) * spacing
            top = self.true_max_surprisal
            for key, measure, q in ((key_c, "counting", 0.0),
                                    (key_g, "generated", 1.0)):
                self._hist[key] = Histogram(
                    surprisal=sp["surprisal"], weights=sp[measure],
                    measure=measure, q=q, grid_spacing=spacing,
                    max_rounding_error=err, true_max_surprisal=top,
                    length=length)
        return {"counting": self._hist[key_c], "generated": self._hist[key_g]}

    def histogram_pair(self, bins: int = 2048) -> dict[str, Histogram]:
        """Both global spectra from ONE transport, on one grid.

        About 1.5x faster than the two single-measure calls, and bit-identical
        to them. The shared grid is the load-bearing part: see
        :meth:`BeamModel.spectrum_pair` and :meth:`atoms` for why the two measures
        have to be split by the same fractions and not merely by the same
        arithmetic.
        """
        return self._pair(bins, None)

    def histograms_by_length(self, bins: int = 2048, *,
                             measure: str = "generated",
                             max_length: int | None = None,
                             ) -> dict[int, Histogram]:
        """Every length-conditioned spectrum, all on the same global grid.

        One call into :meth:`BeamModel.spectra_by_length`, whose note says plainly
        what that buys: 1.26x at order 3 and parity at order 4, because BeamModel's
        per-length trellises are separate dynamic programs with no shared work
        to fuse. What it does buy at every order is one grid, sized from one
        :meth:`BeamModel.accumulator_max` rather than one per length, so the rows
        are directly comparable and summable.

        Rows are bit-identical to the single-length :meth:`histogram` calls.
        """
        hi = int(self.model.config["max_length"])
        if max_length is None:
            max_length = hi
        else:
            max_length = int(max_length)
            if not 0 <= max_length <= hi:
                raise ValueError(
                    f"max_length must be between 0 and {hi}, got {max_length}")
        lengths = [L for L in self.model.observed_lengths if L <= max_length]
        if not lengths:
            return {}
        m = self.model
        spacing = m._grid_spacing(bins)
        sp = m.spectra_by_length(measure=measure, max_length=max(lengths),
                                 bins=bins, spacing=spacing)
        err = (int(m.config["max_length"]) + 1) * spacing
        top = self.true_max_surprisal
        q = 0.0 if measure == "counting" else 1.0
        out = {}
        for L in lengths:
            key = (int(bins), measure, L)
            if key not in self._hist:
                self._hist[key] = Histogram(
                    surprisal=sp["surprisal"], weights=sp["weights"][L].copy(),
                    measure=measure, q=q, grid_spacing=spacing,
                    max_rounding_error=err, true_max_surprisal=top, length=L)
            out[L] = self._hist[key]
        return out

    # ── atoms and discovery ─────────────────────────────────────

    def atoms(self, bins: int = 2048, *, length: int | None = None) -> Atoms:
        """``(probability, multiplicity)`` for every occupied grid bin.

        See :class:`Atoms` for why the probability is the ratio of the two
        spectra rather than ``exp(-surprisal)``.
        """
        key = (int(bins), None if length is None else int(length))
        if key not in self._atoms:
            pair = self._pair(bins, key[1])
            c, g = pair["counting"], pair["generated"]
            cw = c.weights.astype(np.longdouble)
            gw = g.weights.astype(np.longdouble)
            keep = (cw > 0) & (gw > 0)
            self._atoms[key] = Atoms(
                probabilities=gw[keep] / cw[keep],
                multiplicities=cw[keep],
                surprisal=c.surprisal[keep],
                grid_spacing=c.grid_spacing,
                length=key[1])
        return self._atoms[key]

    def expected_richness(self, n: float, *, bins: int = 2048) -> float:
        """``R(n)``: expected distinct sequences after ``n`` draws."""
        return self.atoms(bins).expected_richness(n)

    def novelty(self, n: float, *, bins: int = 2048) -> float:
        """``U(n)``: the chance draw ``n`` is one never drawn before."""
        return self.atoms(bins).novelty(n)

    def discovery_curve(self, draw_counts: Iterable[float], *,
                        bins: int = 2048) -> dict[str, np.ndarray]:
        """``R(n)``, ``U(n)`` and coverage over many depths, one spectrum."""
        return self.atoms(bins).discovery_curve(draw_counts)

    # ── transforms ──────────────────────────────────────────────

    def log_mellin(self, q: float) -> float:
        """``ln sum_s P(s)**q``, exactly."""
        return self.model.log_mellin(q)

    def mellin(self, q: float) -> float:
        """``sum_s P(s)**q``. Overflows to ``inf`` well below ``q=1``."""
        lm = self.model.log_mellin(q)
        return math.inf if lm > math.log(np.finfo(np.float64).max) else math.exp(lm)

    def cgf(self, t: float) -> float:
        """``log M(1 - t)``, the generated-surprisal CGF."""
        return self.model.log_mellin(1.0 - float(t))

    def derivatives(self, q: float, order: int = 4) -> np.ndarray:
        """``out[r] = sum_s P(s)**q * ln(P(s))**r``, through ``order``.

        Assembled from the exact tilted moments: the ``q``-tilted law
        normalises by ``M(q)``, so multiplying its raw moments back by ``M(q)``
        and alternating sign recovers the unnormalised jet.

        ``M(q)`` is astronomically large for ``q`` below about 1, so element
        ``r`` overflows to ``inf`` there. Use :meth:`log_mellin` when that
        happens; it is the same quantity in a representable form.
        """
        order = _check_int("order", order, 0, BEAM_MAX_MOMENT)
        raw = self.model.surprisal_moments(q=float(q), order=order)
        log_mass = self.model.log_mellin(float(q))
        with np.errstate(over="ignore"):
            mass = math.exp(log_mass) if log_mass < 709.0 else math.inf
            out = np.array([mass * ((-1.0) ** r) * raw[r]
                            for r in range(order + 1)], dtype=np.float64)
        return out

    def moments(self, order: int = 4) -> dict[str, float]:
        """Generated-surprisal mass, mean, variance, skewness and kurtosis."""
        order = _check_int("order", order, 2, BEAM_MAX_MOMENT)
        st = self.model.surprisal_stats(q=1.0)
        keep = ("mean", "variance", "std", "skewness", "excess_kurtosis")
        out = {k: st[k] for k in keep}
        out["kurtosis"] = out.pop("excess_kurtosis")
        out["mass"] = math.exp(self.model.log_mellin(1.0))
        return out

    # ── per-length ──────────────────────────────────────────────

    def length_marginals(self) -> dict[int, dict[str, float]]:
        """``{length: {counting, generated, mean_surprisal, variance}}``.

        ``counting`` sums to the support size and ``generated`` to 1, which is
        what makes this a decomposition rather than a summary. Keyed by length
        and spelled as the graph models spell it.
        """
        lm = self.model.length_marginals()
        out = {}
        for i, L in enumerate(lm["length"]):
            if lm["counting"][i] <= 0 and lm["generated"][i] <= 0:
                continue
            out[int(L)] = {
                "counting": float(lm["counting"][i]),
                "generated": float(lm["generated"][i]),
                "mean_surprisal": float(lm["mean_surprisal"][i]),
                "variance": float(lm["variance"][i]),
            }
        return out

    def length_derivatives(self, q: float, order: int = 0,
                           ) -> dict[int, np.ndarray]:
        """Mellin derivatives grouped by sequence length, at any ``q``.

        For length ``L``, element ``r`` is
        ``sum_{len(s)=L} P(s)**q * ln(P(s))**r`` — the unnormalised jet the
        graph models return, so pipeline code reads across unchanged.

        Exact at every ``(q, order)``: :meth:`BeamModel.length_moments` supplies
        each length's share of the ``q``-tilted law and its conditional
        surprisal moments, and the jet is that share times ``M(q)`` with the
        sign of ``(-1)**r``.

        ``M(q)`` is astronomically large for ``q`` below about 1, so the jet
        overflows to ``inf`` there for the same reason :meth:`derivatives`
        does. :meth:`BeamModel.length_moments` is the same content in a form that
        never multiplies by ``M(q)`` and so cannot overflow; use it when the
        shape rather than the scale is what is wanted.
        """
        q = float(q)
        order = _check_int("order", order, 0, BEAM_MAX_MOMENT)
        lm = self.model.length_moments(q=q, order=order)
        log_mass = self.model.log_mellin(q)
        with np.errstate(over="ignore", invalid="ignore"):
            mass = math.exp(log_mass) if log_mass < 709.0 else math.inf
            signs = np.array([(-1.0) ** r for r in range(order + 1)])
            scale = lm["share"] * mass
            jets = lm["moments"] * signs[None, :] * scale[:, None]
        return {int(L): jets[L].copy()
                for L in range(len(lm["share"])) if lm["share"][L] > 0}

    def length_profile(self) -> dict[int, dict[str, float]]:
        """Per-length mass, mean surprisal and variance, as a summary dict."""
        return {L: {"mass": v["generated"], "mean": v["mean_surprisal"],
                    "variance": v["variance"],
                    "std": math.sqrt(max(v["variance"], 0.0))}
                for L, v in self.length_marginals().items()}

    # ── deliberately absent ─────────────────────────────────────

    def diversity_under_edge_thresholds(self, thresholds: Any) -> dict:
        """Not applicable: BeamModel has no prunable edges.

        In the open-world model, hierarchical backoff leaves every table cell
        strictly positive, so an edge sweep would invent a model the smoother
        does not describe. A strict model already retains its exact
        zero-probability boundary rather than an arbitrary threshold.
        :meth:`BeamModel.spectrum` and :meth:`atoms` answer where the mass and
        sequences sit under either regime.
        """
        raise NotImplementedError(
            "BeamModel has no edge pruning: open-world support is fixed by its "
            "smoother, while strict support already has an exact zero "
            "boundary. See the docstring for what to use instead.")


class Calibration:
    """An exact length-conditioned null for the surprisal.

    Built by :meth:`BeamModel.calibrate`. Holds, for each length the model can
    produce, the survival function of ``-ln P`` among sequences of exactly that
    length. Reconstructed by dynamic programming rather than simulated, so
    there is no sampling noise, no minimum-count rule, and nothing to tune.
    """

    __slots__ = ("surprisal", "spacing", "survival", "prior", "_pooled",
                 "_stats")

    def __init__(self, *, surprisal, spacing, survival, prior=None):
        self.surprisal = np.asarray(surprisal, dtype=np.float64)
        self.spacing = float(spacing)
        self.survival = {int(k): np.asarray(v, dtype=np.float64)
                         for k, v in survival.items()}
        # Uniform if no prior is supplied, which is the only honest default
        # when the length mixture is not known.
        self.prior = ({int(k): float(v) for k, v in prior.items()}
                      if prior is not None
                      else dict.fromkeys(self.survival, 1.0))
        self._pooled: np.ndarray | None = None
        self._stats: dict[int | None, tuple[float, float]] = {}

    def __repr__(self) -> str:
        return (f"Calibration(lengths={len(self.survival)}, "
                f"resolution_floor={self.resolution_floor:.2e})")

    @property
    def lengths(self) -> tuple[int, ...]:
        """Lengths this calibration covers."""
        return tuple(sorted(self.survival))

    @property
    def resolution_floor(self) -> float:
        """Smallest tail probability the grid can resolve.

        A p-value at or below this is at the limit of what the reconstruction
        can distinguish from zero, and should be reported as such rather than
        quoted precisely.
        """
        return min(float(v[v > 0].min()) if np.any(v > 0) else 1.0
                   for v in self.survival.values())

    @property
    def pooled_survival(self) -> np.ndarray:
        """The length mixture's survival function, weighted by ``pi(L)``.

        The fallback null for a length the calibration does not cover.
        """
        if self._pooled is None:
            total = sum(self.prior.values())
            acc = np.zeros_like(self.surprisal)
            for L, sf in self.survival.items():
                acc += (self.prior.get(L, 0.0) / total) * sf
            self._pooled = acc
        return self._pooled

    def quantile(self, length: int | None, probability):
        """Surprisal ``u`` with ``P(U <= u) = probability`` at this length.

        ``length=None`` uses the pooled mixture. Linearly interpolated on the
        reconstruction grid, so it is exact to one bin width.
        """
        p = np.atleast_1d(np.asarray(probability, dtype=np.float64))
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must be between 0 and 1")
        sf = (self.pooled_survival if length is None
              else self.survival.get(int(length)))
        if sf is None:
            sf = self.pooled_survival
        # MID-DISTRIBUTION cdf. `1 - sf[i]` is P(U < u_i), strictly below the
        # atom, and inverting that puts every quantile half an atom too high —
        # a systematic bias of half a bin width on the median and so on every
        # SCALE score. Splitting the atom's own mass across the step removes it.
        below = np.clip(1.0 - sf, 0.0, 1.0)
        atom = np.empty_like(sf)
        atom[:-1] = sf[:-1] - sf[1:]
        atom[-1] = sf[-1]
        cdf = np.clip(below + 0.5 * atom, 0.0, 1.0)
        # np.interp needs an increasing x; a survival function is monotone, so
        # the CDF is already sorted up to floating-point flat spots.
        out = np.interp(p, cdf, self.surprisal)
        return float(out[0]) if _is_scalar_arg(probability) else out

    def _median_iqr(self, length: int | None) -> tuple[float, float]:
        """Median and interquartile range of the surprisal at one length."""
        key = None if length is None or int(length) not in self.survival \
            else int(length)
        if key not in self._stats:
            q25, q50, q75 = self.quantile(key, [0.25, 0.5, 0.75])
            iqr = float(q75 - q25)
            # A length whose whole mass falls in one bin has a zero IQR; the
            # grid spacing is the smallest difference that grid can express,
            # and dividing by it is better than dividing by zero.
            self._stats[key] = (float(q50), iqr if iqr > 0 else self.spacing)
        return self._stats[key]

    def median(self, length: int | None = None) -> float:
        """Median surprisal at this length; ``None`` pools over lengths."""
        return self._median_iqr(length)[0]

    def iqr(self, length: int | None = None) -> float:
        """Interquartile range of the surprisal at this length."""
        return self._median_iqr(length)[1]

    def scale(self, length: int, surprisal):
        """``(surprisal - median_L) / IQR_L``, the SCALE anomaly score.

        Length-normalised because ``-ln P`` grows with length, so a raw
        surprisal is not comparable across lengths. An uncovered length falls
        back to the pooled mixture.
        """
        m, sd = self._median_iqr(length)
        u = np.atleast_1d(np.asarray(surprisal, dtype=np.float64))
        out = (u - m) / sd
        return float(out[0]) if _is_scalar_arg(surprisal) else out

    def pvalue(self, length: int, surprisal) -> float | np.ndarray:
        """``P(U >= surprisal)`` for a sequence of this length.

        Returns 1.0 for a length the calibration does not cover, which is the
        conservative answer: an uncovered length is not evidence of anomaly.
        """
        sf = self.survival.get(int(length))
        u = np.atleast_1d(np.asarray(surprisal, dtype=np.float64))
        if sf is None:
            out = np.ones_like(u)
        else:
            idx = np.clip(np.searchsorted(self.surprisal, u, side="left"),
                          0, len(sf) - 1)
            # The survival function is a normalised cumulative sum, so its
            # first entry can land a few ulps above 1. A p-value may not.
            out = np.clip(sf[idx], 0.0, 1.0)
        if _is_scalar_arg(surprisal):
            return float(out[0])
        return out


__all__ = ["BeamModel", "Atoms", "Builder", "Calibration",
           "Histogram", "PseqAnalysis", "effective_threads",
           "memory_budget", "threads", "set_memory_budget",
           "set_threads"]
