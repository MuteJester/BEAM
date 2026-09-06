"""The query object: one model, and every exact question this engine answers.

``BeamQuery`` wraps a fitted model and takes :class:`~BEAM.query.Condition`
objects. Everything it returns is exact -- computed by dynamic programming over
the model's own trellis, with nothing sampled, binned or fitted. A question
with no exact answer is refused rather than estimated.

    >>> q = BeamQuery(model)
    >>> c = Condition(14, model.alphabet).at(4, "G").contains("CASS")
    >>> q.probability(c)                     # P(C)
    >>> q.marginals(c)                       # P(x_i = a | C), shape (14, R)
    >>> q.entropy(c)                         # H(X | C), in nats
    >>> q.mutual_information(c)              # I(X_i ; X_j | C), shape (14, 14)

Cost is not uniform and the object does not hide it. A condition built only
from masks costs one sweep; one carrying an automaton costs a product graph;
and :meth:`mutual_information` costs one sweep per (position, residue) pair.
:meth:`cost` reports all of that before anything runs.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .. import _cbeam as _c
from ..model import alphabets
from .conditions import Condition
from .errors import ImpossibleCondition, QueryTooLarge, UnsupportedQuery

__all__ = ["BeamQuery"]


def _translate(exc: BaseException) -> BaseException:
    """Narrow a binding's built-in exception to this package's own.

    The C layer already distinguishes these cases; the bindings widen them onto
    built-ins because that is the right thing for the rest of the library.
    Narrowing here loses nothing, because each new class keeps the built-in as
    a base.
    """
    msg = str(exc)
    if isinstance(exc, MemoryError):
        return QueryTooLarge(msg)
    if isinstance(exc, RuntimeError):
        return ImpossibleCondition(msg)
    if isinstance(exc, ValueError):
        return UnsupportedQuery(msg)
    return exc


class BeamQuery:
    """Exact probabilistic questions about one fitted model.

    Args:
        model: a finalized :class:`~BEAM.BeamModel`.
    """

    __slots__ = ("_model", "_syms", "_index")

    def __init__(self, model) -> None:
        self._model = model
        self._syms = alphabets.symbols(model.alphabet)
        self._index = {a: i for i, a in enumerate(self._syms)}

    def __repr__(self) -> str:
        return (f"BeamQuery(alphabet={self._model.alphabet!r}, "
                f"residues={len(self._syms)})")

    @property
    def model(self):
        return self._model

    def condition(self, length: int) -> Condition:
        """A fresh, unnarrowed condition over this model's alphabet."""
        return Condition(length, self._model.alphabet)

    # ── plumbing ────────────────────────────────────────────────

    def _check(self, cond: Condition) -> tuple:
        if not isinstance(cond, Condition):
            raise UnsupportedQuery(
                f"expected a Condition, got {type(cond).__name__}")
        if cond.alphabet != self._model.alphabet:
            raise UnsupportedQuery(
                f"this condition is over the {cond.alphabet!r} alphabet and "
                f"the model is over {self._model.alphabet!r}")
        return cond._payload()

    def _call(self, fn, cond: Condition, *extra):
        length, masks, spec = self._check(cond)
        try:
            return fn(self._model._g, length, masks, spec, *extra)
        except (ValueError, RuntimeError, MemoryError) as exc:
            raise _translate(exc) from None

    # ── probability ─────────────────────────────────────────────

    def log_probability(self, cond: Condition) -> float:
        """``ln P(C | length)``, the exact mass the condition selects.

        Returns ``-inf`` for a condition the model cannot produce. That is an
        answer, not a failure: an event that cannot happen has probability
        zero. Conditioning on it is what has no answer, and the methods below
        raise :class:`ImpossibleCondition` instead.
        """
        return float(self._call(_c.query_probability, cond))

    def probability(self, cond: Condition) -> float:
        """``P(C | length)``. Exactly zero where the model cannot produce it."""
        return math.exp(self.log_probability(cond))

    # ── distributions ───────────────────────────────────────────

    def marginals(self, cond: Condition) -> np.ndarray:
        """``P(x_i = a | length, C)`` as a ``(length, n_residues)`` array.

        Every row sums to one. A position the condition pins comes back
        one-hot, which is the correct conditional rather than a placeholder.
        """
        raw, n_res, _ = self._call(_c.query_marginals, cond)
        return np.frombuffer(raw, dtype=np.float64).reshape(-1, n_res).copy()

    def log_partition(self, cond: Condition) -> float:
        """``ln Z`` of the conditioned trellis."""
        _, _, logz = self._call(_c.query_marginals, cond)
        return float(logz)

    def pair_joint(self, cond: Condition, i: int, j: int) -> np.ndarray:
        """``P(x_i = a, x_j = b | C)`` as an ``(n_residues, n_residues)`` array.

        With ``i == j`` the result is diagonal, which is the correct joint of a
        position with itself.
        """
        raw, n_res = self._call(_c.query_pair_joint, cond, int(i), int(j))
        return np.frombuffer(raw, dtype=np.float64).reshape(n_res, n_res).copy()

    # ── information ─────────────────────────────────────────────

    def entropy(self, cond: Condition) -> float:
        """``H(X | length, C)`` in nats: the entropy of the whole sequence."""
        return float(self._call(_c.query_entropy, cond)[0])

    def surprisal_variance(self, cond: Condition) -> float:
        """``Var[-ln P]`` under the condition: the spread of the surprisal."""
        return float(self._call(_c.query_entropy, cond)[1])

    def position_entropy(self, cond: Condition) -> np.ndarray:
        """``H(X_i | C)`` for each position, in nats.

        Not the sequence entropy split up: these sum to MORE than
        :meth:`entropy` wherever positions share information, and the gap is
        exactly what :meth:`mutual_information` measures.
        """
        raw = self._call(_c.query_position_entropy, cond)
        return np.frombuffer(raw, dtype=np.float64).copy()

    def mutual_information(self, cond: Condition) -> np.ndarray:
        """``I(X_i ; X_j | C)`` for every pair, in nats.

        Symmetric, non-negative, with :meth:`position_entropy` on the diagonal
        -- a position tells you exactly itself. Costs one sweep per
        ``(position, residue)`` pair carrying mass, so it is much the most
        expensive query here; :meth:`cost` reports the total first.
        """
        raw = self._call(_c.query_information_matrix, cond)
        n = int(round(math.sqrt(raw.__len__() / 8)))
        return np.frombuffer(raw, dtype=np.float64).reshape(n, n).copy()

    def conditional_entropy(self, cond: Condition, i: int, j: int) -> float:
        """``H(X_i | X_j, C)``: what is left to learn about `i` knowing `j`."""
        mi = self.mutual_information(cond)
        return float(mi[i, i] - mi[i, j])

    # ── expectations ────────────────────────────────────────────

    def expect(self, table, cond: Condition) -> tuple[float, float]:
        """``E[f]`` and ``Var[f]`` for ``f(s) = sum_i table[i, x_i]``.

        Only functionals that decompose over positions are exactly computable,
        and this is all of them: indicators, residue counts and any weighted
        positional score are a table with different entries. An arbitrary
        function of a whole sequence is not here and will not be, since
        computing its expectation means enumerating the support or sampling it.

        Args:
            table: ``(length, n_residues)`` of per-position, per-residue values.
            cond: the condition to take the expectation under.

        Returns:
            ``(mean, variance)``.
        """
        arr = np.ascontiguousarray(table, dtype=np.float64)
        length, _, _ = self._check(cond)
        if arr.shape != (length, len(self._syms)):
            raise UnsupportedQuery(
                f"the functional table is {arr.shape} and this condition needs "
                f"({length}, {len(self._syms)})")
        mean, var, _ = self._call(_c.query_expect, cond, arr.ravel().tolist())
        return float(mean), float(var)

    def expect_count(self, residues: Iterable[str],
                     cond: Condition) -> tuple[float, float]:
        """``E[#positions holding one of `residues`]`` and its variance.

        Equal to the summed marginals down those columns, however correlated
        the positions are -- expectation is linear even when nothing else is.
        """
        length, _, _ = self._check(cond)
        table = np.zeros((length, len(self._syms)))
        for r in residues:
            if r not in self._index:
                raise UnsupportedQuery(
                    f"{r!r} is not one of the {self._model.alphabet!r} "
                    f"residues {self._syms!r}")
            table[:, self._index[r]] = 1.0
        return self.expect(table, cond)

    # ── cross-length: insertions and deletions ──────────────────

    def insertions(self, sequence: str) -> list[dict]:
        """Every sequence one insertion away from `sequence`, scored.

        Each entry carries ``sequence``, ``site`` (the gap inserted at),
        ``residue``, ``log_odds`` against the source, and ``posterior`` over
        the neighbourhood. The odds compare joint probabilities across two
        lengths, so the length prior does not cancel; the posterior asks where
        and which GIVEN that one residue was inserted, and there it does.

        **This does not infer an edit.** The model has no alignment and no
        indel process. These are ratios between what it assigns to two strings
        that differ by one residue, describing where a repertoire tolerates
        insertions. For "given this observed sequence, where was it edited",
        see :class:`~BEAM.prism.PrismScorer`.
        """
        return self._neighbours(sequence, deletions=False)

    def deletions(self, sequence: str) -> list[dict]:
        """Every sequence one deletion away from `sequence`, scored.

        The mirror of :meth:`insertions`, with ``site`` naming the position
        removed and ``residue`` left as ``None``.
        """
        return self._neighbours(sequence, deletions=True)

    def _neighbours(self, sequence: str, *, deletions: bool) -> list[dict]:
        try:
            rows, log_source = _c.query_neighbours(
                self._model._g, sequence, deletions)
        except (ValueError, RuntimeError, MemoryError) as exc:
            raise _translate(exc) from None
        out = []
        for seq, site, res, odds, post in rows:
            out.append({
                "sequence": seq,
                "site": int(site),
                "residue": None if deletions else self._syms[res],
                "log_odds": float(odds),
                "posterior": float(post),
                "log_source": float(log_source),
            })
        out.sort(key=lambda r: -r["posterior"])
        return out

    # ── cost ────────────────────────────────────────────────────

    def cost(self, cond: Condition, *, whole_matrix: bool = False) -> dict:
        """What a query will cost, without running it.

        Args:
            whole_matrix: cost :meth:`mutual_information` rather than a single
                sweep. That is one sweep per ``(position, residue)`` pair, so
                the difference is two orders of magnitude and worth knowing
                before rather than after.

        Returns:
            ``states``, ``edges`` and ``bytes``. ``states`` is an upper bound
            on what the sweep visits, not the live count. ``bytes`` is the
            figure checked against the memory budget.
        """
        s, e, b = self._call(_c.query_cost, cond, bool(whole_matrix))
        return {"states": int(s), "edges": int(e), "bytes": int(b)}
