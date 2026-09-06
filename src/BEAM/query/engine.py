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
        """``H(X | length, C)`` in nats: of the whole sequence, not one position.

        Conditional on the length and EXCLUDING the length prior. Three things
        are called "the entropy" around here and they are different numbers:

        - this one, ``H(X | L, C)``;
        - ``length_marginals()["mean_surprisal"][L]``, which is
          ``E[-ln P | L]`` on the JOINT and so exceeds this by ``-ln pi(L)`` --
          1.54 nats, or 33%, at one measured length;
        - the model's total ``H(X) = H(L) + sum_L pi(L) H(X | L)``, which this
          engine does not produce because its conditions are per-length.

        The variance is unaffected by the shift, so
        :meth:`surprisal_variance` matches the jet's exactly.
        """
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
        """Mutual information between every pair of positions, in nats,
        computed WITHIN the event `cond`.

        Not ``I(X_i ; X_j | C)`` in the information-theoretic sense, which
        averages over the values of ``C``. This is the mutual information of
        the distribution restricted to ``C`` -- ``I`` under ``P(. | C)``. The
        two differ: on one test model an entry read 0.079 here against 0.157
        for the averaged quantity, low by a factor of two, which is the
        direction that makes you miss a coupling. The averaged form is
        ``p * MI(C) + (1 - p) * MI(~C)``, and ``~cond`` is expressible, so it
        is two more calls if that is what you want.

        Symmetric, non-negative, with :meth:`position_entropy` on the diagonal
        -- a position tells you exactly itself. Costs one sweep per
        ``(position, residue)`` pair carrying mass, so it is much the most
        expensive query here; :meth:`cost` reports the total first.
        """
        raw = self._call(_c.query_information_matrix, cond)
        n = int(round(math.sqrt(raw.__len__() / 8)))
        return np.frombuffer(raw, dtype=np.float64).reshape(n, n).copy()

    def conditional_entropy(self, cond: Condition, i: int, j: int) -> float:
        """``H(X_i | X_j, C)``: what is left to learn about `i` knowing `j`.

        From the pair joint, which is one sweep per residue of `j`. Reading it
        off :meth:`mutual_information` instead would be `length * n_residues`
        sweeps for two cells -- 281 rather than 20 on a length-14 amino-acid
        model.
        """
        joint = self.pair_joint(cond, i, j)
        pi = joint.sum(axis=1)
        h_i = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
        nz = joint > 0
        h_ij = -np.sum(joint[nz] * np.log(joint[nz]))
        pj = joint.sum(axis=0)
        h_j = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
        return float(h_ij - h_j)

    def total_correlation(self, cond: Condition) -> float:
        """``sum_i H(X_i) - H(X)``: everything the positions share, at any order.

        The mutual-information matrix sees pairs; this sees the whole
        dependency structure, including interactions no pair exhibits. Zero
        exactly when the positions are independent. Costs nothing new -- both
        halves are already computed.
        """
        return float(self.position_entropy(cond).sum() - self.entropy(cond))

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

        Each entry carries ``sequence``, ``site``, ``residue``,
        ``multiplicity``, ``log_odds`` against the source, ``posterior``, and
        ``length_producible``.

        ``posterior`` is ``P(s\' | s\' in this set)`` -- a distribution over
        STRINGS, not over edit operations. Inserting ``S`` into the ``SS`` of
        ``CASSSQYF`` gives one string reachable four ways; it appears once,
        with ``multiplicity=4``. Weighted per EDIT instead -- multiply by
        ``multiplicity`` and renormalise -- that string is the likeliest
        neighbour, where here it ranks ninth. Per string asks what the model
        makes of a candidate set; per edit asks what happens when a longer
        sequence is drawn and a uniform position removed, which is the story
        :class:`~BEAM.prism.PrismScorer` uses and why it sums repairs with
        multiplicity rather than deduplicating.

        ``site`` is the LEFTMOST gap producing that string -- a canonical
        label, not a claim about where an edit occurred.

        ``length_producible`` is false when the model generates no sequence of
        the neighbours' length at all. Every posterior is then zero for that
        one reason, which is a fact about the length prior rather than about
        any of these strings.

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
            rows, log_source, producible = _c.query_neighbours(
                self._model._g, sequence, deletions)
        except (ValueError, RuntimeError, MemoryError) as exc:
            raise _translate(exc) from None
        out = []
        for seq, site, res, mult, odds, post in rows:
            out.append({
                "sequence": seq,
                "site": int(site),
                "residue": None if deletions else self._syms[res],
                "multiplicity": int(mult),
                "log_odds": float(odds),
                "posterior": float(post),
                "log_source": float(log_source),
                "length_producible": bool(producible),
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
