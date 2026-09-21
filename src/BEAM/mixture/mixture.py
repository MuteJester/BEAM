"""The probability mixture of several fitted models.

See :mod:`BEAM.mixture` for what the object is and what is exact about it.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .. import _cbeam as _c
from ..errors import (BeamTypeError, BeamValueError, BudgetExceeded,
                      ModelMismatch, NoLivePaths)
from .errors import (IncompatibleMembers, MixtureError, MixtureTooLarge,
                     UnsupportedOrder)
from .results import (BetaDiversity, EntropyBounds, EntropyEstimate,
                      SupportSize)

__all__ = ["Mixture"]

_WEIGHT_TOL = 1e-9


def _logsumexp(values: np.ndarray, axis: int | None = None) -> Any:
    """``log sum exp``, with ``-inf`` surviving as an answer.

    numpy's own reduction is not usable here: shifting by the maximum gives
    ``-inf - -inf`` when every term is ``-inf``, and NaN is exactly the wrong
    answer for a sequence no member spells. The whole closed world rests on
    ``-inf`` meaning "probability zero" all the way to the caller, so the
    all-dead case is short-circuited rather than computed.
    """
    values = np.asarray(values, dtype=np.float64)
    mx = np.max(values, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        shifted = np.where(np.isfinite(mx), values - mx, 0.0)
        acc = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    out = np.where(np.isfinite(mx), mx + acc, mx)
    if axis is None:
        return float(out.reshape(()))
    return np.squeeze(out, axis=axis)


def _translate(exc: BaseException) -> BaseException:
    """Narrow a binding's exception to this package's own.

    By MEANING, not by which built-in the failure happens to share -- the same
    rule :func:`BEAM.query.engine._translate` settled on after guessing from
    built-in types reported a non-finalized model as an impossible condition.
    Anything already specific enough passes through untouched.
    """
    if isinstance(exc, MixtureError):
        return exc
    msg = str(exc)
    if isinstance(exc, ModelMismatch):
        return IncompatibleMembers(msg)
    if isinstance(exc, BudgetExceeded):
        return MixtureTooLarge(msg)
    return exc


class Mixture:
    """``pbar(s) = sum_i w_i P_i(s)``: pick a member by ``w``, then a sequence.

    Args:
        models: Two or more finalized :class:`~BEAM.BeamModel` instances that
            agree on alphabet, ``order`` and ``max_length``.
            One model may appear more than once; unlike :meth:`BeamModel.add`,
            nothing here mutates a member, so aliasing is harmless and a
            mixture of a model with itself is a useful degenerate case.
            Compacted models are accepted: the sweeps read the finalized
            tables, not the raw counts.
        weights: Mixing weights, one per model, **summing to 1**. Defaults to
            uniform.

    Weights are REFUSED, not normalised, when they do not sum to 1 (within
    ``1e-9``) or when any is negative. Normalising would give a plausible
    answer to the two mistakes that actually happen -- raw sequence counts
    pasted in as weights, and a member dropped from one list but not the other
    -- and a plausible wrong number is worse here than a refusal. Divide by the
    sum yourself and the intent is on the page.

    Example:
        >>> mix = Mixture([donor_a, donor_b, donor_c])
        >>> mix.hill_number(2.0)          # exact
        41233.7...
        >>> mix.entropy_bounds()          # exact bracket; q=1 has no closed form
        EntropyBounds(lower=..., upper=..., ...)
    """

    __slots__ = ("_models", "_caps", "_w", "_logw", "_ip_cache", "_hill_cache")

    def __init__(self, models: Sequence[Any],
                 weights: Iterable[float] | None = None) -> None:
        members = tuple(models)
        if len(members) == 0:
            raise BeamValueError(
                "a mixture needs at least one member, got none")
        for i, m in enumerate(members):
            if not hasattr(m, "_g") or not hasattr(m, "log_pseq"):
                raise BeamTypeError(
                    f"mixture member {i} is a {type(m).__name__}, not a "
                    f"BeamModel")
        self._models = members
        self._caps = [m._g for m in members]

        n = len(members)
        if weights is None:
            w = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            w = np.asarray(list(weights), dtype=np.float64)
            if w.shape != (n,):
                raise BeamValueError(
                    f"weights must have one entry per member: {n} members and "
                    f"{w.size} weights")
            if not np.all(np.isfinite(w)):
                raise BeamValueError("weights must all be finite")
            if np.any(w < 0.0):
                raise BeamValueError(
                    f"weights must be non-negative, got {w.min()!r}")
            total = float(w.sum())
            if abs(total - 1.0) > _WEIGHT_TOL:
                raise BeamValueError(
                    f"weights must sum to 1, got {total!r}; they are a "
                    f"probability distribution over members, and this is "
                    f"refused rather than normalised because raw counts and a "
                    f"dropped member both land here and both would otherwise "
                    f"produce a plausible answer -- divide by the sum yourself")
        self._w = w
        with np.errstate(divide="ignore"):
            self._logw = np.log(w)

        try:
            _c.mixture_check_shape(self._caps)
        except (ValueError, RuntimeError, MemoryError) as exc:
            raise _translate(exc) from None

        self._ip_cache: dict[tuple[int, ...], float] = {}
        self._hill_cache: dict[float, np.ndarray] = {}

    # ── Identity ────────────────────────────────────────────────

    def __repr__(self) -> str:
        cfg = self.config
        return (f"Mixture({self.n_members} members, {cfg['alphabet']} "
                f"order={cfg['order']} max_length={cfg['max_length']})")

    def __len__(self) -> int:
        return len(self._models)

    @property
    def n_members(self) -> int:
        """Number of members, counting a repeated model once per appearance."""
        return len(self._models)

    @property
    def members(self) -> tuple:
        """The member models, in the order given."""
        return self._models

    @property
    def weights(self) -> np.ndarray:
        """The mixing weights. A copy, so the mixture cannot be re-weighted."""
        return self._w.copy()

    @property
    def config(self) -> dict[str, Any]:
        """The configuration all members share."""
        return self._models[0].config

    @property
    def max_length(self) -> int:
        """The longest sequence any member represents."""
        return int(self.config["max_length"])

    # ── Length structure ────────────────────────────────────────

    @property
    def observed_lengths(self) -> tuple[int, ...]:
        """Lengths the mixture can produce, ascending.

        The UNION of the members' lengths: the mixture produces a length as
        soon as one member does.
        """
        seen: set[int] = set()
        for m in self._models:
            seen.update(m.observed_lengths)
        return tuple(sorted(seen))

    @property
    def length_distribution(self) -> dict[int, float]:
        """``{length: pibar(L)}`` where ``pibar(L) = sum_i w_i pi_i(L)``."""
        out: dict[int, float] = {}
        for w, m in zip(self._w, self._models):
            if w == 0.0:
                continue
            for L, p in m.length_distribution.items():
                out[L] = out.get(L, 0.0) + float(w) * p
        return {L: out[L] for L in sorted(out)}

    def member_weights(self, length: int) -> np.ndarray:
        """The effective member weights WITHIN one length.

        ``w_i pi_i(L) / pibar(L)``, not ``w_i``. Members carry different length
        priors, so conditioning on a length re-weights them: a donor who never
        makes a length-22 CDR3 has weight zero there whatever its share of the
        mixture overall.

        This is the single easiest thing to get wrong in a mixture, because
        using ``w_i`` unchanged still produces a normalised distribution and
        plausible numbers. It is exposed rather than kept private so that it
        can be tested directly against a brute-force sum.

        Raises:
            NoLivePaths: if no member produces ``length``.
        """
        L = int(length)
        num = np.array([float(w) * math.exp(m.log_length_prior(L))
                        if w > 0.0 else 0.0
                        for w, m in zip(self._w, self._models)],
                       dtype=np.float64)
        total = float(num.sum())
        if not (total > 0.0):
            raise NoLivePaths(
                f"no member of this mixture produces length {L}, so there are "
                f"no within-length weights to report")
        return num / total

    # ── Scoring ─────────────────────────────────────────────────

    def log_pseq(self, sequence: str) -> float:
        """``ln pbar(s)``, exactly.

        Exact for any single sequence even though the SUM over sequences is
        not: each member's ``ln P_i(s)`` is exact, and log-sum-exp with the
        weights is arithmetic.

        Returns ``-inf``, never a floored finite value, for a sequence no
        member can produce. The per-length structure needs no special handling
        here -- each ``P_i(s)`` already carries its own ``pi_i(L)``, so the
        within-length re-weighting falls out of the sum.
        """
        vals = np.array([m.log_pseq(sequence) for m in self._models],
                        dtype=np.float64)
        return float(_logsumexp(self._logw + vals))

    def log_pseq_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Vectorised :meth:`log_pseq`."""
        seqs = list(sequences)
        if not seqs:
            return np.empty(0, dtype=np.float64)
        rows = np.stack([np.asarray(m.log_pseq_batch(seqs), dtype=np.float64)
                         for m in self._models])
        return _logsumexp(rows + self._logw[:, None], axis=0)

    def pseq(self, sequence, *, log: bool = False):
        """Sequence probability, for one sequence or many. See :meth:`log_pseq`."""
        if isinstance(sequence, str):
            v = self.log_pseq(sequence)
            return v if log else math.exp(v)
        v = self.log_pseq_batch(sequence)
        return v if log else np.exp(v)

    def sample(self, n: int, *, seed: int | None = None) -> list[str]:
        """Draw ``n`` sequences from the mixture, exactly in proportion to it.

        A member is drawn from ``w`` and then a sequence from that member,
        which IS the mixture by definition. The member counts are drawn
        multinomially and each member sampled once for its whole share rather
        than one draw at a time -- the two are the same distribution, and the
        second costs one C call per draw.

        The result is shuffled, so the order carries no trace of which member
        produced what. Grouping would be a quiet trap for anything that reads a
        prefix.
        """
        n = int(n)
        if n < 0:
            raise BeamValueError(f"n must be non-negative, got {n!r}")
        if n == 0:
            return []
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, self._w)
        out: list[str] = []
        for c, m in zip(counts, self._models):
            if c:
                out.extend(m.sample(int(c),
                                    seed=int(rng.integers(1, 2 ** 31 - 1))))
        rng.shuffle(out)
        return out

    # ── Inner products: the exact engine ────────────────────────

    def log_inner_product(self, *members: int) -> float:
        """``ln sum_s prod_f P_f(s)`` over the members named, exactly.

        With two indices this is ``ln <P_i, P_j>``, whose exponential is the
        probability that an independent draw from each lands on the same
        sequence. One forward sweep over the shared trellis, whose edge weight
        is the sum of the members' edge weights.

        ``-inf`` means the members share no sequence at all. That is an answer.
        """
        idx = tuple(self._check_member(i) for i in members)
        if not idx:
            raise BeamValueError("name at least one member")
        key = tuple(sorted(idx))
        hit = self._ip_cache.get(key)
        if hit is not None:
            return hit
        try:
            total, _by = _c.mixture_inner_product([self._caps[i] for i in key])
        except (ValueError, RuntimeError, MemoryError) as exc:
            raise _translate(exc) from None
        self._ip_cache[key] = float(total)
        return float(total)

    def inner_product(self, *members: int) -> float:
        """:meth:`log_inner_product` on the linear scale. Exactly 0.0 when disjoint."""
        return math.exp(self.log_inner_product(*members))

    def log_gram_matrix(self) -> np.ndarray:
        """``ln <P_i, P_j>`` for every pair, computed ONE PAIR PER THREAD.

        The log scale is the useful one at cohort size: members that share no
        sequence give ``-inf`` here and exactly ``0.0`` in
        :meth:`gram_matrix`, and a matrix of underflowed zeros cannot be told
        apart from one of genuine disjointness.

        Symmetric to the last bit: one triangle is computed and mirrored,
        because the sweep sums the members' edge weights in the order it is
        handed them.

        This is where the parallelism belongs. One inner product spreads over
        the model's ~25 lengths and stops scaling at about four threads --
        measured, 110 ms of core time per pair at one thread and 399 ms at
        sixteen -- whereas pairs are independent, equal-sized and there are
        ``M(M+1)/2`` of them. Asking for the whole matrix at once is therefore
        much faster than asking for its entries: at 512 members it is minutes
        rather than hours.

        Every entry is cached, so :meth:`log_inner_product` and
        :meth:`hill_number` are free afterwards.
        """
        n = self.n_members
        try:
            raw = _c.mixture_gram(self._caps)
        except (ValueError, RuntimeError, MemoryError) as exc:
            raise _translate(exc) from None
        g = np.frombuffer(raw, dtype=np.float64).reshape(n, n).copy()
        for i in range(n):
            for j in range(i, n):
                self._ip_cache[(i, j)] = float(g[i, j])
        return g

    def gram_matrix(self) -> np.ndarray:
        """``G[i, j] = <P_i, P_j>``, the members' pairwise overlap.

        The linear scale of :meth:`log_gram_matrix`, and symmetric for the same
        reason. A pair that shares no sequence reads exactly ``0.0``; use the
        log form if you need to tell that apart from underflow.
        """
        return np.exp(self.log_gram_matrix())

    def _check_member(self, i: int) -> int:
        k = int(i)
        if not 0 <= k < self.n_members:
            raise BeamValueError(
                f"member index {i!r} is outside 0..{self.n_members - 1}")
        return k

    # ── Hill numbers ────────────────────────────────────────────

    @staticmethod
    def _exact_order(q: float) -> int:
        """The integer behind an exactly computable order, or raise saying why.

        ``q = 0`` and integer ``q >= 2`` only. Everything else -- ``q = 1``
        above all, but also ``q = 1.5`` and ``q = -1`` -- needs the log of a
        sum over members, which does not factor over edges.
        """
        f = float(q)
        if f == 0.0:
            return 0
        if f.is_integer() and f >= 2.0:
            return int(f)
        raise UnsupportedOrder(
            f"the Hill number of a mixture has no closed form at q={q!r}; "
            f"only q=0 and integer q>=2 are exactly computable, because "
            f"log(sum_i w_i P_i(s)) does not factor over the trellis. "
            f"At q=1 use entropy_bounds() for an exact bracket or "
            f"entropy_estimate(n) for a sampled value with a standard error; "
            f"both say which they are. This is refused rather than estimated "
            f"so that no unqualified number leaves here.")

    def log_power_sum(self, q: float) -> float:
        """``ln sum_s pbar(s)^q``, exactly. Integer ``q >= 2`` only.

        The power expands into a sum over ORDERED q-tuples of members, which is
        collected here into unordered multisets carrying their multinomial
        multiplicity: ``M(M+1)/2`` inner products at ``q = 2`` rather than
        ``M^2``, and ``C(M+q-1, q)`` in general.

        The multiplicities are exact integers rather than ``lgamma`` of them:
        the whole point of this path is that it is exact, and a q-tuple count
        is small enough to be exact for any q this can afford to run.
        """
        qi = self._exact_order(q)
        if qi == 0:
            raise UnsupportedOrder(
                "log_power_sum is the q>=2 engine; q=0 is the support union, "
                "see support_size()")
        n = self.n_members
        if qi == 2:
            return self._log_power_sum_2()
        fact = [math.factorial(k) for k in range(qi + 1)]
        terms: list[float] = []
        for combo in itertools.combinations_with_replacement(range(n), qi):
            counts = Counter(combo)
            lw = sum(c * self._logw[j] for j, c in counts.items())
            if not math.isfinite(lw):
                continue                       # a member with zero weight
            mult = fact[qi]
            for c in counts.values():
                mult //= fact[c]
            lip = self.log_inner_product(*combo)
            if lip == -math.inf:
                continue                       # these members share nothing
            terms.append(math.log(mult) + lw + lip)
        if not terms:
            return -math.inf
        return float(_logsumexp(np.array(terms, dtype=np.float64)))

    def _log_power_sum_2(self) -> float:
        """``ln sum_s pbar^2``, from the whole Gram matrix in one call.

        The q-tuple loop above is correct at every order but walks the pairs
        one at a time from Python, which at cohort size is the difference
        between minutes and hours: it serialises the only axis that scales.
        q = 2 is the order that is actually used, so it gets the driver that
        hands one pair to each thread, and the arithmetic afterwards is a
        single vectorised reduction.

        The off-diagonal is doubled rather than summed twice, which is the
        whole content of the q = 2 expansion and the easiest thing to get wrong
        when assembling this by hand.
        """
        g = self.log_gram_matrix()
        lw = self._logw
        terms = g + lw[:, None] + lw[None, :]
        iu = np.triu_indices(self.n_members, k=1)
        off = terms[iu] + math.log(2.0)
        allt = np.concatenate([np.diag(terms), off])
        allt = allt[np.isfinite(allt)]
        if allt.size == 0:
            return -math.inf
        return float(_logsumexp(allt))

    def hill_number(self, q: float) -> float:
        """Effective number of sequences at order ``q``. Exact, or it raises.

        ``q = 0`` is the size of the union of the members' supports and
        ``q = 2`` the inverse collision probability of the mixture. Integer
        ``q >= 2`` generalises.

        Raises:
            UnsupportedOrder: at ``q = 1`` and every non-integer order. There
                is no exact answer and none is invented; see
                :class:`~BEAM.mixture.errors.UnsupportedOrder`.
            MixtureTooLarge: at ``q = 0`` when the exact union does not fit the
                memory budget. ``support_size(exact=False)`` gives bounds.
        """
        qi = self._exact_order(q)
        if qi == 0:
            size = self.support_size(exact=True)
            return float(size.value)
        return math.exp(self.log_power_sum(qi) / (1.0 - qi))

    def hill_numbers(self, orders: Iterable[float]) -> np.ndarray:
        """:meth:`hill_number` at several orders at once.

        Every order is validated BEFORE any is computed, so a list containing
        ``1.0`` raises instead of returning a shorter array. A partial answer
        to a list request is worse than no answer: the caller sees fewer
        numbers and no reason, and the entry that vanished is exactly the one
        that needed explaining.
        """
        want = [float(a) for a in orders]
        for a in want:
            self._exact_order(a)
        return np.array([self.hill_number(a) for a in want], dtype=np.float64)

    # ── q = 0: the support union ────────────────────────────────

    def support_size(self, *, exact: bool = True,
                     cap: int = 0) -> SupportSize:
        """``D0``: how many distinct sequences the mixture can produce.

        The UNION of the members' supports. Not the support of a model trained
        on the pooled counts, which is strictly larger because it can splice
        edges learned from different members; see the module docstring.

        Args:
            exact: Run the exact ``(state, member-set)`` sweep. ``False``
                returns only the free bounds and does no work.
            cap: Largest number of live ``(state, member-set)`` pairs a layer
                may hold; ``0`` means as many as the memory budget allows.

        Raises:
            MixtureTooLarge: when ``exact`` is set and the sweep outgrows its
                ceiling. The number of surviving member-sets, not the member
                count, is what binds; the message says where it stopped.
        """
        per = [m.path_count() for m in self._models]
        lower = float(max(per))
        upper = float(sum(per))
        if not exact:
            return SupportSize(value=None, lower=lower, upper=upper,
                               exact=False)
        try:
            total, _by, peak = _c.mixture_support_union(self._caps, int(cap))
        except (ValueError, RuntimeError, MemoryError) as exc:
            raise _translate(exc) from None
        return SupportSize(value=float(total), lower=lower, upper=upper,
                           exact=True, peak_states=int(peak))

    # ── q = 1: bounds and an estimate, never a bare number ──────

    def entropy_bounds(self) -> EntropyBounds:
        """Exact bounds on ``H(pbar)``, in nats. Free.

        ``sum_i w_i H_i <= H(pbar) <= sum_i w_i H_i + H(w)``.

        Both sides come from the members' own exact entropies and the weights,
        and both are attained -- identical members sit on the lower bound,
        disjoint members on the upper -- so the width is the range the answer
        really occupies rather than slack in the method.
        """
        h = np.array([m.entropy() for m in self._models], dtype=np.float64)
        lower = float(np.dot(self._w, h))
        nz = self._w[self._w > 0.0]
        hw = float(-np.sum(nz * np.log(nz)))
        return EntropyBounds(lower=lower, upper=lower + hw, weight_entropy=hw)

    def entropy_estimate(self, n: int = 10000,
                         seed: int | None = None) -> EntropyEstimate:
        """A sampled ``H(pbar)`` with its standard error. NOT exact.

        Draws ``n`` sequences from the mixture and averages ``-ln pbar(s)``.
        Unbiased, because ``ln pbar(s)`` is exact for each individual ``s``
        even though the sum over all of them is not; the only error is Monte
        Carlo and it is reported.

        The returned object carries the exact bracket alongside the estimate.
        An estimate outside it by more than a couple of standard errors is a
        bug, not a small sample.
        """
        n = int(n)
        if n < 2:
            raise BeamValueError(
                f"n must be at least 2 for a standard error, got {n!r}")
        seqs = self.sample(n, seed=seed)
        lp = self.log_pseq_batch(seqs)
        if not np.all(np.isfinite(lp)):
            raise MixtureError(
                "a sequence drawn from this mixture scored -inf under it, "
                "which cannot happen: every draw comes from a member that "
                "gives it positive probability. Report this.")
        s = -lp
        return EntropyEstimate(
            value=float(s.mean()),
            stderr=float(s.std(ddof=1) / math.sqrt(n)),
            n_samples=n, seed=seed, bounds=self.entropy_bounds())

    # ── Beta diversity ──────────────────────────────────────────

    def alpha_diversity(self, q: float) -> float:
        """The members' own ``D_q``, averaged as the POWER MEAN OF ORDER 1-q.

        Arithmetic at ``q = 0``, geometric at ``q = 1``, harmonic at ``q = 2``.
        Exact at every order, because it asks each member for its own Hill
        number and those are exact.

        The naive arithmetic mean at every order is the trap this method exists
        to remove: it is correct only at ``q = 0``, and elsewhere it inflates
        alpha and so silently under-reports beta.
        """
        f = float(q)
        d = np.array([m.hill_number(f) for m in self._models],
                     dtype=np.float64)
        w = self._w
        live = w > 0.0
        if abs(f - 1.0) < 1e-12:
            return float(math.exp(float(np.dot(w[live], np.log(d[live])))))
        p = 1.0 - f
        return float(np.dot(w[live], d[live] ** p) ** (1.0 / p))

    def beta_diversity(self, q: float, *, samples: int = 0,
                       seed: int | None = None) -> BetaDiversity:
        """``D_beta = D_gamma / D_alpha`` at order ``q``.

        How many effectively distinct repertoires the members amount to.
        ``1 <= D_beta <= M``: 1 when they are the same distribution, ``M`` when
        they share nothing. That bound belongs to THE MIXTURE and says nothing
        about a model trained on the pooled counts, whose diversity can exceed
        the sum of the members' because it spells sequences none of them do.

        Exact at ``q = 0`` and integer ``q >= 2``. At ``q = 1`` the gamma has
        no closed form, so the result carries the exact bracket
        ``1 <= beta <= exp(H(w))`` with ``beta`` left as ``None``; pass
        ``samples`` to add a sampled value and its standard error alongside.

        Args:
            samples: Draws for the ``q = 1`` estimate. Refused at an exactly
                computable order, where it would be silently ignored.
        """
        f = float(q)
        alpha = self.alpha_diversity(f)
        if abs(f - 1.0) < 1e-12:
            b = self.entropy_bounds()
            upper = math.exp(b.weight_entropy)
            est = err = None
            if samples:
                e = self.entropy_estimate(samples, seed=seed)
                est = math.exp(e.value - b.lower)
                err = est * e.stderr          # delta method on exp
            return BetaDiversity(order=f, gamma=None, alpha=alpha, beta=None,
                                 lower=1.0, upper=upper, exact=False,
                                 estimate=est, stderr=err)
        if samples:
            raise BeamValueError(
                f"samples is only meaningful at q=1; q={q!r} is exactly "
                f"computable and sampling it would be ignored")
        gamma = self.hill_number(f)
        beta = gamma / alpha
        return BetaDiversity(order=f, gamma=gamma, alpha=alpha, beta=beta,
                             lower=beta, upper=beta, exact=True)
