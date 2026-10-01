"""A repertoire's summaries together WITH its read total.

A repertoire is N reads. The question the chain asks of it is conditional on
N -- "how surprising are n distinct sequences, given N reads" -- and the
answer is not the same as in a world whose read total is itself random. Where
most reads are new sequences, which is where repertoires are, n hardly varies
once N is fixed: on a small model at N = 30 its spread is 1.5, against 4.8
with the total left free.

The Poisson field still does the work. Each sequence's count is independent
Poisson(N P(s)) there, and conditioning on the total is a ratio:

    P(n | N) = P(n, N) / P(N),      P(N) = Poisson(N; N)

so each piece needs the joint probability of its summary and the read total.
Both joints here are sums of independent units, and both are taken by the
same saddlepoint as :mod:`.lattice`: tilt every unit so the observed values
become the mean, take the normal density there, undo the tilt exactly. Its
error is reported with every value, ``sum_j 1/(12 v_j)`` nats over the tilted
conditional variances ``v_j``.

  * :func:`present_and_reads_log_pmf` -- ``P(n distinct, N reads)`` over the
    whole model, its sequences grouped into spectrum atoms. Per sequence the
    pair (seen, count) has the closed form
    ``E[e^{a seen + b c}] = e^{-lam} (1 + e^a (e^{lam e^b} - 1))``.
  * :func:`counts_given_set_log_pmf` -- ``P(N reads | the set seen)`` and
    ``P(abundance pattern, N reads | the set seen)`` over the observed
    sequences, each count zero-truncated Poisson.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, gammainc, gammaln

from .. import _cbeam as _c
from .lattice import categorical_sum_log_pmf

__all__ = ["present_and_reads_log_pmf", "counts_given_set_log_pmf",
           "expected_bins_given_reads"]


def _ztp_reduce(lam, t_bins, edges, allowed, tau):
    """The tilted bin moments summed over the sequences, in C
    (math/ztp_moments.c): log normaliser, bin probabilities, E[count 1_bin],
    E[count], E[count^2] and the Gram matrix of (bin probabilities, E[count]).

    The numpy form this replaces, `_Units.moments`, built a
    (sequences x counts) matrix per Newton step and scattered it into bins
    with `np.add.at`; on a 100,000-sequence repertoire that was 11 of the 12
    seconds a donor cost, to produce the eight numbers per sequence that are
    all the caller sums anyway. `_Units` is kept as the reference the C
    kernel is held to (tests/python/likelihood/test_ztp_moments.py).
    """
    B = len(edges)
    raw = _c.ztp_bin_reduce(np.ascontiguousarray(lam, dtype=float),
                            np.ascontiguousarray(t_bins, dtype=float),
                            np.ascontiguousarray(edges, dtype=float),
                            np.ascontiguousarray(allowed, dtype=float), float(tau))
    v = np.frombuffer(raw, dtype=float)
    return (v[0], v[1:1 + B], v[1 + B:1 + 2 * B], v[1 + 2 * B], v[2 + 2 * B],
            v[3 + 2 * B:].reshape(B + 1, B + 1))

_GRAD_TOL = 1e-10
_MAX_NEWTON = 200
_CHUNK = 20000


def _log_expm1(x):
    """``log(e^x - 1)`` for x > 0, without overflow above or loss below."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    big = x > 30.0
    out[big] = x[big] + np.log1p(-np.exp(-x[big]))
    out[~big] = np.log(np.expm1(x[~big]))
    return out


def _newton(fn, x0, targets):
    """Minimise the convex ``phi`` that ``fn(x) -> (phi, grad, hess)`` gives,
    by damped Newton with backtracking. Returns the last ``(x, phi, hess)``."""
    x = np.asarray(x0, dtype=float)
    phi, grad, hess = fn(x)
    for _ in range(_MAX_NEWTON):
        if np.all(np.abs(grad) <= _GRAD_TOL * np.maximum(1.0, np.abs(targets))):
            return x, phi, hess
        try:
            step = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, -grad, rcond=None)[0]
        big = float(np.max(np.abs(step)))
        if big > 20.0:
            step *= 20.0 / big
        slope = float(grad @ step)
        s = 1.0
        cand = fn(x + step)
        if -slope > 1e-12 * max(1.0, abs(phi)):
            while not (cand[0] <= phi + 1e-4 * s * slope) and s > 1e-12:
                s *= 0.5
                cand = fn(x + s * step)
        x = x + s * step
        phi, grad, hess = cand
    raise RuntimeError(f"the saddlepoint did not converge; gradient {grad}")


def _normal_term(hess):
    """``-(d/2) log 2 pi - log det / 2`` and the error estimate."""
    d = hess.shape[0]
    sign, logdet = np.linalg.slogdet(hess)
    if sign <= 0:
        raise RuntimeError("the saddlepoint Hessian is not positive definite")
    cond = 1.0 / np.diag(np.linalg.inv(hess))
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet), float(np.sum(1.0 / (12.0 * cond)))


# ── (distinct, reads) over the whole model ──────────────────────

def present_and_reads_log_pmf(p, mult, depth: float, n: int, total: int):
    """``log P(n distinct sequences seen, total reads)`` in the Poisson field
    at ``depth``: every sequence independently Poisson(depth * p).

    Args:
        p: per-unit sequence probabilities (spectrum atoms, or sequences).
        mult: how many sequences each unit stands for; real, may be 1e33.
        depth: the Poisson depth, N for a repertoire of N reads.
        n, total: the observed distinct count and read total.

    Returns ``(log_probability, exact, error)``.
    """
    p = np.asarray(p, dtype=float)
    m = np.asarray(mult, dtype=float)
    keep = (p > 0) & (m > 0)
    p, m = p[keep], m[keep]
    lam = depth * p
    n, N = int(n), int(total)
    if n < 0 or N < n or (n == 0) != (N == 0):
        return -math.inf, True, 0.0
    if n == 0:
        return -float(np.sum(m * lam)), True, 0.0
    if N == n:
        # Every seen sequence read once: P(no count above 1) times the chance
        # that exactly n land on "once" given that.
        base = float(np.sum(m * (-lam + np.log1p(lam))))
        lq = np.column_stack([-np.log1p(lam), np.log(lam) - np.log1p(lam)])
        val, exact, err = categorical_sum_log_pmf(lq, m, np.array([0.0, n]), rest=0)
        return base + val, exact, err

    def fn(x):
        a, b = x
        mu = lam * math.exp(b)
        log_h = a + _log_expm1(mu)
        w = expit(log_h)
        w_bar = expit(-log_h)
        rho = mu / -np.expm1(-mu)                 # mean count of a seen sequence
        ec = w * rho
        K = float(np.sum(m * (np.logaddexp(0.0, log_h) - lam)))
        grad = np.array([float(np.sum(m * w)) - n, float(np.sum(m * ec)) - N])
        vii = float(np.sum(m * w * w_bar))
        vic = float(np.sum(m * ec * w_bar))
        vcc = float(np.sum(m * ec * (mu + 1.0 - ec)))
        return K - a * n - b * N, grad, np.array([[vii, vic], [vic, vcc]])

    # Start where the distinct count is right at the depth's own tilt.
    a = 0.0
    for _ in range(200):
        w = expit(a + _log_expm1(lam))
        f = float(np.sum(m * w)) - n
        v = float(np.sum(m * w * (1 - w)))
        if abs(f) <= 1e-9 * max(1.0, n) or v <= 0:
            break
        a -= max(-20.0, min(20.0, f / v))
    x, phi, hess = _newton(fn, np.array([a, 0.0]), np.array([n, N], dtype=float))
    term, err = _normal_term(hess)
    return phi + term, False, err


# ── (pattern, reads) over the observed sequences ────────────────

def _log_poisson_tail(k: int, mu: np.ndarray) -> np.ndarray:
    """``log P(Poisson(mu) >= k)``: a series below k, where it is tiny, and
    the regularised incomplete gamma above."""
    mu = np.asarray(mu, dtype=float)
    if k <= 0:
        return np.zeros_like(mu)
    out = np.empty_like(mu)
    small = mu < k
    if np.any(small):
        ms = mu[small]
        first = k * np.log(ms) - ms - math.lgamma(k + 1)
        total = np.ones(ms.size)
        term = np.ones(ms.size)
        j = 0
        while True:
            j += 1
            term = term * ms / (k + j)
            total += term
            if np.all(term <= 1e-17 * total):
                break
        out[small] = first + np.log(total)
    big = ~small
    out[big] = np.log(gammainc(k, mu[big]))
    return out


def _bin_index(edges, top):
    """The bin of each finite count 1 .. top-1."""
    r = np.arange(1, top)
    return np.searchsorted(np.asarray(edges), r, side="right") - 1


class _Units:
    """Seen sequences, each count zero-truncated Poisson(lam), binned."""

    def __init__(self, lam, edges):
        self.lam = np.asarray(lam, dtype=float)
        self.edges = tuple(edges)
        self.B = len(self.edges)
        self.top = self.edges[-1]
        self.r = np.arange(1, self.top, dtype=float)          # finite counts
        self.rbin = _bin_index(self.edges, self.top)
        self.lgam = gammaln(self.r + 1.0)
        self.lnorm = _log_expm1(self.lam)                    # log(e^lam - 1)

    def probabilities(self, t_bins, tau, allowed, lo=0, hi=None):
        """Tilted P(c = r) for the finite counts (U, R), the top bin's mass
        (U,) and its mean count (U,); sets ``top_r2``, its second moment."""
        lam = self.lam[lo:hi]
        mu = lam * math.exp(tau)
        lw = (np.outer(np.log(mu), self.r) - self.lgam[None, :]
              + t_bins[self.rbin][None, :])
        lw[:, ~allowed[self.rbin]] = -np.inf
        if allowed[-1]:
            lq = _log_poisson_tail(self.top, mu)
            ltop = mu + lq + t_bins[-1]
            r1 = mu * np.exp(_log_poisson_tail(self.top - 1, mu) - lq)
            self.top_r2 = mu * mu * np.exp(_log_poisson_tail(self.top - 2, mu) - lq) + r1
        else:
            ltop = np.full(lam.size, -np.inf)
            r1 = np.zeros(lam.size)
            self.top_r2 = np.zeros(lam.size)
        allw = np.column_stack([lw, ltop])
        e = np.exp(allw - np.max(allw, axis=1)[:, None])
        pr = e / e.sum(axis=1)[:, None]
        return pr[:, :-1], pr[:, -1], r1

    def moments(self, t_bins, tau, allowed, lo=0, hi=None):
        """Per-unit log normaliser and tilted moments for units [lo, hi):
        the bin probabilities (U, B), E[c], E[c^2] and E[1_b c] (U, B)."""
        lam = self.lam[lo:hi]
        U = lam.size
        mu = lam * math.exp(tau)
        lmu = np.log(mu)
        # finite counts: log mu^r / r! - log(e^lam - 1) + t_bin
        lw = (np.outer(lmu, self.r) - self.lgam[None, :]
              + t_bins[self.rbin][None, :])
        lw[:, ~allowed[self.rbin]] = -np.inf
        # the open top bin, closed form: e^mu P(Poisson(mu) >= top)
        if allowed[-1]:
            lq = _log_poisson_tail(self.top, mu)
            ltop = mu + lq + t_bins[-1]
            lq1 = _log_poisson_tail(self.top - 1, mu)
            lq2 = _log_poisson_tail(self.top - 2, mu)
            r1 = mu * np.exp(lq1 - lq)                         # E[c | top]
            r2 = mu * mu * np.exp(lq2 - lq) + r1               # E[c^2 | top]
        else:
            ltop = np.full(U, -np.inf)
            r1 = r2 = np.zeros(U)
        allw = np.column_stack([lw, ltop])
        mx = np.max(allw, axis=1)
        e = np.exp(allw - mx[:, None])
        z = e.sum(axis=1)
        logZ = mx + np.log(z) - self.lnorm[lo:hi]
        pr = e / z[:, None]
        pf, pt = pr[:, :-1], pr[:, -1]
        Pb = np.zeros((U, self.B))
        np.add.at(Pb.T, self.rbin, pf.T)
        Pb[:, -1] += pt
        Rb = np.zeros((U, self.B))
        np.add.at(Rb.T, self.rbin, (pf * self.r).T)
        Rb[:, -1] += pt * r1
        Ec = Rb.sum(axis=1)
        Ec2 = (pf * self.r * self.r).sum(axis=1) + pt * r2
        return logZ, Pb, Ec, Ec2, Rb


def counts_given_set_log_pmf(lam, total: int, edges=None, F=None):
    """``log P(total reads | the set seen)``, or with ``edges`` and ``F``
    ``log P(abundance pattern F, total reads | the set seen)``: each seen
    sequence's count independent zero-truncated Poisson(lam).

    ``edges`` are the lower edges of the count bins, from 1, the last open;
    ``F`` the number of sequences in each. Returns
    ``(log_probability, exact, error)``.
    """
    lam = np.asarray(lam, dtype=float)
    n = lam.size
    N = int(total)
    if edges is None:
        edges, F = (1,), np.array([float(n)])
    edges = tuple(int(e) for e in edges)
    F = np.asarray(F, dtype=float)
    if abs(float(F.sum()) - n) > 1e-9 or N < n:
        return -math.inf, True, 0.0
    if N == n:
        # every count is 1: exact
        if len(edges) > 1 and F[0] != n:
            return -math.inf, True, 0.0
        return float(np.sum(np.log(lam) - _log_expm1(lam))), True, 0.0
    units = _Units(lam, edges)
    B = units.B
    allowed = F > 0
    # the smallest total the allowed bins permit, per unit
    if not allowed[-1] and not np.any(allowed[:-1]):
        return -math.inf, True, 0.0
    finite_width = np.diff(np.asarray(edges + (edges[-1] + 1,)))
    if not allowed[-1] and np.all(finite_width[:-1][allowed[:-1]] == 1):
        # Every non-empty bin holds one count value, so the pattern fixes the
        # total: P(F, N | S) is P(F | S) when N is that total, else zero.
        if int(round(float(np.dot(F[:-1], np.asarray(edges[:-1]))))) != N:
            return -math.inf, True, 0.0
        Pb = np.concatenate([units.moments(np.zeros(B), 0.0, np.ones(B, bool), lo,
                                           min(n, lo + _CHUNK))[1]
                             for lo in range(0, n, _CHUNK)])
        with np.errstate(divide="ignore"):
            return categorical_sum_log_pmf(np.log(Pb), np.ones(n), F)
    # The total at the edge of what the pattern allows -- every count at its
    # bin's floor, or (no open bin occupied) at its ceiling -- leaves nothing
    # for the total to vary over: P(F, N | S) is then the probability that
    # the counts sit exactly there, a categorical sum the lattice takes with
    # its nearly-fixed categories convolved exactly. The joint saddlepoint is
    # singular there, and wrong by nats close to it.
    floors = np.asarray(edges, dtype=float)
    ceils = np.asarray(edges[1:] + (np.inf,), dtype=float) - 1.0
    n_min = float(np.dot(F, floors))
    n_max = float(np.dot(F[allowed], ceils[allowed]))
    if N < n_min or N > n_max:
        return -math.inf, True, 0.0
    if N in (n_min, n_max):
        at = np.where(allowed, floors if N == n_min else ceils, 1.0)
        with np.errstate(divide="ignore"):
            la = (np.outer(np.log(lam), at) - gammaln(at + 1.0)[None, :]
                  - units.lnorm[:, None])
        la[:, ~allowed] = -np.inf
        top = np.max(la, axis=1)
        lw = top + np.log(np.sum(np.exp(la - top[:, None]), axis=1))
        val, exact, err = categorical_sum_log_pmf(la - lw[:, None], np.ones(n), F)
        return float(np.sum(lw)) + val, exact, err
    ref = int(np.argmax(F))
    act = [b for b in range(B) if allowed[b] and b != ref]
    d = len(act) + 1
    targets = np.array([F[b] for b in act] + [float(N)])

    edges_a = np.asarray(edges, dtype=float)
    allowed_a = allowed.astype(float)
    k = len(act)
    take = list(act) + [B]              # the Gram's rows the step uses

    def fn(x):
        t_bins = np.zeros(B)
        t_bins[act] = x[:-1]
        sum_logZ, Pb, Rb, Ec, Ec2, gram = _ztp_reduce(
            lam, t_bins, edges_a, allowed_a, x[-1])
        grad = np.concatenate([Pb[act], [Ec]]) - targets
        hess = np.zeros((d, d))
        hess[:k, :k] = np.diag(Pb[act])
        hess[:k, k] = Rb[act]
        hess[k, :k] = Rb[act]
        hess[k, k] = Ec2
        hess -= gram[np.ix_(take, take)]    # the outer-product term
        return sum_logZ - float(x @ targets), grad, hess

    x, phi, hess = _newton(fn, np.zeros(d), targets)
    term, err = _normal_term(hess)
    if d == 1 and B == 1:
        # The read total alone: with few sequences its distribution is skewed
        # and the normal term's error is larger than 1/(12 v). Add the
        # second-order term from the tilted cumulants, and report its size
        # as the error -- a bound on what is left after it.
        corr = _ztp_sum_correction(lam * math.exp(float(x[-1])))
        return phi + term + corr, False, abs(corr)
    return phi + term, False, err


def _ztp_sum_correction(mu: np.ndarray) -> float:
    """``k4 / (8 k2^2) - 5 k3^2 / (24 k2^3)`` for a sum of zero-truncated
    Poisson(mu): the second-order term of the saddlepoint density."""
    D = -np.expm1(-mu)
    r1 = mu / D
    r2 = (mu * mu + mu) / D
    r3 = (mu ** 3 + 3 * mu * mu + mu) / D
    r4 = (mu ** 4 + 6 * mu ** 3 + 7 * mu * mu + mu) / D
    c2 = r2 - r1 * r1
    c3 = r3 - 3 * r1 * r2 + 2 * r1 ** 3
    c4 = r4 - 4 * r1 * r3 + 6 * r1 * r1 * r2 - 3 * r1 ** 4
    k2, k3, k4 = float(c2.sum()), float(c3.sum()), float((c4 - 3 * c2 * c2).sum())
    if k2 <= 0:
        return 0.0
    return k4 / (8 * k2 * k2) - 5 * k3 * k3 / (24 * k2 ** 3)


def expected_bins_given_reads(lam, total: int, edges) -> np.ndarray:
    """``E[F | the set seen, total reads]``: each sequence's chance of each
    count bin given that all the counts add up to the total, in C
    (math/ztp_moments.c).

    Under the tilt that makes the expected total N, sequence i's count is
    r with probability pi_i(r). Holding the total at N, the others must make
    up N - r, and the chance of that falls as a normal in r around the
    sequence's own tilted mean m_i, with the variance V - v_i of the others:

        P(c_i = r | N)  proportional to  pi_i(r) exp(-(r - m_i)^2 / 2 (V - v_i))

    The first factor alone (the tilt) is off by order 1/n -- 16% in the
    rarest bin of forty sequences; with the second, 9% there and 3% in the
    others. It is only shown beside the observed pattern: the profile's
    probability is the saddlepoint, held to exact convolution.
    """
    lam = np.ascontiguousarray(np.asarray(lam, dtype=float))
    raw = _c.ztp_bin_expect(lam, np.ascontiguousarray(np.asarray(edges, dtype=float)),
                            float(total))
    return np.frombuffer(raw, dtype=float).copy()

