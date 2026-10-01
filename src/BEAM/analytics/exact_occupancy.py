"""Occupancy quantities computed exactly from power sums, with no binned spectrum.

Every quantity here is a sum over all the model's sequences of a function of
P(s): expected richness, how many sequences are drawn exactly j times, how many
two repertoires share, how many only one repertoire in a cohort holds. Such a
function does not factor along a sequence, so the dynamic program cannot sum
it directly -- but its power sums

    M(q) = sum_s P(s)^q

it can, exactly, at any complex q (`BeamModel.log_mellin_line`). Each quantity
is then one of two things:

SERIES, for small depths. 1 - (1-p)^n = sum_k (-1)^{k+1} C(n,k) p^k, so
    R(n) = sum_k (-1)^{k+1} C(n,k) M(k),
with M at the positive integers. For integer n it is a finite sum and exact;
otherwise it converges fast while n * p_max is small.

CONTOUR, for the rest (Mellin-Barnes). The Mellin transform of (1-x)^n on
[0, 1] is B(z, n+1) = Gamma(z) Gamma(n+1) / Gamma(n+1+z), so for -1 < c < 0

    R(n) = -(1/2pi) int B(c+it, n+1) M(-(c+it)) dt,

a one-dimensional integral along a vertical line, taken by the trapezoid rule
on stored values M(q - i k h), q = -c. Richness differences, occupancy counts
and sharing have kernels of the same family.

WHAT MAKES IT EXACT, and not merely accurate, is that each value carries a
rigorous bound, from four terms:

  ALIASING. By Poisson summation the step-h rule sums, for every sequence,
  f(s + m D) e^{-q m D}, D = 2 pi/h, over all integers m, where f is the
  per-sequence function of the surprisal s = -ln P. m = 0 is the answer; the
  rest are bounded from f's own range. For richness 0 <= f <= min(1, n P), so
  m > 0 adds at most D0 e^{-q D} (D0 = M(0), the number of sequences) and
  m < 0 at most n M(1) e^{-(1-q) D}. The planner picks q and h per depth from
  exactly this: q D >= ln(D0 / (eps V)) and (1-q) D >= ln(n / (eps V)).
  Measured on enumerated models, the bound is attained within 0.02% where
  nearly every sequence has f = 1; it carries a 1% margin.

  TRUNCATION beyond the last node, from the kernel's Gamma decay e^{-pi t/2}.

  ROUNDING, which is in the PHASE: a term's phase is t (ln n + s), thousands
  of radians far out on the line, and it is good to eps of that size.

  and the RELATIVE form divides by |value| - bound, the smallest the truth can
  be, never by the value itself, which the error may have inflated.

A value that could not be brought within the requested tolerance is returned
with the bound it has, never silently. Every quantity here is non-negative, so
one no contour resolves at all is returned as 0 with relative bound 1, which
is true of any non-negative number, and marked "unresolved".

What this does NOT replace: counts of sequences in a RANGE of probability --
the spectrum's histograms. Those are threshold counts; see
`docs/api/python/analytics.md`.
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import gammaln, loggamma, polygamma

from .. import _cbeam as _c

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = ["BoundedValue", "ExactOccupancy"]

#: One INFO record per contour line computed or extended: its cost, and how
#: many of the values waiting on it it resolved.
_log = logging.getLogger(__name__)

EPS = float(np.finfo(float).eps)
#: Below this argument the log-gamma ratio is taken directly; above it, as a
#: Taylor series in z, whose radius of convergence is the argument.
_TAYLOR_W = 200.0
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class BoundedValue:
    """A value and a rigorous bound on its RELATIVE error.

    ``|value - truth| <= bound * |truth|``. Deliberately not a float:
    ``float(v)`` raises, so the bound cannot be dropped by accident. Read
    ``v.value`` when the bound has been looked at.
    """
    value: float
    bound: float
    method: str

    def __float__(self):
        raise TypeError("a BoundedValue is not a float; read .value (and .bound)")

    def __repr__(self) -> str:
        return (f"BoundedValue({self.value:.15g}, relative error <= "
                f"{self.bound:.1e}, {self.method})")


@dataclass
class _Line:
    q: float
    h: float
    t: np.ndarray            # 0, h, 2h, ..
    log_m: np.ndarray        # ln M(q - i t)


def _polygammas(w: float, kmax: int) -> np.ndarray:
    """psi^(k)(w) for k = 0..kmax, zero where they underflow."""
    ks = np.arange(kmax + 1)
    with np.errstate(all="ignore"):
        v = polygamma(ks, w)
    v = np.where(np.isfinite(v), v, 0.0)
    return v


def _log_gamma_ratio_c(w, z) -> np.ndarray:
    """ln Gamma(w_k) - ln Gamma(w_k + z_i) for every node z_i and every w_k,
    shape (len(z), len(w)), in C (math/lgamma_ratio.c): Stirling's series for
    the difference, a few rounding units absolute of |z| ln(w + |z|), the
    imaginary part modulo 2 pi. Every thread, and no matrix per term: the
    numpy series it replaces had become most of the non-sweep time of the
    figure steps that ask for tens of thousands of depths."""
    w = np.ascontiguousarray(np.asarray(w, dtype=float).ravel())
    z = np.asarray(z, dtype=np.complex128).ravel()
    zr = np.ascontiguousarray(z.real)
    zi = np.ascontiguousarray(z.imag)
    out = np.frombuffer(_c.log_gamma_ratio(w, zr, zi), dtype=np.complex128)
    return out.reshape(z.size, w.size).copy()


def _log_ratio(w: float, z: np.ndarray) -> np.ndarray:
    """ln Gamma(w) - ln Gamma(w + z), real w >= 1, complex z, as
    `_log_gamma_ratio_c` takes it."""
    z = np.asarray(z, dtype=np.complex128)
    return _log_gamma_ratio_c([float(w)], z.ravel())[:, 0].reshape(z.shape)


def _log_ratio_numpy(w: float, z: np.ndarray) -> np.ndarray:
    """ln Gamma(w) - ln Gamma(w + z), real w >= 1, complex z: the numpy form
    `_log_ratio` replaced, kept as the tests' second opinion.

    Direct below _TAYLOR_W. Above, the two log-gammas are huge and nearly
    equal, so the difference is the Taylor series
        -sum_{k>=0} psi^(k)(w) z^(k+1) / (k+1)!,
    whose terms fall like (|z|/w)^k.
    """
    z = np.asarray(z, dtype=np.complex128)
    if w < _TAYLOR_W:
        return gammaln(w) - loggamma(w + z)
    far = np.abs(z) > 0.3 * w
    if np.any(far):
        # outside the series' reach; only ever needed for magnitudes, far out
        # on the line, where the direct difference is accurate enough
        out = np.empty_like(z)
        out[far] = gammaln(w) - loggamma(w + z[far])
        if np.any(~far):
            out[~far] = _log_ratio_numpy(w, z[~far])
        return out
    pg = _polygammas(w, 90)
    out = np.zeros_like(z)
    zp = z.copy()
    fact = 1.0
    for k in range(90):
        fact *= (k + 1)
        term = -pg[k] * zp / fact
        out += term
        if np.max(np.abs(term)) <= 1e-19 * max(1.0, float(np.max(np.abs(out)))):
            break
        zp = zp * z
    return out


def _log_ratio_matrix(w: np.ndarray, z: np.ndarray) -> np.ndarray:
    """_log_ratio for many w at once: shape (len(z), len(w)), in C."""
    return _log_gamma_ratio_c(w, z)


def _log_ratio_matrix_numpy(w: np.ndarray, z: np.ndarray) -> np.ndarray:
    """_log_ratio_numpy for many w at once: shape (len(z), len(w)) -- the
    numpy form `_log_ratio_matrix` replaced, kept as the tests' second opinion.

    The Taylor series where w >= _TAYLOR_W and |z| <= 0.3 w, the direct
    difference elsewhere (small w, or far out on the line where only a
    magnitude is wanted and the series would not converge).
    """
    w = np.asarray(w, dtype=float)
    z = np.asarray(z, dtype=np.complex128)
    Z = z[:, None]
    out = np.zeros((z.size, w.size), dtype=np.complex128)
    direct = (w[None, :] < _TAYLOR_W) | (np.abs(Z) > 0.3 * w[None, :])
    tay_cols = ~np.all(direct, axis=0)
    if np.any(tay_cols):
        wt = w[tay_cols]
        ks = np.arange(90)
        with np.errstate(all="ignore"):
            pg = polygamma(ks[:, None], wt[None, :])
        pg = np.where(np.isfinite(pg), pg, 0.0)
        acc = np.zeros((z.size, wt.size), dtype=np.complex128)
        zp = Z.copy()
        fact = 1.0
        with np.errstate(all="ignore"):
            for k in range(90):
                fact *= (k + 1)
                term = -pg[k][None, :] * zp / fact
                acc += term
                ok = ~direct[:, tay_cols]
                if not np.any(ok) or np.nanmax(np.abs(term[ok])) <= 1e-19 * max(
                        1.0, float(np.nanmax(np.abs(acc[ok])))):
                    break
                zp = zp * Z
        out[:, tay_cols] = acc
    if np.any(direct):
        rows, cols = np.nonzero(direct)
        out[rows, cols] = gammaln(w[cols]) - loggamma(w[cols] + z[rows])
    return out


def _log_ratio_diff(w1: float, w2: float, z: np.ndarray) -> np.ndarray:
    """_log_ratio(w1, z) - _log_ratio(w2, z), accurate when w1 is close to w2.

    The difference of the Taylor coefficients, psi^(k)(w1) - psi^(k)(w2), is
    itself taken as a Taylor series in d = w1 - w2 about w2, so nothing is
    formed as a difference of two nearly equal numbers.
    """
    d = w1 - w2
    if min(w1, w2) < _TAYLOR_W or abs(d) > 0.25 * w2:
        return _log_ratio(w1, z) - _log_ratio(w2, z)
    pg = _polygammas(w2, 160)
    # dpsi[k] = sum_{m>=1} psi^(k+m)(w2) d^m / m!
    kmax = 90
    dpsi = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        acc, dm, mf = 0.0, 1.0, 1.0
        for m in range(1, 160 - k):
            dm *= d
            mf *= m
            t = pg[k + m] * dm / mf
            acc += t
            if abs(t) <= 1e-19 * abs(acc):
                break
        dpsi[k] = acc
    out = np.zeros_like(z)
    zp = z.copy()
    fact = 1.0
    for k in range(kmax):
        fact *= (k + 1)
        term = -dpsi[k] * zp / fact
        out += term
        if np.max(np.abs(term)) <= 1e-19 * max(1e-300, float(np.max(np.abs(out)))):
            break
        zp = zp * z
    return out


@contextlib.contextmanager
def _locked(path: Path):
    """An exclusive lock beside `path`, where the platform has one."""
    try:
        import fcntl
    except ImportError:          # no advisory locks: merging still helps
        yield
        return
    with open(f"{path}.lock", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ── the sampling deficit ────────────────────────────────────────
#
# D(lam) = E[ln(1 + Y)] - ln lam, Y ~ Poisson(lam), is the entropy a sequence
# of expected count lam loses to being sampled rather than known: the sampled
# Hill D1 of a depth-N Poisson sample is exp(H - S(N)), S(N) = sum_s P D(N P).
#
# Its Mellin transform, for 0 < Re w < 1, is
#
#     D~(w) = Gamma(w) J(w),   J(w) = 1/w + int_0^1 v^-w psi(v) dv,
#     psi(v) = 1/v + 1/ln(1-v) = sum_{n>=1} |G_n| v^(n-1),
#
# (Frullani's integral for ln(1+Y) and for ln lam, then the Mellin transform of
# each exponential), |G_n| the Gregory coefficients: positive, summing to 1,
# so psi rises from 1/2 to 1 and |J(w)| <= 1/|w| + 1/(1 - Re w).
#
# J is taken in three pieces, each to double precision:
#   [0, 1/2]  term by term, sum_n |G_n| 2^(w-n) / (n - w): geometric;
#   [1/2, 1]  the 1/v part in closed form, (2^w - 1)/w, and the 1/ln(1-v)
#             part, after 1 - v = e^-x / 2, as
#             -(1/2) int_0^inf e^-x (1 - e^-x/2)^-w / (ln 2 + x) dx.
#             That integrand is smooth but turns through about 0.69 Im(w)
#             radians, nearly all of it below x = 8: composite Gauss-Legendre
#             there, 128 panels of 16 points, and Gauss-Laguerre beyond, where
#             it no longer oscillates.
# The tests hold D~ to 1e-13 of a 30-digit mpmath reference.

_LN2 = math.log(2.0)


def _gregory_abs(nmax: int) -> np.ndarray:
    """|G_1| .. |G_nmax|, exactly, as the coefficients of psi(v).

    v / (-ln(1-v)) is the reciprocal of sum_k v^k / (k+1); its coefficients
    b_k have b_0 = 1 and b_k = -|G_k| after.
    """
    from fractions import Fraction
    a = [Fraction(1, k + 1) for k in range(nmax + 1)]
    b = [Fraction(1)] + [Fraction(0)] * nmax
    for k in range(1, nmax + 1):
        b[k] = -sum(a[j] * b[k - j] for j in range(1, k + 1))
    return np.array([float(-b[k]) for k in range(1, nmax + 1)])


_GREGORY = _gregory_abs(64)
_LAGUERRE: tuple | None = None
_KERNEL_NODES: tuple | None = None
_X_SPLIT = 8.0


def _kernel_nodes():
    """Nodes and weights for int_0^inf e^-x f(x) dx: composite Gauss-Legendre
    on [0, _X_SPLIT] (weights carrying e^-x), Gauss-Laguerre beyond."""
    global _KERNEL_NODES
    if _KERNEL_NODES is None:
        g, gw = np.polynomial.legendre.leggauss(16)
        edges = np.linspace(0.0, _X_SPLIT, 129)
        a, b = edges[:-1, None], edges[1:, None]
        xs = (0.5 * (b - a) * g[None, :] + 0.5 * (a + b)).ravel()
        ws = (0.5 * (b - a) * gw[None, :]).ravel() * np.exp(-xs)
        from scipy.special import roots_laguerre
        lx, lw = roots_laguerre(60)
        xs = np.concatenate([xs, _X_SPLIT + lx])
        ws = np.concatenate([ws, math.exp(-_X_SPLIT) * lw])
        _KERNEL_NODES = (xs, ws)
    return _KERNEL_NODES


def _laguerre():
    global _LAGUERRE
    if _LAGUERRE is None:
        # scipy's rule: numpy's laggauss overflows its weights at this size
        from scipy.special import roots_laguerre
        x, wt = roots_laguerre(200)
        keep = wt > 0
        _LAGUERRE = (x[keep], wt[keep])
    return _LAGUERRE


def _deficit_kernel(w: np.ndarray) -> np.ndarray:
    """D~(w) for 0 < Re w < 1, elementwise."""
    w = np.asarray(w, dtype=np.complex128).reshape(-1)
    n = np.arange(1, _GREGORY.size + 1, dtype=np.float64)
    head = np.sum(_GREGORY[None, :] * np.exp((w[:, None] - n[None, :]) * _LN2)
                  / (n[None, :] - w[:, None]), axis=1)
    closed = np.expm1(w * _LN2) / w
    x, wt = _kernel_nodes()
    g = (np.exp(-w[:, None] * np.log1p(-0.5 * np.exp(-x))[None, :])
         / (_LN2 + x)[None, :])
    log_part = 0.5 * (g @ wt)
    return np.exp(loggamma(w)) * (1.0 / w + head + closed - log_part)


def _deficit_kernel_abs_bound(w: np.ndarray) -> np.ndarray:
    """An upper bound on |D~(w)|: |Gamma(w)| (1/|w| + 1/(1 - Re w))."""
    w = np.asarray(w, dtype=np.complex128)
    return np.exp(loggamma(w).real) * (1.0 / np.abs(w) + 1.0 / (1.0 - w.real))


_DEFICIT_SERIES: np.ndarray | None = None


def _deficit_series_coeffs(kmax: int) -> np.ndarray:
    """c_k = Delta^k ln(1+x) at 0, over k!, for k = 1..kmax: the coefficients
    of E[ln(1+Y)] = sum_k c_k lam^k.

    |Delta^k ln(1+x)|_0 = int_0^1 u^k / (-ln(1-u)) du, with sign (-1)^(k+1):
    decreasing in k, which is what makes the truncated series' error its
    first omitted term. Taken by Gauss-Laguerre after u = 1 - e^-x.
    """
    global _DEFICIT_SERIES
    if _DEFICIT_SERIES is None or _DEFICIT_SERIES.size < kmax:
        x, wt = _laguerre()
        k = np.arange(1, kmax + 1, dtype=np.float64)
        base = -np.expm1(-x)                          # 1 - e^-x
        # (1 - e^-x)^k / x, in logs: stays finite for every k
        f = np.exp(k[:, None] * np.log(base)[None, :] - np.log(x)[None, :])
        integ = f @ wt
        sign = np.where(k % 2 == 1, 1.0, -1.0)
        _DEFICIT_SERIES = sign * np.exp(np.log(integ) - gammaln(k + 1))
    return _DEFICIT_SERIES[:kmax]


def _check_depth(n, name="n") -> float:
    n = float(n)
    if not math.isfinite(n) or n < 0.0:
        raise ValueError(f"{name} must be a finite non-negative depth, got {n!r}")
    return n


class ExactOccupancy:
    """Occupancy quantities of a model, exactly, with rigorous bounds.

    Args:
        model: a finalized `BeamModel`.
        rel_tol: the relative error each value is planned to. Contour lines
            are designed and computed on demand to meet it, cached, and reused
            by every later query they serve.

    Every method returns `BoundedValue`s. Depths are numbers of independent
    draws (binomial sampling); they need not be integers.
    """

    #: Integer power sums kept for the series route.
    _K_SERIES = 90

    def __init__(self, model, *, rel_tol: float = 1e-10):
        self._m = model
        self.rel_tol = float(rel_tol)
        self._lines: list[_Line] = []
        self._real: dict[float, float] = {}
        self._last_cost = None           # (kind, orders, seconds) of the last line built
        self.log_d0 = float(model.log_mellin(0.0))
        lo, hi = model.surprisal_range()
        self.p_max = math.exp(-lo)
        self.p_min = math.exp(-hi)
        self.s_max = float(hi)
        self._max_len = float(model.config["max_length"])
        self._log_mk_taken = None        # the series route's power sums, on first use
        self._fingerprint_taken = None   # the model's fingerprint, on first use

    # ── bookkeeping ─────────────────────────────────────────────

    @property
    def lines(self) -> list[tuple[float, float, int]]:
        """The contour lines computed so far: (Re q, step h, number of orders)."""
        return [(ln.q, ln.h, ln.t.size) for ln in self._lines]

    def _log_mreal(self, q: float) -> float:
        """ln M(q) at a real q: from a line at that q where there is one (see
        `_line_log_mreal`), otherwise one sweep, kept."""
        if q not in self._real:
            on_line = [ln for ln in self._lines
                       if ln.q == q and ln.t.size and ln.t[0] == 0.0]
            self._real[q] = (float(on_line[0].log_m[0].real) if on_line
                             else float(self._m.log_mellin(q)))
        return self._real[q]

    def _line_log_mreal(self, ln) -> float:
        """ln M(Re q) of a line: its own value at t = 0, where every line
        starts and where it IS the real power sum -- to rounding, which is all
        the bounds that read it need -- rather than a sweep of that real order
        per line per process."""
        if ln.t.size and ln.t[0] == 0.0:
            return float(ln.log_m[0].real)
        return self._log_mreal(ln.q)

    @property
    def _log_mk(self) -> np.ndarray:
        """ln M(k), k = 1 .. _K_SERIES: the series route's power sums, taken on
        first use. They are ninety full sweeps -- about a minute on the
        foundation -- and a workload that only ever reaches the contours never
        needs them."""
        if self._log_mk_taken is None:
            ks = np.arange(1, self._K_SERIES + 1, dtype=float)
            self._log_mk_taken = np.asarray(self._m.log_mellin_many(ks), dtype=float)
        return self._log_mk_taken

    @property
    def _log_m1(self) -> float:
        """ln M(1), one sweep, for the checks that need no other power sum."""
        return self._log_mreal(1.0)

    #: Orders the planning estimates minimise over. Multiples of the 0.02 the
    #: planner snaps q to, so the lines it plans reuse these real power sums.
    _Q_EST = np.round(np.linspace(0.0, 1.0, 11), 2)

    def _log_mgrid(self) -> np.ndarray:
        miss = [q for q in self._Q_EST if float(q) not in self._real]
        if miss:
            for q, v in zip(miss, self._m.log_mellin_many(np.array(miss))):
                self._real[float(q)] = float(v)
        return np.array([self._real[float(q)] for q in self._Q_EST])

    def _estimate(self, kind: str, ns: np.ndarray) -> np.ndarray:
        """A PLANNING estimate of R(n) or U(n): the Chernoff upper bound over
        the grid of real orders, divided by 10.

        1 - (1-p)^n <= (n p)^q, and p (1-p)^(n-1) <= p^q ((1-q)/(e(n-1)))^(1-q),
        for every q in [0, 1]; minimising over q bounds the sum. Measured, the
        bound sits 1 to 15 times above the truth, so a tenth of it is within
        about an order of magnitude either way -- against the endpoints
        min(n, D0) or min(1, D0/n), up to 1e6 above it on the foundation.
        Only the planner reads this; it never enters a value or a bound.
        """
        lm = self._log_mgrid()[None, :]
        q = self._Q_EST[None, :]
        n = np.asarray(ns, dtype=float)[:, None]
        if kind == "richness":
            b = lm + q * np.log(np.maximum(n, 1e-300))
        elif kind == "deficit":
            # D(lam) <= ln(1 + 1/lam) <= lam^-a / a for a in (0, 1]: with
            # q = 1 - a, S(N) <= N^-(1-q) M(q) / (1-q)
            a = 1.0 - q
            with np.errstate(divide="ignore", invalid="ignore"):
                b = np.where(a > 0, lm - a * np.log(np.maximum(n, 1e-300))
                             - np.log(np.where(a > 0, a, 1.0)), np.inf)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                b = lm + (1.0 - q) * np.log((1.0 - q) / (math.e * np.maximum(n - 1.0, 1e-300)))
            b = np.where(q == 1.0, lm, b)
        return np.exp(np.min(b, axis=1)) / 10.0

    def _line(self, q: float, h: float, T: float) -> _Line:
        """A line with this q, step no coarser than h and reach at least T.

        A cached line with the same q and step that falls short of T is
        EXTENDED -- only the missing nodes are computed and appended -- rather
        than recomputed from the start.
        """
        self._last_cost = None
        for ln in self._lines:
            if ln.q == q and ln.h <= h * (1 + 1e-12) and ln.t[-1] >= T:
                return ln
        t0 = time.perf_counter()
        for ln in self._lines:
            if ln.q == q and ln.h == h:
                k0 = ln.t.size
                n = int(math.ceil(T / h)) + 1
                more = self._m.log_mellin_line(q, -h * k0, -h, n - k0)
                ln.log_m = np.concatenate([ln.log_m, np.asarray(more)])
                ln.t = h * np.arange(n)
                self._last_cost = ("extended", n - k0, time.perf_counter() - t0)
                return ln
        n = int(math.ceil(T / h)) + 1
        log_m = self._m.log_mellin_line(q, 0.0, -h, n)
        ln = _Line(q=q, h=h, t=h * np.arange(n), log_m=np.asarray(log_m))
        self._lines.append(ln)
        self._last_cost = ("new", n, time.perf_counter() - t0)
        return ln

    def _fingerprint(self) -> np.ndarray:
        """Exact quantities that identify the model: ln M at a few real
        orders and the config's max_length and order. Taken once: every save
        and load checks it, and each check was four full sweeps."""
        if self._fingerprint_taken is None:
            qs = np.array([0.0, 0.37, 1.73, 3.0])
            self._fingerprint_taken = np.concatenate(
                [np.asarray(self._m.log_mellin_many(qs), dtype=float),
                 [self._max_len, float(self._m.config["order"])]])
        return self._fingerprint_taken

    def save_lines(self, path) -> None:
        """Write the computed lines to `path` (npz), with the model's
        fingerprint, for `load_lines` in a later process.

        Lines already in the file are MERGED in first, under a lock, and the
        file is replaced atomically: processes sharing one file each add what
        they computed and lose nothing another saved, and a reader never sees
        half a file. A file holding another model's lines is refused.
        """
        path = Path(path)
        with _locked(path):
            if path.exists():
                self.load_lines(path)
            arrays = {"fingerprint": self._fingerprint(),
                      "meta": np.array([[ln.q, ln.h, ln.t.size] for ln in self._lines]
                                       ).reshape(-1, 3)}
            for k, ln in enumerate(self._lines):
                arrays[f"logm_{k}"] = ln.log_m
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    np.savez(fh, **arrays)
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(tmp)
                raise

    def load_lines(self, path) -> None:
        """Add the lines in `path` to this object's cache; of two lines with
        the same q and step, the longer is kept. Refuses a file written for a
        different model: a line is M on this model's support."""
        z = np.load(path)
        if not np.array_equal(z["fingerprint"], self._fingerprint()):
            raise ValueError(f"{path} holds lines of a different model")
        for k, (q, h, n) in enumerate(z["meta"]):
            n = int(n)
            same = [ln for ln in self._lines if ln.q == q and ln.h == h]
            if same and same[0].t.size >= n:
                continue
            new = _Line(q=float(q), h=float(h), t=float(h) * np.arange(n),
                        log_m=np.asarray(z[f"logm_{k}"]))
            if same:
                self._lines[self._lines.index(same[0])] = new
            else:
                self._lines.append(new)

    # ── the contour engine ──────────────────────────────────────

    def _integrate(self, ln: _Line, kernel, kernel_abs, sign: float,
                   amp_left: float, amp_right: float, rate_right: float,
                   n_scale: float, extra_phase=None) -> tuple[float, float]:
        """(value, absolute bound) of sign * (1/2pi) int K(z) M(-z) dt on `ln`.

        `kernel(z)` returns K on the nodes z = -q + i t. The integrand is
        conjugate-symmetric in t, so the half-line with weights 1, 2, 2, ..
        is the whole line.
        """
        z = -ln.q + 1j * ln.t
        K = kernel(z)
        I = K * np.exp(ln.log_m)
        w = np.full(ln.t.size, 2.0)
        w[0] = 1.0
        value = sign * (ln.h / _TWO_PI) * float(np.sum(w * I.real))
        D = _TWO_PI / ln.h
        xl = math.exp(-ln.q * D)
        xr = math.exp(-(rate_right - ln.q) * D)
        alias = amp_left * xl / (1 - xl) + amp_right * xr / (1 - xr)
        m_real = math.exp(self._line_log_mreal(ln))       # sum of the moduli
        absK = np.abs(K)
        # KERNEL rounding: the phase t ln n (and, on the direct path, the
        # log-gammas' own size) is carried to eps of its size.
        kphase = 1.0 + ln.t * math.log(max(n_scale, 1.0))
        if extra_phase is not None:
            kphase = kphase + extra_phase
        # DP rounding: the sweep's sums are exact to eps RELATIVE TO THE SUM
        # OF THE MODULI, M(Re q), not to |M(q - i t)|, which cancellation can
        # put far below it; per layer, and in a phase up to t * s_max.
        dphase = self._max_len + ln.t * self.s_max
        floor = EPS * (ln.h / _TWO_PI) * float(
            np.sum(w * (np.abs(I) * kphase + absK * m_real * dphase)))
        trunc = 2.0 * m_real * self._kernel_tail(kernel_abs, ln.q,
                                                  float(ln.t[-1])) / _TWO_PI
        return value, 1.01 * alias + floor + trunc

    @staticmethod
    def _kernel_tail(kernel_abs, q: float, T: float) -> float:
        """Upper bound on int_T^inf |K(-q + i t)| dt.

        |K| decays like e^{-pi t/2} while t is below the depth and like a
        power of t beyond it, so neither law alone is safe. The envelope is
        taken on a geometric grid (left endpoints: an upper sum wherever |K|
        decreases), and the power law fitted over the last decade closes it.
        """
        ts = T * np.geomspace(1.0, 1e8, 481)
        vals = np.asarray(kernel_abs(-q + 1j * ts), dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        # running max from the right makes the envelope non-increasing
        env = np.maximum.accumulate(vals[::-1])[::-1]
        upper = float(np.sum(env[:-1] * np.diff(ts)))
        a = math.log(max(env[-61], 1e-300) / max(env[-1], 1e-300)) / math.log(10.0)
        if env[-1] == 0.0:
            return upper
        if a <= 1.0:
            return math.inf
        return upper + float(env[-1]) * float(ts[-1]) / (a - 1.0)

    def _design(self, value_est: float, amp_left: float, amp_right: float,
                rate_right: float, kernel_abs_at,
                strict: float = 1.0) -> tuple[float, float, float]:
        """(q, h, T) meeting rel_tol for a value of about `value_est`;
        `strict` < 1 asks for proportionally more (a retry after a miss)."""
        target = max(self.rel_tol * 0.05 * strict, 1e-16) * max(value_est, 1e-300)
        L = max(math.log(amp_left / target), 1.0)
        Rr = max(math.log(max(amp_right, 1e-300) / target), 1.0)
        a = rate_right
        q = a * L / (L + Rr)
        q = min(max(round(q / 0.02) * 0.02, 0.02), min(a - 0.02, 0.98))
        # COVERAGE MARGIN. Both alias terms fall exponentially in D, so a line
        # with D 35% beyond this depth's minimum also serves the neighbouring
        # decades, whose requirement differs by less than that. Designed at the
        # bare minimum, each depth range gets a line of its own: 8 lines for
        # the foundation's 1..1e82 where 3 wider ones suffice.
        D = 1.35 * max(L / q, Rr / (a - q))
        h = _TWO_PI / D
        k = math.ceil(math.log(h / 0.12) / math.log(0.92))
        h = 0.12 * 0.92 ** max(k, 0)
        lmq = self._log_mreal(q)
        T = 8.0
        while T < 200.0:
            if kernel_abs_at(-q + 1j * T) * math.exp(lmq) < target:
                break
            T += 4.0
        return q, h, T

    def _solve(self, kernel, kernel_abs, sign, amp_left, amp_right, rate_right,
               n_scale, value_guess, extra_phase_fn=None,
               label="contour") -> BoundedValue:
        """Best value over cached lines; design new ones until rel_tol holds."""
        def on(ln):
            ep = extra_phase_fn(ln) if extra_phase_fn else None
            return self._integrate(ln, kernel, kernel_abs, sign, amp_left,
                                   amp_right, rate_right, n_scale, ep)

        def rel(v, b):
            return b / (abs(v) - b) if abs(v) > b else math.inf

        best = None
        for ln in self._lines:
            v, b = on(ln)
            r = rel(v, b)
            if best is None or r < best[2]:
                best = (v, b, r, ln)
        guess = value_guess
        for attempt in range(4):
            if best is not None and best[2] <= self.rel_tol:
                break
            if best is not None and math.isfinite(best[2]):
                guess = abs(best[0])
            def kabs(zz):
                return float(np.asarray(kernel_abs(np.array([zz])))[0])
            q, h, T = self._design(guess, amp_left, amp_right, rate_right, kabs,
                                   strict=0.1 ** max(attempt - 1, 0))
            ln = self._line(q, h, T)
            v, b = on(ln)
            r = rel(v, b)
            if best is None or r < best[2]:
                best = (v, b, r, ln)
            else:
                guess = abs(v) if math.isfinite(r) else guess / 10
        v, b, r, ln = best
        return BoundedValue(v, r, f"{label} q={ln.q:g} h={ln.h:.4g}")

    # ── series ──────────────────────────────────────────────────

    def _binom_coeffs(self, n: float, kmax: int):
        """(log |C(n,k)|, sign C(n,k)) for k = 0..kmax, generalized binomial,
        and the k at which an integer n terminates the series (or None)."""
        logc = np.zeros(kmax + 1)
        sgn = np.ones(kmax + 1)
        stop = None
        for k in range(1, kmax + 1):
            f = n - (k - 1)
            if f == 0.0:
                stop = k
                logc[k:] = -np.inf
                sgn[k:] = 0.0
                break
            logc[k] = logc[k - 1] + math.log(abs(f)) - math.log(k)
            sgn[k] = sgn[k - 1] * math.copysign(1.0, f)
        return logc, sgn, stop

    def _series_ok(self, n: float) -> bool:
        """Worth trying the series: finite for integer n up to the stored
        power sums, and for any n it converges geometrically in p_max < 1. Whether it is
        ACCURATE is its own bound's business -- an alternating sum with large
        terms says so through the rounding term."""
        return n <= self._K_SERIES or n * self.p_max <= 0.5

    def _richness_series(self, n: float) -> tuple[float, float]:
        """(value, absolute bound) of sum_k (-1)^{k+1} C(n,k) M(k)."""
        K = self._K_SERIES
        logc, sgn, stop = self._binom_coeffs(n, K)
        k = np.arange(1, K + 1)
        alt = np.where(k % 2 == 1, 1.0, -1.0)
        with np.errstate(invalid="ignore"):
            terms = alt * sgn[1:] * np.exp(logc[1:] + self._log_mk)
        terms = np.where(np.isfinite(terms), terms, 0.0)
        value = float(np.sum(terms))
        floor = 16 * EPS * float(np.sum(np.abs(terms)))
        trunc = 0.0 if stop is not None else 2.0 * abs(float(terms[-1]))
        return value, floor + trunc

    # ── public quantities ───────────────────────────────────────

    def _rel(self, value, abs_bound):
        return abs_bound / (abs(value) - abs_bound) if abs(value) > abs_bound else math.inf

    def richness(self, n: float) -> BoundedValue:
        """E[distinct sequences in n draws] = sum_s 1 - (1 - P(s))^n."""
        n = _check_depth(n)
        if n == 0.0:
            return BoundedValue(0.0, 0.0, "exact")
        d0 = math.exp(self.log_d0)
        if n * self.p_min > 800.0:
            # every sequence drawn: (1-p)^n <= e^{-n p_min} for all of them
            return BoundedValue(d0, math.exp(-n * self.p_min), "saturated")
        series = None
        if self._series_ok(n):
            v, b = self._richness_series(n)
            series = BoundedValue(v, self._rel(v, b), "series")
            if series.bound <= self.rel_tol:
                return series
        w = n + 1.0
        direct = w < _TAYLOR_W

        def kernel(z):
            return np.exp(loggamma(z) + _log_ratio(w, z))

        def kernel_abs(z):
            return np.abs(kernel(z))

        extra = ((lambda ln: gammaln(w) + np.abs(loggamma(w - ln.q + 1j * ln.t)))
                 if direct else None)
        contour = self._solve(kernel, kernel_abs, -1.0, d0, n, 1.0, n,
                              min(n, d0), extra_phase_fn=extra)
        return series if series is not None and series.bound < contour.bound \
            else contour

    def richness_difference(self, n1: float, n2: float) -> BoundedValue:
        """R(n1) - R(n2) as one integral, keeping the digits R(n1) and R(n2)
        share: sum_s (1-P)^{n2} - (1-P)^{n1}."""
        n1, n2 = _check_depth(n1, "n1"), _check_depth(n2, "n2")
        if n1 == n2:
            return BoundedValue(0.0, 0.0, "exact")
        if n1 < n2:
            v = self.richness_difference(n2, n1)
            return BoundedValue(-v.value, v.bound, v.method)
        if n2 == 0.0:
            return self.richness(n1)
        series = None
        if self._series_ok(n1) and self._series_ok(n2):
            v1, b1 = self._richness_series(n1)
            v2, b2 = self._richness_series(n2)
            v = v1 - v2
            series = BoundedValue(v, self._rel(v, b1 + b2), "series")
            if series.bound <= self.rel_tol:
                return series
        gap = n1 - n2
        d0 = math.exp(self.log_d0)
        w1, w2 = n1 + 1.0, n2 + 1.0

        def kernel(z):
            lr2 = _log_ratio(w2, z)
            dl = _log_ratio_diff(w1, w2, z)
            return np.exp(loggamma(z) + lr2) * np.expm1(dl)

        def kernel_abs(z):
            # |B1 - B2| <= |B1| + |B2|, valid however far out on the line
            g = loggamma(z)
            return (np.abs(np.exp(g + _log_ratio(w1, z)))
                    + np.abs(np.exp(g + _log_ratio(w2, z))))

        amp_left = d0 * min(1.0, max(gap, 1.0) / w2)
        guess = gap * max(min(n2, d0), 1.0) / max(n1, 1.0)
        extra = ((lambda ln: gammaln(w1) + np.abs(loggamma(w1 - ln.q + 1j * ln.t)))
                 if w2 < _TAYLOR_W else None)
        contour = self._solve(kernel, kernel_abs, -1.0, amp_left, max(gap, 1.0),
                              1.0, n1, guess, extra_phase_fn=extra)
        return series if series is not None and series.bound < contour.bound \
            else contour

    def frequency(self, n: float, j: int) -> BoundedValue:
        """E[sequences drawn exactly j times in n draws],
        sum_s C(n,j) P^j (1-P)^{n-j}, for j >= 1."""
        n = _check_depth(n)
        j = int(j)
        if j < 1:
            raise ValueError(f"j must be at least 1, got {j}")
        if n < j and float(n).is_integer():
            return BoundedValue(0.0, 0.0, "exact")
        logcn = math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        series = None
        if self._series_ok(n) and j + 1 <= self._K_SERIES:
            # C(n,j) sum_i (-1)^i C(n-j, i) M(j+i)
            K = self._K_SERIES - j
            logc, sgn, stop = self._binom_coeffs(n - j, K - 1)
            i = np.arange(K)
            alt = np.where(i % 2 == 0, 1.0, -1.0)
            with np.errstate(invalid="ignore"):
                terms = alt * sgn * np.exp(logcn + logc + self._log_mk[j - 1:j - 1 + K])
            terms = np.where(np.isfinite(terms), terms, 0.0)
            v = float(np.sum(terms))
            b = 16 * EPS * float(np.sum(np.abs(terms))) + (
                0.0 if stop is not None else 2.0 * abs(float(terms[-1])))
            series = BoundedValue(v, self._rel(v, b), "series")
            if series.bound <= self.rel_tol:
                return series
        w = n + 1.0
        d0 = math.exp(self.log_d0)

        def kernel(z):
            return np.exp(loggamma(z + j) - gammaln(j + 1) + _log_ratio(w, z))

        amp_right = math.exp(logcn + self._log_mk[j - 1])
        guess = max(min(n, d0) * 1e-3, 1e-12)
        extra = ((lambda ln: gammaln(w) + np.abs(loggamma(w - ln.q + 1j * ln.t)))
                 if w < _TAYLOR_W else None)
        def kernel_abs(z):
            return np.abs(kernel(z))

        contour = self._solve(kernel, kernel_abs, 1.0, d0, amp_right, float(j),
                              n, guess, extra_phase_fn=extra)
        return series if series is not None and series.bound < contour.bound \
            else contour

    def sharing(self, d1: float, d2: float) -> BoundedValue:
        """E[sequences in both of two independent repertoires of depths d1, d2]
        = R(d1) + R(d2) - R(d1 + d2) = R(d2) - [R(d1 + d2) - R(d1)]."""
        d1, d2 = _check_depth(d1, "d1"), _check_depth(d2, "d2")
        a = self.richness(d2)
        b = self.richness_difference(d1 + d2, d1)
        v = a.value - b.value
        abs_b = a.bound * abs(a.value) / (1 - min(a.bound, 0.5)) \
            + b.bound * abs(b.value) / (1 - min(b.bound, 0.5))
        return BoundedValue(v, self._rel(v, abs_b), "R(d2) - [R(d1+d2) - R(d1)]")

    def private(self, depths) -> BoundedValue:
        """E[sequences seen in exactly one repertoire of a cohort],
        sum_i R(T) - R(T - d_i), T = sum of the depths.

        ONE INTEGRAL FOR THE WHOLE COHORT, not one per repertoire. With
        w = T + 1 fixed, B(z, T+1) - B(z, T-d+1) = -Gamma(z) r(w) expm1(X_d),
        r(w) = Gamma(w)/Gamma(w+z) and X_d = ln r(w-d) - ln r(w), and X_d is
        a power series in d whose coefficients depend only on the node:
            X_d(z) = sum_{m>=1} (-d)^m / m! E_m(z),
            E_m(z) = psi^(m-1)(w) - psi^(m-1)(w+z)
                   = -sum_{k>=1} psi^(m-1+k)(w) z^k / k!.
        So the summed kernel over every depth is a matrix product per line.
        """
        d = np.asarray(depths, dtype=float)
        if d.ndim != 1 or d.size == 0 or not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("depths must be a non-empty 1-D array of finite, "
                             "non-negative numbers")
        T = float(d.sum())
        uniq, counts = np.unique(d[d > 0], return_counts=True)
        if uniq.size == 0:
            return BoundedValue(0.0, 0.0, "exact")
        w = T + 1.0
        if w < _TAYLOR_W or float(uniq[-1]) > 0.25 * w or self._series_ok(T):
            total, abs_b = 0.0, 0.0
            for di, c in zip(uniq, counts):
                v = self.richness_difference(T, T - float(di))
                total += c * v.value
                abs_b += c * v.bound * abs(v.value) / (1 - min(v.bound, 0.5))
            return BoundedValue(total, self._rel(total, abs_b),
                                "sum_i R(T) - R(T - d_i)")
        counts = counts.astype(float)
        mmax = 1
        while mmax < 60 and (uniq[-1] / w) ** mmax / mmax > 1e-19:
            mmax += 1
        pg = _polygammas(w, mmax + 100)
        d0 = math.exp(self.log_d0)

        def e_coeffs(z):
            """E_m(z), m = 1..mmax, shape (mmax, nodes)."""
            out = np.zeros((mmax, z.size), dtype=np.complex128)
            for m in range(1, mmax + 1):
                acc = np.zeros(z.size, dtype=np.complex128)
                zk = np.ones_like(z)
                kf = 1.0
                for k in range(1, 100):
                    zk = zk * z
                    kf *= k
                    term = pg[m - 1 + k] * zk / kf
                    acc += term
                    if np.max(np.abs(term)) <= 1e-19 * max(1e-300, float(np.max(np.abs(acc)))):
                        break
                out[m - 1] = -acc
            return out

        powers = np.stack([(-uniq) ** m / math.factorial(m)
                           for m in range(1, mmax + 1)])        # (mmax, depths)

        def kernel(z):
            z = np.atleast_1d(z)
            E = e_coeffs(z)                                   # (mmax, nodes)
            base = np.exp(loggamma(z) + _log_ratio(w, z))     # Gamma(z) r(w)
            acc = np.zeros(z.size, dtype=np.complex128)
            for lo in range(0, uniq.size, 2048):
                X = E.T @ powers[:, lo:lo + 2048]             # (nodes, chunk)
                acc += np.expm1(X) @ counts[lo:lo + 2048]
            return -base * acc

        grow = w / (w - float(uniq[-1]))

        def kernel_abs(z):
            # |sum_i c_i (B_T - B_{T-d_i})| <= sum_i c_i (|B_T| + |B_{T-d_i}|),
            # and |r(w-d)| <= |r(w)| (w/(w-d)) for 0 <= Re z <= 1 in magnitude
            b = np.abs(np.exp(loggamma(z) + _log_ratio(w, z)))
            return float(counts.sum()) * b * (1.0 + grow)

        amp_left = d0 * float(np.sum(counts * np.minimum(
            1.0, np.maximum(uniq, 1.0) / (w - uniq))))
        amp_right = float(np.sum(counts * np.maximum(uniq, 1.0)))
        guess = float(np.sum(counts * uniq)) * min(T, d0) / w * 0.3
        return self._solve(kernel, kernel_abs, -1.0, amp_left, amp_right, 1.0,
                           T, guess, label="cohort integral")

    # ── the batch engine ────────────────────────────────────────

    _CHUNK = 1024
    #: New lines in a row that resolve nothing before a batch stops planning.
    _FUTILE_LINES = 3

    def _integrate_many(self, ln, kmat, kabs_mat, sign, amp_left, amp_right,
                        rate_right, n_scale, idx, extra=None, alias_fn=None):
        """`_integrate` for many parameters at once: (values, abs bounds,
        abs rounding floors) -- the floor being the part of the bound that
        rounding sets, which no finer or longer line removes.

        `alias_fn(ln, sub, D)`, where given, is the quantity's own aliasing
        bound for the parameters `sub`; otherwise it is the two geometric
        tails `amp_left x_l / (1 - x_l) + amp_right x_r / (1 - x_r)`.
        """
        z = -ln.q + 1j * ln.t
        w = np.full(ln.t.size, 2.0)
        w[0] = 1.0
        D = _TWO_PI / ln.h
        xl = math.exp(-ln.q * D)
        xr = math.exp(-(rate_right - ln.q) * D)
        m_real = math.exp(self._line_log_mreal(ln))
        dphase = (self._max_len + ln.t * self.s_max)[:, None]
        Mv = np.exp(ln.log_m)[:, None]
        vals = np.empty(idx.size)
        bnds = np.empty(idx.size)
        flrs = np.empty(idx.size)
        for lo in range(0, idx.size, self._CHUNK):
            sub = idx[lo:lo + self._CHUNK]
            K = kmat(z, sub)
            I = K * Mv
            vals[lo:lo + sub.size] = sign * (ln.h / _TWO_PI) * (w @ I.real)
            kphase = 1.0 + ln.t[:, None] * np.log(np.maximum(n_scale[sub], 1.0))[None, :]
            if extra is not None:
                kphase = kphase + extra(ln, sub)
            floor = EPS * (ln.h / _TWO_PI) * (
                w @ (np.abs(I) * kphase + np.abs(K) * m_real * dphase))
            tail = self._kernel_tail_many(kabs_mat, ln.q, float(ln.t[-1]), sub)
            if alias_fn is not None:
                alias = alias_fn(ln, sub, D)
            else:
                alias = amp_left[sub] * xl / (1 - xl) + amp_right[sub] * xr / (1 - xr)
            bnds[lo:lo + sub.size] = 1.01 * alias + floor + 2.0 * m_real * tail / _TWO_PI
            flrs[lo:lo + sub.size] = floor
        return vals, bnds, flrs

    @staticmethod
    def _kernel_tail_many(kabs_mat, q, T, idx):
        """`_kernel_tail` for many parameters: an upper bound per column."""
        ts = T * np.geomspace(1.0, 1e8, 481)
        with np.errstate(all="ignore"):
            vals = np.asarray(kabs_mat(-q + 1j * ts, idx), dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        env = np.maximum.accumulate(vals[::-1], axis=0)[::-1]
        upper = np.sum(env[:-1] * np.diff(ts)[:, None], axis=0)
        with np.errstate(all="ignore"):
            a = np.log10(np.maximum(env[-61], 1e-300) / np.maximum(env[-1], 1e-300))
            closing = env[-1] * ts[-1] / (a - 1.0)
        return np.where(env[-1] == 0.0, upper,
                        np.where(a > 1.0, upper + closing, np.inf))

    def _solve_many(self, kmat, kabs_mat, sign, amp_left, amp_right,
                    rate_right, n_scale, guess, idx, extra=None, label="contour",
                    alias_fn=None, abs_floor=0.0):
        """Values and relative bounds for the parameters `idx`, designing and
        computing lines only for those that cached lines do not serve.

        `abs_floor`: a value counts as resolved once its ABSOLUTE error is
        within rel_tol * max(|value|, abs_floor) -- for a quantity only ever
        used through something like exp(H - S), where S's absolute error is
        what matters. The value still reports its true relative bound.
        """
        P = idx.size
        best_v = np.full(P, np.nan)
        best_r = np.full(P, np.inf)
        best_f = np.full(P, np.inf)          # the best line's rounding floor, relative
        best_l = [None] * P

        def rel(v, b):
            with np.errstate(all="ignore"):
                return np.where(np.abs(v) > b, b / (np.abs(v) - b), np.inf)

        def take(ln, pos):
            v, b, fl = self._integrate_many(ln, kmat, kabs_mat, sign, amp_left,
                                            amp_right, rate_right, n_scale,
                                            idx[pos], extra, alias_fn)
            r = rel(v, b)
            better = r < best_r[pos]
            best_v[pos[better]] = v[better]
            best_r[pos[better]] = r[better]
            best_f[pos[better]] = rel(v, fl)[better]
            for j in pos[better]:
                best_l[j] = ln

        allpos = np.arange(P)
        for ln in list(self._lines):
            take(ln, allpos)
        # GREEDY, ONE LINE AT A TIME. A line serves a whole band of parameters,
        # so it is designed for ONE unresolved parameter -- the smallest scale
        # left -- computed, and then every unresolved parameter is evaluated on
        # it before the next is designed. Designing every parameter first and
        # computing a line per distinct design builds many lines where one
        # would do. A parameter its own line did not serve is redesigned with
        # a stricter target, at most four times.
        tries = np.zeros(P, dtype=int)
        order = np.argsort(np.log(np.maximum(n_scale[idx], 1e-300)), kind="stable")

        def unresolved():
            with np.errstate(divide="ignore", invalid="ignore"):
                tol = self.rel_tol * np.maximum(
                    1.0, abs_floor / np.maximum(np.abs(best_v), 1e-300))
            return ~(best_r <= np.where(np.isfinite(best_v), tol, self.rel_tol))

        # WHEN TO STOP. A value can be out of any line's reach -- smaller than
        # the rounding of the sum that forms it -- and then every finer line
        # only costs time (minutes each on a foundation). So a parameter whose
        # own new line did not improve its bound tenfold is finished, and
        # after _FUTILE_LINES new lines in a row that resolved nothing the
        # batch stops. What is left is returned with the bound it has.
        #
        # And before a line is designed for a parameter: if its best line's
        # ROUNDING floor already exceeds its tolerance, a finer or longer line
        # at that q only repeats the rounding. The sum cancels less at a lower
        # q, so the batch gets ONE line at half that q -- its step scaled to
        # keep the same aliasing margin -- and after that such values keep the
        # bounds they have. (On the foundation three lines at q ~0.17 resolved
        # none of 17 deficit values that one at q = 0.12 then resolved.) One,
        # not one per value: halving q halves the step, so each further line
        # doubles in cost, and on the foundation a third or fourth would buy
        # 1e-10 over a proven 1e-7 on values drawn on a log axis.
        futile = 0
        skipped = 0
        lowered = False
        while futile < self._FUTILE_LINES:
            open_ = order[unresolved()[order] & (tries[order] < 4)]
            if open_.size == 0:
                break
            j = int(open_[0])
            i = idx[j]
            if np.isfinite(best_r[j]):
                with np.errstate(divide="ignore", invalid="ignore"):
                    tol_j = self.rel_tol * max(1.0, abs_floor / max(abs(best_v[j]), 1e-300))
                if best_f[j] > 0.5 * tol_j:
                    bl = best_l[j]
                    q_low = round(bl.q / 2.0 / 0.02) * 0.02
                    if lowered or q_low < 0.02:
                        tries[j] = 4
                        skipped += 1
                        continue
                    lowered = True
                    h_low = bl.h * q_low / bl.q   # the same q D, so the same margin
                    T = self._design_T_many(kabs_mat, q_low, idx[[j]],
                                            np.array([max(abs(best_v[j]), abs_floor)]), 1.0)
                    ln = self._line(q_low, h_low, max(T, float(bl.t[-1])))
                    cost = self._last_cost
                    pending = np.nonzero(unresolved())[0]
                    take(ln, pending)
                    if cost is not None:
                        left = int(np.count_nonzero(unresolved()[pending]))
                        _log.info("line q=%g h=%.4g: %s, %d orders in %.1fs, a lower q "
                                  "for values at their rounding floor; resolved %d of "
                                  "%d pending %s values", q_low, h_low, cost[0], cost[1],
                                  cost[2], pending.size - left, pending.size, label)
                        futile = futile + 1 if left == pending.size else 0
                    continue
            strict = 0.1 ** max(tries[j] - 1, 0)
            tries[j] += 1
            before = best_r[j]
            est = abs(best_v[j]) if np.isfinite(best_r[j]) else guess[i]
            est = max(est, abs_floor)
            q, h = self._design_qh(est, amp_left[i], amp_right[i], rate_right, strict)
            # The rounding floor, which the aliasing balance does not see: the
            # highest q whose predicted floor clears the target, or -- where
            # none does even at a hundred times the target, est being a guess
            # good to a factor of ten -- no line at all.
            tol_e = self.rel_tol * max(est, abs_floor)
            qf = self._floor_q(kabs_mat, idx[[j]], 0.1 * tol_e, q)
            if qf is None:
                q_lo = self._floor_q(kabs_mat, idx[[j]], 10.0 * tol_e, q)
                if q_lo is None:
                    tries[j] = 4
                    skipped += 1
                    continue
                qf = q_lo
            if qf < q:
                q, h = self._design_qh(est, amp_left[i], amp_right[i], rate_right,
                                       strict, q=qf)
                # That q is already as low as the floor asks for this value;
                # halving it again for what this line leaves would double the
                # cost for values deeper still (on the foundation: 137 s for 9
                # values of U beyond n = 1e77, against 70 s for 28 here).
                lowered = True
            T = self._design_T_many(kabs_mat, q, idx[[j]], np.array([est]), strict)
            ln = self._line(q, h, T)
            cost = self._last_cost
            pending = np.nonzero(unresolved())[0]
            take(ln, pending)
            if cost is not None:
                left = int(np.count_nonzero(unresolved()[pending]))
                _log.info("line q=%g h=%.4g: %s, %d orders in %.1fs; resolved "
                          "%d of %d pending %s values", q, h, cost[0], cost[1],
                          cost[2], pending.size - left, pending.size, label)
                futile = futile + 1 if left == pending.size else 0
                if unresolved()[j] and not best_r[j] < 0.1 * before:
                    tries[j] = 4
        if skipped:
            _log.info("%d %s values are below the rounding floor of any line "
                      "that could serve them; returned with the bounds they have",
                      skipped, label)
        if futile >= self._FUTILE_LINES:
            _log.info("stopped: %d new lines in a row resolved no %s value; %d "
                      "returned with the bounds they have", futile, label,
                      int(np.count_nonzero(unresolved())))
        out = []
        for j in range(P):
            ln = best_l[j]
            if ln is None:
                # no line resolved it: 0 is within 100% of any non-negative number
                out.append(BoundedValue(0.0, 1.0, f"{label} unresolved"))
                continue
            m = f"{label} q={ln.q:g} h={ln.h:.4g}"
            out.append(BoundedValue(float(best_v[j]), float(best_r[j]), m))
        return out

    def _design_qh(self, value_est, amp_left, amp_right, rate_right, strict,
                   q=None):
        """(q, h) balancing the two aliasing tails for a value of about
        `value_est`; with `q` given, the step for that q."""
        target = max(self.rel_tol * 0.05 * strict, 1e-16) * max(value_est, 1e-300)
        L = max(math.log(amp_left / target), 1.0)
        Rr = max(math.log(max(amp_right, 1e-300) / target), 1.0)
        a = rate_right
        if q is None:
            q = a * L / (L + Rr)
            q = min(max(round(q / 0.02) * 0.02, 0.02), min(a - 0.02, 0.98))
        D = 1.35 * max(L / q, Rr / (a - q))
        k = math.ceil(math.log((_TWO_PI / D) / 0.12) / math.log(0.92))
        return q, 0.12 * 0.92 ** max(k, 0)

    #: Where the predicted rounding floor is compared: the line's nodes up to
    #: t = 60, past which every kernel here has fallen by e^-90 or more.
    _FLOOR_TS = np.linspace(0.0, 60.0, 241)

    def _predicted_floor(self, kabs_mat, params, q: float) -> float:
        """The rounding floor a line at q would have for the parameters
        `params`, before it is built: `_integrate_many`'s floor with |M| at
        most M(q) -- EPS (1/pi) int |K| M(q) (dphase + kphase) dt -- the
        kernel's size taken along the line and ln M(q) interpolated on the
        planning grid. A planning figure, within a small factor of the floor
        the line then has; it never enters a bound."""
        ts = self._FLOOR_TS
        with np.errstate(all="ignore"):
            mags = np.asarray(kabs_mat(-q + 1j * ts, params), dtype=float)
        mags = np.where(np.isfinite(mags), mags, np.inf).max(axis=1)
        lnm = float(np.interp(q, self._Q_EST, self._log_mgrid()))
        phase = 2.0 * self._max_len + ts * (self.s_max + 200.0)
        return EPS / math.pi * math.exp(lnm) * float(_trapezoid(mags * phase, ts))

    def _floor_q(self, kabs_mat, params, target: float, q_hi: float):
        """The largest q at most `q_hi`, on the planner's 0.02 grid, whose
        predicted rounding floor is below `target`; None if there is none.

        WHY. A line's q is chosen to balance its two aliasing tails, and
        nothing in that choice sees the rounding floor, which for a small
        value grows steeply with q: on the foundation, U(1e82) has a floor of
        3e-6 of itself at q = 0.08 and 1e-2 at q = 0.22 -- where the balance
        put three lines that resolved nothing (99 s). Lower is better there
        only down to a point (the richness kernel has a pole at q = 0), so
        the grid is walked down from the balanced q and the first that
        clears is taken."""
        q = round(q_hi / 0.02) * 0.02
        while q >= 0.02 - 1e-12:
            if self._predicted_floor(kabs_mat, params, q) <= target:
                return round(q, 2)
            q = round(q - 0.02, 2)
        return None

    def _design_T_many(self, kabs_mat, q, params, est, strict):
        """The reach that brings every parameter's tail below its target."""
        targets = np.maximum(self.rel_tol * 0.05 * strict, 1e-16) * np.maximum(est, 1e-300)
        Ts = np.arange(8.0, 200.0 + 1e-9, 4.0)
        with np.errstate(all="ignore"):
            mags = np.asarray(kabs_mat(-q + 1j * Ts, params), dtype=float)
        mags = mags * math.exp(self._log_mreal(q))
        ok = mags < targets[None, :]
        first = np.where(ok.any(axis=0), ok.argmax(axis=0), Ts.size - 1)
        return float(Ts[int(first.max())])

    # ── batch quantities ───────────────────────────────────────

    def richness_many(self, ns) -> list[BoundedValue]:
        """`richness` at every depth in `ns`, the contour depths as one batch
        per line, and lines shared between them."""
        ns = np.array([_check_depth(n) for n in np.asarray(ns, dtype=float).ravel()])
        out: list = [None] * ns.size
        d0 = math.exp(self.log_d0)
        todo = []
        for i, n in enumerate(ns):
            if n == 0.0:
                out[i] = BoundedValue(0.0, 0.0, "exact")
            elif n * self.p_min > 800.0:
                out[i] = BoundedValue(d0, math.exp(-n * self.p_min), "saturated")
            else:
                if self._series_ok(n):
                    v, b = self._richness_series(n)
                    sv = BoundedValue(v, self._rel(v, b), "series")
                    if sv.bound <= self.rel_tol:
                        out[i] = sv
                        continue
                    out[i] = sv
                todo.append(i)
        if todo:
            idx = np.array(todo)
            w_all = ns + 1.0

            def kmat(z, sub):
                return np.exp(loggamma(z)[:, None] + _log_ratio_matrix(w_all[sub], z))

            def kabs_mat(z, sub):
                return np.abs(kmat(z, sub))

            def extra(ln, sub):
                ws = w_all[sub]
                if np.all(ws >= _TAYLOR_W):
                    return 0.0
                zz = (-ln.q + 1j * ln.t)[:, None]
                ex = gammaln(ws)[None, :] + np.abs(loggamma(ws[None, :] + zz))
                return np.where(ws[None, :] < _TAYLOR_W, ex, 0.0)

            guess = np.zeros(ns.size)
            guess[idx] = self._estimate("richness", ns[idx])
            res = self._solve_many(kmat, kabs_mat, -1.0, np.full(ns.size, d0),
                                   ns.copy(), 1.0, ns, guess, idx, extra=extra)
            for i, v in zip(todo, res):
                if out[i] is None or v.bound < out[i].bound:
                    out[i] = v
        return out

    def novelty_many(self, ns) -> list[BoundedValue]:
        """U(n) = sum_s P (1 - P)^(n-1), the chance that draw n is a sequence
        not drawn before, at every n >= 1 in `ns`.

        Its own Mellin kernel, Gamma(z+1) Gamma(n) / Gamma(n+1+z), rather than
        R(n) - R(n-1): nothing is subtracted.
        """
        ns = np.array([_check_depth(n) for n in np.asarray(ns, dtype=float).ravel()])
        if np.any(ns < 1.0):
            raise ValueError("novelty needs n >= 1")
        out: list = [None] * ns.size
        d0 = math.exp(self.log_d0)
        todo = []
        K = self._K_SERIES
        for i, n in enumerate(ns):
            if (n - 1.0) * self.p_min - self._log_m1 > 746.0:
                # U(n) <= M(1) e^{-(n-1) p_min}, below the smallest double
                out[i] = BoundedValue(0.0, 1.0, "underflow")
                continue
            if self._series_ok(n - 1.0) and n - 1.0 <= K - 1:
                # sum_k (-1)^k C(n-1, k) M(k+1)
                logc, sgn, stop = self._binom_coeffs(n - 1.0, K - 1)
                k = np.arange(K)
                alt = np.where(k % 2 == 0, 1.0, -1.0)
                with np.errstate(invalid="ignore"):
                    terms = alt * sgn * np.exp(logc + self._log_mk[:K])
                terms = np.where(np.isfinite(terms), terms, 0.0)
                v = float(np.sum(terms))
                b = 16 * EPS * float(np.sum(np.abs(terms))) + (
                    0.0 if stop is not None else 2.0 * abs(float(terms[-1])))
                sv = BoundedValue(v, self._rel(v, b), "series")
                out[i] = sv
                if sv.bound <= self.rel_tol:
                    continue
            todo.append(i)
        if todo:
            idx = np.array(todo)

            def kmat(z, sub):
                return np.exp(loggamma(z + 1.0)[:, None]
                              + _log_ratio_matrix(ns[sub], z + 1.0))

            def kabs_mat(z, sub):
                return np.abs(kmat(z, sub))

            def extra(ln, sub):
                ws = ns[sub]
                if np.all(ws >= _TAYLOR_W):
                    return 0.0
                zz = (-ln.q + 1.0 + 1j * ln.t)[:, None]
                ex = gammaln(ws)[None, :] + np.abs(loggamma(ws[None, :] + zz))
                return np.where(ws[None, :] < _TAYLOR_W, ex, 0.0)

            amp_left = d0 * np.minimum(1.0, 1.0 / np.maximum(ns, 1.0))
            guess = np.zeros(ns.size)
            guess[idx] = self._estimate("novelty", ns[idx])
            res = self._solve_many(kmat, kabs_mat, 1.0, amp_left,
                                   np.ones(ns.size), 1.0, ns, guess, idx,
                                   extra=extra, label="contour")
            for i, v in zip(todo, res):
                if out[i] is None or v.bound < out[i].bound:
                    out[i] = v
        return out

    def poisson_richness_many(self, ns) -> list[BoundedValue]:
        """sum_s 1 - exp(-N P(s)): the richness of a POISSON sample of depth
        N (each sequence's count Poisson(N P)), at every N in `ns`.

        Kernel Gamma(z) N^-z, the Mellin transform of 1 - e^-x for
        -1 < Re z < 0, on the same lines as `richness_many`; the same range
        0 <= f <= min(1, N P) bounds its aliasing. Small N: the alternating
        series sum_k (-1)^(k+1) N^k M(k) / k!, whose terms fall while
        N p_max <= 1/2, so its first omitted term bounds it.
        """
        ns = np.array([_check_depth(n) for n in np.asarray(ns, dtype=float).ravel()])
        out: list = [None] * ns.size
        d0 = math.exp(self.log_d0)
        K = self._K_SERIES
        k = np.arange(1, K + 1, dtype=np.float64)
        todo = []
        for i, n in enumerate(ns):
            if n == 0.0:
                out[i] = BoundedValue(0.0, 0.0, "exact")
            elif n * self.p_min > 800.0:
                out[i] = BoundedValue(d0, math.exp(-n * self.p_min), "saturated")
            elif n * self.p_max <= 0.5:
                terms = np.where(k % 2 == 1, 1.0, -1.0) * np.exp(
                    k * math.log(n) + self._log_mk[:K] - gammaln(k + 1))
                v = float(np.sum(terms[::-1]))
                b = 16 * EPS * float(np.sum(np.abs(terms))) + abs(float(terms[-1]))
                out[i] = BoundedValue(v, self._rel(v, b), "series")
                if out[i].bound > self.rel_tol:
                    todo.append(i)
            else:
                todo.append(i)
        if todo:
            idx = np.array(todo)
            lnn = np.log(ns)

            def kmat(z, sub):
                return np.exp(loggamma(z)[:, None] - z[:, None] * lnn[sub][None, :])

            def kabs_mat(z, sub):
                return np.abs(kmat(z, sub))

            def extra(ln, sub):
                zz = -ln.q + 1j * ln.t
                return (np.abs(loggamma(zz))[:, None]
                        + ln.t[:, None] * np.abs(np.minimum(lnn[sub], 0.0))[None, :])

            guess = np.zeros(ns.size)
            guess[idx] = self._estimate("richness", ns[idx])
            res = self._solve_many(kmat, kabs_mat, -1.0, np.full(ns.size, d0),
                                   ns * math.exp(self._log_m1), 1.0, ns,
                                   guess, idx, extra=extra)
            for i, v in zip(todo, res):
                if out[i] is None or v.bound < out[i].bound:
                    out[i] = v
        return out

    def sampling_deficit_many(self, ns) -> list[BoundedValue]:
        """S(N) = sum_s P(s) D(N P(s)), D(lam) = E[ln(1+Y)] - ln lam with
        Y ~ Poisson(lam), at every N > 0 in `ns`.

        The entropy a Poisson sample of depth N loses to sampling: the sample's
        expected Shannon entropy is H - S(N), H the model's own, so the
        sampled Hill number D1 is exp(H - S(N)). S falls from H - ln N at
        small N to D0 / (2N) at large.

        Large N: the contour int D~(w) N^-w M(1-w) dw / 2 pi i on the same
        lines as richness, w = 1 - q + it (see `_deficit_kernel`). Its
        aliasing comes from D's own range, 0 <= D(lam) <= ln(1 + 1/lam):
        copies at larger lam add at most (D0/N) e^-qD each, and copies at
        smaller lam at most (H + M(1) ln 2 + M(1) j D) e^-(1-q)jD, H the
        exact entropy.

        Small N (N p_max <= 1/2): S = H - M(1) ln N + sum_k c_k N^k M(k+1),
        with c_k = Delta^k ln(1+x)|_0 / k!, alternating and falling, so the
        first omitted term bounds the series.
        """
        ns = np.array([_check_depth(n) for n in np.asarray(ns, dtype=float).ravel()])
        if np.any(ns <= 0.0):
            raise ValueError("the sampling deficit needs N > 0")
        out: list = [None] * ns.size
        d0 = math.exp(self.log_d0)
        m1 = math.exp(self._log_m1)
        H = float(self._m.entropy())
        K = self._K_SERIES - 1
        c = _deficit_series_coeffs(K)
        k = np.arange(1, K + 1, dtype=np.float64)
        todo = []
        for i, n in enumerate(ns):
            if n * self.p_max <= 0.5:
                terms = c * np.exp(k * math.log(n) + self._log_mk[1:K + 1])
                head = H - m1 * math.log(n)
                v = head + float(np.sum(terms[::-1]))
                b = (16 * EPS * (abs(H) + m1 * abs(math.log(n)) + float(np.sum(np.abs(terms))))
                     + 2e-12 * float(np.sum(np.abs(terms)))      # the c_k, to 2e-12
                     + abs(float(terms[-1])))
                out[i] = BoundedValue(v, self._rel(v, b), "series")
                if b > self.rel_tol * max(abs(v), 1.0):
                    todo.append(i)
            else:
                todo.append(i)
        if todo:
            idx = np.array(todo)
            lnn = np.log(ns)

            def kmat(z, sub):
                w = 1.0 + z
                return _deficit_kernel(w)[:, None] * np.exp(-w[:, None] * lnn[sub][None, :])

            def kabs_mat(z, sub):
                w = 1.0 + z
                return (_deficit_kernel_abs_bound(w)[:, None]
                        * np.exp(-w.real[:, None] * lnn[sub][None, :]))

            def extra(ln, sub):
                w = 1.0 - ln.q + 1j * ln.t
                return ((np.abs(loggamma(w)) + 10.0 * (1.0 + ln.t))[:, None]
                        + ln.t[:, None] * np.abs(np.minimum(lnn[sub], 0.0))[None, :])

            def alias_fn(ln, sub, D):
                xl = math.exp(-ln.q * D)
                xr = math.exp(-(1.0 - ln.q) * D)
                return ((d0 / ns[sub]) * xl / (1 - xl)
                        + (H + m1 * _LN2) * xr / (1 - xr)
                        + m1 * D * xr / (1 - xr) ** 2)

            amp_left = d0 / ns
            amp_right = np.full(ns.size, H + m1 * (_LN2 + 60.0))
            guess = np.zeros(ns.size)
            guess[idx] = self._estimate("deficit", ns[idx])
            # S is used as exp(H - S): its ABSOLUTE error is D1's relative
            # one, so it is resolved to rel_tol of max(S, 1 nat). Deep in N,
            # where S ~ D0 / 2N falls to 1e-50 and below, a relative target
            # would be both out of reach and beside the point.
            res = self._solve_many(kmat, kabs_mat, 1.0, amp_left, amp_right, 1.0,
                                   ns, guess, idx, extra=extra,
                                   alias_fn=alias_fn, abs_floor=1.0)
            for i, v in zip(todo, res):
                if out[i] is None or v.bound < out[i].bound:
                    out[i] = v
        return out

    def private_at_depth(self, depths, *, total_depth: float) -> dict:
        """Per repertoire of a cohort of total depth T: "distinct", R(d_i), and
        "private", R(T) - R(T - d_i), the expected number of its sequences no
        other member carries -- as lists of BoundedValues, in input order.

        The private counts are one batch per line through the node-only
        coefficients of `private`; the distinct counts are `richness_many`.
        """
        d = np.asarray(depths, dtype=float).ravel()
        if d.size == 0 or not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("depths must be finite, non-negative and non-empty")
        T = _check_depth(total_depth, "total_depth")
        if np.any(d > T):
            raise ValueError("a depth exceeds the cohort total")
        uniq, inv = np.unique(d, return_inverse=True)
        distinct_u = self.richness_many(uniq)
        w = T + 1.0
        private_u: list = [None] * uniq.size
        fast = (w >= _TAYLOR_W) & (uniq <= 0.25 * w) & (uniq > 0) \
            & (not self._series_ok(T))
        for i in np.nonzero(~fast)[0]:
            private_u[i] = self.richness_difference(T, T - float(uniq[i])) \
                if uniq[i] > 0 else BoundedValue(0.0, 0.0, "exact")
        fi = np.nonzero(fast)[0]
        if fi.size:
            dd = uniq[fi]
            mmax = 1
            while mmax < 60 and (dd.max() / w) ** mmax / mmax > 1e-19:
                mmax += 1
            pg = _polygammas(w, mmax + 100)
            powers = np.stack([(-dd) ** m / math.factorial(m)
                               for m in range(1, mmax + 1)])
            d0 = math.exp(self.log_d0)
            ecache: dict = {}

            def e_coeffs(z):
                key = (z.size, float(z[0].real), float(z[-1].imag))
                if key in ecache:
                    return ecache[key]
                out = np.zeros((mmax, z.size), dtype=np.complex128)
                for m in range(1, mmax + 1):
                    acc = np.zeros(z.size, dtype=np.complex128)
                    zk = np.ones_like(z)
                    kf = 1.0
                    for k in range(1, 100):
                        zk = zk * z
                        kf *= k
                        term = pg[m - 1 + k] * zk / kf
                        acc += term
                        if np.max(np.abs(term)) <= 1e-19 * max(1e-300, float(np.max(np.abs(acc)))):
                            break
                    out[m - 1] = -acc
                ecache[key] = out
                return out

            def kmat(z, sub):
                if np.max(np.abs(z)) > 0.3 * w:
                    return kabs_mat(z, sub).astype(np.complex128)
                E = e_coeffs(z)
                base = np.exp(loggamma(z) + _log_ratio(w, z))
                return -base[:, None] * np.expm1(E.T @ powers[:, sub])

            grow = w / (w - dd)

            def kabs_mat(z, sub):
                b = np.abs(np.exp(loggamma(z) + _log_ratio(w, z)))
                return b[:, None] * (1.0 + grow[sub])[None, :]

            amp_left = d0 * np.minimum(1.0, np.maximum(dd, 1.0) / (w - dd))
            amp_right = np.maximum(dd, 1.0)
            guess = dd * min(T, d0) / w * 0.3
            res = self._solve_many(kmat, kabs_mat, -1.0, amp_left, amp_right,
                                   1.0, np.full(dd.size, T), guess,
                                   np.arange(dd.size), label="cohort")
            for k, i in enumerate(fi):
                private_u[i] = res[k]
        return {"depth": d.tolist(),
                "distinct": [distinct_u[i] for i in inv],
                "private": [private_u[i] for i in inv]}

    def equal_depth_cohort(self, depth: float, n_repertoires: float) -> dict:
        """N repertoires of one depth: {"seen", "private"}, where
        seen = R(N d) and private = N [R(N d) - R((N-1) d)]."""
        depth = _check_depth(depth, "depth")
        N = _check_depth(n_repertoires, "n_repertoires")
        seen = self.richness(N * depth)
        one = self.richness_difference(N * depth, (N - 1.0) * depth)
        private = BoundedValue(N * one.value, one.bound, one.method)
        return {"seen": seen, "private": private}
