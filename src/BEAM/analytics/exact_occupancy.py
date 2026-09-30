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
with the bound it has, never silently.

What this does NOT replace: counts of sequences in a RANGE of probability --
the spectrum's histograms. Those are threshold counts; see
`docs/api/python/analytics.md`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, loggamma, polygamma

__all__ = ["BoundedValue", "ExactOccupancy"]

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


def _log_ratio(w: float, z: np.ndarray) -> np.ndarray:
    """ln Gamma(w) - ln Gamma(w + z), real w >= 1, complex z.

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
            out[~far] = _log_ratio(w, z[~far])
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
        self.log_d0 = float(model.log_mellin(0.0))
        lo, hi = model.surprisal_range()
        self.p_max = math.exp(-lo)
        self.p_min = math.exp(-hi)
        self.s_max = float(hi)
        self._max_len = float(model.config["max_length"])
        ks = np.arange(1, self._K_SERIES + 1, dtype=float)
        self._log_mk = np.asarray(model.log_mellin_many(ks), dtype=float)

    # ── bookkeeping ─────────────────────────────────────────────

    @property
    def lines(self) -> list[tuple[float, float, int]]:
        """The contour lines computed so far: (Re q, step h, number of orders)."""
        return [(ln.q, ln.h, ln.t.size) for ln in self._lines]

    def _log_mreal(self, q: float) -> float:
        if q not in self._real:
            self._real[q] = float(self._m.log_mellin(q))
        return self._real[q]

    def _line(self, q: float, h: float, T: float) -> _Line:
        """A line with this q, step no coarser than h and reach at least T.

        A cached line with the same q and step that falls short of T is
        EXTENDED -- only the missing nodes are computed and appended -- rather
        than recomputed from the start.
        """
        for ln in self._lines:
            if ln.q == q and ln.h <= h * (1 + 1e-12) and ln.t[-1] >= T:
                return ln
        for ln in self._lines:
            if ln.q == q and ln.h == h:
                k0 = ln.t.size
                n = int(math.ceil(T / h)) + 1
                more = self._m.log_mellin_line(q, -h * k0, -h, n - k0)
                ln.log_m = np.concatenate([ln.log_m, np.asarray(more)])
                ln.t = h * np.arange(n)
                return ln
        n = int(math.ceil(T / h)) + 1
        log_m = self._m.log_mellin_line(q, 0.0, -h, n)
        ln = _Line(q=q, h=h, t=h * np.arange(n), log_m=np.asarray(log_m))
        self._lines.append(ln)
        return ln

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
        m_real = math.exp(self._log_mreal(ln.q))       # sum of the moduli
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

    def richness_many(self, ns) -> list[BoundedValue]:
        """`richness` at every depth in `ns`, sharing lines between them."""
        return [self.richness(n) for n in ns]

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

    def equal_depth_cohort(self, depth: float, n_repertoires: float) -> dict:
        """N repertoires of one depth: {"seen", "private"}, where
        seen = R(N d) and private = N [R(N d) - R((N-1) d)]."""
        depth = _check_depth(depth, "depth")
        N = _check_depth(n_repertoires, "n_repertoires")
        seen = self.richness(N * depth)
        one = self.richness_difference(N * depth, (N - 1.0) * depth)
        private = BoundedValue(N * one.value, one.bound, one.method)
        return {"seen": seen, "private": private}
