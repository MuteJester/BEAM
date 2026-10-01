"""P(independent categorical units land in given category totals).

Every summary piece of the repertoire likelihood has this one shape. Each unit
-- a sequence, or a bin of the spectrum holding a real number of sequences --
falls into exactly one of C categories, independently of every other unit:

    how many unique        present / absent, one unit per spectrum bin
    how many shared        in neither / only the first / only the second / both
    the abundance pattern  which count bin a seen sequence's reads fall in

and the piece is the probability of the observed category totals.

HOW. For ANY tilt t, exactly,

    P(X = x) = exp(K(t) - t.x) * P_t(X = x),
    K(t) = sum_u m_u log sum_j q_uj e^{t_j}          (t_ref = 0)

where P_t is the same units with their probabilities tilted by e^{t_j}. The
tilt is chosen by Newton so that grad K(t) = x: under it the observed totals
are the MEAN, and P_t(X = x) is the chance a sum lands on its own mean. K and
its derivatives are exact. For a category whose count is well spread under
the tilt, that chance is the normal density at the mean,
``(2 pi)^(-d/2) det(K'')^(-1/2)`` -- the saddlepoint. Its error is of
Stirling's size, one term per category the normal term covers: about
``sum_j 1/(12 v_j)`` nats, with ``v_j`` that category's variance UNDER THE
TILT. That is the exact leading term for Poisson-like counts, where
``v_j = x_j``; it is an estimate rather than a proven ceiling, and it is
returned with every answer so the caller never has to take it on trust.

NEARLY FIXED CATEGORIES. When the observed total can be reached only by a few
units that are unlikely to go there -- a 150-read clone in a bin the model
gives 1e-100 -- the tilt must be pushed so far that under it the category's
count is almost certain. A normal density at a variance of 1e-15 is 1e7, not
a probability, and the plain saddlepoint returns nonsense there. So every
category whose tilted variance is below four -- where the normal term's error
would pass a fiftieth of a nat -- is taken out of the normal term
and its ``P_t`` computed EXACTLY by convolution: a series for units that
rarely reach it, a direct convolution for the few that often do. The rest keep
the normal term, conditioned on the exact ones through the Schur complement.

WHAT IS EXACT. Categories that force the units are conditioned on before any
of this: an empty one (``P(nothing lands in j)`` is the product of
``1 - q_uj``) and a full one (every unit that can reach it is in it). An
impossible total is ``-inf``. When a single category is left, or nothing is
left for the normal term, the result is flagged exact.

WHY log1p. A spectrum bin can hold 1e20 sequences each seen with probability
1e-20. Its unit contributes ``m * log(1 + 1e-20 * ...)``, and a plain
log-sum-exp rounds the inner sum to 1 and loses the whole contribution. Every
row reduction below keeps its largest term aside and log1p's the rest.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

__all__ = ["categorical_sum_log_pmf"]

_GRAD_TOL = 1e-10
_MAX_NEWTON = 400
_MAX_STEP = 20.0         # nats of tilt one Newton step may move
_EXACT_BELOW = 4.0       # tilted variance under which a category is convolved
_LIGHT = 0.5             # r_D / r_rest up to here: series; above: convolution


def _lse_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise log-sum-exp that keeps a 1e-20 next to a 1."""
    if a.shape[1] == 0:
        return np.full(a.shape[0], -np.inf)
    rows = np.arange(a.shape[0])
    idx = np.argmax(a, axis=1)
    top = a[rows, idx]
    with np.errstate(invalid="ignore"):
        e = np.exp(a - top[:, None])
    e[rows, idx] = 0.0
    e[~np.isfinite(top)] = 0.0
    return top + np.log1p(e.sum(axis=1))


def _lse(v: np.ndarray) -> float:
    top = float(np.max(v)) if v.size else -math.inf
    if not math.isfinite(top):
        return top
    return top + math.log(float(np.sum(np.exp(v - top))))


# ── Exact reductions ───────────────────────────────────────────

def _reduce(log_q, mult, x, rest):
    """Condition exactly on every category whose total forces the units.

    ``rest`` is the index of the category whose total is implied -- whatever
    the others leave -- or None. Its entry in ``x`` is never read. No count
    here is ever formed by subtracting from the number of units: with 1e24
    units, ``1e24 - 12`` is ``1e24`` and the 12 is gone before any test sees
    it. Units that can and cannot reach a category are both summed directly.

    Returns ``(log_q, mult, x, rest, log_factor)``, or None if the totals are
    impossible. Rows of the returned ``log_q`` are renormalised over the
    categories that remain.
    """
    log_factor = 0.0
    x = x.copy()
    while x.size:
        capable = np.isfinite(log_q)
        cap = (mult[:, None] * capable).sum(axis=0)
        noncap = (mult[:, None] * ~capable).sum(axis=0)
        # slack: units able to reach j that are not in it
        slack = cap - x
        tol = 1e-9 * np.maximum(1.0, x)
        empty = x <= 0
        if rest is not None:
            placed = float(np.delete(x, rest).sum())
            slack[rest] = placed - noncap[rest]
            tol[rest] = 1e-9 * max(1.0, placed)
            total = float(mult.sum())
            empty[rest] = placed >= total - 1e-9 * max(1.0, total)
        if np.any(slack < -tol):
            return None
        if x.size == 1:
            if rest is None and np.any(slack > tol):
                return None     # units left over with nowhere to go
            break
        if np.any(empty):
            lq = log_q[:, ~empty]
            stay = _lse_rows(lq)
            if not np.all(np.isfinite(stay)):
                return None
            log_factor += float(np.sum(mult * stay))
            if rest is not None:
                rest = None if empty[rest] else rest - int(empty[:rest].sum())
            log_q, x = lq - stay[:, None], x[~empty]
            continue
        full = np.flatnonzero(slack <= tol)
        if full.size:
            j = int(full[0])
            pin = capable[:, j]
            log_factor += float(np.sum(mult[pin] * log_q[pin, j]))
            keep = np.arange(x.size) != j
            if rest is not None:
                rest = None if j == rest else rest - int(j < rest)
            log_q, mult, x = log_q[~pin][:, keep], mult[~pin], x[keep]
            continue
        break
    return log_q, mult, x, rest, log_factor


# ── The tilted sum ─────────────────────────────────────────────

def _tilt(lq, mult, others, x_o, t):
    """phi = K(t) - t.x, its gradient and Hessian, and the tilted log-probs."""
    a = lq.copy()
    a[:, others] += t
    lse = _lse_rows(a)
    log_r = a - lse[:, None]
    r = np.exp(log_r[:, others])
    mr = mult[:, None] * r
    grad = mr.sum(axis=0) - x_o
    hess = -(mr.T @ r)
    # The diagonal is r (1 - r) with 1 - r summed from the OTHER categories:
    # a unit all but certain to be in j then contributes its true tiny
    # variance instead of the difference of two rounded ones.
    for k, j in enumerate(others):
        rest = _lse_rows(np.delete(log_r, j, axis=1))
        hess[k, k] = float(np.sum(mr[:, k] * np.exp(rest)))
    phi = float(np.sum(mult * lse)) - float(t @ x_o)
    return phi, grad, hess, log_r


def _solve_tilt(lq, mult, others, x_o):
    t = np.zeros(others.size)
    phi, grad, hess, log_r = _tilt(lq, mult, others, x_o, t)
    for _ in range(_MAX_NEWTON):
        if np.all(np.abs(grad) <= _GRAD_TOL * np.maximum(1.0, x_o)):
            return phi, hess, log_r
        try:
            step = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, -grad, rcond=None)[0]
        big = float(np.max(np.abs(step)))
        if big > _MAX_STEP:
            step *= _MAX_STEP / big
        slope = float(grad @ step)
        s = 1.0
        cand = _tilt(lq, mult, others, x_o, t + step)
        # Backtrack only while the predicted decrease is above the rounding of
        # phi. Past that the Armijo test compares noise with noise and rejects
        # every step, while the full Newton step is exactly right.
        if -slope > 1e-12 * max(1.0, abs(phi)):
            while cand[0] > phi + 1e-4 * s * slope and s > 1e-12:
                s *= 0.5
                cand = _tilt(lq, mult, others, x_o, t + s * step)
        t = t + s * step
        phi, grad, hess, log_r = cand
    raise RuntimeError("the saddlepoint equation did not converge in "
                       f"{_MAX_NEWTON} Newton steps; gradient {grad}")


# ── Exact P_t for the nearly fixed categories ──────────────────

def _exact_tilted_log_pmf(log_r, mult, cols, target) -> float:
    """``log P_t(X_cols = target)``, the tilted counts of a few categories,
    by exact convolution.

    Each unit lands in one of ``cols`` or in "elsewhere" (probability a).
    Units that rarely reach ``cols`` (``c = r_cols / a <= 1/2``) enter as the
    power series of ``m log(1 + sum_j c_j z_j)``, which is exact and takes a
    real multiplicity; the few that often do are convolved one copy at a time,
    which needs an integer multiplicity.
    """
    shape = tuple(int(v) + 1 for v in target)
    log_a = _lse_rows(np.delete(log_r, cols, axis=1))
    log_c = log_r[:, cols] - log_a[:, None]
    light = _lse_rows(log_c) <= math.log(_LIGHT)
    heavy = ~light

    # light units: E = exp(L), L_alpha = (-1)^{|a|+1} (|a|-1)!/prod a_j! P_alpha
    lm = np.log(mult[light])
    lc = log_c[light]
    idx = list(itertools.product(*(range(s) for s in shape)))
    L = np.zeros(shape)
    for alpha in idx:
        deg = sum(alpha)
        if deg == 0:
            continue
        with np.errstate(invalid="ignore"):
            terms = lm + (lc * np.asarray(alpha)).sum(axis=1,
                                                      where=np.asarray(alpha) > 0)
        coef = math.exp(math.lgamma(deg) - sum(math.lgamma(v + 1) for v in alpha))
        L[alpha] = (-1) ** (deg + 1) * coef * math.exp(_lse(terms))
    E = np.zeros(shape)
    E[(0,) * len(shape)] = 1.0
    for alpha in sorted(idx, key=sum):
        deg = sum(alpha)
        if deg == 0:
            continue
        acc = 0.0
        for beta in itertools.product(*(range(v + 1) for v in alpha)):
            db = sum(beta)
            if db:
                acc += db * L[beta] * E[tuple(p - q for p, q in zip(alpha, beta))]
        E[alpha] = acc / deg
    log_light = float(np.sum(mult[light] * log_a[light]))

    # heavy units: positive convolution, rescaled as it goes
    m_heavy = mult[heavy]
    if np.any(np.abs(m_heavy - np.round(m_heavy)) > 1e-9):
        raise RuntimeError(
            "a nearly fixed category is reached by spectrum bins holding a "
            "fractional number of sequences, which cannot be convolved "
            "exactly; ask for more bins")
    H = np.zeros(shape)
    H[(0,) * len(shape)] = 1.0
    log_scale = 0.0
    a_h = np.exp(log_a[heavy])
    r_h = np.exp(log_r[heavy][:, cols])
    for u in range(a_h.size):
        for _ in range(int(round(m_heavy[u]))):
            new = a_h[u] * H
            for k in range(len(shape)):
                src = [slice(None)] * len(shape)
                dst = [slice(None)] * len(shape)
                src[k], dst[k] = slice(0, shape[k] - 1), slice(1, shape[k])
                new[tuple(dst)] += r_h[u, k] * H[tuple(src)]
            top = float(new.max())
            if top <= 0:
                return -math.inf
            H = new / top
            log_scale += math.log(top)

    want = tuple(int(v) for v in target)
    total = 0.0
    for beta in itertools.product(*(range(v + 1) for v in want)):
        total += E[beta] * H[tuple(w - b for w, b in zip(want, beta))]
    if not total > 0:
        raise RuntimeError("the exact convolution of a nearly fixed category "
                           f"came out non-positive ({total!r})")
    return log_light + log_scale + math.log(total)


# ── Public ──────────────────────────────────────────────────────

def categorical_sum_log_pmf(log_q, mult, counts, *,
                            rest: int | None = None) -> tuple[float, bool]:
    """``log P(category totals == counts)`` and whether that value is exact.

    Args:
        log_q: ``(U, C)`` natural-log category probabilities per unit. Each row
            is a distribution; ``-inf`` marks a category the unit cannot reach.
        mult: ``(U,)`` multiplicity of each unit, a non-negative real. A unit
            with multiplicity two counts exactly as two copies of it.
        counts: ``(C,)`` observed totals. Must sum to ``sum(mult)``: every
            unit lands somewhere.
        rest: Index of a category whose total is implied -- all the units the
            others leave -- and whose entry in ``counts`` is ignored. Pass it
            whenever that total is ``sum(mult)`` minus something small: with
            1e24 units the subtraction rounds the small part away.

    Returns:
        ``(log_probability, exact, error)``. ``exact`` is True for an
        impossible total and whenever no normal term was needed. ``error`` is
        the leading-order size of the approximation, in nats:
        ``sum_j 1/(12 v_j)`` over the categories the normal term covers, with
        ``v_j`` their variance under the tilt. Zero when exact.
    """
    log_q = np.asarray(log_q, dtype=float)
    mult = np.asarray(mult, dtype=float)
    x = np.asarray(counts, dtype=float)
    if log_q.ndim != 2 or mult.shape != (log_q.shape[0],) \
            or x.shape != (log_q.shape[1],):
        raise ValueError("log_q must be (U, C), mult (U,) and counts (C,); "
                         f"got {log_q.shape}, {mult.shape} and {x.shape}")
    total = float(mult.sum())
    if rest is None:
        if abs(float(x.sum()) - total) > 1e-9 * max(1.0, total):
            raise ValueError(f"counts sum to {float(x.sum())!r} but the units "
                             f"number {total!r}: every unit must land somewhere")
    else:
        rest = int(rest)
        if not 0 <= rest < x.size:
            raise ValueError(f"rest={rest} is not a category index")
        placed = float(np.delete(x, rest).sum())
        if placed > total + 1e-9 * max(1.0, total):
            raise ValueError(f"the explicit counts sum to {placed!r}, more than "
                             f"the {total!r} units there are")
    keep = mult > 0
    reduced = _reduce(log_q[keep], mult[keep], x, rest)
    if reduced is None:
        return -math.inf, True, 0.0
    lq, mult, x, rest, log_factor = reduced
    if x.size <= 1:
        return log_factor, True, 0.0

    ref = int(np.argmax(x)) if rest is None else rest
    others = np.array([j for j in range(x.size) if j != ref])
    x_o = x[others]
    phi, hess, log_r = _solve_tilt(lq, mult, others, x_o)

    fixed = np.diag(hess) < _EXACT_BELOW
    spread = ~fixed
    log_pt = 0.0
    if np.any(fixed):
        log_pt += _exact_tilted_log_pmf(log_r, mult, others[fixed], x_o[fixed])
    d = int(spread.sum())
    err = 0.0
    if d:
        S = hess[np.ix_(spread, spread)]
        if np.any(fixed):
            B = hess[np.ix_(spread, fixed)]
            S = S - B @ np.linalg.pinv(hess[np.ix_(fixed, fixed)]) @ B.T
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise RuntimeError("the saddlepoint Hessian is not positive definite")
        log_pt -= 0.5 * (d * math.log(2.0 * math.pi) + logdet)
        err = float(np.sum(1.0 / (12.0 * np.diag(S))))
    return log_factor + phi + log_pt, d == 0, err
