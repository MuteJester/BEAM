"""The saddlepoint inversion of the surprisal distribution.

Lugannani-Rice, with the tilt solved by a safeguarded Newton iteration, plus
the normal density and distribution it is written in terms of.

These are the implementations behind `BeamModel.saddlepoint` and its private
tilt solver. They take the model as their first argument rather than `self`;
the methods remain on `BeamModel`, carry the public documentation, and
delegate here. Nothing in this module imports `BeamModel` at runtime.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .._simulation_result import SimulationResult
from ..model.validation import _is_scalar_arg

if TYPE_CHECKING:
    from ..model.facade import BeamModel


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def solve_tilt(model: BeamModel, x, cache, *, tol=1e-9, max_iter=80):
    """Behind :meth:`BeamModel._solve_tilt`."""
    def jet_at(t):
        key = round(t, 12)
        if key not in cache:
            cache[key] = model.cgf_jet(key)
        return cache[key]

    lo_t, hi_t = -model.MAX_TILT, model.MAX_TILT
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


def saddlepoint(model: BeamModel, x, *, tol: float = 1e-9,
                max_iter: int = 80) -> dict:
    """Behind :meth:`BeamModel.saddlepoint`."""
    xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
    lo, hi = model.surprisal_range()
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
        t, jet, converged = model._solve_tilt(xv, cache, tol=tol,
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
            k3 = float(model.surprisal_cumulants(q=1.0, order=3)[2])
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
