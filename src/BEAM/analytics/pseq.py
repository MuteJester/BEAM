"""PseqAnalysis: spectra and diversity analytics over one model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only
    from ..model.facade import BeamModel

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .histogram import Histogram
from .atoms import Atoms
from ..constants import BEAM_MAX_MOMENT
from ..model.validation import _check_int, _is_scalar_arg

from .._simulation_result import SimulationResult


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
        """Both global spectra on one grid, split by the same fractions.

        Bit-identical to the two single-measure calls and about as long; it
        holds half their combined peak memory at once. The shared grid is the
        load-bearing part: see
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

        The hierarchical prior leaves every table cell strictly positive, so an
        edge sweep would describe a cut the model does not make.
        :meth:`BeamModel.spectrum` and :meth:`atoms` answer where the mass and
        sequences sit.
        """
        raise NotImplementedError(
            "BeamModel has no edge pruning: every sequence up to max_length is "
            "possible under its prior. See the docstring for what to use "
            "instead.")
