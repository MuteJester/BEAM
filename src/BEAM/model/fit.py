"""Fitting the hierarchical prior's strengths by held-out likelihood.

The prior has one strength per context level and one for the length
distribution. Each is how many pseudo-observations the shorter context is
worth, and no argument fixes that number: it is whatever predicts reads the
model has not seen best. So it is fitted, not chosen.

Held-out reads come from one of two places, and the choice is a question
about what the model is for:

``held_out=None``: the repertoire's own reads, split by binomial thinning.
    Build on a ``fraction`` of every sequence's reads, score the rest. This
    fits the strengths for predicting MORE READS OF THE SAME REPERTOIRE.
    Thinning splits a clone's reads between the halves, so both halves share
    its sequence and the fit leans toward trusting the data.

``held_out=(sequences, counts)``: another repertoire.
    Build on all of this one, score that one. This fits the strengths for
    predicting A DIFFERENT REPERTOIRE, which shares far fewer sequences and
    wants a stronger prior.

Without held-out reads the model already chooses its own strengths, by the
evidence of its counts (see :class:`BeamModel`). This is for a caller who wants
them chosen for a specific prediction instead, and it starts from those.

The search is coordinate-wise golden-section in ``log alpha``: one strength at
a time, the others held, two sweeps. Each evaluation is one re-finalize of one
model -- the counts never change -- plus one scoring pass.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["AlphaFit", "fit_alpha"]

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True, eq=False)
class AlphaFit:
    """Strengths fitted by held-out likelihood. NOT exact: an estimate.

    Attributes:
        alpha: One strength per level 0..order, to pass as ``alpha=``.
        alpha_length: The length prior's, to pass as ``alpha_length=``.
        held_out_log_lik: Mean log-probability per held-out read at the fit.
        default_log_lik: The same at the starting strengths -- the ones the
            model estimated from its own counts unless ``alpha=`` was passed --
            for comparison.
        split: ``"thinning"`` (the repertoire's own reads) or ``"external"``
            (another repertoire), which decides what the fit is FOR; see the
            module docstring.
        fraction: Share of reads built on, under thinning.
        seed: The thinning seed, so the split can be repeated exactly.
        n_evaluations: Re-finalizes the search spent.
    """
    alpha: tuple
    alpha_length: float
    held_out_log_lik: float
    default_log_lik: float
    split: str
    fraction: float | None
    seed: int | None
    n_evaluations: int

    def __repr__(self) -> str:
        a = ", ".join(f"{v:.3g}" for v in self.alpha)
        return (f"AlphaFit(alpha=({a}), alpha_length={self.alpha_length:.3g}; "
                f"held-out {self.held_out_log_lik:.4f} nats/read against "
                f"{self.default_log_lik:.4f} at the start; {self.split} split; "
                f"an estimate, not exact)")


def _merge(sequences: Sequence[str], counts) -> tuple[list, np.ndarray]:
    merged: dict[str, float] = {}
    for s, c in zip(sequences, counts):
        merged[s] = merged.get(s, 0.0) + float(c)
    return list(merged), np.fromiter(merged.values(), dtype=float,
                                     count=len(merged))


def fit_alpha(sequences: Iterable[str], abundances: Iterable[float] | None = None,
              *, held_out: tuple | None = None, fraction: float = 0.5,
              seed: int = 0, bounds: tuple[float, float] = (1e-3, 1e12),
              sweeps: int = 2, tol: float = 0.05, **config) -> AlphaFit:
    """Fit ``alpha`` and ``alpha_length`` to a corpus by held-out likelihood.

    Args:
        sequences, abundances: The corpus, as for :class:`BeamModel`.
            Abundances must be integers when thinning: they are read counts
            and a read goes to one half or the other.
        held_out: ``(sequences, counts)`` of another repertoire to predict,
            or None to thin this one's reads.
        fraction: Share of each sequence's reads built on, under thinning.
        seed: Thinning seed.
        bounds: The search range of every strength.
        sweeps: Passes over the strengths.
        tol: Golden-section stopping width, in ``log alpha``.
        **config: Model configuration (``order``, ``max_length``, ...),
            passed to :class:`BeamModel` unchanged.

    Returns:
        :class:`AlphaFit`. Build the final model with
        ``BeamModel(sequences, abundances=..., alpha=fit.alpha,
        alpha_length=fit.alpha_length, **config)``.
    """
    from .facade import BeamModel

    seqs = list(sequences)
    counts = (np.ones(len(seqs)) if abundances is None
              else np.asarray(list(abundances), dtype=float))
    seqs, counts = _merge(seqs, counts)

    if held_out is None:
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"fraction={fraction!r} must lie strictly "
                             f"between 0 and 1")
        if np.any(counts != np.floor(counts)) or np.any(counts < 1):
            raise ValueError("thinning needs positive integer read counts; "
                             "pass held_out= to fit against another repertoire")
        rng = np.random.default_rng(seed)
        a = rng.binomial(counts.astype(np.int64), fraction).astype(float)
        b = counts - a
        train = [(s, c) for s, c in zip(seqs, a) if c > 0]
        test_seqs = [s for s, c in zip(seqs, b) if c > 0]
        test_counts = b[b > 0]
        split, frac, used_seed = "thinning", float(fraction), int(seed)
    else:
        train = list(zip(seqs, counts))
        test_seqs, test_counts = _merge(*held_out)
        split, frac, used_seed = "external", None, None
    if not train or not test_seqs:
        raise ValueError("the split left one side empty; there is nothing "
                         "to build on or nothing to predict")

    model = BeamModel([s for s, _ in train],
                      abundances=[c for _, c in train], **config)
    cfg = model.config
    theta = [math.log(v) for v in cfg["alpha"]] + [math.log(cfg["alpha_length"])]
    n_total = float(test_counts.sum())
    evaluations = 0

    def score(th) -> float:
        nonlocal evaluations
        evaluations += 1
        model.set_alpha([math.exp(v) for v in th[:-1]], math.exp(th[-1]))
        lp = model.log_pseq_batch(test_seqs)
        return float(np.sum(test_counts * lp)) / n_total

    start = score(theta)
    best = start
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    for _ in range(sweeps):
        for i in range(len(theta)):
            def f(v, i=i):
                th = list(theta); th[i] = v
                return score(th)
            a_, b_ = lo, hi
            c_ = b_ - _GOLDEN * (b_ - a_)
            d_ = a_ + _GOLDEN * (b_ - a_)
            fc, fd = f(c_), f(d_)
            while b_ - a_ > tol:
                if fc > fd:
                    b_, d_, fd = d_, c_, fc
                    c_ = b_ - _GOLDEN * (b_ - a_)
                    fc = f(c_)
                else:
                    a_, c_, fc = c_, d_, fd
                    d_ = a_ + _GOLDEN * (b_ - a_)
                    fd = f(d_)
            cand = (a_ + b_) / 2.0
            fcand = f(cand)
            if fcand >= best:
                theta[i], best = cand, fcand
        # a strength pinned at a bound is reported as found there, not moved

    final = score(theta)
    return AlphaFit(alpha=tuple(math.exp(v) for v in theta[:-1]),
                    alpha_length=math.exp(theta[-1]),
                    held_out_log_lik=final, default_log_lik=start,
                    split=split, fraction=frac, seed=used_seed,
                    n_evaluations=evaluations)
