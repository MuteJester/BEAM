"""SCALE: the BEAM anomaly index, and which residues carry the anomaly.

Three things live here, in increasing order of cost.

**The index.** :meth:`BeamModel.scale_score` already gives the one-channel
score, ``-log10 P(U >= u | L)``, from the exact per-length null. It is a
goodness-of-fit test against a DIFFUSE alternative: thresholding ``-ln P(s)``
is the likelihood-ratio test of "s is a model draw" against "s is uniform over
the support". A sequencing error is not diffuse. It is a LOCAL corruption of a
sequence that was itself a model draw, and against that alternative the total
surprisal is not the matched statistic. :class:`ScaleIndex` adds the two
channels that are, at roughly a thousand times the cost.

**The attribution.** A single left-to-right model cannot say WHERE an error is,
because everything downstream of the error is contaminated by it -- its
estimate is biased late, and no amount of care on one channel removes that
bias. BEAM scores every residue twice, from the N terminus and from the C
terminus, so the contamination runs in opposite directions on the two channels
and cancels. :func:`attribute` returns both profiles.

**The bracket.** For a sequence the model cannot spell at all, an ``-inf`` in
each anchored profile CERTIFIES a window containing the error. Intersecting the
two windows is a proof, not an estimate.

Nothing here is fitted. The reference for the index is the model's own law:
exactly by dynamic programming where a closed form exists (the ``U`` channel),
and by drawing from the model where one does not (``X`` and ``Lambda``). No
labels, no learned weights, no tuned threshold.
"""
from __future__ import annotations

import numpy as np

from .calibration import Calibration

__all__ = ["Attribution", "ScaleIndex", "attribute", "residue_profile",
           "anchor_profile", "crossing_amplitude", "neighbourhood_ratio"]

_AA = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------- profiles

def _explain_arrays(model, sequence):
    """``(forward, backward, weights)`` for one sequence, as float arrays."""
    e = model.explain(sequence)
    f = np.asarray(e["forward"], dtype=np.float64)
    b = np.asarray(e["backward"], dtype=np.float64)
    wf, wb = model.config["weights"]
    return f, b, float(wf), float(wb)


def _dipole(f, b):
    """``(f - b, all_finite)``, with undefined positions contributing nothing.

    Where the model cannot spell a residue at all, one anchor is ``-inf`` and
    the difference is undefined rather than large. Such a position carries no
    DIRECTIONAL information -- it says the error is nearby, which is what the
    certified bracket reports, not which side of it we are on -- so it
    contributes zero to the cumulative profile instead of poisoning it to NaN.
    """
    d = np.subtract(f, b, out=np.zeros_like(f),
                    where=np.isfinite(f) & np.isfinite(b))
    return d, bool(np.all(np.isfinite(f)) and np.all(np.isfinite(b)))


def residue_profile(model, sequence) -> np.ndarray:
    """Per-residue surprisal ``c_i``; one value per residue.

    ``-(wf * f_i + wb * b_i)``, the mixed log-probability of each residue
    negated so that larger means more surprising. This is the "which residue is
    wrong" profile, and it localises substitutions and insertions.

    It is deliberately RAW, not calibrated per position. Building a per
    ``(L, i)`` null and taking the least probable position instead of the most
    surprising one measurably HURTS: on the foundation it costs 0.05 of
    exact-position accuracy on substitutions, because calibrating rewards small
    excesses at intrinsically low-entropy positions. Calibrate the index; do
    not calibrate the attribution.
    """
    f, b, wf, wb = _explain_arrays(model, sequence)
    return -(wf * f + wb * b)


def anchor_profile(model, sequence) -> np.ndarray:
    """Cumulative anchor disagreement ``S_j``; one value per GAP, so ``L + 1``.

    ``S_j = sum_{i < j} (f_i - b_i)``, with ``S_0 = 0``.

    The signature of a point error is a signed DIPOLE in ``f - b``, not a peak
    in its magnitude. The residue just before an error is read by a clean
    N-anchored context but an already-contaminated C-anchored one, so ``f - b``
    rises; at and after the error the roles reverse and it falls. On the
    foundation the mean ``f - b`` around a substitution runs
    ``+1.02`` at offset -1 and ``-0.98`` at +1, against a clean baseline of
    zero everywhere.

    So the error is a ZERO-CROSSING of the disagreement, which is a peak of its
    cumulative sum -- ``argmax_j S_j`` is the estimated site. Applying an
    unsigned statistic such as ``max |f - b|`` to a signed dipole is why naive
    anchor-disagreement scores underperform.

    Indexed on gaps rather than residues because a deletion happens BETWEEN two
    residues: gap ``j`` sits before residue ``j``.
    """
    f, b, _, _ = _explain_arrays(model, sequence)
    d, _ok = _dipole(f, b)
    out = np.zeros(len(d) + 1, dtype=np.float64)
    np.cumsum(d, out=out[1:])
    return out


def crossing_amplitude(model, sequence) -> float:
    """``max_j S_j - min_j S_j``: the size of the dipole.

    A CUSUM range, which is the matched statistic for a change point. This is
    the channel that sees DELETIONS: a deletion barely moves the total
    surprisal (a 2.80 nat spike against a 2.05 background on the foundation)
    but gives the largest dipole of any error class (a 4.0 nat swing), because
    it shifts every residue after it relative to the C anchor.
    """
    f, b, _, _ = _explain_arrays(model, sequence)
    d, ok = _dipole(f, b)
    if not ok:
        # The model cannot spell part of this sequence, so the dipole is not
        # merely large, it is off the scale the null was built on.
        return float("inf")
    c = np.concatenate(([0.0], np.cumsum(d)))
    return float(c.max() - c.min())


# ----------------------------------------------------- one-edit neighbourhood

def _neighbours(sequence: str, max_length: int) -> list[str]:
    """Every string one substitution, deletion or insertion away."""
    s = sequence
    out = []
    for i in range(len(s)):
        for a in _AA:
            if a != s[i]:
                out.append(s[:i] + a + s[i + 1:])
    if len(s) > 1:
        out.extend(s[:i] + s[i + 1:] for i in range(len(s)))
    if len(s) < max_length:
        out.extend(s[:i] + a + s[i:]
                   for i in range(len(s) + 1) for a in _AA)
    return out


def neighbourhood_ratio(model, sequence) -> float:
    """``log sum_{t in N1(s)} P(t) - log P(s)``: the one-edit Bayes factor.

    The likelihood ratio for "s is a corrupted model draw" against "s is a
    model draw", with a uniform kernel over single edits. This is the channel
    that makes the alternative GENERATIVE, which is the theoretical fix to
    testing against "uniform over the support".

    Its only structural assumption is that there is one edit. The uniform
    kernel is an assumption too, and a wrong one -- real errors are not uniform
    over edit type or position -- but choosing a shaped kernel would mean
    fitting one, so it stays uniform.

    Costs about ``40L + 20`` scorings, which is what makes the full index
    roughly a thousand times more expensive than the surprisal alone.
    """
    lp = float(model.log_pseq(sequence))
    nb = _neighbours(sequence, int(model.config["max_length"]))
    if not nb:
        return float("-inf")
    v = np.asarray(model.log_pseq_batch(nb), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("-inf") if np.isfinite(lp) else float("inf")
    m = v.max()
    lse = m + np.log(np.exp(v - m).sum())
    if not np.isfinite(lp):
        # The model cannot spell s but can spell something one edit away.
        # That is the strongest possible evidence of a point error.
        return float("inf")
    return float(lse - lp)


# ------------------------------------------------------------- attribution

class Attribution:
    """Where in a sequence the anomaly sits.

    Attributes:
        sequence: The sequence scored.
        residue: Per-residue surprisal ``c_i``, length ``L``.
        gap: Cumulative anchor disagreement ``S_j``, length ``L + 1``.
        site: Best single guess at the offending residue index.
        gap_site: Best guess at the gap a residue is missing from.
        bracket: ``(lo, hi)`` inclusive residue window certified to contain a
            single error, or ``None``. For an unspellable sequence this is a
            PROOF, not an estimate: on the foundation it contains the true site
            for 100% of substitutions, with a mean width of 4.6 residues, and
            the second anchor narrows it about 40% over the first alone.
            ``None`` on an unspellable sequence means the two anchors certify
            windows that do not overlap, which proves there is MORE THAN ONE
            edit; ``site`` then reports the earliest offending residue.
        spellable: Whether the model gives the sequence finite probability.
    """

    __slots__ = ("sequence", "residue", "gap", "site", "gap_site",
                 "bracket", "spellable")

    def __init__(self, *, sequence, residue, gap, site, gap_site,
                 bracket, spellable):
        self.sequence = sequence
        self.residue = residue
        self.gap = gap
        self.site = site
        self.gap_site = gap_site
        self.bracket = bracket
        self.spellable = spellable

    def __repr__(self) -> str:
        b = "" if self.bracket is None else f", bracket={self.bracket}"
        return (f"Attribution({self.sequence!r}, site={self.site}"
                f"{b}, spellable={self.spellable})")


def attribute(model, sequence) -> Attribution:
    """Localise the anomaly in one sequence.

    Returns both profiles, because substitutions and insertions live on
    residues while deletions live on gaps, and one profile cannot carry both.

    Measured on the 927M foundation against the identifiability set -- the set
    of edit positions producing the same observed string, which has mean size
    1.00 / 1.11 / 1.22 for substitutions / insertions / deletions, because
    deleting one G from a run of Gs is genuinely ambiguous:

    ==========================  =========  =========  =========
    exact-position accuracy     sub        ins        del
    ==========================  =========  =========  =========
    chance                      0.054      0.079      0.085
    N-anchored alone            0.593      0.526      0.331
    residue profile (both)      0.648      0.589      0.213
    anchor crossing (both)      0.247      0.299      0.482
    ==========================  =========  =========  =========

    The decisive number is not the accuracy but the BIAS. Mean signed error is
    ``+0.25 / +0.44 / -0.59`` for the N anchor alone and ``-0.18 / -0.26 /
    -0.26`` for the C anchor alone, against ``-0.00 / +0.05 / -0.37`` for the
    two-channel mixture. One anchor is not merely weaker, it is systematically
    late, and there is no way to correct that from one channel.
    """
    f, b, wf, wb = _explain_arrays(model, sequence)
    c = -(wf * f + wb * b)
    d, _ok = _dipole(f, b)
    s = np.zeros(len(d) + 1, dtype=np.float64)
    np.cumsum(d, out=s[1:])

    spellable = bool(np.isfinite(model.log_pseq(sequence)))

    # An -inf in an anchored profile is a certificate. The N-anchored expert
    # first fails at index `a`, and it reads a context of `order` residues, so
    # the residue that broke it lies in [a - order, a]. The C-anchored expert
    # gives the mirror window. Both hold simultaneously, so intersect them.
    order = int(model.config["order"])
    bad_f = np.flatnonzero(~np.isfinite(f))
    bad_b = np.flatnonzero(~np.isfinite(b))
    lo, hi = 0, len(c) - 1
    certified = False
    if bad_f.size:
        lo = max(lo, int(bad_f[0]) - order)
        hi = min(hi, int(bad_f[0]))
        certified = True
    if bad_b.size:
        lo = max(lo, int(bad_b[-1]))
        hi = min(hi, int(bad_b[-1]) + order)
        certified = True
    bracket = (lo, hi) if certified and lo <= hi else None

    if bracket is not None:
        # Inside a certified window, pick the most surprising residue THERE.
        w = c[bracket[0]:bracket[1] + 1]
        site = bracket[0] + int(np.argmax(np.nan_to_num(w, nan=-np.inf)))
    elif certified:
        # The two certificates are INCONSISTENT, which is itself informative:
        # no single edit explains this sequence, so there is more than one.
        # The first position that broke the N-anchored expert is the earliest
        # offending residue, which is the most defensible single answer.
        site = int(bad_f[0]) if bad_f.size else int(bad_b[-1])
    elif np.any(~np.isfinite(c)):
        # Unspellable with both anchored profiles finite cannot happen through
        # a residue, so this is the length prior. No residue is at fault.
        site = int(np.argmax(np.nan_to_num(c, nan=-np.inf)))
    else:
        site = int(np.argmax(c))
    gap_site = int(np.argmax(s))

    return Attribution(sequence=sequence, residue=c, gap=s, site=site,
                       gap_site=gap_site, bracket=bracket,
                       spellable=spellable)


# ------------------------------------------------------------- the index

def _empirical_sf(null: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``P(N >= v)`` from a sorted null, floored so no p-value is zero.

    The ``+1`` in numerator and denominator is the standard Monte-Carlo
    correction: with ``n`` draws the smallest defensible tail probability is
    ``1 / (n + 1)``, not zero, and a zero would make the Fisher statistic
    infinite on a single channel.
    """
    n = null.size
    ge = n - np.searchsorted(null, values, side="left")
    return (ge + 1.0) / (n + 1.0)


class ScaleIndex:
    """The three-channel SCALE index, calibrated entirely off the model.

    Combines three views of one sequence, each turned into a p-value against
    the model's own null at that length, then fused by Fisher's method and
    referred to the null of the fused statistic:

    ======  ===================================  ==========================
    channel quantity                             null
    ======  ===================================  ==========================
    ``U``   ``-ln P(s)``, the total surprisal    exact, by dynamic programming
    ``X``   crossing amplitude of ``f - b``      model draws at that length
    ``L``   one-edit neighbourhood ratio         model draws at that length
    ======  ===================================  ==========================

    The three were chosen structurally -- one global view, one positional
    two-channel view, one neighbourhood view -- not selected against labels.

    Measured on the 927M foundation, 30,000 clean corpus sequences against one
    uniform single-residue error each, unspellables included and scored
    ``+inf`` rather than dropped:

    ==================  ===============  ===============  ===============
    TPR at 1% FPR       sub              ins              del
    ==================  ===============  ===============  ===============
    one channel (U)     0.244            0.289            0.149
    three channels      0.344            0.467            0.284
    ==================  ===============  ===============  ===============

    The AUC gain is small (about +0.02); the gain is concentrated in the
    low-false-positive corner, which is the only place a screen is ever
    operated. Fitting a supervised classifier on the same three p-values buys a
    further +0.004 AUC, so the no-fitting constraint costs almost nothing.

    Two honest caveats. Combining with ``max`` of ranks instead of Fisher is
    WORSE than the best single channel, because ``max`` has power only when one
    component is extreme; Fisher sums evidence. And Fisher assumes independent
    p-values, which these are not -- referring the fused statistic to its own
    model null keeps the false-positive rate exact, but the combination is
    therefore not the most powerful one available.

    Unlike :meth:`BeamModel.calibrate`, this reintroduces an ``n_sim`` and a
    seed, because no public entry point reconstructs the ``X`` and ``Lambda``
    nulls in closed form. On the ``U`` channel, where both routes exist, 20,000
    draws reproduce the exact null's 99th percentile to within 0.41 nats on
    values of 22 to 62, and the resulting AUC to within 0.001.
    """

    __slots__ = ("calibration", "n_sim", "seed", "_x", "_lam", "_t")

    def __init__(self, *, calibration, n_sim, seed, x, lam, t):
        self.calibration = calibration
        self.n_sim = int(n_sim)
        self.seed = seed
        self._x = x
        self._lam = lam
        self._t = t

    def __repr__(self) -> str:
        return (f"ScaleIndex(lengths={len(self._t)}, "
                f"n_sim={self.n_sim}, seed={self.seed})")

    @property
    def lengths(self) -> tuple[int, ...]:
        """Lengths this index covers."""
        return tuple(sorted(self._t))

    @classmethod
    def build(cls, model, *, calibration: Calibration | None = None,
              n_sim: int = 20000, seed: int = 0, lengths=None) -> "ScaleIndex":
        """Construct the nulls by drawing from the model.

        Args:
            model: The model to calibrate against, and to score with.
            calibration: From :meth:`BeamModel.calibrate`; built if omitted.
            n_sim: Draws per length. 20,000 is where the ``U`` channel's
                simulated null matches its exact one to within 0.001 AUC.
            seed: Base seed; each length is drawn with ``seed + length`` so
                adding a length does not perturb the others.
            lengths: Lengths to cover, default every length the calibration
                covers.
        """
        cal = calibration if calibration is not None else model.calibrate()
        want = sorted(cal.lengths) if lengths is None else sorted(set(lengths))
        x, lam, t = {}, {}, {}
        for L in want:
            draws = model.sample(n_sim, length=L, seed=seed + L)
            if not draws:
                continue
            raw_x = np.array([crossing_amplitude(model, s) for s in draws],
                             dtype=np.float64)
            raw_l = np.array([neighbourhood_ratio(model, s) for s in draws],
                             dtype=np.float64)
            xs, ls = np.sort(raw_x), np.sort(raw_l)
            x[L], lam[L] = xs, ls
            # The fused null must be built from the SAME draws, scored the same
            # way, or its quantiles do not describe the statistic being tested.
            u = -np.asarray(model.log_pseq_batch(draws), dtype=np.float64)
            pu = np.array([cal.pvalue(L, float(v)) for v in u])
            pu = np.maximum(pu, cal.resolution_floor)
            ts = -2.0 * (np.log(pu)
                         + np.log(_empirical_sf(xs, raw_x))
                         + np.log(_empirical_sf(ls, raw_l)))
            t[L] = np.sort(ts)
        return cls(calibration=cal, n_sim=n_sim, seed=seed,
                   x=x, lam=lam, t=t)

    def channels(self, model, sequences) -> dict:
        """The three per-channel p-values, before fusing.

        Returns a dict of arrays keyed ``"U"``, ``"X"`` and ``"L"``. Useful for
        seeing WHICH view flagged a sequence: a deletion typically moves ``X``
        while leaving ``U`` near the middle of its null.
        """
        single = isinstance(sequences, str)
        seqs = [sequences] if single else list(sequences)
        u = -np.asarray(model.log_pseq_batch(seqs), dtype=np.float64)
        pu = np.empty(len(seqs))
        px = np.empty(len(seqs))
        pl = np.empty(len(seqs))
        floor = self.calibration.resolution_floor
        for i, s in enumerate(seqs):
            L = len(s)
            pu[i] = max(self.calibration.pvalue(L, float(u[i])), floor)
            xs, ls = self._x.get(L), self._lam.get(L)
            if xs is None:
                # An uncovered length has no null, so it contributes no
                # evidence. That is the conservative reading, and it matches
                # Calibration returning 1.0 for a length it does not cover.
                px[i] = pl[i] = 1.0
                continue
            px[i] = _empirical_sf(xs, np.array([crossing_amplitude(model, s)]))[0]
            pl[i] = _empirical_sf(ls, np.array([neighbourhood_ratio(model, s)]))[0]
        out = {"U": pu, "X": px, "L": pl}
        if single:
            return {k: float(v[0]) for k, v in out.items()}
        return out

    def score(self, model, sequences):
        """The index: ``-log10 p`` of the fused statistic. Higher is worse.

        A threshold of 2 is a 1% false-positive rate at every covered length,
        the same reading as :meth:`BeamModel.scale_score`, so the two are
        directly comparable at a fixed operating point.
        """
        single = isinstance(sequences, str)
        seqs = [sequences] if single else list(sequences)
        # `seqs` is a list, so channels always returns arrays here.
        ch = self.channels(model, seqs)
        t = -2.0 * (np.log(ch["U"]) + np.log(ch["X"]) + np.log(ch["L"]))
        out = np.empty(len(seqs))
        for i, s in enumerate(seqs):
            null = self._t.get(len(s))
            if null is None:
                out[i] = 0.0
                continue
            p = _empirical_sf(null, np.array([t[i]]))[0]
            out[i] = -np.log10(p)
        # An unspellable sequence keeps an infinite score, as it does on the
        # one-channel index: zero probability is not a small p-value.
        lp = np.asarray(model.log_pseq_batch(seqs), dtype=np.float64)
        out = np.where(np.isneginf(lp), np.inf, out)
        return float(out[0]) if single else out
