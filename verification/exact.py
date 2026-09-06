#!/usr/bin/env python3
"""Check the library's numbers against brute force, not against a recording.

`check.py` is a REPRODUCIBILITY gate: it pins 78 quantities as hex floats and
fails on a one-bit change. It cannot fail on a quantity that has been wrong
since it was written, because it compares each one against a recording of
ITSELF.

That is not hypothetical. `beam_model_positional_marginals` returned rows
summing to 5.9e-264 instead of 1, at `beam_model_config_default()`, in shipped
code, with all eight checks green. It was found by building a second
implementation and comparing. Two more followed: the tilted sweeps overflowed
at any negative `q`, and a zero anchor weight filled the edge weights with
NaNs. All three lived inside quantities the baseline pins.

So this file pins CORRECTNESS. The method is one idea: build a model small
enough that its whole support can be enumerated, score every sequence in it
ONCE, and then check every exported functional of that distribution against a
direct sum over the resulting vector. The enumeration is the cost and it
amortises across some thirty quantities.

Nothing here reads a baseline file. The right-hand side of every comparison is
arithmetic over `log_pseq`, which is itself checked by normalisation and by the
independent reference implementation in `tests/reference/`.
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np

import BEAM

AA = "ACDEFGHIKLMNPQRSTVWY"
NT = "ACGTN"

#: Scalars are compared relatively; probabilities absolutely, since a
#: probability near zero has no useful relative scale. Worst observed on a
#: correct build is ~1e-14, so these leave three orders of headroom.
REL = 1e-9
ABS = 1e-13

#: `BEAM_EXACT_VERBOSE=1` prints how much of each tolerance was actually
#: consumed, which is how the tolerances below were chosen rather than guessed.
VERBOSE = bool(os.environ.get("BEAM_EXACT_VERBOSE"))


def _deviation(got, want, *, abs_=None):
    """Worst deviation, in the units the caller is judging by.

    Returns `inf` for a shape mismatch or a non-finite difference, so that a
    NaN cannot slip through as a small number: `nan <= tol` is False, but so
    is `nan > tol`, and a check that silently passes on NaN is the failure
    mode this whole file exists to prevent.
    """
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if got.shape != want.shape:
        return np.inf
    d = np.abs(got - want)
    if abs_ is None:
        d = d / np.maximum(np.abs(want), 1.0)
    d = d[np.isfinite(d)] if d.size else d
    worst = float(d.max()) if d.size else 0.0
    # A difference that is not finite (inf, or NaN from inf - inf) is a
    # failure, not a missing sample.
    if not np.all(np.isfinite(np.abs(np.asarray(got) - np.asarray(want)))):
        return np.inf
    return worst


class Oracle:
    """One model, its whole support, and the exact law over it.

    `p[k]` is `P(s_k)` — the JOINT probability, length prior included, which is
    what `log_pseq` returns. `cond[L]` is the conditional `P(. | L)` restricted
    to the sequences of that length. Keeping both is what stops the commonest
    mistake in a check like this: the calibration axis is the joint surprisal
    while its weights are conditional on length, and confusing them produces a
    resolution-independent error that looks exactly like a defect.
    """

    def __init__(self, model, alphabet: str, lengths):
        self.m = model
        self.alphabet = alphabet
        self.lengths = list(lengths)
        seqs, idx = [], []
        for L in self.lengths:
            for combo in itertools.product(alphabet, repeat=L):
                seqs.append("".join(combo))
                idx.append(L)
        self.seqs = seqs
        self.len_of = np.asarray(idx)
        self.logp = np.asarray(model.log_pseq_batch(seqs), dtype=float)
        self.p = np.exp(self.logp)
        # Membership of the support is a statement about the LOG probability.
        # In the open world with backoff off a sequence can sit at -2760 nats,
        # which is finite and real and underflows `exp` to exactly zero; using
        # `p > 0` there counts a smaller support than the model has, and then
        # disagrees with `path_count` and `log_mellin(0)` about the size of it.
        # Writing this reference produced three separate versions of that
        # mistake before it found anything about the library.
        self.live = np.isfinite(self.logp)
        self.total = float(self.p.sum())

    def at(self, L):
        """`(sequences, conditional probabilities)` for one length."""
        sel = (self.len_of == L) & self.live
        p = self.p[sel]
        return [s for s, k in zip(self.seqs, sel) if k], p / p.sum()

    def entropy(self):
        """`-sum p ln p`, over the terms that carry any mass.

        The mask is not the support: a sequence whose log probability is -2760
        is IN the support and its `p` underflows to zero, and `0 * ln 0` is
        `0 * -inf`, which is NaN rather than the zero the limit requires. That
        is the same trap as the `weight_forward = 0` defect in the library, and
        it caught this reference too.
        """
        p = self.p[self.live]
        p = p[p > 0.0]
        return float(-(p * np.log(p)).sum())

    def log_mellin(self, q):
        v = q * self.logp[self.live]
        mx = v.max()
        return float(mx + np.log(np.exp(v - mx).sum()))

    def tilted_weights(self, q):
        """Normalised weights of the law tilted to `P^q`, in log space."""
        v = q * self.logp[self.live]
        w = np.exp(v - v.max())
        return w / w.sum()

    def moments(self, q, order):
        """Raw moments of the surprisal under the q-tilted law."""
        s = -self.logp[self.live]
        w = self.tilted_weights(q)
        return np.array([float((w * s ** k).sum()) for k in range(order + 1)])


def tier1(report):
    """The amino-acid oracle: six models, thirty quantities each."""
    rng = np.random.default_rng(20260906)
    corpus = sorted({"".join(rng.choice(list(AA), size=int(rng.integers(3, 5))))
                     for _ in range(400)})
    ab = [1.0 + (i % 13) for i in range(len(corpus))]

    # Six ordinary configurations, and one with backoff switched off.
    #
    # That last one is not decoration. In the open world with zero backoff the
    # unobserved cells sit at `ln(1e-300)`, so a layer's weights span 690 nats
    # -- and that spread is what the tilted sweeps got wrong. With default
    # backoff the spread is a few nats and `q * wmax` is close enough to the
    # true tilted maximum that the defect hides. A grid of well-conditioned
    # models watches this class of bug walk past.
    configs = [(o, s, None) for o in (2, 3, 4) for s in (False, True)]
    configs.append((3, False, 0.0))

    def one(order, strict, kap):
        kw = {} if kap is None else dict(kappa=(kap, kap, kap),
                                         kappa_wide=kap, kappa_len=kap)
        m = BEAM.BeamModel(corpus, abundances=ab, order=order,
                           max_length=4, depth_cap=2, strict=strict, **kw)
        o = Oracle(m, AA, (3, 4))
        tag = (f"aa order {order} {'strict' if strict else 'open'}"
               + ("" if kap is None else " no-backoff"))

        report(f"{tag}: normalisation", o.total, 1.0)
        report(f"{tag}: entropy", m.entropy(), o.entropy())
        report(f"{tag}: path_count", m.path_count(),
               float(o.live.sum()))
        # Negative tilts included deliberately. All three tilted sweeps
        # overflowed at any `q < 0` in shipped code, and a grid of
        # non-negative q would have watched it happen.
        for q in (-2.0, -0.5, 0.0, 0.5, 2.0):
            report(f"{tag}: log_mellin q={q}", m.log_mellin(q),
                   o.log_mellin(q))
        report(f"{tag}: hill 2", m.hill_number(2.0),
               math.exp(o.log_mellin(2.0) / (1.0 - 2.0)))
        lo, hi = m.surprisal_range()
        s = -o.logp[o.live]
        report(f"{tag}: surprisal_range", (lo, hi), (s.min(), s.max()))
        # Negative q reaches `beam_tilted_layer_max`, whose q < 0 branch takes
        # the layer MINIMUM: under a negative tilt the largest tilted weight
        # comes from the smallest raw one. Reverting that to `q * wmax` leaves
        # this 55% wrong at q = -8 on the no-backoff model, or refusing
        # outright -- and a grid stopping at q = 2 watched it pass.
        for q in (-8.0, -2.0, -0.5, 1.0, 2.0):
            report(f"{tag}: surprisal_moments q={q}",
                   m.surprisal_moments(q=q, order=4), o.moments(q, 4))
        # t > 1 is a negative Mellin q. t = 5 is here because the same
        # rescale defect leaves t <= 3 correct on this model while t = 5
        # raises: the boundary is |q| * (weight spread) against 709, and a
        # grid has to CROSS it rather than approach it.
        for t in (-0.3, 0.4, 1.6, 3.0, 5.0):
            j = m.cgf_jet(t)
            # `K(t) = ln E[exp(tY)]` with `Y = -ln P`, which is
            # `log_mellin(1 - t)` -- the facade says so, and taking the
            # definition from the docstring rather than from memory is the
            # difference between checking the library and checking a
            # guess. The first two derivatives are the mean and variance
            # of `Y` under the law tilted to `P^(1-t)`.
            # In log space, like everything else here. On the
            # no-backoff model `(1 - t) * logp` reaches 1400, and
            # exponentiating it directly gives inf / inf = nan -- a
            # reference that overflows before the library does proves
            # nothing about the library.
            w = o.tilted_weights(1.0 - t)
            k1 = float((w * s).sum())
            report(f"{tag}: cgf_jet K t={t}", j["K"], o.log_mellin(1.0 - t))
            report(f"{tag}: cgf_jet K1 t={t}", j["K1"], k1)
            # A tilted VARIANCE at |t| > 1 is conditioning-limited: on
            # the no-backoff model the law concentrates on a handful of
            # paths spanning 2760 nats, and the jet and a centred sum
            # over the support agree to about 3e-6 rather than to 1e-9.
            # The six well-conditioned models agree at the default
            # tolerance, so the slack is specific to the case that needs
            # it rather than a blanket loosening.
            report(f"{tag}: cgf_jet K2 t={t}", j["K2"],
                   float((w * (s - k1) ** 2).sum()),
                   rel=1e-9 if abs(t) <= 1.0 else 1e-6)

        # Ties are genuine in a strict world, so compare the multiset of
        # scores rather than the order, which the tie-break may permute.
        got = sorted(v for _, v in m.top_k_sequences(20))
        want = sorted(o.logp[o.live])[-20:]
        report(f"{tag}: top_k scores", got, want)

        for L in (3, 4):
            seqs, cp = o.at(L)
            marg = np.zeros((L, len(AA)))
            for sq, pr in zip(seqs, cp):
                for i, r in enumerate(sq):
                    marg[i, AA.index(r)] += pr
            report(f"{tag}: positional_marginals L={L}",
                   m.positional_marginals(L), marg, abs_=ABS)

            # A clamp the model can satisfy, chosen from the marginals.
            pos = 1
            res = AA[int(np.argmax(marg[pos]))]
            keep = np.array([sq[pos] == res for sq in seqs])
            mass = float(cp[keep].sum())
            report(f"{tag}: log_clamp_probability L={L}",
                   math.exp(m.log_clamp_probability(L, {pos: res})),
                   mass, abs_=ABS)
            cm = np.zeros((L, len(AA)))
            for sq, pr in zip(np.array(seqs)[keep], cp[keep] / mass):
                for i, r in enumerate(sq):
                    cm[i, AA.index(r)] += pr
            report(f"{tag}: clamped marginals L={L}",
                   m.positional_marginals(L, clamp={pos: res}), cm,
                   abs_=ABS)

    # Per-CONFIGURATION, so that a model which refuses is attributed to
    # itself and the remaining configurations still run. A single raise
    # aborting the tier would hide 279 other quantities behind the first
    # one, which is the difference between a gate that localises a defect
    # and one that only says something is wrong.
    for order, strict, kap in configs:
        try:
            one(order, strict, kap)
        except Exception as exc:                  # noqa: BLE001
            tag = (f"aa order {order} "
                   f"{'strict' if strict else 'open'}"
                   + ("" if kap is None else " no-backoff"))
            report(f"{tag}: raised {type(exc).__name__}: {exc}",
                   float("nan"), 0.0)


def tier2(report):
    """The nucleotide query oracle: the whole query surface, 5**5 support."""
    rng = np.random.default_rng(7)
    corpus = sorted({"".join(rng.choice(list("ACGT"), size=int(rng.integers(4, 6))))
                     for _ in range(300)})
    m = BEAM.BeamModel(corpus, alphabet="nt", order=2, max_length=5,
                       strict=False)
    o = Oracle(m, NT, (5,))
    q = BEAM.BeamQuery(m)
    seqs, cp = o.at(5)
    R = len(NT)

    cases = [
        ("pin", q.condition(5).at(0, "A"), lambda s: s[0] == "A"),
        ("motif", q.condition(5).contains("CG"), lambda s: "CG" in s),
        ("count", q.condition(5).count("A", 2, 3),
         lambda s: 2 <= s.count("A") <= 3),
    ]
    for name, cond, keep in cases:
        sel = np.array([keep(s) for s in seqs])
        mass = float(cp[sel].sum())
        report(f"nt query {name}: probability", q.probability(cond), mass,
               abs_=ABS)
        sub = cp[sel] / mass
        sq = np.array(seqs)[sel]

        marg = np.zeros((5, R))
        for s, pr in zip(sq, sub):
            for i, r in enumerate(s):
                marg[i, NT.index(r)] += pr
        report(f"nt query {name}: marginals", q.marginals(cond), marg, abs_=ABS)
        report(f"nt query {name}: entropy", q.entropy(cond),
               float(-(sub * np.log(sub)).sum()))
        h = np.array([-sum(x * math.log(x) for x in marg[i] if x > 0)
                      for i in range(5)])
        report(f"nt query {name}: position_entropy",
               q.position_entropy(cond), h, abs_=ABS)

        joint = np.zeros((R, R))
        for s, pr in zip(sq, sub):
            joint[NT.index(s[1]), NT.index(s[3])] += pr
        report(f"nt query {name}: pair_joint", q.pair_joint(cond, 1, 3), joint,
               abs_=ABS)

        mi = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                jj = np.zeros((R, R))
                for s, pr in zip(sq, sub):
                    jj[NT.index(s[i]), NT.index(s[j])] += pr
                hij = -sum(x * math.log(x) for x in jj.ravel() if x > 0)
                mi[i, j] = h[i] + h[j] - hij
        report(f"nt query {name}: mutual_information",
               q.mutual_information(cond), np.clip(mi, 0.0, None), abs_=ABS)

        tab = np.zeros((5, R))
        tab[:, NT.index("A")] = 1.0
        cnt = np.array([s.count("A") for s in sq], dtype=float)
        mean = float((sub * cnt).sum())
        report(f"nt query {name}: expect_count",
               q.expect_count("A", cond),
               (mean, float((sub * (cnt - mean) ** 2).sum())))


def run() -> list[str]:
    failures: list[str] = []
    n = 0
    #: Worst deviation seen, as a FRACTION of the tolerance it was judged
    #: against. Printing it turns the tolerances from guesses into a measured
    #: margin: if this ever approaches 1 the budget is being consumed by
    #: something real, and shrinking it is the wrong reaction.
    worst, worst_name = 0.0, "-"
    margins: list = []

    def report(name, got, want, *, abs_=None, rel=REL):
        nonlocal n, worst, worst_name
        n += 1
        dev = _deviation(got, want, abs_=abs_)
        tol = abs_ if abs_ is not None else rel
        if np.isfinite(dev) and dev / tol > worst:
            worst, worst_name = dev / tol, name
        if VERBOSE and np.isfinite(dev):
            margins.append((dev / tol, dev, tol, name))
        if dev > tol:
            failures.append(name)
            g = np.asarray(got, dtype=float).ravel()
            w = np.asarray(want, dtype=float).ravel()
            k = int(np.argmax(np.abs(g - w))) if g.shape == w.shape else 0
            print(f"  FAIL  exact: {name}")
            print(f"          got {g[k]!r} want {w[k]!r}"
                  f" (worst of {g.size})")

    # A tier that RAISES has failed, and must be reported as a failure rather
    # than escaping as a traceback. Reverting the log-space forward sweep
    # makes `positional_marginals` refuse rather than return wrong numbers --
    # correct behaviour for the library, but a gate that dies on it reports
    # nothing and an automated caller cannot tell a refusal from a crash in
    # the gate itself.
    for label, tier in (("tier1", tier1), ("tier2", tier2)):
        try:
            tier(report)
        except Exception as exc:                      # noqa: BLE001
            failures.append(f"{label} raised")
            print(f"  FAIL  exact: {label} raised "
                  f"{type(exc).__name__}: {exc}")
    if VERBOSE:
        print("  margins (deviation as a fraction of its tolerance):")
        for frac, dev, tol, name in sorted(margins, reverse=True)[:14]:
            print(f"    {frac:9.2%}  dev {dev:9.2e}  tol {tol:8.1e}  {name}")
    if not failures:
        print(f"  OK    exact against enumeration ({n} quantities; worst "
              f"{worst:.1%} of its tolerance, at {worst_name})")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
