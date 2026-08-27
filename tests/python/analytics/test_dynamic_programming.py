"""Internal consistency of the DAG.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# ── Internal consistency of the DAG ──────────────────────────────

def test_partition_normalises_by_enumeration():
    """The sharpest check available: sum P(s|L) over EVERY sequence of a short
    length must be exactly 1. If the partition function or any context packing
    is wrong, this cannot come out right by accident."""
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)
    import itertools
    for length in (3, 4):
        total = 0.0
        logpi = m.log_length_prior(length)
        for t in itertools.product(AA, repeat=length):
            lp = m.log_pseq("".join(t))
            if math.isfinite(lp):
                total += math.exp(lp - logpi)
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"L={length}: sum P(s|L) = {total!r}")


def test_length_prior_normalises(model):
    assert model.length_pmf().sum() == pytest.approx(1.0, abs=1e-12)


def test_hill_number_one_is_exp_entropy(model):
    """`hill_number(1)` is a special case that CALLS the entropy DP, so this
    identity is true by construction and cannot fail. It is kept only as a
    guard against the special case being rewired to something else; the checks
    with teeth are the two below."""
    assert model.hill_number(1.0) == pytest.approx(math.exp(model.entropy()),
                                                   rel=1e-9)


def test_hill_number_one_matches_the_limit_from_either_side(model):
    """D1 must be the alpha -> 1 limit of the GENERAL Mellin formula, which is
    a different code path from the entropy DP the alpha == 1 branch shortcuts
    to. If the entropy DP and the tilted sweep disagreed, this would catch it
    and `test_hill_number_one_is_exp_entropy` would not."""
    h = model.entropy()
    eps = 1e-3
    up = model.log_mellin(1.0 + eps) / (1.0 - (1.0 + eps))
    down = model.log_mellin(1.0 - eps) / (1.0 - (1.0 - eps))
    # ln D_(1+e) = H - e*Var/2 + O(e^2), so the two straddle H and their mean
    # cancels the linear term. Both facts are checked: the bracket catches a
    # sign error, the mean catches a scale error.
    assert min(up, down) < h < max(up, down), (up, h, down)
    # rel 1e-5, not tighter: the O(e^2) term of the expansion is ~1e-6 here.
    assert 0.5 * (up + down) == pytest.approx(h, rel=1e-5)
    assert math.log(model.hill_number(1.0)) == pytest.approx(h, rel=1e-12)


def test_entropy_matches_brute_force():
    """The entropy DP against an exhaustive sum over every short sequence.

    `test_hill_number_one_is_exp_entropy` compares the entropy with itself;
    this compares it with `-sum P ln P` computed one sequence at a time."""
    import itertools
    seqs, abund = make_corpus(n=400, seed=5, lo=3, hi=4)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=6)
    h = 0.0
    for length in (3, 4):
        for t in itertools.product(AA, repeat=length):
            lp = m.log_pseq("".join(t))
            if math.isfinite(lp):
                h -= math.exp(lp) * lp
    assert m.entropy() == pytest.approx(h, rel=1e-9)


def test_hill_numbers_are_ordered(model):
    """Hill numbers are non-increasing in alpha, always."""
    d = [model.hill_number(a) for a in (0.0, 0.5, 1.0, 1.5, 2.0)]
    for a, b in zip(d, d[1:]):
        assert b <= a * (1 + 1e-9), d


def test_support_is_all_of_sequence_space():
    """Documented, not desired, and the reason `path_count` carries a warning.

    Hierarchical backoff leaves no cell at zero, so nothing is dead and the
    support is every peptide of every producible length. That is a property of
    the smoother, not of the grammar, and the number must never be quoted as a
    diversity."""
    seqs, abund = make_corpus(n=300, seed=9, lo=5, hi=6)
    m = BeamModel(seqs, abundances=abund, order=2, max_length=8, strict=False)
    assert m.path_count(5) == pytest.approx(20.0 ** 5)
    assert m.path_count(6) == pytest.approx(20.0 ** 6)



def test_spectrum_totals(corpus):
    """The counting measure integrates to the support size and the generated
    measure to 1. Both are checked against a different code path."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)

    counting = m.spectrum(measure="counting", bins=512)
    generated = m.spectrum(measure="generated", bins=512,
                           spacing=counting["spacing"])

    assert counting["total"] == pytest.approx(m.path_count(), rel=1e-9)
    assert generated["total"] == pytest.approx(1.0, rel=1e-9)


def test_spectrum_identity(corpus):
    """`sum over bins of multiplicity * mean probability` must be 1.

    This is the check that catches per-bin error, which totals alone cannot
    see: a spectrum can integrate to the right value while being wrong bin by
    bin. The per-bin probability is `generated / counting`, the mass-weighted
    mean, NOT `exp(-surprisal)` at the bin edge, which carries a systematic
    bias of `exp(spacing)` across the bin.
    """
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    c = m.spectrum(measure="counting", bins=512)
    g = m.spectrum(measure="generated", bins=512, spacing=c["spacing"])
    keep = (c["weights"] > 0) & (g["weights"] > 0)
    p = g["weights"][keep] / c["weights"][keep]
    assert float((c["weights"][keep] * p).sum()) == pytest.approx(1.0, rel=1e-9)
    assert p.max() <= 1.0 + 1e-12


def test_spectrum_mean_is_exact_but_the_shape_is_not(corpus):
    """The documented guarantee is precise, and it is not "every bin is right".

    Linear splitting happens at EVERY layer, not once at the end, so a
    sequence's mass diffuses over several bins rather than the two adjacent to
    its exact surprisal. What survives that exactly is the total and the FIRST
    moment; the second is inflated. Both halves are pinned here, because a
    future change that made the spectrum exact bin-by-bin would be an
    improvement, and one that broke the mean would not be caught by
    `test_spectrum_totals`."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    lm = m.length_marginals()
    length = int(np.argmax(lm["generated"]))

    s = m.spectrum(measure="generated", length=length, bins=512)
    x, w = s["surprisal"], s["weights"]
    total = w.sum()
    mean_grid = float((x * w).sum() / total)
    var_grid = float((x ** 2 * w).sum() / total) - mean_grid ** 2

    assert mean_grid == pytest.approx(lm["mean_surprisal"][length], rel=1e-12)
    # Each split spreads a point across two adjacent bins, adding at most
    # `spacing**2 / 6` of variance, and there are `length + 1` of them (one per
    # layer plus the sink). The inflation is therefore bounded by a quantity
    # the test can compute rather than by a percentage picked to fit: it must
    # be positive, and it must fit inside that budget.
    inflation = var_grid - lm["variance"][length]
    budget = (length + 1) * s["spacing"] ** 2 / 6.0
    assert 0.0 < inflation <= budget, (inflation, budget)


def test_spectrum_per_length_sums_to_pooled(corpus):
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    pooled = m.spectrum(measure="generated", bins=256)
    acc = np.zeros(256)
    for length in range(1, 28):
        if not math.isfinite(m.log_length_prior(length)):
            continue
        acc += m.spectrum(measure="generated", length=length, bins=256,
                          spacing=pooled["spacing"])["weights"]
    assert np.abs(acc - pooled["weights"]).max() < 1e-12


def test_accumulator_max_bounds_the_grid(corpus):
    """The grid must be sized by the accumulator, not by the surprisal: the DP
    accumulates monotonically and only applies a per-length constant at the
    sink. A grid sized to the surprisal silently drops mass off the top when
    that constant is negative."""
    seqs, abund = corpus
    m = BeamModel(seqs, abundances=abund, order=2)
    hi = m.accumulator_max()
    assert hi > 0
    s = m.spectrum(measure="counting", bins=512)
    occupied = np.flatnonzero(s["weights"] > 0)
    assert occupied.size > 0
    # Linear splitting can push a sliver one bin past the exact value per
    # layer, so the occupied range may run up to max_length+1 bins beyond the
    # bound. What must NOT happen is mass reaching the top, which would mean
    # it fell off the end.
    slack = (int(m.config["max_length"]) + 1) * s["spacing"]
    assert s["surprisal"][occupied[-1]] <= hi + slack
    assert s["weights"][-8:].sum() == 0.0, "mass reached the top of the grid"
