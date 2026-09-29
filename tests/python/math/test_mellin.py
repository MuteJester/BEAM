"""The Mellin transform, diversity, and Mellin jets.
"""
from __future__ import annotations

import importlib.util
import math
import random

import numpy as np
import pytest

from BEAM import BeamModel

from _helpers import AA, J_MOTIFS, make_corpus

# Mellin transform and diversity
# ═══════════════════════════════════════════════════════════════

def test_log_mellin_at_zero_is_the_support(model):
    """q=0 counts every supported sequence once, so it must reproduce the
    path-count DP, which is a completely different code path."""
    assert model.log_mellin(0.0) == pytest.approx(math.log(model.path_count()),
                                                  rel=1e-9)


def test_log_mellin_at_one_is_zero(model):
    """q=1 is the total probability, which is 1 by construction. This is the
    cheapest check that the tilted sweep and the normalisation agree."""
    assert model.log_mellin(1.0) == pytest.approx(0.0, abs=1e-9)


def test_log_mellin_matches_brute_force():
    """Against an exhaustive sum over every sequence the model can produce.

    That is every string over the whole alphabet up to max_length, so the
    space is nucleotide and short (19,530 strings); q = 0 then counts them
    all, and q = 1 checks that they carry all the mass."""
    import itertools
    rng = random.Random(5)
    seqs = ["TG" + "".join(rng.choice("ACGT") for _ in range(rng.randint(0, 3)))
            + "C" for _ in range(400)]
    abund = [rng.randint(1, 40) for _ in seqs]
    m = BeamModel(seqs, abundances=abund, alphabet="nt", order=3, max_length=6)
    every = ["".join(t) for L in range(1, 7)
             for t in itertools.product("ACGTN", repeat=L)]
    lp = np.asarray(m.log_pseq_batch(every), dtype=np.float64)
    assert np.isfinite(lp).all(), "the prior leaves no string impossible"
    for q in (0.0, 0.5, 1.0, 2.0):
        total = float(np.exp(q * lp).sum())
        assert m.log_mellin(q) == pytest.approx(math.log(total), rel=1e-9,
                                                abs=1e-12), q


def test_hill_number_matches_mellin(model):
    """D_alpha = (sum P^alpha)^(1/(1-alpha)), computed two ways."""
    for a in (0.5, 2.0, 3.0):
        want = math.exp(model.log_mellin(a) / (1.0 - a))
        assert model.hill_number(a) == pytest.approx(want, rel=1e-9), a


def test_diversity_profile_is_ordered(model):
    d = model.diversity_profile()
    assert d["D0"] >= d["D1"] >= d["D2"] > 0
    assert model.effective_diversity() == pytest.approx(d["D1"], rel=1e-12)


def test_hill_curve_shape(model):
    c = model.hill_curve(orders=[0.0, 1.0, 2.0])
    assert list(c["orders"]) == [0.0, 1.0, 2.0]
    assert len(c["hill"]) == 3
    assert np.all(np.diff(c["hill"]) <= 1e-9 * c["hill"][0])


def test_power_sum_matches_log_mellin(model):
    assert model.power_sum(2.0) == pytest.approx(math.exp(model.log_mellin(2.0)),
                                                 rel=1e-12)


# ═══════════════════════════════════════════════════════════════
# Mellin jets, pooled and per length
# ════════════════════════════════════════════════════════════════════════


def test_derivatives_recover_the_mass_and_the_entropy(model):
    a = model.pseq_analysis()
    d = a.derivatives(1.0, 2)
    assert d[0] == pytest.approx(1.0, rel=1e-12)
    assert -d[1] == pytest.approx(model.entropy(), rel=1e-12)
    # Second derivative gives the surprisal's second raw moment.
    stats = model.surprisal_stats(q=1.0)
    assert d[2] == pytest.approx(stats["variance"] + stats["mean"] ** 2,
                                 rel=1e-10)


def test_derivatives_at_q_zero_is_the_support(model):
    a = model.pseq_analysis()
    assert a.derivatives(0.0, 0)[0] == pytest.approx(model.path_count(),
                                                     rel=1e-9)


def test_mellin_matches_log_mellin(model):
    a = model.pseq_analysis()
    assert math.log(a.mellin(2.0)) == pytest.approx(a.log_mellin(2.0),
                                                    rel=1e-12)
    assert a.cgf(0.0) == pytest.approx(0.0, abs=1e-12)


def test_moments_mean_is_the_entropy(model):
    m = model.pseq_analysis().moments(4)
    assert m["mass"] == pytest.approx(1.0, rel=1e-12)
    assert m["mean"] == pytest.approx(model.entropy(), rel=1e-12)
    assert m["std"] == pytest.approx(math.sqrt(m["variance"]), rel=1e-12)


def test_length_derivatives_q0_is_the_per_length_support(model):
    a = model.pseq_analysis()
    per = a.length_derivatives(0.0, 0)
    for L, jet in per.items():
        assert float(jet[0]) == pytest.approx(model.path_count(L), rel=1e-9)
    assert sum(float(j[0]) for j in per.values()) == pytest.approx(
        model.path_count(), rel=1e-9)


def test_length_derivatives_q1_decomposes_mass_and_entropy(model):
    """Summed over lengths, order 0 must give 1 and order 1 must give minus
    the entropy: the per-length jet has to be a decomposition of the global
    one, not merely per-length numbers that look plausible."""
    per = model.pseq_analysis().length_derivatives(1.0, 2)
    assert sum(float(j[0]) for j in per.values()) == pytest.approx(1.0,
                                                                  rel=1e-12)
    assert -sum(float(j[1]) for j in per.values()) == pytest.approx(
        model.entropy(), rel=1e-9)
    stats = model.surprisal_stats(q=1.0)
    assert sum(float(j[2]) for j in per.values()) == pytest.approx(
        stats["variance"] + stats["mean"] ** 2, rel=1e-9)


@pytest.mark.parametrize("q,order", [(0.0, 0), (0.5, 3), (1.0, 4), (2.0, 2)])
def test_length_derivatives_sum_to_the_pooled_mellin(model, q, order):
    """The per-length jet is now exact at any (q, order), so summing order 0
    over lengths must reproduce M(q) — a different code path."""
    per = model.pseq_analysis().length_derivatives(q, order)
    assert sum(float(j[0]) for j in per.values()) == pytest.approx(
        math.exp(model.log_mellin(q)), rel=1e-9)


def test_length_marginals_are_keyed_by_length(model):
    marg = model.pseq_analysis().length_marginals()
    assert set(marg) == set(model.observed_lengths)
    assert sum(v["generated"] for v in marg.values()) == pytest.approx(1.0,
                                                                      rel=1e-12)
    assert sum(v["counting"] for v in marg.values()) == pytest.approx(
        model.path_count(), rel=1e-9)


def test_edge_thresholding_is_not_applicable(model):
    """BeamModel deliberately has no pruning: hierarchical backoff leaves every
    transition strictly positive, so there is no low-likelihood wiring to
    threshold away. Refusing loudly is the honest answer."""
    with pytest.raises(NotImplementedError, match="no edge pruning"):
        model.pseq_analysis().diversity_under_edge_thresholds([0.0, 0.5])


# ════════════════════════════════════════════════════════════════════════
