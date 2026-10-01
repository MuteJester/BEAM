"""The pieces at a FIXED number of reads, against exact convolution.

A repertoire is N reads. "How surprising is n distinct sequences" means
P(n | N), not P(n) in a world whose read total is itself random, and the two
differ most where most reads are new -- which is where repertoires are: at
N = 30 on the tiny model the spread of n is 1.5, against 4.8 with the total
left free. Every piece is therefore conditioned on N, through joint
probabilities of each summary WITH the read total; the four multiply to the
multinomial exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from _brute import (SLACK, exact_present_and_reads, exact_profile_and_reads,
                    log_multinomial)
from BEAM import repertoire_likelihood
from BEAM.likelihood.fixed_total import (counts_given_set_log_pmf,
                                         present_and_reads_log_pmf)


def _probs(sup):
    return np.array(sorted(sup.values()))


@pytest.mark.parametrize("depth,n,N", [(12, 10, 12), (12, 8, 13), (12, 12, 18),
                                       (25, 20, 25), (25, 14, 30), (25, 22, 22)])
def test_present_and_reads_matches_exact_convolution(sup, depth, n, N):
    p = _probs(sup)
    D = exact_present_and_reads(p, depth, n_max=30, total_max=32)
    want = math.log(D[n, N])
    got, exact, err = present_and_reads_log_pmf(p, np.ones(p.size), depth, n, N)
    assert abs(got - want) <= SLACK * err + 1e-9, (got, want, err)


def test_all_singletons_matches_exact_convolution(sup):
    p = _probs(sup)
    D = exact_present_and_reads(p, 20, n_max=25, total_max=25)
    got, exact, err = present_and_reads_log_pmf(p, np.ones(p.size), 20, 20, 20)
    assert abs(got - math.log(D[20, 20])) <= SLACK * err + 1e-9, (got, math.log(D[20, 20]), err)


def test_real_multiplicities_are_repeated_units(sup):
    p = _probs(sup)[-40:]
    two = present_and_reads_log_pmf(np.repeat(p, 2), np.ones(2 * p.size), 15, 9, 15)
    mult = present_and_reads_log_pmf(p, np.full(p.size, 2.0), 15, 9, 15)
    assert mult[0] == pytest.approx(two[0], abs=1e-9)


LAMS = [0.4, 0.9, 1.7, 0.2, 3.1, 0.05, 6.0, 0.7]
EDGES = (1, 2, 3, 5)


@pytest.mark.parametrize("F,N", [((5, 1, 1, 1), 16), ((6, 1, 1, 0), 12),
                                 ((3, 2, 2, 1), 20), ((8, 0, 0, 0), 8)])
def test_profile_and_reads_matches_exact_convolution(F, N):
    dist = exact_profile_and_reads(LAMS, EDGES, N)
    want = math.log(dist[(F, N)])
    got, exact, err = counts_given_set_log_pmf(np.array(LAMS), N, EDGES, np.array(F, float))
    assert abs(got - want) <= SLACK * err + 1e-9, (got, want, err)


@pytest.mark.parametrize("N", [8, 11, 16, 30])
def test_reads_given_set_matches_exact_convolution(N):
    dist = exact_profile_and_reads(LAMS, (1,), N)
    want = math.log(dist[((len(LAMS),), N)])
    got, exact, err = counts_given_set_log_pmf(np.array(LAMS), N)
    assert abs(got - want) <= SLACK * err + 1e-9, (got, want, err)


def test_the_fixed_N_pieces_multiply_to_the_multinomial(tiny, sup, rep):
    res = repertoire_likelihood(tiny, *rep, bins=2048)
    assert sum(p.log_prob for p in res.pieces) == pytest.approx(
        log_multinomial(sup, *rep), abs=1e-8)
    assert res.log_likelihood == pytest.approx(log_multinomial(sup, *rep), abs=1e-8)


def test_the_richness_piece_is_P_n_given_N(tiny, sup):
    p = _probs(sup)
    seqs = sorted(sup, key=sup.get)[-15:]
    counts = [2, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1]
    N, n = sum(counts), len(seqs)
    D = exact_present_and_reads(p, N, n_max=n + 2, total_max=N + 2)
    log_pN = N * math.log(N) - N - math.lgamma(N + 1)
    want = math.log(D[n, N]) - log_pN
    res = repertoire_likelihood(tiny, seqs, counts, bins=8192)
    rich = res.piece("richness")
    assert abs(rich.log_prob - want) <= SLACK * rich.error + 1e-4, (rich.log_prob, want)


def test_all_singletons_leave_nothing_to_explain_after_the_set(tiny, sup):
    """N = n reads over n sequences: given the set and N, every count is 1,
    so the pattern and the assignment are certain -- exactly, and labelled so."""
    seqs = sorted(sup, key=sup.get)[-30:]
    res = repertoire_likelihood(tiny, seqs, [1] * len(seqs), bins=2048)
    prof, assign = res.piece("profile"), res.piece("assignment")
    assert prof.exact and assign.exact
    assert prof.error == 0.0 and assign.error == 0.0
    assert prof.log_prob == pytest.approx(0.0, abs=1e-12)
    assert assign.log_prob == pytest.approx(0.0, abs=1e-12)


def test_the_binning_error_is_measured_or_said_to_be_missing(tiny, rep):
    a = tiny.pseq_analysis()
    fine, coarse = a.atoms(1024), a.atoms(512)
    with_both = repertoire_likelihood(tiny, *rep, atoms=fine, atoms_coarse=coarse)
    only_fine = repertoire_likelihood(tiny, *rep, atoms=fine)
    r_both, r_fine = with_both.piece("richness"), only_fine.piece("richness")
    assert r_both.log_prob == r_fine.log_prob
    assert "binning error against half the bins" in r_both.method
    assert "NOT estimated" in r_fine.method
    assert r_both.error > r_fine.error


@pytest.mark.parametrize("lams,edges,F,N", [
    # the total is the least the pattern allows: every count at its bin's floor
    ([0.4, 0.9, 1.7, 0.2, 3.1, 0.05, 6.0, 0.7], (1, 2, 6), (5, 1, 2), 5 + 2 + 12),
    ([0.3, 2.2, 0.8, 1.1, 0.6], (1, 3, 6, 11), (3, 0, 2, 0), 3 + 12),
    # the most it allows, with no open bin occupied: every count at its ceiling
    ([0.4, 0.9, 1.7, 0.2, 3.1], (1, 2, 6, 11), (2, 3, 0, 0), 2 + 3 * 5),
])
def test_a_total_at_the_edge_of_the_pattern_is_exact(lams, edges, F, N):
    dist = exact_profile_and_reads(lams, edges, N)
    want = math.log(dist[(F, N)])
    got, exact, err = counts_given_set_log_pmf(np.array(lams), N, edges, np.array(F, float))
    # no slack hiding behind a huge self-reported error: this case is exact
    assert err <= 1e-3, (got, want, err)
    assert abs(got - want) <= max(SLACK * err, 1e-9), (got, want, err)


def test_an_unreliable_split_says_so(tiny, rep):
    """The test repertoire's wide bins hold a few sequences the model makes
    nearly impossible (two at 11 reads with lambda ~ 1e-7, one at 150): the
    joint saddlepoint is degenerate there. Its pieces must say so, while
    their sum stays exact."""
    res = repertoire_likelihood(tiny, *rep, bins=2048)
    for name in ("profile", "assignment"):
        p = res.piece(name)
        assert ("UNRELIABLE" in p.method) == (p.error > 1.0), (name, p.error, p.method)
    assert res.piece("profile").error > 1.0
    total = sum(p.log_prob for p in res.pieces)
    assert total == pytest.approx(res.log_likelihood, abs=1e-8)
