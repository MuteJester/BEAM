"""The clone-size spectrum: how many sequences the model expects at each count.

For a repertoire the model has already been told the SET of sequences in,
each sequence's count is an independent zero-truncated Poisson, so

    E[F_k] = sum_i P(ZTP(lam_i) = k)        Var[F_k] = sum_i p_ik (1 - p_ik)

are EXACT -- plain sums over independent sequences, with no tilt, no
saddlepoint and no binning. That is worth having on its own: the binning the
chain's profile piece needs exists only to keep a saddlepoint's coordinate
count down, and it is what makes a bin like "over 100 reads" degenerate.

Three invariants pin the arithmetic, and they are the reason to trust it:
every seen sequence has some count, the counts add up to the expected read
total, and both agree with a direct computation per sequence.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM.likelihood import abundance_spectrum
from _brute import repertoire, support, tiny_weighted_model


def _lam(model, seqs, counts):
    N = float(sum(counts))
    return N * np.exp(np.asarray(model.log_pseq_batch(list(seqs)), dtype=float))


def _direct(lam, k):
    """E[F_k] and Var[F_k] by the definition, one sequence at a time."""
    e = v = 0.0
    for L in lam:
        p = math.exp(k * math.log(L) - L - math.lgamma(k + 1)) / -math.expm1(-L)
        e += p
        v += p * (1.0 - p)
    return e, v


def test_matches_the_definition_sequence_by_sequence(tiny, rep):
    sp = abundance_spectrum(tiny, *rep)
    lam = _lam(tiny, *rep)
    for k in (1, 2, 3, 7, 150):
        if k not in sp.counts:
            continue
        i = list(sp.counts).index(k)
        e, v = _direct(lam, k)
        assert sp.expected[i] == pytest.approx(e, rel=1e-12), k
        assert sp.sd[i] == pytest.approx(math.sqrt(v), rel=1e-12), k


def test_every_seen_sequence_has_some_count(tiny, rep):
    """sum over ALL k of E[F_k] is the number of sequences, exactly: each one
    was seen, so it has some count."""
    sp = abundance_spectrum(tiny, *rep, max_count=4000)
    assert float(np.sum(sp.expected)) == pytest.approx(sp.n_sequences, rel=1e-9)


def test_the_counts_add_up_to_the_expected_read_total(tiny, rep):
    """sum over k of k E[F_k] is E[reads | these sequences were seen]."""
    sp = abundance_spectrum(tiny, *rep, max_count=4000)
    lam = _lam(tiny, *rep)
    want = float(np.sum(lam / -np.expm1(-lam)))        # sum of ZTP means
    got = float(np.sum(np.asarray(sp.counts, float) * sp.expected))
    assert got == pytest.approx(want, rel=1e-9)


def test_the_observed_counts_are_the_repertoire(tiny, rep):
    sp = abundance_spectrum(tiny, *rep)
    seqs, counts = rep
    for k, o in zip(sp.counts, sp.observed):
        assert o == sum(1 for c in counts if c == k), k
    assert int(np.sum(sp.observed)) == len(counts)


def test_the_z_scores_say_where_the_model_is_wrong(tiny, rep):
    """Where the model expects enough sequences to have a standard deviation
    worth dividing by, z is that division."""
    from BEAM.likelihood.chain import GAUSSIAN_MIN_EXPECTED
    sp = abundance_spectrum(tiny, *rep)
    ok = np.asarray(sp.expected) >= GAUSSIAN_MIN_EXPECTED
    assert np.all(np.isfinite(np.asarray(sp.z)[ok]))
    i = list(sp.counts).index(1)
    assert sp.expected[i] >= GAUSSIAN_MIN_EXPECTED
    assert sp.z[i] == pytest.approx((sp.observed[i] - sp.expected[i]) / sp.sd[i], rel=1e-12)


def test_a_count_no_sequence_reached_is_reported_when_asked(tiny, rep):
    sp = abundance_spectrum(tiny, *rep, counts_wanted=[1, 2, 999])
    assert list(sp.counts) == [1, 2, 999]
    assert sp.observed[-1] == 0
    assert sp.expected[-1] >= 0.0


def test_it_agrees_with_an_enumerated_model(sup):
    """On the enumerable model, against the support's own probabilities."""
    from BEAM.likelihood import abundance_spectrum as spec
    m = tiny_weighted_model(5)
    seqs, counts = repertoire(support(m), 40, seed=4)
    sp = spec(m, seqs, counts)
    lam = _lam(m, seqs, counts)
    for i, k in enumerate(sp.counts):
        e, v = _direct(lam, int(k))
        assert sp.expected[i] == pytest.approx(e, rel=1e-11)
        assert sp.sd[i] == pytest.approx(math.sqrt(v), rel=1e-11)


def test_a_clone_size_the_model_all_but_forbids_gets_no_z(tiny, rep):
    """Where the model expects almost no sequences the count is Poisson-like,
    so a standardised departure is meaningless -- 1e136 for one clone. Those
    sizes report nan and are read off observed against expected instead."""
    sp = abundance_spectrum(tiny, *rep, max_count=400)
    e = np.asarray(sp.expected)
    rare = e < 10.0
    assert np.any(rare)
    assert np.all(np.isnan(np.asarray(sp.z)[rare]))
    assert np.all(np.isfinite(np.asarray(sp.z)[~rare]))
    assert np.nanmax(np.abs(sp.z)) < 1e4            # no runaway standardisation
