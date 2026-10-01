"""``repertoire_likelihood`` and ``pair_likelihood`` against the enumerated
support.

The totals are checked to rounding against direct summation. The pieces are
checked two ways: their sums against exact quantities (the chain rule leaves
no room for anything else), and the saddlepoint ones against exact
convolution within the Stirling bound ``test_lattice.py`` pins.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from BEAM import BeamValueError
from BEAM.likelihood import (DEFAULT_PROFILE_EDGES, pair_likelihood,
                             repertoire_likelihood)
from _brute import (SLACK, exact_categorical_sum, expected_richness, expected_shared,
                    log_multinomial, log_poisson_field, log_presence,
                    ztp_bin_probs)

BINS = 2048


def _pieces(res):
    return {p.name: p for p in res.pieces}


# The enumeration underpins every assertion below, so it is checked first.
def test_the_enumeration_is_the_whole_support(tiny, sup):
    assert len(sup) == tiny.path_count()
    assert sum(sup.values()) == pytest.approx(1.0, abs=1e-12)


def test_the_likelihood_is_the_multinomial(tiny, sup, rep):
    res = repertoire_likelihood(tiny, *rep, bins=BINS)
    assert res.log_likelihood == pytest.approx(log_multinomial(sup, *rep),
                                               abs=1e-8)


def test_the_pieces_sum_to_the_poisson_field(tiny, sup, rep):
    res = repertoire_likelihood(tiny, *rep, bins=BINS)
    total = sum(p.log_prob for p in res.pieces)
    assert total == pytest.approx(log_poisson_field(sup, *rep), abs=1e-8)
    assert res.poissonized_log_likelihood == pytest.approx(total, abs=1e-8)


def test_the_two_totals_differ_by_the_chance_of_exactly_n_reads(tiny, rep):
    res = repertoire_likelihood(tiny, *rep, bins=BINS)
    N = sum(rep[1])
    log_pois_N = N * math.log(N) - N - math.lgamma(N + 1)
    assert res.log_prob_total_reads == pytest.approx(log_pois_N, abs=1e-9)
    assert (res.poissonized_log_likelihood - res.log_likelihood
            == pytest.approx(log_pois_N, abs=1e-8))


def test_richness_and_identity_make_the_presence_probability(tiny, sup, rep):
    p = _pieces(repertoire_likelihood(tiny, *rep, bins=BINS))
    got = p["richness"].log_prob + p["identity"].log_prob
    assert got == pytest.approx(log_presence(sup, rep[0], sum(rep[1])),
                                abs=1e-8)


def _spacing(model, bins=BINS):
    return model.pseq_analysis().atoms(bins).grid_spacing


def test_the_expected_richness_is_the_direct_sum(tiny, sup, rep):
    """Exact up to the spectrum binning, whose relative error is about
    spacing^2 / 8; spacing^2 leaves room without hiding a real error."""
    rich = _pieces(repertoire_likelihood(tiny, *rep, bins=BINS))["richness"]
    assert rich.observed.tolist() == [len(rep[0])]
    assert rich.expected[0] == pytest.approx(
        expected_richness(sup, sum(rep[1])), rel=_spacing(tiny) ** 2)


def test_the_binning_error_vanishes_with_finer_bins(tiny, sup, rep):
    want = expected_richness(sup, sum(rep[1]))
    errs = [abs(_pieces(repertoire_likelihood(tiny, *rep, bins=b))["richness"]
                .expected[0] - want) / want for b in (512, 2048, 8192)]
    assert errs[0] > 10 * errs[1] > 100 * errs[2]


def test_the_richness_piece_matches_exact_convolution(tiny, sup, rep):
    """A Poisson-binomial over the whole support, convolved exactly."""
    N = sum(rep[1])
    units = [(math.exp(-N * p), -math.expm1(-N * p)) for p in sup.values()]
    n = len(rep[0])
    exact = math.log(exact_categorical_sum(units)[(len(units) - n, n)])
    # Fine bins, so the check is on the saddlepoint alone: this repertoire
    # sits ten standard deviations below its expected richness, and that far
    # into the tail the binned spectrum's small error in the mean is amplified
    # past the saddlepoint's own (0.009 nats at 2048 bins, 0.0005 at 8192).
    rich = _pieces(repertoire_likelihood(tiny, *rep, bins=32768))["richness"]
    assert not rich.exact
    assert rich.log_prob == pytest.approx(exact, abs=SLACK * rich.error)


def test_the_profile_piece_matches_exact_convolution(tiny, sup, rep):
    edges = (1, 2, 4)
    N = sum(rep[1])
    units = [ztp_bin_probs(N * sup[s], edges) for s in rep[0]]
    c = np.asarray(rep[1])
    obs = (int((c == 1).sum()), int(((c >= 2) & (c < 4)).sum()),
           int((c >= 4).sum()))
    exact = math.log(exact_categorical_sum(units)[obs])
    prof = _pieces(repertoire_likelihood(tiny, *rep, bins=BINS,
                                         profile_edges=edges))["profile"]
    assert prof.observed.tolist() == list(obs)
    assert prof.expected == pytest.approx(np.sum(units, axis=0), rel=1e-9)
    assert prof.log_prob == pytest.approx(
        exact, abs=max(SLACK * prof.error, 1e-9))


def test_exactness_is_labelled_on_every_piece(tiny, rep):
    res = repertoire_likelihood(tiny, *rep, bins=BINS)
    assert [p.name for p in res.pieces] == ["richness", "identity",
                                            "profile", "assignment"]
    assert all(p.exact is False for p in res.pieces)
    assert all(p.method for p in res.pieces)


def test_the_default_profile_bins(tiny, rep):
    assert DEFAULT_PROFILE_EDGES == (1, 2, 3, 4, 5, 6, 11, 101)
    prof = _pieces(repertoire_likelihood(tiny, *rep, bins=BINS))["profile"]
    c = np.asarray(rep[1])
    edges = DEFAULT_PROFILE_EDGES + (np.inf,)
    want = [int(((c >= lo) & (c < hi)).sum()) for lo, hi in zip(edges, edges[1:])]
    assert prof.observed.tolist() == want
    assert want[-1] == 1          # the 150-read clone


def test_a_sequence_the_model_cannot_represent_makes_it_impossible(tiny, rep):
    """Every string up to max_length is possible; one longer is not."""
    res = repertoire_likelihood(tiny, rep[0] + ["A" * 6], rep[1] + [3],
                                bins=BINS)
    assert res.log_likelihood == -math.inf
    assert res.n_unspellable == 1
    assert res.pieces == ()


def test_repeated_sequences_are_one_sequence_with_their_reads_added(tiny, rep):
    """Two rows of one amino-acid sequence (two nucleotide clonotypes) are one
    sequence of the model, carrying the reads of both."""
    seqs, counts = rep
    i = next(j for j, c in enumerate(counts) if c >= 2)
    split_counts = list(counts)
    split_counts[i] -= 1
    split = repertoire_likelihood(tiny, seqs + [seqs[i]], split_counts + [1],
                                  bins=BINS)
    whole = repertoire_likelihood(tiny, seqs, counts, bins=BINS)
    assert split.log_likelihood == pytest.approx(whole.log_likelihood, abs=1e-9)
    assert split.n_unique == whole.n_unique


@pytest.mark.parametrize("bad", [0, -1, 1.5, float("nan")])
def test_counts_must_be_positive_integers(tiny, rep, bad):
    with pytest.raises(BeamValueError):
        repertoire_likelihood(tiny, rep[0], [bad] + rep[1][1:], bins=BINS)


def test_the_profile_edges_must_start_at_one_and_rise(tiny, rep):
    for edges in [(2, 3), (1, 3, 3), ()]:
        with pytest.raises(BeamValueError):
            repertoire_likelihood(tiny, *rep, bins=BINS, profile_edges=edges)


def test_the_result_is_not_a_number(tiny, rep):
    res = repertoire_likelihood(tiny, *rep, bins=BINS)
    with pytest.raises(TypeError):
        float(res)
    with pytest.raises(TypeError):
        float(res.pieces[0])


def test_spectrum_atoms_can_be_passed_in(tiny, rep):
    atoms = tiny.pseq_analysis().atoms(BINS)
    a = repertoire_likelihood(tiny, *rep, atoms=atoms)
    b = repertoire_likelihood(tiny, *rep, bins=BINS)
    assert [p.log_prob for p in a.pieces] == pytest.approx(
        [p.log_prob for p in b.pieces], abs=1e-12)


# ── Two repertoires under one model ────────────────────────────────

def test_pair_totals_are_the_two_single_totals(tiny, rep, rep2):
    pair = pair_likelihood(tiny, *rep, *rep2, bins=BINS)
    one = repertoire_likelihood(tiny, *rep, bins=BINS)
    two = repertoire_likelihood(tiny, *rep2, bins=BINS)
    assert pair.log_likelihood == pytest.approx(
        one.log_likelihood + two.log_likelihood, abs=1e-8)
    assert pair.poissonized_log_likelihood == pytest.approx(
        one.poissonized_log_likelihood + two.poissonized_log_likelihood,
        abs=1e-8)


def test_pair_pieces_sum_to_both_poisson_fields(tiny, sup, rep, rep2):
    pair = pair_likelihood(tiny, *rep, *rep2, bins=BINS)
    assert [p.name for p in pair.pieces] == ["richness", "sharing", "identity",
                                             "profile", "assignment"]
    assert sum(p.log_prob for p in pair.pieces) == pytest.approx(
        log_poisson_field(sup, *rep) + log_poisson_field(sup, *rep2), abs=1e-8)


def test_pair_sharing_is_counted_and_expected_exactly(tiny, sup, rep, rep2):
    pair = pair_likelihood(tiny, *rep, *rep2, bins=BINS)
    share = _pieces(pair)["sharing"]
    k = len(set(rep[0]) & set(rep2[0]))
    assert pair.n_shared == k
    assert share.observed.tolist() == [k]
    assert share.expected[0] == pytest.approx(
        expected_shared(sup, sum(rep[1]), sum(rep2[1])),
        rel=_spacing(tiny) ** 2)


def test_pair_counts_split_into_shared_and_private(tiny, rep, rep2):
    pair = pair_likelihood(tiny, *rep, *rep2, bins=BINS)
    p = _pieces(pair)
    assert (pair.shared_counts_log_prob + pair.private_counts_log_prob
            == pytest.approx(p["profile"].log_prob + p["assignment"].log_prob,
                             abs=1e-8))


def test_pair_with_a_sequence_it_cannot_represent_is_impossible(tiny, rep, rep2):
    pair = pair_likelihood(tiny, *rep, rep2[0] + ["A" * 6], rep2[1] + [1],
                           bins=BINS)
    assert pair.log_likelihood == -math.inf
    assert pair.n_unspellable == (0, 1)
    assert pair.pieces == ()


def test_the_entry_points_are_exported_at_the_top_level():
    import BEAM
    for name in ("repertoire_likelihood", "pair_likelihood",
                 "RepertoireLikelihood", "PairLikelihood", "Piece"):
        assert name in BEAM.__all__
    assert BEAM.repertoire_likelihood is repertoire_likelihood
    assert BEAM.pair_likelihood is pair_likelihood


def test_richness_survives_a_spectrum_of_1e24_sequences(tiny, rep):
    """The foundation model's support is 4.5e24. A spectrum that size must not
    swallow the observed n: here 2e24 sequences at 1e-25 each (Poisson(59) of
    them seen) and three common ones, checked against their exact convolution."""
    from BEAM import Atoms
    N, n = sum(rep[1]), len(rep[0])
    p = np.array([1e-25, 0.01])
    m = np.array([2e24, 3.0])
    atoms = Atoms(probabilities=p, multiplicities=m, surprisal=-np.log(p),
                  grid_spacing=0.01)
    lam = 2e24 * -math.expm1(-N * 1e-25)
    q = -math.expm1(-N * 0.01)
    exact = math.log(sum(
        math.exp((n - k) * math.log(lam) - lam - math.lgamma(n - k + 1)
                 + math.lgamma(4) - math.lgamma(k + 1) - math.lgamma(4 - k)
                 + k * math.log(q) + (3 - k) * math.log1p(-q))
        for k in range(4)))
    rich = _pieces(repertoire_likelihood(tiny, *rep, atoms=atoms))["richness"]
    assert rich.log_prob == pytest.approx(exact, abs=SLACK * rich.error)
    assert rich.expected[0] == pytest.approx(lam + 3 * q, rel=1e-12)
