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
from BEAM.likelihood.fixed_total import counts_given_set_log_pmf
from _brute import (SLACK, expected_richness_at_reads, expected_shared,
                    exact_profile_and_reads, log_multinomial, log_poisson_field,
                    log_presence)

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


def test_the_pieces_multiply_to_the_multinomial(tiny, sup, rep):
    """Every piece is conditioned on the read total N, so the four multiply to
    P(counts | N) itself; the Poisson-field total is reported beside it."""
    res = repertoire_likelihood(tiny, *rep, bins=BINS)
    total = sum(p.log_prob for p in res.pieces)
    assert total == pytest.approx(log_multinomial(sup, *rep), abs=1e-8)
    assert res.poissonized_log_likelihood == pytest.approx(
        log_poisson_field(sup, *rep), abs=1e-8)


def test_the_two_totals_differ_by_the_chance_of_exactly_n_reads(tiny, rep):
    res = repertoire_likelihood(tiny, *rep, bins=BINS)
    N = sum(rep[1])
    log_pois_N = N * math.log(N) - N - math.lgamma(N + 1)
    assert res.log_prob_total_reads == pytest.approx(log_pois_N, abs=1e-9)
    assert (res.poissonized_log_likelihood - res.log_likelihood
            == pytest.approx(log_pois_N, abs=1e-8))


def test_richness_and_identity_make_the_presence_probability(tiny, sup, rep):
    """P(n | N) P(S | n, N) = P(S, N) / P(N) = P(S) P(N | S) / P(N)."""
    p = _pieces(repertoire_likelihood(tiny, *rep, bins=BINS))
    N = sum(rep[1])
    lam = np.array([N * sup[s] for s in rep[0]])
    log_N_given_S = counts_given_set_log_pmf(lam, N)[0]
    log_pois_N = N * math.log(N) - N - math.lgamma(N + 1)
    got = p["richness"].log_prob + p["identity"].log_prob
    want = log_presence(sup, rep[0], N) + log_N_given_S - log_pois_N
    assert got == pytest.approx(want, abs=1e-8)


def _spacing(model, bins=BINS):
    return model.pseq_analysis().atoms(bins).grid_spacing


def test_the_expected_richness_is_the_direct_sum(tiny, sup, rep):
    """Exact up to the spectrum binning, whose relative error is about
    spacing^2 / 8; spacing^2 leaves room without hiding a real error."""
    rich = _pieces(repertoire_likelihood(tiny, *rep, bins=BINS))["richness"]
    assert rich.observed.tolist() == [len(rep[0])]
    assert rich.expected[0] == pytest.approx(
        expected_richness_at_reads(sup, sum(rep[1])), rel=_spacing(tiny) ** 2)


def test_the_binning_error_vanishes_with_finer_bins(tiny, sup, rep):
    want = expected_richness_at_reads(sup, sum(rep[1]))
    errs = [abs(_pieces(repertoire_likelihood(tiny, *rep, bins=b))["richness"]
                .expected[0] - want) / want for b in (512, 2048, 8192)]
    assert errs[0] > 10 * errs[1] > 100 * errs[2]


def test_the_profile_piece_is_P_F_given_S_and_N(tiny, sup):
    """P(F | S, N) = P(F, N | S) / P(N | S), each convolved exactly over the
    seen sequences' zero-truncated counts (a repertoire small enough for it)."""
    seqs = sorted(sup, key=sup.get)[-12:]
    counts = [1, 1, 2, 1, 3, 1, 1, 5, 1, 2, 1, 1]
    N, edges = sum(counts), (1, 2, 4)
    lam = [N * sup[s] for s in seqs]
    c = np.asarray(counts)
    obs = (int((c == 1).sum()), int(((c >= 2) & (c < 4)).sum()), int((c >= 4).sum()))
    joint = exact_profile_and_reads(lam, edges, N)
    p_N = sum(w for (F, T), w in joint.items() if T == N)
    exact = math.log(joint[(obs, N)] / p_N)
    exp_F = np.sum([np.array(F) * w for (F, T), w in joint.items() if T == N], axis=0) / p_N
    prof = _pieces(repertoire_likelihood(tiny, seqs, counts, bins=BINS,
                                         profile_edges=edges))["profile"]
    assert prof.observed.tolist() == list(obs)
    assert prof.log_prob == pytest.approx(exact, abs=max(SLACK * prof.error, 1e-9))
    # The expectation is approximate (see expected_bins_given_reads): loose
    # at a dozen sequences, within 10% at forty (next test). Only the
    # probability is held to exact convolution.
    assert prof.expected == pytest.approx(exp_F, rel=0.25)


def test_the_expected_profile_tightens_with_more_sequences(tiny, sup):
    seqs = sorted(sup, key=sup.get)[-160:-120]
    rng = np.random.default_rng(3)
    counts = [int(x) for x in 1 + rng.poisson(0.35, size=len(seqs))]
    N, edges = sum(counts), (1, 2, 4)
    lam = [N * sup[s] for s in seqs]
    joint = exact_profile_and_reads(lam, edges, N)
    p_N = sum(w for (F, T), w in joint.items() if T == N)
    exp_F = np.sum([np.array(F) * w for (F, T), w in joint.items() if T == N], axis=0) / p_N
    prof = _pieces(repertoire_likelihood(tiny, seqs, counts, bins=BINS,
                                         profile_edges=edges))["profile"]
    assert prof.expected == pytest.approx(exp_F, rel=0.10)


def test_a_pattern_that_fixes_the_total_is_conditioned_on_exactly(tiny, sup):
    """Counts 1-4 only, each its own bin: the pattern determines N, so
    P(F | S, N) is P(F | S) / P(N | S), with no joint to approximate."""
    seqs = sorted(sup, key=sup.get)[-15:]
    counts = [2, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1]
    N, edges = sum(counts), (1, 2, 3, 4, 5)
    lam = [N * sup[s] for s in seqs]
    joint = exact_profile_and_reads(lam, edges, N)
    p_N = sum(w for (F, T), w in joint.items() if T == N)
    obs = (11, 2, 1, 1, 0)
    prof = _pieces(repertoire_likelihood(tiny, seqs, counts, bins=BINS,
                                         profile_edges=edges))["profile"]
    assert prof.log_prob == pytest.approx(math.log(joint[(obs, N)] / p_N),
                                          abs=max(SLACK * prof.error, 1e-9))


def test_exactness_is_labelled_on_every_piece(tiny, rep):
    res = repertoire_likelihood(tiny, *rep, bins=BINS)
    assert [p.name for p in res.pieces] == ["richness", "identity",
                                            "profile", "assignment"]
    assert all(p.exact is False for p in res.pieces)
    assert all(p.method for p in res.pieces)


def test_the_default_profile_bins(tiny, rep):
    """`DEFAULT_PROFILE_EDGES` is the documented fixed set; it is no longer
    what a call uses unless asked for, since automatic bins replaced it."""
    assert DEFAULT_PROFILE_EDGES == (1, 2, 3, 4, 5, 6, 11, 101)
    prof = _pieces(repertoire_likelihood(tiny, *rep, bins=BINS,
                                         profile_edges=DEFAULT_PROFILE_EDGES))["profile"]
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
    """The foundation model's support is 1.4e35. A spectrum that size must not
    swallow the observed n: here 2e24 sequences at 1e-25 each and three
    common ones, P(n | N) against its exact value. At that probability a rare
    sequence is never read twice (2e-21 expected doubles), so the rare part
    gives n_rare = N_rare ~ Poisson(L); the common part, three Poisson(N/100)
    units, is convolved exactly."""
    from BEAM import Atoms
    N, n = sum(rep[1]), len(rep[0])
    p = np.array([1e-25, 0.01])
    m = np.array([2e24, 3.0])
    atoms = Atoms(probabilities=p, multiplicities=m, surprisal=-np.log(p),
                  grid_spacing=0.01)
    L = 2e24 * N * 1e-25
    mu = N * 0.01
    # common units: joint pmf of (seen, reads)
    reads = np.arange(N + 1)
    pois = np.exp(reads * math.log(mu) - mu - np.array([math.lgamma(r + 1) for r in reads]))
    C = np.zeros((4, N + 1))
    C[0, 0] = 1.0
    for _ in range(3):
        nxt = C * pois[0]
        for r in range(1, N + 1):
            nxt[1:, r:] += C[:-1, :N + 1 - r] * pois[r]
        C = nxt
    joint = sum(C[k, T] * math.exp((n - k) * math.log(L) - L - math.lgamma(n - k + 1))
                for k in range(4) for T in range(N + 1) if N - T == n - k and n - k >= 0)
    mass = 2e24 * 1e-25 + 0.03
    log_pN = N * math.log(N * mass) - N * mass - math.lgamma(N + 1)
    exact = math.log(joint) - log_pN
    rich = _pieces(repertoire_likelihood(tiny, *rep, atoms=atoms))["richness"]
    assert rich.log_prob == pytest.approx(exact, abs=SLACK * rich.error + 1e-6)
    assert rich.expected[0] == pytest.approx(
        2e24 * -math.expm1(N * math.log1p(-1e-25)) + 3 * -math.expm1(N * math.log1p(-0.01)),
        rel=1e-12)
