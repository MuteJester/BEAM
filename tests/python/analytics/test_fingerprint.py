"""Independent sequence enumeration and masked DP references for fingerprints."""
import itertools
import math

import numpy as np
import pytest

import BEAM
from BEAM import BeamModel, BeamQuery
from BEAM.analytics.fingerprint import compare
from BEAM.model import alphabets


def model(reverse=False, *, order=2, weights=(0.7, 0.4)):
    seq = ["ACGT", "CCGT", "AGTA", "TTAC", "GCNA", "ANNC", "ACG", "TCA"]
    counts = [29, 17, 13, 11, 7, 5, 3, 2]
    if reverse:
        counts = counts[::-1]
    return BeamModel(seq, abundances=counts, alphabet="nt", order=order,
                     max_length=4, alpha=2.0, weights=weights)


@pytest.fixture(scope="module")
def pair():
    return model(), model(True)


def direct_law(m, length):
    seq = ["".join(s) for s in itertools.product(alphabets.symbols(m.alphabet), repeat=length)]
    probabilities = np.exp(m.log_pseq_batch(seq) - m.log_length_prior(length))
    assert probabilities.sum() == pytest.approx(1, abs=2e-12)
    return seq, probabilities


def assert_leaf_probabilities(result, a, b):
    for L, row in result.by_length.items():
        seq, p = direct_law(a, L)
        _, q = direct_law(b, L)
        assignments = np.array([[leaf.matches(s) for s in seq] for leaf in row.leaves])
        assert (assignments.sum(axis=0) == 1).all()
        gotp = np.array([leaf.conditional_p for leaf in row.leaves])
        gotq = np.array([leaf.conditional_q for leaf in row.leaves])
        np.testing.assert_allclose(assignments @ p, gotp, atol=2e-12, rtol=2e-11)
        np.testing.assert_allclose(assignments @ q, gotq, atol=2e-12, rtol=2e-11)
        for m, expected in [(a, gotp), (b, gotq)]:
            query = BeamQuery(m)
            for leaf, value in zip(row.leaves, expected):
                cond = query.condition(L)
                for i, mask in enumerate(leaf.masks):
                    cond = cond.oneof(i, [aa for j, aa in enumerate(leaf.alphabet) if mask & (1 << j)])
                assert query.probability(cond) == pytest.approx(value, abs=2e-12, rel=2e-11)
        counts = result.counts(seq)
        assert sum(counts.values()) == len(seq)
        for leaf, indicators in zip(row.leaves, assignments):
            assert counts[L, leaf.node] == indicators.sum()


@pytest.mark.parametrize("order,weights", [(2, (1., 0.)), (2, (0.7, 0.4)), (4, (0., 1.))])
def test_cached_probabilities_against_scoring_and_masked_queries(order, weights):
    a, b = model(order=order, weights=weights), model(True, order=order, weights=weights)
    r = compare(a, b, max_leaves=7, max_depth=3, lookahead=2,
                beam_width=12, max_queries=80)
    assert_leaf_probabilities(r, a, b)
    assert 0 <= r.squared_distance <= r.full_squared_distance + 1e-12
    expected = r.length_squared_distance + sum(
        math.exp((a.log_length_prior(L)+b.log_length_prior(L))/2)*row.squared_distance
        for L, row in r.by_length.items())
    assert r.squared_distance == pytest.approx(expected, abs=3e-13)
    assert all(len(row.leaves) <= 7 and row.queries <= 80 for row in r.by_length.values())


def test_identity_swap_threads_and_unsearched_lengths(pair):
    a, b = pair
    params = dict(lengths=[4], max_leaves=8, beam_width=16, max_queries=70)
    initial_threads = BEAM.threads()
    try:
        BEAM.set_threads(1)
        r = compare(a, b, **params)
        same = compare(a, a, **params)
        swapped = compare(b, a, **params)
        BEAM.set_threads(2)
        parallel = compare(a, b, **params)
    finally:
        BEAM.set_threads(initial_threads)
    assert same.squared_distance < 1e-28
    assert same.coverage is None
    assert r.by_length[3].searched is False
    assert len(r.by_length[3].leaves) == 1
    assert r.squared_distance == swapped.squared_distance == parallel.squared_distance
    for L in r.by_length:
        np.testing.assert_array_equal(r.by_length[L].nodes, swapped.by_length[L].nodes)
        np.testing.assert_array_equal(r.by_length[L].nodes, parallel.by_length[L].nodes)
        np.testing.assert_array_equal(r.by_length[L].conditional_masses,
                                      swapped.by_length[L].conditional_masses[:, ::-1])


def parity_pair(length, active, order):
    seq = ["".join(s) for s in itertools.product("AC", repeat=length)]
    weights = [900 if sum(s[i] == "C" for i in active) % 2 == 0 else 100 for s in seq]
    common = dict(alphabet="nt", order=order, max_length=length,
                  weights=(1., 0.), alpha=1e-6, alpha_length=1e-6)
    return (BeamModel(seq, abundances=weights, **common),
            BeamModel(seq, abundances=[1000-w for w in weights], **common))


@pytest.mark.parametrize("length,active,order,lookahead", [
    (5, (0, 4), 4, 2), (3, (0, 1, 2), 2, 3)])
def test_hidden_gapped_and_three_way_combinations(length, active, order, lookahead):
    a, b = parity_pair(length, active, order)
    base = dict(lengths=[length], max_leaves=12, max_depth=length,
                beam_width=100, max_queries=1000, min_gain=1e-7)
    shallow = compare(a, b, lookahead=lookahead-1, **base)
    deep = compare(a, b, lookahead=lookahead, **base)
    assert shallow.within_squared_distance < 1e-12
    assert deep.within_squared_distance > 0.39
    assert deep.coverage > 0.99
    assert_leaf_probabilities(deep, a, b)


def test_query_budget_and_zero_depth(pair):
    a, b = pair
    limited = compare(a, b, lengths=[4], max_queries=1)
    assert limited.by_length[4].queries == 1
    assert limited.by_length[4].budget_exhausted
    assert_leaf_probabilities(limited, a, b)
    empty = compare(a, b, max_depth=0, compute_full=False)
    assert empty.squared_distance == empty.length_squared_distance
    assert empty.coverage is empty.within_coverage is empty.residual is None


def test_length_only_and_unsupported_observations():
    a = BeamModel(["A"], alphabet="nt", order=2, max_length=2)
    b = BeamModel(["CC"], alphabet="nt", order=2, max_length=2)
    r = compare(a, b, lengths=[])
    assert r.squared_distance == r.length_squared_distance
    assert r.within_squared_distance == 0
    assert all(not row.searched for row in r.by_length.values())
    assert sum(r.counts(["A", "CC"]).values()) == 2
    with pytest.raises(ValueError):
        r.counts(["ZZ"])
    with pytest.raises(ValueError):
        r.counts(["AAA"])


def test_single_residue_length_with_long_context():
    a = BeamModel(["A", "C"], abundances=[9, 1], alphabet="nt", order=4, max_length=2)
    b = BeamModel(["A", "C"], abundances=[1, 9], alphabet="nt", order=4, max_length=2)
    r = compare(a, b, lengths=[1], lookahead=3)
    assert_leaf_probabilities(r, a, b)
    assert r.by_length[1].nodes[:, 5].max() <= 1


@pytest.mark.parametrize("kw", [dict(lookahead=4), dict(max_queries=0),
    dict(max_queries=2**32), dict(max_depth=-1), dict(max_leaves=1.5),
    dict(beam_width=True), dict(min_mass=math.nan), dict(min_gain=0), dict(lengths=[99])])
def test_invalid_budgets_refused(pair, kw):
    with pytest.raises(ValueError):
        compare(*pair, **kw)


def test_budget_refusal_precedes_cache_allocation(pair):
    initial = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(1024)
        with pytest.raises(BEAM.BudgetExceeded):
            compare(*pair, lengths=[4], compute_full=False)
    finally:
        BEAM.set_memory_budget(initial)


def test_shipped_order_amino_acid_gaps():
    seq = ["CASSYF", "CASSAF", "CASRYF", "CAGGWF", "CASGAF", "WYAACF"]
    a = BeamModel(seq, abundances=[90, 20, 30, 40, 50, 60], order=4, max_length=6, alpha=2.)
    b = BeamModel(seq, abundances=[20, 90, 50, 30, 60, 40], order=4, max_length=6, alpha=2.)
    r = compare(a, b, lengths=[6], max_queries=8, max_leaves=4, beam_width=4)
    for m, side in [(a, "conditional_p"), (b, "conditional_q")]:
        q = BeamQuery(m)
        for leaf in r.by_length[6].leaves:
            c = q.condition(6)
            for i, mask in enumerate(leaf.masks):
                c = c.oneof(i, [aa for j, aa in enumerate(leaf.alphabet) if mask & (1 << j)])
            assert q.probability(c) == pytest.approx(getattr(leaf, side), abs=1e-12, rel=2e-11)
