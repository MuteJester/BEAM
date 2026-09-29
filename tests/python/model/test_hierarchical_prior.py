"""The hierarchical Bayesian prior, against arithmetic that owes nothing to it.

Every row of either table is the posterior mean of a Dirichlet centred on its
parent, the same depth with one residue less of context:

    P_k(x) = (n_k(x) + alpha[k] * P_{k-1}(x)) / (n_k + alpha[k]),

level 0 the depth alone and the uniform 1/R below it. The length prior is the
same rule over lengths. What is checked here:

  * the tables, read back through ``log_pseq`` with one anchor switched off,
    equal a plain-Python evaluation of that rule from the training counts --
    seen contexts, unseen ones, junk, and a length the corpus never had;
  * every sequence up to ``max_length`` is possible, and the probabilities
    sum to one over the whole enumerated space;
  * sparse and dense storage are the same model bit for bit;
  * the strengths are validated, stored, re-applied and fitted as documented.
"""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict

import numpy as np
import pytest

from BEAM import BeamModel, fit_alpha

AA = "ACDEFGHIKLMNPQRSTVWY"
NT = "ACGTN"


def _corpus(n=300, seed=2):
    rng = random.Random(seed)
    seqs = ["CASS" + "".join(rng.choice("LGQTEYRS") for _ in range(rng.randint(2, 8)))
            + "QYF" for _ in range(n)]
    return seqs, [rng.randint(1, 20) for _ in seqs]


def _reference(seqs, counts, order, alpha, direction):
    """P(x | depth, context) by the rule, straight from the training counts.

    Contexts are the ``order`` nearest residues, sentinel-padded past either
    end. A level-k row pools every top row sharing its k nearest residues.
    """
    cnt = [defaultdict(Counter) for _ in range(order + 1)]
    for s, c in zip(seqs, counts):
        L = len(s)
        for i in range(L):
            d, ctx = _depth_and_context(s, i, order, direction)
            for k in range(order + 1):
                cnt[k][(d, tuple(ctx[:k]))][s[i]] += c

    def prob(d, ctx, x):
        p = 1.0 / 20.0
        for k in range(order + 1):
            row = cnt[k].get((d, tuple(ctx[:k])))
            n = sum(row.values()) if row else 0
            p = ((row[x] if row else 0) + alpha[k] * p) / (n + alpha[k])
        return p
    return prob


def _depth_and_context(s, i, order, direction):
    L = len(s)
    if direction == "forward":
        return i, [s[i - j] if i - j >= 0 else "#" for j in range(1, order + 1)]
    return L - 1 - i, [s[i + j] if i + j < L else "#" for j in range(1, order + 1)]


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_the_tables_are_the_rule_applied_to_the_counts(direction):
    """With one anchor off, log P(s) - log pi(L) is the sum of that anchor's
    per-residue terms: a proper chain, so Z(L) is one."""
    seqs, counts = _corpus()
    order, alpha = 3, (2.0, 5.0, 7.0, 11.0)
    weights = (1.0, 0.0) if direction == "forward" else (0.0, 1.0)
    m = BeamModel(seqs, abundances=counts, order=order, alpha=alpha,
                  weights=weights)
    prob = _reference(seqs, counts, order, alpha, direction)
    rng = random.Random(7)
    probe = (seqs[:40]
             + ["CASS" + "".join(rng.choice(AA) for _ in range(rng.randint(1, 10)))
                + "F" for _ in range(80)]
             + ["WWWW", "C", "Y" * 27])
    for s in probe:
        want = sum(math.log(prob(*_depth_and_context(s, i, order, direction), s[i]))
                   for i in range(len(s)))
        got = m.log_pseq(s) - m.log_length_prior(len(s))
        assert got == pytest.approx(want, abs=1e-12), s


def test_the_length_prior_is_the_rule_over_lengths():
    seqs, counts = _corpus()
    m = BeamModel(seqs, abundances=counts, order=3, alpha_length=4.0)
    n_len = Counter()
    for s, c in zip(seqs, counts):
        n_len[len(s)] += c
    N = sum(counts)
    for L in range(1, 28):
        want = math.log((n_len[L] + 4.0 / 27) / (N + 4.0))
        assert m.log_length_prior(L) == pytest.approx(want, abs=1e-14), L


@pytest.fixture(scope="module")
def small_nt():
    rng = random.Random(1)
    seqs = ["".join(rng.choice("ACGT") for _ in range(rng.randint(3, 5)))
            for _ in range(60)]
    return seqs


@pytest.mark.parametrize("sparse", [True, False])
def test_every_sequence_is_possible_and_the_total_is_one(small_nt, sparse):
    m = BeamModel(small_nt, alphabet="nt", order=2, max_length=5, sparse=sparse)
    total, n = 0.0, 0
    for L in range(1, 6):
        for t in itertools.product(NT, repeat=L):
            lp = m.log_pseq("".join(t))
            assert math.isfinite(lp)
            total += math.exp(lp)
            n += 1
    assert n == sum(5 ** L for L in range(1, 6))
    assert total == pytest.approx(1.0, abs=1e-12)
    assert m.path_count() == n


def test_sparse_and_dense_are_the_same_model_bit_for_bit(small_nt):
    a = BeamModel(small_nt, alphabet="nt", order=3, max_length=6, sparse=True)
    b = BeamModel(small_nt, alphabet="nt", order=3, max_length=6, sparse=False)
    rng = random.Random(5)
    probe = ["".join(rng.choice(NT) for _ in range(rng.randint(1, 6)))
             for _ in range(3000)]
    assert np.array_equal(a.log_pseq_batch(probe), b.log_pseq_batch(probe))
    assert a.entropy() == b.entropy()


def test_junk_is_far_less_likely_than_a_cdr3():
    seqs, counts = _corpus(n=800)
    m = BeamModel(seqs, abundances=counts)
    assert m.log_pseq("CASSLGQTEYF") - m.log_pseq("WWWWWWWWWWW") > 30


def test_without_alpha_the_strengths_come_from_the_counts():
    m = BeamModel(["CASSLF", "CASSQF", "CASRLF"] * 3, order=3)
    cfg = m.config
    assert cfg["alpha_auto"] and cfg["alpha_length_auto"]
    assert len(cfg["alpha"]) == 4
    assert all(math.isfinite(a) and a > 0 for a in cfg["alpha"])
    assert "strict" not in cfg


def test_fixed_strengths_are_kept_and_reported_as_fixed():
    m = BeamModel(["CASSLF", "CASSQF"], order=2, alpha=(1.0, 2.0, 3.0),
                  alpha_length=4.0)
    cfg = m.config
    assert cfg["alpha"] == (1.0, 2.0, 3.0) and cfg["alpha_length"] == 4.0
    assert not cfg["alpha_auto"] and not cfg["alpha_length_auto"]


def _evidence(rows, parents, a):
    """log prod_rows DirMult(counts | a * parent), straight from lgamma."""
    tot = 0.0
    for n, m in zip(rows, parents):
        tot += math.lgamma(a) - math.lgamma(sum(n.values()) + a)
        for x, c in n.items():
            tot += math.lgamma(c + a * m[x]) - math.lgamma(a * m[x])
    return tot


def test_the_estimated_strengths_maximise_the_evidence():
    """Each level's strength is the one that makes that level's counts most
    probable with the Dirichlet integrated out, against parents built with the
    strengths fitted below it. Recomputed here level by level from the
    training strings, both anchors pooled, and checked against a grid and
    against its immediate neighbours."""
    seqs, counts = _corpus(n=400, seed=12)
    order = 2
    m = BeamModel(seqs, abundances=counts, order=order, max_length=20)
    fitted = m.config["alpha"]
    cnt = [defaultdict(Counter) for _ in range(order + 1)]
    for s, c in zip(seqs, counts):
        for direction in ("forward", "backward"):
            for i in range(len(s)):
                d, ctx = _depth_and_context(s, i, order, direction)
                for k in range(order + 1):
                    cnt[k][(direction, d, tuple(ctx[:k]))][s[i]] += c
    post = {}
    for k in range(order + 1):
        keys = list(cnt[k])
        rows = [cnt[k][q] for q in keys]
        parents = [defaultdict(lambda: 0.05) if k == 0
                   else post[(q[0], q[1], q[2][:k - 1])] for q in keys]
        best = _evidence(rows, parents, fitted[k])
        for a in [10 ** e for e in (-2, -1, 0, 1, 2, 3, 4, 5)] + \
                 [fitted[k] * 1.01, fitted[k] / 1.01]:
            assert _evidence(rows, parents, a) <= best + 1e-6, (k, a)
        for q, n in zip(keys, rows):
            tot = sum(n.values())
            par = post[(q[0], q[1], q[2][:k - 1])] if k else None
            post[q] = defaultdict(float, {
                x: (n[x] + fitted[k] * (par[x] if par else 0.05)) / (tot + fitted[k])
                for x in AA})


def test_the_estimate_does_not_depend_on_the_thread_count():
    import BEAM
    seqs, counts = _corpus(n=600, seed=13)
    got = []
    for t in (1, 4):
        BEAM.set_threads(t)
        try:
            got.append(BeamModel(seqs, abundances=counts, order=3).config["alpha"])
        finally:
            BEAM.set_threads(0)
    assert got[0] == got[1]


def test_auto_hands_a_fixed_model_back_to_the_data():
    seqs, counts = _corpus()
    fixed = BeamModel(seqs, abundances=counts, order=3, alpha=10.0,
                      alpha_length=10.0)
    fixed.set_alpha("auto", "auto")
    auto = BeamModel(seqs, abundances=counts, order=3)
    assert fixed.config == auto.config
    probe = seqs[:20] + ["CASSWWWF"]
    assert np.array_equal(fixed.log_pseq_batch(probe), auto.log_pseq_batch(probe))


def test_one_number_sets_every_level():
    m = BeamModel(["CASSLF", "CASSQF"], order=2, alpha=3.5)
    assert m.config["alpha"] == (3.5, 3.5, 3.5)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_a_strength_must_be_finite_and_positive(bad):
    with pytest.raises(ValueError):
        BeamModel(["CASSLF"], order=2, alpha=bad)
    with pytest.raises(ValueError):
        BeamModel(["CASSLF"], order=2, alpha_length=bad)


def test_one_strength_per_level_or_none():
    with pytest.raises(ValueError, match="levels"):
        BeamModel(["CASSLF"], order=3, alpha=(1.0, 2.0))


def test_set_alpha_is_building_with_that_alpha():
    seqs, counts = _corpus()
    alpha, al = (1.5, 30.0, 0.2, 800.0), 2.5
    a = BeamModel(seqs, abundances=counts, order=3)
    a.set_alpha(alpha, al)
    b = BeamModel(seqs, abundances=counts, order=3, alpha=alpha, alpha_length=al)
    probe = seqs[:50] + ["CASSWWWF", "C", "QQQQQQ"]
    assert np.array_equal(a.log_pseq_batch(probe), b.log_pseq_batch(probe))
    assert a.config == b.config


def test_set_alpha_refuses_without_changing_the_model():
    seqs, counts = _corpus()
    m = BeamModel(seqs, abundances=counts, order=3)
    before = m.log_pseq(seqs[0])
    with pytest.raises(ValueError):
        m.set_alpha((1.0, 1.0, -1.0, 1.0))
    assert m.log_pseq(seqs[0]) == before


def test_estimated_strengths_survive_save_and_load(tmp_path):
    seqs, counts = _corpus()
    m = BeamModel(seqs, abundances=counts, order=3)
    p = tmp_path / "auto.beam"
    m.save(p)
    back = BeamModel.from_file(p)
    assert back.config == m.config and back.config["alpha_auto"]


def test_the_strengths_survive_save_and_load(tmp_path):
    seqs, counts = _corpus()
    m = BeamModel(seqs, abundances=counts, order=3, alpha=(1.0, 2.0, 3.0, 4.0),
                  alpha_length=5.0)
    p = tmp_path / "m.beam"
    m.save(p)
    back = BeamModel.from_file(p)
    assert back.config == m.config
    probe = seqs[:30] + ["CASSWWWF"]
    assert np.array_equal(back.log_pseq_batch(probe), m.log_pseq_batch(probe))


def test_an_unseen_context_in_a_tempered_model_is_tempered_too():
    """An unobserved row is its ancestor, so tempering must reach the ancestor
    rows and the uniform row as well as the stored top rows."""
    seqs, counts = _corpus()
    m = BeamModel(seqs, abundances=counts, order=3)
    t = 1.7
    mt = m.tempered(t)
    for s in ["CASSWWWWF", "YYYY", seqs[0]]:
        assert mt.log_pseq(s) == pytest.approx(
            t * m.log_pseq(s) - m.log_mellin(t), abs=1e-9), s


def test_fit_alpha_improves_the_held_out_likelihood():
    seqs, counts = _corpus(n=1500, seed=4)
    fit = fit_alpha(seqs, counts, order=3, max_length=20, sweeps=1)
    assert fit.held_out_log_lik > fit.default_log_lik
    assert len(fit.alpha) == 4 and fit.split == "thinning"
    with pytest.raises(TypeError):
        float(fit)


def test_fit_alpha_is_repeatable_by_seed():
    seqs, counts = _corpus(n=600, seed=6)
    a = fit_alpha(seqs, counts, order=2, max_length=20, sweeps=1, seed=3)
    b = fit_alpha(seqs, counts, order=2, max_length=20, sweeps=1, seed=3)
    assert a.alpha == b.alpha and a.alpha_length == b.alpha_length


def test_fit_alpha_against_another_repertoire():
    seqs, counts = _corpus(n=600, seed=6)
    other, other_counts = _corpus(n=400, seed=9)
    fit = fit_alpha(seqs, counts, held_out=(other, other_counts), order=2,
                    max_length=20, sweeps=1)
    assert fit.split == "external" and fit.seed is None
    assert fit.held_out_log_lik >= fit.default_log_lik


def test_thinning_needs_integer_read_counts():
    with pytest.raises(ValueError, match="integer"):
        fit_alpha(["CASSLF", "CASSQF"], [1.5, 2.0], order=2)
