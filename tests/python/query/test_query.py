"""The Python query API: exactness, the cost ladder, and the refusals.

Where a quantity can be checked against brute force it is, and the nucleotide
alphabet is used for that on purpose -- 5**5 is 3,125 sequences and walkable in
Python, where 20**5 is not. The identities are checked on amino acids, where
they are not enumerable but are still required to hold.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from BEAM import BeamModel
from BEAM.query import (BeamQuery, Condition, ImpossibleCondition,
                        QueryTooLarge, UnsupportedQuery)

NT = "ACGTN"
EXACT = 1e-12


@pytest.fixture(scope="module")
def nt_model():
    """Lengths 4 to 6, so that a length-5 sequence has producible neighbours
    one residue either side. A corpus of one length makes every insertion
    impossible, which is a correct answer and an uninformative test."""
    seqs = ["ACGT", "ACGA", "TCGT",
            "ACGTA", "ACGTT", "AGGTA", "ACCTA", "ACGAA", "TCGTA", "ACGTG",
            "ACGTAC", "ACGTTA", "AGGTAC", "ACGTAA"]
    return BeamModel(seqs, alphabet="nt", order=2, max_length=8, strict=False)


@pytest.fixture(scope="module")
def aa_model():
    import random
    random.seed(11)
    seqs = ["CASS" + "".join(random.choice("LGQPV") for _ in range(4)) + "QYF"
            for _ in range(500)]
    return BeamModel(seqs, order=3)


def _support(model, length, alphabet=NT):
    """Every sequence of `length` with non-zero probability, and its mass."""
    out = {}
    for combo in itertools.product(alphabet, repeat=length):
        s = "".join(combo)
        p = math.exp(model.log_pseq(s))
        if p > 0.0:
            out[s] = p
    total = sum(out.values())
    return {s: p / total for s, p in out.items()}


# ── Conditions, before any query runs ───────────────────────────

def test_a_fresh_condition_selects_everything():
    c = Condition(6, "aa")
    assert c.is_free and not c.needs_product
    assert c.pattern() == "______"


def test_builders_narrow_and_do_not_mutate():
    a = Condition(5, "aa")
    b = a.at(1, "C").oneof(3, "FY").exclude(4, "A")
    assert a.is_free, "the original is untouched"
    # `_` is untouched, a letter is pinned, `.` is narrowed to several --
    # excluding one residue of twenty still narrows the position.
    assert b.pattern() == "_C_..", "pinned, a pair, and one residue removed"
    assert not b.is_free and not b.needs_product, "masks stay on the free tier"


def test_the_automaton_tier_is_visible():
    c = Condition(8, "aa").contains("CASS")
    assert c.needs_product, "a motif cannot be a per-position restriction"
    assert c.pattern() == "________", "and does not show in the mask picture"


def test_a_condition_refuses_what_it_cannot_express():
    c = Condition(5, "aa")
    with pytest.raises(UnsupportedQuery, match="not one of"):
        c.at(0, "B")
    with pytest.raises(UnsupportedQuery, match="outside"):
        c.at(9, "C")
    with pytest.raises(UnsupportedQuery, match="past the end"):
        c.run(3, "CASS")
    with pytest.raises(UnsupportedQuery, match="cannot occur"):
        c.contains("CASSLGQ")
    with pytest.raises(UnsupportedQuery, match="empty"):
        c.count("AC", 3, 2)


def test_conditions_of_different_shape_do_not_combine():
    a, b = Condition(5, "aa"), Condition(6, "aa")
    with pytest.raises(UnsupportedQuery, match="cannot combine"):
        _ = a & b


# ── Exactness, against the whole support ───────────────────────

def test_probability_matches_enumeration(nt_model):
    q = BeamQuery(nt_model)
    sup = _support(nt_model, 5)
    cases = [
        (q.condition(5).at(0, "A"), lambda s: s[0] == "A"),
        (q.condition(5).oneof(2, "GC"), lambda s: s[2] in "GC"),
        (q.condition(5).at(0, "A").at(3, "T"), lambda s: s[0] == "A" and s[3] == "T"),
        (q.condition(5).contains("CG"), lambda s: "CG" in s),
        (q.condition(5).count("A", 2, 3), lambda s: 2 <= s.count("A") <= 3),
    ]
    for cond, keep in cases:
        want = sum(p for s, p in sup.items() if keep(s))
        assert 0.0 < want < 1.0, "the case selects a real fraction"
        assert q.probability(cond) == pytest.approx(want, abs=EXACT)


def test_boolean_algebra_matches_enumeration(nt_model):
    q = BeamQuery(nt_model)
    sup = _support(nt_model, 5)
    a, ka = q.condition(5).at(0, "A"), lambda s: s[0] == "A"
    b, kb = q.condition(5).contains("GT"), lambda s: "GT" in s

    for cond, keep in ((a & b, lambda s: ka(s) and kb(s)),
                       (a | b, lambda s: ka(s) or kb(s)),
                       (~a, lambda s: not ka(s)),
                       (~(a & b), lambda s: not (ka(s) and kb(s)))):
        want = sum(p for s, p in sup.items() if keep(s))
        assert q.probability(cond) == pytest.approx(want, abs=EXACT)

    # De Morgan, through the engine rather than the automaton algebra.
    assert q.probability(~(a & b)) == pytest.approx(
        q.probability((~a) | (~b)), abs=EXACT)


def test_marginals_and_entropy_match_enumeration(nt_model):
    q = BeamQuery(nt_model)
    sup = _support(nt_model, 5)
    cond = q.condition(5).at(0, "A")
    kept = {s: p for s, p in sup.items() if s[0] == "A"}
    mass = sum(kept.values())

    want = np.zeros((5, len(NT)))
    for s, p in kept.items():
        for i, r in enumerate(s):
            want[i, NT.index(r)] += p / mass
    assert np.allclose(q.marginals(cond), want, atol=EXACT)

    h = -sum((p / mass) * math.log(p / mass) for p in kept.values())
    assert q.entropy(cond) == pytest.approx(h, abs=EXACT)
    assert q.marginals(cond)[0, 0] == pytest.approx(1.0, abs=EXACT), "pinned"


# ── Identities, on a model too large to enumerate ──────────────

def test_the_information_matrix_obeys_its_identities(aa_model):
    q = BeamQuery(aa_model)
    c = q.condition(11)
    mi, h = q.mutual_information(c), q.position_entropy(c)
    assert np.allclose(mi, mi.T, atol=EXACT), "symmetric"
    assert (mi >= 0.0).all(), "never negative"
    assert np.allclose(np.diag(mi), h, atol=EXACT), "diagonal is H(X_i)"
    assert h.sum() >= q.entropy(c) - EXACT, "subadditive"
    for i in range(11):
        for j in range(11):
            assert q.conditional_entropy(c, i, j) >= -EXACT


def test_pair_joint_agrees_with_the_marginals(aa_model):
    q = BeamQuery(aa_model)
    c = q.condition(11)
    m = q.marginals(c)
    j = q.pair_joint(c, 4, 6)
    assert j.sum() == pytest.approx(1.0, abs=EXACT)
    assert np.allclose(j.sum(axis=1), m[4], atol=EXACT)
    assert np.allclose(j.sum(axis=0), m[6], atol=EXACT)
    self_j = q.pair_joint(c, 5, 5)
    assert np.allclose(np.diag(self_j), m[5], atol=EXACT), "a position with itself"
    assert self_j.sum() == pytest.approx(1.0, abs=EXACT)


def test_expected_count_is_the_summed_marginals(aa_model):
    q = BeamQuery(aa_model)
    c = q.condition(11)
    m = q.marginals(c)
    syms = "ACDEFGHIKLMNPQRSTVWY"
    for residues in ("G", "LG", "QYF"):
        want = sum(m[:, syms.index(r)].sum() for r in residues)
        mean, var = q.expect_count(residues, c)
        assert mean == pytest.approx(want, abs=EXACT)
        assert var >= 0.0


def test_an_indicator_expectation_is_the_marginal(aa_model):
    q = BeamQuery(aa_model)
    c = q.condition(11)
    m = q.marginals(c)
    table = np.zeros((11, 20))
    table[4, 9] = 1.0                      # L at position 4
    mean, var = q.expect(table, c)
    p = m[4, 9]
    assert mean == pytest.approx(p, abs=EXACT)
    assert var == pytest.approx(p * (1.0 - p), abs=EXACT), "a Bernoulli"


# ── Cross-length ───────────────────────────────────────────────

def test_insertions_and_deletions_round_trip(nt_model):
    q = BeamQuery(nt_model)
    ins = q.insertions("ACGTA")
    assert ins and all(len(r["sequence"]) == 6 for r in ins)
    assert sum(r["posterior"] for r in ins) == pytest.approx(1.0, abs=EXACT)
    assert ins == sorted(ins, key=lambda r: -r["posterior"]), "ranked"
    for r in ins[:5]:
        assert r["residue"] in NT
        undone = r["sequence"][:r["site"]] + r["sequence"][r["site"] + 1:]
        assert undone == "ACGTA", "undoing the recorded edit gives the source"

    dele = q.deletions("ACGTA")
    assert all(r["residue"] is None for r in dele), "a deletion adds nothing"
    assert all(len(r["sequence"]) == 4 for r in dele)


def test_a_run_collapses_its_aliases(nt_model):
    q = BeamQuery(nt_model)
    # Deleting either A of the run leaves the same string, so it appears once.
    got = {r["sequence"] for r in q.deletions("AACGT")}
    assert "ACGT" in got
    assert len(got) == len(q.deletions("AACGT")), "no duplicate spellings"


# ── Refusals ───────────────────────────────────────────────────

def test_conditioning_on_the_impossible_is_refused(nt_model):
    q = BeamQuery(nt_model)
    bad = q.condition(5).at(2, "G").at(2, "T")     # contradiction
    assert q.log_probability(bad) == -math.inf, "asking P is fine"
    with pytest.raises(ImpossibleCondition):
        q.marginals(bad)
    with pytest.raises(ImpossibleCondition):
        q.entropy(bad)


def test_a_condition_over_the_wrong_alphabet_is_refused(nt_model):
    q = BeamQuery(nt_model)
    with pytest.raises(UnsupportedQuery, match="alphabet"):
        q.probability(Condition(5, "aa"))
    with pytest.raises(UnsupportedQuery, match="Condition"):
        q.probability("not a condition")


def test_a_functional_of_the_wrong_shape_is_refused(aa_model):
    q = BeamQuery(aa_model)
    with pytest.raises(UnsupportedQuery, match="table"):
        q.expect(np.zeros((3, 20)), q.condition(11))
    with pytest.raises(UnsupportedQuery, match="not one of"):
        q.expect_count("BZ", q.condition(11))


def test_a_query_over_the_memory_budget_is_refused(aa_model):
    import BEAM
    q = BeamQuery(aa_model)
    saved = BEAM.memory_budget()
    try:
        BEAM.set_memory_budget(1024)
        with pytest.raises(QueryTooLarge, match="memory budget"):
            q.marginals(q.condition(11))
    finally:
        BEAM.set_memory_budget(saved)
    q.marginals(q.condition(11))           # and it works again afterwards


# ── Cost ───────────────────────────────────────────────────────

def test_cost_reports_the_ladder(aa_model):
    q = BeamQuery(aa_model)
    plain = q.condition(11)
    motif = q.condition(11).contains("QYF")
    one = q.cost(plain)
    with_aut = q.cost(motif)
    whole = q.cost(plain, whole_matrix=True)

    assert with_aut["edges"] > one["edges"], "an automaton costs a product"
    assert with_aut["bytes"] > one["bytes"]
    assert whole["edges"] == one["edges"] * (11 * 20 + 1), "a sweep per pin"
    assert whole["bytes"] > one["bytes"], "and holds accumulators across them"
    assert q.condition(11).at(3, "S").is_free is False
